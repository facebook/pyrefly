/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::Cell;
use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::mem;

use pyrefly_types::dimension::ShapeError;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::simplify::intersect;
use pyrefly_types::special_form::SpecialForm;
use pyrefly_types::tensor::TensorShape;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Union;
use pyrefly_util::gas::Gas;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::recurser::Guard;
use pyrefly_util::recurser::Recurser;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::AttrSubsetError;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::solver::type_order::TypeOrder;
use crate::types::callable::Callable;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::PrefixParam;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::module::ModuleType;
use crate::types::simplify::simplify_tuples;
use crate::types::simplify::unions;
use crate::types::simplify::unions_with_literals;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::types::types::Var;

/// Error message when a variable has leaked from one module to another.
///
/// We have a rule that `Var`'s should not leak from one module to another, but it has happened.
/// The easiest debugging technique is to look at the `Solutions` and see if there is a `Var(Unique)`
/// in the output. The usual cause is that we failed to visit all the necessary `Type` fields.
const VAR_LEAK: &str = "Internal error: a variable has leaked from one module to another.";

/// A number chosen such that all practical types are less than this depth,
/// but low enough to avoid stack overflow. Rust's default stack size is 8MB,
/// and each recursive call to is_subset_eq can use several KB of stack space
/// due to large enums (Type) and lock guards.
const INITIAL_GAS: Gas = Gas::new(200);

/// Accumulated bounds for a solver variable.
#[derive(Clone, Debug, Default)]
struct Bounds {
    lower: Vec<Type>,
    upper: Vec<Type>,
}

impl Bounds {
    fn new() -> Self {
        Self {
            lower: Vec::new(),
            upper: Vec::new(),
        }
    }

    fn extend(&mut self, other: Bounds) {
        self.lower.extend(other.lower);
        self.upper.extend(other.upper);
    }

    fn is_empty(&self) -> bool {
        self.lower.is_empty() && self.upper.is_empty()
    }
}

#[derive(Clone, Debug)]
enum Variable {
    /// A "partial type" (terminology borrowed from mypy) for an empty container.
    ///
    /// Pyrefly only creates partial types for assignments, and will attempt to
    /// determine the type ("pin" it) using the first use of the name assigned.
    ///
    /// It will attempt to infer the type from the first downstream use; if the
    /// type cannot be determined it becomes `Any`.
    ///
    /// The TextRange is the location of the empty container literal (e.g., `[]` or `{}`),
    /// used for error reporting when the type cannot be inferred.
    PartialContained(TextRange),
    /// A "partial type" (see above) representing a type variable that was not
    /// solved as part of a generic function or constructor call.
    ///
    /// Behaves similar to `PartialContained`, but it has the ability to use
    /// the default type if the first use does not pin.
    PartialQuantified(Quantified),
    /// A variable due to generic instantiation, `def f[T](x: T): T` with `f(1)`
    Quantified {
        quantified: Quantified,
        bounds: Bounds,
    },
    /// A variable caused by general recursion, e.g. `x = f(); def f(): return x`.
    Recursive,
    /// A variable that used to decompose a type, e.g. getting T from Awaitable[T]
    Unwrap(Bounds),
    /// A variable whose answer has been determined
    Answer(Type),
}

impl Variable {
    fn finished(q: &Quantified) -> Self {
        if q.default().is_some() {
            Variable::Answer(q.as_gradual_type())
        } else {
            Variable::PartialQuantified(q.clone())
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variable::PartialContained(_) => write!(f, "PartialContained"),
            Variable::PartialQuantified(q)
            | Variable::Quantified {
                quantified: q,
                bounds: _,
            } => {
                let label = if matches!(self, Variable::PartialQuantified(_)) {
                    "PartialQuantified"
                } else {
                    "Quantified"
                };
                let k = q.kind;
                if let Some(t) = &q.default {
                    write!(f, "{label}({k}, default={t})")
                } else {
                    write!(f, "{label}({k})")
                }
            }
            Variable::Recursive => write!(f, "Recursive"),
            Variable::Unwrap(_) => write!(f, "Unwrap"),
            Variable::Answer(t) => write!(f, "{t}"),
        }
    }
}

#[derive(Debug)]
#[must_use = "Quantified vars must be finalized. Pass to finish_quantified."]
pub struct QuantifiedHandle(Vec<Var>);

impl QuantifiedHandle {
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Split the handle into (vars in ty, vars not in ty)
    pub fn partition_by(self, ty: &Type) -> (Self, Self) {
        let vars_in_ty = ty.collect_maybe_placeholder_vars();
        let (left, right) = self.0.into_iter().partition(|var| vars_in_ty.contains(var));
        (QuantifiedHandle(left), QuantifiedHandle(right))
    }
}

/// The solver tracks variables as a mapping from Var to Variable.
/// We use union-find to unify two vars, using RefCell for interior
/// mutability.
///
/// Note that RefCell means we need to be careful about how we access
/// variables. Access is "mutable xor shared" like ordinary references,
/// except with runtime instead of static enforcement.
#[derive(Debug, Default)]
struct Variables(SmallMap<Var, RefCell<VariableNode>>);

/// A union-find node. We store the parent pointer in a Cell so that we
/// can implement path compression. We use a separate Cell instead of using
/// the RefCell around the node, because we might find that two vars point
/// to the same root, which would cause us to borrow_mut twice and panic.
#[derive(Clone, Debug)]
enum VariableNode {
    Goto(Cell<Var>),
    Root(Variable, usize),
}

impl Display for VariableNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariableNode::Goto(x) => write!(f, "Goto({})", x.get()),
            VariableNode::Root(x, _) => write!(f, "{x}"),
        }
    }
}

impl Variables {
    fn get<'a>(&'a self, x: Var) -> Ref<'a, Variable> {
        let root = self.get_root(x);
        let variable = self.get_node(root).borrow();
        Ref::map(variable, |v| match v {
            VariableNode::Root(v, _) => v,
            _ => unreachable!(),
        })
    }

    fn get_mut<'a>(&'a self, x: Var) -> RefMut<'a, Variable> {
        let root = self.get_root(x);
        let variable = self.get_node(root).borrow_mut();
        RefMut::map(variable, |v| match v {
            VariableNode::Root(v, _) => v,
            _ => unreachable!(),
        })
    }

    /// Unification for vars. Currently unification order matters, since unification is destructive.
    /// This function will always preserve the "Variable" information from `y`, even when `x` has
    /// higher rank, for backwards compatibility reasons. Otherwise, this is standard union by rank.
    fn unify(&self, x: Var, y: Var) {
        let x_root = self.get_root(x);
        let y_root = self.get_root(y);
        if x_root != y_root {
            let mut x_node = self.get_node(x_root).borrow_mut();
            let mut y_node = self.get_node(y_root).borrow_mut();
            match (&mut *x_node, &mut *y_node) {
                (VariableNode::Root(x, x_rank), VariableNode::Root(y, y_rank)) => {
                    if x_rank > y_rank {
                        // X has higher rank, preserve the Variable data from Y
                        std::mem::swap(x, y);
                        *y_node = VariableNode::Goto(Cell::new(x_root));
                    } else {
                        if x_rank == y_rank {
                            *y_rank += 1;
                        }
                        *x_node = VariableNode::Goto(Cell::new(y_root));
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a Var, Ref<'a, VariableNode>)> {
        self.0.iter().map(|(x, y)| (x, y.borrow()))
    }

    /// Insert a fresh variable. If we already have a record of this variable,
    /// this function will panic. To update an existing variable, use `update`.
    fn insert_fresh(&mut self, x: Var, v: Variable) {
        assert!(
            self.0
                .insert(x, RefCell::new(VariableNode::Root(v, 0)))
                .is_none()
        );
    }

    /// Update an existing variable. If the variable does not exist, this will
    /// panic. To insert a new variable, use `insert_fresh`.
    fn update(&self, x: Var, v: Variable) {
        *self.get_mut(x) = v;
    }

    fn recurse<'a>(&self, x: Var, recurser: &'a VarRecurser) -> Option<Guard<'a, Var>> {
        let root = self.get_root(x);
        recurser.recurse(root)
    }

    /// Get root using path compression.
    fn get_root(&self, x: Var) -> Var {
        match &*self.get_node(x).borrow() {
            VariableNode::Root(..) => x,
            VariableNode::Goto(parent) => {
                let root = self.get_root(parent.get());
                parent.set(root);
                root
            }
        }
    }

    fn get_node(&self, x: Var) -> &RefCell<VariableNode> {
        assert_ne!(
            x,
            Var::ZERO,
            "Internal error: unexpected Var::ZERO, which is a dummy value."
        );
        self.0.get(&x).expect(VAR_LEAK)
    }
}

/// A recurser for Vars which is aware of unification.
/// Prefer this over Recurser<Var> and use Solver::recurse.
pub struct VarRecurser(Recurser<Var>);

impl VarRecurser {
    pub fn new() -> Self {
        Self(Recurser::new())
    }

    fn recurse<'a>(&'a self, var: Var) -> Option<Guard<'a, Var>> {
        self.0.recurse(var)
    }
}

#[derive(Debug)]
pub enum PinError {
    ImplicitPartialContained(TextRange),
    UnfinishedQuantified(Quantified),
}

/// Snapshot of solver variable state.
/// IMPORTANT: this struct is deliberately opaque.
/// Var state should not be exposed outside this file.
pub struct VarSnapshot(Vec<(Var, VarState)>);

struct VarState {
    node: VariableNode,
    variable: Variable,
    error: Option<TypeVarSpecializationError>,
}

#[derive(Debug)]
pub struct Solver {
    variables: Mutex<Variables>,
    instantiation_errors: RwLock<SmallMap<Var, TypeVarSpecializationError>>,
    /// Cross-call cache for protocol conformance results.
    /// Only caches results for types that contain no Vars, to ensure
    /// soundness across different subset contexts.
    protocol_cache: Mutex<HashMap<(Type, Type), Result<(), SubsetError>>>,
    pub infer_with_first_use: bool,
    pub heap: TypeHeap,
    pub tensor_shapes: bool,
    pub strict_callable_subtyping: bool,
    pub spec_compliant_overloads: bool,
}

impl Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (x, y) in self.variables.lock().iter() {
            writeln!(f, "{x} = {y}")?;
        }
        Ok(())
    }
}

/// A number chosen such that all practical types are less than this depth,
/// but we don't want to stack overflow.
const TYPE_LIMIT: usize = 20;

/// A new bound to add to a variable.
enum NewBound {
    /// The new bound should replace the existing bound.
    UpdateExistingBound(Type),
    /// The new bound should be appended to the existing bounds.
    AddBound(Type),
}

impl Solver {
    /// Create a new solver.
    pub fn new(
        infer_with_first_use: bool,
        tensor_shapes: bool,
        strict_callable_subtyping: bool,
        spec_compliant_overloads: bool,
    ) -> Self {
        Self {
            variables: Default::default(),
            instantiation_errors: Default::default(),
            protocol_cache: Default::default(),
            infer_with_first_use,
            heap: TypeHeap::new(),
            tensor_shapes,
            strict_callable_subtyping,
            spec_compliant_overloads,
        }
    }

    pub fn recurse<'a>(&self, var: Var, recurser: &'a VarRecurser) -> Option<Guard<'a, Var>> {
        self.variables.lock().recurse(var, recurser)
    }

    /// Look up a cached protocol conformance result.
    pub fn check_protocol_cache(&self, got: &Type, want: &Type) -> Option<Result<(), SubsetError>> {
        self.protocol_cache
            .lock()
            .get(&(got.clone(), want.clone()))
            .cloned()
    }

    /// Store a protocol conformance result.
    pub fn store_protocol_cache(&self, got: Type, want: Type, result: Result<(), SubsetError>) {
        self.protocol_cache.lock().insert((got, want), result);
    }

    /// Force all non-recursive Vars in `vars`.
    /// TODO: deduplicate Variable-to-gradual-type logic with `force_var`.
    pub fn pin_placeholder_type(&self, var: Var, pin_partial_types: bool) -> Option<PinError> {
        let variables = self.variables.lock();
        let mut variable = variables.get_mut(var);
        match &mut *variable {
            Variable::Recursive | Variable::Answer(..) => {
                // Nothing to do if we have an answer already, and we want to skip recursive Vars
                // which do not represent placeholder types.
                None
            }
            Variable::Quantified {
                quantified: q,
                bounds: _,
            } => {
                // A Variable::Quantified should always be finished (see `finish_quantified`) by
                // the code that creates it, because we need to know when we're done collecting
                // constraints. If we see a Quantified while pinning other placeholder types, that
                // means we forgot to finish it.
                let result = Some(PinError::UnfinishedQuantified(q.clone()));
                *variable = Variable::Answer(q.as_gradual_type());
                result
            }
            Variable::PartialQuantified(q) => {
                if pin_partial_types {
                    *variable = Variable::Answer(q.as_gradual_type());
                }
                None
            }
            Variable::PartialContained(range) if pin_partial_types => {
                let range = *range;
                *variable = Variable::Answer(self.heap.mk_any_implicit());
                Some(PinError::ImplicitPartialContained(range))
            }
            Variable::PartialContained(_) => None,
            Variable::Unwrap(bounds) => {
                *variable = Variable::Answer(
                    self.solve_bounds(mem::take(bounds))
                        .unwrap_or_else(Type::any_implicit),
                );
                None
            }
        }
    }

    /// Check whether a Var represents a partial/placeholder type that would be
    /// pinned by `pin_placeholder_type` with `pin_partial_types=true`.
    /// This excludes Quantified (which represents an error case, not a normal partial type)
    /// and focuses on the types created specifically for first-use inference.
    pub fn var_is_partial(&self, var: Var) -> bool {
        let variables = self.variables.lock();
        let variable = variables.get(var);
        matches!(
            &*variable,
            Variable::PartialQuantified(_) | Variable::PartialContained(_) | Variable::Unwrap(_)
        )
    }

    /// Snapshot the current state of the given vars so they can be restored later.
    pub fn snapshot_vars(&self, vars: &[Var]) -> VarSnapshot {
        if vars.is_empty() {
            return VarSnapshot(Vec::new()); // avoid acquiring locks
        }
        let variables = self.variables.lock();
        let errors = self.instantiation_errors.read();
        VarSnapshot(
            vars.iter()
                .map(|v| {
                    (
                        *v,
                        VarState {
                            node: variables.get_node(*v).borrow().clone(),
                            variable: variables.get(*v).clone(),
                            error: errors.get(v).cloned(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// Restore vars to a previously saved snapshot.
    pub fn restore_vars(&self, snapshot: VarSnapshot) {
        if snapshot.0.is_empty() {
            return; // avoid acquiring locks
        }
        let variables = self.variables.lock();
        let mut errors = self.instantiation_errors.write();
        for (var, state) in snapshot.0 {
            *variables.get_node(var).borrow_mut() = state.node;
            variables.update(var, state.variable);
            match state.error {
                Some(e) => {
                    errors.insert(var, e);
                }
                None => {
                    if errors.contains_key(&var) {
                        errors.shift_remove(&var);
                    }
                }
            }
        }
    }

    /// Finish the type returned from a function call. This entails expanding solved variables,
    /// erasing unsolved variables without defaults from unions, and canonicalizing dimension
    /// expressions so that all-literal SizeExpr trees fold to single literals.
    pub fn finish_function_return(&self, mut t: Type) -> Type {
        self.expand_with_limit(&mut t, TYPE_LIMIT, &VarRecurser::new(), false);
        self.erase_unsolved_variables(&mut t);
        self.simplify_mut(&mut t);
        // After variable expansion, dimension expressions may have all-literal operands
        // (e.g., (64 + 2 - 3 - 1) // 2 + 1) that should fold to a single literal (32).
        // Without this, Sequential chaining compounds symbolic expressions across layers.
        self.simplify_forced_type(t)
    }

    /// Expand a type. All variables that have been bound will be replaced with non-Var types,
    /// even if they are recursive (using `Any` for self-referential occurrences).
    /// Variables that have not yet been bound will remain as Var.
    ///
    /// In addition, if the type exceeds a large depth, it will be replaced with `Any`.
    pub fn expand_vars(&self, mut t: Type) -> Type {
        self.expand_vars_mut(&mut t);
        t
    }

    /// Like `expand`, but when you have a `&mut`.
    pub fn expand_vars_mut(&self, t: &mut Type) {
        self.expand_with_limit(t, TYPE_LIMIT, &VarRecurser::new(), false);
        // After we substitute bound variables, we may be able to simplify some types
        self.simplify_mut(t);
    }

    /// Expand, but if the resulting type will be greater than limit levels deep, return an `Any`.
    /// Avoids producing things that stack overflow later in the process.
    fn expand_with_limit(
        &self,
        t: &mut Type,
        limit: usize,
        recurser: &VarRecurser,
        expand_unfinished_variables: bool,
    ) {
        if limit == 0 {
            // TODO: Should probably add an error here, and use any_error,
            // but don't have any good location information to hand.
            *t = self.heap.mk_any_implicit();
        } else if let Type::Var(x) = t {
            let lock = self.variables.lock();
            if let Some(_guard) = lock.recurse(*x, recurser) {
                let variable = lock.get(*x);
                match &*variable {
                    Variable::Answer(ty) => {
                        *t = ty.clone();
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(t, limit - 1, recurser, expand_unfinished_variables);
                    }
                    Variable::Quantified {
                        quantified: _,
                        bounds,
                    }
                    | Variable::Unwrap(bounds)
                        if expand_unfinished_variables
                            && let Some(bound) = self.solve_bounds(bounds.clone()) =>
                    {
                        *t = bound;
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(t, limit - 1, recurser, expand_unfinished_variables);
                    }
                    _ => {}
                }
            } else {
                *t = self.heap.mk_any_implicit();
            }
        } else {
            t.recurse_mut(&mut |t| {
                self.expand_with_limit(t, limit - 1, recurser, expand_unfinished_variables)
            });
        }
    }

    /// Expand `Variable::Unwrap` to its answer or its lower bounds accumulated so far.
    pub fn expand_unwrap(&self, v: Var) -> Type {
        let variables = self.variables.lock();
        match &*variables.get(v) {
            Variable::Answer(t) => t.clone(),
            Variable::Unwrap(bounds) if let Some(bound) = self.solve_bounds(bounds.clone()) => {
                bound
            }
            _ => v.to_type(&self.heap),
        }
    }

    /// Public wrapper to expand a dimension type by resolving bound Vars.
    /// Used by subset checking to expand Vars before comparing dimension expressions.
    pub fn expand_dimension(&self, dim_ty: &mut Type) {
        self.expand_with_limit(dim_ty, TYPE_LIMIT, &VarRecurser::new(), true);
    }

    /// Given a `Var`, ensures that the solver has an answer for it (or inserts Any if not already),
    /// and returns that answer. Note that if the `Var` is already bound to something that contains a
    /// `Var` (including itself), then we will return the answer.
    pub fn force_var(&self, v: Var) -> Type {
        let lock = self.variables.lock();
        let mut e = lock.get_mut(v);
        let result = match &mut *e {
            Variable::Answer(t) => t.clone(),
            _ => {
                let ty = match &mut *e {
                    Variable::Quantified {
                        quantified: q,
                        bounds,
                    } => self
                        .solve_bounds(mem::take(bounds))
                        .unwrap_or_else(|| q.as_gradual_type()),
                    Variable::PartialQuantified(q) => q.as_gradual_type(),
                    Variable::Unwrap(bounds) => self
                        .solve_bounds(mem::take(bounds))
                        .unwrap_or_else(|| self.heap.mk_any_implicit()),
                    _ => self.heap.mk_any_implicit(),
                };
                *e = Variable::Answer(ty.clone());
                ty
            }
        };
        // Simplify dimension expressions after forcing
        // This ensures Tensor[(10 * 20)] becomes Tensor[200]
        self.simplify_forced_type(result)
    }

    /// Simplify dimension expressions in a forced type
    fn simplify_forced_type(&self, mut ty: Type) -> Type {
        // Use transform_mut to visit every Type node and simplify dimensions
        ty.transform_mut(&mut |t| {
            let simplified = canonicalize(t.clone());
            if &simplified != t {
                *t = simplified;
            }
        });
        ty
    }

    fn deep_force_mut_with_limit(&self, t: &mut Type, limit: usize, recurser: &VarRecurser) {
        if limit == 0 {
            // TODO: Should probably add an error here, and use any_error,
            // but don't have any good location information to hand.
            *t = self.heap.mk_any_implicit();
        } else if let Type::Var(v) = t {
            if let Some(_guard) = self.recurse(*v, recurser) {
                *t = self.force_var(*v);
                self.deep_force_mut_with_limit(t, limit - 1, recurser);
            } else {
                *t = self.heap.mk_any_implicit();
            }
        } else {
            t.recurse_mut(&mut |t| self.deep_force_mut_with_limit(t, limit - 1, recurser));
            // After forcing all Vars recursively, simplify dimension expressions
            // This handles cases like Tensor[(10 * 20)] after Vars are forced to 10 and 20
            let simplified = canonicalize(t.clone());
            if &simplified != t {
                *t = simplified;
            }
        }
    }

    /// A version of `deep_force` that works in-place on a `Type`.
    pub fn deep_force_mut(&self, t: &mut Type) {
        self.deep_force_mut_with_limit(t, TYPE_LIMIT, &VarRecurser::new());
        // After forcing, we might be able to simplify some unions
        self.simplify_mut(t);
    }

    /// Simplify a type as much as we can.
    fn simplify_mut(&self, t: &mut Type) {
        t.transform_mut(&mut |x| {
            if let Type::Union(box Union {
                members: xs,
                display_name: original_name,
            }) = x
            {
                let mut merged = unions(mem::take(xs), &self.heap);
                // Preserve union display names during simplification
                if let Type::Union(box Union { display_name, .. }) = &mut merged {
                    *display_name = original_name.clone();
                }
                *x = merged;
            }
            if let Type::Intersect(y) = x {
                *x = intersect(mem::take(&mut y.0), y.1.clone(), &self.heap);
            }
            if let Type::Tuple(tuple) = x {
                *x = self
                    .heap
                    .mk_tuple(simplify_tuples(mem::take(tuple), &self.heap));
            }
            // Flatten Tensor[prefix, *tuple[...], suffix] after TypeVarTuple resolution
            if let Type::Tensor(tensor) = x
                && let TensorShape::Unpacked(box (prefix, middle, suffix)) = &mut tensor.shape
                && let Type::Tuple(tuple_variant) = middle
            {
                match tuple_variant {
                    Tuple::Concrete(elements) => {
                        let mut new_dims = prefix.clone();
                        new_dims.extend(elements.clone());
                        new_dims.extend(suffix.clone());
                        tensor.shape = TensorShape::Concrete(new_dims);
                    }
                    Tuple::Unpacked(box (tuple_prefix, tuple_middle, tuple_suffix)) => {
                        let mut new_prefix = prefix.clone();
                        new_prefix.extend(tuple_prefix.clone());
                        let mut new_suffix = tuple_suffix.clone();
                        new_suffix.extend(suffix.clone());
                        tensor.shape = TensorShape::Unpacked(Box::new((
                            new_prefix,
                            tuple_middle.clone(),
                            new_suffix,
                        )));
                    }
                    _ => {}
                }
            }
            // When a param spec is resolved, collapse any Concatenate and Callable types that use it
            if let Type::Concatenate(ts, box Type::ParamSpecValue(paramlist)) = x {
                let params = mem::take(paramlist).prepend_types(ts).into_owned();
                *x = self.heap.mk_param_spec_value(params);
            }
            if let Type::Concatenate(ts, box Type::Concatenate(ts2, pspec)) = x {
                let combined: Box<[PrefixParam]> = ts.iter().chain(ts2.iter()).cloned().collect();
                *x = self.heap.mk_concatenate(combined, (**pspec).clone());
            }
            let (callable, kind) = match x {
                Type::Callable(c) => (Some(&mut **c), None),
                Type::Function(box Function {
                    signature: c,
                    metadata: k,
                }) => (Some(c), Some(k)),
                _ => (None, None),
            };
            if let Some(Callable {
                params: Params::ParamSpec(ts, pspec),
                ret,
            }) = callable
            {
                let new_callable = |c| {
                    if let Some(k) = kind {
                        self.heap.mk_function(Function {
                            signature: c,
                            metadata: k.clone(),
                        })
                    } else {
                        self.heap.mk_callable_from(c)
                    }
                };
                match pspec {
                    Type::ParamSpecValue(paramlist) => {
                        let params = mem::take(paramlist).prepend_types(ts).into_owned();
                        let new_callable = new_callable(Callable::list(params, ret.clone()));
                        *x = new_callable;
                    }
                    Type::Ellipsis if ts.is_empty() => {
                        *x = new_callable(Callable::ellipsis(ret.clone()));
                    }
                    Type::Concatenate(ts2, pspec) => {
                        *x = new_callable(Callable::concatenate(
                            ts.iter().chain(ts2.iter()).cloned().collect(),
                            (**pspec).clone(),
                            ret.clone(),
                        ));
                    }
                    _ => {}
                }
            } else if let Some(Callable {
                params: Params::List(param_list),
                ret: _,
            }) = callable
            {
                // When a Varargs has a concrete unpacked tuple, expand it to positional-only params
                // e.g., (*args: Unpack[tuple[int, str]]) -> (int, str, /)
                let mut new_params = Vec::new();
                for param in mem::take(param_list).into_items() {
                    match param {
                        Param::Varargs(_, Type::Unpack(box Type::Tuple(Tuple::Concrete(elts)))) => {
                            for elt in elts {
                                new_params.push(Param::PosOnly(None, elt, Required::Required));
                            }
                        }
                        _ => new_params.push(param),
                    }
                }
                *param_list = ParamList::new(new_params);
            }
        });
    }

    /// In unions, convert any Variable::Unsolved without a default into Never.
    /// See test::generic_basic::test_typevar_or_none for why we need to do this.
    fn erase_unsolved_variables(&self, t: &mut Type) {
        t.transform_mut(&mut |x| match x {
            Type::Union(box Union { members: xs, .. }) => {
                let erase_type = |x: &Type| match x {
                    Type::Var(v) => {
                        let lock = self.variables.lock();
                        let variable = lock.get(*v);
                        match &*variable {
                            Variable::PartialQuantified(q) => {
                                let erase = q.default.is_none();
                                drop(variable);
                                drop(lock);
                                erase
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                };
                let mut erase_xs = Vec::new();
                // We only want to erase variables from the union if
                // (1) there is at least one variable to erase, and
                // (2) we don't erase the entire union.
                let mut should_erase = false;
                for x in xs.iter() {
                    let erase = erase_type(x);
                    if let Some(prev) = erase_xs.last()
                        && *prev != erase
                    {
                        should_erase = true;
                    }
                    erase_xs.push(erase);
                }
                if should_erase {
                    for (x, erase) in xs.iter_mut().zip(erase_xs) {
                        if erase {
                            *x = self.heap.mk_never();
                        }
                    }
                }
            }
            _ => {}
        })
    }

    /// Like [`expand`], but also forces variables that haven't yet been bound
    /// to become `Any`, both in the result and in the `Solver` going forward.
    /// Guarantees there will be no `Var` in the result.
    ///
    /// In addition, if the type exceeds a large depth, it will be replaced with `Any`.
    pub fn deep_force(&self, mut t: Type) -> Type {
        self.deep_force_mut(&mut t);
        t
    }

    /// Generate a fresh variable based on code that is unspecified inside a container,
    /// e.g. `[]` with an unknown type of element.
    /// The `range` parameter is the location of the empty container literal.
    pub fn fresh_partial_contained(&self, uniques: &UniqueFactory, range: TextRange) -> Var {
        let v = Var::new(uniques);
        self.variables
            .lock()
            .insert_fresh(v, Variable::PartialContained(range));
        v
    }

    // Generate a fresh variable used to decompose a type, e.g. getting T from Awaitable[T]
    // Also used for lambda parameters, where the var is created during bindings, but solved during
    // the answers phase by contextually typing against an annotation.
    pub fn fresh_unwrap(&self, uniques: &UniqueFactory) -> Var {
        let v = Var::new(uniques);
        self.variables
            .lock()
            .insert_fresh(v, Variable::Unwrap(Bounds::new()));
        v
    }

    fn fresh_quantified_vars(
        &self,
        qs: &[&Quantified],
        uniques: &UniqueFactory,
    ) -> QuantifiedHandle {
        let vs = qs.map(|_| Var::new(uniques));
        let mut lock = self.variables.lock();
        for (v, q) in vs.iter().zip(qs.iter()) {
            lock.insert_fresh(
                *v,
                Variable::Quantified {
                    quantified: (*q).clone(),
                    bounds: Bounds::new(),
                },
            );
        }
        QuantifiedHandle(vs)
    }

    /// Generate fresh variables and substitute them in replacing a `Forall`.
    pub fn fresh_quantified(
        &self,
        params: &TParams,
        t: Type,
        uniques: &UniqueFactory,
    ) -> (QuantifiedHandle, Type) {
        if params.is_empty() {
            return (QuantifiedHandle::empty(), t);
        }

        let qs = params.iter().collect::<Vec<_>>();
        let vs = self.fresh_quantified_vars(&qs, uniques);
        let ts = vs.0.map(|v| v.to_type(&self.heap));
        let t = t.subst(&qs.into_iter().zip(&ts).collect());
        (vs, t)
    }

    /// Partially instantiate a generic function using the first argument.
    /// Mainly, we use this to create a function type from a bound function,
    /// but also for calling the staticmethod `__new__`.
    ///
    /// Unlike fresh_quantified, which creates vars for every tparam, we only
    /// instantiate the tparams that appear in the first parameter.
    ///
    /// Returns a callable with the first parameter removed, substituted with
    /// instantiations provided by applying the first argument.
    pub fn instantiate_callable_self(
        &self,
        tparams: &TParams,
        self_obj: &Type,
        self_param: &Type,
        mut callable: Callable,
        uniques: &UniqueFactory,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Callable {
        // Collect tparams that appear in the first parameter.
        let mut qs = Vec::new();
        self_param.for_each_quantified(&mut |q| {
            if tparams.iter().any(|tparam| *tparam == *q) {
                qs.push(q);
            }
        });

        if qs.is_empty() {
            return callable;
        }

        // Substitute fresh vars for the quantifieds in the self param.
        let vs = self.fresh_quantified_vars(&qs, uniques);
        let ts = vs.0.map(|v| v.to_type(&self.heap));
        let mp = qs.into_iter().zip(&ts).collect();
        let self_param = self_param.clone().subst(&mp);
        callable.visit_mut(&mut |t| t.subst_mut(&mp));
        drop(mp);

        // Solve for the vars created above.
        is_subset(self_obj, &self_param);

        // Either we have solutions, or we fall back to Any. We don't want Variable::Partial.
        // If this errors, then the definition is invalid, and we should have raised an error at
        // the definition site.
        let _specialization_errors = self.finish_quantified(vs, false);

        callable
    }

    pub fn has_instantiation_errors(&self, vs: &QuantifiedHandle) -> bool {
        let lock = self.instantiation_errors.read();
        vs.0.iter().any(|v| lock.contains_key(v))
    }

    /// Have these vars picked up any new instantiation errors since they were snapshotted?
    pub fn has_new_instantiation_errors(&self, snapshot: &VarSnapshot) -> bool {
        let lock = self.instantiation_errors.read();
        snapshot
            .0
            .iter()
            .any(|(v, state)| state.error.is_none() && lock.contains_key(v))
    }

    /// Returns true if the given type is a Var that points to a partial
    /// (PartialQuantified or PartialContained) variable.
    pub fn is_partial(&self, ty: &Type) -> bool {
        if let Type::Var(v) = ty {
            matches!(
                *self.variables.lock().get(*v),
                Variable::PartialQuantified(_) | Variable::PartialContained(_)
            )
        } else {
            false
        }
    }

    /// Add a bound to the variable if it is a Quantified or Unwrap
    fn get_new_bound(
        &self,
        existing_bound: Option<Type>,
        bound: Type,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> NewBound {
        // Check if the new bound can absorb or be absorbed into the existing bound.
        // Examples: `float` absorbs `int`, `list[Any]` absorbs `list[int]`.
        // TODO(https://github.com/facebook/pyrefly/issues/105): there are a few fishy things:
        // * We're only checking against the first bound.
        // * We're keeping `Any` separate so it can be filtered out in `solve_one_bounds`.
        // * We're relying on `is_subset` to pin vars.
        let updated_bound = existing_bound.and_then(|first| {
            let can_absorb = |t: &Type| !t.is_any() && t.collect_all_vars().is_empty();
            if !can_absorb(&first) || !can_absorb(&bound) {
                let _ = is_subset(&bound, &first); // Ignore the result, just pin vars
                None
            } else if is_subset(&bound.materialize(), &first).is_ok() {
                Some(first)
            } else if is_subset(&first.materialize(), &bound).is_ok() {
                Some(bound.clone())
            } else {
                None
            }
        });
        if let Some(updated_bound) = updated_bound {
            NewBound::UpdateExistingBound(updated_bound)
        } else {
            NewBound::AddBound(bound)
        }
    }

    fn add_bound(&self, bounds: &mut Vec<Type>, bound: NewBound) {
        match bound {
            NewBound::UpdateExistingBound(new_first) => {
                if let Some(old_first) = bounds.first_mut() {
                    *old_first = new_first;
                } else {
                    *bounds = vec![new_first];
                }
            }
            NewBound::AddBound(bound) => {
                bounds.push(bound);
            }
        }
    }

    fn validate_bound_consistency(
        &self,
        bound: &Type,
        existing_bounds: &Vec<Type>,
        kind: QuantifiedKind,
    ) -> Result<(), SubsetError> {
        if kind == QuantifiedKind::TypeVarTuple
            && let Type::Tuple(Tuple::Concrete(elts)) = bound
        {
            // Validate that the tuple length is consistent.
            for t in existing_bounds {
                if let Type::Tuple(Tuple::Concrete(existing_elts)) = t {
                    if elts.len() == existing_elts.len() {
                        // We only need to validate against the first tuple encountered.
                        // If subsequent ones are a different length, we would've already reported
                        // a violation when adding them.
                        return Ok(());
                    } else {
                        return Err(SubsetError::Other);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn add_lower_bound(
        &self,
        v: Var,
        bound: Type,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), SubsetError> {
        let lock = self.variables.lock();
        let e = lock.get(v);
        let (first_bound, upper_bound, res) = match &*e {
            Variable::Quantified {
                quantified: _,
                bounds,
            }
            | Variable::Unwrap(bounds) => (
                bounds.lower.first().cloned(),
                self.get_current_bound(bounds.upper.clone()),
                if let Variable::Quantified { quantified, .. } = &*e {
                    self.validate_bound_consistency(&bound, &bounds.lower, quantified.kind())
                } else {
                    Ok(())
                },
            ),
            _ => return Ok(()),
        };
        drop(e);
        drop(lock);
        let res = res.and_then(|_| {
            upper_bound.map_or(Ok(()), |upper_bound| is_subset(&bound, &upper_bound))
        });
        let new_bound = if res.is_ok() {
            self.get_new_bound(first_bound, bound, is_subset)
        } else {
            // TODO(https://github.com/facebook/pyrefly/issues/105): don't throw away the bound.
            NewBound::AddBound(Type::any_error())
        };
        let lock = self.variables.lock();
        match &mut *lock.get_mut(v) {
            Variable::Quantified {
                quantified: _,
                bounds,
            }
            | Variable::Unwrap(bounds) => self.add_bound(&mut bounds.lower, new_bound),
            _ => {}
        }
        res
    }

    pub fn add_upper_bound(
        &self,
        v: Var,
        bound: Type,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), SubsetError> {
        let lock = self.variables.lock();
        let e = lock.get(v);
        let (first_bound, lower_bound, res) = match &*e {
            Variable::Quantified {
                quantified: _,
                bounds,
            }
            | Variable::Unwrap(bounds) => (
                bounds.upper.first().cloned(),
                self.get_current_bound(bounds.lower.clone()),
                if let Variable::Quantified { quantified, .. } = &*e {
                    self.validate_bound_consistency(&bound, &bounds.upper, quantified.kind())
                } else {
                    Ok(())
                },
            ),
            _ => return Ok(()),
        };
        drop(e);
        drop(lock);
        let res = res.and_then(|_| {
            lower_bound.map_or(Ok(()), |lower_bound| is_subset(&lower_bound, &bound))
        });
        let new_bound = if res.is_ok() {
            self.get_new_bound(first_bound, bound, is_subset)
        } else {
            // TODO(https://github.com/facebook/pyrefly/issues/105): don't throw away the bound.
            NewBound::AddBound(Type::any_error())
        };
        let lock = self.variables.lock();
        match &mut *lock.get_mut(v) {
            Variable::Quantified {
                quantified: _,
                bounds,
            }
            | Variable::Unwrap(bounds) => self.add_bound(&mut bounds.upper, new_bound),
            _ => {}
        }
        res
    }

    /// Get current bound from a set of bounds of an unfinished variable.
    fn get_current_bound(&self, bounds: Vec<Type>) -> Option<Type> {
        if bounds.is_empty() {
            return None;
        }
        Some(unions(bounds, &self.heap))
    }

    /// Solve one set of bounds (upper or lower)
    fn solve_one_bounds(&self, mut bounds: Vec<Type>) -> Option<Type> {
        if bounds.is_empty() {
            return None;
        }
        // Keeping `Any` bounds causes `Any` to propagate to too many places,
        // so we filter them out unless `Any` is the only solution.
        if bounds.iter().any(|t| !t.is_any()) {
            bounds.retain(|t| !t.is_any());
        }
        Some(unions(bounds, &self.heap))
    }

    fn solve_bounds(&self, bounds: Bounds) -> Option<Type> {
        // Prefer non-Any bound > Any bound > no bound
        let lower_bound = self.solve_one_bounds(bounds.lower);
        if lower_bound.as_ref().is_none_or(|b| b.is_any()) {
            self.solve_one_bounds(bounds.upper).or(lower_bound)
        } else {
            lower_bound
        }
    }

    /// Called after a quantified function has been called. Given `def f[T](x: int): list[T]`,
    /// after the generic has completed.
    /// If `infer_with_first_use` is true, the variable `T` will be have like an
    /// empty container and get pinned by the first subsequent usage.
    /// If `infer_with_first_use` is false, the variable `T` will be replaced with `Any`
    pub fn finish_quantified(
        &self,
        vs: QuantifiedHandle,
        infer_with_first_use: bool,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        let lock = self.variables.lock();
        let mut err = Vec::new();
        for v in vs.0 {
            let mut e = lock.get_mut(v);
            match &mut *e {
                Variable::Answer(_) => {
                    // We pin the quantified var to a type when it first appears in a subset constraint,
                    // and at that point we check the instantiation with the bound.
                    if let Some(e) = self.instantiation_errors.read().get(&v) {
                        err.push(e.clone());
                    }
                }
                Variable::Quantified {
                    quantified: q,
                    bounds,
                } => {
                    if let Some(e) = self.instantiation_errors.read().get(&v) {
                        err.push(e.clone());
                    }
                    *e = if let Some(bound) = self.solve_bounds(mem::take(bounds)) {
                        Variable::Answer(bound)
                    } else if infer_with_first_use {
                        Variable::finished(q)
                    } else {
                        Variable::Answer(q.as_gradual_type())
                    };
                }
                _ => {}
            }
        }
        match Vec1::try_from_vec(err) {
            Ok(err) => Err(err),
            Err(_) => Ok(()),
        }
    }

    pub fn finish_all_quantified(&self, ty: &Type) -> Result<(), Vec1<TypeVarSpecializationError>> {
        let vs = QuantifiedHandle(ty.collect_maybe_placeholder_vars());
        self.finish_quantified(vs, self.infer_with_first_use)
    }

    /// Given targs which contain quantified (as come from `instantiate`), replace the quantifieds
    /// with fresh vars. We can avoid substitution because tparams can not appear in the bounds of
    /// another tparam. tparams can appear in the default, but those are not in quantified form yet.
    pub fn freshen_class_targs(
        &self,
        targs: &mut TArgs,
        uniques: &UniqueFactory,
    ) -> QuantifiedHandle {
        let mut vs = Vec::new();
        let mut lock = self.variables.lock();
        targs.iter_paired_mut().for_each(|(param, t)| {
            if let Type::Quantified(q) = t
                && **q == *param
            {
                let v = Var::new(uniques);
                vs.push(v);
                *t = v.to_type(&self.heap);
                lock.insert_fresh(
                    v,
                    Variable::Quantified {
                        quantified: param.clone(),
                        bounds: Bounds::new(),
                    },
                );
            }
        });
        QuantifiedHandle(vs)
    }

    /// Solve each fresh var created in freshen_class_targs. If we still have a Var, we do not
    /// yet have an instantiation, but one might come later. E.g., __new__ did not provide an
    /// instantiation, but __init__ will.
    pub fn generalize_class_targs(&self, targs: &mut TArgs) {
        // Expanding targs might require the variables lock, so do that first.
        targs
            .as_mut()
            .iter_mut()
            .for_each(|t| self.expand_vars_mut(t));
        let lock = self.variables.lock();
        targs.iter_paired_mut().for_each(|(param, t)| {
            if let Type::Var(v) = t {
                let mut e = lock.get_mut(*v);
                if let Variable::Quantified {
                    quantified: q,
                    bounds,
                } = &mut *e
                    && *q == *param
                {
                    if bounds.is_empty() {
                        *t = param.clone().to_type(&self.heap);
                    } else {
                        // If the variable has already been solved, finalize its type now.
                        *e = Variable::Answer(
                            self.solve_bounds(mem::take(bounds))
                                .unwrap_or_else(|| q.as_gradual_type()),
                        );
                    }
                }
            }
        })
    }

    /// Finalize the tparam instantiations. Any targs which don't yet have an instantiation
    /// will resolve to their default, if one exists. Otherwise, create a "partial" var and
    /// try to find an instantiation at the first use, like finish_quantified.
    pub fn finish_class_targs(&self, targs: &mut TArgs, uniques: &UniqueFactory) {
        // The default can refer to a tparam from earlier in the list, so we maintain a
        // small scope data structure during the traversal.
        let mut seen_params = SmallMap::new();
        let mut new_targs: Vec<Option<Type>> = Vec::with_capacity(targs.len());
        targs.iter_paired().enumerate().for_each(|(i, (param, t))| {
            let new_targ = if let Type::Quantified(q) = t
                && **q == *param
            {
                if let Some(default) = param.default() {
                    // Note that TypeVars are stored in Type::TypeVar form, and have not yet been
                    // converted to Quantified form, so we do that now.
                    // TODO: deal with code duplication in get_tparam_default
                    let mut t = default.clone();
                    t.transform_mut(&mut |t| {
                        let name = match t {
                            Type::TypeVar(t) => Some(t.qname().id()),
                            Type::TypeVarTuple(t) => Some(t.qname().id()),
                            Type::ParamSpec(p) => Some(p.qname().id()),
                            Type::Quantified(q) => Some(q.name()),
                            _ => None,
                        };
                        if let Some(name) = name {
                            *t = if let Some(i) = seen_params.get(name) {
                                let new_targ: &Option<Type> = &new_targs[*i];
                                new_targ
                                    .as_ref()
                                    .unwrap_or_else(|| &targs.as_slice()[*i])
                                    .clone()
                            } else {
                                param.as_gradual_type()
                            }
                        }
                    });
                    Some(t)
                } else if self.infer_with_first_use {
                    let v = Var::new(uniques);
                    self.variables.lock().insert_fresh(v, Variable::finished(q));
                    Some(v.to_type(&self.heap))
                } else {
                    Some(q.as_gradual_type())
                }
            } else {
                None
            };
            seen_params.insert(param.name(), i);
            new_targs.push(new_targ);
        });
        drop(seen_params);
        new_targs
            .into_iter()
            .zip(targs.as_mut().iter_mut())
            .for_each(|(new_targ, targ)| {
                if let Some(new_targ) = new_targ {
                    *targ = new_targ;
                }
            })
    }

    /// Generate a fresh variable used to tie recursive bindings.
    pub fn fresh_recursive(&self, uniques: &UniqueFactory) -> Var {
        let v = Var::new(uniques);
        self.variables.lock().insert_fresh(v, Variable::Recursive);
        v
    }

    pub fn for_display(&self, mut t: Type) -> Type {
        self.expand_with_limit(&mut t, TYPE_LIMIT, &VarRecurser::new(), true);
        self.simplify_mut(&mut t);
        t.deterministic_printing()
    }

    /// Generate an error message that `got <: want` failed.
    pub fn error(
        &self,
        got: &Type,
        want: &Type,
        errors: &ErrorCollector,
        loc: TextRange,
        tcc: &dyn Fn() -> TypeCheckContext,
        subset_error: SubsetError,
    ) {
        let tcc = tcc();
        let msg = tcc.kind.format_error(
            &self.for_display(got.clone()),
            &self.for_display(want.clone()),
            errors.module().name(),
        );
        let mut msg_lines = vec1![msg];
        if let Some(subset_error_msg) = subset_error.to_error_msg() {
            msg_lines.push(subset_error_msg);
        }
        let extra_annotations = tcc.annotations;
        match tcc.context {
            Some(ctx) => {
                errors.add_with_annotations(
                    loc,
                    ErrorInfo::Context(&|| ctx.clone()),
                    msg_lines,
                    extra_annotations,
                );
            }
            None => {
                errors.add_with_annotations(
                    loc,
                    ErrorInfo::Kind(tcc.kind.as_error_kind()),
                    msg_lines,
                    extra_annotations,
                );
            }
        }
    }

    /// Union a list of types together. In the process may cause some variables to be forced.
    pub fn unions<Ans: LookupAnswer>(
        &self,
        mut branches: Vec<Type>,
        type_order: TypeOrder<Ans>,
    ) -> Type {
        if branches.is_empty() {
            return self.heap.mk_never();
        }
        if branches.len() == 1 {
            return branches.pop().unwrap();
        }

        // We want to union modules differently, by merging their module sets
        let mut modules: SmallMap<Vec<Name>, ModuleType> = SmallMap::new();
        let mut branches = branches
            .into_iter()
            .filter_map(|x| match x {
                // Maybe we should force x before looking at it, but that causes issues with
                // recursive variables that we can't examine.
                // In practice unlikely anyone has a recursive variable which evaluates to a module.
                Type::Module(m) => {
                    match modules.entry(m.parts().to_owned()) {
                        Entry::Occupied(mut e) => {
                            e.get_mut().merge(&m);
                        }
                        Entry::Vacant(e) => {
                            e.insert(m);
                        }
                    }
                    None
                }
                t => Some(t),
            })
            .collect::<Vec<_>>();
        branches.extend(modules.into_values().map(Type::Module));
        unions_with_literals(
            branches,
            type_order.stdlib(),
            &|cls| type_order.get_enum_member_count(cls),
            &self.heap,
        )
    }

    /// Record a variable that is used recursively.
    pub fn record_recursive(&self, var: Var, ty: Type) -> Type {
        fn expand(
            t: Type,
            variables: &Variables,
            recurser: &VarRecurser,
            heap: &TypeHeap,
            res: &mut Vec<Type>,
        ) {
            match t {
                Type::Var(v) if let Some(_guard) = variables.recurse(v, recurser) => {
                    let variable = variables.get(v);
                    match &*variable {
                        Variable::Answer(t) => {
                            let t = t.clone();
                            drop(variable);
                            expand(t, variables, recurser, heap, res);
                        }
                        _ => res.push(v.to_type(heap)),
                    }
                }
                Type::Union(box Union { members: ts, .. }) => {
                    for t in ts {
                        expand(t, variables, recurser, heap, res);
                    }
                }
                _ => res.push(t),
            }
        }

        let lock = self.variables.lock();
        let variable = lock.get(var);
        match &*variable {
            Variable::Answer(forced) => {
                // An answer was already forced - use it, not the type from analysis.
                //
                // This can only happen in a fixpoint, and we'll catch it with a fixpoint non-convergence
                // error if it does not eventually converge.
                let forced = forced.clone();
                drop(variable);
                drop(lock);
                forced
            }
            _ => {
                drop(variable);
                // If you are recording `@1 = @1 | something` then the `@1` can't contribute any
                // possibilities, so just ignore it.
                let mut res = Vec::new();
                // First expand all union/var into a list of the possible unions
                expand(ty, &lock, &VarRecurser::new(), &self.heap, &mut res);
                // Then remove any reference to self, before unioning it back together
                res.retain(|x| x != &Type::Var(var));
                let ty = unions(res, &self.heap);
                lock.update(var, Variable::Answer(ty.clone()));
                ty
            }
        }
    }

    /// Is `got <: want`? If you aren't sure, return `false`.
    /// May cause partial variables to be resolved to an answer.
    pub fn is_subset_eq<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), SubsetError> {
        self.is_subset_eq_impl(got, want, type_order)
    }

    fn is_subset_eq_impl<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), SubsetError> {
        let mut subset = self.subset(type_order);
        subset.is_subset_eq(got, want)
    }

    pub fn is_consistent<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), SubsetError> {
        let mut subset = self.subset(type_order);
        subset.is_consistent(got, want)
    }

    pub fn is_equivalent<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), SubsetError> {
        let mut subset = self.subset(type_order);
        subset.is_equivalent(got, want)
    }

    fn subset<'a, Ans: LookupAnswer>(&'a self, type_order: TypeOrder<'a, Ans>) -> Subset<'a, Ans> {
        Subset {
            solver: self,
            type_order,
            gas: INITIAL_GAS,
            subset_cache: SmallMap::new(),
            class_protocol_assumptions: SmallSet::new(),
            coinductive_assumptions_used: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeVarSpecializationError {
    pub name: Name,
    pub got: Type,
    pub want: Type,
    #[allow(dead_code)]
    pub error: SubsetError,
}

impl TypeVarSpecializationError {
    pub fn to_error_msg<Ans: LookupAnswer>(self, ans: &AnswersSolver<Ans>) -> String {
        TypeCheckKind::TypeVarSpecialization(self.name).format_error(
            &ans.for_display(self.got),
            &ans.for_display(self.want),
            ans.module().name(),
        )
    }
}

#[derive(Debug, Clone)]
pub enum TypedDictSubsetError {
    /// TypedDict `got` is missing a field that `want` requires
    MissingField { got: Name, want: Name, field: Name },
    /// TypedDict field in `got` is ReadOnly but `want` requires read-write
    ReadOnlyMismatch { got: Name, want: Name, field: Name },
    /// TypedDict field in `got` is not required but `want` requires it
    RequiredMismatch { got: Name, want: Name, field: Name },
    /// TypedDict field in `got` is required cannot be, since it is `NotRequired` and read-write in `want`
    NotRequiredReadWriteMismatch { got: Name, want: Name, field: Name },
    /// TypedDict invariant field type mismatch (read-write fields must have exactly the same type)
    InvariantFieldMismatch {
        got: Name,
        got_field_ty: Type,
        want: Name,
        want_field_ty: Type,
        field: Name,
    },
    /// TypedDict covariant field type mismatch (readonly field type in `got` is not a subtype of `want`)
    CovariantFieldMismatch {
        got: Name,
        got_field_ty: Type,
        want: Name,
        want_field_ty: Type,
        field: Name,
    },
}

impl TypedDictSubsetError {
    fn to_error_msg(self) -> String {
        match self {
            TypedDictSubsetError::MissingField { got, want, field } => {
                format!("Field `{field}` is present in `{want}` and absent in `{got}`")
            }
            TypedDictSubsetError::ReadOnlyMismatch { got, want, field } => {
                format!("Field `{field}` is read-write in `{want}` but is `ReadOnly` in `{got}`")
            }
            TypedDictSubsetError::RequiredMismatch { got, want, field } => {
                format!("Field `{field}` is required in `{want}` but is `NotRequired` in `{got}`")
            }
            TypedDictSubsetError::NotRequiredReadWriteMismatch { got, want, field } => {
                format!(
                    "Field `{field}` is `NotRequired` and read-write in `{want}`, so it cannot be required in `{got}`"
                )
            }
            TypedDictSubsetError::InvariantFieldMismatch {
                got,
                got_field_ty,
                want,
                want_field_ty,
                field,
            } => format!(
                "Field `{field}` in `{got}` has type `{}`, which is not consistent with `{}` in `{want}` (read-write fields must have the same type)",
                got_field_ty.deterministic_printing(),
                want_field_ty.deterministic_printing()
            ),
            TypedDictSubsetError::CovariantFieldMismatch {
                got,
                got_field_ty,
                want,
                want_field_ty,
                field,
            } => format!(
                "Field `{field}` in `{got}` has type `{}`, which is not assignable to `{}`, the type of `{want}.{field}` (read-only fields are covariant)",
                got_field_ty.deterministic_printing(),
                want_field_ty.deterministic_printing()
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpenTypedDictSubsetError {
    /// `got` is missing a field in `want`
    MissingField { got: Name, want: Name, field: Name },
    /// `got` may contain unknown fields contradicting the `extra_items` type in `want`
    UnknownFields {
        got: Name,
        want: Name,
        extra_items: Type,
    },
}

impl OpenTypedDictSubsetError {
    fn to_error_msg(self) -> String {
        let (msg, got) = match self {
            Self::MissingField { got, want, field } => (
                format!(
                    "`{got}` is an open TypedDict with unknown extra items, which may include `{want}` item `{field}` with an incompatible type"
                ),
                got,
            ),
            Self::UnknownFields {
                got,
                want,
                extra_items: Type::Never(_),
            } => (
                format!(
                    "`{got}` is an open TypedDict with unknown extra items, which cannot be unpacked into closed TypedDict `{want}`",
                ),
                got,
            ),
            Self::UnknownFields {
                got,
                want,
                extra_items,
            } => (
                format!(
                    "`{got}` is an open TypedDict with unknown extra items, which may not be compatible with `extra_items` type `{}` in `{want}`",
                    extra_items.deterministic_printing(),
                ),
                got,
            ),
        };
        format!("{msg}. Hint: add `closed=True` to the definition of `{got}` to close it.")
    }
}

/// If a got <: want check fails, the failure reason
#[derive(Debug, Clone)]
pub enum SubsetError {
    /// The name of a positional parameter differs between `got` and `want`.
    PosParamName(Name, Name),
    /// Instantiations for quantified vars are incompatible with bounds
    TypeVarSpecialization(Vec1<TypeVarSpecializationError>),
    /// `got` is missing an attribute that the Protocol `want` requires
    /// The first element is the name of the protocol, the second is the name of the attribute
    MissingAttribute(Name, Name),
    /// Attribute in `got` is incompatible with the same attribute in Protocol `want`
    /// The first element is the name of `want, the second element is `got`, and the third element is the name of the attribute
    IncompatibleAttribute(Box<(Name, Type, Name, AttrSubsetError)>),
    /// TypedDict subset check failed
    TypedDict(Box<TypedDictSubsetError>),
    /// Errors involving arbitrary unknown fields in open TypedDicts
    OpenTypedDict(Box<OpenTypedDictSubsetError>),
    /// Tensor shape check failed
    TensorShape(ShapeError),
    /// An invariant was violated - used for cases that should be unreachable when - if there is ever a bug - we
    /// would prefer to not panic and get a text location for reproducing rather than just a crash report.
    /// Note: always use `ErrorCollector::internal_error` to log internal errors.
    InternalError(String),
    /// Protocol class names cannot be assigned to `type[P]` when `P` is a protocol
    TypeOfProtocolNeedsConcreteClass(Name),
    /// A `type` cannot accept special forms like `Callable`
    TypeCannotAcceptSpecialForms(SpecialForm),
    // TODO(rechen): replace this with specific reasons
    Other,
}

impl SubsetError {
    pub fn to_error_msg(self) -> Option<String> {
        match self {
            SubsetError::PosParamName(got, want) => Some(format!(
                "Positional parameter name mismatch: got `{got}`, want `{want}`"
            )),
            SubsetError::TypeVarSpecialization(_) => {
                // TODO
                None
            }
            SubsetError::MissingAttribute(protocol, attribute) => Some(format!(
                "Protocol `{protocol}` requires attribute `{attribute}`"
            )),
            SubsetError::IncompatibleAttribute(box (protocol, got, attribute, err)) => {
                Some(err.to_error_msg(&Name::new(format!("{got}")), &protocol, &attribute))
            }
            SubsetError::TypedDict(err) => Some(err.to_error_msg()),
            SubsetError::OpenTypedDict(err) => Some(err.to_error_msg()),
            SubsetError::TensorShape(err) => Some(err.to_string()),
            SubsetError::InternalError(msg) => Some(format!("Pyrefly internal error: {msg}")),
            SubsetError::TypeOfProtocolNeedsConcreteClass(want) => Some(format!(
                "Only concrete classes may be assigned to `type[{want}]` because `{want}` is a protocol"
            )),
            SubsetError::TypeCannotAcceptSpecialForms(form) => Some(format!(
                "`type` cannot accept special form `{}` as an argument",
                form
            )),
            SubsetError::Other => None,
        }
    }
}

/// Cached result for a recursive subset check. Used by `Subset::subset_cache`.
#[derive(Clone, Debug)]
pub(crate) enum SubsetCacheEntry {
    /// Currently being computed — used for coinductive cycle detection.
    /// Treated as `Ok(())`: if we encounter a pair already being checked,
    /// we optimistically assume the check succeeds (coinductive reasoning).
    InProgress,
    /// Computed and succeeded.
    Ok,
    /// Computed and failed.
    Err(SubsetError),
}

/// A helper to implement subset ergonomically.
/// Should only be used within `crate::subset`, which implements part of it.
pub struct Subset<'a, Ans: LookupAnswer> {
    pub(crate) solver: &'a Solver,
    pub type_order: TypeOrder<'a, Ans>,
    gas: Gas,
    /// Memoization cache for recursive subset checks (protocols and recursive type aliases).
    /// Doubles as a cycle detector: `InProgress` entries break cycles via coinductive
    /// reasoning by optimistically returning `Ok(())`.
    ///
    /// Unlike a stack-based cycle detector (which removes entries on return and forces
    /// re-computation from sibling call paths), this cache persists `Ok` results across
    /// the entire query, preventing exponential re-checking when the same `(got, want)`
    /// pair is encountered from multiple sibling call paths (e.g., different methods of
    /// a protocol each requiring the same structural subtype check).
    ///
    /// On failure, entries added *during* the failing computation are rolled back
    /// (popped from the end of the map back to the saved size), because intermediate
    /// `Ok` entries may have depended on a coinductive assumption that the failure
    /// invalidated. For example, if checking `A <: P1` internally succeeds on
    /// `A <: P2` (via the coinductive assumption that `A <: P1` holds) but then
    /// `A <: P1` ultimately fails, the cached success for `A <: P2` is unsound and
    /// must be discarded. Only entries added during the failing computation are
    /// removed; entries from earlier (independent) computations are preserved.
    /// This works because `SmallMap` preserves insertion order.
    pub subset_cache: SmallMap<(Type, Type), SubsetCacheEntry>,
    /// Class-level recursive assumptions for protocol checks.
    /// When checking `got <: protocol` where got's type arguments contain Vars
    /// (indicating we're in a recursive pattern), we track (got_class, protocol_class)
    /// pairs to detect cycles. This enables coinductive reasoning for recursive protocols
    /// like Functor/Maybe without falsely assuming success for unrelated protocol checks.
    pub class_protocol_assumptions: SmallSet<(Class, Class)>,
    /// Tracks whether a coinductive assumption (InProgress → Ok) was used during
    /// the current computation. Used to avoid caching protocol results in the
    /// persistent cross-call cache when they depend on coinductive assumptions.
    pub coinductive_assumptions_used: bool,
}

impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
    pub fn is_consistent(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.is_subset_eq(got, want)?;
        self.is_subset_eq(want, got)
    }

    pub fn is_equivalent(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.is_consistent(&got.materialize(), want)?;
        self.is_consistent(got, &want.materialize())
    }

    pub fn is_subset_eq(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        if self.gas.stop() {
            // We really have no idea. Just give up for now.
            return Err(SubsetError::Other);
        }
        if matches!(got, Type::Materialization) {
            return self.is_subset_eq(
                &self
                    .solver
                    .heap
                    .mk_class_type(self.type_order.stdlib().object().clone()),
                want,
            );
        } else if matches!(want, Type::Materialization) {
            return self.is_subset_eq(got, &self.solver.heap.mk_never());
        }
        let res = self.is_subset_eq_var(got, want);
        self.gas.restore();
        res
    }

    /// For a constrained TypeVar, find the narrowest constraint that `ty` is assignable to.
    ///
    /// Per the typing spec, a constrained TypeVar (`T = TypeVar("T", int, str)`) must resolve
    /// to exactly one of its constraint types — never a subtype like `bool` or `Literal[42]`.
    /// This method finds the best (narrowest) matching constraint by checking assignability
    /// and preferring the most specific constraint when multiple match.
    fn find_matching_constraint<'c>(
        &mut self,
        ty: &Type,
        constraints: &'c [Type],
    ) -> Option<&'c Type> {
        if ty.is_any() {
            return None;
        }
        let matching: Vec<&Type> = constraints
            .iter()
            .filter(|c| self.is_subset_eq(ty, c).is_ok())
            .collect();
        if matching.is_empty() {
            return None;
        }
        // Pick the narrowest matching constraint: the one that is a subtype of all others.
        let mut best = matching[0];
        for &candidate in &matching[1..] {
            if self.is_subset_eq(candidate, best).is_ok() {
                best = candidate;
            }
        }
        Some(best)
    }

    /// is_subset_eq_var(t1, Quantified)
    fn is_subset_eq_quantified(
        &mut self,
        t1: &Type,
        q: &Quantified,
        upper_bound: Option<&Type>,
    ) -> (Type, Option<TypeVarSpecializationError>) {
        let t1_p = {
            let t1_p = t1
                .clone()
                .promote_implicit_literals(self.type_order.stdlib());
            if let Some(upper_bound) = upper_bound {
                // Don't promote literals if doing so would violate a literal upper bound.
                if self.is_subset_eq(&t1_p, upper_bound).is_ok() {
                    t1_p
                } else {
                    t1.clone()
                }
            } else {
                t1_p
            }
        };
        let bound = q.bound_type(self.type_order.stdlib(), &self.solver.heap);
        // For constrained TypeVars, promote to the matching constraint type.
        if let Restriction::Constraints(ref constraints) = q.restriction {
            // Try promoted type first, then fall back to original (for literal bounds).
            if let Some(constraint) = self.find_matching_constraint(&t1_p, constraints) {
                (constraint.clone(), None)
            } else if let Some(constraint) = self.find_matching_constraint(t1, constraints) {
                (constraint.clone(), None)
            } else if let Err(err_p) = self.is_subset_eq(&t1_p, &bound) {
                // No individual constraint matched, but the type may still
                // be assignable to the constraint union (e.g. an abstract
                // `AnyStr` satisfies `str | bytes`). Fall back to bound
                // checking, mirroring the non-constraint code path.
                if self.is_subset_eq(t1, &bound).is_err() {
                    let specialization_error = TypeVarSpecializationError {
                        name: q.name().clone(),
                        got: t1_p.clone(),
                        want: bound,
                        error: err_p,
                    };
                    (t1_p.clone(), Some(specialization_error))
                } else {
                    (t1.clone(), None)
                }
            } else {
                (t1_p.clone(), None)
            }
        } else if let Err(err_p) = self.is_subset_eq(&t1_p, &bound) {
            // If the promoted type fails, try again with the original type, in case the bound itself is literal.
            // This could be more optimized, but errors are rare, so this code path should not be hot.
            if self.is_subset_eq(t1, &bound).is_err() {
                // If the original type is also an error, use the promoted type.
                let specialization_error = TypeVarSpecializationError {
                    name: q.name().clone(),
                    got: t1_p.clone(),
                    want: bound,
                    error: err_p,
                };
                (t1_p.clone(), Some(specialization_error))
            } else {
                (t1.clone(), None)
            }
        } else {
            (t1_p.clone(), None)
        }
    }

    /// Implementation of Var subset cases, calling onward to solve non-Var cases.
    ///
    /// This function does two things: it checks that got <: want, and it solves free variables assuming that
    /// got <: want.
    ///
    /// Before solving, for Quantified and Partial variables we will generally
    /// promote literals when a variable appears on the left side of an
    /// inequality, but not when it is on the left. This means that, e.g.:
    /// - if `f[T](x: T) -> T: ...`, then `f(1)` gets solved to `int`
    /// - if `f(x: Literal[0]): ...`, then `x = []; f(x[0])` results in `x: list[Literal[0]]`
    fn is_subset_eq_var(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        match (got, want) {
            _ if got == want => Ok(()),
            (Type::Var(v1), Type::Var(v2)) => {
                let variables = self.solver.variables.lock();
                // Variable unification is destructive, so we have to copy bounds first.
                let root1 = variables.get_root(*v1);
                let root2 = variables.get_root(*v2);
                if root1 == root2 {
                    // same variable after unification, nothing to do
                } else {
                    let mut v1_mut = variables.get_mut(*v1);
                    let mut v2_mut = variables.get_mut(*v2);
                    match (&mut *v1_mut, &mut *v2_mut) {
                        (
                            Variable::Quantified {
                                quantified: _,
                                bounds: v1_bounds,
                            }
                            | Variable::Unwrap(v1_bounds),
                            Variable::Quantified {
                                quantified: _,
                                bounds: v2_bounds,
                            }
                            | Variable::Unwrap(v2_bounds),
                        ) => {
                            v1_bounds.extend(mem::take(v2_bounds));
                            *v2_bounds = v1_bounds.clone();
                        }
                        _ => {}
                    }
                    drop(v1_mut);
                    drop(v2_mut);
                }

                let variable1 = variables.get(*v1);
                let variable2 = variables.get(*v2);
                match (&*variable1, &*variable2) {
                    (Variable::Answer(t1), Variable::Answer(t2)) => {
                        let t1 = t1.clone();
                        let t2 = t2.clone();
                        drop(variable1);
                        drop(variable2);
                        drop(variables);
                        self.is_subset_eq(&t1, &t2)
                    }
                    (_, Variable::Answer(t2)) => {
                        let t2 = t2.clone();
                        drop(variable1);
                        drop(variable2);
                        drop(variables);
                        self.is_subset_eq(got, &t2)
                    }
                    (Variable::Answer(t1), _) => {
                        let t1 = t1.clone();
                        drop(variable1);
                        drop(variable2);
                        drop(variables);
                        self.is_subset_eq(&t1, want)
                    }
                    // When both variables are quantified, we need to preserve the stricter bound.
                    // The `unify` function preserves the Variable data from its second argument,
                    // so we call it with the stricter bound in the v2 position.
                    (
                        Variable::Quantified {
                            quantified: q1,
                            bounds: _,
                        },
                        Variable::Quantified {
                            quantified: q2,
                            bounds: _,
                        },
                    )
                    | (Variable::PartialQuantified(q1), Variable::PartialQuantified(q2)) => {
                        let r1_restricted = q1.restriction().is_restricted();
                        let r2_restricted = q2.restriction().is_restricted();
                        let b1 = q1.bound_type(self.type_order.stdlib(), &self.solver.heap);
                        let b2 = q2.bound_type(self.type_order.stdlib(), &self.solver.heap);
                        drop(variable1);
                        drop(variable2);

                        match (r1_restricted, r2_restricted) {
                            (false, false) => {
                                // Neither has a restriction, order doesn't matter
                                variables.unify(*v1, *v2);
                            }
                            (true, false) => {
                                // Only v1 has a restriction, preserve v1's data
                                variables.unify(*v2, *v1);
                            }
                            (false, true) => {
                                // Only v2 has a restriction, preserve v2's data
                                variables.unify(*v1, *v2);
                            }
                            (true, true) => {
                                // Both have restrictions, need to compare bounds
                                drop(variables);

                                let b1_subtype_of_b2 = self.is_subset_eq(&b1, &b2).is_ok();
                                let b2_subtype_of_b1 = self.is_subset_eq(&b2, &b1).is_ok();

                                // Unify in the correct order to preserve the stricter bound.
                                // unify(x, y) preserves y's Variable data.
                                if b1_subtype_of_b2 && b2_subtype_of_b1 {
                                    // Bounds are equivalent, order doesn't matter
                                    self.solver.variables.lock().unify(*v1, *v2);
                                } else if b1_subtype_of_b2 {
                                    // b1 is stricter (subtype of b2), preserve v1's data
                                    self.solver.variables.lock().unify(*v2, *v1);
                                } else if b2_subtype_of_b1 {
                                    // b2 is stricter (subtype of b1), preserve v2's data
                                    self.solver.variables.lock().unify(*v1, *v2);
                                } else {
                                    // Bounds are incompatible
                                    return Err(SubsetError::Other);
                                }
                            }
                        }
                        Ok(())
                    }
                    (
                        _,
                        Variable::Quantified {
                            quantified: _,
                            bounds: _,
                        },
                    ) => {
                        drop(variable1);
                        drop(variable2);
                        // `unify` preserves the Variable in its second argument. When a Quantified
                        // and a non-Quantified are unified, we preserve the non-Quantified to
                        // avoid leaking unsolved type parameters across bindings.
                        variables.unify(*v2, *v1);
                        Ok(())
                    }
                    (_, _) => {
                        drop(variable1);
                        drop(variable2);
                        variables.unify(*v1, *v2);
                        Ok(())
                    }
                }
            }
            (Type::Var(v1), t2) => {
                let variables = self.solver.variables.lock();
                let v1_ref = variables.get(*v1);
                match &*v1_ref {
                    Variable::Answer(t1) => {
                        let t1 = t1.clone();
                        drop(v1_ref);
                        drop(variables);
                        self.is_subset_eq(&t1, t2)
                    }
                    Variable::Quantified {
                        quantified: q,
                        bounds: _,
                    } if q.kind() == QuantifiedKind::ParamSpec => {
                        // TODO(https://github.com/facebook/pyrefly/issues/105): figure out what to
                        // do with ParamSpec.
                        drop(v1_ref);
                        variables.update(*v1, Variable::Answer(t2.clone()));
                        Ok(())
                    }
                    Variable::Quantified { .. } | Variable::Unwrap(_) => {
                        drop(v1_ref);
                        drop(variables);
                        self.solver
                            .add_upper_bound(*v1, t2.clone(), &mut |got, want| {
                                self.is_subset_eq(got, want)
                            })
                    }
                    Variable::PartialQuantified(q) => {
                        let name = q.name.clone();
                        let restriction = q.restriction().clone();
                        let bound = q.bound_type(self.type_order.stdlib(), &self.solver.heap);
                        drop(v1_ref);

                        // For constrained TypeVars, promote to the matching constraint type
                        // rather than pinning to the raw argument type.
                        if let Restriction::Constraints(ref constraints) = restriction {
                            variables.update(*v1, Variable::Answer(t2.clone()));
                            drop(variables);
                            if let Some(constraint) = self.find_matching_constraint(t2, constraints)
                            {
                                let constraint = constraint.clone();
                                self.solver
                                    .variables
                                    .lock()
                                    .update(*v1, Variable::Answer(constraint));
                            } else if let Err(e) = self.is_subset_eq(t2, &bound) {
                                // No individual constraint matched, but the type may still
                                // be assignable to the constraint union (e.g. an abstract
                                // `AnyStr` satisfies `str | bytes`). Only error if it fails
                                // the union bound check too.
                                self.solver.instantiation_errors.write().insert(
                                    *v1,
                                    TypeVarSpecializationError {
                                        name,
                                        got: t2.clone(),
                                        want: bound,
                                        error: e,
                                    },
                                );
                            }
                        } else {
                            variables.update(*v1, Variable::Answer(t2.clone()));
                            drop(variables);
                            if let Err(e) = self.is_subset_eq(t2, &bound) {
                                self.solver.instantiation_errors.write().insert(
                                    *v1,
                                    TypeVarSpecializationError {
                                        name,
                                        got: t2.clone(),
                                        want: bound,
                                        error: e,
                                    },
                                );
                            }
                        }
                        // Widen None to None | Any for PartialQuantified, matching
                        // the PartialContained behavior (see comment there).
                        let variables = self.solver.variables.lock();
                        let v1_current = variables.get(*v1);
                        if let Variable::Answer(t) = &*v1_current
                            && t.is_none()
                        {
                            let widened = self
                                .solver
                                .heap
                                .mk_union(vec![t.clone(), Type::any_implicit()]);
                            drop(v1_current);
                            variables.update(*v1, Variable::Answer(widened));
                        }
                        Ok(())
                    }
                    Variable::PartialContained(_) => {
                        drop(v1_ref);
                        // When an empty container's element is pinned to None, widen to
                        // None | Any. A bare None in the first use almost always means the
                        // container will later hold some other (unknown) type, analogous
                        // to how `self.x = None` is inferred as `None | Any` for attributes.
                        let answer = if t2.is_none() {
                            self.solver
                                .heap
                                .mk_union(vec![t2.clone(), Type::any_implicit()])
                        } else {
                            t2.clone()
                        };
                        variables.update(*v1, Variable::Answer(answer));
                        Ok(())
                    }
                    Variable::Recursive => {
                        drop(v1_ref);
                        variables.update(*v1, Variable::Answer(t2.clone()));
                        Ok(())
                    }
                }
            }
            (t1, Type::Var(v2)) => {
                let variables = self.solver.variables.lock();
                let v2_ref = variables.get(*v2);
                match &*v2_ref {
                    Variable::Answer(t2) => {
                        let t2 = t2.clone();
                        drop(v2_ref);
                        drop(variables);
                        self.is_subset_eq(t1, &t2)
                    }
                    Variable::Quantified {
                        quantified: q,
                        bounds,
                    } => {
                        let q = q.clone();
                        let upper_bound = self.solver.get_current_bound(bounds.upper.clone());
                        drop(v2_ref);
                        drop(variables);
                        let (answer, specialization_error) =
                            self.is_subset_eq_quantified(t1, &q, upper_bound.as_ref());
                        if let Some(specialization_error) = specialization_error {
                            self.solver
                                .instantiation_errors
                                .write()
                                .insert(*v2, specialization_error);
                        }
                        if q.kind() == QuantifiedKind::ParamSpec
                            || matches!(q.restriction(), Restriction::Constraints(_))
                        {
                            // If the TypeVar has constraints, we write the answer immediately to
                            // enforce that we always match the same constraint.
                            //
                            // TODO(https://github.com/facebook/pyrefly/issues/105): figure out
                            // what to do with ParamSpec.
                            self.solver
                                .variables
                                .lock()
                                .update(*v2, Variable::Answer(answer));
                            Ok(())
                        } else {
                            self.solver.add_lower_bound(*v2, answer, &mut |got, want| {
                                self.is_subset_eq(got, want)
                            })
                        }
                    }
                    Variable::PartialQuantified(q) => {
                        let q = q.clone();
                        drop(v2_ref);
                        drop(variables);
                        let (answer, specialization_error) =
                            self.is_subset_eq_quantified(t1, &q, None);
                        if let Some(specialization_error) = specialization_error {
                            self.solver
                                .instantiation_errors
                                .write()
                                .insert(*v2, specialization_error);
                        }
                        // Widen None to None | Any for PartialQuantified, matching
                        // the PartialContained behavior (see comment there).
                        let variables = self.solver.variables.lock();
                        if answer.is_none() {
                            let widened = self
                                .solver
                                .heap
                                .mk_union(vec![answer.clone(), Type::any_implicit()]);
                            variables.update(*v2, Variable::Answer(widened));
                        } else {
                            variables.update(*v2, Variable::Answer(answer));
                        }
                        Ok(())
                    }
                    Variable::PartialContained(_) => {
                        let t1_p = t1
                            .clone()
                            .promote_implicit_literals(self.type_order.stdlib());
                        drop(v2_ref);
                        // Widen None to None | Any (see comment at the other
                        // PartialContained pinning site above).
                        let answer = if t1_p.is_none() {
                            self.solver.heap.mk_union(vec![t1_p, Type::any_implicit()])
                        } else {
                            t1_p
                        };
                        variables.update(*v2, Variable::Answer(answer));
                        Ok(())
                    }
                    Variable::Unwrap(_) => {
                        drop(v2_ref);
                        drop(variables);
                        self.solver
                            .add_lower_bound(*v2, t1.clone(), &mut |got, want| {
                                self.is_subset_eq(got, want)
                            })
                    }
                    Variable::Recursive => {
                        drop(v2_ref);
                        variables.update(*v2, Variable::Answer(t1.clone()));
                        Ok(())
                    }
                }
            }
            _ => self.is_subset_eq_impl(got, want),
        }
    }

    pub fn finish_quantified(
        &self,
        vs: QuantifiedHandle,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        self.solver
            .finish_quantified(vs, self.solver.infer_with_first_use)
    }
}
