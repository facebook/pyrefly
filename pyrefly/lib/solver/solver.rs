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
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;
use std::sync::Arc;

use itertools::Either;
use itertools::Itertools;
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
use crate::types::callable::FuncFlags;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
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
use crate::types::typed_dict::TypedDict;
use crate::types::types::CallableResidual;
use crate::types::types::CallableResidualKind;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Overload;
use crate::types::types::OverloadBranchProjection;
use crate::types::types::OverloadResidualIdentity;
use crate::types::types::OverloadType;
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
const MAX_RESIDUAL_FINALIZE_ITERS: usize = 8;

/// Accumulated bounds for a solver variable.
#[derive(Clone, Debug, Default)]
struct Bounds {
    // TODO(https://github.com/facebook/pyrefly/issues/105): use `SmallSet<Type>`; bounds should
    // not be order-dependent.
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidualIdentity {
    witness_hash: u64,
    /// Vars that are allowed to observe the residualized answer for this candidate.
    /// This is captured during subset checks and threaded into `Variable::ResidualAnswer`.
    target_vars: SmallSet<Var>,
}

/// Full per-branch capture used transiently during overload probing.
/// Decomposed into per-var results by `record_overload_residuals_for_witness`.
#[derive(Clone, Debug)]
pub(crate) struct OverloadBranchCapture {
    branch_index: usize,
    values: SmallMap<Var, Variable>,
}

/// Shared context for an overload capture event. Arc'd to avoid duplication across vars.
#[derive(Clone, Debug)]
struct OverloadResidualWitness {
    identity: ResidualIdentity,
}

/// Per-var, per-branch capture at solver level.
#[derive(Clone, Debug)]
struct OverloadVarBranchCapture {
    branch_index: usize,
    value: Variable,
}

/// Per-var overload residual, stored on `Variable::Quantified`.
#[derive(Clone, Debug)]
struct OverloadResidualForVar {
    witness: Arc<OverloadResidualWitness>,
    branches: Vec<OverloadVarBranchCapture>,
}

/// Witness-keyed pruning decisions threaded through finishing.
#[derive(Clone, Debug, Default)]
struct OverloadWitnessPruningDecision {
    surviving_branch_indices: SmallSet<usize>,
    all_pruned: bool,
    all_pruned_cause: Option<OverloadAllPrunedCause>,
}

type OverloadPruningByWitness = HashMap<OverloadResidualIdentity, OverloadWitnessPruningDecision>;

enum OverloadResidualIdentityAnalysis {
    None,
    Single {
        identity: OverloadResidualIdentity,
        branch_indices: Vec<usize>,
    },
    Multiple,
}

#[derive(Clone, Debug)]
struct OverloadAllPrunedCause {
    quantified_name: Name,
    solved_ty: Type,
}

/// Snapshot of a solved quantified var after the bounds-first pass in
/// `finish_quantified`.
///
/// Used by witness-level overload pruning: eliminate branches whose
/// branch-implied bounds are incompatible with the solved type.
struct OverloadBranchSubstitutionResult {
    substituted: bool,
    marker_remaining: bool,
}
#[derive(Clone, Debug)]
struct SolvedVarWithResiduals {
    quantified_name: Name,
    solved_ty: Type,
    overload_residuals: Vec<OverloadResidualForVar>,
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
        /// Residual candidates captured during subset checks.
        /// Kept separate from concrete bounds to avoid accidental coupling.
        residuals: Vec<ResidualIdentity>,
        /// Overload residual candidates captured during overload dispatch.
        overload_residuals: Vec<OverloadResidualForVar>,
    },
    /// A variable caused by general recursion, e.g. `x = f(); def f(): return x`.
    Recursive,
    /// A variable that used to decompose a type, e.g. getting T from Awaitable[T]
    Unwrap(Bounds),
    /// A variable whose answer has been determined
    Answer(Type),
    /// A variable whose answer is a residual that is only visible to selected vars.
    ResidualAnswer {
        target_vars: SmallSet<Var>,
        ty: Type,
    },
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
                residuals: _,
                overload_residuals: _,
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
            Variable::ResidualAnswer { target_vars, ty } => {
                write!(f, "ResidualAnswer({ty}, targets={target_vars:?})")
            }
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
    Root(Box<Variable>, usize),
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
            VariableNode::Root(v, _) => v.as_ref(),
            _ => unreachable!(),
        })
    }

    fn get_mut<'a>(&'a self, x: Var) -> RefMut<'a, Variable> {
        let root = self.get_root(x);
        let variable = self.get_node(root).borrow_mut();
        RefMut::map(variable, |v| match v {
            VariableNode::Root(v, _) => v.as_mut(),
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
                .insert(x, RefCell::new(VariableNode::Root(Box::new(v), 0)))
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
    /// Cross-call cache for TypedDict subset results.
    /// Like protocol_cache, only caches Var-free types.
    typed_dict_cache: Mutex<HashMap<(TypedDict, TypedDict), Result<(), SubsetError>>>,
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

/// Result of `with_snapshot`, which performs an `is_subset_eq` call with var snapshotting.
pub enum SubsetWithSnapshotResult {
    /// `is_subset_eq` call was successful.
    Ok,
    /// `is_subset_eq` call failed.
    Err(SubsetError),
}

impl SubsetWithSnapshotResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, SubsetWithSnapshotResult::Ok)
    }
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
            typed_dict_cache: Default::default(),
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

    pub fn check_typed_dict_cache(
        &self,
        got: &TypedDict,
        want: &TypedDict,
    ) -> Option<Result<(), SubsetError>> {
        self.typed_dict_cache
            .lock()
            .get(&(got.clone(), want.clone()))
            .cloned()
    }

    pub fn store_typed_dict_cache(
        &self,
        got: TypedDict,
        want: TypedDict,
        result: Result<(), SubsetError>,
    ) {
        self.typed_dict_cache.lock().insert((got, want), result);
    }

    /// Force all non-recursive Vars in `vars`.
    /// TODO: deduplicate Variable-to-gradual-type logic with `force_var`.
    pub fn pin_placeholder_type(&self, var: Var, pin_partial_types: bool) -> Option<PinError> {
        let variables = self.variables.lock();
        let mut variable = variables.get_mut(var);
        match &mut *variable {
            Variable::Recursive | Variable::Answer(..) | Variable::ResidualAnswer { .. } => {
                // Nothing to do if we have an answer already, and we want to skip recursive Vars
                // which do not represent placeholder types.
                None
            }
            Variable::Quantified {
                quantified: q,
                bounds: _,
                ..
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

    /// Witnesses track both origin and deferred vars for residual plumbing,
    /// but overload branch capture snapshots only quantified vars. Only
    /// quantified vars can carry the per-branch residual candidates that we
    /// later materialize at finishing boundaries.
    pub(crate) fn var_is_quantified(&self, var: Var) -> bool {
        let variables = self.variables.lock();
        matches!(&*variables.get(var), Variable::Quantified { .. })
    }

    /// Witnesses track both origin and deferred vars for residual plumbing,
    /// but overload branch capture snapshots only quantified vars. Only
    /// quantified vars can carry the per-branch residual candidates that we
    /// later materialize at finishing boundaries.
    pub(crate) fn overload_capture_quantified_vars(
        &self,
        witness: &ResidualWitnessContext,
    ) -> Vec<Var> {
        let variables = self.variables.lock();
        witness
            .capture_candidate_vars()
            .into_iter()
            .filter(|var| matches!(&*variables.get(*var), Variable::Quantified { .. }))
            .collect()
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

    /// Snapshots the given vars, calls `f`, and rolls back the vars if the call fails.
    /// Note that this only rolls back the var state and not:
    /// * `Ok` entries left in `subset_cache` (the rollback in `is_subset_eq_impl` only fires on
    ///   `Err` from the speculative call, not on `Ok`-with-instantiation-errors), or
    /// * `coinductive_assumptions_used`, which is one-way.
    pub fn with_snapshot(
        &self,
        vars: &[Var],
        f: impl FnOnce() -> Result<(), SubsetError>,
    ) -> SubsetWithSnapshotResult {
        if vars.is_empty() {
            // Fast path - no var snapshotting needed.
            return f().map_or_else(SubsetWithSnapshotResult::Err, |_| {
                SubsetWithSnapshotResult::Ok
            });
        }
        let snapshot = self.snapshot_vars(vars);
        let res = match (f(), self.has_new_instantiation_errors(&snapshot)) {
            (Ok(()), false) => SubsetWithSnapshotResult::Ok,
            (Ok(()), true) => SubsetWithSnapshotResult::Err(SubsetError::Other),
            (Err(e), _) => SubsetWithSnapshotResult::Err(e),
        };
        if !res.is_ok() {
            self.restore_vars(snapshot);
        }
        res
    }

    // Partially sort a list of types for matching (is_subset_eq).
    // Sort non-var elements before var elements, so that if we match a non-var, we
    // don't pin the vars. Within var-containing members, try wrapped vars (e.g.
    // `type[T]`) before bare vars (e.g. `T`), so that more specific patterns are
    // tried first. This prevents cases like `T | type[T]` from incorrectly matching
    // bare `T` when `type[T]` would produce a better (bound-satisfying) solution.
    pub fn partial_sort_by_vars<'a>(
        &self,
        ts: &'a [Type],
    ) -> impl Iterator<Item = (&'a Type, Vec<Var>)> {
        let (vars, nonvars): (Vec<_>, Vec<_>) = ts.iter().partition_map(|t| {
            let vs = t.collect_maybe_placeholder_vars();
            if !vs.is_empty() {
                Either::Left((t, vs))
            } else {
                Either::Right((t, vs))
            }
        });
        let (bare_vars, wrapped_vars): (Vec<_>, Vec<_>) = vars
            .into_iter()
            .partition(|(t, _)| matches!(t, Type::Var(_)));
        nonvars.into_iter().chain(wrapped_vars).chain(bare_vars)
    }

    pub(crate) fn extract_overload_branch_capture(
        &self,
        branch_index: usize,
        vars: &[Var],
    ) -> OverloadBranchCapture {
        let variables = self.variables.lock();
        let values: SmallMap<Var, Variable> = vars
            .iter()
            .map(|var| (*var, variables.get(*var).clone()))
            .collect();
        OverloadBranchCapture {
            branch_index,
            values,
        }
    }

    /// Finish the type returned from a function call. This entails expanding solved variables,
    /// erasing unsolved variables without defaults from unions, and canonicalizing dimension
    /// expressions so that all-literal SizeExpr trees fold to single literals.
    pub fn finish_function_return(&self, mut t: Type) -> Type {
        self.expand_with_limit(&mut t, TYPE_LIMIT, &VarRecurser::new(), false, None);
        t = self.finalize_callable_residuals_preserving_class_targs_at_boundary(t);
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
        self.expand_with_limit(t, TYPE_LIMIT, &VarRecurser::new(), false, None);
        // After we substitute bound variables, we may be able to simplify some types
        self.simplify_mut(t);
    }

    // Canonical fallback policy for residual elimination outside callable-preserving paths.
    fn residual_fallback_type(&self, residual: &CallableResidual) -> Type {
        match &residual.kind {
            CallableResidualKind::Generic { quantified } => quantified.as_gradual_type(),
            CallableResidualKind::Overload { branches, .. } => branches
                .iter()
                .min_by_key(|branch| branch.branch_index)
                .expect("overload residual should include at least one branch")
                .ty
                .clone(),
        }
    }

    fn flatten_residual_for_non_target_read(&self, ty: &Type) -> Type {
        let mut flattened = ty.clone();
        flattened.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner {
                *inner = self.residual_fallback_type(residual);
            }
        });
        flattened
    }

    fn promote_callable_to_forall(&self, callable: Callable) -> Type {
        let callable_ty = self
            .heap
            .mk_callable(callable.params.clone(), callable.ret.clone());
        let mut tparams = Vec::new();
        callable_ty.for_each_quantified(&mut |q| tparams.push(q.clone()));
        tparams.sort();
        tparams.dedup();
        Forallable::Callable(callable).forall(Arc::new(TParams::new(tparams)))
    }

    fn promote_function_to_forall(&self, func: Function) -> Type {
        let callable_ty = self
            .heap
            .mk_callable(func.signature.params.clone(), func.signature.ret.clone());
        let mut tparams = Vec::new();
        callable_ty.for_each_quantified(&mut |q| tparams.push(q.clone()));
        tparams.sort();
        tparams.dedup();
        Forallable::Function(func).forall(Arc::new(TParams::new(tparams)))
    }

    fn generic_residual_quantified(&self, residual: &CallableResidual) -> Quantified {
        match &residual.kind {
            CallableResidualKind::Generic { quantified } => quantified.clone(),
            CallableResidualKind::Overload { .. } => {
                unreachable!("generic_residual_quantified called on overload residual")
            }
        }
    }

    /// Analyze overload residual markers in one pass.
    /// Returns:
    /// - `None`: no overload residual markers in this type
    /// - `Single`: exactly one witness identity with the branch-index intersection
    /// - `Multiple`: more than one witness identity appears in the same type
    fn analyze_overload_residual_identity(&self, ty: &Type) -> OverloadResidualIdentityAnalysis {
        let mut first: Option<OverloadResidualIdentity> = None;
        let mut intersection: Option<SmallSet<usize>> = None;
        let mut conflict = false;
        ty.universe(&mut |inner| {
            if conflict {
                return;
            }
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload { identity, branches } = &residual.kind
            {
                let branch_indices: SmallSet<usize> =
                    branches.iter().map(|branch| branch.branch_index).collect();
                match &first {
                    None => {
                        first = Some(identity.clone());
                        intersection = Some(branch_indices);
                    }
                    Some(existing) if existing == identity => {
                        let current = intersection.take().expect(
                            "matching overload residual identity must have intersection state",
                        );
                        intersection = Some(
                            current
                                .into_iter()
                                .filter(|idx| branch_indices.contains(idx))
                                .collect(),
                        );
                    }
                    Some(_) => conflict = true,
                }
            }
        });
        if conflict {
            OverloadResidualIdentityAnalysis::Multiple
        } else if let Some(identity) = first {
            let mut branch_indices = intersection
                .expect("matching overload residual identity must produce an intersection set")
                .into_iter()
                .collect::<Vec<_>>();
            branch_indices.sort_unstable();
            OverloadResidualIdentityAnalysis::Single {
                identity,
                branch_indices,
            }
        } else {
            OverloadResidualIdentityAnalysis::None
        }
    }

    fn strip_overload_residual_identity(&self, ty: &mut Type, identity: &OverloadResidualIdentity) {
        ty.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload {
                    identity: marker_identity,
                    ..
                } = &residual.kind
                && marker_identity == identity
            {
                *inner = self.residual_fallback_type(residual);
            }
        });
    }

    fn substitute_overload_residual_identity_branch(
        &self,
        ty: &mut Type,
        identity: &OverloadResidualIdentity,
        branch_index: usize,
    ) -> OverloadBranchSubstitutionResult {
        let mut substituted = false;
        let mut marker_remaining = false;
        ty.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload {
                    identity: marker_identity,
                    branches,
                } = &residual.kind
                && marker_identity == identity
            {
                let branch = branches
                    .iter()
                    .find(|branch| branch.branch_index == branch_index)
                    .expect("selected overload branch index must exist on every matching marker");
                let branch_ty = &branch.ty;
                let branch_contains_identity = branch_ty.any(|candidate| {
                    if let Type::CallableResidual(candidate_residual) = candidate
                        && let CallableResidualKind::Overload {
                            identity: candidate_identity,
                            ..
                        } = &candidate_residual.kind
                    {
                        candidate_identity == identity
                    } else {
                        false
                    }
                });
                marker_remaining |= branch_contains_identity;
                *inner = branch_ty.clone();
                substituted = true;
            }
        });
        OverloadBranchSubstitutionResult {
            substituted,
            marker_remaining,
        }
    }

    fn overload_metadata_for_reconstruction(&self, ty: &Type) -> FuncMetadata {
        ty.visit_toplevel_func_metadata(&|metadata| Some(metadata.clone()))
            .unwrap_or(FuncMetadata {
                kind: FunctionKind::Overload,
                flags: FuncFlags::default(),
            })
    }

    fn overload_signature_from_branch_type(
        &self,
        branch_ty: Type,
        metadata: &FuncMetadata,
    ) -> Option<OverloadType> {
        match branch_ty {
            Type::Function(function) => Some(OverloadType::Function(*function)),
            Type::Forall(forall) => match forall.body {
                Forallable::Function(function) => Some(OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: function,
                })),
                Forallable::Callable(callable) => Some(OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: Function {
                        signature: callable,
                        metadata: metadata.clone(),
                    },
                })),
                Forallable::TypeAlias(_) => None,
            },
            Type::Callable(callable) => Some(OverloadType::Function(Function {
                signature: *callable,
                metadata: metadata.clone(),
            })),
            _ => None,
        }
    }

    fn try_combine_reconstructed_overload(
        &self,
        original_ty: &Type,
        reconstructed: &[Type],
    ) -> Option<Type> {
        let metadata = self.overload_metadata_for_reconstruction(original_ty);
        let signatures = reconstructed
            .iter()
            .cloned()
            .map(|branch_ty| self.overload_signature_from_branch_type(branch_ty, &metadata))
            .collect::<Option<Vec<_>>>()?;
        let signatures = Vec1::try_from_vec(signatures).ok()?;
        Some(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    /// Finalize callable residuals after a substitution (either a return type or a class field)
    /// once (as in, perform one round of finalization - if a residual resolves to another
    /// residual, the second one will still be present).
    ///
    /// - Recursively replace `CallableResidual`s that are not nested in a Callable with flattened
    ///   fallback types (for example generic residuals become the gradual fallback for that quantified)
    /// - Transform callables that contain `CallableResidual`s (for example, if there are generic
    ///   residuals, then replace them with the Quantified, and also promote to a Forall with
    ///   those quantifieds scoped as type parameters).
    ///
    /// Returns
    /// - did_we_encounter_any_residuals: used to terminate recursive
    ///   transforms, which we need because residuals can nest (in particular,
    ///   an overload residual can contain a generic residual)
    /// - did_we_encounter_any_residuals_inside_a_callable: used to
    ///
    ///
    /// TODO(stroxler): See if this can be simplified or otherwise made clearer at the top of the
    /// stack, including after overload support gets added and after optimizations.
    fn finalize_callable_residuals_mut(
        &self,
        ty: &mut Type,
        callable_slot: bool,
        preserve_class_targs: bool,
    ) -> (bool, bool) {
        match ty {
            Type::CallableResidual(residual) => match &residual.kind {
                CallableResidualKind::Generic { .. } => {
                    if !callable_slot {
                        *ty = self.residual_fallback_type(residual);
                        return (true, false);
                    }
                    *ty = self
                        .heap
                        .mk_quantified(self.generic_residual_quantified(residual));
                    (true, true)
                }
                CallableResidualKind::Overload { .. } => {
                    *ty = self.residual_fallback_type(residual);
                    (true, false)
                }
            },
            Type::Callable(callable) => {
                let mut changed = false;
                let mut consumed_residual = false;
                match &mut callable.params {
                    Params::List(params) => {
                        for param in params.items_mut() {
                            let (param_changed, param_consumed) = self
                                .finalize_callable_residuals_mut(
                                    param.as_type_mut(),
                                    true,
                                    preserve_class_targs,
                                );
                            changed |= param_changed;
                            consumed_residual |= param_consumed;
                        }
                    }
                    Params::ParamSpec(prefix, p) => {
                        for prefix_param in prefix.iter_mut() {
                            let prefix_ty = match prefix_param {
                                PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => ty,
                            };
                            let (prefix_changed, prefix_consumed) = self
                                .finalize_callable_residuals_mut(
                                    prefix_ty,
                                    true,
                                    preserve_class_targs,
                                );
                            changed |= prefix_changed;
                            consumed_residual |= prefix_consumed;
                        }
                        let (paramspec_changed, paramspec_consumed) =
                            self.finalize_callable_residuals_mut(p, true, preserve_class_targs);
                        changed |= paramspec_changed;
                        consumed_residual |= paramspec_consumed;
                    }
                    Params::Ellipsis | Params::Materialization => {}
                }
                let (ret_changed, ret_consumed) = self.finalize_callable_residuals_mut(
                    &mut callable.ret,
                    true,
                    preserve_class_targs,
                );
                changed |= ret_changed;
                consumed_residual |= ret_consumed;
                if consumed_residual {
                    if callable_slot {
                        (changed, true)
                    } else {
                        let promoted = self.promote_callable_to_forall((**callable).clone());
                        *ty = promoted;
                        (true, true)
                    }
                } else {
                    (changed, false)
                }
            }
            Type::Function(function) => {
                if !function.signature.contains_callable_residual() {
                    return (false, false);
                }
                let mut signature = function.signature.clone();
                let (changed, consumed_residual) = self.finalize_callable_residuals_mut(
                    &mut signature.ret,
                    true,
                    preserve_class_targs,
                );
                let (params_changed, params_consumed) = match &mut signature.params {
                    Params::List(params) => {
                        let mut any_changed = false;
                        let mut any_consumed = false;
                        for param in params.items_mut() {
                            let (param_changed, param_consumed) = self
                                .finalize_callable_residuals_mut(
                                    param.as_type_mut(),
                                    true,
                                    preserve_class_targs,
                                );
                            any_changed |= param_changed;
                            any_consumed |= param_consumed;
                        }
                        (any_changed, any_consumed)
                    }
                    Params::ParamSpec(prefix, p) => {
                        let mut any_changed = false;
                        let mut any_consumed = false;
                        for prefix_param in prefix.iter_mut() {
                            let prefix_ty = match prefix_param {
                                PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => ty,
                            };
                            let (prefix_changed, prefix_consumed) = self
                                .finalize_callable_residuals_mut(
                                    prefix_ty,
                                    true,
                                    preserve_class_targs,
                                );
                            any_changed |= prefix_changed;
                            any_consumed |= prefix_consumed;
                        }
                        let (paramspec_changed, paramspec_consumed) =
                            self.finalize_callable_residuals_mut(p, true, preserve_class_targs);
                        any_changed |= paramspec_changed;
                        any_consumed |= paramspec_consumed;
                        (any_changed, any_consumed)
                    }
                    Params::Ellipsis | Params::Materialization => (false, false),
                };
                let changed = changed | params_changed;
                let consumed_residual = consumed_residual | params_consumed;
                if !changed {
                    return (false, false);
                }
                function.signature = signature;
                if consumed_residual && !callable_slot {
                    let promoted = self.promote_function_to_forall((**function).clone());
                    *ty = promoted;
                    (true, true)
                } else {
                    (true, consumed_residual)
                }
            }
            Type::ClassType(_) if preserve_class_targs && !callable_slot => (false, false),
            _ => {
                let mut changed = false;
                let mut consumed_residual = false;
                ty.recurse_mut(&mut |inner| {
                    let (inner_changed, inner_consumed) = self.finalize_callable_residuals_mut(
                        inner,
                        callable_slot,
                        preserve_class_targs,
                    );
                    changed |= inner_changed;
                    consumed_residual |= inner_consumed;
                });
                (changed, consumed_residual)
            }
        }
    }

    /// Finalize callable residuals at a substitution boundary (a return type or class field).
    ///
    /// Because residuals can resolve to residuals, we do this until all residuals are removed.
    ///
    /// TODO(stroxler): We might be able to simplify this if we can guarantee that residuals
    /// nest at most one deep; I *think* we only really need to handle generic residuals inside
    /// an overload residual, which means two passes should suffice, but it is not trivial
    /// to be certain about this.
    fn finalize_callable_residuals_at_boundary_impl(
        &self,
        ty: Type,
        preserve_class_targs: bool,
    ) -> Type {
        let mut active_overload_identities = Vec::new();
        self.finalize_callable_residuals_at_boundary_impl_with_active(
            ty,
            preserve_class_targs,
            &mut active_overload_identities,
        )
    }

    fn finalize_callable_residuals_at_boundary_impl_with_active(
        &self,
        mut ty: Type,
        preserve_class_targs: bool,
        active_overload_identities: &mut Vec<OverloadResidualIdentity>,
    ) -> Type {
        // Preserve class targs at this boundary so residualized class arguments
        // can be finalized later at the access/call site with proper context.
        let skip_overload_reconstruction = preserve_class_targs && matches!(ty, Type::ClassType(_));

        // Reconstruction is only safe when all overload residual markers in
        // the type share a single witness identity. Multiple witnesses would
        // produce a cross-product explosion; strip all markers and fall
        // through to generic residual finalization instead.
        if !skip_overload_reconstruction
            && let OverloadResidualIdentityAnalysis::Single {
                identity,
                branch_indices,
            } = self.analyze_overload_residual_identity(&ty)
        {
            if active_overload_identities.contains(&identity) {
                unreachable!(
                    "detected recursive overload residual identity cycle during finalization",
                );
            }
            if branch_indices.is_empty() {
                // TODO(T235420905): This path is intended to be unreachable for coherent
                // witnesses, but we have seen CI panic signatures around this stack. Audit
                // the top-of-stack identity/branch-coherence invariant and decide whether
                // to harden this boundary or tighten upstream residual capture.
                self.strip_overload_residual_identity(&mut ty, &identity);
            } else {
                active_overload_identities.push(identity.clone());
                let mut reconstructed = Vec::with_capacity(branch_indices.len());
                for branch_index in branch_indices {
                    let mut branch_ty = ty.clone();
                    let substitution_result = self.substitute_overload_residual_identity_branch(
                        &mut branch_ty,
                        &identity,
                        branch_index,
                    );
                    if !substitution_result.substituted {
                        unreachable!(
                            "selected overload residual identity must be present during reconstruction",
                        );
                    }
                    if substitution_result.marker_remaining {
                        unreachable!(
                            "overload residual substitution did not eliminate active identity"
                        );
                    }
                    reconstructed.push(
                        self.finalize_callable_residuals_at_boundary_impl_with_active(
                            branch_ty,
                            preserve_class_targs,
                            active_overload_identities,
                        ),
                    );
                }
                let popped = active_overload_identities
                    .pop()
                    .expect("active_overload_identities push/pop must stay balanced");
                debug_assert_eq!(popped, identity);
                if let Some(overload_ty) =
                    self.try_combine_reconstructed_overload(&ty, &reconstructed)
                {
                    ty = overload_ty;
                } else {
                    // TODO(stroxler): This behavior causes us to explode residuals into unions
                    // whenever they appear in a non-Callable type. The behavior isn't exactly
                    // bad, but it would be cleaner to flatten them inline. Fix this later.
                    ty = unions(reconstructed, &self.heap);
                    self.simplify_mut(&mut ty);
                }
            }
        }

        for _ in 0..MAX_RESIDUAL_FINALIZE_ITERS {
            if !self
                .finalize_callable_residuals_mut(&mut ty, false, preserve_class_targs)
                .0
            {
                return ty;
            }
        }
        self.flatten_residual_for_non_target_read(&ty)
    }

    pub(crate) fn finalize_callable_residuals_at_boundary(&self, ty: Type) -> Type {
        self.finalize_callable_residuals_at_boundary_impl(ty, false)
    }

    fn finalize_callable_residuals_preserving_class_targs_at_boundary(&self, ty: Type) -> Type {
        self.finalize_callable_residuals_at_boundary_impl(ty, true)
    }

    fn residual_read_for_query_var(
        &self,
        query_var: Option<Var>,
        target_vars: &SmallSet<Var>,
        ty: &Type,
    ) -> Type {
        if query_var.is_some_and(|q| target_vars.contains(&q)) {
            ty.clone()
        } else {
            self.flatten_residual_for_non_target_read(ty)
        }
    }

    /// Expand, but if the resulting type will be greater than limit levels deep, return an `Any`.
    /// Avoids producing things that stack overflow later in the process.
    fn expand_with_limit(
        &self,
        t: &mut Type,
        limit: usize,
        recurser: &VarRecurser,
        expand_unfinished_variables: bool,
        query_var: Option<Var>,
    ) {
        if limit == 0 {
            // TODO: Should probably add an error here, and use any_error,
            // but don't have any good location information to hand.
            *t = self.heap.mk_any_implicit();
        } else if let Type::Var(x) = t {
            let query_var = query_var.or(Some(*x));
            let lock = self.variables.lock();
            if let Some(_guard) = lock.recurse(*x, recurser) {
                let variable = lock.get(*x);
                match &*variable {
                    Variable::Answer(ty) => {
                        *t = ty.clone();
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(
                            t,
                            limit - 1,
                            recurser,
                            expand_unfinished_variables,
                            query_var,
                        );
                    }
                    Variable::ResidualAnswer { target_vars, ty } => {
                        *t = self.residual_read_for_query_var(query_var, target_vars, ty);
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(
                            t,
                            limit - 1,
                            recurser,
                            expand_unfinished_variables,
                            query_var,
                        );
                    }
                    Variable::Quantified {
                        quantified: _,
                        bounds,
                        ..
                    }
                    | Variable::Unwrap(bounds)
                        if expand_unfinished_variables
                            && let Some(bound) = self.solve_bounds(bounds.clone()) =>
                    {
                        *t = bound;
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(
                            t,
                            limit - 1,
                            recurser,
                            expand_unfinished_variables,
                            query_var,
                        );
                    }
                    _ => {}
                }
            } else {
                *t = self.heap.mk_any_implicit();
            }
        } else {
            t.recurse_mut(&mut |t| {
                self.expand_with_limit(
                    t,
                    limit - 1,
                    recurser,
                    expand_unfinished_variables,
                    query_var,
                )
            });
        }
    }

    /// Expand `Variable::Unwrap` to its answer or its lower bounds accumulated so far.
    pub fn expand_unwrap(&self, v: Var) -> Type {
        let variables = self.variables.lock();
        match &*variables.get(v) {
            Variable::Answer(t) => t.clone(),
            Variable::ResidualAnswer { target_vars, ty } => {
                self.residual_read_for_query_var(Some(v), target_vars, ty)
            }
            Variable::Unwrap(bounds) if let Some(bound) = self.solve_bounds(bounds.clone()) => {
                bound
            }
            _ => v.to_type(&self.heap),
        }
    }

    /// Public wrapper to expand a dimension type by resolving bound Vars.
    /// Used by subset checking to expand Vars before comparing dimension expressions.
    pub fn expand_dimension(&self, dim_ty: &mut Type) {
        self.expand_with_limit(dim_ty, TYPE_LIMIT, &VarRecurser::new(), true, None);
    }

    /// Given a `Var`, ensures that the solver has an answer for it (or inserts Any if not already),
    /// and returns that answer. Note that if the `Var` is already bound to something that contains a
    /// `Var` (including itself), then we will return the answer.
    pub fn force_var(&self, v: Var) -> Type {
        let lock = self.variables.lock();
        let mut e = lock.get_mut(v);
        let result = match &mut *e {
            Variable::Answer(t) => t.clone(),
            Variable::ResidualAnswer { target_vars, ty } => {
                self.residual_read_for_query_var(Some(v), target_vars, ty)
            }
            _ => {
                let ty = match &mut *e {
                    Variable::Quantified {
                        quantified: q,
                        bounds,
                        ..
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

    /// Normalize a type for export-like boundaries that must not leak solver-internal
    /// placeholders such as callable residuals.
    ///
    /// This expands already-solved vars while leaving unfinished vars in place; boundary
    /// reads must be non-forcing.
    ///
    /// This is the canonical boundary normalization entry point used by report/query
    /// surfaces and other serialization/display-adjacent consumers.
    pub fn for_export_boundary(&self, mut t: Type) -> Type {
        self.expand_vars_mut(&mut t);
        t = self.finalize_callable_residuals_at_boundary(t);
        self.simplify_mut(&mut t);
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
                    residuals: Vec::new(),
                    overload_residuals: Vec::new(),
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
                ..
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
                ..
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
                ..
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
                ..
            }
            | Variable::Unwrap(bounds) => self.add_bound(&mut bounds.upper, new_bound),
            _ => {}
        }
        res
    }

    /// Get current bound from a set of bounds of an unfinished variable.
    /// TODO(https://github.com/facebook/pyrefly/issues/105): the current solver design requires us
    /// to repeatedly clone and union together intermediate bounds to validate every new bound we
    /// add. Consider a less wasteful strategy, such as validating when we finish the variable.
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
        // Callable residual bounds are fallback-only. If we also learned a concrete
        // non-Any bound, prefer that and discard residual markers.
        if bounds
            .iter()
            .any(|t| !t.is_any() && !matches!(t, Type::CallableResidual(_)))
        {
            bounds.retain(|t| !t.is_any() && !matches!(t, Type::CallableResidual(_)));
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
        // TODO(https://github.com/facebook/pyrefly/issues/105): consider using polarity to
        // determine whether we use the lower or upper bound.
        let lower_bound = self.solve_one_bounds(bounds.lower);
        if lower_bound.as_ref().is_none_or(|b| b.is_any()) {
            self.solve_one_bounds(bounds.upper).or(lower_bound)
        } else {
            lower_bound
        }
    }

    pub(crate) fn record_generic_residuals_for_witness(&self, witness: &ResidualWitnessContext) {
        let lock = self.variables.lock();
        for &var in witness
            .origin_vars
            .iter()
            .chain(witness.deferred_vars.iter())
        {
            let mut variable = lock.get_mut(var);
            if let Variable::Quantified { residuals, .. } = &mut *variable
                && !residuals.contains(&witness.identity)
            {
                residuals.push(witness.identity.clone());
            }
        }
    }

    /// Record per-var overload residuals from full branch captures.
    /// Decomposes each branch's full capture into per-var entries.
    pub(crate) fn record_overload_residuals_for_witness(
        &self,
        witness: &ResidualWitnessContext,
        branch_captures: Vec<OverloadBranchCapture>,
    ) {
        let shared_witness = Arc::new(OverloadResidualWitness {
            identity: witness.identity.clone(),
        });
        let lock = self.variables.lock();
        for &var in witness
            .origin_vars
            .iter()
            .chain(witness.deferred_vars.iter())
        {
            let mut variable = lock.get_mut(var);
            if let Variable::Quantified {
                overload_residuals, ..
            } = &mut *variable
            {
                let branches: Vec<OverloadVarBranchCapture> = branch_captures
                    .iter()
                    .filter_map(|capture| {
                        capture
                            .values
                            .get(&var)
                            .map(|value| OverloadVarBranchCapture {
                                branch_index: capture.branch_index,
                                value: value.clone(),
                            })
                    })
                    .collect();
                if !branches.is_empty() {
                    overload_residuals.push(OverloadResidualForVar {
                        witness: shared_witness.clone(),
                        branches,
                    });
                }
            }
        }
    }

    fn merge_residual_candidates(
        left: &mut Vec<ResidualIdentity>,
        right: &mut Vec<ResidualIdentity>,
    ) {
        for residual in mem::take(right) {
            if !left.contains(&residual) {
                left.push(residual);
            }
        }
        *right = left.clone();
    }

    fn materialize_generic_residual_type(&self, q: &Quantified) -> Type {
        Type::CallableResidual(Box::new(CallableResidual {
            kind: CallableResidualKind::Generic {
                quantified: q.clone(),
            },
        }))
    }

    fn materialize_overload_residual_type(
        &self,
        residual: &OverloadResidualForVar,
        overload_pruning_by_witness: &OverloadPruningByWitness,
    ) -> Type {
        let identity = OverloadResidualIdentity {
            witness_hash: residual.witness.identity.witness_hash,
        };
        let pruning_decision = overload_pruning_by_witness.get(&identity);
        if pruning_decision.is_some_and(|decision| decision.all_pruned) {
            // All candidate branches were pruned for this witness.
            // Return Never immediately and avoid any branch materialization work.
            return Type::never();
        }
        let surviving_branch_indices = pruning_decision
            .map(|decision| decision.surviving_branch_indices.clone())
            .unwrap_or_else(|| {
                residual
                    .branches
                    .iter()
                    .map(|branch| branch.branch_index)
                    .collect()
            });
        let surviving_branches = residual
            .branches
            .iter()
            .filter(|branch| surviving_branch_indices.contains(&branch.branch_index))
            .map(|branch| OverloadBranchProjection {
                branch_index: branch.branch_index,
                ty: self.materialize_overload_residual_branch_value(
                    &branch.value,
                    overload_pruning_by_witness,
                ),
            })
            .collect::<Vec<_>>();
        match surviving_branches.len() {
            0 => {
                unreachable!(
                    "overload residual pruning produced no surviving branches without all_pruned"
                )
            }
            1 => {
                surviving_branches
                    .into_iter()
                    .next()
                    .expect("single surviving overload branch must exist")
                    .ty
            }
            _ => {
                let first_ty = surviving_branches
                    .first()
                    .expect("multiple surviving overload branches must have first branch")
                    .ty
                    .clone();
                if surviving_branches
                    .iter()
                    .all(|branch| branch.ty == first_ty)
                {
                    first_ty
                } else {
                    Type::CallableResidual(Box::new(CallableResidual {
                        kind: CallableResidualKind::Overload {
                            identity,
                            branches: surviving_branches,
                        },
                    }))
                }
            }
        }
    }

    fn materialize_overload_residual_branch_value(
        &self,
        value: &Variable,
        overload_pruning_by_witness: &OverloadPruningByWitness,
    ) -> Type {
        match value {
            Variable::Answer(ty) | Variable::ResidualAnswer { ty, .. } => ty.clone(),
            Variable::Quantified {
                quantified,
                bounds,
                residuals,
                overload_residuals,
            } => {
                if let Some(bound) = self.solve_bounds(bounds.clone()) {
                    return bound;
                }
                if overload_residuals.len() == 1 {
                    return self.materialize_overload_residual_type(
                        overload_residuals.first().expect(
                            "overload_residuals.len() == 1 must provide one overload residual",
                        ),
                        overload_pruning_by_witness,
                    );
                }
                if residuals.len() == 1 {
                    return self.materialize_generic_residual_type(quantified);
                }
                quantified.as_gradual_type()
            }
            Variable::PartialQuantified(q) => q.as_gradual_type(),
            Variable::PartialContained(_) => {
                unreachable!("overload residual capture should not include PartialContained vars")
            }
            Variable::Recursive => {
                unreachable!("overload residual capture should not include Recursive vars")
            }
            Variable::Unwrap(_) => {
                unreachable!("overload residual capture should not include Unwrap vars")
            }
        }
    }

    fn are_branch_bounds_compatible_with_solved_type(
        &self,
        branch_value: &Variable,
        solved_ty: &Type,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> bool {
        let bounds = match branch_value {
            Variable::Quantified { bounds, .. } | Variable::Unwrap(bounds) => bounds,
            Variable::Answer(branch_ty) | Variable::ResidualAnswer { ty: branch_ty, .. } => {
                // If this branch already collapsed to a concrete type, treat
                // compatibility as type equivalence against the solved type.
                return is_subset(branch_ty, solved_ty).is_ok()
                    && is_subset(solved_ty, branch_ty).is_ok();
            }
            Variable::PartialQuantified(_) => {
                unreachable!("overload residual capture should not include PartialQuantified vars")
            }
            Variable::PartialContained(_) => {
                unreachable!("overload residual capture should not include PartialContained vars")
            }
            Variable::Recursive => {
                unreachable!("overload residual capture should not include Recursive vars")
            }
        };
        bounds
            .lower
            .iter()
            .all(|lower| is_subset(lower, solved_ty).is_ok())
            && bounds
                .upper
                .iter()
                .all(|upper| is_subset(solved_ty, upper).is_ok())
    }

    fn compute_overload_pruning_by_witness(
        &self,
        solved_vars_with_residuals: &[SolvedVarWithResiduals],
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> OverloadPruningByWitness {
        let mut all_branch_indices_by_witness: HashMap<OverloadResidualIdentity, SmallSet<usize>> =
            HashMap::new();
        let mut surviving_branch_indices_by_witness: HashMap<
            OverloadResidualIdentity,
            SmallSet<usize>,
        > = HashMap::new();
        let mut all_pruned_cause_by_witness: HashMap<
            OverloadResidualIdentity,
            OverloadAllPrunedCause,
        > = HashMap::new();

        for solved_var in solved_vars_with_residuals {
            let mut surviving_per_witness_for_solved_var: HashMap<
                OverloadResidualIdentity,
                SmallSet<usize>,
            > = HashMap::new();
            for residual in &solved_var.overload_residuals {
                let identity = OverloadResidualIdentity {
                    witness_hash: residual.witness.identity.witness_hash,
                };
                let all_branch_indices = all_branch_indices_by_witness
                    .entry(identity.clone())
                    .or_default();
                let surviving_for_solved_var = surviving_per_witness_for_solved_var
                    .entry(identity.clone())
                    .or_default();
                for branch in &residual.branches {
                    all_branch_indices.insert(branch.branch_index);
                    if self.are_branch_bounds_compatible_with_solved_type(
                        &branch.value,
                        &solved_var.solved_ty,
                        is_subset,
                    ) {
                        surviving_for_solved_var.insert(branch.branch_index);
                    }
                }
                if surviving_for_solved_var.is_empty() {
                    all_pruned_cause_by_witness
                        .entry(identity)
                        .or_insert_with(|| OverloadAllPrunedCause {
                            quantified_name: solved_var.quantified_name.clone(),
                            solved_ty: solved_var.solved_ty.clone(),
                        });
                }
            }
            for (identity, surviving_for_solved_var) in surviving_per_witness_for_solved_var {
                if let Some(existing_surviving) =
                    surviving_branch_indices_by_witness.get_mut(&identity)
                {
                    existing_surviving.retain(|idx| surviving_for_solved_var.contains(idx));
                } else {
                    surviving_branch_indices_by_witness.insert(identity, surviving_for_solved_var);
                }
            }
        }

        all_branch_indices_by_witness
            .into_keys()
            .map(|identity| {
                let surviving_branch_indices = surviving_branch_indices_by_witness
                    .get(&identity)
                    .cloned()
                    .unwrap_or_else(|| {
                        unreachable!(
                            "all observed overload residual witnesses must have surviving candidates"
                        )
                    });
                let all_pruned = surviving_branch_indices.is_empty();
                let all_pruned_cause = if all_pruned {
                    Some(
                        all_pruned_cause_by_witness
                            .get(&identity)
                            .cloned()
                            .unwrap_or_else(|| {
                                unreachable!(
                                    "all-pruned witness decisions must include a solved-type cause"
                                )
                            }),
                    )
                } else {
                    None
                };
                (
                    identity,
                    OverloadWitnessPruningDecision {
                        surviving_branch_indices: surviving_branch_indices.clone(),
                        all_pruned,
                        all_pruned_cause,
                    },
                )
            })
            .collect()
    }

    /// Called after a quantified function has been called. Given
    /// `def f[T](x: int): list[T]`, this runs after generic solving
    /// completes.
    ///
    /// If `infer_with_first_use` is true, unresolved `T` behaves like an
    /// empty-container partial type and may be pinned by first use.
    /// If `infer_with_first_use` is false, unresolved `T` is replaced with
    /// gradual (`Any`-like) fallback.
    pub fn finish_quantified(
        &self,
        vs: QuantifiedHandle,
        infer_with_first_use: bool,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        self.finish_quantified_with_subset(vs, infer_with_first_use, &mut |_got, _want| Ok(()))
    }

    pub(crate) fn finish_quantified_with_subset(
        &self,
        vs: QuantifiedHandle,
        infer_with_first_use: bool,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        let mut err = Vec::new();
        let mut solved_vars_with_residuals = Vec::new();
        let mut reported_all_pruned_witnesses = SmallSet::new();
        let lock = self.variables.lock();
        for &v in &vs.0 {
            let mut variable = lock.get_mut(v);
            match &mut *variable {
                Variable::Answer(_) | Variable::ResidualAnswer { .. } => {
                    // We pin the quantified var to a type when it first appears in a subset constraint,
                    // and at that point we check the instantiation with the bound.
                    if let Some(e) = self.instantiation_errors.read().get(&v) {
                        err.push(e.clone());
                    }
                }
                Variable::Quantified {
                    quantified: q,
                    bounds,
                    overload_residuals,
                    ..
                } => {
                    if let Some(e) = self.instantiation_errors.read().get(&v) {
                        err.push(e.clone());
                    }
                    let original_bounds = mem::take(bounds);
                    if let Some(bound) = self.solve_bounds(original_bounds.clone()) {
                        let quantified_name = q.name().clone();
                        let overload_residuals = overload_residuals.clone();
                        let solved_ty = bound.clone();
                        *variable = Variable::Answer(bound);
                        solved_vars_with_residuals.push(SolvedVarWithResiduals {
                            quantified_name,
                            solved_ty,
                            overload_residuals,
                        });
                    } else {
                        *bounds = original_bounds;
                    }
                }
                _ => {}
            }
        }
        drop(lock);

        let overload_pruning_by_witness =
            self.compute_overload_pruning_by_witness(&solved_vars_with_residuals, is_subset);

        let lock = self.variables.lock();
        for &v in &vs.0 {
            let mut e = lock.get_mut(v);
            if let Variable::Quantified {
                quantified: q,
                bounds,
                residuals,
                overload_residuals,
            } = &mut *e
            {
                let solved_bound = self.solve_bounds(mem::take(bounds));
                let overload_residual_candidate =
                    if solved_bound.is_none() && overload_residuals.len() == 1 {
                        overload_residuals.first().cloned()
                    } else {
                        None
                    };
                let overload_all_pruned =
                    overload_residual_candidate
                        .as_ref()
                        .is_some_and(|candidate| {
                            let identity = OverloadResidualIdentity {
                                witness_hash: candidate.witness.identity.witness_hash,
                            };
                            overload_pruning_by_witness
                                .get(&identity)
                                .is_some_and(|decision| decision.all_pruned)
                        });
                if overload_all_pruned && let Some(candidate) = overload_residual_candidate.as_ref()
                {
                    let witness_hash = candidate.witness.identity.witness_hash;
                    if reported_all_pruned_witnesses.insert(witness_hash) {
                        let all_pruned_cause = overload_pruning_by_witness
                            .get(&OverloadResidualIdentity { witness_hash })
                            .and_then(|decision| decision.all_pruned_cause.as_ref())
                            .unwrap_or_else(|| {
                                unreachable!(
                                    "all-pruned witness diagnostics require solved-type cause"
                                )
                            });
                        err.push(TypeVarSpecializationError {
                            name: q.name().clone(),
                            got: Type::never(),
                            want: q.as_gradual_type(),
                            message_override: Some(format!(
                                "Overload type was not compatible with the solved type `{}` of type variable `{}`",
                                all_pruned_cause.solved_ty.clone().deterministic_printing(),
                                all_pruned_cause.quantified_name,
                            )),
                            error: SubsetError::Other,
                        });
                    }
                }
                let residual_candidate = if solved_bound.is_none() && residuals.len() == 1 {
                    residuals.first().cloned()
                } else {
                    None
                };
                *e = if let Some(bound) = solved_bound {
                    Variable::Answer(bound)
                } else if overload_all_pruned {
                    Variable::Answer(Type::never())
                } else if let Some(overload_residual) = overload_residual_candidate {
                    Variable::ResidualAnswer {
                        target_vars: overload_residual.witness.identity.target_vars.clone(),
                        ty: self.materialize_overload_residual_type(
                            &overload_residual,
                            &overload_pruning_by_witness,
                        ),
                    }
                } else if let Some(ResidualIdentity { target_vars, .. }) = residual_candidate {
                    Variable::ResidualAnswer {
                        target_vars,
                        ty: self.materialize_generic_residual_type(q),
                    }
                } else if infer_with_first_use {
                    Variable::finished(q)
                } else {
                    Variable::Answer(q.as_gradual_type())
                };
            }
        }
        drop(lock);

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
                        residuals: Vec::new(),
                        overload_residuals: Vec::new(),
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
                    residuals,
                    overload_residuals,
                    ..
                } = &mut *e
                    && *q == *param
                {
                    if bounds.is_empty() && residuals.is_empty() && overload_residuals.is_empty() {
                        *t = param.clone().to_type(&self.heap);
                    } else if !bounds.is_empty() {
                        // If the variable has bounds, finalize its type now.
                        *e = Variable::Answer(
                            self.solve_bounds(mem::take(bounds))
                                .unwrap_or_else(|| q.as_gradual_type()),
                        );
                    }
                    // Otherwise (residuals but no bounds): leave the var as
                    // Quantified so finish_quantified can materialize residuals.
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

    pub fn for_display(&self, t: Type) -> Type {
        let mut t = t;
        self.expand_with_limit(&mut t, TYPE_LIMIT, &VarRecurser::new(), true, None);
        t = self.finalize_callable_residuals_at_boundary(t);
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
        note: Option<String>,
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
        if let Some(note) = note {
            msg_lines.push(note);
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
            query_var: Var,
            residual_read: &dyn Fn(Var, &SmallSet<Var>, &Type) -> Type,
            res: &mut Vec<Type>,
        ) {
            match t {
                Type::Var(v) if let Some(_guard) = variables.recurse(v, recurser) => {
                    let variable = variables.get(v);
                    match &*variable {
                        Variable::Answer(t) => {
                            let t = t.clone();
                            drop(variable);
                            expand(t, variables, recurser, heap, query_var, residual_read, res);
                        }
                        Variable::ResidualAnswer { target_vars, ty } => {
                            let t = residual_read(query_var, target_vars, ty);
                            drop(variable);
                            expand(t, variables, recurser, heap, query_var, residual_read, res);
                        }
                        _ => res.push(v.to_type(heap)),
                    }
                }
                Type::Union(box Union { members: ts, .. }) => {
                    for t in ts {
                        expand(t, variables, recurser, heap, query_var, residual_read, res);
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
            Variable::ResidualAnswer {
                target_vars,
                ty: forced,
            } => {
                let forced = self.residual_read_for_query_var(Some(var), target_vars, forced);
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
                let residual_read = |query_var: Var, target_vars: &SmallSet<Var>, ty: &Type| {
                    self.residual_read_for_query_var(Some(query_var), target_vars, ty)
                };
                expand(
                    ty,
                    &lock,
                    &VarRecurser::new(),
                    &self.heap,
                    var,
                    &residual_read,
                    &mut res,
                );
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

    pub(crate) fn is_subset_eq_with_call_context<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
        call_context: &CallContext,
    ) -> Result<(), SubsetError> {
        let mut subset = self.subset(type_order);
        subset.is_subset_eq_with_context(got, want, call_context)
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

    pub fn finish_quantified_with_type_order<Ans: LookupAnswer>(
        &self,
        vs: QuantifiedHandle,
        infer_with_first_use: bool,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        let mut subset = self.subset(type_order);
        subset.finish_quantified_with_infer(vs, infer_with_first_use)
    }

    fn subset<'a, Ans: LookupAnswer>(&'a self, type_order: TypeOrder<'a, Ans>) -> Subset<'a, Ans> {
        Subset {
            solver: self,
            type_order,
            gas: INITIAL_GAS,
            active_call_context: CallContext::outside(),
            subset_cache: SmallMap::new(),
            class_protocol_assumptions: SmallSet::new(),
            coinductive_assumptions_used: false,
            witness_deferred_vars: SmallMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeVarSpecializationError {
    pub name: Name,
    pub got: Type,
    pub want: Type,
    pub message_override: Option<String>,
    #[allow(dead_code)]
    pub error: SubsetError,
}

impl TypeVarSpecializationError {
    pub fn to_error_msg<Ans: LookupAnswer>(self, ans: &AnswersSolver<Ans>) -> String {
        if let Some(message_override) = self.message_override {
            return message_override;
        }
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
pub enum SubsetCacheEntry {
    /// Currently being computed — used for coinductive cycle detection.
    /// Treated as `Ok(())`: if we encounter a pair already being checked,
    /// we optimistically assume the check succeeds (coinductive reasoning).
    InProgress,
    /// Computed and succeeded.
    Ok,
    /// Computed and failed.
    Err(SubsetError),
}

/// Which side of a call argument check we are currently analyzing.
///
/// `NotAnalyzingACall` is required because `is_subset_eq` is also called in
/// contexts that are unrelated to callable argument-vs-parameter checks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ArgumentSide {
    Got,
    Want,
    #[default]
    NotAnalyzingACall,
}

impl ArgumentSide {
    fn negated(self) -> Self {
        match self {
            Self::Got => Self::Want,
            Self::Want => Self::Got,
            Self::NotAnalyzingACall => Self::NotAnalyzingACall,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub(crate) enum SubsetCacheContext {
    #[default]
    Default,
    Witness {
        witness_hash: u64,
        argument_side: ArgumentSide,
    },
}

// The context in which we are collecting residuals.
// - The `identity` is the particular Forall type that appeared as an argument
//   in a higher-order call
// - The `origin_vars` are `Vars` that correspond to scoped type parameters
//   inside of that argument (the "origin" of the generic behavior)
// - The `deferred_vars` are `Vars` that correspond to call-scope vars from
//   the higher-order call we were making; these might get "deferred" in the
//   sense that instead of finishing to a concrete type we may finish to a
//   CallableResidual if no other constraints on these types appear.
//
// TODO(stroxler): Rethink the names of fields here. It would be difficult to restack.
#[derive(Clone, Debug)]
pub struct ResidualWitnessContext {
    identity: ResidualIdentity,
    argument_side: ArgumentSide,
    origin_vars: SmallSet<Var>,
    deferred_vars: SmallSet<Var>,
}

impl ResidualWitnessContext {
    pub(crate) fn witness_hash(&self) -> u64 {
        self.identity.witness_hash
    }

    pub(crate) fn capture_candidate_vars(&self) -> SmallSet<Var> {
        self.origin_vars
            .iter()
            .chain(self.deferred_vars.iter())
            .copied()
            .collect()
    }

    pub(crate) fn extend_deferred_vars(&mut self, vars: SmallSet<Var>) {
        self.deferred_vars.extend(vars);
    }
}

#[derive(Clone, Debug, Default)]
pub struct CallContext {
    witness: Option<ResidualWitnessContext>,
    argument_side: ArgumentSide,
}

impl CallContext {
    pub fn outside() -> Self {
        Self::default()
    }

    pub fn with_argument_side(&self, argument_side: ArgumentSide) -> Self {
        Self {
            witness: self.witness.clone(),
            argument_side,
        }
    }

    pub(crate) fn with_witness_and_side(
        witness: ResidualWitnessContext,
        argument_side: ArgumentSide,
    ) -> Self {
        Self {
            witness: Some(witness),
            argument_side,
        }
    }

    pub fn residual_witness(&self) -> Option<&ResidualWitnessContext> {
        self.witness.as_ref()
    }

    pub fn residual_witness_mut(&mut self) -> Option<&mut ResidualWitnessContext> {
        self.witness.as_mut()
    }

    pub(crate) fn with_preserved_side(&self) -> Self {
        self.clone()
    }

    pub fn with_negated_side(&self) -> Self {
        Self {
            witness: self.witness.clone(),
            argument_side: self.argument_side.negated(),
        }
    }

    pub(crate) fn argument_side(&self) -> ArgumentSide {
        self.argument_side
    }

    pub(crate) fn residual_hooks_enabled(&self) -> bool {
        match &self.witness {
            Some(witness) => {
                self.argument_side == witness.argument_side
                    && !matches!(self.argument_side, ArgumentSide::NotAnalyzingACall)
            }
            None => false,
        }
    }

    pub(crate) fn subset_cache_context(&self) -> SubsetCacheContext {
        if let Some(witness) = &self.witness {
            // Context-scoped cache keying preserves memoization while keeping
            // witness/polarity-sensitive side effects isolated. Most checks run
            // under Default context and keep prior cache behavior.
            SubsetCacheContext::Witness {
                witness_hash: witness.identity.witness_hash,
                argument_side: self.argument_side,
            }
        } else {
            SubsetCacheContext::Default
        }
    }
}

/// A helper to implement subset ergonomically.
/// Should only be used within `crate::subset`, which implements part of it.
pub struct Subset<'a, Ans: LookupAnswer> {
    pub(crate) solver: &'a Solver,
    pub type_order: TypeOrder<'a, Ans>,
    gas: Gas,
    active_call_context: CallContext,
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
    pub subset_cache: SmallMap<(Type, Type, SubsetCacheContext), SubsetCacheEntry>,
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
    witness_deferred_vars: SmallMap<u64, SmallSet<Var>>,
}

impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
    fn type_witness_hash(ty: &Type) -> u64 {
        let mut hasher = DefaultHasher::new();
        ty.hash(&mut hasher);
        hasher.finish()
    }

    pub fn snapshot_witness_deferred_vars(&self) -> SmallMap<u64, SmallSet<Var>> {
        self.witness_deferred_vars.clone()
    }

    pub(crate) fn restore_witness_deferred_vars(
        &mut self,
        deferred_vars: SmallMap<u64, SmallSet<Var>>,
    ) {
        self.witness_deferred_vars = deferred_vars;
    }

    pub(crate) fn make_forall_witness_context(
        &mut self,
        got: &Type,
        vars: &QuantifiedHandle,
        want: &Type,
        call_context: &CallContext,
    ) -> CallContext {
        let argument_side = call_context.argument_side();
        CallContext::with_witness_and_side(
            ResidualWitnessContext {
                identity: ResidualIdentity {
                    witness_hash: Self::type_witness_hash(got),
                    target_vars: want.collect_maybe_placeholder_vars().into_iter().collect(),
                },
                argument_side,
                origin_vars: vars.0.iter().copied().collect(),
                deferred_vars: SmallSet::new(),
            },
            argument_side,
        )
    }

    pub fn is_consistent(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.is_subset_eq(got, want)?;
        self.is_subset_eq(want, got)
    }

    pub fn is_equivalent(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.is_consistent(&got.materialize(), want)?;
        self.is_consistent(got, &want.materialize())
    }

    pub fn is_subset_eq(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        let call_context = self.active_call_context.clone();
        self.is_subset_eq_with_context(got, want, &call_context)
    }

    pub fn is_subset_eq_with_context(
        &mut self,
        got: &Type,
        want: &Type,
        call_context: &CallContext,
    ) -> Result<(), SubsetError> {
        let old_context = mem::replace(&mut self.active_call_context, call_context.clone());
        if self.gas.stop() {
            // We really have no idea. Just give up for now.
            self.active_call_context = old_context;
            return Err(SubsetError::Other);
        }
        if matches!(got, Type::Materialization) {
            let res = self.is_subset_eq_with_context(
                &self
                    .solver
                    .heap
                    .mk_class_type(self.type_order.stdlib().object().clone()),
                want,
                call_context,
            );
            self.active_call_context = old_context;
            return res;
        } else if matches!(want, Type::Materialization) {
            let res =
                self.is_subset_eq_with_context(got, &self.solver.heap.mk_never(), call_context);
            self.active_call_context = old_context;
            return res;
        }
        let res = self.is_subset_eq_var(got, want, call_context);
        self.gas.restore();
        self.active_call_context = old_context;
        res
    }

    pub(crate) fn with_active_call_context<T>(
        &mut self,
        call_context: &CallContext,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old_context = mem::replace(&mut self.active_call_context, call_context.clone());
        let res = f(self);
        self.active_call_context = old_context;
        res
    }

    pub(crate) fn take_witness_deferred_vars(
        &mut self,
        witness_hash: u64,
    ) -> Option<SmallSet<Var>> {
        self.witness_deferred_vars.shift_remove(&witness_hash)
    }

    pub(crate) fn active_argument_side(&self) -> ArgumentSide {
        self.active_call_context.argument_side()
    }

    pub(crate) fn make_overload_witness_context(
        &mut self,
        got: &Type,
        eligible_vars: &[Var],
        argument_side: ArgumentSide,
    ) -> CallContext {
        let target_vars: SmallSet<Var> = eligible_vars.iter().copied().collect();
        let origin_vars = target_vars.clone();
        CallContext::with_witness_and_side(
            ResidualWitnessContext {
                identity: ResidualIdentity {
                    witness_hash: Self::type_witness_hash(got),
                    target_vars,
                },
                argument_side,
                origin_vars,
                deferred_vars: SmallSet::new(),
            },
            argument_side,
        )
    }

    pub(crate) fn active_overload_residual_witness(&self) -> Option<ResidualWitnessContext> {
        if !self.active_call_context.residual_hooks_enabled() {
            return None;
        }
        let mut witness = self.active_call_context.residual_witness()?.clone();
        if let Some(deferred_vars) = self.witness_deferred_vars.get(&witness.witness_hash()) {
            witness.extend_deferred_vars(deferred_vars.clone());
        }
        Some(witness)
    }

    fn record_deferred_residual_target_vars(&mut self, origin_var: Var, other: &Type) {
        if !self.active_call_context.residual_hooks_enabled() {
            return;
        }
        let Some(witness) = self.active_call_context.residual_witness_mut() else {
            return;
        };
        if !witness.origin_vars.contains(&origin_var) {
            return;
        }
        let witness_hash = witness.identity.witness_hash;
        let target_vars = witness.identity.target_vars.clone();
        let deferred_vars = self.witness_deferred_vars.entry(witness_hash).or_default();
        for var in other.collect_maybe_placeholder_vars() {
            if target_vars.contains(&var) {
                deferred_vars.insert(var);
            }
        }
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
                        message_override: None,
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
                    message_override: None,
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
    fn is_subset_eq_var(
        &mut self,
        got: &Type,
        want: &Type,
        call_context: &CallContext,
    ) -> Result<(), SubsetError> {
        if let Some(witness) = call_context.residual_witness() {
            let _ = (
                &witness.identity,
                &witness.origin_vars,
                &witness.deferred_vars,
            );
        }
        match (got, want) {
            _ if got == want => Ok(()),
            (Type::Var(v1), Type::Var(v2)) => {
                self.record_deferred_residual_target_vars(*v1, want);
                self.record_deferred_residual_target_vars(*v2, got);
                let variables = self.solver.variables.lock();
                // Variable unification is destructive, so we have to copy bounds first.
                let root1 = variables.get_root(*v1);
                let root2 = variables.get_root(*v2);
                if root1 == root2 {
                    // same variable after unification, nothing to do
                } else {
                    // TODO(https://github.com/facebook/pyrefly/issues/105): unifying vars in this
                    // scenario is probably wrong. v1 <: v2 should mean that v2 gains v1 as a lower
                    // bound and v1 gains v2 as an upper bound, not that the two are now equal.
                    let mut v1_mut = variables.get_mut(*v1);
                    let mut v2_mut = variables.get_mut(*v2);
                    match (&mut *v1_mut, &mut *v2_mut) {
                        (
                            Variable::Quantified {
                                quantified: _,
                                bounds: v1_bounds,
                                residuals: v1_residuals,
                                overload_residuals: v1_overload_residuals,
                            },
                            Variable::Quantified {
                                quantified: _,
                                bounds: v2_bounds,
                                residuals: v2_residuals,
                                overload_residuals: v2_overload_residuals,
                            },
                        ) => {
                            v1_bounds.extend(mem::take(v2_bounds));
                            *v2_bounds = v1_bounds.clone();
                            Solver::merge_residual_candidates(v1_residuals, v2_residuals);
                            // Overload residual merge semantics (identity-aware dedupe/pruning)
                            // are not wired yet; for now keep all captured candidates.
                            v1_overload_residuals.extend(mem::take(v2_overload_residuals));
                            *v2_overload_residuals = v1_overload_residuals.clone();
                        }
                        (
                            Variable::Quantified {
                                quantified: _,
                                bounds: v1_bounds,
                                ..
                            }
                            | Variable::Unwrap(v1_bounds),
                            Variable::Quantified {
                                quantified: _,
                                bounds: v2_bounds,
                                ..
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
                let solved1 = match &*variable1 {
                    Variable::Answer(t1) => Some(t1.clone()),
                    Variable::ResidualAnswer { target_vars, ty } => Some(
                        self.solver
                            .residual_read_for_query_var(Some(*v1), target_vars, ty),
                    ),
                    _ => None,
                };
                let solved2 = match &*variable2 {
                    Variable::Answer(t2) => Some(t2.clone()),
                    Variable::ResidualAnswer { target_vars, ty } => Some(
                        self.solver
                            .residual_read_for_query_var(Some(*v2), target_vars, ty),
                    ),
                    _ => None,
                };
                if let (Some(t1), Some(t2)) = (solved1.clone(), solved2.clone()) {
                    drop(variable1);
                    drop(variable2);
                    drop(variables);
                    self.is_subset_eq(&t1, &t2)
                } else if let Some(t2) = solved2 {
                    drop(variable1);
                    drop(variable2);
                    drop(variables);
                    self.is_subset_eq(got, &t2)
                } else if let Some(t1) = solved1 {
                    drop(variable1);
                    drop(variable2);
                    drop(variables);
                    self.is_subset_eq(&t1, want)
                } else {
                    match (&*variable1, &*variable2) {
                        // When both variables are quantified, we need to preserve the stricter bound.
                        // The `unify` function preserves the Variable data from its second argument,
                        // so we call it with the stricter bound in the v2 position.
                        (
                            Variable::Quantified {
                                quantified: q1,
                                bounds: _,
                                ..
                            },
                            Variable::Quantified {
                                quantified: q2,
                                bounds: _,
                                ..
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
                                ..
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
            }
            (Type::Var(v1), t2) => {
                self.record_deferred_residual_target_vars(*v1, t2);
                let variables = self.solver.variables.lock();
                let v1_ref = variables.get(*v1);
                match &*v1_ref {
                    Variable::Answer(t1) => {
                        let t1 = t1.clone();
                        drop(v1_ref);
                        drop(variables);
                        self.is_subset_eq(&t1, t2)
                    }
                    Variable::ResidualAnswer {
                        target_vars,
                        ty: t1,
                    } => {
                        let t1 =
                            self.solver
                                .residual_read_for_query_var(Some(*v1), target_vars, t1);
                        drop(v1_ref);
                        drop(variables);
                        self.is_subset_eq(&t1, t2)
                    }
                    Variable::Quantified {
                        quantified: q,
                        bounds: _,
                        ..
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
                                        message_override: None,
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
                                        message_override: None,
                                        error: e,
                                    },
                                );
                            }
                        }
                        // Widen None to None | Any for PartialQuantified, matching
                        // the PartialContained behavior (see comment there).
                        let variables = self.solver.variables.lock();
                        let v1_current = variables.get(*v1);
                        if let Variable::Answer(t) | Variable::ResidualAnswer { ty: t, .. } =
                            &*v1_current
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
                self.record_deferred_residual_target_vars(*v2, t1);
                let variables = self.solver.variables.lock();
                let v2_ref = variables.get(*v2);
                match &*v2_ref {
                    Variable::Answer(t2) => {
                        let t2 = t2.clone();
                        drop(v2_ref);
                        drop(variables);
                        self.is_subset_eq(t1, &t2)
                    }
                    Variable::ResidualAnswer {
                        target_vars,
                        ty: t2,
                    } => {
                        let t2 =
                            self.solver
                                .residual_read_for_query_var(Some(*v2), target_vars, t2);
                        drop(v2_ref);
                        drop(variables);
                        self.is_subset_eq(t1, &t2)
                    }
                    Variable::Quantified {
                        quantified: q,
                        bounds,
                        ..
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
            _ => self.is_subset_eq_impl(got, want, call_context),
        }
    }

    pub fn finish_quantified(
        &mut self,
        vs: QuantifiedHandle,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        self.finish_quantified_with_infer(vs, self.solver.infer_with_first_use)
    }

    pub fn finish_quantified_with_infer(
        &mut self,
        vs: QuantifiedHandle,
        infer_with_first_use: bool,
    ) -> Result<(), Vec1<TypeVarSpecializationError>> {
        self.solver
            .finish_quantified_with_subset(vs, infer_with_first_use, &mut |got, want| {
                self.is_subset_eq_probe_for_pruning(got, want)
            })
    }
}
