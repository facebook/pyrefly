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
use std::fmt;
use std::fmt::Display;
use std::mem;

use pyrefly_types::quantified::Quantified;
use pyrefly_types::simplify::intersect;
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
use crate::alt::attr::AttrSubsetError;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::solver::type_order::TypeOrder;
use crate::types::callable::Callable;
use crate::types::callable::Function;
use crate::types::callable::Params;
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

const INITIAL_GAS: Gas = Gas::new(1000);

#[derive(Debug)]
enum Variable {
    /// A "partial type" (terminology borrowed from mypy) for an empty container.
    ///
    /// Pyrefly only creates partial types for assignments, and will attempt to
    /// determine the type ("pin" it) using the first use of the name assigned.
    ///
    /// It will attempt to infer the type from the first downsteam use; if the
    /// type cannot be determined it becomes `Any`.
    PartialContained,
    /// A "partial type" (see above) representing a type variable that was not
    /// solved as part of a generic function or constructor call.
    ///
    /// Behaves similar to `PartialContained`, but it has the ability to use
    /// the default type if the first use does not pin.
    PartialQuantified(Box<Quantified>),
    /// A variable due to generic instantiation, `def f[T](x: T): T` with `f(1)`
    Quantified(Box<Quantified>),
    /// A variable caused by general recursion, e.g. `x = f(); def f(): return x`.
    Recursive,
    /// A loop-recursive variable, e.g. `x = None; while x is None: x = f()`
    /// The Variable tracks the prior type bound to this name before the loop.
    LoopRecursive(Type, LoopBound),
    /// A variable that used to decompose a type, e.g. getting T from Awaitable[T]
    Unwrap,
    /// A variable used for a parameter type (either a function or lambda parameter).
    ///
    /// These are created at binding time, and used so that two bindings (the parameter
    /// and either the expression containing the lambda body or the function def)
    /// can communicate out-of-band about the parameter type.
    Parameter,
    /// A variable whose answer has been determined
    Answer(Type),
}

impl Variable {
    fn finished(q: &Quantified) -> Self {
        if let Some(d) = q.default() {
            Variable::Answer(d.clone())
        } else {
            Variable::PartialQuantified(Box::new(q.clone()))
        }
    }
}

fn default(q: &Quantified) -> Type {
    q.default().cloned().unwrap_or_else(Type::any_implicit)
}

/// The restrictions placed on a `LoopRecursive` Var during recursive solve of
/// the loop Phi.
#[derive(Debug)]
enum LoopBound {
    /// The variable may have been used recursively, and in the process we
    /// accumulated some upper bounds on it. It is a type error if the final
    /// solved Phi is not a subtype of these.
    UpperBounds(Vec<Type>),
    /// At some point we forced the Var - for example to resolve method access
    /// or to solve a type variable in a generic function call. The type should be
    /// exactly the loop prior. It is a type error if the final solved Phi does not match.
    Prior,
}

impl Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variable::PartialContained => write!(f, "PartialContained"),
            Variable::PartialQuantified(q) | Variable::Quantified(q) => {
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
            Variable::LoopRecursive(t, _) => write!(f, "LoopRecursive(prior={t}, _)"),
            Variable::Recursive => write!(f, "Recursive"),
            Variable::Parameter => write!(f, "Parameter"),
            Variable::Unwrap => write!(f, "Unwrap"),
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
#[derive(Debug)]
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
pub struct Solver {
    variables: Mutex<Variables>,
    instantiation_errors: RwLock<SmallMap<Var, TypeVarSpecializationError>>,
    pub infer_with_first_use: bool,
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

impl Solver {
    /// Create a new solver.
    pub fn new(infer_with_first_use: bool) -> Self {
        Self {
            variables: Default::default(),
            instantiation_errors: Default::default(),
            infer_with_first_use,
        }
    }

    pub fn recurse<'a>(&self, var: Var, recurser: &'a VarRecurser) -> Option<Guard<'a, Var>> {
        self.variables.lock().recurse(var, recurser)
    }

    /// Force all non-recursive Vars in `vars`.
    ///
    /// TODO: deduplicate Variable-to-gradual-type logic with `force_var`.
    pub fn pin_placeholder_type(&self, var: Var) {
        let variables = self.variables.lock();
        let mut variable = variables.get_mut(var);
        match &mut *variable {
            Variable::LoopRecursive(..) | Variable::Recursive | Variable::Answer(..) => {
                // Nothing to do if we have an answer already, and we want to skip recursive Vars
                // which do not represent placeholder types.
            }
            Variable::Quantified(q) => {
                *variable = Variable::Answer(q.as_gradual_type());
            }
            Variable::PartialQuantified(q) => {
                *variable = Variable::Answer(default(q));
            }
            Variable::PartialContained | Variable::Unwrap => {
                *variable = Variable::Answer(Type::any_implicit());
            }
            Variable::Parameter => {
                unreachable!("Unexpected Variable::Parameter")
            }
        }
    }

    /// Finish the type returned from a function call. This entails expanding solved variables and
    /// erasing unsolved variables without defaults from unions.
    pub fn finish_function_return(&self, mut t: Type) -> Type {
        self.expand_with_limit(&mut t, TYPE_LIMIT, &VarRecurser::new());
        self.erase_unsolved_variables(&mut t);
        self.simplify_mut(&mut t);
        t
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
        self.expand_with_limit(t, TYPE_LIMIT, &VarRecurser::new());
        // After we substitute bound variables, we may be able to simplify some types
        self.simplify_mut(t);
    }

    /// Expand, but if the resulting type will be greater than limit levels deep, return an `Any`.
    /// Avoids producing things that stack overflow later in the process.
    fn expand_with_limit(&self, t: &mut Type, limit: usize, recurser: &VarRecurser) {
        if limit == 0 {
            // TODO: Should probably add an error here, and use any_error,
            // but don't have any good location information to hand.
            *t = Type::any_implicit();
        } else if let Type::Var(x) = t {
            let lock = self.variables.lock();
            if let Some(_guard) = lock.recurse(*x, recurser) {
                let variable = lock.get(*x);
                match &*variable {
                    Variable::Answer(ty) | Variable::LoopRecursive(ty, LoopBound::Prior) => {
                        *t = ty.clone();
                        drop(variable);
                        drop(lock);
                        self.expand_with_limit(t, limit - 1, recurser);
                    }
                    _ => {}
                }
            } else {
                *t = Type::any_implicit();
            }
        } else {
            t.recurse_mut(&mut |t| self.expand_with_limit(t, limit - 1, recurser));
        }
    }

    /// Given a `Var`, ensures that the solver has an answer for it (or inserts Any if not already),
    /// and returns that answer. Note that if the `Var` is already bound to something that contains a
    /// `Var` (including itself), then we will return the answer.
    pub fn force_var(&self, v: Var) -> Type {
        let lock = self.variables.lock();
        let mut e = lock.get_mut(v);
        match &mut *e {
            Variable::Answer(t) => t.clone(),
            Variable::LoopRecursive(prior_type, bound) => {
                *bound = LoopBound::Prior;
                prior_type.clone()
            }
            _ => {
                let ty = match &mut *e {
                    Variable::Quantified(q) => q.as_gradual_type(),
                    Variable::PartialQuantified(q) => default(q),
                    _ => Type::any_implicit(),
                };
                *e = Variable::Answer(ty.clone());
                ty
            }
        }
    }

    fn deep_force_mut_with_limit(&self, t: &mut Type, limit: usize, recurser: &VarRecurser) {
        if limit == 0 {
            // TODO: Should probably add an error here, and use any_error,
            // but don't have any good location information to hand.
            *t = Type::any_implicit();
        } else if let Type::Var(v) = t {
            if let Some(_guard) = self.recurse(*v, recurser) {
                *t = self.force_var(*v);
                self.deep_force_mut_with_limit(t, limit - 1, recurser);
            } else {
                *t = Type::any_implicit();
            }
        } else {
            t.recurse_mut(&mut |t| self.deep_force_mut_with_limit(t, limit - 1, recurser));
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
                let mut merged = unions(mem::take(xs));
                // Preserve union display names during simplification
                if let Type::Union(box Union { display_name, .. }) = &mut merged {
                    *display_name = original_name.clone();
                }
                *x = merged;
            }
            if let Type::Intersect(y) = x {
                *x = intersect(mem::take(&mut y.0), y.1.clone());
            }
            if let Type::Tuple(tuple) = x {
                *x = Type::Tuple(simplify_tuples(mem::take(tuple)));
            }
            // When a param spec is resolved, collapse any Concatenate and Callable types that use it
            if let Type::Concatenate(ts, box Type::ParamSpecValue(paramlist)) = x {
                let params = mem::take(paramlist).prepend_types(ts).into_owned();
                *x = Type::ParamSpecValue(params);
            }
            if let Type::Concatenate(ts, box Type::Concatenate(ts2, pspec)) = x {
                *x = Type::Concatenate(
                    ts.iter().chain(ts2.iter()).cloned().collect(),
                    pspec.clone(),
                );
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
                        Type::Function(Box::new(Function {
                            signature: c,
                            metadata: k.clone(),
                        }))
                    } else {
                        Type::Callable(Box::new(c))
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
                            *x = Type::never();
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
    pub fn fresh_partial_contained(&self, uniques: &UniqueFactory) -> Var {
        let v = Var::new(uniques);
        self.variables
            .lock()
            .insert_fresh(v, Variable::PartialContained);
        v
    }

    /// Generate a fresh variable for out-of-band logic that allows two bindings to communicate
    /// about a type without it being explicitly in a binding result. Used for function parameters,
    /// where the var is created during the bindings phase, then solved during answers, where we
    /// can determine whether the first argument should be Self or type[Self] based on decorators.
    ///
    /// Parameter vars must be solved before they appear in a constraint, by calling solve_parameter.
    /// If a parameter var appears in a constraint, we will panic.
    pub fn fresh_parameter(&self, uniques: &UniqueFactory) -> Var {
        let v = Var::new(uniques);
        self.variables.lock().insert_fresh(v, Variable::Parameter);
        v
    }

    /// Solve a parameter var (created using fresh_parameter) to a concrete type. This must happen
    /// before the var can appear in a constraint, or else we will panic.
    pub fn solve_parameter(&self, v: Var, t: Type) {
        let lock = self.variables.lock();
        let mut v = lock.get_mut(v);
        match &mut *v {
            Variable::Answer(_) => {}
            Variable::Parameter => {
                *v = Variable::Answer(t);
            }
            _ => {
                panic!("Expected a parameter, got {}", v);
            }
        }
    }

    // Generate a fresh variable used to decompose a type, e.g. getting T from Awaitable[T]
    // Also used for lambda parameters, where the var is created during bindings, but solved during
    // the answers phase by contextually typing against an annotation.
    pub fn fresh_unwrap(&self, uniques: &UniqueFactory) -> Var {
        let v = Var::new(uniques);
        self.variables.lock().insert_fresh(v, Variable::Unwrap);
        v
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

        let vs: Vec<_> = params.iter().map(|_| Var::new(uniques)).collect();
        let ts = vs.map(|v| v.to_type());
        let t = t.subst(&params.iter().map(|p| &p.quantified).zip(&ts).collect());
        let mut lock = self.variables.lock();
        for (v, param) in vs.iter().zip(params.iter()) {
            lock.insert_fresh(*v, Variable::Quantified(Box::new(param.quantified.clone())));
        }
        (QuantifiedHandle(vs), t)
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
            if tparams.iter().any(|tparam| tparam.quantified == *q) {
                qs.push(q.clone());
            }
        });

        if qs.is_empty() {
            return callable;
        }

        // Substitute fresh vars for the quantifieds in the self param.
        let vs = qs.map(|_| Var::new(uniques));
        let ts = vs.map(|v| v.to_type());
        let mp = qs.iter().zip(&ts).collect();
        let self_param = self_param.clone().subst(&mp);
        callable.visit_mut(&mut |t| t.subst_mut(&mp));
        drop(mp);

        let mut lock = self.variables.lock();
        for (v, q) in vs.iter().zip(qs.into_iter()) {
            lock.insert_fresh(*v, Variable::Quantified(Box::new(q)));
        }
        drop(lock);

        // Solve for the vars created above. If this errors, then the definition
        // is invalid, and we should have raised an error at the definition site.
        is_subset(self_obj, &self_param);

        // Either we have solutions, or we fall back to Any. We don't use finish_quantified
        // because we don't want Variable::Partial.
        for v in vs {
            self.force_var(v);
        }

        callable
    }

    pub fn has_instantiation_errors(&self, vs: &QuantifiedHandle) -> bool {
        let lock = self.instantiation_errors.read();
        vs.0.iter().any(|v| lock.contains_key(v))
    }

    /// Called after a quantified function has been called. Given `def f[T](x: int): list[T]`,
    /// after the generic has completed.
    /// If `infer_with_first_use` is true, the variable `T` will be have like an
    /// empty container and get pinned by the first subsequent usage.
    /// If `infer_with_first_use` is false, the variable `T` will be replaced with `Any`
    pub fn finish_quantified(
        &self,
        vs: QuantifiedHandle,
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
                Variable::Quantified(q) => {
                    if self.infer_with_first_use {
                        *e = Variable::finished(q);
                    } else {
                        *e = Variable::Answer(default(q))
                    }
                }
                _ => {}
            }
        }
        match Vec1::try_from_vec(err) {
            Ok(err) => Err(err),
            Err(_) => Ok(()),
        }
    }

    /// Given targs which contain quantified (as come from `instantiate`), replace the quantifieds
    /// with fresh vars. We can avoid substitution because tparams can not appear in the bounds of
    /// another tparam. tparams can appear in the default, but those are not in quantified form yet.
    pub fn freshen_class_targs(&self, targs: &mut TArgs, uniques: &UniqueFactory) {
        let mut lock = self.variables.lock();
        targs.iter_paired_mut().for_each(|(param, t)| {
            if let Type::Quantified(q) = t
                && **q == param.quantified
            {
                let v = Var::new(uniques);
                *t = v.to_type();
                lock.insert_fresh(v, Variable::Quantified(Box::new(param.quantified.clone())));
            }
        })
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
            if let Type::Var(v) = t
                && !matches!(&*lock.get(*v), Variable::Answer(_))
            {
                *t = param.quantified.clone().to_type();
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
                && **q == param.quantified
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
                                param.quantified.as_gradual_type()
                            }
                        }
                    });
                    Some(t)
                } else if self.infer_with_first_use {
                    let v = Var::new(uniques);
                    self.variables.lock().insert_fresh(v, Variable::finished(q));
                    Some(v.to_type())
                } else {
                    Some(default(q))
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

    pub fn fresh_loop_recursive(&self, uniques: &UniqueFactory, prior_type: Type) -> Var {
        let v = Var::new(uniques);
        self.variables.lock().insert_fresh(
            v,
            Variable::LoopRecursive(prior_type, LoopBound::UpperBounds(vec![])),
        );
        v
    }

    pub fn for_display(&self, t: Type) -> Type {
        let mut t = self.expand_vars(t);
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
        match tcc.context {
            Some(ctx) => {
                errors.add(loc, ErrorInfo::Context(&|| ctx.clone()), msg_lines);
            }
            None => {
                errors.add(loc, ErrorInfo::Kind(tcc.kind.as_error_kind()), msg_lines);
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
            return Type::never();
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
        unions_with_literals(branches, type_order.stdlib(), &|cls| {
            type_order.get_enum_member_count(cls)
        })
    }

    /// Record a variable that is used recursively.
    pub fn record_recursive<Ans: LookupAnswer>(
        &self,
        var: Var,
        ty: Type,
        type_order: TypeOrder<Ans>,
        errors: &ErrorCollector,
        range: TextRange,
    ) -> Type {
        fn expand(t: Type, variables: &Variables, recurser: &VarRecurser, res: &mut Vec<Type>) {
            match t {
                Type::Var(v) if let Some(_guard) = variables.recurse(v, recurser) => {
                    let variable = variables.get(v);
                    match &*variable {
                        Variable::Answer(t) => {
                            let t = t.clone();
                            drop(variable);
                            expand(t, variables, recurser, res);
                        }
                        _ => res.push(v.to_type()),
                    }
                }
                Type::Union(box Union { members: ts, .. }) => {
                    for t in ts {
                        expand(t, variables, recurser, res);
                    }
                }
                _ => res.push(t),
            }
        }

        let lock = self.variables.lock();
        let variable = lock.get(var);
        match &*variable {
            Variable::Answer(forced) => {
                let forced = forced.clone();
                drop(variable);
                drop(lock);
                // We got forced into choosing a type to satisfy a subset constraint, so check we are OK with that.
                // Since we have already used `forced`, and will continue to do so, important that what we expect
                // is more restrictive (so the `forced` is an over-approximation).
                if self.is_subset_eq(&ty, &forced, type_order).is_err() {
                    // Poor error message, but overall, this is a terrible experience for users.
                    self.error(
                        &ty,
                        &forced,
                        errors,
                        range,
                        &|| TypeCheckContext::of_kind(TypeCheckKind::CycleBreaking),
                        SubsetError::Other,
                    );
                }
                // In order to minimize the blast radius of poor cycle-handling, we currently produce
                // inconsistent results - any other binding that saved an answer which depended on
                // this one sees the forced type, but anything downstream of this sees the computed
                // type.
                //
                // This is both highly unpredictable in terms of end user experience, and nondeterministic
                // because we can definitely get non-idempotent errors in some cases.
                //
                // TODO(stroxler): Probably remove this - it regresses a CRTP example, so we should
                // remove it in a dedicated diff.
                ty
            }
            _ => {
                // If we just recorded a LoopRecursive answer, we may now need to check the result
                // against bounds introduced during the recursion.
                let bounds_to_check = match &*variable {
                    Variable::LoopRecursive(prior, LoopBound::Prior) => Some(vec![prior.clone()]),
                    Variable::LoopRecursive(_, LoopBound::UpperBounds(bounds)) => {
                        Some(bounds.clone())
                    }
                    _ => None,
                };
                drop(variable);
                // If you are recording `@1 = @1 | something` then the `@1` can't contribute any
                // possibilities, so just ignore it.
                let mut res = Vec::new();
                // First expand all union/var into a list of the possible unions
                expand(ty, &lock, &VarRecurser::new(), &mut res);
                // Then remove any reference to self, before unioning it back together
                res.retain(|x| x != &Type::Var(var));
                let ty = unions(res);
                match bounds_to_check {
                    Some(bounds) => {
                        lock.update(var, Variable::Answer(ty.clone()));
                        drop(lock);
                        bounds.iter().for_each(|bound| {
                            if self.is_subset_eq(&ty, bound, type_order).is_err() {
                                self.error(
                                    &ty,
                                    bound,
                                    errors,
                                    range,
                                    &|| TypeCheckContext::of_kind(TypeCheckKind::CycleBreaking),
                                    SubsetError::Other,
                                );
                            }
                        });
                    }
                    None => {
                        lock.update(var, Variable::Answer(ty.clone()));
                    }
                }
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

    pub fn is_equal<Ans: LookupAnswer>(
        &self,
        got: &Type,
        want: &Type,
        type_order: TypeOrder<Ans>,
    ) -> Result<(), SubsetError> {
        let mut subset = self.subset(type_order);
        subset.is_equal(got, want)
    }

    fn subset<'a, Ans: LookupAnswer>(&'a self, type_order: TypeOrder<'a, Ans>) -> Subset<'a, Ans> {
        Subset {
            solver: self,
            type_order,
            gas: INITIAL_GAS,
            recursive_assumptions: SmallSet::new(),
            class_protocol_assumptions: SmallSet::new(),
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
    #[allow(dead_code)]
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
    /// An invariant was violated - used for cases that should be unreachabile when - if there is ever a bug - we
    /// would prefer to not panic and get a text location for reproducing rather than just a crash report.
    InternalError(String),
    /// Protocol class names cannot be assigned to `type[P]` when `P` is a protocol
    TypeOfProtocolNeedsConcreteClass(Name),
    // TODO(rechen): replace this with specific reasons
    Other,
}

impl SubsetError {
    fn internal_error(msg: &str) -> Self {
        Self::InternalError(msg.to_owned())
    }

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
            SubsetError::InternalError(msg) => Some(format!("Pyrefly internal error: {msg}")),
            SubsetError::TypeOfProtocolNeedsConcreteClass(want) => Some(format!(
                "Only concrete classes may be assigned to `type[{want}]` because `{want}` is a protocol"
            )),
            SubsetError::Other => None,
        }
    }
}

/// A helper to implement subset ergonomically.
/// Should only be used within `crate::subset`, which implements part of it.
pub struct Subset<'a, Ans: LookupAnswer> {
    solver: &'a Solver,
    pub type_order: TypeOrder<'a, Ans>,
    gas: Gas,
    /// Recursive assumptions of pairs of types that is_subset_eq returns true for.
    /// Used for structural typechecking of protocols.
    pub recursive_assumptions: SmallSet<(Type, Type)>,
    /// Class-level recursive assumptions for protocol checks.
    /// When checking `got <: protocol` where got's type arguments contain Vars
    /// (indicating we're in a recursive pattern), we track (got_class, protocol_class)
    /// pairs to detect cycles. This enables coinductive reasoning for recursive protocols
    /// like Functor/Maybe without falsely assuming success for unrelated protocol checks.
    pub class_protocol_assumptions: SmallSet<(Class, Class)>,
}

impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
    pub fn is_equal(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.is_subset_eq(got, want)?;
        self.is_subset_eq(want, got)
    }

    pub fn is_subset_eq(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        if self.gas.stop() {
            // We really have no idea. Just give up for now.
            return Err(SubsetError::Other);
        }
        let res = self.is_subset_eq_var(got, want);
        self.gas.restore();
        res
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
                let variable1 = variables.get(*v1);
                let variable2 = variables.get(*v2);
                match (&*variable1, &*variable2) {
                    (_, Variable::LoopRecursive(t2, _)) => {
                        // When a LoopRecursive variable appears as `want`, use its prior type instead.
                        // This handles nested loops where an inner loop variable's prior depends on
                        // an outer loop variable that is still being analyzed.
                        // See: https://github.com/facebook/pyrefly/issues/1565
                        let t2 = t2.clone();
                        drop(variable1);
                        drop(variable2);
                        drop(variables);
                        self.is_subset_eq(got, &t2)
                    }
                    (Variable::Parameter, _) | (_, Variable::Parameter) => {
                        Err(SubsetError::internal_error(
                            "Did not expect a `Variable::Parameter` to ever appear in is_subset_eq",
                        ))
                    }
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
                    (Variable::LoopRecursive(t1, bound), _) => {
                        // If we have to solve a type variable against a loop recursive type, it means
                        // we need to commit to the loop recursion type, so we
                        // should pin it to the loop prior.
                        let already_pinned = matches!(bound, LoopBound::Prior);
                        let mut t1 = t1.clone();
                        drop(variable1);
                        drop(variable2);
                        // Note: we have to drop `variable1` and `variable2` before we can work with
                        // a mutable reference because the `.get` with path compression requires no
                        // mutable refs to already be taken when calling `.get` or `.get_mut`
                        if !already_pinned {
                            let mut variable1 = variables.get_mut(*v1);
                            match &mut *variable1 {
                                Variable::LoopRecursive(_, bound) => {
                                    // TODO(stroxler): track a range here for better error messages.
                                    *bound = LoopBound::Prior;
                                }
                                // Reachable through data races - if the current thread picked up a
                                // LoopRecursive that another thread is solving, the other thread might write
                                // the final answer while we don't hold the lock. Note that the resulting
                                // behavior is nondeterministic.
                                Variable::Answer(t1_answer) => {
                                    t1 = t1_answer.clone();
                                }
                                _ => unreachable!(
                                    "Did not expect a LoopRecursive to become any other Variable kind besides Answer."
                                ),
                            }
                            drop(variable1)
                        }
                        drop(variables);
                        self.is_subset_eq(&t1, want)
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
                    Variable::Parameter => Err(SubsetError::internal_error(
                        "Did not expect a `Variable::Parameter` to ever appear in is_subset_eq",
                    )),
                    Variable::Answer(t1) => {
                        let t1 = t1.clone();
                        drop(v1_ref);
                        drop(variables);
                        self.is_subset_eq(&t1, t2)
                    }
                    Variable::Quantified(q) | Variable::PartialQuantified(q) => {
                        let name = q.name.clone();
                        let bound = q.restriction().as_type(self.type_order.stdlib());
                        drop(v1_ref);
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
                        Ok(())
                    }
                    Variable::LoopRecursive(t1, ..) => {
                        let t1 = t1.clone();
                        drop(v1_ref);
                        drop(variables);
                        match self.is_subset_eq(&t1, t2) {
                            Ok(()) => {
                                // If the check passed, we need to track an upper bound. That requires
                                // retaking the locks.
                                let variables = self.solver.variables.lock();
                                let mut variable1 = variables.get_mut(*v1);
                                match &mut *variable1 {
                                    Variable::LoopRecursive(_, bound) => {
                                        match bound {
                                            LoopBound::UpperBounds(ubs) => ubs.push(t2.clone()),
                                            LoopBound::Prior => {
                                                // Nothing to do; `t1` is assignable to `t2`
                                            }
                                        }
                                        Ok(())
                                    }
                                    // This is reachable through data races - if the current thread picked up a
                                    // LoopRecursive that another thread is solving, the other thread might write
                                    // the final answer while we don't hold the lock. Note that the resulting
                                    // behavior is nondeterministic.
                                    Variable::Answer(t1) => {
                                        let t1 = t1.clone();
                                        drop(variable1);
                                        drop(variables);
                                        self.is_subset_eq(&t1, t2)
                                    }
                                    _ => unreachable!(
                                        "Did not expect a LoopRecursive variable to become {:?}",
                                        variable1
                                    ),
                                }
                            }
                            Err(e) => Err(e),
                        }
                    }
                    Variable::PartialContained | Variable::Unwrap | Variable::Recursive => {
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
                    Variable::LoopRecursive(t2, _) => {
                        // When a LoopRecursive variable appears as `want`, use its prior type instead.
                        // This handles nested loops where an inner loop variable's prior depends on
                        // an outer loop variable that is still being analyzed.
                        // See: https://github.com/facebook/pyrefly/issues/1565
                        let t2 = t2.clone();
                        drop(v2_ref);
                        drop(variables);
                        self.is_subset_eq(t1, &t2)
                    }
                    Variable::Parameter => Err(SubsetError::internal_error(
                        "Did not expect a `Variable::Parameter` to ever appear in is_subset_eq",
                    )),
                    Variable::Answer(t2) => {
                        let t2 = t2.clone();
                        drop(v2_ref);
                        drop(variables);
                        self.is_subset_eq(t1, &t2)
                    }
                    Variable::Quantified(q) | Variable::PartialQuantified(q) => {
                        let t1_p = t1.clone().promote_literals(self.type_order.stdlib());
                        let name = q.name.clone();
                        let bound = q.restriction().as_type(self.type_order.stdlib());
                        drop(v2_ref);
                        variables.update(*v2, Variable::Answer(t1_p.clone()));
                        drop(variables);
                        if let Err(err_p) = self.is_subset_eq(&t1_p, &bound) {
                            // If the promoted type fails, try again with the original type, in case the bound itself is literal.
                            // This could be more optimized, but errors are rare, so this code path should not be hot.
                            self.solver
                                .variables
                                .lock()
                                .update(*v2, Variable::Answer(t1.clone()));
                            if self.is_subset_eq(t1, &bound).is_err() {
                                // If the original type is also an error, use the promoted type.
                                self.solver
                                    .variables
                                    .lock()
                                    .update(*v2, Variable::Answer(t1_p.clone()));
                                self.solver.instantiation_errors.write().insert(
                                    *v2,
                                    TypeVarSpecializationError {
                                        name,
                                        got: t1_p.clone(),
                                        want: bound,
                                        error: err_p,
                                    },
                                );
                            }
                        }
                        Ok(())
                    }
                    Variable::PartialContained => {
                        let t1_p = t1.clone().promote_literals(self.type_order.stdlib());
                        drop(v2_ref);
                        variables.update(*v2, Variable::Answer(t1_p));
                        Ok(())
                    }
                    Variable::Unwrap | Variable::Recursive => {
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
        self.solver.finish_quantified(vs)
    }
}
