/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::types::Union;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_bases::ClassBases;
use crate::types::callable::Params;
use crate::types::class::Class;
use crate::types::tuple::Tuple;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Variance;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::Type;

// This is our variance inference algorithm, which determines variance based on visiting the structure of the type.
// There are a couple of TODO that I [zeina] would like to revisit as I figure them out. There are several types that I'm not visiting (and did not visit similar ones in pyre1),
// And I'm not yet clear what variance inference should do on those:

// Those types are:
// - Concatenate
// - Intersect (Our variance inference algorithm is not defined on this. Unclear to me yet what to do on this type.)
// - Forall (I suspect that we should not visit this, since the forall type is related to a function, and variance makes no sense in the absence of a class definition)
// - Unpack (potentially just visit the inner type recursively?)
// - SpecialForm
// - ParamSpecValue
// - Args and Kwargs
// - SuperInstance
// - TypeGuard
// - TypeIs

// We need to visit the types that we know are required to be visited for variance inference, and appear in the context of a class with type variables.
// For example, SelfType is intentionally skipped and should not be visited because it should not be included in the variance calculation.

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, Default, VisitMut)]
pub struct VarianceMap(SmallMap<Name, Variance>);

impl Display for VarianceMap {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{{")?;
        for (key, value) in self.0.iter() {
            write!(f, "{key}: {value}, ")?;
        }
        write!(f, "}}")
    }
}

impl VarianceMap {
    pub fn get(&self, parameter: &Name) -> Variance {
        self.0
            .get(parameter)
            .copied()
            .unwrap_or(Variance::Invariant)
    }
}

#[derive(Debug, Clone)]
pub struct VarianceViolation {
    pub range: TextRange,
    pub var_name: Name,
    pub position_variance: Variance,
    pub declared_variance: PreInferenceVariance,
}

impl VarianceViolation {
    pub fn format_message(&self) -> String {
        format!(
            "Type variable `{}` is {} but is used in {} position",
            self.var_name, self.declared_variance, self.position_variance
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct VarianceResult {
    pub variance_map: VarianceMap,
    pub violations: Vec<VarianceViolation>,
}

#[derive(Debug, Clone, Copy)]
struct InferenceStatus {
    inferred_variance: Variance,
    has_variance_inferred: bool,
    specified_variance: Option<Variance>,
}
type InferenceMap = SmallMap<Name, InferenceStatus>;

// A map from class name to tparam environment
// Why is this not Class or ClassObject
type VarianceEnv = SmallMap<Class, InferenceMap>;

fn handle_tuple_type(
    tuple: &Tuple,
    variance: Variance,
    inj: bool,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, Variance, bool, PreInferenceVariance),
) {
    match tuple {
        Tuple::Concrete(concrete_types) => {
            for ty in concrete_types {
                on_type(variance, inj, ty, on_edge, on_var);
            }
        }
        Tuple::Unbounded(unbounded_ty) => {
            on_type(variance, inj, unbounded_ty, on_edge, on_var);
        }
        Tuple::Unpacked(boxed_parts) => {
            let (before, middle, after) = &**boxed_parts;
            for ty in before {
                on_type(variance, inj, ty, on_edge, on_var);
            }
            on_type(variance, inj, middle, on_edge, on_var);
            for ty in after {
                on_type(variance, inj, ty, on_edge, on_var);
            }
        }
    }
}

fn on_type(
    variance: Variance,
    inj: bool,
    typ: &Type,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, Variance, bool, PreInferenceVariance),
) {
    match typ {
        Type::Type(t) => {
            on_type(variance, inj, t, on_edge, on_var);
        }

        Type::Function(t) => {
            // Walk return type covariantly
            on_type(variance, inj, &t.signature.ret, on_edge, on_var);

            // Walk parameters contravariantly
            match &t.signature.params {
                Params::List(param_list) => {
                    for param in param_list.items().iter() {
                        let ty = param.as_type();
                        on_type(variance.inv(), inj, ty, on_edge, on_var);
                    }
                }
                Params::Ellipsis | Params::Materialization => {
                    // Unknown params
                }
                Params::ParamSpec(prefix, param_spec) => {
                    for (ty, _) in prefix.iter() {
                        on_type(variance.inv(), inj, ty, on_edge, on_var);
                    }
                    on_type(variance.inv(), inj, param_spec, on_edge, on_var);
                }
            }
        }

        Type::ClassType(class) => {
            let params = on_edge(class.class_object());
            let targs = class.targs().as_slice();

            // If targs is empty, nothing to do
            if targs.is_empty() {
                return;
            }

            // Zip params (from on_edge) with targs
            // Note: if params.len() != targs.len(), zip will stop at the shorter one
            for (status, ty) in params.values().zip(targs) {
                // Use specified_variance if available (for externally defined TypeVars
                // with explicit variance like covariant=True), otherwise use inferred.
                let effective_variance = status
                    .specified_variance
                    .unwrap_or(status.inferred_variance);
                on_type(
                    variance.compose(effective_variance),
                    status.has_variance_inferred,
                    ty,
                    on_edge,
                    on_var,
                );
            }
        }
        Type::Quantified(q) => {
            on_var(q.name(), variance, inj, q.variance());
        }
        Type::Union(box Union { members: tys, .. }) => {
            for ty in tys {
                on_type(variance, inj, ty, on_edge, on_var);
            }
        }
        Type::Overload(t) => {
            let sigs = &t.signatures;
            for sig in sigs {
                on_type(variance, inj, &sig.as_type(), on_edge, on_var);
            }
        }
        Type::Callable(t) => {
            // Walk return type covariantly
            on_type(variance, inj, &t.ret, on_edge, on_var);

            // Walk parameters contravariantly
            match &t.params {
                Params::List(param_list) => {
                    for param in param_list.items().iter() {
                        let ty = param.as_type();
                        on_type(variance.inv(), inj, ty, on_edge, on_var);
                    }
                }
                Params::Ellipsis | Params::Materialization => {
                    // Unknown params
                }
                Params::ParamSpec(prefix, param_spec) => {
                    for (ty, _) in prefix.iter() {
                        on_type(variance.inv(), inj, ty, on_edge, on_var);
                    }
                    on_type(variance.inv(), inj, param_spec, on_edge, on_var);
                }
            }
        }
        Type::Tuple(t) => {
            handle_tuple_type(t, variance, inj, on_edge, on_var);
        }
        Type::Forall(forall) => {
            // Methods with type parameters are wrapped in Forall. We need to visit
            // the body to find class-level type variables used within.
            on_type(
                variance,
                inj,
                &forall.body.clone().as_type(),
                on_edge,
                on_var,
            );
        }

        _ => {}
    }
}

fn on_class(
    class: &Class,
    heap: &TypeHeap,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, Variance, bool, PreInferenceVariance),
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
) {
    fn is_private_field(name: &Name) -> bool {
        let starts_with_underscore = name.starts_with('_');
        let ends_with_double_underscore = name.ends_with("__");

        starts_with_underscore && !ends_with_double_underscore
    }

    for base_type in get_class_bases(class).iter() {
        on_type(
            Variance::Covariant,
            true,
            &heap.mk_class_type(base_type.clone()),
            on_edge,
            on_var,
        );
    }

    let fields = get_fields(class);

    // todo zeina: check if we need to check for things like __init_subclass__
    // in pyre 1, we didn't need to.
    for (name, field) in fields.iter() {
        if name == &dunder::INIT {
            continue;
        }

        if let Some((ty, _, read_only)) = field.for_variance_inference() {
            // TODO: We need a much better way to distinguish between fields and methods than this
            // currently, class field representation isn't good enough but we need to fix that soon
            let variance = if ty.is_toplevel_callable()
                || is_private_field(name)
                || read_only
                || field.is_final()
            {
                Variance::Covariant
            } else {
                Variance::Invariant
            };
            on_type(variance, true, ty, on_edge, on_var);
        }
    }
}

/// Check a type variable for variance violations.
fn check_typevar(
    name: &Name,
    position_variance: Variance,
    declared_variance: PreInferenceVariance,
    range: TextRange,
    violations: &mut Vec<VarianceViolation>,
) {
    let is_valid = match declared_variance {
        PreInferenceVariance::Covariant => position_variance == Variance::Covariant,
        PreInferenceVariance::Contravariant => position_variance == Variance::Contravariant,
        // Invariant type variables can be used in any position (covariant, contravariant, or both)
        PreInferenceVariance::Invariant => true,
        // PEP695: variance will be inferred, no check needed
        PreInferenceVariance::Undefined => true,
    };
    if !is_valid {
        violations.push(VarianceViolation {
            range,
            var_name: name.clone(),
            position_variance,
            declared_variance,
        });
    }
}

/// Check method for variance violations (shallow - only direct TypeVars in params/return).
fn check_method_shallow(typ: &Type, range: TextRange, violations: &mut Vec<VarianceViolation>) {
    match typ {
        Type::Forall(forall) => {
            check_method_shallow(&forall.body.clone().as_type(), range, violations);
        }
        Type::Function(t) => {
            // Check return type (covariant position)
            if let Type::Quantified(q) = &t.signature.ret {
                check_typevar(
                    q.name(),
                    Variance::Covariant,
                    q.variance(),
                    range,
                    violations,
                );
            }
            // Check parameters (contravariant position)
            if let Params::List(param_list) = &t.signature.params {
                for param in param_list.items().iter() {
                    if let Type::Quantified(q) = param.as_type() {
                        check_typevar(
                            q.name(),
                            Variance::Contravariant,
                            q.variance(),
                            range,
                            violations,
                        );
                    }
                }
            }
        }
        Type::Callable(t) => {
            // Check return type (covariant position)
            if let Type::Quantified(q) = &t.ret {
                check_typevar(
                    q.name(),
                    Variance::Covariant,
                    q.variance(),
                    range,
                    violations,
                );
            }
            // Check parameters (contravariant position)
            if let Params::List(param_list) = &t.params {
                for param in param_list.items().iter() {
                    if let Type::Quantified(q) = param.as_type() {
                        check_typevar(
                            q.name(),
                            Variance::Contravariant,
                            q.variance(),
                            range,
                            violations,
                        );
                    }
                }
            }
        }
        _ => {}
    }
}

/// Check a class for variance violations.
///
/// Checking behavior:
/// - Base classes: DEEP checking (recurse into all nested generics)
/// - Methods: SHALLOW checking (only direct TypeVar usage, not nested Callables)
/// - Fields: NO checking (mutable fields constrain variance during inference only)
fn check_class_violations(
    class: &Class,
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
    get_tparams: &impl Fn(&Class) -> Arc<TParams>,
) -> Vec<VarianceViolation> {
    let mut violations = Vec::new();

    // Check base classes deeply using on_type for traversal
    for (base_type, range) in get_class_bases(class).iter_with_ranges() {
        let mut on_var =
            |name: &Name, variance: Variance, _inj: bool, declared: PreInferenceVariance| {
                check_typevar(name, variance, declared, range, &mut violations);
            };
        let mut on_edge = |c: &Class| initial_inference_map(get_tparams(c).as_vec());
        on_type(
            Variance::Covariant,
            true,
            &base_type.clone().to_type(),
            &mut on_edge,
            &mut on_var,
        );
    }

    // Check methods shallowly
    let fields = get_fields(class);
    for (name, field) in fields.iter() {
        if name == &dunder::INIT || name == &dunder::NEW {
            continue;
        }
        if let Some((ty, _, _)) = field.for_variance_inference()
            && ty.is_toplevel_callable()
        {
            let range = class
                .field_decl_range(name)
                .unwrap_or_else(|| class.range());
            check_method_shallow(ty, range, &mut violations);
        }
    }

    violations
}

fn initial_inference_status(gp: &TParam) -> InferenceStatus {
    let variance = pre_to_post_variance(gp.variance());
    let (specified_variance, has_variance_inferred) = match variance {
        Variance::Bivariant => (None, false),
        _ => (Some(variance), true),
    };
    InferenceStatus {
        inferred_variance: variance,
        has_variance_inferred,
        specified_variance,
    }
}

fn initial_inference_map(tparams: &[TParam]) -> InferenceMap {
    tparams
        .iter()
        .map(|p| (p.name().clone(), initial_inference_status(p)))
        .collect::<InferenceMap>()
}

fn pre_to_post_variance(pre_variance: PreInferenceVariance) -> Variance {
    match pre_variance {
        PreInferenceVariance::Covariant => Variance::Covariant,
        PreInferenceVariance::Contravariant => Variance::Contravariant,
        PreInferenceVariance::Invariant => Variance::Invariant,
        PreInferenceVariance::Undefined => Variance::Bivariant,
    }
}

fn initialize_environment_impl<'a>(
    class: &'a Class,
    heap: &TypeHeap,
    environment: &mut VarianceEnv,
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
    get_tparams: &impl Fn(&Class) -> Arc<TParams>,
) -> InferenceMap {
    if let Some(params) = environment.get(class) {
        return params.clone();
    }

    let params = initial_inference_map(get_tparams(class).as_vec());

    environment.insert(class.dupe(), params.clone());
    let mut on_var = |_name: &Name, _variance: Variance, _inj: bool, _: PreInferenceVariance| {};

    // get the variance results of a given class c
    let mut on_edge = |c: &Class| {
        initialize_environment_impl(
            c,
            heap,
            environment,
            get_class_bases,
            get_fields,
            get_tparams,
        )
    };

    on_class(
        class,
        heap,
        &mut on_edge,
        &mut on_var,
        get_class_bases,
        get_fields,
    );

    params
}

fn initialize_environment<'a>(
    class: &'a Class,
    heap: &TypeHeap,
    environment: &mut VarianceEnv,
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
    get_tparams: &impl Fn(&Class) -> Arc<TParams>,
) {
    let mut on_var = |_name: &Name, _variance: Variance, _inj: bool, _: PreInferenceVariance| {};
    let mut on_edge = |c: &Class| {
        initialize_environment_impl(
            c,
            heap,
            environment,
            get_class_bases,
            get_fields,
            get_tparams,
        )
    };
    on_class(
        class,
        heap,
        &mut on_edge,
        &mut on_var,
        get_class_bases,
        get_fields,
    );
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn compute_variance_env(&self, class: &Class) -> VarianceEnv {
        fn fixpoint<'a, Ans: LookupAnswer>(
            solver: &AnswersSolver<'a, Ans>,
            mut env: VarianceEnv,
        ) -> VarianceEnv {
            let mut changed = true;

            while changed {
                changed = false;
                let mut new_environment: VarianceEnv = SmallMap::new();

                for (my_class, params) in env.iter() {
                    let mut new_params = params.clone();

                    let mut on_var =
                        |name: &Name,
                         variance: Variance,
                         has_inferred: bool,
                         _: PreInferenceVariance| {
                            if let Some(old_status) = new_params.get_mut(name) {
                                let new_inferred_variance =
                                    variance.union(old_status.inferred_variance);
                                // Mark as inferred if:
                                // 1. It was already marked as inferred, OR
                                // 2. The caller says this is an injective (reliable) constraint, OR
                                // 3. The inferred variance is no longer Bivariant (we found a constraint)
                                // Case 3 fixes self-referential types where `has_inferred` is always false
                                // but we still discover variance constraints through the fixpoint iteration.
                                let new_has_variance_inferred = old_status.has_variance_inferred
                                    || has_inferred
                                    || new_inferred_variance != Variance::Bivariant;
                                if new_inferred_variance != old_status.inferred_variance
                                    || new_has_variance_inferred != old_status.has_variance_inferred
                                {
                                    old_status.inferred_variance = new_inferred_variance;
                                    old_status.has_variance_inferred = new_has_variance_inferred;
                                    changed = true;
                                }
                            }
                        };
                    let mut on_edge = |c: &Class| env.get(c).cloned().unwrap_or_default();
                    on_class(
                        my_class,
                        solver.heap,
                        &mut on_edge,
                        &mut on_var,
                        &|c| solver.get_base_types_for_class(c),
                        &|c| solver.get_class_field_map(c),
                    );
                    new_environment.insert(my_class.dupe(), new_params);
                }
                env = new_environment;
            }
            env
        }

        let mut environment = VarianceEnv::new();
        let initial_inference_map_for_class =
            initial_inference_map(self.get_class_tparams(class).as_vec());
        let need_inference = initial_inference_map_for_class
            .iter()
            .any(|(_, status)| status.specified_variance.is_none());
        environment.insert(class.dupe(), initial_inference_map_for_class);
        if !need_inference {
            environment
        } else {
            initialize_environment(
                class,
                self.heap,
                &mut environment,
                &|c| self.get_base_types_for_class(c),
                &|c| self.get_class_field_map(c),
                &|c| self.get_class_tparams(c),
            );
            fixpoint(self, environment)
        }
    }

    /// Compute variance for a class, optionally checking for violations.
    ///
    /// When `check_violations` is true, also checks that type variables with
    /// declared variance are used in compatible positions.
    pub fn compute_variance(&self, class: &Class, check_violations: bool) -> VarianceResult {
        let env = self.compute_variance_env(class);
        let class_variances = env
            .get(class)
            .expect("class name must be present in environment")
            .iter()
            .map(|(name, status)| {
                (
                    name.clone(),
                    if let Some(specified_variance) = status.specified_variance {
                        specified_variance
                    } else if status.has_variance_inferred {
                        status.inferred_variance
                    } else {
                        Variance::Bivariant
                    },
                )
            })
            .collect::<SmallMap<_, _>>();
        let variance_map = VarianceMap(class_variances);

        let violations = if check_violations {
            check_class_violations(
                class,
                &|c| self.get_base_types_for_class(c),
                &|c| self.get_class_field_map(c),
                &|c| self.get_class_tparams(c),
            )
        } else {
            Vec::new()
        };

        VarianceResult {
            variance_map,
            violations,
        }
    }
}
