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
use pyrefly_types::types::Union;
use ruff_python_ast::name::Name;
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

fn on_class(
    class: &Class,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, Variance, bool),
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
) {
    fn is_private_field(name: &Name) -> bool {
        let starts_with_underscore = name.starts_with('_');
        let ends_with_double_underscore = name.ends_with("__");

        starts_with_underscore && !ends_with_double_underscore
    }

    fn handle_tuple_type(
        tuple: &Tuple,
        variance: Variance,
        inj: bool,
        on_edge: &mut impl FnMut(&Class) -> InferenceMap,
        on_var: &mut impl FnMut(&Name, Variance, bool),
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
        on_var: &mut impl FnMut(&Name, Variance, bool),
    ) {
        match typ {
            Type::Type(t) => {
                on_type(variance, inj, t, on_edge, on_var);
            }

            Type::Function(t) => {
                on_type(
                    variance,
                    inj,
                    &Type::Callable(Box::new(t.signature.clone())),
                    on_edge,
                    on_var,
                );
            }

            Type::ClassType(class) if !class.tparams().is_empty() => {
                let params = on_edge(class.class_object());

                let targs = class.targs().as_slice();

                for (status, ty) in params.values().zip(targs) {
                    on_type(
                        variance.compose(status.inferred_variance),
                        status.has_variance_inferred,
                        ty,
                        on_edge,
                        on_var,
                    );
                }
            }
            Type::Quantified(q) => {
                on_var(q.name(), variance, inj);
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

            _ => {}
        }
    }

    for base_type in get_class_bases(class).iter() {
        on_type(
            Variance::Covariant,
            true,
            &base_type.clone().to_type(),
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

fn initial_inference_status(gp: &TParam) -> InferenceStatus {
    let variance = pre_to_post_variance(gp.variance);
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
    let mut on_var = |_name: &Name, _variance: Variance, _inj: bool| {};

    // get the variance results of a given class c
    let mut on_edge = |c: &Class| {
        initialize_environment_impl(c, environment, get_class_bases, get_fields, get_tparams)
    };

    on_class(
        class,
        &mut on_edge,
        &mut on_var,
        get_class_bases,
        get_fields,
    );

    params
}

fn initialize_environment<'a>(
    class: &'a Class,
    environment: &mut VarianceEnv,
    get_class_bases: &impl Fn(&Class) -> Arc<ClassBases>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
    get_tparams: &impl Fn(&Class) -> Arc<TParams>,
) {
    let mut on_var = |_name: &Name, _variance: Variance, _inj: bool| {};
    let mut on_edge = |c: &Class| {
        initialize_environment_impl(c, environment, get_class_bases, get_fields, get_tparams)
    };
    on_class(
        class,
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

                    let mut on_var = |name: &Name, variance: Variance, has_inferred: bool| {
                        if let Some(old_status) = new_params.get_mut(name) {
                            let new_inferred_variance =
                                variance.union(old_status.inferred_variance);
                            let new_has_variance_inferred =
                                old_status.has_variance_inferred || has_inferred;
                            if new_inferred_variance != old_status.inferred_variance
                                || new_has_variance_inferred != old_status.has_variance_inferred
                            {
                                old_status.inferred_variance = new_inferred_variance;
                                old_status.has_variance_inferred = new_has_variance_inferred;
                                changed = true;
                            }
                        }
                    };
                    let mut on_edge = |c: &Class| env.get(c).cloned().unwrap_or_else(SmallMap::new);
                    on_class(
                        my_class,
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
                &mut environment,
                &|c| self.get_base_types_for_class(c),
                &|c| self.get_class_field_map(c),
                &|c| self.get_class_tparams(c),
            );
            fixpoint(self, environment)
        }
    }

    pub fn variance_map(&self, class: &Class) -> Arc<VarianceMap> {
        let class_variances = self
            .compute_variance_env(class)
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
                        // Rare case where the variance does not appear to be constrained in any way
                        Variance::Bivariant
                    },
                )
            })
            .collect::<SmallMap<_, _>>();
        Arc::new(VarianceMap(class_variances))
    }
}
