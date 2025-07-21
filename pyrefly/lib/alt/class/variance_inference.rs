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
use pyrefly_python::dunder;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMetadata;
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

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, Default)]
pub struct VarianceMap(SmallMap<Name, Variance>);

impl VisitMut<Type> for VarianceMap {
    fn recurse_mut(&mut self, _visitor: &mut dyn FnMut(&mut Type)) {
        // No-op: VarianceMap does not contain any Type
    }
}

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

type Injectivity = bool;
type TypeParam = (Name, Variance, Injectivity);
type TParamArray = Vec<TypeParam>;

// A map from class name to tparam environment
// Why is this not Class or ClassObject
type VarianceEnv = SmallMap<Class, TParamArray>;

fn on_class(
    class: &Class,
    on_edge: &mut impl FnMut(&Class) -> TParamArray,
    on_var: &mut impl FnMut(&Name, Variance, Injectivity),
    get_metadata: &impl Fn(&Class) -> Arc<ClassMetadata>,
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
        inj: Injectivity,
        on_edge: &mut impl FnMut(&Class) -> TParamArray,
        on_var: &mut impl FnMut(&Name, Variance, Injectivity),
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
        inj: Injectivity,
        typ: &Type,
        on_edge: &mut impl FnMut(&Class) -> TParamArray,
        on_var: &mut impl FnMut(&Name, Variance, Injectivity),
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

                for (i, param) in params.iter().enumerate() {
                    if let Some(ty) = targs.get(i) {
                        let (_, variance_param, inj_param) = param;
                        on_type(
                            variance.compose(*variance_param),
                            *inj_param,
                            ty,
                            on_edge,
                            on_var,
                        );
                    }
                }
            }
            Type::Quantified(q) => {
                on_var(q.name(), variance, inj);
            }
            Type::Union(t) => {
                for ty in t {
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
                            let ty = param.param_to_type();
                            on_type(variance.inv(), inj, ty, on_edge, on_var);
                        }
                    }
                    Params::Ellipsis => {
                        // Unknown params
                    }
                    Params::ParamSpec(prefix, param_spec) => {
                        for ty in prefix.iter() {
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

    let metadata = get_metadata(class);
    let base_types = metadata.bases_with_metadata();

    for base_type in base_types {
        on_type(
            Variance::Covariant,
            true,
            &base_type.0.clone().to_type(),
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

        if let Some((ty, _, read_only, descriptor_getter, descriptor_setter)) =
            field.for_variance_inference()
        {
            // Case 1: Regular attribute

            // TODO: We need a much better way to distinguish between fields and methods than this
            // currently, class field representation isn't good enough but we need to fix that soon
            if descriptor_getter.is_none() && descriptor_setter.is_none() {
                let variance = if ty.is_function_type()
                    || is_private_field(name)
                    || read_only
                    || field.is_final()
                {
                    Variance::Covariant
                } else {
                    Variance::Invariant
                };
                on_type(variance, true, ty, on_edge, on_var);
            } else {
                // Case 2: Descriptor or property (has getter and/or setter)
                // Not too sure about this yet, will need to investigate further.

                // Getter: covariant on return type
                if let Some(typ) = descriptor_getter {
                    on_type(Variance::Covariant, true, typ, on_edge, on_var);
                }

                // Setter: contravariant on value being written
                if let Some(typ) = descriptor_setter {
                    on_type(Variance::Contravariant, true, typ, on_edge, on_var);
                }
            }
        }
    }
}

fn default_variance_and_inj(gp: &TParam, contains_bivariant: &mut bool) -> (Variance, Injectivity) {
    let variance = pre_to_post_variance(gp.variance, contains_bivariant);
    let inj = match variance {
        Variance::Bivariant => false,
        _ => true,
    };
    (variance, inj)
}

fn from_gp_to_decl(gp: &TParam, contains_bivariant: &mut bool) -> (Name, Variance, Injectivity) {
    let (variance, inj) = default_variance_and_inj(gp, contains_bivariant);
    (gp.name().clone(), variance, inj)
}

fn params_from_gp(tparams: &[TParam], contains_bivariant: &mut bool) -> TParamArray {
    let mut params: Vec<(Name, Variance, Injectivity)> = tparams
        .iter()
        .map(|param| (param.name().clone(), Variance::Bivariant, false))
        .collect();

    for (i, param) in tparams.iter().enumerate() {
        let (name, variance, inj) = from_gp_to_decl(param, contains_bivariant);
        params[i] = (name, variance, inj);
    }
    params
}

fn convert_gp_to_map(tparams: &TParams, contains_bivariant: &mut bool) -> SmallMap<Name, Variance> {
    let mut lookup = SmallMap::new();

    for param in tparams.iter() {
        lookup.insert(
            param.name().clone(),
            pre_to_post_variance(param.variance, contains_bivariant),
        );
    }

    lookup
}

fn pre_to_post_variance(
    pre_variance: PreInferenceVariance,
    contains_bivariant: &mut bool,
) -> Variance {
    match pre_variance {
        PreInferenceVariance::PCovariant => Variance::Covariant,
        PreInferenceVariance::PContravariant => Variance::Contravariant,
        PreInferenceVariance::PInvariant => Variance::Invariant,
        PreInferenceVariance::PUndefined => {
            *contains_bivariant = true;
            Variance::Bivariant
        }
    }
}

fn loop_fn<'a>(
    class: &'a Class,
    environment: &mut VarianceEnv,
    contains_bivariant: &mut bool,
    get_metadata: &impl Fn(&Class) -> Arc<ClassMetadata>,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, Arc<ClassField>>,
    get_tparams: &impl Fn(&Class) -> Arc<TParams>,
) -> TParamArray {
    if let Some(params) = environment.get(class) {
        return params.clone();
    }

    let params: Vec<(Name, Variance, bool)> =
        params_from_gp(get_tparams(class).as_vec(), contains_bivariant);

    environment.insert(class.dupe(), params.clone());
    let mut on_var = |_name: &Name, _variance: Variance, _inj: Injectivity| {};

    // get the variance results of a given class c
    let mut on_edge = |c: &Class| {
        loop_fn(
            c,
            environment,
            contains_bivariant,
            get_metadata,
            get_fields,
            get_tparams,
        )
    };

    on_class(class, &mut on_edge, &mut on_var, get_metadata, get_fields);

    params
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn variance_map(&self, class: &Class) -> Arc<VarianceMap> {
        let mut contains_bivariant: bool = false;

        let post_inference_initial =
            convert_gp_to_map(&self.get_class_tparams(class), &mut contains_bivariant);

        fn to_map(
            params: &TParamArray,
            post_inference_initial: &SmallMap<Name, Variance>,
        ) -> SmallMap<Name, Variance> {
            let mut map = SmallMap::new();

            for (name, variance, inj) in params.iter() {
                let inferred_variance = match post_inference_initial.get(name) {
                    Some(&Variance::Bivariant) => match (*variance, *inj) {
                        (_, false) => Variance::Bivariant,
                        (Variance::Bivariant, _) => Variance::Bivariant,
                        (Variance::Covariant, _) => Variance::Covariant,
                        (Variance::Contravariant, _) => Variance::Contravariant,
                        (Variance::Invariant, _) => Variance::Invariant,
                    },
                    Some(&res) => res,
                    None => {
                        panic!("Impossible. Class name {name} must be present in variance map",)
                    }
                };
                map.insert(name.clone(), inferred_variance);
            }
            map
        }

        fn fixpoint<'a, Ans: LookupAnswer>(
            solver: &AnswersSolver<'a, Ans>,
            env: &VarianceEnv,
        ) -> VarianceEnv {
            let mut environment_prime: VarianceEnv = SmallMap::new();
            let mut changed = false;

            for (my_class, params) in env.iter() {
                let mut params_prime = params.clone();

                let mut on_var = |name: &Name, variance: Variance, inj: Injectivity| {
                    for (n, variance_prime, inj_prime) in params_prime.iter_mut() {
                        if n == name {
                            *variance_prime = variance.union(*variance_prime);
                            *inj_prime = *inj_prime || inj;
                        }
                    }
                };

                let mut on_edge = |c: &Class| env.get(c).cloned().unwrap_or_else(Vec::new);

                on_class(
                    my_class,
                    &mut on_edge,
                    &mut on_var,
                    &|c| solver.get_metadata_for_class(c),
                    &|c| solver.get_class_field_map(c),
                );

                if params != &params_prime {
                    changed = true;
                }

                environment_prime.insert(my_class.dupe(), params_prime);
            }

            if changed {
                fixpoint(solver, &environment_prime)
            } else {
                environment_prime
            }
        }

        if !contains_bivariant {
            Arc::new(VarianceMap(post_inference_initial))
        } else {
            let mut environment = VarianceEnv::new();

            loop_fn(
                class,
                &mut environment,
                &mut contains_bivariant,
                &|c| self.get_metadata_for_class(c),
                &|c| self.get_class_field_map(c),
                &|c| self.get_class_tparams(c),
            );

            let environment = fixpoint(self, &environment);

            let params = environment
                .get(class)
                .expect("class name must be present in environment");

            let class_variances = to_map(params, &post_inference_initial);
            Arc::new(VarianceMap(class_variances))
        }
    }
}
