use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::variance_inference::variance_visitor::Injectivity;
use crate::alt::class::variance_inference::variance_visitor::TParamArray;
use crate::alt::class::variance_inference::variance_visitor::VarianceEnv;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::types::class::Class;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Variance;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::Type;

// TODO zeina: This algorithm still needs to handle class properties.
// To do so, we have to thread class property information to the algorithm
// After this, we simply call on_type on all class property signatures
// we have to special case some properties like __init__ and private properties

#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
pub struct VarianceMap(pub SmallMap<String, Variance>);

impl VisitMut<Type> for VarianceMap {
    fn visit_mut(&mut self, _visitor: &mut dyn FnMut(&mut Type)) {
        // No-op: VarianceMap does not contain any Type
    }

    fn recurse_mut(&mut self, _visitor: &mut dyn FnMut(&mut Type)) {
        // No-op: VarianceMap does not contain any Type
    }
}

impl Display for VarianceMap {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{{")?;
        for (key, value) in self.0.iter() {
            write!(f, "{}: {}, ", key, value)?;
        }
        write!(f, "}}")
    }
}

pub mod variance_visitor {
    use std::sync::Arc;

    use starlark_map::small_map::SmallMap;

    use crate::alt::class::class_field::ClassField;
    use crate::alt::types::class_metadata::ClassMetadata;
    use crate::types::callable::Params;
    use crate::types::class::Class;
    use crate::types::tuple::Tuple;
    use crate::types::type_var::Variance;
    use crate::types::types::Type;
    pub type Injectivity = bool;
    pub type TypeParam = (String, Variance, Injectivity);
    pub type TParamArray = Vec<TypeParam>;

    // A map from class name to tparam environment
    pub type VarianceEnv = SmallMap<String, TParamArray>;

    pub fn on_class(
        class: &Class,
        on_edge: &mut impl FnMut(&mut SmallMap<String, Arc<Class>>, &Class) -> TParamArray,
        on_var: &mut impl FnMut(&str, Variance, Injectivity),
        get_metadata: &impl Fn(&Class) -> Arc<ClassMetadata>,
        get_fields: &impl Fn(&Class) -> SmallMap<String, Arc<ClassField>>,
        class_lookup_map: &mut SmallMap<String, Arc<Class>>,
    ) {
        fn is_private_field(name: &str) -> bool {
            let starts_with_underscore = name.starts_with('_');
            let ends_with_double_underscore = name.ends_with("__");

            starts_with_underscore && !ends_with_double_underscore
        }

        fn handle_tuple_type(
            tuple: &Tuple,
            variance: Variance,
            inj: Injectivity,
            on_edge: &mut impl FnMut(&mut SmallMap<String, Arc<Class>>, &Class) -> TParamArray,
            on_var: &mut impl FnMut(&str, Variance, Injectivity),
            class_lookup_map: &mut SmallMap<String, Arc<Class>>,
        ) {
            match tuple {
                Tuple::Concrete(concrete_types) => {
                    for ty in concrete_types {
                        on_type(variance, inj, ty, on_edge, on_var, class_lookup_map);
                    }
                }
                Tuple::Unbounded(unbounded_ty) => {
                    on_type(
                        variance,
                        inj,
                        unbounded_ty,
                        on_edge,
                        on_var,
                        class_lookup_map,
                    );
                }
                Tuple::Unpacked(boxed_parts) => {
                    let (before, middle, after) = &**boxed_parts;
                    for ty in before {
                        on_type(variance, inj, ty, on_edge, on_var, class_lookup_map);
                    }
                    on_type(variance, inj, middle, on_edge, on_var, class_lookup_map);
                    for ty in after {
                        on_type(variance, inj, ty, on_edge, on_var, class_lookup_map);
                    }
                }
            }
        }

        fn on_type(
            variance: Variance,
            inj: Injectivity,
            typ: &Type,
            on_edge: &mut impl FnMut(&mut SmallMap<String, Arc<Class>>, &Class) -> TParamArray,
            on_var: &mut impl FnMut(&str, Variance, Injectivity),
            class_lookup_map: &mut SmallMap<String, Arc<Class>>,
        ) {
            match typ {
                Type::Type(t) => {
                    on_type(variance, inj, t, on_edge, on_var, class_lookup_map);
                }

                Type::Function(t) => {
                    on_type(
                        variance,
                        inj,
                        &Type::Callable(Box::new(t.signature.clone())),
                        on_edge,
                        on_var,
                        class_lookup_map,
                    );
                }

                Type::ClassType(class) => {
                    class_lookup_map.insert(
                        class.name().as_str().to_owned(),
                        Arc::new(class.class_object().clone()),
                    );

                    let params = on_edge(class_lookup_map, class.class_object());

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
                                class_lookup_map,
                            );
                        }
                    }
                }
                Type::Quantified(q) => {
                    on_var(q.name().as_str(), variance, inj);
                }
                Type::Union(t) => {
                    for ty in t {
                        on_type(variance, inj, ty, on_edge, on_var, class_lookup_map);
                    }
                }
                Type::Overload(t) => {
                    let sigs = &t.signatures;
                    for sig in sigs {
                        on_type(
                            variance,
                            inj,
                            &sig.as_type(),
                            on_edge,
                            on_var,
                            class_lookup_map,
                        );
                    }
                }
                Type::Callable(t) => {
                    // Walk return type covariantly
                    on_type(variance, inj, &t.ret, on_edge, on_var, class_lookup_map);

                    // Walk parameters contravariantly
                    match &t.params {
                        Params::List(param_list) => {
                            for param in param_list.items().iter() {
                                let ty = param.param_to_type();
                                on_type(variance.inv(), inj, ty, on_edge, on_var, class_lookup_map);
                            }
                        }
                        Params::Ellipsis => {
                            // Unknown params
                        }
                        Params::ParamSpec(prefix, param_spec) => {
                            for ty in prefix.iter() {
                                on_type(variance.inv(), inj, ty, on_edge, on_var, class_lookup_map);
                            }
                            on_type(
                                variance.inv(),
                                inj,
                                param_spec,
                                on_edge,
                                on_var,
                                class_lookup_map,
                            );
                        }
                    }
                }
                Type::Tuple(t) => {
                    handle_tuple_type(t, variance, inj, on_edge, on_var, class_lookup_map);
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
                class_lookup_map,
            );
        }

        let fields = get_fields(class);

        // todo zeina: check if we need to check for things like __init_subclass__
        // in pyre 1, we didn't need to.
        for (name, field) in fields.iter() {
            if name == "__init__" {
                continue;
            }

            if let Some((ty, _, readonly, descriptor_getter, descriptor_setter)) =
                field.for_variance_inference()
            {
                // Case 1: Regular attribute

                // TODO: We need a much better way to distinguish between fields and methods than this
                // currently, class field representation isn't good enough but we need to fix that soon
                if descriptor_getter.is_none() && descriptor_setter.is_none() {
                    // TODO BE: Move this to a helper function and make sure any code using the same logic uses it
                    let is_function = matches!(
                        ty,
                        Type::Function { .. }
                            | Type::Overload { .. }
                            | Type::BoundMethod { .. }
                            | Type::Callable { .. }
                    );
                    let variance =
                        if is_function || is_private_field(name) || readonly || field.is_final() {
                            Variance::Covariant
                        } else {
                            Variance::Invariant
                        };

                    on_type(variance, true, ty, on_edge, on_var, class_lookup_map);
                } else {
                    // Case 2: Descriptor or property (has getter and/or setter)
                    // Not too sure about this yet, will need to investigate further.

                    // Getter: covariant on return type
                    if let Some(typ) = descriptor_getter {
                        on_type(
                            Variance::Covariant,
                            true,
                            typ,
                            on_edge,
                            on_var,
                            class_lookup_map,
                        );
                    }

                    // Setter: contravariant on value being written
                    if let Some(typ) = descriptor_setter {
                        on_type(
                            Variance::Contravariant,
                            true,
                            typ,
                            on_edge,
                            on_var,
                            class_lookup_map,
                        );
                    }
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

fn from_gp_to_decl(gp: &TParam, contains_bivariant: &mut bool) -> (String, Variance, Injectivity) {
    let (variance, inj) = default_variance_and_inj(gp, contains_bivariant);
    (gp.name().as_str().to_owned(), variance, inj)
}

pub fn params_from_gp(tparams: &[TParam], contains_bivariant: &mut bool) -> TParamArray {
    let mut params: Vec<(String, Variance, Injectivity)> = tparams
        .iter()
        .map(|param| (param.name().as_str().to_owned(), Variance::Bivariant, false))
        .collect();

    for (i, param) in tparams.iter().enumerate() {
        let (name, variance, inj) = from_gp_to_decl(param, contains_bivariant);
        params[i] = (name, variance, inj);
    }
    params
}

pub fn convert_gp_to_map(
    tparams: &TParams,
    contains_bivariant: &mut bool,
) -> SmallMap<String, Variance> {
    let mut lookup = SmallMap::new();

    for param in tparams.iter() {
        lookup.insert(
            param.name().as_str().to_owned(),
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
    get_fields: &impl Fn(&Class) -> SmallMap<String, Arc<ClassField>>,
    class_lookup_map: &mut SmallMap<String, Arc<Class>>,
) -> TParamArray {
    let class_name = class.name().as_str().to_owned();

    if let Some(params) = environment.get(&class_name) {
        return params.clone();
    }

    let params: Vec<(String, Variance, bool)> =
        params_from_gp(class.tparams().as_vec(), contains_bivariant);

    environment.insert(class_name.clone(), params.clone());
    let mut on_var = |_name: &str, _variance: Variance, _inj: Injectivity| {};

    // get the variance results of a given class c
    let mut on_edge = |map: &mut SmallMap<String, Arc<Class>>, c: &Class| {
        loop_fn(
            c,
            environment,
            contains_bivariant,
            get_metadata,
            get_fields,
            map,
        )
    };

    variance_visitor::on_class(
        class,
        &mut on_edge,
        &mut on_var,
        get_metadata,
        get_fields,
        class_lookup_map,
    );

    params
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn variance_map(&self, class: &Class) -> Arc<VarianceMap> {
        let mut contains_bivariant: bool = false;
        let mut class_lookup_map: SmallMap<String, Arc<Class>> = SmallMap::new();

        let post_inference_initial = convert_gp_to_map(class.tparams(), &mut contains_bivariant);

        fn to_map(
            params: &TParamArray,
            post_inference_initial: &SmallMap<String, Variance>,
        ) -> SmallMap<String, Variance> {
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
                    None => panic!(
                        "Impossible. Class name {} must be present in variance map",
                        name
                    ),
                };
                map.insert(name.clone(), inferred_variance);
            }
            map
        }

        fn fixpoint<'a, Ans: LookupAnswer>(
            solver: &AnswersSolver<'a, Ans>,
            class: &Class,
            env: &VarianceEnv,
            class_lookup_map: &mut SmallMap<String, Arc<Class>>,
        ) -> VarianceEnv {
            let mut environment_prime: VarianceEnv = SmallMap::new();
            let mut changed = false;

            for (class_name, params) in env.iter() {
                let mut params_prime = params.clone();

                let metadata = solver.get_metadata_for_class(class);
                let ancestor_class = metadata.ancestors(solver.stdlib).find(|ancestor| {
                    let class_obj = ancestor.class_object();
                    class_obj.name() == class_name
                });

                // TODO zeina: If our invariants are right, "continue" should be replace with a panic
                // after we stop visiting monomorphic types
                let my_class = if let Some(ancestor) = ancestor_class {
                    Arc::new(ancestor.class_object().clone())
                } else if class.name() == class_name {
                    Arc::new(class.clone())
                } else if let Some(class_arc) = class_lookup_map.get(class_name) {
                    class_arc.clone()
                } else {
                    continue;
                };

                let mut on_var = |name: &str, variance: Variance, inj: Injectivity| {
                    for (n, variance_prime, inj_prime) in params_prime.iter_mut() {
                        if n == name {
                            *variance_prime = variance.union(*variance_prime);
                            *inj_prime = *inj_prime || inj;
                        }
                    }
                };

                let mut on_edge = |_map: &mut SmallMap<String, Arc<Class>>, c: &Class| {
                    env.get(c.name().as_str()).cloned().unwrap_or_else(Vec::new)
                };

                variance_visitor::on_class(
                    &my_class,
                    &mut on_edge,
                    &mut on_var,
                    &|c| solver.get_metadata_for_class(c),
                    &|c| solver.get_class_field_map(c),
                    class_lookup_map,
                );

                if params != &params_prime {
                    changed = true;
                }

                environment_prime.insert(class_name.clone(), params_prime);
            }

            if changed {
                fixpoint(solver, class, &environment_prime, class_lookup_map)
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
                &mut class_lookup_map,
            );

            let environment = fixpoint(self, class, &environment, &mut class_lookup_map);

            let class_name = class.name().as_str();

            let params = environment
                .get(class_name)
                .expect("class name must be present in environment");

            let class_variances = to_map(params, &post_inference_initial);
            Arc::new(VarianceMap(class_variances))
        }
    }
}
