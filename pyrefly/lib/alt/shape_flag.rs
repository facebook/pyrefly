/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::dunder;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::function::FuncMetadata;
use pyrefly_types::function::Function;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::type_var::FlagDomain;
use pyrefly_types::type_var::FlagMember;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::TParams;
use pyrefly_types::types::TParamsSource;
use pyrefly_types::types::Type;
use pyrefly_types::types::Var;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::binding::FunctionDefData;
use crate::binding::shape_type::TypeParameterBound;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

#[derive(Clone, PartialEq, Eq)]
enum ClassFlagSourceKind {
    PosOnly(usize),
    Pos(usize, Name),
    KwOnly(Name),
}

#[derive(Clone, PartialEq, Eq)]
struct ClassFlagSource {
    kind: ClassFlagSourceKind,
    required: Required,
}

pub(crate) fn shape_flag_vars(tparams: &TParams, vars: &[Var]) -> Option<Arc<SmallSet<Var>>> {
    assert_eq!(
        tparams.len(),
        vars.len(),
        "fresh callable variables must align with type parameters"
    );
    let vars = tparams
        .iter()
        .zip(vars)
        .filter_map(|(tparam, var)| {
            matches!(tparam.restriction(), Restriction::Flag(_)).then_some(*var)
        })
        .collect::<SmallSet<_>>();
    (!vars.is_empty()).then(|| Arc::new(vars))
}

pub(crate) fn extend_shape_flag_vars_from_targs(
    vars: &mut Option<Arc<SmallSet<Var>>>,
    targs: &TArgs,
) {
    let class_vars = targs.iter_paired().filter_map(|(tparam, ty)| {
        if matches!(tparam.restriction(), Restriction::Flag(_))
            && let Type::Var(var) = ty
        {
            Some(*var)
        } else {
            None
        }
    });
    for var in class_vars {
        Arc::make_mut(vars.get_or_insert_with(|| Arc::new(SmallSet::new()))).insert(var);
    }
}

impl<Ans: LookupAnswer> AnswersSolver<'_, '_, Ans> {
    /// Record constructor parameters whose source annotations directly name a type parameter of
    /// the defining class. A subclass may bind an ordinary base-class parameter to a `Flag`, and
    /// this provenance is lost when aliases are resolved.
    pub(crate) fn record_shape_flag_constructor_sources(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        defining_cls: Option<&Class>,
        metadata: &mut FuncMetadata,
    ) {
        if stmt.name.id != dunder::INIT && stmt.name.id != dunder::NEW {
            return;
        }
        let Some(cls) = defining_cls else {
            return;
        };
        let class_tparams = self.get_class_tparams(cls);
        let sources = params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| {
                let name = param.name()?;
                let parameter = stmt.parameters.find(name.as_str())?;
                let Some(Expr::Name(name)) = parameter.annotation() else {
                    return None;
                };
                let Type::Quantified(quantified) = param.as_type() else {
                    return None;
                };
                (matches!(
                    param,
                    Param::PosOnly(..) | Param::Pos(..) | Param::KwOnly(..)
                ) && quantified.name() == &name.id
                    && class_tparams
                        .iter()
                        .any(|tparams| tparams.iter().any(|tparam| tparam == &**quantified)))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if !sources.is_empty() {
            metadata.flags.shape_flag_constructor_sources = Some(Box::new(sources));
        }
    }

    pub(crate) fn reject_legacy_shape_flag_bound(
        &self,
        bound: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> bool {
        if matches!(bound, Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag")) {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                "`shape_extensions.Flag` is supported only as a direct PEP 695 type parameter bound"
                    .to_owned(),
            );
            true
        } else {
            false
        }
    }

    pub(crate) fn resolve_shape_type_parameter_bound(
        &self,
        bound: &TypeParameterBound,
        errors: &ErrorCollector,
    ) -> Restriction {
        match bound {
            TypeParameterBound::ShapeFlag {
                domain: Some(domain),
                ..
            } => {
                let domain_ty =
                    self.expr_untype(domain, TypeFormContext::TypeVarConstraint, errors);
                if domain_ty.is_error() {
                    return Restriction::Unrestricted;
                }
                match FlagDomain::from_type(&domain_ty) {
                    Some(flag_domain) => Restriction::Flag(flag_domain),
                    None => {
                        self.error(
                            errors,
                            domain.range(),
                            ErrorKind::InvalidTypeVar,
                            format!(
                                "`Flag` domain must resolve to a nonempty union of `int`, `bool`, `str`, `None`, and integer tuples of one fixed arity or `tuple[int, ...]`, got `{domain_ty}`"
                            ),
                        );
                        Restriction::Unrestricted
                    }
                }
            }
            TypeParameterBound::ShapeFlag {
                domain: None,
                range,
            } => {
                self.error(
                    errors,
                    *range,
                    ErrorKind::InvalidTypeVar,
                    "`shape_extensions.Flag` requires one domain argument: `int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these"
                        .to_owned(),
                );
                Restriction::Unrestricted
            }
            TypeParameterBound::Ordinary(bound) => {
                let bound_ty = self.expr_untype(bound, TypeFormContext::TypeVarConstraint, errors);
                if matches!(&bound_ty, Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag"))
                {
                    // TODO: Distinguish quoted canonical bounds from aliases so quoted syntax gets
                    // its intended behavior or a quoted-form diagnostic rather than an alias error.
                    self.error(
                        errors,
                        bound.range(),
                        ErrorKind::InvalidTypeVar,
                        "`shape_extensions.Flag` must be used directly rather than through a type alias"
                            .to_owned(),
                    );
                    Restriction::Unrestricted
                } else {
                    Restriction::Bound(bound_ty)
                }
            }
        }
    }

    pub(crate) fn validate_shape_flag_type_parameter_scope(
        &self,
        tparams: &[Quantified],
        source: &TParamsSource,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if matches!(source, TParamsSource::TypeAlias)
            && tparams
                .iter()
                .any(|tparam| matches!(tparam.restriction(), Restriction::Flag(_)))
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                "`Flag` type parameters are not supported on type aliases".to_owned(),
            );
        }
    }

    /// Check that each class `Flag` is bound by exactly one direct parameter in every constructor
    /// signature that mentions it.
    pub(crate) fn check_shape_flag_constructor_sources(
        &self,
        cls: &Class,
        errors: &ErrorCollector,
    ) {
        let tparams = self.get_class_tparams(cls);
        let flag_tparams = tparams
            .iter()
            .flat_map(|tparams| tparams.iter())
            .filter_map(|tparam| match tparam.restriction() {
                Restriction::Flag(domain) => Some((tparam, domain)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if flag_tparams.is_empty() {
            return;
        }

        let class_type = self.as_class_type_unchecked(cls);
        let dunder_new = self.get_dunder_new(&class_type, false);
        let dunder_init = self.get_dunder_init(&class_type, dunder_new.is_none());
        let phases = [dunder_new, dunder_init];

        for (tparam, domain) in flag_tparams {
            let source_ty = self.heap.mk_quantified(tparam.clone());
            let mut phase_sources: Vec<Vec<Vec<ClassFlagSource>>> = Vec::new();
            for phase in phases.iter().flatten() {
                let mut signatures = Vec::new();
                let mut visit_function = |function: &Function| {
                    let (Params::List(params) | Params::Partial(params)) =
                        &function.signature.params
                    else {
                        signatures.push(Vec::new());
                        return;
                    };
                    signatures.push(
                        function
                            .metadata
                            .flags
                            .shape_flag_constructor_sources
                            .as_deref()
                            .into_iter()
                            .flatten()
                            .filter_map(|index| {
                                // Constructor signatures still contain their implicit `self` or
                                // `cls` parameter here, while calls bind it separately. Remove
                                // that common leading slot before comparing caller positions.
                                let caller_positional_index = params
                                    .items()
                                    .iter()
                                    .take(*index)
                                    .filter(|param| {
                                        matches!(param, Param::PosOnly(..) | Param::Pos(..))
                                    })
                                    .count()
                                    .checked_sub(1)?;
                                // Decorators and partial application may change the effective
                                // signature after source indices were recorded. Treat a stale
                                // index as a missing source so the class is rejected safely.
                                let param = params.items().get(*index)?;
                                match param {
                                    Param::PosOnly(_, ty, required) if ty == &source_ty => {
                                        Some(ClassFlagSource {
                                            kind: ClassFlagSourceKind::PosOnly(
                                                caller_positional_index,
                                            ),
                                            required: required.clone(),
                                        })
                                    }
                                    Param::Pos(name, ty, required) if ty == &source_ty => {
                                        Some(ClassFlagSource {
                                            kind: ClassFlagSourceKind::Pos(
                                                caller_positional_index,
                                                name.clone(),
                                            ),
                                            required: required.clone(),
                                        })
                                    }
                                    Param::KwOnly(name, ty, required) if ty == &source_ty => {
                                        Some(ClassFlagSource {
                                            kind: ClassFlagSourceKind::KwOnly(name.clone()),
                                            required: required.clone(),
                                        })
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                    );
                };
                match phase {
                    Type::Function(function) => visit_function(function),
                    Type::Forall(forall) => {
                        if let Forallable::Function(function) = &forall.body {
                            visit_function(function);
                        }
                    }
                    Type::BoundMethod(method) => match &method.func {
                        BoundMethodType::Function(function) => visit_function(function),
                        BoundMethodType::Forall(forall) => visit_function(&forall.body),
                        BoundMethodType::Overload(overload) => {
                            for signature in overload.signatures.iter() {
                                match signature {
                                    OverloadType::Function(function) => visit_function(function),
                                    OverloadType::Forall(forall) => visit_function(&forall.body),
                                }
                            }
                        }
                    },
                    Type::Overload(overload) => {
                        for signature in overload.signatures.iter() {
                            match signature {
                                OverloadType::Function(function) => visit_function(function),
                                OverloadType::Forall(forall) => visit_function(&forall.body),
                            }
                        }
                    }
                    _ => {}
                }
                phase_sources.push(signatures);
            }

            let mut mentioned = false;
            let mut expected_source: Option<&ClassFlagSource> = None;
            for signatures in &phase_sources {
                if signatures.iter().all(|sources| sources.is_empty()) {
                    continue;
                }
                mentioned = true;
                for sources in signatures {
                    if sources.len() != 1 {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorKind::InvalidTypeVar,
                            format!(
                                "`Flag` type parameter `{}` must directly annotate exactly one constructor parameter, found {}",
                                tparam.name(),
                                sources.len(),
                            ),
                        );
                    } else {
                        let source = &sources[0];
                        if let Required::Optional(Some(default)) = &source.required
                            && !domain.accepts_literal(&default.ty)
                        {
                            self.error(
                                errors,
                                cls.range(),
                                ErrorKind::InvalidTypeVar,
                                format!(
                                    "Default for parameter binding `Flag[{domain}]` type parameter `{}` must be a `{domain}` literal, got `{}`",
                                    tparam.name(),
                                    default.ty,
                                ),
                            );
                        }
                        if let Some(expected) = expected_source {
                            if source.kind != expected.kind {
                                self.error(
                                    errors,
                                    cls.range(),
                                    ErrorKind::InvalidTypeVar,
                                    format!(
                                        "`Flag` type parameter `{}` must bind from the same constructor argument in every signature",
                                        tparam.name(),
                                    ),
                                );
                            } else if source.required != expected.required {
                                self.error(
                                    errors,
                                    cls.range(),
                                    ErrorKind::InvalidTypeVar,
                                    format!(
                                        "Parameter binding `Flag` type parameter `{}` must have the same default in every constructor signature",
                                        tparam.name(),
                                    ),
                                );
                            }
                        } else {
                            expected_source = Some(source);
                        }
                    }
                }
            }
            if !mentioned {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Flag` type parameter `{}` must directly annotate exactly one constructor parameter, found 0",
                        tparam.name(),
                    ),
                );
            }
        }
    }

    pub(crate) fn validate_shape_flag_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Restriction::Flag(domain) = restriction else {
            return None;
        };
        // Tuple type expressions infer as `type[...]`; defaults store the corresponding value.
        let default = match default {
            Type::Type(inner) => inner.as_ref(),
            _ => default,
        };
        if domain.accepts_literal(default) {
            Some(default.clone())
        } else {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!(
                    "Default for `Flag[{domain}]` type parameter `{name}` must be a `{domain}` literal, got `{default}`"
                ),
            );
            Some(self.heap.mk_any_error())
        }
    }

    pub(crate) fn validate_shape_flag_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        for (tparam, domain) in tparams
            .iter()
            .filter_map(|tparam| match tparam.restriction() {
                Restriction::Flag(domain) => Some((tparam, domain)),
                _ => None,
            })
        {
            let sources = stmt
                .parameters
                .iter()
                .zip(params)
                .enumerate()
                .filter_map(|(index, (parameter, param))| {
                    let single_value_parameter = matches!(
                        param,
                        Param::PosOnly(..) | Param::Pos(..) | Param::KwOnly(..)
                    );
                    // Scalar sources require direct syntax. Unpacked sources use the resolved type
                    // so equivalent `Unpack` spellings are treated identically.
                    match (parameter.annotation(), param) {
                        (Some(Expr::Name(name)), _)
                            if single_value_parameter && name.id == *tparam.name() =>
                        {
                            Some((index, name.range(), false))
                        }
                        (
                            Some(annotation),
                            Param::Varargs(_, Type::Unpack(inner)),
                        ) if matches!(&**inner, Type::Quantified(q) if q.as_ref() == tparam) =>
                        {
                            Some((index, annotation.range(), true))
                        }
                        _ => None,
                    }
                })
                .collect::<Vec<_>>();
            if sources.len() != 1 {
                self.error(
                    errors,
                    stmt.name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Flag` type parameter `{}` must directly annotate exactly one function parameter, found {}",
                        tparam.name(),
                        sources.len(),
                    ),
                );
                continue;
            }
            let (source_index, source_range, unpacked) = sources[0];
            if unpacked && !domain.contains(FlagMember::IntTuple) {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "Unpacked parameter binding `Flag[{domain}]` type parameter `{}` requires a domain containing an integer tuple",
                        tparam.name(),
                    ),
                );
                continue;
            }
            let runtime_default = match &params[source_index] {
                Param::PosOnly(_, _, Required::Optional(default))
                | Param::Pos(_, _, Required::Optional(default))
                | Param::KwOnly(_, _, Required::Optional(default)) => default.as_ref(),
                _ => None,
            };
            if let Some(default) = runtime_default
                && !domain.accepts_literal(&default.ty)
            {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "Default for parameter binding `Flag[{domain}]` type parameter `{}` must be a `{domain}` literal, got `{}`",
                        tparam.name(),
                        default.ty,
                    ),
                );
            }
        }
    }

    pub(crate) fn is_shape_flag_parameter_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Quantified(q) if matches!(q.restriction(), Restriction::Flag(_)))
    }
}
