/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared helpers for experimental shape-extension types and restrictions.

use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::dimension::Int;
use pyrefly_types::function::Function;
use pyrefly_types::named_ints::NamedInt;
use pyrefly_types::named_ints::NamedInts;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_var::FlagDomain;
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

/// Returns whether `ty` is the normalized upper bound for an `IntTuple`-bounded `TypeVar`.
///
/// Other tuple bounds are ordinary type bounds and must not enable shape-specific parsing.
pub(crate) fn is_int_tuple_bound(ty: &Type, int_type: &Type) -> bool {
    match ty {
        Type::IntTuple(_) => true,
        Type::Tuple(Tuple::Unbounded(inner)) => inner.as_ref() == int_type,
        _ => false,
    }
}

pub(crate) fn shape_extension_vars(tparams: &TParams, vars: &[Var]) -> Option<Arc<SmallSet<Var>>> {
    assert_eq!(
        tparams.len(),
        vars.len(),
        "fresh callable variables must align with type parameters"
    );
    let vars = tparams
        .iter()
        .zip(vars)
        .filter_map(|(tparam, var)| {
            tparam
                .restriction()
                .uses_direct_value_source()
                .then_some(*var)
        })
        .collect::<SmallSet<_>>();
    (!vars.is_empty()).then(|| Arc::new(vars))
}

pub(crate) fn extend_shape_extension_vars_from_targs(
    vars: &mut Option<Arc<SmallSet<Var>>>,
    targs: &TArgs,
) {
    let class_vars = targs.iter_paired().filter_map(|(tparam, ty)| {
        if tparam.restriction().uses_direct_value_source()
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

pub(crate) fn capture_named_ints_source(ty: &Type) -> Option<&Type> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    let [source] = cls.targs().as_slice() else {
        return None;
    };
    cls.has_qname("shape_extensions", "CaptureNamedInts")
        .then_some(source)
}

pub(crate) struct NamedIntsCapture<'a> {
    source: &'a Type,
    entries: Vec<NamedInt>,
    open: bool,
}

fn visit_functions(ty: &Type, visit: &mut impl FnMut(&Function)) {
    match ty {
        Type::Function(function) => visit(function),
        Type::Forall(forall) => {
            if let Forallable::Function(function) = &forall.body {
                visit(function);
            }
        }
        Type::BoundMethod(method) => match &method.func {
            BoundMethodType::Function(function) => visit(function),
            BoundMethodType::Forall(forall) => visit(&forall.body),
            BoundMethodType::Overload(overload) => {
                overload
                    .signatures
                    .iter()
                    .for_each(|signature| match signature {
                        OverloadType::Function(function) => visit(function),
                        OverloadType::Forall(forall) => visit(&forall.body),
                    })
            }
        },
        Type::Overload(overload) => {
            overload
                .signatures
                .iter()
                .for_each(|signature| match signature {
                    OverloadType::Function(function) => visit(function),
                    OverloadType::Forall(forall) => visit(&forall.body),
                })
        }
        _ => {}
    }
}

impl<'a> NamedIntsCapture<'a> {
    pub(crate) fn new(source: &'a Type) -> Self {
        Self {
            source,
            entries: Vec::new(),
            open: false,
        }
    }

    pub(crate) fn insert(&mut self, name: Name, value: Int, required: bool) {
        self.entries.push(NamedInt {
            name,
            value,
            required,
        });
    }

    pub(crate) fn mark_open(&mut self) {
        self.open = true;
    }

    pub(crate) fn finish(self) -> (&'a Type, Type) {
        (
            self.source,
            Type::NamedInts(Box::new(NamedInts::new(self.entries, self.open))),
        )
    }
}

pub(crate) fn direct_function_parameter_sources(
    stmt: &FunctionDefData,
    params: &[Param],
    tparam: &Quantified,
) -> Vec<(usize, TextRange, bool)> {
    stmt.parameters
        .iter()
        .zip(params)
        .enumerate()
        .filter_map(|(index, (parameter, param))| {
            let single_value_parameter =
                matches!(param, Param::PosOnly(..) | Param::Pos(..) | Param::KwOnly(..));
            // Scalar sources require direct syntax. Unpacked sources use the resolved type so
            // equivalent `Unpack` spellings are treated identically.
            match (parameter.annotation(), param) {
                (Some(Expr::Name(name)), _)
                    if single_value_parameter && name.id == *tparam.name() =>
                {
                    Some((index, name.range(), false))
                }
                (Some(annotation), Param::Varargs(_, Type::Unpack(inner)))
                    if matches!(&**inner, Type::Quantified(q) if q.as_ref() == tparam) =>
                {
                    Some((index, annotation.range(), true))
                }
                _ => None,
            }
        })
        .collect()
}

impl<Ans: LookupAnswer> AnswersSolver<'_, '_, Ans> {
    /// Gives `IntListLiteral` and `MapIntTuples` their ordinary function-body types.
    pub(crate) fn shape_extension_parameter_body_type(&self, ty: Type) -> Type {
        self.int_list_literal_parameter_body_type(self.map_int_tuples_parameter_body_type(ty))
    }

    pub(crate) fn check_named_ints_constructor_sources(
        &self,
        cls: &Class,
        errors: &ErrorCollector,
    ) {
        let tparams = self.get_class_tparams(cls);
        let named_ints_tparams = tparams
            .iter()
            .flat_map(|tparams| tparams.iter())
            .filter(|tparam| tparam.restriction().is_named_ints())
            .collect::<Vec<_>>();
        if named_ints_tparams.is_empty() {
            return;
        }
        let class_type = self.as_class_type_unchecked(cls);
        let dunder_new = self.get_dunder_new(&class_type, false);
        let dunder_init = self.get_dunder_init(&class_type, dunder_new.is_none());

        for tparam in named_ints_tparams {
            let source_ty = self.heap.mk_quantified(tparam.clone());
            let mut mentioned = false;
            for phase in [&dunder_new, &dunder_init].into_iter().flatten() {
                let mut counts = Vec::new();
                visit_functions(phase, &mut |function| {
                    let count = match &function.signature.params {
                        Params::List(params) | Params::Partial(params) => params
                            .items()
                            .iter()
                            .filter(|param| {
                                matches!(param, Param::Kwargs(_, ty) if capture_named_ints_source(ty) == Some(&source_ty))
                            })
                            .count(),
                        _ => 0,
                    };
                    counts.push(count);
                });
                if counts.iter().all(|count| *count == 0) {
                    continue;
                }
                mentioned = true;
                for count in counts.into_iter().filter(|count| *count != 1) {
                    self.error(
                        errors,
                        cls.range(),
                        ErrorKind::InvalidTypeVar,
                        format!(
                            "`NamedInts` type parameter `{}` must source exactly one constructor `CaptureNamedInts` parameter, found {count}",
                            tparam.name(),
                        ),
                    );
                }
            }
            if !mentioned {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`NamedInts` type parameter `{}` must source exactly one constructor `CaptureNamedInts` parameter, found 0",
                        tparam.name(),
                    ),
                );
            }
        }
    }

    pub(crate) fn validate_shape_extension_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        self.validate_shape_flag_type_parameter_default(name, default, range, restriction, errors)
            .or_else(|| {
                self.validate_shape_index_type_parameter_default(
                    name,
                    default,
                    range,
                    restriction,
                    errors,
                )
            })
            .or_else(|| {
                restriction.is_named_ints().then(|| {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidTypeVar,
                        format!("`NamedInts` type parameter `{name}` cannot have a default"),
                    );
                    self.heap.mk_any_error()
                })
            })
    }

    pub(crate) fn validate_shape_extension_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        self.validate_shape_flag_function_parameters(stmt, params, tparams, errors);
        self.validate_shape_index_function_parameters(stmt, params, tparams, errors);
        for tparam in tparams
            .iter()
            .filter(|tparam| tparam.restriction().is_named_ints())
        {
            let sources = params
                .iter()
                .filter(|param| {
                    let ty = match param {
                        Param::PosOnly(_, ty, _)
                        | Param::Pos(_, ty, _)
                        | Param::Varargs(_, ty)
                        | Param::KwOnly(_, ty, _)
                        | Param::Kwargs(_, ty) => ty,
                    };
                    matches!(capture_named_ints_source(ty), Some(Type::Quantified(q)) if q.as_ref() == tparam)
                })
                .count();
            if sources != 1 {
                self.error(
                    errors,
                    stmt.name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`NamedInts` type parameter `{}` must source exactly one `CaptureNamedInts` parameter, found {sources}",
                        tparam.name(),
                    ),
                );
            }
        }
        for (parameter, param) in stmt.parameters.iter().zip(params) {
            let ty = match param {
                Param::PosOnly(_, ty, _)
                | Param::Pos(_, ty, _)
                | Param::Varargs(_, ty)
                | Param::KwOnly(_, ty, _)
                | Param::Kwargs(_, ty) => ty,
            };
            let is_capture = matches!(ty, Type::ClassType(cls) if cls.has_qname("shape_extensions", "CaptureNamedInts"));
            if !is_capture {
                continue;
            }
            let range = parameter
                .annotation()
                .map_or_else(|| stmt.name.range(), Ranged::range);
            let Some(source) = capture_named_ints_source(ty) else {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidTypeVar,
                    "`CaptureNamedInts` requires exactly one type argument".to_owned(),
                );
                continue;
            };
            if !matches!(source, Type::Quantified(q) if q.restriction().is_named_ints()) {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidTypeVar,
                    "`CaptureNamedInts` argument must be a `NamedInts` type parameter".to_owned(),
                );
            }
            if !matches!(param, Param::Kwargs(..)) {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidTypeVar,
                    "`CaptureNamedInts` is supported only as a `**kwargs` annotation".to_owned(),
                );
            }
        }
    }

    pub(crate) fn reject_legacy_shape_extension_bound(
        &self,
        bound: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> bool {
        let kind = match bound {
            Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag") => "Flag",
            Type::ClassType(cls) if cls.has_qname("shape_extensions", "Index") => "Index",
            Type::ClassType(cls) if cls.has_qname("shape_extensions", "NamedInts") => "NamedInts",
            _ => return false,
        };
        self.error(
            errors,
            range,
            ErrorKind::InvalidTypeVar,
            format!(
                "`shape_extensions.{kind}` is supported only as a direct PEP 695 type parameter bound"
            ),
        );
        true
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
                    Some(flag_domain) => Restriction::flag(flag_domain),
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
            TypeParameterBound::ShapeIndex => Restriction::index(),
            TypeParameterBound::ShapeNamedInts => Restriction::named_ints(),
            TypeParameterBound::Ordinary(bound) => {
                let bound_ty = self.expr_untype(bound, TypeFormContext::TypeVarConstraint, errors);
                let aliased_kind = match &bound_ty {
                    Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag") => {
                        Some("Flag")
                    }
                    Type::ClassType(cls) if cls.has_qname("shape_extensions", "Index") => {
                        Some("Index")
                    }
                    Type::ClassType(cls) if cls.has_qname("shape_extensions", "NamedInts") => {
                        Some("NamedInts")
                    }
                    _ => None,
                };
                if let Some(kind) = aliased_kind {
                    // TODO: Distinguish quoted canonical bounds from aliases so quoted syntax gets
                    // its intended behavior or a quoted-form diagnostic rather than an alias error.
                    self.error(
                        errors,
                        bound.range(),
                        ErrorKind::InvalidTypeVar,
                        format!(
                            "`shape_extensions.{kind}` must be used directly rather than through a type alias"
                        ),
                    );
                    Restriction::Unrestricted
                } else {
                    Restriction::Bound(bound_ty)
                }
            }
        }
    }

    pub(crate) fn validate_shape_extension_type_parameter_scope(
        &self,
        tparams: &[Quantified],
        source: &TParamsSource,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if matches!(source, TParamsSource::TypeAlias) {
            if tparams.iter().any(|tparam| tparam.restriction().is_flag()) {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidTypeVar,
                    "`Flag` type parameters are not supported on type aliases".to_owned(),
                );
            }
            if tparams
                .iter()
                .any(|tparam| tparam.restriction().is_named_ints())
            {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidTypeVar,
                    "`NamedInts` type parameters are not supported on type aliases".to_owned(),
                );
            }
        }
        let index_source = match source {
            TParamsSource::Function => None,
            TParamsSource::Class => Some("classes"),
            TParamsSource::TypeAlias => Some("type aliases"),
        };
        if let Some(source_name) = index_source
            && tparams.iter().any(|tparam| tparam.restriction().is_index())
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!("`Index` type parameters are not supported on {source_name}"),
            );
        }
    }

    /// Parse a shape-extension annotation, leaving other `Annotated` forms alone.
    pub fn parse_shape_annotation(
        &self,
        value: &Expr,
        slice: &Expr,
        range: TextRange,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        self.parse_jaxtyping_type_form(value, slice, range, errors)
            .or_else(|| self.parse_shaped_annotation(slice, range, type_form_context, errors))
    }
}
