/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solver integration for the experimental `shape_extensions.ScalarAsShape` marker.

use std::cmp::Ordering;

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_alias::TypeAliasIndex;
use pyrefly_types::types::AnyStyle;
use pyrefly_types::types::Type;
use pyrefly_util::visit::Visit;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::TypeCheckOptions;
use crate::alt::callable::ArgMap;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::TypeCheckContext;
use crate::solver::solver::CallContext;
use crate::solver::solver::Subset;
use crate::solver::solver::SubsetError;
use crate::solver::solver::SubsetWithSnapshotResult;
use crate::solver::solver::VarSnapshot;
use crate::types::class::Class;
use crate::types::type_var::Restriction;

#[derive(Clone)]
struct ScalarAsShape {
    source: Type,
    shape: Type,
}

type AliasKey = (ModuleName, TypeAliasIndex);

impl<'solver, 'subset, Ans: LookupAnswer> Subset<'solver, 'subset, Ans> {
    pub(crate) fn is_subset_callable_parameter(
        &mut self,
        actual: &Type,
        parameter: &Type,
    ) -> Result<(), SubsetError> {
        let actual = if let Some((ordinary, markers)) = split_scalar_as_shape_parameter(actual) {
            let mut accepted = ordinary.into_iter().collect::<Vec<_>>();
            let mut first_shape_error = None;
            for marker in markers {
                match self.with_speculative_subset_branch(&[&marker.shape], |me| {
                    me.is_subset_eq(
                        &IntTuple::new(Vec::new()).to_shape_arg_type(),
                        &marker.shape,
                    )
                }) {
                    SubsetWithSnapshotResult::Ok => accepted.push(marker.source),
                    SubsetWithSnapshotResult::Err(error) => {
                        first_shape_error.get_or_insert(error);
                    }
                }
            }
            if accepted.is_empty() {
                return Err(first_shape_error.unwrap_or(SubsetError::Other));
            }
            self.solver.unions(accepted, self.type_order)
        } else {
            actual.clone()
        };
        let Some((ordinary, markers)) = split_scalar_as_shape_parameter(parameter) else {
            return self.is_subset_eq(&actual, parameter);
        };
        let actual_members = match &actual {
            Type::Union(union) => union.members.iter().collect::<Vec<_>>(),
            _ => vec![&actual],
        };
        for actual in actual_members {
            if let Some(ordinary) = &ordinary
                && self
                    .with_speculative_subset_branch(&[actual, ordinary], |me| {
                        me.is_subset_eq(actual, ordinary)
                    })
                    .is_ok()
            {
                continue;
            }
            if matches!(actual, Type::Any(_) | Type::Never(_)) {
                continue;
            }
            let mut shape_error = None;
            let mut matched = false;
            for marker in &markers {
                let mut source_matched = false;
                let result = self.with_speculative_subset_branch(
                    &[actual, &marker.source, &marker.shape],
                    |me| {
                        me.is_subset_eq(actual, &marker.source)?;
                        source_matched = true;
                        me.is_subset_eq(
                            &IntTuple::new(Vec::new()).to_shape_arg_type(),
                            &marker.shape,
                        )
                    },
                );
                match result {
                    SubsetWithSnapshotResult::Ok => {
                        matched = true;
                        break;
                    }
                    SubsetWithSnapshotResult::Err(error) => {
                        if source_matched {
                            shape_error.get_or_insert(error);
                        }
                    }
                }
            }
            if matched {
                continue;
            }
            if let Some(error) = shape_error {
                return Err(error);
            }
            let mut accepted = ordinary.clone().into_iter().collect::<Vec<_>>();
            accepted.extend(markers.iter().map(|marker| marker.source.clone()));
            return self
                .is_subset_eq(actual, &self.solver.unions(accepted, self.type_order))
                .and(Err(SubsetError::Other));
        }
        Ok(())
    }
}

fn alias_key(alias: &TypeAliasData) -> Option<AliasKey> {
    match alias {
        TypeAliasData::Ref(reference) => Some((reference.module_name, reference.index)),
        TypeAliasData::Value(_) => None,
    }
}

fn scalar_as_shape(ty: &Type) -> Option<ScalarAsShape> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if !cls.has_qname("shape_extensions", "ScalarAsShape") {
        return None;
    }
    let [source, shape] = cls.targs().as_slice() else {
        return None;
    };
    Some(ScalarAsShape {
        source: source.clone(),
        shape: shape.clone(),
    })
}

fn contains_scalar_as_shape(ty: &Type) -> bool {
    ty.any(|ty| scalar_as_shape(ty).is_some())
}

fn is_direct_parameter_position(context: TypeFormContext<'_>) -> bool {
    match context {
        TypeFormContext::ParameterAnnotation
        | TypeFormContext::ParameterArgsAnnotation
        | TypeFormContext::ParameterKwargsAnnotation
        | TypeFormContext::TypeAlias => true,
        TypeFormContext::UnionMember(parent) => is_direct_parameter_position(*parent),
        _ => false,
    }
}

fn scalar_as_shape_actual_members(actual: &Type) -> Vec<Type> {
    if let Type::Union(union) = actual {
        return union.members.clone();
    }
    let restriction = match actual {
        Type::Quantified(q) if q.is_type_var() => Some(q.restriction()),
        Type::TypeVar(type_var) => Some(type_var.restriction()),
        _ => None,
    };
    match restriction {
        Some(Restriction::Bound(Type::Union(union))) => union.members.clone(),
        Some(Restriction::Bound(bound)) => vec![bound.clone()],
        Some(Restriction::Constraints(constraints)) => constraints.clone(),
        Some(Restriction::Flag(_) | Restriction::Unrestricted) | None => vec![actual.clone()],
    }
}

fn split_scalar_as_shape_parameter(ty: &Type) -> Option<(Option<Type>, Vec<ScalarAsShape>)> {
    if let Some(marker) = scalar_as_shape(ty) {
        return Some((None, vec![marker]));
    }
    let Type::Union(union) = ty else {
        return None;
    };
    let mut ordinary = Vec::new();
    let mut markers = Vec::new();
    for member in &union.members {
        if let Some(marker) = scalar_as_shape(member) {
            markers.push(marker);
        } else {
            ordinary.push(member.clone());
        }
    }
    if markers.is_empty() {
        None
    } else {
        Some((
            (!ordinary.is_empty()).then(|| Type::union(ordinary)),
            markers,
        ))
    }
}

pub(crate) fn is_scalar_as_shape_parameter(ty: &Type) -> bool {
    let is_marker = |ty: &Type| matches!(ty, Type::ClassType(cls) if cls.has_qname("shape_extensions", "ScalarAsShape"));
    is_marker(ty) || matches!(ty, Type::Union(union) if union.members.iter().any(is_marker))
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn collect_scalar_as_shape_markers(
        &self,
        ty: &Type,
        markers: &mut Vec<ScalarAsShape>,
        invalid_position: &mut bool,
    ) {
        let mut worklist = vec![(ty.clone(), true, Vec::new())];
        let mut seen_aliases: Vec<(TypeAliasData, bool)> = Vec::new();
        while let Some((candidate, direct, path)) = worklist.pop() {
            if let Type::TypeAlias(alias) | Type::UntypedAlias(alias) = &candidate {
                let mut path = path;
                if let Some(key) = alias_key(alias) {
                    if path.contains(&key) {
                        if let TypeAliasData::Ref(reference) = &**alias
                            && let Some(args) = &reference.args
                        {
                            // Stopping recursive alias expansion does not change the syntactic
                            // position of its specialization arguments.
                            worklist.extend(
                                args.as_slice()
                                    .iter()
                                    .cloned()
                                    .map(|arg| (arg, direct, path.clone())),
                            );
                        }
                        continue;
                    }
                    path.push(key);
                }
                if seen_aliases
                    .iter()
                    .any(|(seen, seen_direct)| seen == &**alias && *seen_direct == direct)
                {
                    continue;
                }
                seen_aliases.push(((**alias).clone(), direct));
                worklist.push((self.untype_alias(alias), direct, path));
            } else if let Some(marker) = scalar_as_shape(&candidate) {
                *invalid_position |= !direct;
                worklist.push((marker.source.clone(), false, path.clone()));
                worklist.push((marker.shape.clone(), false, path));
                markers.push(marker);
            } else if let Type::Union(union) = &candidate
                && direct
            {
                worklist.extend(
                    union
                        .members
                        .iter()
                        .cloned()
                        .map(|member| (member, true, path.clone())),
                );
            } else {
                candidate.recurse(&mut |child| worklist.push((child.clone(), false, path.clone())));
            }
        }
    }

    pub(crate) fn is_scalar_as_shape_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("shape_extensions", "ScalarAsShape")
    }

    pub(crate) fn validate_scalar_as_shape_annotation(
        &self,
        ty: Type,
        range: TextRange,
        context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        let mut markers = Vec::new();
        let mut invalid_position = false;
        self.collect_scalar_as_shape_markers(&ty, &mut markers, &mut invalid_position);
        if markers.is_empty() {
            return ty;
        }
        if markers.iter().any(|marker| {
            matches!(marker.source, Type::Any(AnyStyle::Implicit))
                && matches!(marker.shape, Type::Any(AnyStyle::Implicit))
        }) {
            return self.error(
                errors,
                range,
                ErrorKind::BadSpecialization,
                "`ScalarAsShape` requires two type arguments".to_owned(),
            );
        }
        if !is_direct_parameter_position(context) || invalid_position {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                "`shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member"
                    .to_owned(),
            );
        }
        for marker in markers {
            if marker.source.any(|source| {
                matches!(
                    source,
                    Type::Any(AnyStyle::Explicit | AnyStyle::Implicit) | Type::Never(_)
                )
            }) {
                return self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "First argument to `ScalarAsShape` may not contain `Any` or `Never`, got `{}`",
                        self.for_display(marker.source)
                    ),
                );
            }
            if !self.is_int_tuple_dsl_argument(&marker.shape) {
                return self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Second argument to `ScalarAsShape` must be an `IntTuple`, got `{}`",
                        self.for_display(marker.shape)
                    ),
                );
            }
        }
        ty
    }

    pub(crate) fn scalar_as_shape_parameter_body_type(&self, ty: Type) -> Type {
        let Some((ordinary, markers)) = split_scalar_as_shape_parameter(&ty) else {
            return ty;
        };
        let mut body = ordinary.into_iter().collect::<Vec<_>>();
        body.extend(markers.into_iter().map(|marker| marker.source));
        self.unions(body)
    }

    pub(crate) fn compare_scalar_as_shape_overloads(
        &self,
        left: &ArgMap,
        right: &ArgMap,
    ) -> Ordering {
        // Preserve normal overload selection unless this call actually used the marker conversion.
        if left.has_gradual_argument
            || right.has_gradual_argument
            || left.scalar_as_shape_conversions == 0 && right.scalar_as_shape_conversions == 0
        {
            return Ordering::Equal;
        }
        let at_least_as_specific = |left: &ArgMap, right: &ArgMap| {
            left.matched_params.len() == right.matched_params.len()
                && left.matched_params.iter().zip(&right.matched_params).all(
                    |((left_range, left), (right_range, right))| {
                        if left_range != right_range {
                            return false;
                        }
                        let left = self.scalar_as_shape_parameter_body_type(left.ty.clone());
                        let right = self.scalar_as_shape_parameter_body_type(right.ty.clone());
                        let snapshot = self
                            .solver()
                            .snapshot_for_speculative_inference(&[&left, &right]);
                        let result = self.is_subset_eq(&left, &right);
                        self.solver().restore_vars(snapshot);
                        result
                    },
                )
        };
        match (
            at_least_as_specific(left, right),
            at_least_as_specific(right, left),
        ) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (true, true) => right
                .scalar_as_shape_conversions
                .cmp(&left.scalar_as_shape_conversions),
            (false, false) => Ordering::Equal,
        }
    }

    pub(crate) fn scalar_as_shape_preserves_expanded_gradual_ambiguity(
        &self,
        argmap: &ArgMap,
    ) -> bool {
        argmap.has_unmaterialized_gradual_argument && argmap.scalar_as_shape_conversions > 0
    }

    pub(crate) fn check_scalar_as_shape_argument(
        &self,
        actual: &Type,
        parameter: &Type,
        range: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
        call_context: &CallContext<'_>,
    ) -> bool {
        let (ordinary, markers) = split_scalar_as_shape_parameter(parameter)
            .expect("callers check for a ScalarAsShape parameter");
        let check_speculative = |got: &Type,
                                 want: &Type,
                                 attempt_errors: &ErrorCollector,
                                 check_tcc: &dyn Fn() -> TypeCheckContext,
                                 snapshot: &VarSnapshot| {
            self.check_type_with_options(
                got,
                want,
                range,
                TypeCheckOptions::new(attempt_errors, check_tcc).with_call_context(call_context),
            )
            .is_none()
                && !self.solver().has_new_instantiation_errors(snapshot)
        };

        let actual_members = scalar_as_shape_actual_members(actual);
        let empty_shape = IntTuple::new(Vec::new()).to_shape_arg_type();
        let shape_tcc = || tcc().for_scalar_as_shape();
        let transaction = self
            .solver()
            .snapshot_for_speculative_inference(&[actual, parameter]);
        let mut converted = false;
        for actual in &actual_members {
            if let Some(ordinary) = &ordinary {
                let snapshot = self
                    .solver()
                    .snapshot_for_speculative_inference(&[actual, ordinary]);
                let attempt_errors = self.error_collector();
                if check_speculative(actual, ordinary, &attempt_errors, tcc, &snapshot) {
                    errors.extend(attempt_errors);
                    continue;
                }
                self.solver().restore_vars(snapshot);
            }

            if matches!(actual, Type::Any(_) | Type::Never(_)) {
                continue;
            }

            let mut first_shape_errors = None;
            let mut matched = false;
            for marker in &markers {
                let snapshot = self.solver().snapshot_for_speculative_inference(&[
                    actual,
                    &marker.source,
                    &marker.shape,
                ]);
                let attempt_errors = self.error_collector();
                if check_speculative(actual, &marker.source, &attempt_errors, tcc, &snapshot) {
                    let shape_matched = check_speculative(
                        &empty_shape,
                        &marker.shape,
                        &attempt_errors,
                        &shape_tcc,
                        &snapshot,
                    );
                    if shape_matched {
                        errors.extend(attempt_errors);
                        converted = true;
                        matched = true;
                        break;
                    }
                    if attempt_errors.is_empty() {
                        self.report_type_error(
                            &empty_shape,
                            &marker.shape,
                            &attempt_errors,
                            range,
                            &shape_tcc,
                            SubsetError::Other,
                        );
                    }
                    self.solver().restore_vars(snapshot);
                    if first_shape_errors.is_none() {
                        first_shape_errors = Some(attempt_errors);
                    }
                    continue;
                }
                self.solver().restore_vars(snapshot);
            }
            if matched {
                continue;
            }
            self.solver().restore_vars(transaction);
            if let Some(first_shape_errors) = first_shape_errors {
                errors.extend(first_shape_errors);
                return false;
            }

            let mut accepted = ordinary.clone().into_iter().collect::<Vec<_>>();
            accepted.extend(markers.iter().map(|marker| marker.source.clone()));
            let accepted = self.unions(accepted);
            self.report_type_error(actual, &accepted, errors, range, tcc, SubsetError::Other);
            return false;
        }
        converted
    }

    pub(crate) fn report_scalar_as_shape_default_error(
        &self,
        ty: &Type,
        default_range: Option<TextRange>,
        errors: &ErrorCollector,
    ) -> bool {
        let Some(range) = default_range.filter(|_| contains_scalar_as_shape(ty)) else {
            return false;
        };
        self.error(
            errors,
            range,
            ErrorKind::InvalidAnnotation,
            "A parameter using `ScalarAsShape` may not have a default".to_owned(),
        );
        true
    }
}
