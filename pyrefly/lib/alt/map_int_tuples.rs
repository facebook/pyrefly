/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solver integration for the experimental `MapIntTuples` shape operation.
//!
//! A map normally applies its lambda forward to each known `IntTuple` source member. In parameter
//! annotations the direction is reversed: Pyrefly matches each argument member against the lambda
//! body to infer the symbolic `IntTuples` source. This module owns both interpretations and the
//! annotation and call-boundary policy that selects between them.

use pyrefly_types::callable::Param;
use pyrefly_types::equality::TypeEq;
use pyrefly_types::equality::TypeEqCtx;
use pyrefly_types::map_int_tuples::MapIntTuplesInterpretation;
use pyrefly_types::map_int_tuples::TypeLambda;
use pyrefly_types::map_int_tuples::map_int_tuples_mapper_binder;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::TypeCheckOptions;
use crate::alt::solve::Iterable;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::solver::solver::CallContext;
use crate::solver::solver::SubsetError;
use crate::types::class::Class;

/// The parts of a `MapIntTuples` parameter pattern needed during argument matching.
pub(crate) struct MapIntTuplesParameterPattern<'a> {
    mapper: &'a TypeLambda,
    mapped_member: &'a Type,
    source: &'a Type,
}

/// Returns a `MapIntTuples` parameter pattern retained in a callable annotation.
pub(crate) fn map_int_tuples_parameter_pattern(
    ty: &Type,
) -> Option<MapIntTuplesParameterPattern<'_>> {
    let ty = match ty {
        Type::Unpack(inner) => inner.as_ref(),
        ty => ty,
    };
    let Type::TypeLevelDslCall(call) = ty else {
        return None;
    };
    let Some((mapper, MapIntTuplesInterpretation::ParameterPattern { mapped_member }, source)) =
        call.as_map_int_tuples()
    else {
        return None;
    };
    Some(MapIntTuplesParameterPattern {
        mapper,
        mapped_member,
        source,
    })
}

/// An argument matched against a shape-specific `MapIntTuples` parameter pattern.
/// Expressions are preserved until inversion so a list literal can act as tuple-like syntax while
/// retaining each element's shape. Arbitrary list values remain ordinary list types. This
/// pragmatic exception lets existing Torch call sites adopt shape annotations without changing
/// their runtime containers.
pub(crate) enum MapIntTuplesPatternArgument<'a> {
    Expr(&'a Expr),
    Type(Type),
}

/// Replaces deferred shape-DSL computations with their gradual results while preserving the
/// ordinary type structure around them.
///
/// For example, the mapper body `Box[append_one(S)]` cannot be inverted through `append_one`, so
/// its validation view is `Box[IntTuple]`: the unknown call result is a gradual shape, while the
/// known `Box` constructor still rejects non-box values and boxes whose argument cannot represent
/// a shape.
fn approximate_map_int_tuples_pattern_view(mut ty: Type) -> Type {
    ty.transform_mut(&mut |ty| {
        if let Type::TypeLevelDslCall(call) = ty {
            *ty = call.fallback();
        }
    });
    debug_assert!(
        !ty.any(|ty| matches!(ty, Type::TypeLevelDslCall(_))),
        "post-order approximation must replace nested shape-DSL calls before their parents"
    );
    ty
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    pub(crate) fn resolve_map_int_tuples_mapper_parameter(&self, name: &Identifier) -> Type {
        let binder = map_int_tuples_mapper_binder(self.module().name(), name);
        self.heap.mk_type_of(binder.to_type(self.heap))
    }

    /// Whether this class is the experimental `shape_extensions.MapIntTuples` operation.
    pub(crate) fn is_map_int_tuples_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("shape_extensions", "MapIntTuples")
    }

    /// Whether this type is the unsubscripted runtime shim for experimental `MapIntTuples`.
    pub(crate) fn is_bare_map_int_tuples(&self, ty: &Type) -> bool {
        matches!(
            ty,
            Type::ClassType(cls)
                if cls.targs().is_empty()
                    && self.is_map_int_tuples_class(cls.class_object())
        )
    }

    /// Parses the unary type-level function accepted by `MapIntTuples`.
    ///
    /// Its parameter is always constrained to a shapeless `IntTuple`.
    fn parse_map_int_tuples_mapper(
        &self,
        mapper: &Expr,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Option<TypeLambda> {
        let Expr::Lambda(lambda) = mapper else {
            self.error(
                errors,
                mapper.range(),
                ErrorKind::InvalidAnnotation,
                "First argument to `MapIntTuples` must be a lambda".to_owned(),
            );
            return None;
        };
        let Some(parameters) = lambda.parameters.as_ref().filter(|parameters| {
            parameters.posonlyargs.is_empty()
                && parameters.args.len() == 1
                && parameters.vararg.is_none()
                && parameters.kwonlyargs.is_empty()
                && parameters.kwarg.is_none()
        }) else {
            self.error(
                errors,
                mapper.range(),
                ErrorKind::InvalidAnnotation,
                "Mapper for `MapIntTuples` must have exactly one positional parameter".to_owned(),
            );
            return None;
        };
        let parameter = &parameters.args[0];
        if parameter.default.is_some() {
            self.error(
                errors,
                mapper.range(),
                ErrorKind::InvalidAnnotation,
                "Mapper for `MapIntTuples` must have exactly one positional parameter without a default"
                    .to_owned(),
            );
            return None;
        }

        let name = parameter.name();
        if !self.bindings().is_type_level_lambda_parameter(name) {
            self.error(
                errors,
                mapper.range(),
                ErrorKind::InvalidAnnotation,
                "First argument to `MapIntTuples` must be a directly recognized mapper lambda"
                    .to_owned(),
            );
            return None;
        }
        // A mapper is a type-level callable, so its body is its return position. This permits
        // shape-DSL calls in the body without making them legal in the enclosing annotation.
        let context = TypeFormContext::TypeLevelLambdaReturn(&type_form_context);
        Some(TypeLambda::new(
            map_int_tuples_mapper_binder(self.module().name(), name),
            self.expr_untype(&lambda.body, context, errors),
        ))
    }

    /// Parses `MapIntTuples[mapper, source]`, reducing concrete sources eagerly and retaining
    /// symbolic sources until generic specialization supplies their members.
    pub(crate) fn parse_map_int_tuples(
        &self,
        arguments: &[Expr],
        range: TextRange,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        let fallback = || self.heap.mk_unbounded_tuple(Type::any_error());
        let [mapper, source] = arguments else {
            self.error(
                errors,
                range,
                ErrorKind::BadSpecialization,
                format!(
                    "Expected 2 type arguments for `MapIntTuples`, got {}",
                    arguments.len()
                ),
            );
            return fallback();
        };
        let mapper_range = mapper.range();
        let mapper = self.parse_map_int_tuples_mapper(mapper, type_form_context, errors);
        let source_context = TypeFormContext::TypeArgument(&type_form_context);
        let source_type = self.expr_untype(source, source_context, errors);
        let source_is_deferred = match &source_type {
            Type::Any(_) | Type::Never(_) | Type::Tuple(_) => false,
            Type::Quantified(_) | Type::TypeVar(_) | Type::TypeLevelDslCall(_)
                if self.is_int_tuples_dsl_argument(&source_type) =>
            {
                true
            }
            Type::Union(_) if self.is_int_tuples_dsl_argument(&source_type) => false,
            _ => {
                self.error(
                    errors,
                    source.range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Source argument to `MapIntTuples` must be an `IntTuples` value, got `{}`",
                        self.for_display(source_type)
                    ),
                );
                return fallback();
            }
        };
        let Some(mapper) = mapper else {
            return fallback();
        };
        let call = TypeLevelDslCall::map_int_tuples(mapper, source_type);
        if source_is_deferred {
            return Type::TypeLevelDslCall(Box::new(call));
        }
        match call.evaluate() {
            Ok(mut mapped) => {
                for error in mapped.finalize_type_level_dsl_at_boundary() {
                    self.error(
                        errors,
                        mapper_range,
                        ErrorKind::InvalidAnnotation,
                        error.to_string(),
                    );
                }
                mapped
            }
            Err(error) => {
                self.error(
                    errors,
                    source.range(),
                    ErrorKind::InvalidAnnotation,
                    error.to_string(),
                );
                fallback()
            }
        }
    }

    fn make_map_int_tuples_parameter_pattern(&self, ty: Type) -> Type {
        let Type::TypeLevelDslCall(mut call) = ty else {
            return ty;
        };
        let Some(map) = call.as_map_int_tuples_mut() else {
            return Type::TypeLevelDslCall(call);
        };
        let (mapper, _, _) = map.parts();
        let member = approximate_map_int_tuples_pattern_view(
            mapper.apply(IntTuple::shapeless().to_shape_arg_type()),
        );
        map.make_parameter_pattern(member);
        Type::TypeLevelDslCall(call)
    }

    /// Whether runtime untyping must preserve an unpacked `MapIntTuples` call until the variadic
    /// parameter annotation root interprets it. Other calls in this variadic-parameter annotation
    /// remain invalid.
    pub(crate) fn should_preserve_unpacked_map_int_tuples_call(
        &self,
        ty: &Type,
        type_form_context: TypeFormContext<'_>,
    ) -> bool {
        type_form_context == TypeFormContext::ParameterArgsAnnotation
            && matches!(ty, Type::Unpack(inner)
                if matches!(inner.as_ref(), Type::TypeLevelDslCall(call)
                    if call.as_map_int_tuples().is_some()))
    }

    /// Applies the experimental `MapIntTuples` interpretation at its two supported annotation
    /// roots: ordinary parameters and unpacked variadic parameters.
    pub(crate) fn interpret_map_int_tuples_at_annotation_root(
        &self,
        ty: Type,
        type_form_context: TypeFormContext<'_>,
    ) -> Type {
        match type_form_context {
            TypeFormContext::ParameterAnnotation => self.make_map_int_tuples_parameter_pattern(ty),
            TypeFormContext::ParameterArgsAnnotation => {
                let Type::Unpack(inner) = ty else {
                    return ty;
                };
                self.heap
                    .mk_unpack(self.make_map_int_tuples_parameter_pattern(*inner))
            }
            _ => ty,
        }
    }

    /// Projects a `MapIntTuples` pattern to the ordinary collection type visible in the function
    /// body: `Sequence[...]` for an ordinary parameter or `tuple[..., ...]` for an unpacked
    /// variadic parameter. The callable signature retains the pattern for later argument solving.
    pub(crate) fn map_int_tuples_parameter_body_type(&self, ty: Type) -> Type {
        let (call, unpacked) = match &ty {
            Type::TypeLevelDslCall(call) => (call, false),
            Type::Unpack(inner) => {
                let Type::TypeLevelDslCall(call) = inner.as_ref() else {
                    return ty;
                };
                (call, true)
            }
            Type::Tuple(Tuple::Unpacked(tuple)) => {
                let (prefix, middle, suffix) = tuple.parts();
                if !prefix.is_empty() || !suffix.is_empty() {
                    return ty;
                }
                let Type::TypeLevelDslCall(call) = middle else {
                    return ty;
                };
                (call, true)
            }
            _ => return ty,
        };
        let Some((_, MapIntTuplesInterpretation::ParameterPattern { mapped_member }, _)) =
            call.as_map_int_tuples()
        else {
            return ty;
        };
        if unpacked {
            Type::unbounded_tuple(mapped_member.clone())
        } else {
            self.heap
                .mk_class_type(self.stdlib.sequence(mapped_member.clone()))
        }
    }

    /// Attempts to recover the `IntTuple` used to construct one mapped argument member.
    ///
    /// For example, matching `Box[IntTuple[2, 3]]` against the mapper body `Box[S]` solves the
    /// synthetic mapper parameter `S` to `IntTuple[2, 3]`. The probe uses ordinary generic
    /// inference and always rolls back its constraints. Only the whole-pattern check may commit
    /// them.
    fn infer_map_int_tuples_pattern_member(
        &self,
        mapper: &TypeLambda,
        mapper_uses_parameter: bool,
        actual: &Type,
    ) -> Option<Type> {
        if matches!(actual, Type::Any(_) | Type::Never(_) | Type::Union(_))
            || matches!(mapper.body(), Type::Union(_))
            || !mapper_uses_parameter
        {
            return None;
        }

        // Snapshot only state that existed before this probe. The synthetic mapper variable is a
        // linear obligation owned by `handle`; restoring it after `finish_quantified` would
        // resurrect an unfinished quantified variable.
        let snapshot = self
            .solver()
            .snapshot_for_speculative_inference(&[actual, mapper.body()]);
        let tparams = TParams::new(vec![mapper.parameter().clone()]);
        let (handle, target) =
            self.solver()
                .fresh_quantified(&tparams, mapper.body().clone(), self.uniques);
        let Some(&var) = handle.vars().first() else {
            unreachable!("a MapIntTuples mapper always contributes one fresh parameter");
        };
        let matched = self
            .solver()
            .is_subset_eq(actual, &target, self.type_order(), None)
            .is_ok();
        let finished = self.finish_quantified(handle, false).is_ok();
        let candidate = self.solver().expand(Type::Var(var));
        self.solver().restore_vars(snapshot);
        debug_assert!(
            !self.solver().var_is_quantified(var),
            "a MapIntTuples inference probe must finalize its synthetic mapper variable"
        );

        if !matched
            || !finished
            || matches!(candidate, Type::Any(_))
            || candidate.any(|ty| matches!(ty, Type::Var(_)))
            || !self.is_int_tuple_dsl_argument(&candidate)
        {
            None
        } else {
            Some(candidate)
        }
    }

    /// Inverts an inferred tuple or `Sequence` argument into the map's symbolic source and an
    /// ordinary `Sequence` view used to validate the argument.
    fn invert_map_int_tuples_parameter_pattern(
        &self,
        mapper: &TypeLambda,
        actual: &Type,
        exact_members: Option<&[Type]>,
    ) -> (Type, Type) {
        fn invert_member<Ans: LookupAnswer>(
            solver: &AnswersSolver<Ans>,
            mapper: &TypeLambda,
            mapper_uses_parameter: bool,
            gradual: &Type,
            mapped_member_types: &mut Vec<Type>,
            member: &Type,
        ) -> Type {
            match solver.infer_map_int_tuples_pattern_member(mapper, mapper_uses_parameter, member)
            {
                Some(candidate) => {
                    mapped_member_types.push(approximate_map_int_tuples_pattern_view(
                        mapper.apply(candidate.clone()),
                    ));
                    candidate
                }
                None => {
                    mapped_member_types.push(gradual.clone());
                    IntTuple::shapeless().to_shape_arg_type()
                }
            }
        }

        fn map_tuple<Ans: LookupAnswer>(
            solver: &AnswersSolver<Ans>,
            mapper: &TypeLambda,
            mapper_uses_parameter: bool,
            gradual: &Type,
            mapped_member_types: &mut Vec<Type>,
            tuple: &Tuple,
        ) -> Tuple {
            let map = |members: &[Type], mapped_member_types: &mut Vec<Type>| {
                members
                    .iter()
                    .map(|member| {
                        invert_member(
                            solver,
                            mapper,
                            mapper_uses_parameter,
                            gradual,
                            mapped_member_types,
                            member,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            match tuple {
                Tuple::Concrete(members) => Tuple::Concrete(map(members, mapped_member_types)),
                Tuple::Unbounded(member) => Tuple::Unbounded(Box::new(invert_member(
                    solver,
                    mapper,
                    mapper_uses_parameter,
                    gradual,
                    mapped_member_types,
                    member,
                ))),
                Tuple::Unpacked(unpacked) => {
                    let (prefix, middle, suffix) = unpacked.parts();
                    let prefix = map(prefix, mapped_member_types);
                    let middle = match middle {
                        Type::Tuple(tuple) => Type::Tuple(map_tuple(
                            solver,
                            mapper,
                            mapper_uses_parameter,
                            gradual,
                            mapped_member_types,
                            tuple,
                        )),
                        _ => Type::unbounded_tuple({
                            mapped_member_types.push(gradual.clone());
                            IntTuple::shapeless().to_shape_arg_type()
                        }),
                    };
                    let suffix = map(suffix, mapped_member_types);
                    Tuple::unpacked(prefix, middle, suffix)
                }
            }
        }

        let mut mapper_uses_parameter = false;
        mapper.body().for_each_quantified(&mut |quantified| {
            mapper_uses_parameter |= quantified == mapper.parameter();
        });
        let gradual = approximate_map_int_tuples_pattern_view(
            mapper.apply(IntTuple::shapeless().to_shape_arg_type()),
        );
        let mut mapped_member_types = Vec::new();
        let source = if let Some(members) = exact_members {
            Type::concrete_tuple(
                members
                    .iter()
                    .map(|member| {
                        invert_member(
                            self,
                            mapper,
                            mapper_uses_parameter,
                            &gradual,
                            &mut mapped_member_types,
                            member,
                        )
                    })
                    .collect(),
            )
        } else if let Type::Tuple(tuple) = actual {
            Type::Tuple(map_tuple(
                self,
                mapper,
                mapper_uses_parameter,
                &gradual,
                &mut mapped_member_types,
                tuple,
            ))
        } else if matches!(actual, Type::Any(_) | Type::Never(_) | Type::Union(_)) {
            mapped_member_types.push(gradual.clone());
            Type::unbounded_tuple(IntTuple::shapeless().to_shape_arg_type())
        } else {
            let iterables =
                self.iterate(actual, TextRange::default(), &self.error_swallower(), None);
            match iterables.as_slice() {
                [Iterable::FixedLen(members)] => Type::concrete_tuple(
                    members
                        .iter()
                        .map(|member| {
                            invert_member(
                                self,
                                mapper,
                                mapper_uses_parameter,
                                &gradual,
                                &mut mapped_member_types,
                                member,
                            )
                        })
                        .collect(),
                ),
                [Iterable::OfType(member)] => Type::unbounded_tuple(invert_member(
                    self,
                    mapper,
                    mapper_uses_parameter,
                    &gradual,
                    &mut mapped_member_types,
                    member,
                )),
                [
                    Iterable::Unpacked {
                        prefix,
                        middle,
                        suffix,
                    },
                ] => {
                    let prefix = prefix
                        .iter()
                        .map(|member| {
                            invert_member(
                                self,
                                mapper,
                                mapper_uses_parameter,
                                &gradual,
                                &mut mapped_member_types,
                                member,
                            )
                        })
                        .collect();
                    let middle = Type::unbounded_tuple(invert_member(
                        self,
                        mapper,
                        mapper_uses_parameter,
                        &gradual,
                        &mut mapped_member_types,
                        middle,
                    ));
                    let suffix = suffix
                        .iter()
                        .map(|member| {
                            invert_member(
                                self,
                                mapper,
                                mapper_uses_parameter,
                                &gradual,
                                &mut mapped_member_types,
                                member,
                            )
                        })
                        .collect();
                    Type::unpacked_tuple(prefix, middle, suffix)
                }
                _ => {
                    mapped_member_types.push(gradual.clone());
                    Type::unbounded_tuple(IntTuple::shapeless().to_shape_arg_type())
                }
            }
        };
        let member = if mapped_member_types.is_empty() {
            self.heap.mk_never()
        } else {
            self.unions(mapped_member_types)
        };
        (
            source,
            self.heap.mk_class_type(self.stdlib.sequence(member)),
        )
    }

    /// Checks an argument against a `MapIntTuples` parameter pattern and commits the recovered
    /// source only when both the ordinary sequence view and the source bound match.
    pub(crate) fn check_map_int_tuples_parameter_pattern(
        &self,
        pattern: MapIntTuplesParameterPattern<'_>,
        argument: MapIntTuplesPatternArgument,
        range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
        call_context: &CallContext,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let MapIntTuplesParameterPattern {
            mapper,
            mapped_member,
            source: map_source,
        } = pattern;
        let infer_or_error = |expr| {
            let error_count = arg_errors.len_hard();
            let ty = self.expr_infer(expr, arg_errors);
            if arg_errors.len_hard() > error_count {
                self.heap.mk_any_error()
            } else {
                ty
            }
        };
        let (actual, exact_members) = match argument {
            MapIntTuplesPatternArgument::Type(ty) => (ty, None),
            MapIntTuplesPatternArgument::Expr(expr) => match expr {
                Expr::List(list) if list.elts.iter().any(|elt| matches!(elt, Expr::Starred(_))) => {
                    // Preserve ordinary expression diagnostics, constraints, and dependencies
                    // even though exact inversion cannot represent a starred list's cardinality.
                    let _ = infer_or_error(expr);
                    (
                        self.error_with_context(
                            call_errors,
                            range,
                            ErrorKind::BadArgumentType,
                            "Starred list elements are not supported by a `MapIntTuples` parameter pattern"
                                .to_owned(),
                            context,
                        ),
                        None,
                    )
                }
                Expr::List(list) if list.elts.is_empty() => {
                    // Ordinary inference gives an empty list a fresh element variable. Keep that
                    // validation behavior while exact inversion records its zero cardinality.
                    (infer_or_error(expr), Some(Vec::new()))
                }
                Expr::List(list) => {
                    let members = list.elts.iter().map(infer_or_error).collect::<Vec<_>>();
                    let element = self.unions(
                        members
                            .iter()
                            .cloned()
                            .map(|ty| ty.promote_implicit_literals(self.stdlib))
                            .collect(),
                    );
                    (
                        self.heap.mk_class_type(self.stdlib.list(element)),
                        Some(members),
                    )
                }
                _ => (infer_or_error(expr), None),
            },
        };
        // Ordinary argument inference above is not speculative and remains committed. Snapshot
        // before inversion and validation, which may constrain existing solver state. The mapper
        // probe creates and finalizes its own temporary variable after this snapshot.
        let transaction_snapshot = self.solver().snapshot_for_speculative_inference(&[
            &actual,
            mapper.body(),
            mapped_member,
            map_source,
        ]);
        let (source, inferred_validation_type) =
            self.invert_map_int_tuples_parameter_pattern(mapper, &actual, exact_members.as_deref());
        let view_error = self.check_type_with_options(
            &actual,
            &inferred_validation_type,
            range,
            TypeCheckOptions::new(call_errors, tcc).with_call_context(call_context),
        );
        let view_has_instantiation_errors = self
            .solver()
            .has_new_instantiation_errors(&transaction_snapshot);
        if view_error.is_none() && view_has_instantiation_errors {
            self.report_type_error(
                &actual,
                &inferred_validation_type,
                call_errors,
                range,
                tcc,
                SubsetError::Other,
            );
        }
        let view_matched = view_error.is_none() && !view_has_instantiation_errors;
        let source_snapshot = self
            .solver()
            .snapshot_for_speculative_inference(&[&source, map_source]);
        let source_matched = view_matched
            && self
                .solver()
                .is_subset_eq(&source, map_source, self.type_order(), None)
                .is_ok()
            && !self.solver().has_new_instantiation_errors(&source_snapshot);
        let matched = source_matched
            && !self
                .solver()
                .has_new_instantiation_errors(&transaction_snapshot);
        if view_matched && !source_matched {
            self.error_with_context(
                call_errors,
                range,
                ErrorKind::BadArgumentType,
                format!(
                    "Shapes `{}` recovered from this argument are not assignable to the `MapIntTuples` source `{}`",
                    self.for_display(source),
                    self.for_display(map_source.clone()),
                ),
                context,
            );
        }
        if !matched {
            self.solver().restore_vars(transaction_snapshot);
        }
        actual
    }

    /// Rejects multiple parameter patterns that would infer the same symbolic source.
    pub(crate) fn validate_map_int_tuples_parameter_patterns(
        &self,
        params: &[Param],
        mut ranges: impl Iterator<Item = TextRange>,
        errors: &ErrorCollector,
    ) {
        let mut sources: Vec<&Type> = Vec::new();
        for param in params {
            let range = ranges
                .next()
                .expect("each callable parameter has a source range");
            let ty = match param {
                Param::PosOnly(_, ty, _)
                | Param::Pos(_, ty, _)
                | Param::Varargs(_, ty)
                | Param::KwOnly(_, ty, _)
                | Param::Kwargs(_, ty) => ty,
            };
            let Some(pattern) = map_int_tuples_parameter_pattern(ty) else {
                continue;
            };
            if sources
                .iter()
                .any(|seen| pattern.source.type_eq(seen, &mut TypeEqCtx::default()))
            {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    "An `IntTuples` source may have only one `MapIntTuples` parameter pattern"
                        .to_owned(),
                );
            } else {
                sources.push(pattern.source);
            }
        }
        assert!(
            ranges.next().is_none(),
            "a callable cannot have source ranges without parameters"
        );
    }
}
