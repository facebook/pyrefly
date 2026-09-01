/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solver integration for the experimental `shape_extensions.MapIntTuples` operation.
//!
//! `MapIntTuples` is a shape-specific type operator rather than part of the type-level DSL
//! evaluator. Keeping its solver behavior here prevents the general annotation and DSL paths from
//! accumulating details of its mapper binding, evaluation, and parameter-pattern semantics.

use pyrefly_types::map_int_tuples::TypeLambda;
use pyrefly_types::map_int_tuples::map_int_tuples_mapper_binder;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::class::Class;

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
        let context = TypeFormContext::TypeArgument(&type_form_context);
        Some(TypeLambda::new(
            map_int_tuples_mapper_binder(self.module().name(), name),
            self.expr_untype(&lambda.body, context, errors),
        ))
    }

    /// Reduces `MapIntTuples[mapper, source]` eagerly to an ordinary tuple type.
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
        let mapper = self.parse_map_int_tuples_mapper(mapper, type_form_context, errors);
        let source_context = TypeFormContext::TypeArgument(&type_form_context);
        let source_type = self.expr_untype(source, source_context, errors);
        let mapped = match source_type {
            Type::Any(_) => mapper.map(|mapper| {
                Ok(self
                    .heap
                    .mk_unbounded_tuple(mapper.apply(IntTuple::shapeless().to_shape_arg_type())))
            }),
            never @ Type::Never(_) => mapper.map(|_| Ok(never)),
            Type::Tuple(tuple) => mapper.map(|mapper| mapper.apply_to_tuple(tuple)),
            source_type => {
                self.error(
                    errors,
                    source.range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Source argument to `MapIntTuples` must be an `IntTuples` value, got `{}`",
                        self.for_display(source_type)
                    ),
                );
                None
            }
        };
        match mapped.transpose() {
            Ok(Some(mapped)) => mapped,
            Ok(None) => fallback(),
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
}
