/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solving policy for the experimental `shape_extensions.Index` restriction.

use std::slice;

use pyrefly_types::callable::Param;
use pyrefly_types::callable::Required;
use pyrefly_types::dimension::Int;
use pyrefly_types::shape_index::lower_index_type;
use pyrefly_types::shaped_array::type_to_dim;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::shape_extension::direct_function_parameter_sources;
use crate::alt::solve::TypeFormContext;
use crate::binding::binding::FunctionDefData;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

impl<Ans: LookupAnswer> AnswersSolver<'_, '_, Ans> {
    /// Preserve shape-extension integer expressions used as slice bounds while leaving ordinary
    /// Python slice-bound inference unchanged when the expression is not symbolic shape arithmetic.
    pub(crate) fn infer_shape_index_slice_bound(
        &self,
        expr: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        if !self.solver().tensor_shapes {
            return self.expr_infer(expr, errors);
        }
        let parse_errors = self.error_swallower();
        let parse_symbolic = |expr| {
            self.parse_dimension_list_for_type_shape_dsl_int_argument(
                slice::from_ref(expr),
                TypeFormContext::TypeExpression,
                &parse_errors,
            )
            .ok()
            .and_then(|dimensions| dimensions.into_iter().next())
            .and_then(|ty| type_to_dim(&ty))
            .filter(|dimension| !matches!(dimension, Int::Literal(_) | Int::Int))
        };
        let mut base = expr;
        let mut is_negative = false;
        while let Expr::UnaryOp(unary) = base
            && unary.op == UnaryOp::USub
        {
            is_negative = !is_negative;
            base = &unary.operand;
        }
        if let Some(dimension) = parse_symbolic(base) {
            // Negative-bound normalization requires an uncanonicalized outer `-1` product.
            // Count the complete unary-minus chain so only odd parity receives that marker.
            return if is_negative {
                self.heap
                    .mk_int(Int::Mul(Box::new(Int::Literal(-1)), Box::new(dimension)))
            } else {
                self.heap.mk_int(dimension)
            };
        }
        self.expr_infer(expr, errors)
    }

    pub(crate) fn parse_index_shape_type_level_dsl_call(
        &self,
        call: &ExprCall,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        if !call.arguments.keywords.is_empty() {
            return self.error(
                errors,
                call.range(),
                ErrorKind::InvalidAnnotation,
                "`index_shape` does not accept keyword arguments".to_owned(),
            );
        }
        if let Some(argument) = call
            .arguments
            .args
            .iter()
            .find(|argument| matches!(argument, Expr::Starred(_)))
        {
            return self.error(
                errors,
                argument.range(),
                ErrorKind::InvalidAnnotation,
                "`index_shape` does not accept starred arguments".to_owned(),
            );
        }
        if call.arguments.args.len() != 2 {
            return self.error(
                errors,
                call.range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected 2 arguments for `index_shape`, got {}",
                    call.arguments.args.len()
                ),
            );
        }

        let argument_context = TypeFormContext::TypeArgument(&type_form_context);
        let shape = self.expr_untype(&call.arguments.args[0], argument_context, errors);
        let index = self.expr_untype(&call.arguments.args[1], argument_context, errors);
        if shape.is_error() {
            return shape;
        }
        if index.is_error() {
            return index;
        }

        let mut invalid = false;
        if !self.is_int_tuple_dsl_argument(&shape) {
            self.error(
                errors,
                call.arguments.args[0].range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected an `IntTuple` first argument to `index_shape`, got `{}`",
                    self.for_display(shape.clone())
                ),
            );
            invalid = true;
        }
        if !lower_index_type(&index).is_valid() {
            self.error(
                errors,
                call.arguments.args[1].range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected an `Index` second argument to `index_shape`, got `{}`",
                    self.for_display(index.clone())
                ),
            );
            invalid = true;
        }
        if invalid {
            return self.heap.mk_any_error();
        }

        Type::TypeLevelDslCall(Box::new(TypeLevelDslCall::index_shape(shape, index)))
    }

    pub(crate) fn validate_shape_index_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !restriction.is_index() {
            return None;
        }
        // Tuple type expressions infer as `type[...]`; defaults store the corresponding value.
        let default = match default {
            Type::Type(inner) => inner.as_ref(),
            _ => default,
        };
        if lower_index_type(default).is_valid() {
            Some(default.clone())
        } else {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!(
                    "Default for `Index` type parameter `{name}` is not a valid index value: `{default}`"
                ),
            );
            Some(self.heap.mk_any_error())
        }
    }

    pub(crate) fn validate_shape_index_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        for tparam in tparams
            .iter()
            .filter(|tparam| tparam.restriction().is_index())
        {
            let sources = direct_function_parameter_sources(stmt, params, tparam);
            if sources.len() != 1 {
                self.error(
                    errors,
                    stmt.name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Index` type parameter `{}` must directly annotate exactly one function parameter, found {}",
                        tparam.name(),
                        sources.len(),
                    ),
                );
                continue;
            }
            let (source_index, source_range, unpacked) = sources[0];
            if unpacked {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Index` type parameter `{}` cannot bind an unpacked parameter",
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
                && !lower_index_type(&default.ty).is_valid()
            {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "Default for parameter binding `Index` type parameter `{}` is not a valid index value: `{}`",
                        tparam.name(),
                        default.ty,
                    ),
                );
            }
        }
    }
}
