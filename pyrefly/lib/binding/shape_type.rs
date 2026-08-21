/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::meta_shape_dsl::ShapeDslFunction;
use pyrefly_types::meta_shape_dsl::convert_shape_dsl_function;
use pyrefly_types::type_level_dsl::ParsedTypeShapeDslFunction;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::binding::ShapedArrayMetadata;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::config::error_kind::ErrorKind;
use crate::export::special::SpecialExport;

#[derive(Clone, Debug)]
pub enum TypeParameterBound {
    Ordinary(Expr),
    ShapeFlag {
        domain: Option<Expr>,
        range: TextRange,
    },
}

pub(super) struct ShapeFunctionMetadata {
    pub shape_dsl_def: Option<Arc<ShapeDslFunction>>,
    pub type_shape_dsl_def: Option<Arc<ParsedTypeShapeDslFunction>>,
    pub uses_shape_dsl_ir_name: Option<ShortIdentifier>,
}
impl BindingsBuilder<'_> {
    /// Bind shape-specific function decorators before ordinary decorator processing consumes them.
    pub(super) fn record_shape_function_metadata(
        &mut self,
        function: &StmtFunctionDef,
        is_top_level: bool,
    ) -> ShapeFunctionMetadata {
        let is_shape_dsl = function.decorator_list.iter().any(|decorator| {
            self.as_special_export(&decorator.expression) == Some(SpecialExport::ShapeDslFunction)
        });
        let is_type_shape_dsl = function.decorator_list.iter().any(|decorator| {
            self.as_special_export(&decorator.expression)
                == Some(SpecialExport::TypeShapeDslFunction)
        });
        if is_shape_dsl && is_type_shape_dsl {
            self.error(
                function.name.range(),
                ErrorKind::InvalidArgument,
                "`@shape_dsl_function` and `@type_shape_dsl_function` cannot be combined"
                    .to_owned(),
            );
        }

        let type_shape_dsl_def = if is_type_shape_dsl && !is_shape_dsl {
            match ParsedTypeShapeDslFunction::try_new(function.clone(), is_top_level) {
                Ok(definition) => Some(Arc::new(definition)),
                Err(error) => {
                    self.error(
                        error.range,
                        ErrorKind::InvalidArgument,
                        format!("@type_shape_dsl_function {}", error.message),
                    );
                    None
                }
            }
        } else {
            None
        };

        let uses_shape_dsl_ir_name = function.decorator_list.iter().find_map(|decorator| {
            let call = decorator.expression.as_call_expr()?;
            if self.as_special_export(&call.func) != Some(SpecialExport::UsesShapeDsl) {
                return None;
            }
            let name = call.arguments.args.first()?.as_name_expr()?;
            Some(ShortIdentifier::expr_name(name))
        });

        let shape_dsl_def = if is_shape_dsl && !is_type_shape_dsl {
            if let Some(vararg) = &function.parameters.vararg {
                self.error(
                    vararg.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: *args parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(kwarg) = &function.parameters.kwarg {
                self.error(
                    kwarg.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: **kwargs parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(keyword_only) = function.parameters.kwonlyargs.first() {
                self.error(
                    keyword_only.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: keyword-only parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(positional_only) = function.parameters.posonlyargs.first() {
                self.error(
                    positional_only.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: positional-only parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }

            match convert_shape_dsl_function(function) {
                Ok(dsl_function) => {
                    let dsl_function = Arc::new(dsl_function);
                    self.metadata
                        .push_shape_dsl(function.name.id.clone(), Arc::clone(&dsl_function));
                    Some(dsl_function)
                }
                Err(error) => {
                    self.error(
                        error.range,
                        ErrorKind::InvalidArgument,
                        format!("@shape_dsl_function: {}", error.message),
                    );
                    None
                }
            }
        } else {
            None
        };

        ShapeFunctionMetadata {
            shape_dsl_def,
            type_shape_dsl_def,
            uses_shape_dsl_ir_name,
        }
    }

    /// Extract `@shaped_array(shape="Shape")` metadata from class decorators.
    pub(super) fn extract_shaped_array_metadata(
        &mut self,
        decorators: &[Decorator],
    ) -> Option<Box<ShapedArrayMetadata>> {
        let mut metadata = None;
        let mut seen_shaped_array = false;
        for decorator in decorators {
            let Some(call) = decorator.expression.as_call_expr() else {
                if self.as_special_export(&decorator.expression) == Some(SpecialExport::ShapedArray)
                {
                    if seen_shaped_array {
                        self.error(
                            decorator.range(),
                            ErrorKind::InvalidArgument,
                            "Duplicate `@shaped_array` decorator".to_owned(),
                        );
                        continue;
                    }
                    seen_shaped_array = true;
                    self.error(
                        decorator.range(),
                        ErrorKind::InvalidArgument,
                        "`@shaped_array` requires a `shape` keyword argument".to_owned(),
                    );
                }
                continue;
            };
            if self.as_special_export(&call.func) != Some(SpecialExport::ShapedArray) {
                continue;
            }
            if seen_shaped_array {
                self.error(
                    decorator.range(),
                    ErrorKind::InvalidArgument,
                    "Duplicate `@shaped_array` decorator".to_owned(),
                );
                continue;
            }
            seen_shaped_array = true;

            let mut invalid = false;
            if let Some(arg) = call.arguments.args.first() {
                self.error(
                    arg.range(),
                    ErrorKind::InvalidArgument,
                    "`@shaped_array` expects `shape` as a keyword argument".to_owned(),
                );
                invalid = true;
            }

            let mut shape_keyword = None;
            for keyword in &call.arguments.keywords {
                let Some(arg) = &keyword.arg else {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        "Unpacking is not supported in `@shaped_array`".to_owned(),
                    );
                    invalid = true;
                    continue;
                };
                if arg.as_str() == "shape" {
                    if shape_keyword.is_none() {
                        shape_keyword = Some(keyword);
                    }
                } else {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        format!(
                            "Unexpected keyword argument `{}` for `@shaped_array`; expected `shape`",
                            arg.id
                        ),
                    );
                    invalid = true;
                }
            }

            let Some(shape_keyword) = shape_keyword else {
                if !invalid {
                    self.error(
                        call.range(),
                        ErrorKind::InvalidArgument,
                        "`@shaped_array` requires a `shape` keyword argument".to_owned(),
                    );
                }
                continue;
            };
            let Expr::StringLiteral(shape) = &shape_keyword.value else {
                self.error(
                    shape_keyword.value.range(),
                    ErrorKind::InvalidArgument,
                    "`@shaped_array` `shape` argument must be a string literal".to_owned(),
                );
                continue;
            };
            if !invalid {
                metadata = Some(Box::new(ShapedArrayMetadata {
                    shape_name: Name::new(shape.value.to_str()),
                    range: shape_keyword.value.range(),
                }));
            }
        }
        metadata
    }

    /// Extract `capture_init` names from `@uses_shape_dsl` on a class's `forward` method.
    pub(super) fn extract_capture_init(&mut self, body: &[Stmt]) -> Option<Vec<Name>> {
        let forward = body
            .iter()
            .filter_map(|stmt| stmt.as_function_def_stmt())
            .find(|function| function.name.as_str() == "forward")?;

        forward.decorator_list.iter().find_map(|decorator| {
            let call = decorator.expression.as_call_expr()?;
            if self.as_special_export(&call.func) != Some(SpecialExport::UsesShapeDsl) {
                return None;
            }
            let capture_init = call.arguments.keywords.iter().find(|keyword| {
                keyword
                    .arg
                    .as_ref()
                    .is_some_and(|arg| arg.as_str() == "capture_init")
            })?;
            let list = capture_init.value.as_list_expr()?;
            Some(
                list.elts
                    .iter()
                    .filter_map(|element| {
                        if let Some(string) = element.as_string_literal_expr() {
                            Some(Name::new(string.value.to_str()))
                        } else {
                            self.error(
                                element.range(),
                                ErrorKind::InvalidArgument,
                                "`capture_init` entries must be string literals".to_owned(),
                            );
                            None
                        }
                    })
                    .collect(),
            )
        })
    }

    /// Record binding dependencies for a type parameter bound, including the
    /// shape-specific handling required by `shape_extensions.Flag`.
    pub(super) fn record_type_parameter_bound(
        &mut self,
        bound_expr: &mut Expr,
        usage: &mut Usage,
    ) -> TypeParameterBound {
        match bound_expr {
            Expr::Subscript(subscript) if self.is_shape_flag(&subscript.value) => {
                self.ensure_expr(&mut subscript.value, usage);
                self.ensure_type_with_usage(&mut subscript.slice, None, usage);
                TypeParameterBound::ShapeFlag {
                    domain: Some((*subscript.slice).clone()),
                    range: subscript.range,
                }
            }
            marker if self.is_shape_flag(marker) => {
                let range = marker.range();
                self.ensure_expr(marker, usage);
                TypeParameterBound::ShapeFlag {
                    domain: None,
                    range,
                }
            }
            bound => {
                self.ensure_type_with_usage(bound, None, usage);
                TypeParameterBound::Ordinary(bound.clone())
            }
        }
    }

    fn is_shape_flag(&self, expr: &Expr) -> bool {
        if let Expr::Name(name) = expr
            && SpecialExport::new(&name.id) == Some(SpecialExport::Flag)
            && SpecialExport::Flag.defined_in(self.module_info.name())
        {
            return self.scopes.current_binding_is_module_binding(&name.id)
                && matches!(
                    self.scopes.binding_idx_for_name(&name.id),
                    Some((idx, _)) if self.binding_is_class_def(idx)
                );
        }
        self.as_special_export(expr) == Some(SpecialExport::Flag)
    }
}
