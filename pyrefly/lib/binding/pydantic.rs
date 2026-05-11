/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice::Iter;

use ruff_python_ast::DictItem;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprDict;
use ruff_python_ast::Keyword;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use starlark_map::Hashed;

use crate::binding::bindings::BindingsBuilder;
use crate::export::special::SpecialExport;

// special pydantic constants
pub const VALIDATION_ALIAS: Name = Name::new_static("validation_alias");
pub const VALIDATE_BY_NAME: Name = Name::new_static("validate_by_name");
pub const VALIDATE_BY_ALIAS: Name = Name::new_static("validate_by_alias");
pub const GT: Name = Name::new_static("gt");
pub const LT: Name = Name::new_static("lt");
pub const GE: Name = Name::new_static("ge");
pub const LE: Name = Name::new_static("le");
pub const ROOT: Name = Name::new_static("root");
pub const STRICT: Name = Name::new_static("strict");
pub const STRICT_DEFAULT: bool = false;
pub const FROZEN: Name = Name::new_static("frozen");
pub const FROZEN_DEFAULT: bool = false;
pub const EXTRA: Name = Name::new_static("extra");
pub const FIELD_VALIDATOR: Name = Name::new_static("field_validator");

// An abstraction to iterate over configuration values, whether `ConfigDict()` or a dict display
// is used.
enum PydanticConfigExpr<'a> {
    ExprCall(&'a ExprCall),
    ExprDict(&'a ExprDict),
}

impl<'a> PydanticConfigExpr<'a> {
    fn iter(&self) -> PydanticConfigExprIter<'_> {
        match *self {
            Self::ExprCall(expr_call) => {
                PydanticConfigExprIter::ExprCall(expr_call.arguments.keywords.iter())
            }
            Self::ExprDict(expr_dict) => PydanticConfigExprIter::ExprDict(expr_dict.items.iter()),
        }
    }
}

enum PydanticConfigExprIter<'a> {
    ExprCall(Iter<'a, Keyword>),
    ExprDict(Iter<'a, DictItem>),
}

impl<'a> Iterator for PydanticConfigExprIter<'a> {
    type Item = (&'a str, &'a Expr);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ExprCall(kw_iter) => {
                kw_iter.find_map(|kw| kw.arg.as_ref().map(|arg| (arg.as_str(), &kw.value)))
            }
            Self::ExprDict(items_iter) => items_iter.find_map(|i| {
                i.key.as_ref().and_then(|key| {
                    key.as_string_literal_expr()
                        .map(|s| (s.value.to_str(), &i.value))
                })
            }),
        }
    }
}

/// If a class body contains a `model_config` attribute assigned to a `pydantic.ConfigDict`, the
/// configuration options from the `ConfigDict`. In the answers phase, this will be merged with
/// configuration options from the class keywords to produce a full Pydantic model configuration.
#[derive(Debug, Clone, Default)]
pub struct PydanticConfigDict {
    pub frozen: Option<bool>,
    pub extra: Option<bool>,
    pub strict: Option<bool>,
    pub validate_by_name: Option<bool>,
    pub validate_by_alias: Option<bool>,
}

impl<'a> BindingsBuilder<'a> {
    fn get_pydantic_config_expr<'b>(&self, e: &'b Expr) -> Option<PydanticConfigExpr<'b>> {
        if let Some(call) = e.as_call_expr()
            && matches!(
                self.as_special_export(&call.func),
                Some(SpecialExport::PydanticConfigDict) | Some(SpecialExport::BuiltinsDict)
            )
        {
            return Some(PydanticConfigExpr::ExprCall(call));
        } else if let Some(expr_dict) = e.as_dict_expr() {
            return Some(PydanticConfigExpr::ExprDict(expr_dict));
        }

        None
    }

    /// Scan a class body for `@field_validator(...)` decorators with `mode='before'` or
    /// `mode='plain'`, and return the field names those validators target. When a before/plain
    /// validator is present, the corresponding `__init__` parameter should accept `Any` because
    /// the validator can transform arbitrary input into the declared type.
    // TODO: `mode='wrap'` validators also receive raw input, but they additionally receive
    // the inner validator as a callable. Supporting them requires more work.
    pub fn extract_field_validator_fields(&self, body: &[Stmt]) -> Vec<Name> {
        body.iter()
            .filter_map(|stmt| stmt.as_function_def_stmt())
            .flat_map(|func_def| &func_def.decorator_list)
            .filter_map(|decorator| {
                let call = decorator.expression.as_call_expr()?;
                let is_field_validator = match &*call.func {
                    Expr::Name(n) => n.id == FIELD_VALIDATOR,
                    Expr::Attribute(a) => a.attr.id == FIELD_VALIDATOR,
                    _ => false,
                };
                if !is_field_validator {
                    return None;
                }
                // Check for `mode='before'` or `mode='plain'` keyword.
                let has_before_or_plain_mode = call.arguments.keywords.iter().any(|kw| {
                    kw.arg.as_ref().is_some_and(|a| a.as_str() == "mode")
                        && matches!(
                            &kw.value,
                            Expr::StringLiteral(s)
                                if matches!(s.value.to_str(), "before" | "plain")
                        )
                });
                if !has_before_or_plain_mode {
                    return None;
                }
                Some(&call.arguments.args)
            })
            .flatten()
            .filter_map(|arg| {
                arg.as_string_literal_expr()
                    .map(|s| Name::new(s.value.to_str()))
            })
            .collect()
    }

    // The goal of this function is to extract pydantic metadata (https://docs.pydantic.dev/latest/concepts/models/) from expressions.
    // TODO: Consider propagating the entire expression instead of the value
    // in case it is aliased.
    pub fn extract_pydantic_config_dict(
        &self,
        e: &Expr,
        name: &Hashed<Name>,
        pydantic_config_dict: &mut PydanticConfigDict,
    ) {
        if name.as_str() == "model_config"
            && let Some(pydantic_config_expr) = self.get_pydantic_config_expr(e)
        {
            for (name, value) in pydantic_config_expr.iter() {
                if name == FROZEN
                    && let Expr::BooleanLiteral(bl) = value
                {
                    pydantic_config_dict.frozen = Some(bl.value);
                } else if name == EXTRA
                    && let Some(extra) = value.as_string_literal_expr()
                {
                    let extra_value = extra.value.to_str();
                    pydantic_config_dict.extra = if matches!(extra_value, "allow" | "ignore") {
                        Some(true)
                    } else if extra_value == "forbid" {
                        Some(false)
                    } else {
                        None
                    }
                } else if name == STRICT
                    && let Expr::BooleanLiteral(bl) = value
                {
                    pydantic_config_dict.strict = Some(bl.value);
                } else if name == VALIDATE_BY_NAME
                    && let Expr::BooleanLiteral(bl) = value
                {
                    pydantic_config_dict.validate_by_name = Some(bl.value);
                } else if name == VALIDATE_BY_ALIAS
                    && let Expr::BooleanLiteral(bl) = value
                {
                    pydantic_config_dict.validate_by_alias = Some(bl.value);
                }
            }
        }
    }
}
