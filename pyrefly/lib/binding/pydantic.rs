/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use ruff_python_ast::DictItem;
use ruff_python_ast::Expr;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use starlark_map::Hashed;

use crate::binding::bindings::BindingsBuilder;
use crate::export::special::SpecialExport;

// special pydantic constants
pub const FROZEN_DEFAULT: bool = false;
pub const VALIDATION_ALIAS: Name = Name::new_static("validation_alias");
pub const VALIDATE_BY_NAME: Name = Name::new_static("validate_by_name");
pub const VALIDATE_BY_ALIAS: Name = Name::new_static("validate_by_alias");
pub const GT: Name = Name::new_static("gt");
pub const LT: Name = Name::new_static("lt");
pub const GE: Name = Name::new_static("ge");
pub const ROOT: Name = Name::new_static("root");
pub const STRICT: Name = Name::new_static("strict");
const FROZEN: Name = Name::new_static("frozen");
const EXTRA: Name = Name::new_static("extra");

// An abstraction to iterate over configuration values, whether `ConfigDict()` or a dict display
// is used.
enum PydanticConfigExpr<'a> {
    ExprCall(&'a ruff_python_ast::ExprCall),
    ExprDict(&'a ruff_python_ast::ExprDict),
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
    ExprCall(std::slice::Iter<'a, Keyword>),
    ExprDict(std::slice::Iter<'a, DictItem>),
}

impl<'a> Iterator for PydanticConfigExprIter<'a> {
    type Item = (&'a str, &'a Expr);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ExprCall(kw_iter) => {
                let nxt = kw_iter.find(|&kw| kw.arg.is_some());
                match nxt {
                    Some(kw) => Some((kw.arg.as_ref().unwrap().as_str(), &kw.value)),
                    None => None,
                }
            }
            Self::ExprDict(items_iter) => {
                let nxt = items_iter.find(|&i| {
                    i.key
                        .as_ref()
                        .is_some_and(|key| key.is_string_literal_expr())
                });
                match nxt {
                    Some(item) => Some((
                        item.key
                            .as_ref()
                            .unwrap()
                            .as_string_literal_expr()
                            .unwrap()
                            .value
                            .to_str(),
                        &item.value,
                    )),
                    None => None,
                }
            }
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
    pub validation_flags: PydanticValidationFlags,
}

/// Flags that control whether a Pydantic model's fields are populated by their names or their aliases.
/// See https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_by_name.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
pub struct PydanticValidationFlags {
    pub validate_by_name: bool,
    pub validate_by_alias: bool,
}

impl Default for PydanticValidationFlags {
    fn default() -> Self {
        Self {
            validate_by_name: false,
            validate_by_alias: true,
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    fn get_pydantic_config_expr<'b>(&self, e: &'b Expr) -> Option<PydanticConfigExpr<'b>> {
        if let Some(call) = e.as_call_expr()
            && (matches!(
                self.as_special_export(&call.func),
                Some(SpecialExport::PydanticConfigDict)
            ) || call
                .func
                .as_name_expr()
                .is_some_and(|name| name.id == "dict"))
        {
            return Some(PydanticConfigExpr::ExprCall(call));
        } else if let Some(expr_dict) = e.as_dict_expr() {
            return Some(PydanticConfigExpr::ExprDict(expr_dict));
        }

        None
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
                    pydantic_config_dict.validation_flags.validate_by_name = bl.value;
                } else if name == VALIDATE_BY_ALIAS
                    && let Expr::BooleanLiteral(bl) = value
                {
                    pydantic_config_dict.validation_flags.validate_by_alias = bl.value;
                }
            }
        }
    }
}
