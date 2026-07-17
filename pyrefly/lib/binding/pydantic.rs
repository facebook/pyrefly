/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice::Iter;

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
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
pub const ALIAS_GENERATOR: Name = Name::new_static("alias_generator");

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, VisitMut)]
pub enum PydanticAliasGenerator {
    ToCamel,
    ToPascal,
    ToSnake,
}

impl PydanticAliasGenerator {
    pub fn from_special_export(export: SpecialExport) -> Option<Self> {
        match export {
            SpecialExport::PydanticToCamel => Some(Self::ToCamel),
            SpecialExport::PydanticToPascal => Some(Self::ToPascal),
            SpecialExport::PydanticToSnake => Some(Self::ToSnake),
            _ => None,
        }
    }

    pub fn generate(&self, field_name: &str) -> String {
        match self {
            Self::ToCamel => Self::to_camel(field_name),
            Self::ToPascal => Self::to_pascal(field_name),
            Self::ToSnake => Self::to_snake(field_name),
        }
    }

    fn to_pascal(field_name: &str) -> String {
        let mut title = String::with_capacity(field_name.len());
        let mut previous_is_cased = false;
        for ch in field_name.chars() {
            if ch.is_alphabetic() {
                if previous_is_cased {
                    title.extend(ch.to_lowercase());
                } else {
                    title.extend(ch.to_uppercase());
                }
                previous_is_cased = true;
            } else {
                title.push(ch);
                previous_is_cased = false;
            }
        }
        let title_chars: Vec<_> = title.chars().collect();
        let mut pascal = String::with_capacity(title.len());
        for (i, ch) in title_chars.iter().enumerate() {
            if *ch == '_'
                && i > 0
                && title_chars[i - 1].is_ascii_alphanumeric()
                && title_chars
                    .get(i + 1)
                    .is_some_and(|next| next.is_ascii_digit() || next.is_ascii_uppercase())
            {
                continue;
            }
            pascal.push(*ch);
        }
        pascal
    }

    fn to_camel(field_name: &str) -> String {
        let pascal = Self::to_pascal(field_name);
        let mut camel = String::with_capacity(pascal.len());
        let mut lowercased = false;
        for ch in pascal.chars() {
            if !lowercased && ch != '_' {
                camel.extend(ch.to_lowercase());
                lowercased = true;
            } else {
                camel.push(ch);
            }
        }
        camel
    }

    fn to_snake(field_name: &str) -> String {
        // Mirror Pydantic's `pydantic.alias_generators.to_snake`, which applies these
        // underscore-insertions between adjacent characters, then replaces `-` with `_`
        // and lowercases everything:
        //   1. within a run of uppercase letters followed by `Upper+lower`, split before
        //      the trailing uppercase (`HTTPResponse` -> `http_response`)
        //   2. between a lowercase letter and an uppercase letter (`fooBar` -> `foo_bar`)
        //   3. between a digit and an uppercase letter (`foo2Bar` -> `foo_2_bar`)
        //   4. between a letter and a digit (`foo2bar` -> `foo_2bar`)
        // The insertions depend only on the original characters, so a single pass over
        // adjacent pairs reproduces the sequential regex substitutions.
        let chars: Vec<char> = field_name.chars().collect();
        let mut snake = String::with_capacity(field_name.len() + 4);
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 {
                let prev = chars[i - 1];
                let next = chars.get(i + 1).copied();
                let insert_underscore = (prev.is_ascii_uppercase()
                    && ch.is_ascii_uppercase()
                    && next.is_some_and(|n| n.is_ascii_lowercase()))
                    || (prev.is_ascii_lowercase() && ch.is_ascii_uppercase())
                    || (prev.is_ascii_digit() && ch.is_ascii_uppercase())
                    || (prev.is_ascii_alphabetic() && ch.is_ascii_digit());
                if insert_underscore {
                    snake.push('_');
                }
            }
            if ch == '-' {
                snake.push('_');
            } else {
                snake.extend(ch.to_lowercase());
            }
        }
        snake
    }
}

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
    pub alias_generator: Option<PydanticAliasGenerator>,
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

    fn extract_alias_generator(&self, value: &Expr) -> Option<PydanticAliasGenerator> {
        PydanticAliasGenerator::from_special_export(self.as_special_export(value)?)
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
                } else if name == ALIAS_GENERATOR {
                    pydantic_config_dict.alias_generator = self.extract_alias_generator(value);
                }
            }
        }
    }
}
