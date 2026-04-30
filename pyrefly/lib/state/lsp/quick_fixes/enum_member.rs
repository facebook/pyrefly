/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::error::error::Error;

pub(crate) fn replace_with_enum_member_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    let replacement = enum_member_suggestion(error)?;
    let literal_range = enclosing_string_literal_range(ast, error.range())?;
    Some((
        format!("Replace with `{replacement}`"),
        module_info.dupe(),
        literal_range,
        replacement.to_owned(),
    ))
}

fn enum_member_suggestion(error: &Error) -> Option<&str> {
    let details = error.msg_details()?;
    for line in details.lines() {
        let Some(suggestion) = line.trim().strip_prefix("Did you mean `") else {
            continue;
        };
        let Some(suggestion) = suggestion.strip_suffix("`?") else {
            continue;
        };
        if is_qualified_identifier(suggestion) {
            return Some(suggestion);
        }
    }
    None
}

fn is_qualified_identifier(s: &str) -> bool {
    s.split('.').count() >= 2
        && s.split('.').all(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
                && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
        })
}

fn enclosing_string_literal_range(ast: &ModModule, error_range: TextRange) -> Option<TextRange> {
    for node in Ast::locate_node(ast, error_range.start()) {
        if let AnyNodeRef::ExprStringLiteral(literal) = node
            && literal.range().contains_range(error_range)
        {
            return Some(literal.range());
        }
    }
    None
}
