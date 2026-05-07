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
use crate::error::error::ErrorQuickFix;

pub(crate) fn replace_with_enum_member_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    let replacement = enum_member_replacement(error)?;
    let literal_range = enclosing_string_literal_range(ast, error.range())?;
    Some((
        format!("Replace with `{replacement}`"),
        module_info.dupe(),
        literal_range,
        replacement.to_owned(),
    ))
}

fn enum_member_replacement(error: &Error) -> Option<&str> {
    let ErrorQuickFix::ReplaceWithEnumMember { replacement } = error.quick_fixes().first()?;
    Some(replacement.as_str())
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
