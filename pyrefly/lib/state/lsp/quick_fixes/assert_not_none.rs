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
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::error::error::Error;
use crate::error::error::ErrorQuickFix;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_statement_range;
use crate::state::lsp::quick_fixes::extract_shared::line_indent_and_start;

/// Insert an assertion before a standalone statement that uses an optional name.
pub(crate) fn assert_not_none_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    if !error
        .quick_fixes()
        .iter()
        .any(|fix| matches!(fix, ErrorQuickFix::AssertNotNone))
    {
        return None;
    }
    let name = Ast::locate_node(ast, error.range().start())
        .into_iter()
        .find_map(|node| match node {
            AnyNodeRef::ExprName(name) if name.range().contains_range(error.range()) => Some(name),
            AnyNodeRef::ExprAttribute(attribute)
                if attribute.range().contains_range(error.range())
                    && let Expr::Name(name) = attribute.value.as_ref() =>
            {
                Some(name)
            }
            _ => None,
        })?;
    let statement = find_enclosing_statement_range(ast, error.range())?;
    let source = module_info.contents().as_str();
    let (indent, line_start) = line_indent_and_start(source, statement.start())?;

    // Inserting before a statement is only valid when it starts on its own line.
    if source[line_start.to_usize()..statement.start().to_usize()] != indent {
        return None;
    }

    let assertion = format!("assert {} is not None", name.id);
    Some((
        format!("Add `{assertion}`"),
        module_info.dupe(),
        TextRange::new(line_start, line_start),
        format!("{indent}{assertion}\n"),
    ))
}
