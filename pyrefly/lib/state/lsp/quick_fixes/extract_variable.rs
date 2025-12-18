/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_function::LocalRefactorCodeAction;

const DEFAULT_VARIABLE_PREFIX: &str = "extracted_value";

/// Builds extract-variable refactor actions for the supplied selection.
///
/// Returns `None` when the selection is not a single, non-empty expression or the
/// expression cannot be safely rewritten into a variable binding. Otherwise,
/// returns a single [`LocalRefactorCodeAction`] that inserts a binding above the
/// enclosing statement and replaces the selection with the new identifier.
pub(crate) fn extract_variable_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    if selection.is_empty() {
        return None;
    }
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let selection_text = module_info.code_at(selection);
    if selection_text.trim().is_empty() {
        return None;
    }
    let (leading_ws, expression_text, trailing_ws, expression_range) =
        split_selection(selection_text, selection)?;
    if !is_exact_expression(ast.as_ref(), expression_range) {
        return None;
    }
    let statement_range = find_enclosing_statement_range(ast.as_ref(), expression_range)?;
    let (statement_indent, insert_position) =
        line_indent_and_start(module_info.contents(), statement_range.start())?;
    let variable_name = generate_variable_name(module_info.contents());
    let assignment = format!("{statement_indent}{variable_name} = {expression_text}\n");
    let replacement_text = format!("{leading_ws}{variable_name}{trailing_ws}");
    let insert_edit = (
        module_info.dupe(),
        TextRange::at(insert_position, TextSize::new(0)),
        assignment,
    );
    let replace_edit = (module_info.dupe(), selection, replacement_text);
    let action = LocalRefactorCodeAction {
        title: format!("Extract into variable `{variable_name}`"),
        edits: vec![insert_edit, replace_edit],
        kind: CodeActionKind::REFACTOR_EXTRACT,
    };
    Some(vec![action])
}

fn split_selection<'a>(
    selection_text: &'a str,
    selection_range: TextRange,
) -> Option<(&'a str, &'a str, &'a str, TextRange)> {
    let trimmed_start = selection_text.trim_start_matches(char::is_whitespace);
    let leading_len = selection_text.len() - trimmed_start.len();
    let trimmed = trimmed_start.trim_end_matches(char::is_whitespace);
    let trailing_len = trimmed_start.len() - trimmed.len();
    if trimmed.is_empty() || trimmed.contains('\n') {
        return None;
    }
    let leading_ws = &selection_text[..leading_len];
    let trailing_ws = &selection_text[selection_text.len() - trailing_len..];
    let leading_size = TextSize::try_from(leading_len).ok()?;
    let trailing_size = TextSize::try_from(trailing_len).ok()?;
    let expr_start = selection_range.start() + leading_size;
    let expr_end = selection_range.end() - trailing_size;
    if expr_start >= expr_end {
        return None;
    }
    Some((
        leading_ws,
        trimmed,
        trailing_ws,
        TextRange::new(expr_start, expr_end),
    ))
}

fn is_exact_expression(ast: &ModModule, selection: TextRange) -> bool {
    Ast::locate_node(ast, selection.start())
        .into_iter()
        .any(|node| node.as_expr_ref().is_some() && node.range() == selection)
}

fn find_enclosing_statement_range(ast: &ModModule, selection: TextRange) -> Option<TextRange> {
    let covering_nodes = Ast::locate_node(ast, selection.start());
    for node in covering_nodes {
        if let Some(stmt) = node.as_stmt_ref()
            && stmt.range().contains_range(selection)
        {
            return Some(stmt.range());
        }
    }
    None
}

fn line_indent_and_start(source: &str, position: TextSize) -> Option<(String, TextSize)> {
    let mut idx = position.to_usize();
    if idx > source.len() {
        idx = source.len();
    }
    let line_start = source[..idx]
        .rfind('\n')
        .map(|start| start + 1)
        .unwrap_or(0);
    let indent = source[line_start..idx]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let insert_position = TextSize::try_from(line_start).ok()?;
    Some((indent, insert_position))
}

fn generate_variable_name(source: &str) -> String {
    let mut counter = 1;
    loop {
        let candidate = if counter == 1 {
            DEFAULT_VARIABLE_PREFIX.to_owned()
        } else {
            format!("{DEFAULT_VARIABLE_PREFIX}_{counter}")
        };
        let check_space = format!("{candidate} =");
        let check_tab = format!("{candidate}\t=");
        if !source.contains(&check_space) && !source.contains(&check_tab) {
            return candidate;
        }
        counter += 1;
    }
}
