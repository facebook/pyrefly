/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_statement_range;
use crate::state::lsp::quick_fixes::extract_shared::is_exact_expression;
use crate::state::lsp::quick_fixes::extract_shared::line_indent_and_start;
use crate::state::lsp::quick_fixes::extract_shared::split_selection;
use crate::state::lsp::quick_fixes::extract_shared::unique_name;
use crate::state::lsp::quick_fixes::extract_shared::validate_non_empty_selection;

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
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let selection_text = validate_non_empty_selection(selection, module_info.code_at(selection))?;
    let (leading_ws, expression_text, trailing_ws, expression_range) =
        split_selection(selection_text, selection)?;
    if !is_exact_expression(ast.as_ref(), expression_range) {
        return None;
    }
    let statement_range = find_enclosing_statement_range(ast.as_ref(), expression_range)?;
    let (statement_indent, insert_position) =
        line_indent_and_start(module_info.contents(), statement_range.start())?;
    let source = module_info.contents();
    let variable_name = unique_name(DEFAULT_VARIABLE_PREFIX, |name| {
        let check_space = format!("{name} =");
        let check_tab = format!("{name}\t=");
        source.contains(&check_space) || source.contains(&check_tab)
    });
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
