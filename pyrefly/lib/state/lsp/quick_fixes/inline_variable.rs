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
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::reference_in_disallowed_scope;

pub(crate) fn inline_variable_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let position = selection.start();
    let identifier = transaction.identifier_at(handle, position)?;
    if !matches!(identifier.context, IdentifierContext::Expr(_)) {
        return None;
    }
    let module_info = transaction.get_module_info(handle)?;
    let defs = transaction.find_definition(handle, position, FindPreference::default());
    let def = defs.into_iter().find(|def| {
        def.module.path() == module_info.path()
            && matches!(
                def.metadata.symbol_kind(),
                Some(SymbolKind::Variable | SymbolKind::Constant)
            )
    })?;
    let ast = transaction.get_ast(handle)?;
    let (assignment_range, value_expr) =
        find_simple_assignment(ast.as_ref(), def.definition_range)?;
    if value_expr_contains_name(value_expr, identifier.identifier.id.as_str()) {
        return None;
    }
    if has_other_store(
        ast.as_ref(),
        identifier.identifier.id.as_str(),
        def.definition_range,
    ) {
        return None;
    }
    let references = transaction.find_local_references(handle, def.definition_range.start());
    if references.is_empty() {
        return None;
    }
    if references
        .iter()
        .any(|range| range.start() < assignment_range.start())
    {
        return None;
    }
    if references.iter().any(|range| {
        *range != def.definition_range && reference_in_disallowed_scope(ast.as_ref(), *range)
    }) {
        return None;
    }
    let scope_range = enclosing_scope_range(
        ast.as_ref(),
        def.definition_range,
        module_info.contents().len(),
    )?;
    if references_in_nested_scope(ast.as_ref(), &references, scope_range, def.definition_range) {
        return None;
    }
    let value_text = module_info.code_at(value_expr.range());
    if value_text.trim().is_empty() {
        return None;
    }
    if value_text.contains('\n') {
        return None;
    }
    let replacement = format!("({value_text})");
    let mut edits = Vec::new();
    for range in references {
        if range == def.definition_range {
            continue;
        }
        edits.push((module_info.dupe(), range, replacement.clone()));
    }
    if edits.is_empty() {
        return None;
    }
    let remove_range = expand_statement_removal_range(module_info.contents(), assignment_range);
    edits.push((module_info.dupe(), remove_range, String::new()));
    Some(vec![LocalRefactorCodeAction {
        title: format!("Inline variable `{}`", identifier.identifier.id),
        edits,
        kind: CodeActionKind::REFACTOR_INLINE,
    }])
}

fn find_simple_assignment(
    ast: &ModModule,
    definition_range: TextRange,
) -> Option<(TextRange, &Expr)> {
    fn find_in_stmts<'a>(
        stmts: &'a [Stmt],
        definition_range: TextRange,
    ) -> Option<(TextRange, &'a Expr)> {
        for stmt in stmts {
            match stmt {
                Stmt::Assign(assign) => {
                    if assign.targets.len() != 1 {
                        continue;
                    }
                    if let Expr::Name(name) = &assign.targets[0]
                        && name.range() == definition_range
                    {
                        return Some((stmt.range(), assign.value.as_ref()));
                    }
                }
                Stmt::FunctionDef(function_def) => {
                    if let Some(found) = find_in_stmts(&function_def.body, definition_range) {
                        return Some(found);
                    }
                }
                Stmt::ClassDef(class_def) => {
                    if let Some(found) = find_in_stmts(&class_def.body, definition_range) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }

    find_in_stmts(&ast.body, definition_range)
}

fn value_expr_contains_name(expr: &Expr, name: &str) -> bool {
    let mut found = false;
    expr.visit(&mut |node| {
        if let Expr::Name(expr_name) = node
            && expr_name.id.as_str() == name
            && matches!(expr_name.ctx, ExprContext::Load)
        {
            found = true;
        }
    });
    found
}

fn has_other_store(ast: &ModModule, name: &str, definition_range: TextRange) -> bool {
    let mut found = false;
    ast.visit(&mut |node: &Expr| {
        if let Expr::Name(expr_name) = node
            && expr_name.id.as_str() == name
            && matches!(expr_name.ctx, ExprContext::Store)
            && expr_name.range() != definition_range
        {
            found = true;
        }
    });
    found
}

fn enclosing_scope_range(
    ast: &ModModule,
    definition_range: TextRange,
    module_len: usize,
) -> Option<TextRange> {
    let covering_nodes = Ast::locate_node(ast, definition_range.start());
    for node in covering_nodes {
        if let Some(function_def) = node.as_stmt_function_def()
            && function_def.range().contains_range(definition_range)
        {
            return Some(function_def.range());
        }
        if let Some(class_def) = node.as_stmt_class_def()
            && class_def.range().contains_range(definition_range)
        {
            return Some(class_def.range());
        }
    }
    let module_len = TextSize::try_from(module_len).ok()?;
    Some(TextRange::new(TextSize::new(0), module_len))
}

fn references_in_nested_scope(
    ast: &ModModule,
    references: &[TextRange],
    scope_range: TextRange,
    definition_range: TextRange,
) -> bool {
    let mut nested_ranges = Vec::new();
    collect_nested_scopes(&ast.body, scope_range, &mut nested_ranges);
    references
        .iter()
        .filter(|r| **r != definition_range)
        .any(|r| nested_ranges.iter().any(|nested| nested.contains_range(*r)))
}

fn collect_nested_scopes(stmts: &[Stmt], scope_range: TextRange, ranges: &mut Vec<TextRange>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if scope_range.contains_range(function_def.range()) {
                    if function_def.range() != scope_range {
                        ranges.push(function_def.range());
                    }
                    collect_nested_scopes(&function_def.body, scope_range, ranges);
                }
            }
            Stmt::ClassDef(class_def) => {
                if scope_range.contains_range(class_def.range()) {
                    if class_def.range() != scope_range {
                        ranges.push(class_def.range());
                    }
                    collect_nested_scopes(&class_def.body, scope_range, ranges);
                }
            }
            _ => {}
        }
    }
}

/// Expands a statement range to include surrounding whitespace and newlines for clean removal.
///
/// When removing a statement, we want to delete the entire line if the statement is alone on it,
/// including leading indentation and the trailing newline. This prevents leaving behind blank
/// lines or orphaned whitespace after the removal.
///
/// The function handles three cases:
/// 1. Statement alone on a line: expands to include leading whitespace and trailing newline
/// 2. Statement at end of file with no trailing newline: expands backward to consume preceding whitespace
/// 3. Statement with other content on the same line: returns the original range unchanged
fn expand_statement_removal_range(source: &str, stmt_range: TextRange) -> TextRange {
    let mut start = stmt_range.start().to_usize();
    let mut end = stmt_range.end().to_usize();
    let bytes = source.as_bytes();
    if start <= source.len() {
        let line_start = source[..start].rfind('\n').map(|idx| idx + 1).unwrap_or(0);
        if source[line_start..start]
            .chars()
            .all(|ch| ch == ' ' || ch == '\t')
        {
            start = line_start;
        }
    }
    if end < bytes.len() {
        if bytes[end] == b'\r' && end + 1 < bytes.len() && bytes[end + 1] == b'\n' {
            end += 2;
        } else if bytes[end] == b'\n' || bytes[end] == b'\r' {
            end += 1;
        }
    }
    if end == stmt_range.end().to_usize() {
        while start > 0 && bytes[start - 1].is_ascii_whitespace() && bytes[start - 1] != b'\n' {
            start -= 1;
        }
        if start > 0 && bytes[start - 1] == b'\n' {
            start -= 1;
        }
    }
    let start = TextSize::try_from(start).unwrap_or(stmt_range.start());
    let end = TextSize::try_from(end).unwrap_or(stmt_range.end());
    TextRange::new(start, end)
}
