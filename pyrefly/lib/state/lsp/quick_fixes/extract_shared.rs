/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

pub(super) fn split_selection<'a>(
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

pub(super) fn is_exact_expression(ast: &ModModule, selection: TextRange) -> bool {
    Ast::locate_node(ast, selection.start())
        .into_iter()
        .any(|node| node.as_expr_ref().is_some() && node.range() == selection)
}

pub(super) fn line_indent_and_start(
    source: &str,
    position: TextSize,
) -> Option<(String, TextSize)> {
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

pub(super) fn first_parameter_name(parameters: &Parameters) -> Option<String> {
    if let Some(param) = parameters.posonlyargs.first() {
        return Some(param.name().id.to_string());
    }
    parameters
        .args
        .first()
        .map(|param| param.name().id.to_string())
}

pub(super) fn function_has_decorator(function_def: &StmtFunctionDef, decorator: &str) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|d| decorator_matches_name(&d.expression, decorator))
}

pub(super) fn decorator_matches_name(decorator: &Expr, expected: &str) -> bool {
    match decorator {
        Expr::Name(identifier) => identifier.id.as_str() == expected,
        Expr::Attribute(attribute) => attribute.attr.as_str() == expected,
        Expr::Call(call) => decorator_matches_name(call.func.as_ref(), expected),
        _ => false,
    }
}

/// Given a selection range, returns the first non-whitespace position within it.
/// If the selection is empty, returns the start position.
pub(super) fn selection_anchor(source: &str, selection: TextRange) -> TextSize {
    if selection.is_empty() {
        return selection.start();
    }
    let start = selection.start().to_usize().min(source.len());
    let end = selection.end().to_usize().min(source.len());
    if start >= end {
        return selection.start();
    }
    if let Some(offset) = source[start..end]
        .char_indices()
        .find(|(_, ch)| !matches!(ch, ' ' | '\t' | '\n' | '\r'))
        .map(|(idx, _)| idx)
    {
        TextSize::try_from(start + offset).unwrap_or(selection.start())
    } else {
        selection.start()
    }
}

/// Extracts the name from a statement that defines a named symbol.
/// Returns `None` for statements that don't define a single named symbol.
pub(super) fn member_name_from_stmt(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::FunctionDef(func_def) => Some(func_def.name.id.to_string()),
        Stmt::ClassDef(class_def) => Some(class_def.name.id.to_string()),
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            if let Expr::Name(name) = &assign.targets[0] {
                Some(name.id.to_string())
            } else {
                None
            }
        }
        Stmt::AnnAssign(assign) => {
            if let Expr::Name(name) = assign.target.as_ref() {
                Some(name.id.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}
