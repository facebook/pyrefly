/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::code_at_range;
use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::extract_shared::statement_removal_range_from_range;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

struct UnwrapBlockTarget {
    outer_range: TextRange,
    body_range: TextRange,
    from_indent: String,
    to_indent: String,
}

#[derive(Clone, Copy, Debug)]
struct QuoteState {
    delimiter: char,
    triple: bool,
}

/// Builds unwrap-block refactor actions for the block under the cursor.
pub(crate) fn unwrap_block_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let position = selection_anchor(source, selection);
    let target = find_unwrap_block_target_in_body(&ast.body, source, position)?;
    let removal_range = statement_removal_range_from_range(source, target.outer_range)?;
    let replacement = reindent_block(
        code_at_range(source, target.body_range)?,
        &target.from_indent,
        &target.to_indent,
    );
    Some(vec![LocalRefactorCodeAction {
        title: "Unwrap block".to_owned(),
        edits: vec![(module_info.dupe(), removal_range, replacement)],
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

fn find_unwrap_block_target_in_body(
    body: &[Stmt],
    source: &str,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    for stmt in body {
        if !range_contains_position(stmt.range(), position) {
            continue;
        }
        if let Some(target) = find_unwrap_block_target_in_stmt(stmt, source, position) {
            return Some(target);
        }
    }
    None
}

fn find_unwrap_block_target_in_stmt(
    stmt: &Stmt,
    source: &str,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    match stmt {
        Stmt::FunctionDef(function_def) => {
            find_unwrap_block_target_in_body(&function_def.body, source, position)
        }
        Stmt::ClassDef(class_def) => {
            find_unwrap_block_target_in_body(&class_def.body, source, position)
        }
        Stmt::If(if_stmt) => find_unwrap_if_target(if_stmt, source, position),
        Stmt::For(for_stmt) => {
            if let Some(target) = find_unwrap_block_target_in_body(&for_stmt.body, source, position)
            {
                return Some(target);
            }
            if !for_stmt.orelse.is_empty() {
                return None;
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &for_stmt.body,
                None,
                source,
                position,
            )
        }
        Stmt::While(while_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&while_stmt.body, source, position)
            {
                return Some(target);
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&while_stmt.orelse, source, position)
            {
                return Some(target);
            }
            if !while_stmt.orelse.is_empty() {
                return None;
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &while_stmt.body,
                None,
                source,
                position,
            )
        }
        Stmt::With(with_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&with_stmt.body, source, position)
            {
                return Some(target);
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &with_stmt.body,
                None,
                source,
                position,
            )
        }
        Stmt::Try(try_stmt) => {
            let (to_indent, _) = line_indent_and_start(source, stmt.range().start())?;
            if let Some(target) = find_unwrap_block_target_in_body(&try_stmt.body, source, position)
            {
                return Some(target);
            }
            for handler in &try_stmt.handlers {
                if let Some(handler) = handler.as_except_handler()
                    && let Some(target) =
                        find_unwrap_block_target_in_body(&handler.body, source, position)
                {
                    return Some(target);
                }
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&try_stmt.orelse, source, position)
            {
                return Some(target);
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&try_stmt.finalbody, source, position)
            {
                return Some(target);
            }
            for (index, handler) in try_stmt.handlers.iter().enumerate() {
                let Some(handler) = handler.as_except_handler() else {
                    continue;
                };
                let next_clause_start = try_stmt
                    .handlers
                    .get(index + 1)
                    .and_then(|next| next.as_except_handler())
                    .map(|next| next.range().start())
                    .or_else(|| {
                        try_stmt
                            .orelse
                            .first()
                            .map(Ranged::range)
                            .map(|range| range.start())
                    })
                    .or_else(|| {
                        try_stmt
                            .finalbody
                            .first()
                            .map(Ranged::range)
                            .map(|range| range.start())
                    });
                if let Some(target) = build_unwrap_block_target(
                    stmt.range(),
                    handler.range().start(),
                    &handler.body,
                    next_clause_start,
                    source,
                    position,
                ) {
                    return Some(target);
                }
            }
            let else_start = try_stmt.orelse.first().and_then(|stmt| {
                find_try_clause_start(source, stmt.range().start(), &to_indent, "else")
            });
            if let Some(else_start) = else_start
                && let Some(target) = build_unwrap_block_target(
                    stmt.range(),
                    else_start,
                    &try_stmt.orelse,
                    try_stmt.finalbody.first().and_then(|stmt| {
                        find_try_clause_start(source, stmt.range().start(), &to_indent, "finally")
                    }),
                    source,
                    position,
                )
            {
                return Some(target);
            }
            let finally_start = try_stmt.finalbody.first().and_then(|stmt| {
                find_try_clause_start(source, stmt.range().start(), &to_indent, "finally")
            });
            build_unwrap_block_target(
                stmt.range(),
                finally_start?,
                &try_stmt.finalbody,
                None,
                source,
                position,
            )
        }
        Stmt::Match(match_stmt) => {
            for case in &match_stmt.cases {
                if let Some(target) = find_unwrap_block_target_in_body(&case.body, source, position)
                {
                    return Some(target);
                }
            }
            None
        }
        _ => None,
    }
}

fn find_unwrap_if_target(
    if_stmt: &StmtIf,
    source: &str,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    if let Some(target) = find_unwrap_block_target_in_body(&if_stmt.body, source, position) {
        return Some(target);
    }
    for clause in &if_stmt.elif_else_clauses {
        if let Some(target) = find_unwrap_block_target_in_body(&clause.body, source, position) {
            return Some(target);
        }
    }

    let next_clause_start = if_stmt
        .elif_else_clauses
        .first()
        .map(|clause| clause.range.start());
    if let Some(target) = build_unwrap_block_target(
        if_stmt.range(),
        if_stmt.range().start(),
        &if_stmt.body,
        next_clause_start,
        source,
        position,
    ) {
        return Some(target);
    }

    for (index, clause) in if_stmt.elif_else_clauses.iter().enumerate() {
        let next_clause_start = if_stmt
            .elif_else_clauses
            .get(index + 1)
            .map(|next_clause| next_clause.range.start());
        if let Some(target) = build_unwrap_block_target(
            if_stmt.range(),
            clause.range.start(),
            &clause.body,
            next_clause_start,
            source,
            position,
        ) {
            return Some(target);
        }
    }

    None
}

fn build_unwrap_block_target(
    outer_range: TextRange,
    header_start: TextSize,
    body: &[Stmt],
    next_clause_start: Option<TextSize>,
    source: &str,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    let first_stmt = body.first()?;
    let (to_indent, _) = line_indent_and_start(source, outer_range.start())?;
    let (_, header_line_start) = line_indent_and_start(source, header_start)?;
    let (from_indent, first_stmt_line_start) =
        line_indent_and_start(source, first_stmt.range().start())?;
    let colon_position = find_suite_colon(source, header_line_start, first_stmt_line_start)?;
    if position < header_line_start || position > colon_position + TextSize::new(1) {
        return None;
    }
    let block_start = line_end_position(source, colon_position);
    if block_start > first_stmt.range().start() {
        return None;
    }
    let block_end = if let Some(next_clause_start) = next_clause_start {
        line_indent_and_start(source, next_clause_start)?.1
    } else {
        line_end_position(source, outer_range.end())
    };
    if block_start >= block_end {
        return None;
    }
    Some(UnwrapBlockTarget {
        outer_range,
        body_range: TextRange::new(block_start, block_end),
        from_indent,
        to_indent,
    })
}

fn find_suite_colon(
    source: &str,
    header_line_start: TextSize,
    body_line_start: TextSize,
) -> Option<TextSize> {
    let header_start = header_line_start.to_usize();
    let header_end = body_line_start.to_usize().min(source.len());
    if header_start >= header_end {
        return None;
    }
    let header = &source[header_start..header_end];
    let colon = find_suite_colon_in_header(header)?;
    TextSize::try_from(header_start + colon).ok()
}

fn find_suite_colon_in_header(header: &str) -> Option<usize> {
    let mut quote = None::<QuoteState>;
    let mut escaped = false;
    let mut in_comment = false;
    let mut last_colon = None;

    for (index, ch) in header.char_indices() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        if let Some(state) = quote {
            if state.triple {
                if ch == state.delimiter
                    && header[index..].starts_with(triple_quote_delimiter(state.delimiter))
                {
                    quote = None;
                } else if ch == '\\' {
                    escaped = true;
                }
            } else {
                match ch {
                    '\\' => escaped = true,
                    current if current == state.delimiter => quote = None,
                    _ => {}
                }
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(QuoteState {
                    delimiter: ch,
                    triple: header[index..].starts_with(triple_quote_delimiter(ch)),
                });
            }
            '#' => in_comment = true,
            ':' => last_colon = Some(index),
            _ => {}
        }
    }

    last_colon
}

fn triple_quote_delimiter(ch: char) -> &'static str {
    match ch {
        '\'' => "'''",
        '"' => "\"\"\"",
        _ => unreachable!("only quote delimiters are supported"),
    }
}

fn find_try_clause_start(
    source: &str,
    body_start: TextSize,
    outer_indent: &str,
    keyword: &str,
) -> Option<TextSize> {
    let mut line_end = body_start.to_usize().min(source.len());
    let expected_prefix = format!("{outer_indent}{keyword}");
    while line_end > 0 {
        let search_end = line_end.saturating_sub(1);
        let line_start = source[..search_end]
            .rfind('\n')
            .map(|index| index + 1)
            .unwrap_or(0);
        let line = source[line_start..line_end].trim_end_matches('\n');
        if line.starts_with(&expected_prefix) && find_suite_colon_in_header(line).is_some() {
            return TextSize::try_from(line_start).ok();
        }
        if line_start == 0 {
            break;
        }
        line_end = line_start;
    }
    None
}

fn range_contains_position(range: TextRange, position: TextSize) -> bool {
    range.start() <= position && position < range.end()
}
