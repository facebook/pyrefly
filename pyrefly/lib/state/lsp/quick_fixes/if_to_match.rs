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
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

/// Builds an if/elif-to-match refactor for simple same-subject narrowing chains.
pub(crate) fn if_to_match_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let if_stmt = find_if_at_selection(ast.as_ref(), selection_point)?;
    let replacement = build_match_replacement(source, if_stmt)?;
    let (_, line_start) = line_indent_and_start(source, if_stmt.range().start())?;
    let replace_range =
        TextRange::new(line_start, line_end_position(source, if_stmt.range().end()));

    Some(vec![LocalRefactorCodeAction {
        title: "Convert if/elif chain to match".to_owned(),
        edits: vec![(module_info.dupe(), replace_range, replacement)],
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

fn find_if_at_selection(ast: &ModModule, selection_point: TextSize) -> Option<&StmtIf> {
    Ast::locate_node(ast, selection_point)
        .into_iter()
        .find_map(|node| match node {
            AnyNodeRef::StmtIf(if_stmt) => Some(if_stmt),
            _ => None,
        })
}

fn build_match_replacement(source: &str, if_stmt: &StmtIf) -> Option<String> {
    let (base_indent, _) = line_indent_and_start(source, if_stmt.range().start())?;
    let body_indent = branch_body_indent(source, &if_stmt.body, &base_indent)?;
    let mut subject = None;
    let mut cases = Vec::new();
    let if_end = line_end_position(source, if_stmt.range().end());
    let first_body_end = if_stmt
        .elif_else_clauses
        .first()
        .and_then(|clause| line_start_position(source, clause.range.start()))
        .unwrap_or(if_end);
    cases.push(branch_case(
        source,
        &if_stmt.test,
        TextRange::new(
            line_end_position(source, if_stmt.test.range().end()),
            first_body_end,
        ),
        &mut subject,
    )?);
    for (index, clause) in if_stmt.elif_else_clauses.iter().enumerate() {
        let body_end = if_stmt
            .elif_else_clauses
            .get(index + 1)
            .and_then(|next_clause| line_start_position(source, next_clause.range.start()))
            .unwrap_or(if_end);
        match &clause.test {
            Some(test) => cases.push(branch_case(
                source,
                test,
                TextRange::new(line_end_position(source, test.range().end()), body_end),
                &mut subject,
            )?),
            None => cases.push(else_case(
                source,
                TextRange::new(line_end_position(source, clause.range.start()), body_end),
            )?),
        }
    }
    if cases.iter().filter(|case| case.pattern != "_").count() < 2 {
        return None;
    }

    let subject = subject?;
    let case_indent = format!("{base_indent}{body_indent}");
    let mut replacement = format!("{base_indent}match {subject}:\n");
    for case in cases {
        replacement.push_str(&case_indent);
        replacement.push_str("case ");
        replacement.push_str(&case.pattern);
        replacement.push_str(":\n");
        replacement.push_str(&indent_branch_body(&case.body_text, &body_indent));
    }
    Some(replacement)
}

struct MatchCaseText {
    pattern: String,
    body_text: String,
}

fn branch_case(
    source: &str,
    test: &Expr,
    body_range: TextRange,
    subject: &mut Option<String>,
) -> Option<MatchCaseText> {
    let (candidate_subject, pattern) = case_subject_and_pattern(source, test)?;
    match subject {
        Some(subject) if subject != &candidate_subject => return None,
        Some(_) => {}
        None => *subject = Some(candidate_subject),
    }
    Some(MatchCaseText {
        pattern,
        body_text: source_text(source, body_range).to_owned(),
    })
}

fn else_case(source: &str, body_range: TextRange) -> Option<MatchCaseText> {
    Some(MatchCaseText {
        pattern: "_".to_owned(),
        body_text: source_text(source, body_range).to_owned(),
    })
}

fn case_subject_and_pattern(source: &str, test: &Expr) -> Option<(String, String)> {
    if let Expr::Compare(compare) = test
        && compare.ops.as_ref() == [CmpOp::Eq]
    {
        let subject = expr_text(source, &compare.left);
        let pattern = value_pattern(source, compare.comparators.first()?)?;
        return Some((subject, pattern));
    }
    if let Expr::Call(call) = test
        && let Expr::Name(func) = call.func.as_ref()
        && func.id.as_str() == "isinstance"
        && call.arguments.args.len() == 2
        && call.arguments.keywords.is_empty()
    {
        let subject = expr_text(source, &call.arguments.args[0]);
        let class_name = class_pattern_name(source, &call.arguments.args[1])?;
        return Some((subject, format!("{class_name}()")));
    }
    None
}

fn value_pattern(source: &str, value: &Expr) -> Option<String> {
    match value {
        Expr::StringLiteral(_)
        | Expr::BytesLiteral(_)
        | Expr::NumberLiteral(_)
        | Expr::BooleanLiteral(_)
        | Expr::NoneLiteral(_)
        | Expr::Attribute(_) => Some(expr_text(source, value)),
        _ => None,
    }
}

fn class_pattern_name(source: &str, value: &Expr) -> Option<String> {
    match value {
        Expr::Name(_) | Expr::Attribute(_) => Some(expr_text(source, value)),
        _ => None,
    }
}

fn expr_text(source: &str, expr: &Expr) -> String {
    source_text(source, expr.range()).trim().to_owned()
}

fn branch_body_indent(source: &str, body: &[Stmt], base_indent: &str) -> Option<String> {
    let (indent, _) = line_indent_and_start(source, body.first()?.range().start())?;
    Some(
        indent
            .strip_prefix(base_indent)
            .filter(|indent| !indent.is_empty())
            .unwrap_or("    ")
            .to_owned(),
    )
}

fn indent_branch_body(body: &str, extra_indent: &str) -> String {
    let mut result = String::new();
    for line in body.split_inclusive('\n') {
        if line.trim().is_empty() {
            result.push_str(line);
        } else {
            result.push_str(extra_indent);
            result.push_str(line);
        }
    }
    if !body.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn source_text(source: &str, range: TextRange) -> &str {
    &source[range.start().to_usize().min(source.len())..range.end().to_usize().min(source.len())]
}

fn line_end_position(source: &str, position: TextSize) -> TextSize {
    let idx = position.to_usize().min(source.len());
    if let Some(offset) = source[idx..].find('\n') {
        TextSize::try_from(idx + offset + 1).unwrap_or(position)
    } else {
        TextSize::try_from(source.len()).unwrap_or(position)
    }
}

fn line_start_position(source: &str, position: TextSize) -> Option<TextSize> {
    line_indent_and_start(source, position).map(|(_, start)| start)
}
