/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::state::lsp::quick_fixes::extract_shared::line_indent_and_start;
use crate::state::lsp::quick_fixes::extract_shared::needs_pass_after_removal;
use crate::state::lsp::quick_fixes::extract_shared::statement_removal_range;

pub(crate) fn remove_unused_variable_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    unused_name: &str,
    unused_range: TextRange,
) -> Option<(String, Module, TextRange, String)> {
    let context = find_unused_assignment_context(&ast.body, unused_range)?;
    let removal_range = statement_removal_range(module_info.contents(), context.stmt)?;
    let replacement = if needs_pass_after_removal(context.parent_body, context.stmt.range()) {
        let (indent, _) =
            line_indent_and_start(module_info.contents(), context.stmt.range().start())?;
        format!("{indent}pass\n")
    } else {
        String::new()
    };
    Some((
        format!("Remove unused variable `{unused_name}`"),
        module_info.dupe(),
        removal_range,
        replacement,
    ))
}

struct UnusedAssignmentContext<'a> {
    stmt: &'a Stmt,
    parent_body: &'a [Stmt],
}

fn find_unused_assignment_context(
    body: &[Stmt],
    unused_range: TextRange,
) -> Option<UnusedAssignmentContext<'_>> {
    for stmt in body {
        if removable_assignment_range(stmt) == Some(unused_range) {
            return Some(UnusedAssignmentContext {
                stmt,
                parent_body: body,
            });
        }
        if let Some(context) = find_unused_assignment_context_in_child_body(stmt, unused_range) {
            return Some(context);
        }
    }
    None
}

fn find_unused_assignment_context_in_child_body(
    stmt: &Stmt,
    unused_range: TextRange,
) -> Option<UnusedAssignmentContext<'_>> {
    match stmt {
        Stmt::FunctionDef(function_def) => {
            find_unused_assignment_context(&function_def.body, unused_range)
        }
        Stmt::ClassDef(class_def) => find_unused_assignment_context(&class_def.body, unused_range),
        Stmt::For(for_stmt) => find_unused_assignment_context(&for_stmt.body, unused_range)
            .or_else(|| find_unused_assignment_context(&for_stmt.orelse, unused_range)),
        Stmt::While(while_stmt) => find_unused_assignment_context(&while_stmt.body, unused_range)
            .or_else(|| find_unused_assignment_context(&while_stmt.orelse, unused_range)),
        Stmt::If(if_stmt) => {
            find_unused_assignment_context(&if_stmt.body, unused_range).or_else(|| {
                if_stmt
                    .elif_else_clauses
                    .iter()
                    .find_map(|clause| find_unused_assignment_context(&clause.body, unused_range))
            })
        }
        Stmt::With(with_stmt) => find_unused_assignment_context(&with_stmt.body, unused_range),
        Stmt::Match(match_stmt) => match_stmt
            .cases
            .iter()
            .find_map(|case| find_unused_assignment_context(&case.body, unused_range)),
        Stmt::Try(try_stmt) => find_unused_assignment_context(&try_stmt.body, unused_range)
            .or_else(|| {
                try_stmt.handlers.iter().find_map(|handler| {
                    let ExceptHandler::ExceptHandler(handler) = handler;
                    find_unused_assignment_context(&handler.body, unused_range)
                })
            })
            .or_else(|| find_unused_assignment_context(&try_stmt.orelse, unused_range))
            .or_else(|| find_unused_assignment_context(&try_stmt.finalbody, unused_range)),
        _ => None,
    }
}

fn removable_assignment_range(stmt: &Stmt) -> Option<TextRange> {
    match stmt {
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            simple_name_range(&assign.targets[0])
        }
        Stmt::AnnAssign(assign) => simple_name_range(assign.target.as_ref()),
        _ => None,
    }
}

fn simple_name_range(expr: &Expr) -> Option<TextRange> {
    match expr {
        Expr::Name(name) => Some(name.range()),
        _ => None,
    }
}
