/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use serde::Serialize;
use starlark_map::Hashed;

use crate::binding::binding::Key;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::types::PysaType;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GlobalVariable {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<PysaType>,
    pub location: PysaLocation,
}

fn visit_assign_target(
    target: &Expr,
    global_variables: &mut HashMap<Name, GlobalVariable>,
    context: &ModuleContext,
) {
    Ast::expr_lvalue(target, &mut |global: &ExprName| {
        let type_ = context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&Key::Definition(ShortIdentifier::expr_name(
                global,
            ))))
            .and_then(|idx| context.answers.get_idx(idx));
        if let Some(type_) = type_.as_ref()
            && type_.ty().is_type_variable()
        {
            // Don't export type variable globals.
            return;
        }
        let location = PysaLocation::new(context.module_info.display_range(global.range()));
        global_variables
            .entry(global.id.clone())
            .or_insert(GlobalVariable {
                type_: type_.map(|type_| PysaType::from_type(type_.ty(), context)),
                location,
            });
    });
}

fn visit_statement(
    stmt: &Stmt,
    global_variables: &mut HashMap<Name, GlobalVariable>,
    context: &ModuleContext,
) {
    match stmt {
        Stmt::Assign(assign) => {
            for t in &assign.targets {
                visit_assign_target(t, global_variables, context);
            }
        }
        Stmt::AnnAssign(assign) => {
            visit_assign_target(&assign.target, global_variables, context);
        }
        Stmt::AugAssign(assign) => {
            visit_assign_target(&assign.target, global_variables, context);
        }
        Stmt::For(for_stmt) => {
            visit_statements(for_stmt.body.iter(), global_variables, context);
            visit_statements(for_stmt.orelse.iter(), global_variables, context);
        }
        Stmt::While(while_stmt) => {
            visit_statements(while_stmt.body.iter(), global_variables, context);
            visit_statements(while_stmt.orelse.iter(), global_variables, context);
        }
        Stmt::If(if_stmt) => {
            visit_statements(if_stmt.body.iter(), global_variables, context);
            for elif_else_clause in &if_stmt.elif_else_clauses {
                visit_statements(elif_else_clause.body.iter(), global_variables, context);
            }
        }
        Stmt::With(with_stmt) => {
            visit_statements(with_stmt.body.iter(), global_variables, context);
        }
        Stmt::Match(match_stmt) => {
            for case in &match_stmt.cases {
                visit_statements(case.body.iter(), global_variables, context);
            }
        }
        Stmt::Try(try_stmt) => {
            visit_statements(try_stmt.body.iter(), global_variables, context);
            visit_statements(try_stmt.orelse.iter(), global_variables, context);
            visit_statements(try_stmt.finalbody.iter(), global_variables, context);
            for ruff_python_ast::ExceptHandler::ExceptHandler(except_handler) in &try_stmt.handlers
            {
                visit_statements(except_handler.body.iter(), global_variables, context);
            }
        }
        Stmt::FunctionDef(_)
        | Stmt::ClassDef(_)
        | Stmt::Expr(_)
        | Stmt::Return(_)
        | Stmt::Delete(_)
        | Stmt::Raise(_)
        | Stmt::Assert(_)
        | Stmt::TypeAlias(_)
        | Stmt::Import(_)
        | Stmt::ImportFrom(_)
        | Stmt::Global(_)
        | Stmt::Nonlocal(_)
        | Stmt::Pass(_)
        | Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::IpyEscapeCommand(_) => {
            // do nothing.
        }
    }
}

fn visit_statements<'a>(
    statements: impl Iterator<Item = &'a Stmt>,
    global_variables: &mut HashMap<Name, GlobalVariable>,
    context: &ModuleContext,
) {
    for stmt in statements {
        visit_statement(stmt, global_variables, context);
    }
}

pub fn export_global_variables(context: &ModuleContext) -> HashMap<Name, GlobalVariable> {
    let mut global_variables = HashMap::new();
    visit_statements(context.ast.body.iter(), &mut global_variables, context);
    global_variables
}
