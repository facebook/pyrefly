/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::types::Type;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
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
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::slow_fun_monitor::slow_fun_monitor_scope;
use crate::report::pysa::step_logger::StepLogger;
use crate::report::pysa::types::PysaType;
use crate::report::pysa::types::is_bound_method_like;
use crate::report::pysa::types::preprocess_type;
use crate::state::state::Transaction;

/// Represents information about a global variable, collected in a pre-analysis step.
/// See `GlobalVariable` for the type exported to Pysa. This only store memory-efficient information.
#[derive(Debug, Clone)]
pub struct GlobalVariableBase {
    pub type_: Option<Type>,
    pub name: Name,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GlobalVariable {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<PysaType>,
    pub location: PysaLocation,
}

/// A reference to a global variable.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalVariableRef {
    pub module_id: ModuleId,
    pub module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    pub name: Name,
}

#[derive(Debug, Clone)]
pub struct ModuleGlobalVariables(HashMap<ShortIdentifier, GlobalVariableBase>);

pub struct WholeProgramGlobalVariables(dashmap::ReadOnlyView<ModuleId, ModuleGlobalVariables>);

impl ModuleGlobalVariables {
    fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, short_identifier: ShortIdentifier) -> Option<&GlobalVariableBase> {
        self.0.get(&short_identifier)
    }
}

impl WholeProgramGlobalVariables {
    pub fn new() -> Self {
        WholeProgramGlobalVariables(dashmap::DashMap::new().into_read_only())
    }

    pub fn get_for_module(&self, module_id: ModuleId) -> Option<&ModuleGlobalVariables> {
        self.0.get(&module_id)
    }
}

impl GlobalVariable {
    fn from_base(
        identifier: ShortIdentifier,
        base: &GlobalVariableBase,
        context: &ModuleContext,
    ) -> Self {
        Self {
            type_: base
                .type_
                .as_ref()
                .map(|type_| PysaType::from_type(type_, context)),
            location: PysaLocation::new(context.module_info.display_range(identifier.range())),
        }
    }
}

fn visit_assign_target(
    target: &Expr,
    global_variables: &mut ModuleGlobalVariables,
    context: &ModuleContext,
) {
    Ast::expr_lvalue(target, &mut |global: &ExprName| {
        let short_identifier = ShortIdentifier::expr_name(global);
        let type_ = context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&Key::Definition(short_identifier)))
            .and_then(|idx| context.answers.get_idx(idx));
        if let Some(type_) = type_.as_ref()
            && (type_.ty().is_type_variable() || is_bound_method_like(type_.ty()))
        {
            // Don't export:
            // - Type variable, such as `T = TypeVar("T")`
            // - Aliases to bound method, such as `inst = Random(); random = inst.random`
            return;
        }
        global_variables
            .0
            .entry(short_identifier)
            .or_insert(GlobalVariableBase {
                type_: type_.map(|type_| preprocess_type(type_.ty(), context)),
                name: global.id.clone(),
            });
    });
}

fn visit_statement(
    stmt: &Stmt,
    global_variables: &mut ModuleGlobalVariables,
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
    global_variables: &mut ModuleGlobalVariables,
    context: &ModuleContext,
) {
    for stmt in statements {
        visit_statement(stmt, global_variables, context);
    }
}

pub fn collect_global_variables(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> WholeProgramGlobalVariables {
    let step = StepLogger::start("Indexing global variables", "Indexed global variables");

    let global_variables = dashmap::DashMap::new();

    ThreadPool::new().install(|| {
        slow_fun_monitor_scope(|slow_function_monitor| {
            handles.par_iter().for_each(|handle| {
                let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();
                let context =
                    ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                let globals_for_module = slow_function_monitor.monitor_function(
                    move || {
                        let mut global_variables = ModuleGlobalVariables::new();
                        visit_statements(context.ast.body.iter(), &mut global_variables, &context);
                        global_variables
                    },
                    format!("Indexing global variables for {}", handle.module().as_str(),),
                    /* max_time_in_seconds */ 4,
                );
                global_variables.insert(module_id, globals_for_module);
            });
        })
    });

    step.finish();
    WholeProgramGlobalVariables(global_variables.into_read_only())
}

pub fn export_global_variables(
    global_variables: &WholeProgramGlobalVariables,
    context: &ModuleContext,
) -> HashMap<Name, GlobalVariable> {
    let globals_for_module = global_variables.0.get(&context.module_id).unwrap();

    let mut global_variables = HashMap::new();
    for (short_identifier, global) in &globals_for_module.0 {
        let new_global = GlobalVariable::from_base(*short_identifier, global, context);
        global_variables
            .entry(global.name.clone())
            .and_modify(|existing_global: &mut GlobalVariable| {
                // Preserve the entry with the lowest location.
                if new_global.location < existing_global.location {
                    *existing_global = new_global.clone();
                }
            })
            .or_insert(new_global);
    }
    global_variables
}
