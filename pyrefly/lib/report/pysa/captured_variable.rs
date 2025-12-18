/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_build::handle::Handle;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use serde::Serialize;
use starlark_map::Hashed;
use starlark_map::small_set::SmallSet;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::ScopeExportedFunctionFlags;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::call_graph::FunctionTrait;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::slow_fun_monitor::slow_fun_monitor_scope;
use crate::report::pysa::step_logger::StepLogger;
use crate::state::state::Transaction;

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapturedVariable {
    pub name: Name,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapturedVariableRef<Function: FunctionTrait> {
    pub outer_function: Function,
    pub name: Name,
}

impl<Function: FunctionTrait> CapturedVariableRef<Function> {
    #[cfg(test)]
    pub fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> CapturedVariableRef<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        CapturedVariableRef {
            outer_function: map(self.outer_function),
            name: self.name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleCapturedVariables<Function: FunctionTrait>(
    HashMap<Function, HashMap<CapturedVariable, Function>>,
);

impl<Function: FunctionTrait> ModuleCapturedVariables<Function> {
    #[cfg(test)]
    pub fn into_iter(
        self,
    ) -> impl Iterator<Item = (Function, HashMap<CapturedVariable, Function>)> {
        self.0.into_iter()
    }

    pub fn get<'a>(
        &'a self,
        function_ref: &Function,
    ) -> Option<&'a HashMap<CapturedVariable, Function>> {
        self.0.get(function_ref)
    }
}

pub struct WholeProgramCapturedVariables(
    dashmap::ReadOnlyView<ModuleId, ModuleCapturedVariables<FunctionRef>>,
);

impl WholeProgramCapturedVariables {
    pub fn new() -> Self {
        WholeProgramCapturedVariables(dashmap::DashMap::new().into_read_only())
    }

    pub fn get_for_module(
        &self,
        module_id: ModuleId,
    ) -> Option<&ModuleCapturedVariables<FunctionRef>> {
        self.0.get(&module_id)
    }
}

static SCOPE_EXPORTED_FUNCTION_FLAGS: ScopeExportedFunctionFlags = ScopeExportedFunctionFlags {
    include_top_level: true,
    include_class_top_level: true,
    include_function_decorators: super::ast_visitor::ExportFunctionDecorators::InParentScope,
    include_class_decorators: super::ast_visitor::ExportClassDecorators::InParentScope,
    include_default_arguments: super::ast_visitor::ExportDefaultArguments::InParentScope,
};

struct DefinitionToFunctionMapVisitor<'a> {
    definition_to_function_map: &'a mut HashMap<Idx<Key>, FunctionRef>,
    module_context: &'a ModuleContext<'a>,
}

impl<'a> DefinitionToFunctionMapVisitor<'a> {
    fn bind_name(&mut self, key: Key, scopes: &Scopes) {
        if let Some(idx) = self
            .module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key))
            && let Some(current_function) = scopes.current_exported_function(
                self.module_context.module_id,
                self.module_context.module_info.name(),
                &SCOPE_EXPORTED_FUNCTION_FLAGS,
            )
        {
            assert!(
                self.definition_to_function_map
                    .insert(idx, current_function)
                    .is_none(),
                "Found multiple definitions for {:?}",
                &key,
            );
        }
    }

    fn bind_assign_target(&mut self, target: &Expr, scopes: &Scopes) {
        Ast::expr_lvalue(target, &mut |name: &ExprName| {
            self.bind_name(Key::Definition(ShortIdentifier::expr_name(name)), scopes);
        });
    }
}

impl<'a> AstScopedVisitor for DefinitionToFunctionMapVisitor<'a> {
    fn visit_statement(&mut self, stmt: &Stmt, scopes: &Scopes) {
        match stmt {
            Stmt::Assign(x) => {
                for target in &x.targets {
                    self.bind_assign_target(target, scopes);
                }
            }
            Stmt::AnnAssign(x) => self.bind_assign_target(&x.target, scopes),
            Stmt::AugAssign(x) => self.bind_assign_target(&x.target, scopes),
            _ => (),
        }
    }

    fn enter_function_scope(&mut self, function_def: &StmtFunctionDef, scopes: &Scopes) {
        for p in function_def.parameters.iter_non_variadic_params() {
            self.bind_name(
                Key::Definition(ShortIdentifier::new(&p.parameter.name)),
                scopes,
            );
        }
        if let Some(args) = &function_def.parameters.vararg {
            self.bind_name(Key::Definition(ShortIdentifier::new(&args.name)), scopes);
        }
        if let Some(kwargs) = &function_def.parameters.kwarg {
            self.bind_name(Key::Definition(ShortIdentifier::new(&kwargs.name)), scopes);
        }
    }

    fn visit_type_annotations() -> bool {
        false
    }
}

fn build_definition_to_function_map(context: &ModuleContext) -> HashMap<Idx<Key>, FunctionRef> {
    let mut definition_to_function_map = HashMap::new();
    let mut visitor = DefinitionToFunctionMapVisitor {
        definition_to_function_map: &mut definition_to_function_map,
        module_context: context,
    };
    visit_module_ast(&mut visitor, context);
    definition_to_function_map
}

struct CapturedVariableVisitor<'a> {
    // Map from a captured variable to the function that defines it, if any.
    captured_variables: &'a mut HashMap<FunctionRef, HashMap<CapturedVariable, FunctionRef>>,
    definition_to_function_map: &'a HashMap<Idx<Key>, FunctionRef>,
    module_context: &'a ModuleContext<'a>,
    current_exported_function: Option<FunctionRef>,
}

impl<'a> CapturedVariableVisitor<'a> {
    fn check_capture(&mut self, key: Key, name: &Name) {
        if let Some(definition) = self.get_definition_from_usage(key)
            && let Some(current_function) = &self.current_exported_function
            && definition != *current_function
        {
            self.captured_variables
                .entry(current_function.clone())
                .or_default()
                .insert(CapturedVariable { name: name.clone() }, definition);
        }
    }

    fn get_definition_from_usage(&self, key: Key) -> Option<FunctionRef> {
        let idx = self
            .module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key))?;
        let binding = self.module_context.bindings.get(idx);
        match binding {
            Binding::Forward(definition_idx) => {
                self.get_definition_from_idx(
                    *definition_idx,
                    /* seen */ SmallSet::new(),
                    /* depth */ 0,
                )
            }
            _ => None,
        }
    }

    fn get_definition_from_idx(
        &self,
        idx: Idx<Key>,
        mut seen: SmallSet<Idx<Key>>,
        mut depth: u32,
    ) -> Option<FunctionRef> {
        if let Some(function_ref) = self.definition_to_function_map.get(&idx) {
            return Some(function_ref.clone());
        }

        // Avoid cycles in bindings.
        if seen.contains(&idx) {
            return None;
        }
        seen.insert(idx);

        // Avoid a bottleneck with very deep AST nodes.
        if depth >= 10 {
            return None;
        }
        depth += 1;

        let binding = self.module_context.bindings.get(idx);
        match binding {
            Binding::Forward(idx) => self.get_definition_from_idx(*idx, seen, depth),
            Binding::Phi(_, elements) => {
                for idx in elements {
                    if let Some(function_ref) =
                        self.get_definition_from_idx(*idx, seen.clone(), depth)
                    {
                        return Some(function_ref);
                    }
                }
                None
            }
            Binding::CompletedPartialType(idx, _) => {
                self.get_definition_from_idx(*idx, seen, depth)
            }
            _ => None,
        }
    }
}

impl<'a> AstScopedVisitor for CapturedVariableVisitor<'a> {
    fn on_scope_update(&mut self, scopes: &Scopes) {
        self.current_exported_function = scopes.current_exported_function(
            self.module_context.module_id,
            self.module_context.module_info.name(),
            &SCOPE_EXPORTED_FUNCTION_FLAGS,
        );
    }

    fn visit_expression(
        &mut self,
        expr: &Expr,
        _scopes: &Scopes,
        _parent_expression: Option<&Expr>,
        _current_statement: Option<&Stmt>,
    ) {
        if self.current_exported_function.is_none() {
            return;
        }

        match expr {
            Expr::Name(x) => {
                self.check_capture(Key::BoundName(ShortIdentifier::expr_name(x)), x.id());
            }
            _ => (),
        }
    }

    fn visit_statement(&mut self, stmt: &Stmt, _scopes: &Scopes) {
        if self.current_exported_function.is_none() {
            return;
        }

        match stmt {
            Stmt::Nonlocal(nonlocal) => {
                for identifier in &nonlocal.names {
                    self.check_capture(
                        Key::MutableCapture(ShortIdentifier::new(identifier)),
                        identifier.id(),
                    );
                }
            }
            _ => (),
        }
    }

    fn visit_type_annotations() -> bool {
        false
    }
}

pub fn collect_captured_variables_for_module(
    context: &ModuleContext,
) -> ModuleCapturedVariables<FunctionRef> {
    let definition_to_function_map = build_definition_to_function_map(context);
    let mut captured_variables = HashMap::new();
    let mut visitor = CapturedVariableVisitor {
        captured_variables: &mut captured_variables,
        definition_to_function_map: &definition_to_function_map,
        module_context: context,
        current_exported_function: None,
    };
    visit_module_ast(&mut visitor, context);
    ModuleCapturedVariables(captured_variables)
}

pub fn collect_captured_variables(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> WholeProgramCapturedVariables {
    let step = StepLogger::start("Indexing captured variables", "Indexed captured variables");

    let captured_variables = dashmap::DashMap::new();

    ThreadPool::new().install(|| {
        slow_fun_monitor_scope(|slow_function_monitor| {
            handles.par_iter().for_each(|handle| {
                let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();
                let context =
                    ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                let captureds_for_module = slow_function_monitor.monitor_function(
                    move || collect_captured_variables_for_module(&context),
                    format!(
                        "Indexing captured variables for {}",
                        handle.module().as_str(),
                    ),
                    /* max_time_in_seconds */ 4,
                );
                captured_variables.insert(module_id, captureds_for_module);
            });
        })
    });

    step.finish();
    WholeProgramCapturedVariables(captured_variables.into_read_only())
}

pub fn export_captured_variables_for_module(
    captured_variables: &WholeProgramCapturedVariables,
    context: &ModuleContext,
) -> HashMap<FunctionRef, Vec<CapturedVariable>> {
    captured_variables
        .get_for_module(context.module_id)
        .unwrap()
        .clone()
        .0
        .into_iter()
        .map(|(k, v)| (k, v.keys().cloned().collect::<Vec<_>>()))
        .collect()
}
