/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
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
use crate::graph::index::Idx;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::ScopeId;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionRef;

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub struct CapturedVariable {
    pub name: Name,
}

#[derive(Debug, Clone)]
pub struct ModuleCapturedVariables(HashMap<FunctionRef, HashSet<CapturedVariable>>);

impl ModuleCapturedVariables {
    #[cfg(test)]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[cfg(test)]
    pub fn into_iter(self) -> impl Iterator<Item = (FunctionRef, HashSet<CapturedVariable>)> {
        self.0.into_iter()
    }

    pub fn get<'a>(&'a self, function_ref: &FunctionRef) -> Option<&'a HashSet<CapturedVariable>> {
        self.0.get(function_ref)
    }
}

struct DefinitionToScopeMapVisitor<'a> {
    definition_to_scope_map: &'a mut HashMap<Idx<Key>, ScopeId>,
    module_context: &'a ModuleContext<'a>,
}

impl<'a> DefinitionToScopeMapVisitor<'a> {
    fn bind_name(&mut self, key: Key, scopes: &Scopes) {
        if let Some(idx) = self
            .module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key))
        {
            assert!(
                self.definition_to_scope_map
                    .insert(idx, ScopeId::from_scopes(scopes))
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

impl<'a> AstScopedVisitor for DefinitionToScopeMapVisitor<'a> {
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
}

fn build_definition_to_scope_map(context: &ModuleContext) -> HashMap<Idx<Key>, ScopeId> {
    let mut definition_to_scope_map = HashMap::new();
    let mut visitor = DefinitionToScopeMapVisitor {
        definition_to_scope_map: &mut definition_to_scope_map,
        module_context: context,
    };
    visit_module_ast(&mut visitor, context);
    definition_to_scope_map
}

struct CapturedVariableVisitor<'a> {
    captured_variables: &'a mut HashMap<FunctionRef, HashSet<CapturedVariable>>,
    definition_to_scope_map: &'a HashMap<Idx<Key>, ScopeId>,
    module_context: &'a ModuleContext<'a>,
    current_exported_function: Option<FunctionRef>,
}

impl<'a> CapturedVariableVisitor<'a> {
    fn check_capture(&mut self, key: Key, name: &Name, scopes: &Scopes) {
        if let Some(scope_id) = self.get_definition_scope_from_usage(key)
            && scope_id != ScopeId::from_scopes(scopes)
            && let Some(current_function) = &self.current_exported_function
        {
            self.captured_variables
                .entry(current_function.clone())
                .or_default()
                .insert(CapturedVariable { name: name.clone() });
        }
    }

    fn get_definition_scope_from_usage(&self, key: Key) -> Option<ScopeId> {
        let idx = self
            .module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key))?;
        let binding = self.module_context.bindings.get(idx);
        match binding {
            Binding::Forward(definition_idx) => {
                self.get_definition_scope_from_idx(*definition_idx, SmallSet::new())
            }
            _ => None,
        }
    }

    fn get_definition_scope_from_idx(
        &self,
        idx: Idx<Key>,
        mut seen: SmallSet<Idx<Key>>,
    ) -> Option<ScopeId> {
        if let Some(scope_id) = self.definition_to_scope_map.get(&idx) {
            return Some(*scope_id);
        }

        // Avoid cycles in bindings.
        if seen.contains(&idx) {
            return None;
        }
        seen.insert(idx);

        let binding = self.module_context.bindings.get(idx);
        match binding {
            Binding::Forward(idx) => self.get_definition_scope_from_idx(*idx, seen),
            Binding::Phi(elements) => {
                for idx in elements {
                    if let Some(scope_id) = self.get_definition_scope_from_idx(*idx, seen.clone()) {
                        return Some(scope_id);
                    }
                }
                None
            }
            Binding::CompletedPartialType(idx, _) => self.get_definition_scope_from_idx(*idx, seen),
            _ => None,
        }
    }
}

impl<'a> AstScopedVisitor for CapturedVariableVisitor<'a> {
    fn on_scope_update(&mut self, scopes: &Scopes) {
        self.current_exported_function = scopes.current_exported_function(
            self.module_context.module_id,
            self.module_context.module_info.name(),
            /* include_top_level */ true,
            /* include_class_top_level */ true,
            /* include_decorators_in_decorated_definition */ false,
        );
    }

    fn visit_expression(&mut self, expr: &Expr, scopes: &Scopes) {
        if self.current_exported_function.is_none() {
            return;
        }

        match expr {
            Expr::Name(x) => {
                self.check_capture(
                    Key::BoundName(ShortIdentifier::expr_name(x)),
                    x.id(),
                    scopes,
                );
            }
            _ => (),
        }
    }

    fn visit_statement(&mut self, stmt: &Stmt, scopes: &Scopes) {
        if self.current_exported_function.is_none() {
            return;
        }

        match stmt {
            Stmt::Nonlocal(nonlocal) => {
                for identifier in &nonlocal.names {
                    self.check_capture(
                        Key::MutableCapture(ShortIdentifier::new(identifier)),
                        identifier.id(),
                        scopes,
                    );
                }
            }
            _ => (),
        }
    }
}

pub fn export_captured_variables(context: &ModuleContext) -> ModuleCapturedVariables {
    let definition_to_scope_map = build_definition_to_scope_map(context);
    let mut captured_variables = HashMap::new();
    let mut visitor = CapturedVariableVisitor {
        captured_variables: &mut captured_variables,
        definition_to_scope_map: &definition_to_scope_map,
        module_context: context,
        current_exported_function: None,
    };
    visit_module_ast(&mut visitor, context);
    ModuleCapturedVariables(captured_variables)
}
