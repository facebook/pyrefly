/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_python_ast::identifier::Identifier;
use ruff_python_ast::visitor;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::report::pysa::ClassRef;
use crate::report::pysa::DefinitionRef;
use crate::report::pysa::FunctionBaseDefinition;
use crate::report::pysa::FunctionId;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::ModuleId;
use crate::report::pysa::PysaLocation;
use crate::report::pysa::WholeProgramFunctionDefinitions;
use crate::state::lsp::FindPreference;

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CallTarget<Target> {
    pub(crate) target: Target,
    pub(crate) implicit_receiver: bool,
    pub(crate) receiver_class: Option<ClassRef>,
}

impl<Target> CallTarget<Target> {
    #[cfg(test)]
    fn map_target<TargetForTest, MapTarget>(&self, map: MapTarget) -> CallTarget<TargetForTest>
    where
        MapTarget: Fn(&Target) -> TargetForTest,
    {
        CallTarget {
            target: map(&self.target),
            implicit_receiver: self.implicit_receiver,
            receiver_class: self.receiver_class.clone(),
        }
    }

    #[cfg(test)]
    pub fn with_implicit_receiver(mut self, implicit_receiver: bool) -> Self {
        self.implicit_receiver = implicit_receiver;
        self
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub struct CallCallees<Target> {
    pub(crate) call_targets: Vec<CallTarget<Target>>,
}

impl<Target> CallCallees<Target> {
    #[cfg(test)]
    fn map_target<TargetForTest, MapTarget>(&self, map: MapTarget) -> CallCallees<TargetForTest>
    where
        MapTarget: Fn(&Target) -> TargetForTest,
    {
        CallCallees {
            call_targets: self
                .call_targets
                .iter()
                .map(|call_target| CallTarget::map_target(call_target, &map))
                .collect(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub struct AttributeAccessCallees<Target> {
    pub(crate) callable_targets: Vec<CallTarget<Target>>,
}

impl<Target> AttributeAccessCallees<Target> {
    #[cfg(test)]
    fn map_target<TargetForTest, MapTarget>(
        &self,
        map: MapTarget,
    ) -> AttributeAccessCallees<TargetForTest>
    where
        MapTarget: Fn(&Target) -> TargetForTest,
    {
        AttributeAccessCallees {
            callable_targets: self
                .callable_targets
                .iter()
                .map(|call_target| CallTarget::map_target(call_target, &map))
                .collect(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub struct IdentifierCallees<Target> {
    pub(crate) callable_targets: Vec<CallTarget<Target>>,
}

impl<Target> IdentifierCallees<Target> {
    #[cfg(test)]
    fn map_target<TargetForTest, MapTarget>(
        &self,
        map: MapTarget,
    ) -> IdentifierCallees<TargetForTest>
    where
        MapTarget: Fn(&Target) -> TargetForTest,
    {
        IdentifierCallees {
            callable_targets: self
                .callable_targets
                .iter()
                .map(|call_target| CallTarget::map_target(call_target, &map))
                .collect(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum ExpressionCallees<Target> {
    Call(CallCallees<Target>),
    Identifier(IdentifierCallees<Target>),
    AttributeAccess(AttributeAccessCallees<Target>),
}

impl<Target> ExpressionCallees<Target> {
    #[cfg(test)]
    pub fn map_target<TargetForTest, MapTarget>(
        &self,
        map: MapTarget,
    ) -> ExpressionCallees<TargetForTest>
    where
        MapTarget: Fn(&Target) -> TargetForTest,
    {
        match self {
            ExpressionCallees::Call(call_callees) => {
                ExpressionCallees::Call(call_callees.map_target(map))
            }
            ExpressionCallees::Identifier(identifier_callees) => {
                ExpressionCallees::Identifier(identifier_callees.map_target(map))
            }
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                ExpressionCallees::AttributeAccess(attribute_access_callees.map_target(map))
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CallGraphs<Target, Location>(
    pub(crate) HashMap<Target, HashMap<Location, ExpressionCallees<Target>>>,
);

impl<Target, Location> PartialEq for CallGraphs<Target, Location>
where
    Target: PartialEq + Eq + std::hash::Hash,
    Location: PartialEq + Eq + std::hash::Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<Target, Location> Eq for CallGraphs<Target, Location>
where
    Target: PartialEq + Eq + std::hash::Hash,
    Location: PartialEq + Eq + std::hash::Hash,
{
}

impl<Target, Location> CallGraphs<Target, Location> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[allow(dead_code)]
    pub fn from_map(map: HashMap<Target, HashMap<Location, ExpressionCallees<Target>>>) -> Self {
        Self(map)
    }
}

#[allow(dead_code)]
struct CallGraphVisitor<'a> {
    module_context: &'a ModuleContext<'a>,
    module_id: ModuleId,
    module_name: ModuleName,
    function_base_definitions: &'a WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    // A stack where the top element is always the current callable that we are
    // building a call graph for. The stack is updated each time we enter and exit
    // a function definition or a class definition.
    definition_nesting: Vec<DefinitionRef>,
    call_graphs: &'a mut CallGraphs<DefinitionRef, PysaLocation>,
    in_exported_definition: bool,
}

impl<'a> CallGraphVisitor<'a> {
    fn current_definition(&self) -> DefinitionRef {
        self.definition_nesting.last().unwrap().clone()
    }

    fn add_callees(&mut self, location: TextRange, callees: ExpressionCallees<DefinitionRef>) {
        assert!(
            self.call_graphs
                .0
                .entry(self.current_definition())
                .or_default()
                .insert(
                    PysaLocation(self.module_context.module_info.display_range(location)),
                    callees,
                )
                .is_none(),
            "Adding callees to the same location"
        );
    }

    fn has_implicit_receiver(
        &self,
        definition_ref: &DefinitionRef,
        explicit_receiver: bool,
    ) -> bool {
        let (is_staticmethod, is_classmethod, is_method) = self
            .function_base_definitions
            .get_and_map(
                self.module_id,
                &definition_ref.function_id,
                |function_definition| {
                    (
                        function_definition.is_staticmethod,
                        function_definition.is_classmethod,
                        FunctionBaseDefinition::is_method(function_definition),
                    )
                },
            )
            .unwrap_or((false, false, false));

        if is_staticmethod {
            false
        } else if is_classmethod {
            true
        } else {
            !explicit_receiver && is_method
        }
    }

    fn resolve_name(&self, name: &ExprName) -> Vec<CallTarget<DefinitionRef>> {
        let identifier = Ast::expr_name_identifier(name.clone());
        self.module_context
            .transaction
            .find_definition_for_name_use(
                self.module_context.handle,
                &identifier,
                &FindPreference::default(),
            )
            .map_or(vec![], |d| vec![d])
            .iter()
            .filter_map(|definition| {
                DefinitionRef::from_find_definition_item_with_docstring(
                    definition,
                    self.function_base_definitions,
                    self.module_context,
                )
                .map(|definition_ref| {
                    let implicit_receiver = self.has_implicit_receiver(
                        &definition_ref,
                        false, // explicit_receiver
                    );
                    CallTarget {
                        target: definition_ref,
                        implicit_receiver,
                        receiver_class: None,
                    }
                })
            })
            .collect::<Vec<_>>()
    }

    fn resolve_attribute_access(
        &self,
        attribute: &ExprAttribute,
    ) -> Vec<CallTarget<DefinitionRef>> {
        self.module_context
            .transaction
            .find_definition_for_attribute(
                self.module_context.handle,
                attribute.value.range(),
                &attribute.attr,
                &FindPreference::default(),
            )
            .iter()
            .filter_map(|definition| {
                DefinitionRef::from_find_definition_item_with_docstring(
                    definition,
                    self.function_base_definitions,
                    self.module_context,
                )
                .map(|definition_ref| {
                    let implicit_receiver = self.has_implicit_receiver(
                        &definition_ref,
                        false, // explicit_receiver
                    );
                    let receiver_class = self
                        .module_context
                        .answers
                        .get_type_trace(attribute.value.range())
                        .and_then(|type_| match type_ {
                            Type::ClassType(class_type) => Some(ClassRef::from_class(
                                class_type.class_object(),
                                self.module_context.module_ids,
                            )),
                            _ => None,
                        });
                    CallTarget {
                        target: definition_ref,
                        implicit_receiver,
                        receiver_class,
                    }
                })
            })
            .collect::<Vec<_>>()
    }
}

impl<'a> Visitor<'a> for CallGraphVisitor<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                let function_id = FunctionId::Function {
                    location: PysaLocation(
                        self.module_context
                            .module_info
                            .display_range(function_def.identifier()),
                    ),
                };
                if let Some(function_name) = self.function_base_definitions.get_and_map(
                    self.module_id,
                    &function_id,
                    |function_definition| function_definition.name.clone(),
                ) {
                    self.definition_nesting.push(DefinitionRef {
                        module_id: self.module_id,
                        module_name: self.module_name,
                        function_id,
                        identifier: function_name,
                    });
                    visitor::walk_stmt(self, stmt);
                    self.definition_nesting.pop();
                } else {
                    let current_in_exported_definition = self.in_exported_definition;
                    self.in_exported_definition = false;
                    visitor::walk_stmt(self, stmt);
                    self.in_exported_definition = current_in_exported_definition;
                }
            }
            Stmt::ClassDef(_class_def) => {
                // TODO: Push the class id into `definition_nesting`
                visitor::walk_stmt(self, stmt);
                // TODO: Pop the class id from `definition_nesting`
            }
            _ => {
                visitor::walk_stmt(self, stmt);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        visitor::walk_expr(self, expr);
        if !self.in_exported_definition {
            return;
        }

        match expr {
            Expr::Call(call) => {
                let call_targets = match &*call.func {
                    Expr::Name(name) => self.resolve_name(name),
                    Expr::Attribute(attribute) => self.resolve_attribute_access(attribute),
                    _ => Vec::new(),
                };
                if !call_targets.is_empty() {
                    self.add_callees(
                        expr.range(),
                        ExpressionCallees::Call(CallCallees { call_targets }),
                    );
                }
            }
            Expr::Name(name) => {
                // TODO: Avoid visiting when the parent expression is a `Call`
                let callable_targets = self.resolve_name(name);
                if !callable_targets.is_empty() {
                    self.add_callees(
                        expr.range(),
                        ExpressionCallees::Identifier(IdentifierCallees { callable_targets }),
                    );
                }
            }
            Expr::Attribute(attribute) => {
                // TODO: Avoid visiting when the parent expression is a `Call`
                let callable_targets = self.resolve_attribute_access(attribute);
                if !callable_targets.is_empty() {
                    self.add_callees(
                        expr.range(),
                        ExpressionCallees::AttributeAccess(AttributeAccessCallees {
                            callable_targets,
                        }),
                    );
                }
            }
            _ => (),
        };
    }
}

#[allow(dead_code)]
pub fn build_call_graphs_for_module(
    context: &ModuleContext,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
) -> CallGraphs<DefinitionRef, PysaLocation> {
    let mut call_graphs = CallGraphs::new();

    let module_name = context.module_info.name();
    let module_toplevel = DefinitionRef {
        module_id: context.module_id,
        module_name,
        function_id: FunctionId::ModuleTopLevel,
        identifier: "$module_top_level".to_owned(),
    };
    let mut visitor = CallGraphVisitor {
        module_context: context,
        module_id: context.module_id,
        module_name,
        definition_nesting: vec![module_toplevel],
        call_graphs: &mut call_graphs,
        function_base_definitions,
        in_exported_definition: true,
    };

    for stmt in &context.ast.body {
        visitor.visit_stmt(stmt);
    }
    call_graphs
}
