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
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::report::pysa::class::ClassRef;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::state::lsp::FindPreference;

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ImplicitReceiver {
    TrueWithClassReceiver,
    TrueWithObjectReceiver,
    False,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CallTarget<Target> {
    pub(crate) target: Target,
    pub(crate) implicit_receiver: ImplicitReceiver,
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
            implicit_receiver: self.implicit_receiver.clone(),
            receiver_class: self.receiver_class.clone(),
        }
    }

    #[cfg(test)]
    pub fn with_implicit_receiver(mut self, implicit_receiver: ImplicitReceiver) -> Self {
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

macro_rules! debug_println {
    ($debug:expr, $($arg:tt)*) => {
        if $debug {
            tracing::info!($($arg)*);
        }
    };
}

fn has_toplevel_call(body: &[Stmt], callee_name: &'static str) -> bool {
    body.iter().any(|stmt| match stmt {
        Stmt::Expr(stmt_expr) => match &*stmt_expr.value {
            Expr::Call(call) => match &*call.func {
                Expr::Name(name) => name.id == callee_name,
                _ => false,
            },
            _ => false,
        },
        _ => false,
    })
}

// This also strips `Optional[T]` since that is represented by `Union`
fn strip_none_from_union(type_: &Type) -> Type {
    match type_ {
        Type::Union(types) => {
            if let Ok(none_index) = types.binary_search(&Type::None) {
                let mut new_types = types.clone();
                new_types.remove(none_index);
                match new_types.len() {
                    0 => panic!("Unexpected union type `{:#?}`", type_),
                    1 => new_types.into_iter().next().unwrap(),
                    _ => Type::Union(new_types),
                }
            } else {
                type_.clone()
            }
        }
        _ => type_.clone(),
    }
}

struct MethodMetadata {
    is_staticmethod: bool,
    is_classmethod: bool,
    is_method: bool,
}

fn has_implicit_receiver(
    method_metadata: &MethodMetadata,
    is_receiver_class_def: bool,
) -> ImplicitReceiver {
    if method_metadata.is_staticmethod {
        ImplicitReceiver::False
    } else if method_metadata.is_classmethod {
        if is_receiver_class_def {
            // C.f() where f is a class method
            ImplicitReceiver::TrueWithClassReceiver
        } else {
            // c.f() where f is a class method
            ImplicitReceiver::TrueWithObjectReceiver
        }
    } else if method_metadata.is_method {
        if is_receiver_class_def {
            // C.g(c) where g is a method
            ImplicitReceiver::False
        } else {
            // c.g() where g is a method
            ImplicitReceiver::TrueWithObjectReceiver
        }
    } else {
        ImplicitReceiver::False
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
    definition_nesting: Vec<FunctionRef>,
    call_graphs: &'a mut CallGraphs<FunctionRef, PysaLocation>,
    in_exported_definition: bool,
    debug_current_definition: bool,
}

impl<'a> CallGraphVisitor<'a> {
    fn current_definition(&self) -> FunctionRef {
        self.definition_nesting.last().unwrap().clone()
    }

    fn add_callees(&mut self, location: TextRange, callees: ExpressionCallees<FunctionRef>) {
        assert!(
            self.call_graphs
                .0
                .entry(self.current_definition())
                .or_default()
                .insert(
                    PysaLocation::new(self.module_context.module_info.display_range(location)),
                    callees,
                )
                .is_none(),
            "Adding callees to the same location"
        );
    }

    fn receiver_class_from_type(
        &self,
        receiver_type: &Type,
        is_class_method: bool,
    ) -> (Option<ClassRef>, bool) {
        let receiver_type = strip_none_from_union(receiver_type);
        match receiver_type {
            Type::ClassType(class_type) => (
                Some(ClassRef::from_class(
                    class_type.class_object(),
                    self.module_context.module_ids,
                )),
                false, // Whether the receiver is `type(SomeClass)`
            ),
            Type::ClassDef(class_def) => {
                // The receiver is the class itself. Technically, the receiver class type should be `type(SomeClass)`.
                // However, we strip away the `type` part since it is implied by the `is_class_method` flag.
                (
                    if is_class_method {
                        Some(ClassRef::from_class(
                            &class_def,
                            self.module_context.module_ids,
                        ))
                    } else {
                        None
                    },
                    true,
                )
            }
            _ => (None, false),
        }
    }

    fn get_method_metadata(&self, function_ref: &FunctionRef) -> MethodMetadata {
        self.function_base_definitions
            .get(self.module_id, &function_ref.function_id)
            .map_or(
                MethodMetadata {
                    is_staticmethod: false,
                    is_classmethod: false,
                    is_method: false,
                },
                |function_definition| MethodMetadata {
                    is_staticmethod: function_definition.is_staticmethod,
                    is_classmethod: function_definition.is_classmethod,
                    is_method: FunctionBaseDefinition::is_method(function_definition),
                },
            )
    }

    fn resolve_name(&self, name: &ExprName) -> Vec<CallTarget<FunctionRef>> {
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
                FunctionRef::from_find_definition_item_with_docstring(
                    definition,
                    self.function_base_definitions,
                    self.module_context,
                )
                .map(|function_ref| {
                    let method_metadata = self.get_method_metadata(&function_ref);
                    let implicit_receiver = has_implicit_receiver(
                        &method_metadata,
                        false, // We don't know receiver type
                    );
                    CallTarget {
                        target: function_ref,
                        implicit_receiver,
                        receiver_class: None,
                    }
                })
            })
            .collect::<Vec<_>>()
    }

    fn resolve_attribute_access(&self, attribute: &ExprAttribute) -> Vec<CallTarget<FunctionRef>> {
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
                FunctionRef::from_find_definition_item_with_docstring(
                    definition,
                    self.function_base_definitions,
                    self.module_context,
                )
                .map(|function_ref| {
                    let method_metadata = self.get_method_metadata(&function_ref);
                    let (receiver_class, is_receiver_class_def) = self
                        .module_context
                        .answers
                        .get_type_trace(attribute.value.range())
                        .map_or((None, false), |receiver_type| {
                            self.receiver_class_from_type(
                                &receiver_type,
                                method_metadata.is_classmethod,
                            )
                        });
                    let implicit_receiver =
                        has_implicit_receiver(&method_metadata, is_receiver_class_def);
                    CallTarget {
                        target: function_ref,
                        implicit_receiver,
                        receiver_class,
                    }
                })
            })
            .collect::<Vec<_>>()
    }

    // Enable debug logs by adding `pysa_dump()` to the top level statements of the definition of interest
    const DEBUG_FUNCTION_NAME: &'static str = "pysa_dump";

    fn set_debug_for_body(&mut self, body: &[Stmt]) {
        self.debug_current_definition = has_toplevel_call(body, Self::DEBUG_FUNCTION_NAME);
    }

    fn unset_debug(&mut self) {
        self.debug_current_definition = false;
    }
}

impl<'a> Visitor<'a> for CallGraphVisitor<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                let function_id = FunctionId::Function {
                    location: PysaLocation::new(
                        self.module_context
                            .module_info
                            .display_range(function_def.identifier()),
                    ),
                };
                if let Some(function_definition) = self
                    .function_base_definitions
                    .get(self.module_id, &function_id)
                {
                    self.set_debug_for_body(&function_def.body);
                    self.definition_nesting.push(FunctionRef {
                        module_id: self.module_id,
                        module_name: self.module_name,
                        function_id,
                        function_name: function_definition.name.clone(),
                    });
                    visitor::walk_stmt(self, stmt);
                    self.definition_nesting.pop();
                    self.unset_debug();
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
                debug_println!(self.debug_current_definition, "Visiting call: {:#?}", call);

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
) -> CallGraphs<FunctionRef, PysaLocation> {
    let mut call_graphs = CallGraphs::new();

    let module_name = context.module_info.name();
    let module_toplevel = FunctionRef {
        module_id: context.module_id,
        module_name,
        function_id: FunctionId::ModuleTopLevel,
        function_name: Name::from("$toplevel"),
    };
    let mut visitor = CallGraphVisitor {
        module_context: context,
        module_id: context.module_id,
        module_name,
        definition_nesting: vec![module_toplevel],
        call_graphs: &mut call_graphs,
        function_base_definitions,
        in_exported_definition: true,
        debug_current_definition: false,
    };

    for stmt in &context.ast.body {
        visitor.visit_stmt(stmt);
    }
    call_graphs
}
