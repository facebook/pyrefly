/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::hash::Hash;

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::class::Class;
use pyrefly_types::types::Type;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::ClassFieldDefinition;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::get_class_field_declaration;
use crate::report::pysa::class::get_context_from_class;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::override_graph::OverrideGraph;
use crate::report::pysa::override_graph::get_last_definition;
use crate::report::pysa::types::has_superclass;
use crate::state::lsp::FindPreference;

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Copy)]
pub enum ImplicitReceiver {
    TrueWithClassReceiver,
    TrueWithObjectReceiver,
    False,
}

pub trait FunctionTrait: std::fmt::Debug + PartialEq + Eq + Clone + Hash + Serialize {}

impl FunctionTrait for FunctionRef {}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize)]
pub enum Target<Function: FunctionTrait> {
    Function(Function), // Either a function or a method
    Override(Function),
    #[allow(dead_code)]
    Object(String),
}

impl<Function: FunctionTrait> Target<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> Target<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        match self {
            Target::Function(function) => Target::Function(map(function)),
            Target::Override(function) => Target::Override(map(function)),
            Target::Object(object) => Target::Object(object),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallTarget<Function: FunctionTrait> {
    pub(crate) target: Target<Function>,
    // `TrueWithClassReceiver` or `TrueWithObjectReceiver` if the call has an implicit receiver,
    // such as calling an instance or a class method.
    // For instance, `x.foo(0)` should be treated as `C.foo(x, 0)`. As another example, `C.foo(0)`
    // should be treated as `C.foo(C, 0)`.
    pub(crate) implicit_receiver: ImplicitReceiver,
    // True if this is an implicit call to the `__call__` method.
    pub(crate) implicit_dunder_call: bool,
    // The class of the receiver object at this call site, if any.
    pub(crate) receiver_class: Option<ClassRef>,
    // True if calling a class method.
    pub(crate) is_class_method: bool,
    // True if calling a static method.
    pub(crate) is_static_method: bool,
}

impl<Function: FunctionTrait> CallTarget<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> CallTarget<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        CallTarget {
            target: self.target.map_function(map),
            implicit_receiver: self.implicit_receiver,
            receiver_class: self.receiver_class,
            implicit_dunder_call: self.implicit_dunder_call,
            is_class_method: self.is_class_method,
            is_static_method: self.is_static_method,
        }
    }

    #[cfg(test)]
    pub fn with_implicit_receiver(mut self, implicit_receiver: ImplicitReceiver) -> Self {
        self.implicit_receiver = implicit_receiver;
        self
    }

    pub fn with_implicit_dunder_call(mut self, implicit_dunder_call: bool) -> Self {
        self.implicit_dunder_call = implicit_dunder_call;
        self
    }

    #[cfg(test)]
    pub fn with_is_class_method(mut self, is_class_method: bool) -> Self {
        self.is_class_method = is_class_method;
        self
    }

    #[cfg(test)]
    pub fn with_is_static_method(mut self, is_static_method: bool) -> Self {
        self.is_static_method = is_static_method;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallCallees<Function: FunctionTrait> {
    pub(crate) call_targets: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> CallCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> CallCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        CallCallees {
            call_targets: self
                .call_targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, &map))
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.call_targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.call_targets.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttributeAccessCallees<Function: FunctionTrait> {
    pub(crate) callable_targets: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> AttributeAccessCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> AttributeAccessCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        AttributeAccessCallees {
            callable_targets: self
                .callable_targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, &map))
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.callable_targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.callable_targets.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdentifierCallees<Function: FunctionTrait> {
    pub(crate) callable_targets: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> IdentifierCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> IdentifierCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        IdentifierCallees {
            callable_targets: self
                .callable_targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, &map))
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.callable_targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.callable_targets.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ExpressionCallees<Function: FunctionTrait> {
    Call(CallCallees<Function>),
    Identifier(IdentifierCallees<Function>),
    AttributeAccess(AttributeAccessCallees<Function>),
}

impl<Function: FunctionTrait> ExpressionCallees<Function> {
    #[cfg(test)]
    pub fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: MapFunction,
    ) -> ExpressionCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        match self {
            ExpressionCallees::Call(call_callees) => {
                ExpressionCallees::Call(call_callees.map_function(map))
            }
            ExpressionCallees::Identifier(identifier_callees) => {
                ExpressionCallees::Identifier(identifier_callees.map_function(map))
            }
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                ExpressionCallees::AttributeAccess(attribute_access_callees.map_function(map))
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            ExpressionCallees::Call(call_callees) => call_callees.is_empty(),
            ExpressionCallees::Identifier(identifier_callees) => identifier_callees.is_empty(),
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                attribute_access_callees.is_empty()
            }
        }
    }

    pub fn all_targets<'a>(&'a self) -> Box<dyn Iterator<Item = &'a CallTarget<Function>> + 'a> {
        match self {
            ExpressionCallees::Call(call_callees) => Box::new(call_callees.all_targets()),
            ExpressionCallees::Identifier(identifier_callees) => {
                Box::new(identifier_callees.all_targets())
            }
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                Box::new(attribute_access_callees.all_targets())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallGraph<Function: FunctionTrait>(HashMap<PysaLocation, ExpressionCallees<Function>>);

impl<Function: FunctionTrait> CallGraph<Function> {
    #[cfg(test)]
    pub fn from_map(map: HashMap<PysaLocation, ExpressionCallees<Function>>) -> Self {
        Self(map)
    }

    #[cfg(test)]
    pub fn into_iter(self) -> impl Iterator<Item = (PysaLocation, ExpressionCallees<Function>)> {
        self.0.into_iter()
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = (&PysaLocation, &ExpressionCallees<Function>)> {
        self.0.iter()
    }
}

impl<Function: FunctionTrait> Default for CallGraph<Function> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallGraphs<Function: FunctionTrait>(HashMap<Function, CallGraph<Function>>);

impl<Function: FunctionTrait> CallGraphs<Function> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[cfg(test)]
    pub fn from_map(map: HashMap<Function, CallGraph<Function>>) -> Self {
        Self(map)
    }

    pub fn into_iter(self) -> impl Iterator<Item = (Function, CallGraph<Function>)> {
        self.0.into_iter()
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = (&Function, &CallGraph<Function>)> {
        self.0.iter()
    }

    #[cfg(test)]
    pub fn intersect(&mut self, other: &Self) {
        self.0.retain(|target, _| other.0.contains_key(target));
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

struct CallGraphVisitor<'a> {
    call_graphs: &'a mut CallGraphs<FunctionRef>,
    module_context: &'a ModuleContext<'a>,
    module_id: ModuleId,
    module_name: ModuleName,
    function_base_definitions: &'a WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    override_graph: &'a OverrideGraph,
    current_function: Option<FunctionRef>, // The current function, if it is exported.
    debug: bool,                           // Enable logging for the current function or class body.
    debug_scopes: Vec<bool>,               // The value of the debug flag for each scope.
}

impl<'a> CallGraphVisitor<'a> {
    fn add_callees(&mut self, location: TextRange, callees: ExpressionCallees<FunctionRef>) {
        assert!(
            self.call_graphs
                .0
                .entry(self.current_function.clone().unwrap())
                .or_default()
                .0
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

    fn create_callee_from_class_field(
        &self,
        class: &Class,
        field_name: &Name,
    ) -> Option<FunctionRef> {
        let context = get_context_from_class(class, self.module_context);
        get_class_field_declaration(class, field_name, &context).and_then(|field_binding| {
            match field_binding {
                BindingClassField {
                    definition: ClassFieldDefinition::MethodLike { definition, .. },
                    ..
                } => {
                    let binding = context.bindings.get(*definition);
                    if let Binding::Function(key_decorated_function, ..) = binding {
                        Some(FunctionRef::from_decorated_function(
                            &get_last_definition(*key_decorated_function, &context),
                            &context,
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
    }

    // Figure out what target to pick for an indirect call that resolves to implementation_target.
    // E.g., if the receiver type is A, and A derives from Base, and the target is Base.method, then
    // targeting the override tree of Base.method is wrong, as it would include all siblings for A.//
    // Instead, we have the following cases:
    // a) receiver type matches implementation_target's declaring type -> override implementation_target
    // b) no implementation_target override entries are subclasses of A -> real implementation_target
    // c) some override entries are subclasses of A -> search upwards for actual implementation,
    //    and override all those where the override name is
    //  1) the override target if it exists in the override shared mem
    //  2) the real target otherwise
    fn compute_indirect_targets(
        &self,
        receiver_class: Option<&ClassRef>,
        callee: FunctionRef,
    ) -> Vec<Target<FunctionRef>> {
        // TODO: Optimize by avoiding to call this function when we are sure this is a direct call
        if receiver_class.is_none() {
            return vec![Target::Function(callee)];
        }
        let receiver_class = receiver_class.unwrap();

        let callee_class = self
            .function_base_definitions
            .get(callee.module_id, &callee.function_id)
            .and_then(|definition| definition.defining_class.clone());
        let callee_class = callee_class
            .unwrap_or_else(|| panic!("Expect a callee class for callee `{:#?}`", callee));

        let get_actual_target = |callee: FunctionRef| {
            if self.override_graph.overrides_exist(&callee) {
                Target::Override(callee)
            } else {
                Target::Function(callee)
            }
        };
        if &callee_class == receiver_class {
            // case a
            vec![get_actual_target(callee)]
        } else if let Some(overriding_classes) = self.override_graph.get_overriding_classes(&callee)
        {
            // case c
            let mut callees = overriding_classes
                .iter()
                .filter_map(|overriding_class| {
                    if has_superclass(
                        &overriding_class.class,
                        &receiver_class.class,
                        self.module_context,
                    ) {
                        self.create_callee_from_class_field(
                            &overriding_class.class,
                            &callee.function_name,
                        )
                        .map(&get_actual_target)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            callees.push(Target::Function(callee));
            callees
        } else {
            // case b
            vec![Target::Function(callee)]
        }
    }

    fn resolve_name(&self, name: &ExprName) -> Vec<CallTarget<FunctionRef>> {
        let create_call_target = |function_ref: FunctionRef| {
            let method_metadata = self.get_method_metadata(&function_ref);
            let implicit_receiver = has_implicit_receiver(
                &method_metadata,
                false, // We don't know receiver type since we are given an `ExprName`
            );
            CallTarget {
                target: Target::Function(function_ref),
                implicit_receiver,
                receiver_class: None,
                implicit_dunder_call: false,
                is_class_method: method_metadata.is_classmethod,
                is_static_method: method_metadata.is_staticmethod,
            }
        };
        let identifier = Ast::expr_name_identifier(name.clone());
        let go_to_definitions = self
            .module_context
            .transaction
            .find_definition_for_name_use(
                &self.module_context.handle,
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
                .map(&create_call_target)
            })
            .collect::<Vec<_>>();
        if !go_to_definitions.is_empty() {
            go_to_definitions
        } else {
            // There is no go-to-definition when for example an `ExprName` is a class definition,
            // a local variable, or a parameter.
            let callee_type = self.module_context.answers.get_type_trace(name.range());
            let call_target = match callee_type {
                Some(Type::ClassType(class_type)) => {
                    // Since we are calling an instance of a class, look up method `__call__`
                    self.create_callee_from_class_field(
                        class_type.class_object(),
                        &Name::new("__call__"),
                    )
                    .map(|function_ref| {
                        create_call_target(function_ref).with_implicit_dunder_call(true)
                    })
                }
                _ => None,
            };
            call_target.into_iter().collect::<Vec<_>>()
        }
    }

    fn resolve_attribute_access(&self, attribute: &ExprAttribute) -> Vec<CallTarget<FunctionRef>> {
        let receiver_type = self
            .module_context
            .answers
            .get_type_trace(attribute.value.range());
        self.module_context
            .transaction
            .find_definition_for_attribute(
                &self.module_context.handle,
                attribute.value.range(),
                &attribute.attr,
                &FindPreference::default(),
            )
            .iter()
            .flat_map(|definition| {
                FunctionRef::from_find_definition_item_with_docstring(
                    definition,
                    self.function_base_definitions,
                    self.module_context,
                )
                .map(|function_ref| {
                    let method_metadata = self.get_method_metadata(&function_ref);
                    if let Some(ref receiver_type) = receiver_type {
                        let (receiver_class, is_receiver_class_def) = self
                            .receiver_class_from_type(
                                receiver_type,
                                method_metadata.is_classmethod,
                            );
                        let implicit_receiver =
                            has_implicit_receiver(&method_metadata, is_receiver_class_def);
                        self.compute_indirect_targets(receiver_class.as_ref(), function_ref)
                            .into_iter()
                            .map(|target| CallTarget {
                                target,
                                implicit_receiver,
                                receiver_class: receiver_class.clone(),
                                implicit_dunder_call: false,
                                is_class_method: method_metadata.is_classmethod,
                                is_static_method: method_metadata.is_staticmethod,
                            })
                            .collect::<Vec<_>>()
                    } else {
                        vec![CallTarget {
                            target: Target::Function(function_ref),
                            implicit_receiver: has_implicit_receiver(
                                &method_metadata,
                                /* is_receiver_class_def */ false,
                            ),
                            receiver_class: None,
                            implicit_dunder_call: false,
                            is_class_method: method_metadata.is_classmethod,
                            is_static_method: method_metadata.is_staticmethod,
                        }]
                    }
                })
                .unwrap_or_default()
            })
            .collect::<Vec<_>>()
    }

    fn resolve_call(&self, call: &ExprCall) -> Vec<CallTarget<FunctionRef>> {
        debug_println!(self.debug, "Visiting call: {:#?}", call);

        match &*call.func {
            Expr::Name(name) => self.resolve_name(name),
            Expr::Attribute(attribute) => self.resolve_attribute_access(attribute),
            _ => Vec::new(),
        }
    }

    fn resolve_expression(&self, expr: &Expr) -> Option<ExpressionCallees<FunctionRef>> {
        match expr {
            Expr::Call(call) => Some(ExpressionCallees::Call(CallCallees {
                call_targets: self.resolve_call(call),
            })),
            Expr::Name(name) => {
                // TODO: Avoid visiting when the parent expression is a `Call`
                Some(ExpressionCallees::Identifier(IdentifierCallees {
                    callable_targets: self.resolve_name(name),
                }))
            }
            Expr::Attribute(attribute) => {
                // TODO: Avoid visiting when the parent expression is a `Call`
                Some(ExpressionCallees::AttributeAccess(AttributeAccessCallees {
                    callable_targets: self.resolve_attribute_access(attribute),
                }))
            }
            _ => None,
        }
    }

    // Enable debug logs by adding `pysa_dump()` to the top level statements of the definition of interest
    const DEBUG_FUNCTION_NAME: &'static str = "pysa_dump";

    fn enter_debug_scope(&mut self, body: &[Stmt]) {
        self.debug = has_toplevel_call(body, Self::DEBUG_FUNCTION_NAME);
        self.debug_scopes.push(self.debug);
    }

    fn exit_debug_scope(&mut self) {
        self.debug_scopes.pop();
        self.debug = self.debug_scopes.last().copied().unwrap();
    }
}

impl<'a> AstScopedVisitor for CallGraphVisitor<'a> {
    fn on_scope_update(&mut self, scopes: &Scopes) {
        self.current_function = scopes.current_exported_function(
            self.module_id,
            self.module_name,
            /* include_top_level */ true,
            /* include_class_top_level */ true,
            /* include_decorators_in_decorated_definition */ true,
        );
    }

    fn enter_function_scope(&mut self, function_def: &StmtFunctionDef, _: &Scopes) {
        self.enter_debug_scope(&function_def.body);
    }

    fn enter_class_scope(&mut self, class_def: &StmtClassDef, _: &Scopes) {
        self.enter_debug_scope(&class_def.body);
    }

    fn exit_function_scope(&mut self, _function_def: &StmtFunctionDef, _: &Scopes) {
        self.exit_debug_scope();
    }

    fn exit_class_scope(&mut self, _function_def: &StmtClassDef, _: &Scopes) {
        self.exit_debug_scope();
    }

    fn enter_toplevel_scope(&mut self, ast: &ModModule, _: &Scopes) {
        self.enter_debug_scope(&ast.body);
    }

    fn visit_expression(&mut self, expr: &Expr, _: &Scopes) {
        if self.current_function.is_none() {
            return;
        }

        let callees = self.resolve_expression(expr);
        if let Some(callees) = callees
            && !callees.is_empty()
        {
            self.add_callees(expr.range(), callees);
        }
    }
}

fn resolve_call(
    call: &ExprCall,
    function_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    module_context: &ModuleContext,
    override_graph: &OverrideGraph,
) -> CallCallees<FunctionRef> {
    let mut call_graphs = CallGraphs::new();
    let visitor = CallGraphVisitor {
        call_graphs: &mut call_graphs,
        module_context,
        module_id: module_context.module_id,
        module_name: module_context.module_info.name(),
        function_base_definitions: function_definitions,
        current_function: None,
        debug: false,
        debug_scopes: Vec::new(),
        override_graph,
    };
    CallCallees {
        call_targets: visitor.resolve_call(call),
    }
}

fn resolve_expression(
    expression: &Expr,
    function_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    module_context: &ModuleContext,
    override_graph: &OverrideGraph,
) -> Option<ExpressionCallees<FunctionRef>> {
    let mut call_graphs = CallGraphs::new();
    let visitor = CallGraphVisitor {
        call_graphs: &mut call_graphs,
        module_context,
        module_id: module_context.module_id,
        module_name: module_context.module_info.name(),
        function_base_definitions: function_definitions,
        current_function: None,
        debug: false,
        debug_scopes: Vec::new(),
        override_graph,
    };
    visitor.resolve_expression(expression)
}

// Requires `context` to be the module context of the decorators.
pub fn resolve_decorator_callees(
    decorators: &[Decorator],
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    context: &ModuleContext,
) -> HashMap<PysaLocation, Vec<Target<FunctionRef>>> {
    let mut decorator_callees = HashMap::new();
    // We do not care about overrides for now
    let override_graph = OverrideGraph::new();

    for decorator in decorators {
        let (range, callees) = match &decorator.expression {
            Expr::Call(call) => {
                // Decorator factor, e.g `@foo(1)`. We export the callee of `foo`.
                let callees =
                    resolve_call(call, function_base_definitions, context, &override_graph);
                (
                    (*call.func).range(),
                    callees
                        .all_targets()
                        .map(|call_target| call_target.target.clone())
                        .collect(),
                )
            }
            expr => {
                let callees =
                    resolve_expression(expr, function_base_definitions, context, &override_graph);
                (
                    expr.range(),
                    if let Some(callees) = callees {
                        callees
                            .all_targets()
                            .map(|call_target| call_target.target.clone())
                            .collect()
                    } else {
                        Vec::new()
                    },
                )
            }
        };

        if !callees.is_empty() {
            let location = PysaLocation::new(context.module_info.display_range(range));
            assert!(
                decorator_callees.insert(location, callees).is_none(),
                "Found multiple decorators at the same location"
            );
        }
    }

    decorator_callees
}

pub fn export_call_graphs(
    context: &ModuleContext,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    override_graph: &OverrideGraph,
) -> CallGraphs<FunctionRef> {
    let mut call_graphs = CallGraphs::new();

    let mut visitor = CallGraphVisitor {
        call_graphs: &mut call_graphs,
        module_context: context,
        module_id: context.module_id,
        module_name: context.module_info.name(),
        function_base_definitions,
        current_function: None,
        debug: false,
        debug_scopes: Vec::new(),
        override_graph,
    };

    visit_module_ast(&mut visitor, context);
    call_graphs
}
