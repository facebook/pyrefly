/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Not;

use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::types::BoundMethod;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::Type;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ArgOrKeyword;
use ruff_python_ast::Arguments;
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
use starlark_map::Hashed;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::KeyDecoratedFunction;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::ScopeExportedFunctionFlags;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::get_class_field;
use crate::report::pysa::class::get_context_from_class;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionNode;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::function::should_export_decorated_function;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::override_graph::OverrideGraph;
use crate::report::pysa::types::ScalarTypeProperties;
use crate::report::pysa::types::has_superclass;
use crate::state::lsp::FindPreference;

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Copy, Hash, PartialOrd, Ord)]
pub enum ImplicitReceiver {
    TrueWithClassReceiver,
    TrueWithObjectReceiver,
    False,
}

impl ImplicitReceiver {
    // Required to pass by ref to use in `serde(skip_serializing_if=..)`
    #![allow(clippy::trivially_copy_pass_by_ref)]
    fn is_false(&self) -> bool {
        *self == ImplicitReceiver::False
    }
}

pub trait FunctionTrait:
    std::fmt::Debug + PartialEq + Eq + Clone + Hash + Serialize + PartialOrd + Ord
{
}

impl FunctionTrait for FunctionRef {}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, PartialOrd, Ord)]
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
        map: &MapFunction,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Hash, PartialOrd, Ord)]
pub struct CallTarget<Function: FunctionTrait> {
    pub(crate) target: Target<Function>,
    // `TrueWithClassReceiver` or `TrueWithObjectReceiver` if the call has an implicit receiver,
    // such as calling an instance or a class method.
    // For instance, `x.foo(0)` should be treated as `C.foo(x, 0)`. As another example, `C.foo(0)`
    // should be treated as `C.foo(C, 0)`.
    #[serde(skip_serializing_if = "ImplicitReceiver::is_false")]
    pub(crate) implicit_receiver: ImplicitReceiver,
    // True if this is an implicit call to the `__call__` method.
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub(crate) implicit_dunder_call: bool,
    // The class of the receiver object at this call site, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) receiver_class: Option<ClassRef>,
    // True if calling a class method.
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub(crate) is_class_method: bool,
    // True if calling a static method.
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub(crate) is_static_method: bool,
    // The return type of the call expression, or `None` for object targets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) return_type: Option<ScalarTypeProperties>,
}

impl<Function: FunctionTrait> CallTarget<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
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
            return_type: self.return_type,
        }
    }

    #[cfg(test)]
    pub fn with_implicit_receiver(mut self, implicit_receiver: ImplicitReceiver) -> Self {
        self.implicit_receiver = implicit_receiver;
        self
    }

    #[cfg(test)]
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

    #[cfg(test)]
    pub fn with_return_type(mut self, return_type: Option<ScalarTypeProperties>) -> Self {
        self.return_type = return_type;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum UnresolvedReason {
    LambdaArgument,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Unresolved {
    False,
    True(UnresolvedReason),
}

impl Unresolved {
    fn is_resolved(&self) -> bool {
        match self {
            Unresolved::False => true,
            Unresolved::True(_) => false,
        }
    }

    fn join(self, other: Self) -> Self {
        match (&self, &other) {
            (Unresolved::True(left), Unresolved::True(right)) => {
                if left == right {
                    self.clone()
                } else {
                    Unresolved::True(UnresolvedReason::Mixed)
                }
            }
            (Unresolved::True(..), Unresolved::False) => self.clone(),
            (Unresolved::False, Unresolved::True(..)) => other.clone(),
            (Unresolved::False, Unresolved::False) => Unresolved::False,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HigherOrderParameter<Function: FunctionTrait> {
    pub(crate) index: u32,
    pub(crate) call_targets: Vec<CallTarget<Function>>,
    #[serde(skip_serializing_if = "Unresolved::is_resolved")]
    pub(crate) unresolved: Unresolved,
}

impl<Function: FunctionTrait> HigherOrderParameter<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> HigherOrderParameter<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        HigherOrderParameter {
            index: self.index,
            call_targets: self
                .call_targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, map))
                .collect(),
            unresolved: self.unresolved,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallCallees<Function: FunctionTrait> {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) call_targets: Vec<CallTarget<Function>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) init_targets: Vec<CallTarget<Function>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) new_targets: Vec<CallTarget<Function>>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub(crate) higher_order_parameters: HashMap<u32, HigherOrderParameter<Function>>,
    #[serde(skip_serializing_if = "Unresolved::is_resolved")]
    pub(crate) unresolved: Unresolved,
}

impl<Function: FunctionTrait> CallCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> CallCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        let map_call_targets = |targets: Vec<CallTarget<Function>>| {
            targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, map))
                .collect()
        };
        CallCallees {
            call_targets: map_call_targets(self.call_targets),
            init_targets: map_call_targets(self.init_targets),
            new_targets: map_call_targets(self.new_targets),
            higher_order_parameters: self
                .higher_order_parameters
                .into_iter()
                .map(|(k, v)| (k, HigherOrderParameter::map_function(v, map)))
                .collect(),
            unresolved: self.unresolved,
        }
    }

    fn is_empty(&self) -> bool {
        self.call_targets.is_empty() && self.init_targets.is_empty() && self.new_targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.call_targets
            .iter()
            .chain(self.init_targets.iter())
            .chain(self.new_targets.iter())
    }

    fn dedup_and_sort(&mut self) {
        self.call_targets.sort();
        self.call_targets.dedup();
        self.init_targets.sort();
        self.init_targets.dedup();
        self.new_targets.sort();
        self.new_targets.dedup();
    }

    fn with_higher_order_parameters(
        &mut self,
        higher_order_parameters: HashMap<u32, HigherOrderParameter<Function>>,
    ) {
        self.higher_order_parameters = higher_order_parameters;
    }

    fn join_in_place(&mut self, other: Self) {
        self.call_targets.extend(other.call_targets);
        self.init_targets.extend(other.init_targets);
        self.new_targets.extend(other.new_targets);
        self.higher_order_parameters
            .extend(other.higher_order_parameters);
        self.unresolved = self.unresolved.clone().join(other.unresolved);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttributeAccessCallees<Function: FunctionTrait> {
    /// When the attribute access is called, the callees it may resolve to
    pub(crate) if_called: CallCallees<Function>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) property_setters: Vec<CallTarget<Function>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) property_getters: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> AttributeAccessCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> AttributeAccessCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        let map_call_targets = |targets: Vec<CallTarget<Function>>| {
            targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, map))
                .collect()
        };
        AttributeAccessCallees {
            if_called: self.if_called.map_function(map),
            property_setters: map_call_targets(self.property_setters),
            property_getters: map_call_targets(self.property_getters),
        }
    }

    fn is_empty(&self) -> bool {
        self.if_called.is_empty()
            && self.property_setters.is_empty()
            && self.property_getters.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.if_called
            .all_targets()
            .chain(self.property_setters.iter())
            .chain(self.property_getters.iter())
    }

    fn dedup_and_sort(&mut self) {
        self.if_called.dedup_and_sort();
        self.property_setters.sort();
        self.property_setters.dedup();
        self.property_getters.sort();
        self.property_getters.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdentifierCallees<Function: FunctionTrait> {
    /// When the attribute access is called, the callees it may resolve to
    pub(crate) if_called: CallCallees<Function>,
}

impl<Function: FunctionTrait> IdentifierCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> IdentifierCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        IdentifierCallees {
            if_called: self.if_called.map_function(map),
        }
    }

    fn is_empty(&self) -> bool {
        self.if_called.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.if_called.all_targets()
    }

    fn dedup_and_sort(&mut self) {
        self.if_called.dedup_and_sort();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DefineCallees<Function: FunctionTrait> {
    pub(crate) define_targets: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> DefineCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> DefineCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        DefineCallees {
            define_targets: self
                .define_targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, map))
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.define_targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.define_targets.iter()
    }

    fn dedup_and_sort(&mut self) {
        self.define_targets.sort();
        self.define_targets.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ExpressionCallees<Function: FunctionTrait> {
    Call(CallCallees<Function>),
    Identifier(IdentifierCallees<Function>),
    AttributeAccess(AttributeAccessCallees<Function>),
    Define(DefineCallees<Function>),
}

impl<Function: FunctionTrait> ExpressionCallees<Function> {
    #[cfg(test)]
    pub fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
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
            ExpressionCallees::Define(define_callees) => {
                ExpressionCallees::Define(define_callees.map_function(map))
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
            ExpressionCallees::Define(define_callees) => define_callees.is_empty(),
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
            ExpressionCallees::Define(define_callees) => Box::new(define_callees.all_targets()),
        }
    }

    fn dedup_and_sort(&mut self) {
        match self {
            ExpressionCallees::Call(call_callees) => {
                call_callees.dedup_and_sort();
            }
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                attribute_access_callees.dedup_and_sort();
            }
            ExpressionCallees::Identifier(identifier_callees) => {
                identifier_callees.dedup_and_sort();
            }
            ExpressionCallees::Define(define_callees) => {
                define_callees.dedup_and_sort();
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

    fn dedup_and_sort(&mut self) {
        for callees in self.0.values_mut() {
            callees.dedup_and_sort();
        }
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

    fn dedup_and_sort(&mut self) {
        for callees in self.0.values_mut() {
            callees.dedup_and_sort();
        }
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

fn has_implicit_receiver(
    base_definition: Option<&FunctionBaseDefinition>,
    is_receiver_class_def: bool,
) -> ImplicitReceiver {
    let is_classmethod = base_definition.is_some_and(|definition| definition.is_classmethod);
    let is_staticmethod = base_definition.is_some_and(|definition| definition.is_staticmethod);
    let is_method = base_definition.is_some_and(|definition| definition.is_method());
    if is_staticmethod {
        ImplicitReceiver::False
    } else if is_classmethod {
        if is_receiver_class_def {
            // C.f() where f is a class method
            ImplicitReceiver::TrueWithClassReceiver
        } else {
            // c.f() where f is a class method
            ImplicitReceiver::TrueWithObjectReceiver
        }
    } else if is_method {
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

fn extract_function_from_bound_method(
    bound_method: &BoundMethod,
) -> Vec<&pyrefly_types::callable::Function> {
    match &bound_method.func {
        BoundMethodType::Function(function) => vec![function],
        BoundMethodType::Forall(forall) => vec![&forall.body],
        BoundMethodType::Overload(overload) => overload
            .signatures
            .iter()
            .map(|overload_type| match overload_type {
                OverloadType::Function(function) => function,
                OverloadType::Forall(forall) => &forall.body,
            })
            .collect(),
    }
}

fn find_class_type_for_new_method(new_method_parameters: &Params) -> Option<&Type> {
    // TODO: We assume the first parameter of `__new__` is the class but this may not always be the case.
    match new_method_parameters {
        Params::List(param_list) => param_list.items().first().and_then(|param| match param {
            Param::PosOnly(_, type_, _) => Some(type_),
            Param::Pos(_, type_, _) => Some(type_),
            _ => None,
        }),
        _ => None,
    }
    .and_then(|type_| match type_ {
        Type::Type(type_) => Some(&**type_),
        _ => None,
    })
}

fn method_name_from_function(function: &pyrefly_types::callable::Function) -> Cow<'_, Name> {
    function.metadata.kind.function_name()
}

fn receiver_type_from_callee_type(callee_type: Option<&Type>) -> Option<&Type> {
    match callee_type {
        Some(Type::BoundMethod(bound_method)) => Some(&bound_method.obj),
        _ => None,
    }
}

// Whether the call is non-dynamically dispatched
fn is_direct_call(callee: AnyNodeRef, callee_type: Option<&Type>) -> bool {
    fn is_super_call(callee: AnyNodeRef) -> bool {
        match callee {
            AnyNodeRef::ExprCall(call) => is_super_call(call.func.as_ref().into()),
            AnyNodeRef::ExprName(name) => name.id == "super",
            AnyNodeRef::ExprAttribute(attribute) => is_super_call(attribute.value.as_ref().into()),
            _ => false,
        }
    }

    is_super_call(callee) || {
        match callee_type {
            Some(Type::BoundMethod(box BoundMethod {
                obj: Type::ClassDef(_) | Type::Type(_),
                ..
            })) => true,
            Some(Type::BoundMethod(box BoundMethod {
                obj: Type::ClassType(_) | Type::SelfType(_),
                ..
            })) => {
                // Dynamic dispatch if calling a method via an attribute lookup
                // on an instance
                false
            }
            Some(Type::BoundMethod(bound_method)) => {
                eprintln!(
                    "For callee `{:#?}`, unknown object type in bound method `{:#?}`",
                    callee, bound_method
                );
                // `true` would skip overrides, which may lead to false negatives. But we prefer false positives since
                // we are blind to false negatives.
                false
            }
            Some(Type::Function(_)) => true,
            Some(Type::Union(types)) => is_direct_call(callee, Some(types.first().unwrap())),
            _ => false,
        }
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
            Type::ClassType(class_type)
            | Type::SelfType(class_type)
            | Type::SuperInstance(box (class_type, _)) => (
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

    fn get_base_definition(&self, function_ref: &FunctionRef) -> Option<&FunctionBaseDefinition> {
        self.function_base_definitions
            .get(function_ref.module_id, &function_ref.function_id)
    }

    fn function_ref_from_class_field(
        &self,
        class: &Class,
        field_name: &Name,
    ) -> Option<FunctionRef> {
        let context = get_context_from_class(class, self.module_context);
        let class_field = get_class_field(class, field_name, &context)?;
        let function = FunctionNode::exported_function_from_class_field(
            class,
            field_name,
            class_field,
            &context,
        )?;
        Some(function.as_function_ref(&context))
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
    fn compute_targets_for_virtual_call(
        &self,
        callee_type: Option<&Type>,
        callee: FunctionRef,
    ) -> Vec<Target<FunctionRef>> {
        let receiver_type = receiver_type_from_callee_type(callee_type);
        if receiver_type.is_none() {
            return vec![Target::Function(callee)];
        }
        let receiver_type = receiver_type.unwrap();
        let callee_definition = self.get_base_definition(&callee);
        let (receiver_class, _) = self.receiver_class_from_type(
            receiver_type,
            callee_definition.is_some_and(|definition| definition.is_classmethod),
        );
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
        if callee_class == receiver_class {
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
                        self.function_ref_from_class_field(
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

    fn call_target_from_function_ref(
        &self,
        function_ref: FunctionRef,
        return_type: Option<ScalarTypeProperties>,
        receiver_type: Option<&Type>,
        // For example, `f` in call expr `f(1)` or `__call__` in call expr `c.__call__(1)`
        callee_expr_suffix: Option<&str>,
        is_override_target: bool,
        override_implicit_receiver: Option<ImplicitReceiver>,
    ) -> CallTarget<FunctionRef> {
        let function_definiton = self.get_base_definition(&function_ref);
        let is_classmethod = function_definiton.is_some_and(|definition| definition.is_classmethod);
        let is_staticmethod =
            function_definiton.is_some_and(|definition| definition.is_staticmethod);
        let (receiver_class, is_receiver_class_def) = match receiver_type {
            Some(receiver_type) => self.receiver_class_from_type(receiver_type, is_classmethod),
            None => (None, false),
        };
        CallTarget {
            implicit_receiver: override_implicit_receiver.unwrap_or(has_implicit_receiver(
                function_definiton,
                is_receiver_class_def,
            )),
            receiver_class,
            implicit_dunder_call: function_ref.function_name == dunder::CALL
                && callee_expr_suffix.is_some_and(|suffix| suffix != dunder::CALL.as_str()),
            is_class_method: is_classmethod,
            is_static_method: is_staticmethod || function_ref.function_name == dunder::NEW,
            return_type,
            target: if is_override_target {
                Target::Override(function_ref)
            } else {
                Target::Function(function_ref)
            },
        }
    }

    fn call_targets_from_static_or_virtual_call(
        &self,
        function_ref: FunctionRef,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        precise_receiver_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
        override_implicit_receiver: Option<ImplicitReceiver>,
        override_is_direct_call: Option<bool>,
    ) -> Vec<CallTarget<FunctionRef>> {
        let is_direct_call = match override_is_direct_call {
            Some(override_is_direct_call) => override_is_direct_call,
            None => is_direct_call(callee_expr, callee_type),
        };
        let receiver_type = if precise_receiver_type.is_some() {
            precise_receiver_type
        } else {
            // Since `Type::BoundMethod` does not always has the most precise receiver type, we use it as a fallback
            receiver_type_from_callee_type(callee_type)
        };
        if is_direct_call {
            vec![self.call_target_from_function_ref(
                function_ref,
                return_type,
                receiver_type,
                callee_expr_suffix,
                /* is_override_target */ false,
                override_implicit_receiver,
            )]
        } else {
            self.compute_targets_for_virtual_call(callee_type, function_ref)
                .into_iter()
                .map(|target| match target {
                    Target::Function(function_ref) => self.call_target_from_function_ref(
                        function_ref,
                        return_type,
                        receiver_type,
                        callee_expr_suffix,
                        /* is_override_target */ false,
                        override_implicit_receiver,
                    ),
                    Target::Override(function_ref) => self.call_target_from_function_ref(
                        function_ref,
                        return_type,
                        receiver_type,
                        callee_expr_suffix,
                        /* is_override_target */ true,
                        override_implicit_receiver,
                    ),
                    Target::Object(_) => CallTarget {
                        target,
                        implicit_receiver: ImplicitReceiver::False,
                        receiver_class: None,
                        implicit_dunder_call: false,
                        is_class_method: false,
                        is_static_method: false,
                        return_type,
                    },
                })
                .collect::<Vec<_>>()
        }
    }

    fn call_targets_from_method_name(
        &self,
        method: &Name,
        defining_class: Option<&Type>,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        is_bound_method: bool,
        callee_expr_suffix: Option<&str>,
        override_implicit_receiver: Option<ImplicitReceiver>,
        override_is_direct_call: Option<bool>,
    ) -> Vec<CallTarget<FunctionRef>> {
        let call_targets = match defining_class {
            Some(Type::ClassType(class_type)) => {
                self.function_ref_from_class_field(class_type.class_object(), method)
                    .map(|function_ref| {
                        let receiver_type = if is_bound_method {
                            // For a bound method, its receiver is either `self` or `cls`. For `self`, the receiver
                            // is the defining class. For `cls`, technically the receiver is the type of the class
                            // but we need to be consistent with `receiver_class_from_type`.
                            defining_class
                        } else {
                            None
                        };
                        self.call_targets_from_static_or_virtual_call(
                            function_ref,
                            callee_expr,
                            callee_type,
                            receiver_type,
                            return_type,
                            callee_expr_suffix,
                            override_implicit_receiver,
                            override_is_direct_call,
                        )
                    })
                    .unwrap_or_default()
            }
            Some(Type::Union(types)) => types
                .iter()
                .flat_map(|type_| {
                    self.call_targets_from_method_name(
                        method,
                        Some(type_),
                        callee_expr,
                        callee_type,
                        return_type,
                        is_bound_method,
                        callee_expr_suffix,
                        override_implicit_receiver,
                        override_is_direct_call,
                    )
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        };
        if call_targets.is_empty() {
            debug_println!(
                self.debug,
                "Cannot find call targets for method `{:#?}` in class `{:#?}`",
                method,
                defining_class,
            );
        }
        call_targets
    }

    fn call_targets_from_new_method(
        &self,
        new_method: &pyrefly_types::callable::Function,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> Vec<CallTarget<FunctionRef>> {
        let class_type = find_class_type_for_new_method(&new_method.signature.params);
        self.call_targets_from_method_name(
            &method_name_from_function(new_method),
            class_type,
            callee_expr,
            callee_type,
            return_type,
            /* is_bound_method */ false,
            callee_expr_suffix,
            /* override_implicit_receiver*/ None,
            /* override_is_direct_call */ None,
        )
    }

    fn resolve_constructor_callees(
        &self,
        init_method: Option<Type>,
        new_method: Option<Type>,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> CallCallees<FunctionRef> {
        let object_class = self.module_context.stdlib.object();
        let object_init_method = || {
            self.call_targets_from_method_name(
                &dunder::INIT,
                Some(&Type::ClassType(object_class.clone())),
                callee_expr,
                callee_type,
                return_type,
                /* is_bound_method */ false,
                callee_expr_suffix,
                /* override_implicit_receiver*/
                Some(ImplicitReceiver::TrueWithObjectReceiver),
                /* override_is_direct_call */
                Some(true), // Too expensive to merge models for overrides on `__init__`
            )
        };
        let object_new_method = || {
            self.call_targets_from_method_name(
                &dunder::NEW,
                Some(&Type::ClassType(object_class.clone())),
                callee_expr,
                callee_type,
                return_type,
                /* is_bound_method */ false,
                callee_expr_suffix,
                /* override_implicit_receiver*/ None,
                /* override_is_direct_call */
                Some(true), // Too expensive to merge models for overrides on `__new__`
            )
        };

        let mut init_targets = init_method.as_ref().map_or(
            object_init_method().into_iter().collect::<Vec<_>>(),
            |init_method| match init_method {
                Type::BoundMethod(bound_method) => {
                    extract_function_from_bound_method(bound_method)
                        .into_iter()
                        .flat_map(|function| {
                            self.call_targets_from_method_name(
                                &method_name_from_function(function),
                                Some(&bound_method.obj),
                                callee_expr,
                                callee_type,
                                return_type,
                                /* is_bound_method */ true,
                                callee_expr_suffix,
                                /* override_implicit_receiver*/ None,
                                /* override_is_direct_call */ None,
                            )
                        })
                        .collect::<Vec<_>>()
                }
                _ => vec![],
            },
        );
        if init_method.is_some() && init_targets.is_empty() {
            // TODO(T243217129): Remove this to treat the callees as obscure when unresolved
            init_targets.extend(object_init_method());
        }

        let mut new_targets = new_method.as_ref().map_or(
            object_new_method().into_iter().collect::<Vec<_>>(),
            |new_method| match new_method {
                Type::Function(function) => self
                    .call_targets_from_new_method(
                        function,
                        callee_expr,
                        callee_type,
                        return_type,
                        callee_expr_suffix,
                    )
                    .into_iter()
                    .collect::<Vec<_>>(),
                Type::Overload(overload) => overload
                    .signatures
                    .iter()
                    .flat_map(|overload_type| {
                        let function = match overload_type {
                            OverloadType::Function(function) => function,
                            OverloadType::Forall(forall) => &forall.body,
                        };
                        self.call_targets_from_new_method(
                            function,
                            callee_expr,
                            callee_type,
                            return_type,
                            callee_expr_suffix,
                        )
                    })
                    .collect::<Vec<_>>(),
                _ => vec![],
            },
        );
        if new_method.is_some() && new_targets.is_empty() {
            // TODO(T243217129): Remove this to treat the callees as obscure when unresolved
            new_targets.extend(object_new_method());
        }

        CallCallees {
            call_targets: vec![],
            init_targets,
            new_targets,
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        }
    }

    fn resolve_pyrefly_target(
        &self,
        pyrefly_target: Option<&crate::alt::call::CallTarget>,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> Option<CallCallees<FunctionRef>> {
        match pyrefly_target {
            Some(crate::alt::call::CallTarget::BoundMethod(type_, target)) => {
                // Calling a method on a class instance.
                let call_targets = self
                    .call_targets_from_method_name(
                        &method_name_from_function(&target.1),
                        Some(type_),
                        callee_expr,
                        callee_type,
                        return_type,
                        /* is_bound_method */ true,
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                    )
                    .into_iter()
                    .collect::<Vec<_>>();
                Some(CallCallees {
                    call_targets,
                    init_targets: vec![],
                    new_targets: vec![],
                    higher_order_parameters: HashMap::new(),
                    unresolved: Unresolved::False,
                })
            }
            Some(crate::alt::call::CallTarget::Function(function)) => {
                // Sometimes this means calling a function (e.g., static method) on a class instance. Sometimes
                // this could be simply calling a module top-level function, which can be handled when the stack
                // of D85441657 enables uniquely identifying a definition from a type.
                let call_targets = self
                    .call_targets_from_method_name(
                        &method_name_from_function(&function.1),
                        callee_type, // For static methods, we find them within the callee type
                        callee_expr,
                        callee_type,
                        return_type,
                        /* is_bound_method */ false,
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                    )
                    .into_iter()
                    .collect::<Vec<_>>();
                Some(CallCallees {
                    call_targets,
                    init_targets: vec![],
                    new_targets: vec![],
                    higher_order_parameters: HashMap::new(),
                    unresolved: Unresolved::False,
                })
            }
            Some(crate::alt::call::CallTarget::Class(class_type, _)) => {
                // Constructing a class instance.
                self.module_context
                    .transaction
                    .ad_hoc_solve(&self.module_context.handle, |solver| {
                        let new_method = solver.get_dunder_new(class_type);
                        let overrides_new = new_method.is_some();
                        let init_method = solver
                            .get_dunder_init(class_type, /* get_object_init */ !overrides_new);
                        (init_method, new_method)
                    })
                    .map(|(init_method, new_method)| {
                        self.resolve_constructor_callees(
                            init_method,
                            new_method,
                            callee_expr,
                            callee_type,
                            return_type,
                            callee_expr_suffix,
                        )
                    })
            }
            Some(crate::alt::call::CallTarget::Union(targets)) => targets
                .iter()
                .flat_map(|target| {
                    self.resolve_pyrefly_target(
                        Some(target),
                        callee_expr,
                        callee_type,
                        return_type,
                        callee_expr_suffix,
                    )
                })
                .reduce(|mut so_far, call_target| {
                    so_far.join_in_place(call_target);
                    so_far
                }),
            _ => {
                debug_println!(
                    self.debug,
                    "Unrecognized pyrefly target `{:#?}` for `{:#?}`",
                    pyrefly_target,
                    callee_expr,
                );
                None
            }
        }
    }

    fn resolve_repr(
        &self,
        function_ref: FunctionRef,
        call_arguments: Option<&ruff_python_ast::Arguments>,
        callee_expr: AnyNodeRef,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> Vec<CallTarget<FunctionRef>> {
        if function_ref.module_name == ModuleName::builtins()
            && function_ref.function_name == "repr"
        {
            // Find the actual `__repr__`
            let actual_repr = call_arguments
                .as_ref()
                .and_then(|arguments| {
                    arguments.find_positional(0).and_then(|argument| {
                        self.module_context.answers.get_type_trace(argument.range())
                    })
                })
                .map(|first_argument_type| {
                    self.call_targets_from_method_name(
                        &Name::new_static("__repr__"),
                        Some(&first_argument_type),
                        callee_expr,
                        callee_type,
                        return_type,
                        /* is_bound_method */ true,
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                    )
                });
            if actual_repr
                .as_ref()
                .is_some_and(|actual_repr| !actual_repr.is_empty())
            {
                return actual_repr.unwrap();
            }
        }
        self.call_targets_from_static_or_virtual_call(
            function_ref.clone(),
            callee_expr,
            callee_type,
            /* precise_receiver_type */ None,
            return_type,
            callee_expr_suffix,
            /* override_implicit_receiver*/ None,
            /* override_is_direct_call */ None,
        )
    }

    fn resolve_name(
        &self,
        name: &ExprName,
        call_arguments: Option<&ruff_python_ast::Arguments>,
        return_type: Option<ScalarTypeProperties>,
    ) -> Option<CallCallees<FunctionRef>> {
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
            })
            .collect::<Vec<_>>();

        let callee_type = self.module_context.answers.get_type_trace(name.range());
        let callee_expr_suffix = Some(name.id.as_str());
        if !go_to_definitions.is_empty() {
            let go_to_definitions = go_to_definitions
                .into_iter()
                .flat_map(|function_ref| {
                    self.resolve_repr(
                        function_ref,
                        call_arguments,
                        AnyNodeRef::from(name),
                        callee_type.as_ref(),
                        return_type,
                        callee_expr_suffix,
                    )
                })
                .collect::<Vec<_>>();
            Some(CallCallees {
                call_targets: go_to_definitions,
                init_targets: vec![],
                new_targets: vec![],
                higher_order_parameters: HashMap::new(),
                unresolved: Unresolved::False,
            })
        } else {
            // There is no go-to-definition when for example an `ExprName` is a class definition,
            // a local variable, or a parameter.
            let pyrefly_target = self
                .module_context
                .transaction
                .ad_hoc_solve(&self.module_context.handle, |solver| {
                    callee_type
                        .as_ref()
                        .and_then(|type_| solver.as_call_target(type_.clone()))
                })
                .flatten();
            self.resolve_pyrefly_target(
                pyrefly_target.as_ref(),
                AnyNodeRef::from(name),
                callee_type.as_ref(),
                return_type,
                callee_expr_suffix,
            )
        }
    }

    fn resolve_attribute_access(
        &self,
        attribute: &ExprAttribute,
        return_type: Option<ScalarTypeProperties>,
        assignment_targets: Option<&Vec<&Expr>>,
    ) -> Option<AttributeAccessCallees<FunctionRef>> {
        let (property_callees, non_property_callees): (Vec<FunctionRef>, Vec<FunctionRef>) = self
            .module_context
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
            })
            .partition(|function_ref| {
                self.get_base_definition(function_ref)
                    .is_some_and(|definition| {
                        definition.is_property_getter || definition.is_property_setter
                    })
            });

        let in_assignment_lhs = assignment_targets.is_some_and(|assignment_targets| {
            assignment_targets
                .iter()
                .any(|assignment_target| match assignment_target {
                    Expr::Attribute(assignment_target_attribute) => {
                        assignment_target_attribute.range() == attribute.range()
                    }
                    _ => false,
                })
        });
        // Go-to-definition always resolves to the property getters, even when used as a left hand side of an
        // assignment. Therefore we use heuristics to differentiate setters from getters.
        let (property_setters, property_getters) = if in_assignment_lhs {
            (property_callees, vec![])
        } else {
            (vec![], property_callees)
        };

        let callee_expr_suffix = Some(attribute.attr.id.as_str());
        let receiver_type = self
            .module_context
            .answers
            .get_type_trace(attribute.value.range());
        let callee_type = self
            .module_context
            .answers
            .get_type_trace(attribute.range());
        let callee_expr = AnyNodeRef::from(attribute);
        Some(AttributeAccessCallees {
            if_called: CallCallees {
                call_targets: non_property_callees
                    .into_iter()
                    .flat_map(|function| {
                        self.call_targets_from_static_or_virtual_call(
                            function,
                            callee_expr,
                            callee_type.as_ref(),
                            receiver_type.as_ref(),
                            return_type,
                            callee_expr_suffix,
                            /* override_implicit_receiver*/ None,
                            /* override_is_direct_call */ None,
                        )
                    })
                    .collect(),
                init_targets: vec![],
                new_targets: vec![],
                higher_order_parameters: HashMap::new(),
                unresolved: Unresolved::False,
            },
            property_setters: property_setters
                .into_iter()
                .flat_map(|function| {
                    self.call_targets_from_static_or_virtual_call(
                        function,
                        callee_expr,
                        callee_type.as_ref(),
                        receiver_type.as_ref(),
                        Some(ScalarTypeProperties::none()),
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                    )
                })
                .collect(),
            property_getters: {
                // We cannot get the return types by treating the property getter expressions as callable types.
                // Hence we use the types of the whole expressions.
                let return_type = self
                    .module_context
                    .answers
                    .get_type_trace(attribute.range())
                    .map(|type_| ScalarTypeProperties::from_type(&type_, self.module_context));
                property_getters
                    .into_iter()
                    .flat_map(|function| {
                        self.call_targets_from_static_or_virtual_call(
                            function,
                            callee_expr,
                            callee_type.as_ref(),
                            receiver_type.as_ref(),
                            return_type,
                            callee_expr_suffix,
                            /* override_implicit_receiver*/ None,
                            /* override_is_direct_call */ None,
                        )
                    })
                    .collect()
            },
        })
    }

    fn resolve_higher_order_parameters(
        &self,
        call_arguments: &Arguments,
    ) -> HashMap<u32, HigherOrderParameter<FunctionRef>> {
        // TODO: Filter the results with `filter_implicit_dunder_calls`
        call_arguments
            .arguments_source_order()
            .enumerate()
            .filter_map(|(index, argument)| {
                let argument = match argument {
                    ArgOrKeyword::Arg(argument) => argument,
                    ArgOrKeyword::Keyword(keyword) => &keyword.value,
                };
                let index = index.try_into().unwrap();
                match argument {
                    Expr::Lambda(_) => Some((
                        index,
                        HigherOrderParameter {
                            index,
                            call_targets: vec![],
                            unresolved: Unresolved::True(UnresolvedReason::LambdaArgument),
                        },
                    )),
                    _ => {
                        self.resolve_expression(
                            argument, /* parent_expression */ None,
                            /* assignment_targets */ None,
                        )
                        .map(|callees| {
                            let (call_targets, unresolved) = match callees {
                                ExpressionCallees::Call(callees) => {
                                    (callees.call_targets, callees.unresolved)
                                }
                                ExpressionCallees::AttributeAccess(callees) => (
                                    [
                                        callees.if_called.call_targets,
                                        callees.property_getters,
                                        callees.property_setters,
                                    ]
                                    .concat(),
                                    callees.if_called.unresolved,
                                ),
                                ExpressionCallees::Identifier(callees) => {
                                    (callees.if_called.call_targets, callees.if_called.unresolved)
                                }
                                ExpressionCallees::Define(_) => unreachable!(),
                            };
                            (
                                index,
                                HigherOrderParameter {
                                    index,
                                    call_targets,
                                    unresolved,
                                },
                            )
                        })
                    }
                }
            })
            .collect()
    }

    fn resolve_call(
        &self,
        call: &ExprCall,
        assignment_targets: Option<&Vec<&Expr>>,
    ) -> Option<CallCallees<FunctionRef>> {
        let return_type = self
            .module_context
            .answers
            .get_type_trace(call.range())
            .map(|type_| ScalarTypeProperties::from_type(&type_, self.module_context));

        let higher_order_parameters = self.resolve_higher_order_parameters(&call.arguments);

        let mut callees = match &*call.func {
            Expr::Name(name) => {
                let callees = self.resolve_name(name, Some(&call.arguments), return_type);
                debug_println!(
                    self.debug,
                    "Resolved call `{:#?}` into `{:#?}`",
                    call,
                    callees
                );
                callees
            }
            Expr::Attribute(attribute) => {
                let callees =
                    self.resolve_attribute_access(attribute, return_type, assignment_targets);
                debug_println!(
                    self.debug,
                    "Resolved call `{:#?}` into `{:#?}`",
                    call,
                    callees
                );
                callees.map(|callees| callees.if_called)
            }
            _ => None,
        };
        if let Some(callees) = callees.as_mut() {
            callees.with_higher_order_parameters(higher_order_parameters);
        }
        callees
    }

    // Use this only when we are not analyzing a call expression (e.g., `foo` in `x = foo`), because
    // for a call expression, we could simply query its type (e.g., query the type of `c(1)`).
    fn get_return_type_for_callee(&self, callee_type: Option<&Type>) -> ScalarTypeProperties {
        let return_type_from_function = |function: &pyrefly_types::callable::Function| {
            ScalarTypeProperties::from_type(&function.signature.ret, self.module_context)
        };
        callee_type.map_or(ScalarTypeProperties::none(), |type_| match type_ {
            Type::Function(function) => return_type_from_function(function),
            Type::BoundMethod(bound_method) => extract_function_from_bound_method(bound_method)
                .into_iter()
                .map(return_type_from_function)
                .reduce(|so_far, property| so_far.join(property))
                .unwrap_or(ScalarTypeProperties::none()),
            Type::Callable(callable) => {
                ScalarTypeProperties::from_type(&callable.ret, self.module_context)
            }
            _ => ScalarTypeProperties::none(),
        })
    }

    fn resolve_expression(
        &self,
        expr: &Expr,
        parent_expression: Option<&Expr>,
        assignment_targets: Option<&Vec<&Expr>>,
    ) -> Option<ExpressionCallees<FunctionRef>> {
        debug_println!(self.debug, "Resolving callees for expression `{:#?}`", expr);
        let is_nested_callee_or_base =
            parent_expression.is_some_and(|parent_expression| match parent_expression {
                // For example, avoid visiting `x.__call__` in `x.__call__(1)`
                Expr::Call(callee) if expr.range() == callee.func.range() => true,
                // For example, avoid visiting `x` in `x.__call__` or `x.y` in `x.y.__call__`
                Expr::Attribute(_) => true,
                _ => false,
            });
        let return_type_when_called = || {
            self.get_return_type_for_callee(
                self.module_context
                    .answers
                    .get_type_trace(expr.range())
                    .as_ref(),
            )
        };
        match expr {
            Expr::Call(call) => self
                .resolve_call(call, assignment_targets)
                .map(ExpressionCallees::Call),
            Expr::Name(name) if !is_nested_callee_or_base => self
                .resolve_name(
                    name,
                    /* call_arguments */ None,
                    Some(return_type_when_called()),
                )
                .map(|call_callees| {
                    ExpressionCallees::Identifier(IdentifierCallees {
                        if_called: call_callees,
                    })
                }),
            Expr::Attribute(attribute) if !is_nested_callee_or_base => self
                .resolve_attribute_access(
                    attribute,
                    Some(return_type_when_called()),
                    assignment_targets,
                )
                .map(ExpressionCallees::AttributeAccess),
            _ => None,
        }
    }

    fn resolve_function_def(
        &self,
        function_def: &StmtFunctionDef,
    ) -> Option<ExpressionCallees<FunctionRef>> {
        let is_inner_function = match self.current_function {
            Some(FunctionRef {
                module_id: _,
                module_name: _,
                function_id: FunctionId::ClassTopLevel { .. },
                function_name: _,
            }) => false,
            Some(FunctionRef {
                module_id: _,
                module_name: _,
                function_id: FunctionId::ModuleTopLevel,
                function_name: _,
            }) => false,
            Some(FunctionRef {
                module_id: _,
                module_name: _,
                function_id: FunctionId::FunctionDecoratedTarget { .. },
                function_name: _,
            }) => false,
            _ => true,
        };
        if !is_inner_function {
            return None;
        }
        let key = KeyDecoratedFunction(ShortIdentifier::new(&function_def.name));
        self.module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key))
            .and_then(|idx| {
                let decorated_function = DecoratedFunction::from_bindings_answers(
                    idx,
                    &self.module_context.bindings,
                    &self.module_context.answers,
                );
                if should_export_decorated_function(&decorated_function, self.module_context) {
                    let return_type = decorated_function
                        .ty
                        .callable_return_type()
                        .map(|type_| ScalarTypeProperties::from_type(&type_, self.module_context));
                    let target = self.call_target_from_function_ref(
                        FunctionRef::from_decorated_function(
                            &decorated_function,
                            self.module_context,
                        ),
                        return_type,
                        /* receiver_type */ None,
                        /* callee_expr_suffix */ None,
                        /* is_override_target */ false,
                        /* override_implicit_receiver*/ None,
                    );
                    Some(ExpressionCallees::Define(DefineCallees {
                        define_targets: vec![target],
                    }))
                } else {
                    None
                }
            })
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
            ScopeExportedFunctionFlags {
                include_top_level: true,
                include_class_top_level: true,
                include_function_decorators:
                    super::ast_visitor::ExportFunctionDecorators::InDecoratedTarget,
                include_class_decorators: super::ast_visitor::ExportClassDecorators::InParentScope,
                include_default_arguments: super::ast_visitor::ExportDefaultArguments::InFunction,
            },
        );
        if let Some(current_function) = &self.current_function {
            // Always insert an empty call graph for the function.
            // This way we can error on missing call graphs in Pysa.
            self.call_graphs
                .0
                .entry(current_function.clone())
                .or_default();
        }
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

    fn visit_type_annotations() -> bool {
        false
    }

    fn visit_expression(
        &mut self,
        expr: &Expr,
        _: &Scopes,
        parent_expression: Option<&Expr>,
        assignment_targets: Option<&Vec<&Expr>>,
    ) {
        if self.current_function.is_none() {
            return;
        }

        let callees = self.resolve_expression(expr, parent_expression, assignment_targets);
        if let Some(callees) = callees
            && !callees.is_empty()
        {
            self.add_callees(expr.range(), callees);
        }
    }

    fn visit_statement(&mut self, stmt: &Stmt, _scopes: &Scopes) {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                let callees = self.resolve_function_def(function_def);
                if let Some(callees) = callees
                    && !callees.is_empty()
                {
                    self.add_callees(function_def.range(), callees);
                }
            }
            _ => {}
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
    visitor
        .resolve_call(call, /* assignment_targets */ None)
        .unwrap_or(CallCallees {
            call_targets: vec![],
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        })
}

fn resolve_expression(
    expression: &Expr,
    function_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    module_context: &ModuleContext,
    override_graph: &OverrideGraph,
    parent_expression: Option<&Expr>,
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
    visitor.resolve_expression(
        expression,
        parent_expression,
        /* assignment_targets */ None,
    )
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

    let is_object_new_or_init_target = |target: &Target<FunctionRef>| match target {
        Target::Function(function_ref) | Target::Override(function_ref) => {
            function_ref.module_name == ModuleName::builtins()
                && (function_ref.function_name == dunder::INIT
                    || function_ref.function_name == dunder::NEW)
        }
        Target::Object(_) => false,
    };

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
                        .filter(|call_target| !is_object_new_or_init_target(call_target))
                        .collect(),
                )
            }
            expr => {
                let callees = resolve_expression(
                    expr,
                    function_base_definitions,
                    context,
                    &override_graph,
                    /* parent_expression */ None,
                );
                (
                    expr.range(),
                    if let Some(callees) = callees {
                        callees
                            .all_targets()
                            .map(|call_target| call_target.target.clone())
                            .filter(|call_target| !is_object_new_or_init_target(call_target))
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
    call_graphs.dedup_and_sort();
    call_graphs
}
