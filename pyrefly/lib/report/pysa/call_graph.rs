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

use dupe::Dupe;
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
use ruff_python_ast::Comprehension;
use ruff_python_ast::ConversionFlag;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprFString;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtWith;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use starlark_map::Hashed;
use vec1::Vec1;

use crate::alt::call::CallTargetLookup;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::KeyDecoratedFunction;
use crate::error::collector::ErrorCollector;
use crate::error::style::ErrorStyle;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::ScopeExportedFunctionFlags;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::get_class_field;
use crate::report::pysa::class::get_context_from_class;
use crate::report::pysa::collect::CollectNoDuplicateKeys;
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Hash, PartialOrd, Ord)]
pub enum OriginKind {
    GetAttrConstantLiteral,
    ComparisonOperator,
    GeneratorIter,
    GeneratorNext,
    WithEnter,
    ForDecoratedTarget,
    SubscriptGetItem,
    SubscriptSetItem,
    FormatStringArtificial,
    FormatStringStringify,
}

impl std::fmt::Display for OriginKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GetAttrConstantLiteral => write!(f, "get-attr-constant-literal"),
            Self::ComparisonOperator => write!(f, "comparison"),
            Self::GeneratorIter => write!(f, "generator-iter"),
            Self::GeneratorNext => write!(f, "generator-next"),
            Self::WithEnter => write!(f, "with-enter"),
            Self::ForDecoratedTarget => write!(f, "for-decorated-target"),
            Self::SubscriptGetItem => write!(f, "subscript-get-item"),
            Self::SubscriptSetItem => write!(f, "subscript-set-item"),
            Self::FormatStringArtificial => write!(f, "format-string-artificial"),
            Self::FormatStringStringify => write!(f, "format-string-stringify"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Hash, PartialOrd, Ord)]
pub struct Origin {
    kind: OriginKind,
    location: PysaLocation,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, PartialOrd, Ord)]
pub enum ExpressionIdentifier {
    Regular(PysaLocation),
    ArtificialAttributeAccess(Origin),
    ArtificialCall(Origin),
}

impl std::fmt::Display for ExpressionIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular(location) => write!(f, "{}", location.as_key()),
            Self::ArtificialAttributeAccess(Origin { kind, location }) => {
                write!(
                    f,
                    "{}|artificial-attribute-access|{}",
                    location.as_key(),
                    kind,
                )
            }
            Self::ArtificialCall(Origin { kind, location }) => {
                write!(f, "{}|artificial-call|{}", location.as_key(), kind,)
            }
        }
    }
}

impl ExpressionIdentifier {
    pub fn as_key(&self) -> String {
        format!("{}", self)
    }

    fn regular(location: TextRange, module: &pyrefly_python::module::Module) -> Self {
        ExpressionIdentifier::Regular(PysaLocation::new(module.display_range(location)))
    }
}

pub trait ExpressionIdTrait:
    std::fmt::Debug + PartialEq + Eq + Clone + Hash + Serialize + PartialOrd + Ord
{
}

impl ExpressionIdTrait for ExpressionIdentifier {}

impl ExpressionIdTrait for String {}

impl Serialize for ExpressionIdentifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.as_key())
    }
}

#[derive(Debug)]
struct DunderAttrCallees {
    callees: CallCallees<FunctionRef>,
    attr_type: Option<Type>,
}

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
    FormatString,
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
            Target::FormatString => Target::FormatString,
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

    fn format_string_target() -> Self {
        CallTarget {
            target: Target::FormatString,
            return_type: None,
            implicit_receiver: ImplicitReceiver::False,
            implicit_dunder_call: false,
            receiver_class: None,
            is_class_method: false,
            is_static_method: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnresolvedReason {
    // Argument is a lambda.
    LambdaArgument,
    // Unexpected pyrefly::CallTarget type.
    UnexpectedPyreflyTarget,
    // Empty pyrefly::CallTarget type.
    EmptyPyreflyTarget,
    // Could not find the given field on a class.
    UnknownClassField,
    // Failure to create a target from a pyrefly::CallTarget::Function.
    UnsupportedFunctionTarget,
    // Unexpected type, expecting a class or union of classes.
    UnexpectedDefiningClass,
    // Unexpected __init__ method returned by pyrefly.
    UnexpectedInitMethod,
    // Unexpected __new__ method returned by pyrefly.
    UnexpectedNewMethod,
    // Unexpected expression for the callee (currently handle name and attribute access only).
    UnexpectedCalleeExpression,
    // Pyrefly failed to resolved a magic dunder attribute.
    UnresolvedMagicDunderAttr,
    // No base type when trying to resolve a magic dunder attribute on this type.
    UnresolvedMagicDunderAttrDueToNoBase,
    // No attribute when trying to resolve a magic dunder attribute on a type.
    UnresolvedMagicDunderAttrDueToNoAttribute,
    // Set of different reasons.
    Mixed,
}

impl UnresolvedReason {
    fn join(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            UnresolvedReason::Mixed
        }
    }
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
        match (self, other) {
            (Unresolved::True(left), Unresolved::True(right)) => Unresolved::True(left.join(right)),
            (left @ Unresolved::True(..), Unresolved::False) => left,
            (Unresolved::False, right @ Unresolved::True(..)) => right,
            (Unresolved::False, Unresolved::False) => Unresolved::False,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaybeResolved<T> {
    Resolved(T),
    PartiallyResolved(T, UnresolvedReason),
    Unresolved(UnresolvedReason),
}

impl<T> MaybeResolved<T> {
    fn is_unresolved(&self) -> bool {
        matches!(self, MaybeResolved::Unresolved(_))
    }

    fn flatten(self) -> (Option<T>, Unresolved) {
        match self {
            MaybeResolved::Resolved(resolved) => (Some(resolved), Unresolved::False),
            MaybeResolved::PartiallyResolved(resolved, unresolved) => {
                (Some(resolved), Unresolved::True(unresolved))
            }
            MaybeResolved::Unresolved(unresolved) => (None, Unresolved::True(unresolved)),
        }
    }
}

impl<T> MaybeResolved<Vec1<T>> {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (MaybeResolved::Resolved(mut resolved), MaybeResolved::Resolved(other)) => {
                resolved.extend(other);
                MaybeResolved::Resolved(resolved)
            }
            (
                MaybeResolved::Resolved(mut resolved),
                MaybeResolved::PartiallyResolved(other_resolved, other_unresolved),
            ) => {
                resolved.extend(other_resolved);
                MaybeResolved::PartiallyResolved(resolved, other_unresolved)
            }
            (MaybeResolved::Resolved(resolved), MaybeResolved::Unresolved(unresolved)) => {
                MaybeResolved::PartiallyResolved(resolved, unresolved)
            }
            (
                MaybeResolved::PartiallyResolved(mut resolved, unresolved),
                MaybeResolved::PartiallyResolved(other_resolved, other_unresolved),
            ) => {
                resolved.extend(other_resolved);
                MaybeResolved::PartiallyResolved(resolved, unresolved.join(other_unresolved))
            }
            (
                MaybeResolved::PartiallyResolved(resolved, unresolved),
                MaybeResolved::Unresolved(other_unresolved),
            ) => MaybeResolved::PartiallyResolved(resolved, unresolved.join(other_unresolved)),
            (
                MaybeResolved::Unresolved(unresolved),
                MaybeResolved::Unresolved(other_unresolved),
            ) => MaybeResolved::Unresolved(unresolved.join(other_unresolved)),
            (left, right) => right.join(left),
        }
    }
}

impl MaybeResolved<Vec1<CallTarget<FunctionRef>>> {
    fn into_call_callees(self) -> CallCallees<FunctionRef> {
        match self {
            MaybeResolved::Resolved(call_targets) => CallCallees {
                call_targets: call_targets.into_vec(),
                new_targets: vec![],
                init_targets: vec![],
                higher_order_parameters: HashMap::new(),
                unresolved: Unresolved::False,
            },
            MaybeResolved::PartiallyResolved(call_targets, unresolved) => CallCallees {
                call_targets: call_targets.into_vec(),
                new_targets: vec![],
                init_targets: vec![],
                higher_order_parameters: HashMap::new(),
                unresolved: Unresolved::True(unresolved),
            },
            MaybeResolved::Unresolved(unresolved) => CallCallees {
                call_targets: vec![],
                new_targets: vec![],
                init_targets: vec![],
                higher_order_parameters: HashMap::new(),
                unresolved: Unresolved::True(unresolved),
            },
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

    fn dedup_and_sort(&mut self) {
        self.call_targets.sort();
        self.call_targets.dedup();
    }

    fn join_in_place(&mut self, other: Self) {
        assert_eq!(self.index, other.index);
        self.call_targets.extend(other.call_targets);
        self.unresolved = self.unresolved.clone().join(other.unresolved);
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
    fn new(call_targets: Vec1<CallTarget<Function>>) -> Self {
        CallCallees {
            call_targets: call_targets.into_vec(),
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        }
    }

    pub fn new_unresolved(unresolved: UnresolvedReason) -> Self {
        CallCallees {
            call_targets: vec![],
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::True(unresolved),
        }
    }

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
        self.call_targets.is_empty()
            && self.init_targets.is_empty()
            && self.new_targets.is_empty()
            && self.higher_order_parameters.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.call_targets
            .iter()
            .chain(self.init_targets.iter())
            .chain(self.new_targets.iter())
            .chain(
                self.higher_order_parameters
                    .values()
                    .flat_map(|higher_order_parameter| higher_order_parameter.call_targets.iter()),
            )
    }

    fn dedup_and_sort(&mut self) {
        self.call_targets.sort();
        self.call_targets.dedup();
        self.init_targets.sort();
        self.init_targets.dedup();
        self.new_targets.sort();
        self.new_targets.dedup();
        self.higher_order_parameters
            .values_mut()
            .for_each(|higher_order_parameter| higher_order_parameter.dedup_and_sort());
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
        for (index, higher_order_parameter) in other.higher_order_parameters.into_iter() {
            self.higher_order_parameters
                .entry(index)
                .and_modify(|existing| existing.join_in_place(higher_order_parameter.clone()))
                .or_insert(higher_order_parameter);
        }
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

    #[allow(dead_code)]
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
pub struct FormatStringArtificialCallees<Function: FunctionTrait> {
    pub(crate) targets: Vec<CallTarget<Function>>,
}

impl<Function: FunctionTrait> FormatStringArtificialCallees<Function> {
    #[cfg(test)]
    fn map_function<OutputFunction: FunctionTrait, MapFunction>(
        self,
        map: &MapFunction,
    ) -> FormatStringArtificialCallees<OutputFunction>
    where
        MapFunction: Fn(Function) -> OutputFunction,
    {
        FormatStringArtificialCallees {
            targets: self
                .targets
                .into_iter()
                .map(|call_target| CallTarget::map_function(call_target, map))
                .collect(),
        }
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub fn all_targets(&self) -> impl Iterator<Item = &CallTarget<Function>> {
        self.targets.iter()
    }

    fn dedup_and_sort(&mut self) {
        self.targets.sort();
        self.targets.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ExpressionCallees<Function: FunctionTrait> {
    Call(CallCallees<Function>),
    Identifier(IdentifierCallees<Function>),
    AttributeAccess(AttributeAccessCallees<Function>),
    Define(DefineCallees<Function>),
    FormatStringArtificial(FormatStringArtificialCallees<Function>),
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
            ExpressionCallees::FormatStringArtificial(callees) => {
                ExpressionCallees::FormatStringArtificial(callees.map_function(map))
            }
        }
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        match self {
            ExpressionCallees::Call(call_callees) => call_callees.is_empty(),
            ExpressionCallees::Identifier(identifier_callees) => identifier_callees.is_empty(),
            ExpressionCallees::AttributeAccess(attribute_access_callees) => {
                attribute_access_callees.is_empty()
            }
            ExpressionCallees::Define(define_callees) => define_callees.is_empty(),
            ExpressionCallees::FormatStringArtificial(callees) => callees.is_empty(),
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
            ExpressionCallees::FormatStringArtificial(callees) => Box::new(callees.all_targets()),
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
            ExpressionCallees::FormatStringArtificial(callees) => {
                callees.dedup_and_sort();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallGraph<ExpressionId: ExpressionIdTrait, Function: FunctionTrait>(
    HashMap<ExpressionId, ExpressionCallees<Function>>,
);

impl<ExpressionId: ExpressionIdTrait, Function: FunctionTrait> CallGraph<ExpressionId, Function> {
    #[cfg(test)]
    pub fn from_map(map: HashMap<ExpressionId, ExpressionCallees<Function>>) -> Self {
        Self(map)
    }

    #[cfg(test)]
    pub fn into_iter(self) -> impl Iterator<Item = (ExpressionId, ExpressionCallees<Function>)> {
        self.0.into_iter()
    }

    fn dedup_and_sort(&mut self) {
        for callees in self.0.values_mut() {
            callees.dedup_and_sort();
        }
    }
}

impl<ExpressionId: ExpressionIdTrait, Function: FunctionTrait> Default
    for CallGraph<ExpressionId, Function>
{
    fn default() -> Self {
        Self(HashMap::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallGraphs<ExpressionId: ExpressionIdTrait, Function: FunctionTrait>(
    HashMap<Function, CallGraph<ExpressionId, Function>>,
);

impl<ExpressionId: ExpressionIdTrait, Function: FunctionTrait> CallGraphs<ExpressionId, Function> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[cfg(test)]
    pub fn from_map(map: HashMap<Function, CallGraph<ExpressionId, Function>>) -> Self {
        Self(map)
    }

    pub fn into_iter(self) -> impl Iterator<Item = (Function, CallGraph<ExpressionId, Function>)> {
        self.0.into_iter()
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

    fn add_callees(
        &mut self,
        function: Function,
        expression_identifier: ExpressionId,
        callees: ExpressionCallees<Function>,
    ) {
        assert!(
            self.0
                .entry(function)
                .or_default()
                .0
                .insert(expression_identifier, callees)
                .is_none(),
            "Adding callees to the same location"
        );
    }

    fn remove_callees(&mut self, function: Function, expression_identifier: ExpressionId) {
        self.0
            .entry(function)
            .or_default()
            .0
            .remove(&expression_identifier);
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
) -> Vec1<&pyrefly_types::callable::Function> {
    match &bound_method.func {
        BoundMethodType::Function(function) => Vec1::new(function),
        BoundMethodType::Forall(forall) => Vec1::new(&forall.body),
        BoundMethodType::Overload(overload) => Vec1::try_from_vec(
            overload
                .signatures
                .iter()
                .map(|overload_type| match overload_type {
                    OverloadType::Function(function) => function,
                    OverloadType::Forall(forall) => &forall.body,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap(),
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

fn assignment_targets(statement: Option<&Stmt>) -> Option<&[Expr]> {
    match statement {
        Some(Stmt::Assign(assign)) => Some(&assign.targets),
        Some(Stmt::AugAssign(assign)) => Some(std::slice::from_ref(assign.target.as_ref())),
        Some(Stmt::AnnAssign(assign)) => Some(std::slice::from_ref(assign.target.as_ref())),
        _ => None,
    }
}

enum DirectCall {
    True,
    False,
    UnknownCallee,
}

impl DirectCall {
    fn from_bool(b: bool) -> Self {
        if b { Self::True } else { Self::False }
    }

    fn or(&self, other: Self) -> Self {
        match (self, other) {
            // When there is sufficient evidence to tell whether this is a direct call, use that answer
            (Self::True, _) => Self::True,
            (_, Self::True) => Self::True,
            (Self::False, _) => Self::False,
            (_, Self::False) => Self::False,
            (Self::UnknownCallee, Self::UnknownCallee) => Self::UnknownCallee,
        }
    }

    fn is_super_call(callee: Option<AnyNodeRef>) -> Self {
        if callee.is_none() {
            return Self::UnknownCallee;
        }
        let callee = callee.unwrap();
        match callee {
            AnyNodeRef::ExprCall(call) => {
                Self::is_super_call(Some(AnyNodeRef::from(call.func.as_ref())))
            }
            AnyNodeRef::ExprName(name) => Self::from_bool(name.id == "super"),
            AnyNodeRef::ExprAttribute(attribute) => {
                Self::is_super_call(Some(AnyNodeRef::from(attribute.value.as_ref())))
            }
            _ => Self::from_bool(false),
        }
    }

    // Whether the call is non-dynamically dispatched
    fn is_direct_call(callee: Option<AnyNodeRef>, callee_type: Option<&Type>, debug: bool) -> Self {
        Self::is_super_call(callee).or({
            match callee_type {
                Some(Type::BoundMethod(box BoundMethod {
                    obj: Type::ClassDef(_) | Type::Type(_),
                    ..
                })) => Self::from_bool(true),
                Some(Type::BoundMethod(box BoundMethod {
                    obj: Type::ClassType(_) | Type::SelfType(_),
                    ..
                })) => {
                    // Dynamic dispatch if calling a method via an attribute lookup
                    // on an instance
                    Self::from_bool(false)
                }
                Some(Type::BoundMethod(bound_method)) => {
                    debug_println!(
                        debug,
                        "For callee `{:#?}`, unknown object type in bound method `{:#?}`",
                        callee,
                        bound_method
                    );
                    // `true` would skip overrides, which may lead to false negatives. But we prefer false positives since
                    // we are blind to false negatives.
                    Self::from_bool(false)
                }
                Some(Type::Function(_)) => Self::from_bool(true),
                Some(Type::Union(types)) => {
                    Self::is_direct_call(callee, Some(types.first().unwrap()), debug)
                }
                _ => Self::from_bool(false),
            }
        })
    }
}

struct CallGraphVisitor<'a> {
    call_graphs: &'a mut CallGraphs<ExpressionIdentifier, FunctionRef>,
    module_context: &'a ModuleContext<'a>,
    module_id: ModuleId,
    module_name: ModuleName,
    function_base_definitions: &'a WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    override_graph: &'a OverrideGraph,
    current_function: Option<FunctionRef>, // The current function, if it is exported.
    debug: bool,                           // Enable logging for the current function or class body.
    debug_scopes: Vec<bool>,               // The value of the debug flag for each scope.
    error_collector: ErrorCollector,
}

impl<'a> CallGraphVisitor<'a> {
    fn pysa_location(&self, location: TextRange) -> PysaLocation {
        PysaLocation::new(self.module_context.module_info.display_range(location))
    }

    fn add_callees(
        &mut self,
        expression_identifier: ExpressionIdentifier,
        callees: ExpressionCallees<FunctionRef>,
    ) {
        if let Some(current_function) = self.current_function.clone() {
            self.call_graphs
                .add_callees(current_function, expression_identifier, callees);
        }
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
    ) -> Vec1<Target<FunctionRef>> {
        let receiver_type = receiver_type_from_callee_type(callee_type);
        if receiver_type.is_none() {
            return Vec1::new(Target::Function(callee));
        }
        let receiver_type = receiver_type.unwrap();
        let callee_definition = self.get_base_definition(&callee);
        let (receiver_class, _) = self.receiver_class_from_type(
            receiver_type,
            callee_definition.is_some_and(|definition| definition.is_classmethod),
        );
        if receiver_class.is_none() {
            return Vec1::new(Target::Function(callee));
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
            Vec1::new(get_actual_target(callee))
        } else if let Some(overriding_classes) = self.override_graph.get_overriding_classes(&callee)
        {
            // case c
            let callees = overriding_classes
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
            Vec1::from_vec_push(callees, Target::Function(callee))
        } else {
            // case b
            Vec1::new(Target::Function(callee))
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
        callee_expr: Option<AnyNodeRef>,
        callee_type: Option<&Type>,
        precise_receiver_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
        override_implicit_receiver: Option<ImplicitReceiver>,
        override_is_direct_call: Option<bool>,
        unknown_callee_as_direct_call: bool,
    ) -> Vec1<CallTarget<FunctionRef>> {
        let is_direct_call = match override_is_direct_call {
            Some(override_is_direct_call) => DirectCall::from_bool(override_is_direct_call),
            None => DirectCall::is_direct_call(callee_expr, callee_type, self.debug),
        };
        let is_direct_call = match is_direct_call {
            DirectCall::True => true,
            DirectCall::False => false,
            DirectCall::UnknownCallee => unknown_callee_as_direct_call,
        };
        let receiver_type = if precise_receiver_type.is_some() {
            precise_receiver_type
        } else {
            // Since `Type::BoundMethod` does not always has the most precise receiver type, we use it as a fallback
            receiver_type_from_callee_type(callee_type)
        };
        if is_direct_call {
            Vec1::new(self.call_target_from_function_ref(
                function_ref,
                return_type,
                receiver_type,
                callee_expr_suffix,
                /* is_override_target */ false,
                override_implicit_receiver,
            ))
        } else {
            self.compute_targets_for_virtual_call(callee_type, function_ref)
                .mapped(|target| match target {
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
                    Target::Object(_) | Target::FormatString => CallTarget {
                        target,
                        implicit_receiver: ImplicitReceiver::False,
                        receiver_class: None,
                        implicit_dunder_call: false,
                        is_class_method: false,
                        is_static_method: false,
                        return_type,
                    },
                })
        }
    }

    fn call_targets_from_method_name(
        &self,
        method: &Name,
        defining_class: Option<&Type>,
        callee_expr: Option<AnyNodeRef>,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        is_bound_method: bool,
        callee_expr_suffix: Option<&str>,
        override_implicit_receiver: Option<ImplicitReceiver>,
        override_is_direct_call: Option<bool>,
        unknown_callee_as_direct_call: bool,
    ) -> MaybeResolved<Vec1<CallTarget<FunctionRef>>> {
        let call_targets_from_method_name_with_class = |class| {
            self.function_ref_from_class_field(class, method)
                .map(|function_ref| {
                    let receiver_type = if is_bound_method {
                        // For a bound method, its receiver is either `self` or `cls`. For `self`, the receiver
                        // is the defining class. For `cls`, technically the receiver is the type of the class
                        // but we need to be consistent with `receiver_class_from_type`.
                        defining_class
                    } else {
                        None
                    };
                    MaybeResolved::Resolved(self.call_targets_from_static_or_virtual_call(
                        function_ref,
                        callee_expr,
                        callee_type,
                        receiver_type,
                        return_type,
                        callee_expr_suffix,
                        override_implicit_receiver,
                        override_is_direct_call,
                        unknown_callee_as_direct_call,
                    ))
                })
                .unwrap_or(MaybeResolved::Unresolved(
                    UnresolvedReason::UnknownClassField,
                ))
        };

        let call_targets = match defining_class {
            Some(Type::ClassType(class_type)) => {
                call_targets_from_method_name_with_class(class_type.class_object())
            }
            Some(Type::Union(types)) => types
                .iter()
                .map(|type_| {
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
                        unknown_callee_as_direct_call,
                    )
                })
                .reduce(|left, right| left.join(right))
                .unwrap(),
            Some(Type::Function(_)) => {
                // For now, we can't create a pysa target from a Type::Function,
                // because it does not provide enough information to uniquely identify
                // a function.
                MaybeResolved::Unresolved(UnresolvedReason::UnsupportedFunctionTarget)
            }
            Some(Type::LiteralString) => {
                let str_class = self.module_context.stdlib.str().class_object();
                call_targets_from_method_name_with_class(str_class)
            }
            _ => MaybeResolved::Unresolved(UnresolvedReason::UnexpectedDefiningClass),
        };
        if call_targets.is_unresolved() {
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
        callee_expr: Option<AnyNodeRef>,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> MaybeResolved<Vec1<CallTarget<FunctionRef>>> {
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
            /* unknown_callee_as_direct_call */ true,
        )
    }

    fn resolve_constructor_callees(
        &self,
        init_method: Option<Type>,
        new_method: Option<Type>,
        callee_expr: Option<AnyNodeRef>,
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
                Some(true), // Too expensive to merge models for overrides on `object.__init__`
                /* unknown_callee_as_direct_call */ true,
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
                Some(true), // Too expensive to merge models for overrides on `object.__new__`
                /* unknown_callee_as_direct_call */ true,
            )
        };

        let mut init_targets = init_method
            .as_ref()
            .map_or(object_init_method(), |init_method| match init_method {
                Type::BoundMethod(bound_method) => {
                    extract_function_from_bound_method(bound_method)
                        .into_iter()
                        .map(|function| {
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
                                /* unknown_callee_as_direct_call */ true,
                            )
                        })
                        .reduce(|left, right| left.join(right))
                        .unwrap()
                }
                _ => MaybeResolved::Unresolved(UnresolvedReason::UnexpectedInitMethod),
            });
        if init_targets.is_unresolved() {
            // TODO(T243217129): Remove this to treat the callees as obscure when unresolved
            init_targets = object_init_method();
        }

        let mut new_targets = new_method
            .as_ref()
            .map_or(object_new_method(), |new_method| match new_method {
                Type::Function(function) => self.call_targets_from_new_method(
                    function,
                    callee_expr,
                    callee_type,
                    return_type,
                    callee_expr_suffix,
                ),
                Type::Overload(overload) => overload
                    .signatures
                    .iter()
                    .map(|overload_type| {
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
                    .reduce(|left, right| left.join(right))
                    .unwrap(),
                _ => MaybeResolved::Unresolved(UnresolvedReason::UnexpectedNewMethod),
            });
        if new_targets.is_unresolved() {
            // TODO(T243217129): Remove this to treat the callees as obscure when unresolved
            new_targets = object_new_method();
        }

        let (init_targets, init_unresolved) = init_targets.flatten();
        let (new_targets, new_unresolved) = new_targets.flatten();
        CallCallees {
            call_targets: vec![],
            init_targets: init_targets.map(Vec1::into_vec).unwrap_or(vec![]),
            new_targets: new_targets.map(Vec1::into_vec).unwrap_or(vec![]),
            higher_order_parameters: HashMap::new(),
            unresolved: init_unresolved.join(new_unresolved),
        }
    }

    fn resolve_pyrefly_target(
        &self,
        pyrefly_target: Option<crate::alt::call::CallTargetLookup>,
        callee_expr: Option<AnyNodeRef>,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
        unknown_callee_as_direct_call: bool,
    ) -> CallCallees<FunctionRef> {
        match pyrefly_target {
            Some(CallTargetLookup::Ok(box crate::alt::call::CallTarget::BoundMethod(
                type_,
                target,
            ))) => {
                // Calling a method on a class instance.
                self.call_targets_from_method_name(
                    &method_name_from_function(&target.1),
                    Some(&type_),
                    callee_expr,
                    callee_type,
                    return_type,
                    /* is_bound_method */ true,
                    callee_expr_suffix,
                    /* override_implicit_receiver*/ None,
                    /* override_is_direct_call */ None,
                    unknown_callee_as_direct_call,
                )
                .into_call_callees()
            }
            Some(CallTargetLookup::Ok(box crate::alt::call::CallTarget::Function(function))) => {
                // Sometimes this means calling a function (e.g., static method) on a class instance. Sometimes
                // this could be simply calling a module top-level function, which can be handled when the stack
                // of D85441657 enables uniquely identifying a definition from a type.
                self.call_targets_from_method_name(
                    &method_name_from_function(&function.1),
                    callee_type, // For static methods, we find them within the callee type
                    callee_expr,
                    callee_type,
                    return_type,
                    /* is_bound_method */ false,
                    callee_expr_suffix,
                    /* override_implicit_receiver*/ None,
                    /* override_is_direct_call */ None,
                    unknown_callee_as_direct_call,
                )
                .into_call_callees()
            }
            Some(CallTargetLookup::Ok(box crate::alt::call::CallTarget::Class(class_type, _))) => {
                // Constructing a class instance.
                let (init_method, new_method) = self
                    .module_context
                    .transaction
                    .ad_hoc_solve(&self.module_context.handle, |solver| {
                        let new_method = solver.get_dunder_new(&class_type);
                        let overrides_new = new_method.is_some();
                        let init_method = solver.get_dunder_init(
                            &class_type,
                            /* get_object_init */ !overrides_new,
                        );
                        (init_method, new_method)
                    })
                    .unwrap();
                self.resolve_constructor_callees(
                    init_method,
                    new_method,
                    callee_expr,
                    callee_type,
                    return_type,
                    callee_expr_suffix,
                )
            }
            Some(CallTargetLookup::Ok(box crate::alt::call::CallTarget::Union(targets)))
            | Some(CallTargetLookup::Error(targets)) => {
                if targets.is_empty() {
                    debug_println!(
                        self.debug,
                        "Empty pyrefly target CallTargetLookup::Ok([]) or Error([]) for `{:#?}`",
                        callee_expr,
                    );
                    CallCallees::new_unresolved(UnresolvedReason::EmptyPyreflyTarget)
                } else {
                    targets
                        .into_iter()
                        .map(|target| {
                            self.resolve_pyrefly_target(
                                Some(CallTargetLookup::Ok(Box::new(target))),
                                callee_expr,
                                callee_type,
                                return_type,
                                callee_expr_suffix,
                                unknown_callee_as_direct_call,
                            )
                        })
                        .reduce(|mut so_far, call_target| {
                            so_far.join_in_place(call_target);
                            so_far
                        })
                        .unwrap()
                }
            }
            _ => {
                debug_println!(
                    self.debug,
                    "Unrecognized pyrefly target `{:#?}` for `{:#?}`",
                    pyrefly_target,
                    callee_expr,
                );
                CallCallees::new_unresolved(UnresolvedReason::UnexpectedPyreflyTarget)
            }
        }
    }

    fn resolve_repr(
        &self,
        function_ref: FunctionRef,
        call_arguments: Option<&ruff_python_ast::Arguments>,
        callee_expr: Option<AnyNodeRef>,
        callee_type: Option<&Type>,
        return_type: Option<ScalarTypeProperties>,
        callee_expr_suffix: Option<&str>,
    ) -> CallCallees<FunctionRef> {
        if function_ref.module_name == ModuleName::builtins()
            && function_ref.function_name == "repr"
        {
            // Find the actual `__repr__`
            let actual_repr = call_arguments
                .as_ref()
                .and_then(|arguments| arguments.find_positional(0))
                .and_then(|argument| self.module_context.answers.get_type_trace(argument.range()))
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
                        /* unknown_callee_as_direct_call */ true,
                    )
                });
            if let Some(actual_repr) = actual_repr
                && !actual_repr.is_unresolved()
            {
                return actual_repr.into_call_callees();
            }
        }
        CallCallees::new(self.call_targets_from_static_or_virtual_call(
            function_ref.clone(),
            callee_expr,
            callee_type,
            /* precise_receiver_type */ None,
            return_type,
            callee_expr_suffix,
            /* override_implicit_receiver*/ None,
            /* override_is_direct_call */ None,
            /* unknown_callee_as_direct_call */ true,
        ))
    }

    fn resolve_name(
        &self,
        name: &ExprName,
        call_arguments: Option<&ruff_python_ast::Arguments>,
        return_type: Option<ScalarTypeProperties>,
    ) -> IdentifierCallees<FunctionRef> {
        let identifier = Ast::expr_name_identifier(name.clone());
        let go_to_definitions = self
            .module_context
            .transaction
            .find_definition_for_name_use(
                &self.module_context.handle,
                &identifier,
                FindPreference::default(),
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
        let callee_expr = Some(AnyNodeRef::from(name));
        let callee_expr_suffix = Some(name.id.as_str());
        if !go_to_definitions.is_empty() {
            let go_to_definitions = go_to_definitions
                .into_iter()
                .map(|function_ref| {
                    self.resolve_repr(
                        function_ref,
                        call_arguments,
                        callee_expr,
                        callee_type.as_ref(),
                        return_type,
                        callee_expr_suffix,
                    )
                })
                .reduce(|mut left, right| {
                    left.join_in_place(right);
                    left
                })
                .unwrap();
            return IdentifierCallees {
                if_called: go_to_definitions,
            };
        }

        // There is no go-to-definition when for example an `ExprName` is a class definition,
        // a local variable, or a parameter.
        let pyrefly_target = self
            .module_context
            .transaction
            .ad_hoc_solve(&self.module_context.handle, |solver| {
                callee_type
                    .as_ref()
                    .map(|type_| solver.as_call_target(type_.clone()))
            })
            .flatten();
        let callees = self.resolve_pyrefly_target(
            pyrefly_target,
            callee_expr,
            callee_type.as_ref(),
            return_type,
            callee_expr_suffix,
            /* unknown_callee_as_direct_call */ true,
        );
        IdentifierCallees { if_called: callees }
    }

    fn call_targets_from_magic_dunder_attr(
        &self,
        base: Option<&Type>,
        attribute: Option<&Name>,
        range: TextRange,
        callee_expr: Option<AnyNodeRef>,
        unknown_callee_as_direct_call: bool,
        resolve_context: &str,
    ) -> DunderAttrCallees {
        if let Some(base) = base
            && let Some(attribute) = attribute
        {
            struct ResolvedDunderAttr {
                target: CallTargetLookup,
                attr_type: Type,
            }
            self.module_context
                .transaction
                .ad_hoc_solve(&self.module_context.handle, |solver| {
                    solver
                        .type_of_magic_dunder_attr(
                            base,
                            attribute,
                            range,
                            &self.error_collector,
                            None,
                            resolve_context,
                            /* allow_getattr_fallback */ true,
                        )
                        .map(|type_| ResolvedDunderAttr {
                            target: solver.as_call_target(type_.clone()),
                            attr_type: type_,
                        })
                })
                .flatten()
                .map(
                    |ResolvedDunderAttr {
                         target,
                         attr_type: callee_type,
                     }| {
                        let return_type =
                            if let Some(return_type) = callee_type.callable_return_type() {
                                ScalarTypeProperties::from_type(&return_type, self.module_context)
                            } else {
                                ScalarTypeProperties::none()
                            };
                        DunderAttrCallees {
                            callees: self.resolve_pyrefly_target(
                                Some(target),
                                callee_expr,
                                Some(&callee_type),
                                Some(return_type),
                                Some(attribute.as_str()),
                                unknown_callee_as_direct_call,
                            ),
                            attr_type: Some(callee_type),
                        }
                    },
                )
                .unwrap_or(DunderAttrCallees {
                    callees: CallCallees::new_unresolved(
                        UnresolvedReason::UnresolvedMagicDunderAttr,
                    ),
                    attr_type: None,
                })
        } else {
            let reason = if base.is_none() {
                UnresolvedReason::UnresolvedMagicDunderAttrDueToNoBase
            } else if attribute.is_none() {
                UnresolvedReason::UnresolvedMagicDunderAttrDueToNoAttribute
            } else {
                unreachable!();
            };
            DunderAttrCallees {
                callees: CallCallees::new_unresolved(reason),
                attr_type: None,
            }
        }
    }

    // Resolve the attribute access via `__getattr__`
    fn resolve_magic_dunder_attr(
        &self,
        attribute: &Name,
        receiver_type: Option<&Type>,
        callee_expr: Option<AnyNodeRef>, // This is `base.attribute`
        callee_range: TextRange,
    ) -> AttributeAccessCallees<FunctionRef> {
        let DunderAttrCallees { callees, .. } = self.call_targets_from_magic_dunder_attr(
            /* base */ receiver_type,
            /* attribute */ Some(attribute),
            callee_range,
            callee_expr,
            /* unknown_callee_as_direct_call */ true,
            "resolve_magic_dunder_attr",
        );
        AttributeAccessCallees {
            if_called: callees,
            // Property getters and setters are always found via the normal attribute lookup
            property_setters: vec![],
            property_getters: vec![],
        }
    }

    fn resolve_attribute_access(
        &self,
        base: &Expr,
        attribute: &Name,
        callee_expr: Option<AnyNodeRef>, // This is `base.attribute`
        callee_type: Option<&Type>,
        callee_range: TextRange,
        return_type: Option<ScalarTypeProperties>,
        assignment_targets: Option<&[Expr]>,
    ) -> AttributeAccessCallees<FunctionRef> {
        let go_to_definitions = self
            .module_context
            .transaction
            .find_definition_for_attribute(
                &self.module_context.handle,
                base.range(),
                attribute,
                FindPreference::default(),
            );

        let callee_expr_suffix = Some(attribute.as_str());
        let receiver_type = self.module_context.answers.get_type_trace(base.range());

        if go_to_definitions.is_empty() {
            return self.resolve_magic_dunder_attr(
                attribute,
                receiver_type.as_ref(),
                callee_expr,
                callee_range,
            );
        }

        let (property_callees, non_property_callees): (Vec<FunctionRef>, Vec<FunctionRef>) =
            go_to_definitions
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
                        assignment_target_attribute.range() == callee_range
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

        let unknown_callee_as_direct_call = true;
        let if_called = CallCallees {
            call_targets: non_property_callees
                .into_iter()
                .flat_map(|function| {
                    self.call_targets_from_static_or_virtual_call(
                        function,
                        callee_expr,
                        callee_type,
                        receiver_type.as_ref(),
                        return_type,
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                        unknown_callee_as_direct_call,
                    )
                })
                .collect(),
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        };
        AttributeAccessCallees {
            if_called,
            property_setters: property_setters
                .into_iter()
                .flat_map(|function| {
                    self.call_targets_from_static_or_virtual_call(
                        function,
                        callee_expr,
                        callee_type,
                        receiver_type.as_ref(),
                        Some(ScalarTypeProperties::none()),
                        callee_expr_suffix,
                        /* override_implicit_receiver*/ None,
                        /* override_is_direct_call */ None,
                        unknown_callee_as_direct_call,
                    )
                })
                .collect(),
            property_getters: {
                // We cannot get the return types by treating the property getter expressions as callable types.
                // Hence we use the types of the whole expressions.
                let return_type = callee_type
                    .as_ref()
                    .map(|type_| ScalarTypeProperties::from_type(type_, self.module_context));
                property_getters
                    .into_iter()
                    .flat_map(|function| {
                        self.call_targets_from_static_or_virtual_call(
                            function,
                            callee_expr,
                            callee_type,
                            receiver_type.as_ref(),
                            return_type,
                            callee_expr_suffix,
                            /* override_implicit_receiver*/ None,
                            /* override_is_direct_call */ None,
                            unknown_callee_as_direct_call,
                        )
                    })
                    .collect()
            },
        }
    }

    fn resolve_higher_order_parameters(
        &self,
        call_arguments: Option<&ruff_python_ast::Arguments>,
    ) -> HashMap<u32, HigherOrderParameter<FunctionRef>> {
        if call_arguments.is_none() {
            return HashMap::new();
        }
        // TODO: Filter the results with `filter_implicit_dunder_calls`
        call_arguments
            .unwrap()
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
                        let callees = self.resolve_call(
                            /* callee */ argument,
                            /* return_type */
                            Some(
                                self.get_return_type_for_callee(
                                    self.module_context
                                        .answers
                                        .get_type_trace(argument.range())
                                        .as_ref(),
                                ),
                            ),
                            /* arguments */ None,
                            /* assignment_targets */ None,
                        );
                        let call_targets = callees.call_targets;
                        if call_targets.is_empty() {
                            None
                        } else {
                            Some((
                                index,
                                HigherOrderParameter {
                                    index,
                                    call_targets,
                                    unresolved: callees.unresolved,
                                },
                            ))
                        }
                    }
                }
            })
            .collect_no_duplicate_keys()
            .expect("Found duplicate higher order parameters")
    }

    fn resolve_call(
        &self,
        callee: &Expr,
        return_type: Option<ScalarTypeProperties>,
        arguments: Option<&ruff_python_ast::Arguments>,
        assignment_targets: Option<&[Expr]>,
    ) -> CallCallees<FunctionRef> {
        let higher_order_parameters = self.resolve_higher_order_parameters(arguments);

        let mut callees = match callee {
            Expr::Name(name) => {
                let callees = self.resolve_name(name, arguments, return_type);
                debug_println!(
                    self.debug,
                    "Resolved call `{:#?}` with arguments `{:#?}` into `{:#?}`",
                    callee,
                    arguments,
                    callees
                );
                callees.if_called
            }
            Expr::Attribute(attribute) => {
                let callee_expr = Some(AnyNodeRef::from(attribute));
                let callee_type = self
                    .module_context
                    .answers
                    .get_type_trace(attribute.range());
                let callees = self.resolve_attribute_access(
                    &attribute.value,
                    attribute.attr.id(),
                    callee_expr,
                    callee_type.as_ref(),
                    attribute.range(),
                    return_type,
                    assignment_targets,
                );
                debug_println!(
                    self.debug,
                    "Resolved call `{:#?}` into `{:#?}`",
                    callee,
                    callees
                );
                callees.if_called
            }
            _ => CallCallees::new_unresolved(UnresolvedReason::UnexpectedCalleeExpression),
        };
        callees.with_higher_order_parameters(higher_order_parameters);
        callees
    }

    // Use this only when we are not analyzing a call expression (e.g., `foo` in `x = foo`), because
    // for a call expression, we could simply query its type (e.g., query the type of `c(1)`).
    fn get_return_type_for_callee(&self, callee_type: Option<&Type>) -> ScalarTypeProperties {
        callee_type
            .and_then(|ty| ty.callable_return_type())
            .map(|return_type| ScalarTypeProperties::from_type(&return_type, self.module_context))
            .unwrap_or(ScalarTypeProperties::none())
    }

    fn resolve_and_register_call(
        &mut self,
        call: &ExprCall,
        return_type: Option<ScalarTypeProperties>,
        expression_identifier: ExpressionIdentifier,
        assignment_targets: Option<&[Expr]>,
    ) {
        let callee = &call.func;
        let callees = ExpressionCallees::Call(self.resolve_call(
            callee,
            return_type,
            Some(&call.arguments),
            assignment_targets,
        ));
        self.add_callees(expression_identifier, callees);

        match callee.as_ref() {
            Expr::Name(name) if name.id == "getattr" => {
                let base = call.arguments.find_positional(0);
                let attribute = call.arguments.find_positional(1);
                match (base, attribute) {
                    (Some(base), Some(Expr::StringLiteral(attribute))) => {
                        let callees =
                            ExpressionCallees::AttributeAccess(self.resolve_attribute_access(
                                base,
                                &Name::new(attribute.value.to_str()),
                                /* callee_expr */ None,
                                /* callee_type */ None,
                                call.range(),
                                return_type,
                                assignment_targets,
                            ));
                        let expression_identifier =
                            ExpressionIdentifier::ArtificialAttributeAccess(Origin {
                                kind: OriginKind::GetAttrConstantLiteral,
                                location: self.pysa_location(call.range()),
                            });
                        self.add_callees(expression_identifier, callees);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn resolve_and_register_compare(&mut self, compare: &ExprCompare) {
        let Some(left_comparator_type) = &compare
            .comparators
            .first()
            .and_then(|left| self.module_context.answers.get_type_trace(left.range()))
        else {
            return;
        };

        for (operator, right_comparator) in compare.ops.iter().zip(compare.comparators.iter()) {
            let callee_name = dunder::rich_comparison_dunder(*operator);
            let DunderAttrCallees { callees, .. } = self.call_targets_from_magic_dunder_attr(
                /* base */ Some(left_comparator_type),
                /* attribute */ callee_name.as_ref(),
                compare.range(),
                /* callee_expr */ None,
                /* unknown_callee_as_direct_call */ true,
                "resolve_expression_for_exprcompare",
            );
            let expression_identifier = ExpressionIdentifier::ArtificialCall(Origin {
                kind: OriginKind::ComparisonOperator,
                location: self.pysa_location(right_comparator.range()),
            });
            self.add_callees(expression_identifier, ExpressionCallees::Call(callees));
        }
    }

    fn resolve_and_register_comprehension(&mut self, generators: &[Comprehension]) {
        for generator in generators.iter() {
            let iter_range = generator.iter.range();
            let (iter_callee_name, next_callee_name) = if generator.is_async {
                (dunder::AITER, dunder::ANEXT)
            } else {
                (dunder::ITER, dunder::NEXT)
            };
            let DunderAttrCallees {
                callees: iter_callees,
                attr_type: iter_callee_type,
            } = self.call_targets_from_magic_dunder_attr(
                /* base */
                self.module_context
                    .answers
                    .get_type_trace(iter_range)
                    .as_ref(),
                /* attribute */ Some(&iter_callee_name),
                iter_range,
                /* callee_expr */ None,
                /* unknown_callee_as_direct_call */ true,
                "resolve_expression_for_comprehension_iter",
            );
            let iter_identifier = ExpressionIdentifier::ArtificialCall(Origin {
                kind: OriginKind::GeneratorIter,
                location: self.pysa_location(iter_range),
            });
            self.add_callees(iter_identifier, ExpressionCallees::Call(iter_callees));

            let DunderAttrCallees {
                callees: next_callees,
                ..
            } = self.call_targets_from_magic_dunder_attr(
                /* base */
                iter_callee_type
                    .and_then(|iter_callee_type| iter_callee_type.callable_return_type())
                    .as_ref(),
                /* attribute */ Some(&next_callee_name),
                iter_range,
                /* callee_expr */ None,
                /* unknown_callee_as_direct_call */ true,
                "resolve_expression_for_comprehension_iter_next",
            );
            let next_identifier = ExpressionIdentifier::ArtificialCall(Origin {
                kind: OriginKind::GeneratorNext,
                location: self.pysa_location(iter_range),
            });
            self.add_callees(next_identifier, ExpressionCallees::Call(next_callees))
        }
    }

    fn resolve_and_register_subscript(
        &mut self,
        subscript: &ExprSubscript,
        assignment_targets: Option<&[Expr]>,
        current_statement_location: Option<TextRange>,
    ) {
        let subscript_range = subscript.range();
        let is_assignment_target = assignment_targets.is_some_and(|assignment_targets| {
            assignment_targets
                .iter()
                .any(|assignment_target| assignment_target.range() == subscript_range)
        });
        let (callee_name, origin_kind, callee_location) = if is_assignment_target {
            (
                dunder::SETITEM,
                OriginKind::SubscriptSetItem,
                current_statement_location.unwrap(),
            )
        } else {
            (
                dunder::GETITEM,
                OriginKind::SubscriptGetItem,
                subscript_range,
            )
        };
        let value_range = subscript.value.range();
        let DunderAttrCallees { callees, .. } = self.call_targets_from_magic_dunder_attr(
            /* base */
            self.module_context
                .answers
                .get_type_trace(value_range)
                .as_ref(),
            /* attribute */ Some(&callee_name),
            value_range,
            /* callee_expr */ None,
            /* unknown_callee_as_direct_call */ true,
            "resolve_expression_for_subscript",
        );
        let identifier = ExpressionIdentifier::ArtificialCall(Origin {
            kind: origin_kind,
            location: self.pysa_location(callee_location),
        });
        self.add_callees(identifier, ExpressionCallees::Call(callees))
    }

    fn resolve_and_register_fstring(&mut self, fstring: &ExprFString) {
        self.add_callees(
            ExpressionIdentifier::ArtificialCall(Origin {
                kind: OriginKind::FormatStringArtificial,
                location: self.pysa_location(fstring.range()),
            }),
            ExpressionCallees::FormatStringArtificial(FormatStringArtificialCallees {
                targets: vec![CallTarget::format_string_target()],
            }),
        );

        for element in fstring.value.elements() {
            if let Some(interpolation) = element.as_interpolation() {
                {
                    let expression_range = interpolation.expression.range();
                    let callee_name = match interpolation.conversion {
                        ConversionFlag::None => dunder::FORMAT,
                        ConversionFlag::Str => dunder::STR,
                        ConversionFlag::Ascii => dunder::ASCII,
                        ConversionFlag::Repr => dunder::REPR,
                    };
                    let DunderAttrCallees { callees, .. } = self
                        .call_targets_from_magic_dunder_attr(
                            /* base */
                            self.module_context
                                .answers
                                .get_type_trace(expression_range)
                                .as_ref(),
                            /* attribute */ Some(&callee_name),
                            expression_range,
                            /* callee_expr */ None,
                            /* unknown_callee_as_direct_call */ true,
                            "resolve_expression_for_fstring",
                        );
                    self.add_callees(
                        ExpressionIdentifier::ArtificialCall(Origin {
                            kind: OriginKind::FormatStringStringify,
                            location: self.pysa_location(expression_range),
                        }),
                        ExpressionCallees::Call(callees),
                    );
                }
            }
        }
    }

    fn resolve_and_register_expression(
        &mut self,
        expr: &Expr,
        parent_expression: Option<&Expr>,
        current_statement: Option<&Stmt>,
    ) {
        let is_nested_callee_or_base =
            parent_expression.is_some_and(|parent_expression| match parent_expression {
                // For example, avoid visiting `x.__call__` in `x.__call__(1)`
                Expr::Call(callee) if expr.range() == callee.func.range() => true,
                // For example, avoid visiting `x` in `x.__call__` or `x.y` in `x.y.__call__`
                Expr::Attribute(_) => true,
                _ => false,
            });
        let expr_type = || self.module_context.answers.get_type_trace(expr.range());
        let regular_expression_identifier =
            ExpressionIdentifier::regular(expr.range(), &self.module_context.module_info);
        match expr {
            Expr::Call(call) => {
                debug_println!(self.debug, "Resolving callees for call `{:#?}`", expr);
                let return_type_from_expr = expr_type()
                    .as_ref()
                    .map(|type_| ScalarTypeProperties::from_type(type_, self.module_context));
                self.resolve_and_register_call(
                    call,
                    return_type_from_expr,
                    regular_expression_identifier,
                    assignment_targets(current_statement),
                );
            }
            Expr::Name(name) if !is_nested_callee_or_base => {
                debug_println!(self.debug, "Resolving callees for name `{:#?}`", expr);
                let callees = self.resolve_name(
                    name,
                    /* call_arguments */ None,
                    Some(self.get_return_type_for_callee(expr_type().as_ref())), // This is the return type when `expr` is called
                );
                if !callees.is_empty() {
                    self.add_callees(
                        regular_expression_identifier,
                        ExpressionCallees::Identifier(callees),
                    );
                }
            }
            Expr::Attribute(attribute) if !is_nested_callee_or_base => {
                debug_println!(self.debug, "Resolving callees for attribute `{:#?}`", expr);
                let callee_expr = Some(AnyNodeRef::from(attribute));
                let callees = self.resolve_attribute_access(
                    &attribute.value,
                    attribute.attr.id(),
                    callee_expr,
                    /* callee_type */ expr_type().as_ref(),
                    attribute.range(),
                    /* return_type */
                    Some(self.get_return_type_for_callee(expr_type().as_ref())), // This is the return type when `expr` is called
                    assignment_targets(current_statement),
                );
                if !callees.is_empty() {
                    self.add_callees(
                        regular_expression_identifier,
                        ExpressionCallees::AttributeAccess(callees),
                    );
                }
            }
            Expr::Compare(compare) => {
                debug_println!(self.debug, "Resolving callees for compare `{:#?}`", expr);
                self.resolve_and_register_compare(compare);
            }
            Expr::ListComp(comp) => {
                debug_println!(self.debug, "Resolving callees for list comp `{:#?}`", expr);
                self.resolve_and_register_comprehension(&comp.generators);
            }
            Expr::SetComp(comp) => {
                debug_println!(self.debug, "Resolving callees for set comp `{:#?}`", expr);
                self.resolve_and_register_comprehension(&comp.generators);
            }
            Expr::Generator(generator) => {
                debug_println!(self.debug, "Resolving callees for generator `{:#?}`", expr);
                self.resolve_and_register_comprehension(&generator.generators);
            }
            Expr::DictComp(comp) => {
                debug_println!(self.debug, "Resolving callees for dict comp `{:#?}`", expr);
                self.resolve_and_register_comprehension(&comp.generators);
            }
            Expr::Subscript(subscript) => {
                debug_println!(self.debug, "Resolving callees for subscript `{:#?}`", expr);
                self.resolve_and_register_subscript(
                    subscript,
                    assignment_targets(current_statement),
                    current_statement.map(|stmt| stmt.range()),
                );
            }
            Expr::FString(fstring) => {
                debug_println!(self.debug, "Resolving callees for fstring `{:#?}`", expr);
                self.resolve_and_register_fstring(fstring);
            }
            _ => {
                debug_println!(self.debug, "Nothing to resolve in expression `{:#?}`", expr);
            }
        };
    }

    fn resolve_and_register_function_def(&mut self, function_def: &StmtFunctionDef) {
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
            return;
        }
        let key = KeyDecoratedFunction(ShortIdentifier::new(&function_def.name));
        let callees = self
            .module_context
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
            });
        if let Some(callees) = callees {
            self.add_callees(
                ExpressionIdentifier::regular(
                    function_def.range(),
                    &self.module_context.module_info,
                ),
                callees,
            );
        }
    }

    fn resolve_and_register_with_statement(&mut self, stmt_with: &StmtWith) {
        for item in stmt_with.items.iter() {
            let context_expr_range = item.context_expr.range();
            let callee_name = if stmt_with.is_async {
                dunder::AENTER
            } else {
                dunder::ENTER
            };
            let DunderAttrCallees { callees, .. } = self.call_targets_from_magic_dunder_attr(
                /* base */
                self.module_context
                    .answers
                    .get_type_trace(context_expr_range)
                    .as_ref(),
                /* attribute */ Some(&callee_name),
                context_expr_range,
                /* callee_expr */ None,
                /* unknown_callee_as_direct_call */ true,
                "resolve_and_register_with_statement",
            );
            let expression_identifier = ExpressionIdentifier::ArtificialCall(Origin {
                kind: OriginKind::WithEnter,
                location: self.pysa_location(context_expr_range),
            });
            self.add_callees(expression_identifier, ExpressionCallees::Call(callees));
        }
    }

    fn resolve_and_register_decorator_callees(
        &mut self,
        decorators: &[Decorator],
        decorated_target: FunctionRef,
    ) {
        for decorator in decorators.iter() {
            debug_println!(
                self.debug,
                "Resolving callees for decorator call `{:#?}`",
                decorator
            );
            let callee_type = self
                .module_context
                .answers
                .get_type_trace(decorator.expression.range());
            let return_type = self.get_return_type_for_callee(callee_type.as_ref());
            let callees = self.resolve_call(
                /* callee */ &decorator.expression,
                /* return_type */ Some(return_type),
                /* arguments */ None,
                /* assignment_targets */ None,
            );
            self.call_graphs.add_callees(
                decorated_target.clone(),
                ExpressionIdentifier::ArtificialCall(Origin {
                    kind: OriginKind::ForDecoratedTarget,
                    location: self.pysa_location(decorator.expression.range()),
                }),
                ExpressionCallees::Call(callees),
            );
            // Remove callees for the underlying expression, to avoid duplicates.
            if matches!(decorator.expression, Expr::Name(_))
                || matches!(decorator.expression, Expr::Attribute(_))
            {
                self.call_graphs.remove_callees(
                    decorated_target.clone(),
                    ExpressionIdentifier::Regular(self.pysa_location(decorator.expression.range())),
                );
            }
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

    fn exit_function_scope(&mut self, function_def: &StmtFunctionDef, scopes: &Scopes) {
        // Register artificial callees for decorated targets.
        if !function_def.decorator_list.is_empty() {
            let current_function = scopes
                .current_exported_function(
                    self.module_id,
                    self.module_name,
                    ScopeExportedFunctionFlags {
                        include_top_level: false,
                        include_class_top_level: false,
                        include_function_decorators:
                            super::ast_visitor::ExportFunctionDecorators::Ignore,
                        include_class_decorators: super::ast_visitor::ExportClassDecorators::Ignore,
                        include_default_arguments:
                            super::ast_visitor::ExportDefaultArguments::Ignore,
                    },
                )
                .and_then(|function_ref| function_ref.get_decorated_target());
            if let Some(decorated_target) = current_function {
                self.resolve_and_register_decorator_callees(
                    &function_def.decorator_list,
                    decorated_target,
                );
            }
        }

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
        current_statement: Option<&Stmt>,
    ) {
        if self.current_function.is_none() {
            return;
        }
        self.resolve_and_register_expression(expr, parent_expression, current_statement);
    }

    fn visit_statement(&mut self, stmt: &Stmt, _scopes: &Scopes) {
        match stmt {
            Stmt::FunctionDef(function_def) => self.resolve_and_register_function_def(function_def),
            Stmt::With(stmt_with) => self.resolve_and_register_with_statement(stmt_with),
            _ => {}
        }
    }
}

fn resolve_call(
    call: &ExprCall,
    function_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    module_context: &ModuleContext,
    override_graph: &OverrideGraph,
) -> Vec<CallTarget<FunctionRef>> {
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
        error_collector: ErrorCollector::new(module_context.module_info.dupe(), ErrorStyle::Never),
    };
    let callees = visitor.resolve_call(
        /* callee */ &call.func,
        /* return_type */
        module_context
            .answers
            .get_type_trace(call.range())
            .map(|type_| ScalarTypeProperties::from_type(&type_, module_context)),
        /* arguments */ Some(&call.arguments),
        /* assignment_targets */ None,
    );
    callees.all_targets().cloned().collect()
}

fn resolve_expression(
    expression: &Expr,
    function_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    module_context: &ModuleContext,
    override_graph: &OverrideGraph,
    parent_expression: Option<&Expr>,
) -> Vec<CallTarget<FunctionRef>> {
    // This needs to be provided. Otherwise the callees won't be registered into `call_graphs`.
    let current_function = FunctionRef {
        module_id: module_context.module_id,
        module_name: module_context.module_info.name(),
        function_id: FunctionId::ModuleTopLevel,
        function_name: Name::new("artificial_function"),
    };
    let mut call_graphs = CallGraphs::new();
    let mut visitor = CallGraphVisitor {
        call_graphs: &mut call_graphs,
        module_context,
        module_id: module_context.module_id,
        module_name: module_context.module_info.name(),
        function_base_definitions: function_definitions,
        current_function: Some(current_function.clone()),
        debug: false,
        debug_scopes: Vec::new(),
        override_graph,
        error_collector: ErrorCollector::new(module_context.module_info.dupe(), ErrorStyle::Never),
    };
    visitor.resolve_and_register_expression(
        expression,
        parent_expression,
        /* current_statement */ None,
    );
    let expression_identifier =
        ExpressionIdentifier::regular(expression.range(), &module_context.module_info);
    call_graphs
        .0
        .entry(current_function)
        .or_default()
        .0
        .get(&expression_identifier)
        .map(|callees| callees.all_targets().cloned().collect::<Vec<_>>())
        .unwrap_or_default()
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
        Target::Object(_) | Target::FormatString => false,
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
                        .into_iter()
                        .map(|call_target| call_target.target)
                        .filter(|target| !is_object_new_or_init_target(target))
                        .collect::<Vec<_>>(),
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
                    callees
                        .into_iter()
                        .map(|call_target| call_target.target)
                        .filter(|target| !is_object_new_or_init_target(target))
                        .collect::<Vec<_>>(),
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
) -> CallGraphs<ExpressionIdentifier, FunctionRef> {
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
        error_collector: ErrorCollector::new(context.module_info.dupe(), ErrorStyle::Never),
    };

    visit_module_ast(&mut visitor, context);
    call_graphs.dedup_and_sort();
    call_graphs
}
