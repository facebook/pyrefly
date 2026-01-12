/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Not;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::class::Class;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Overload;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use serde::Serialize;
use starlark_map::Hashed;

use crate::alt::class::class_field::ClassField;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecoratedFunction;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::resolve_decorator_callees;
use crate::report::pysa::captured_variable::CapturedVariable;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::get_all_classes;
use crate::report::pysa::class::get_class_field_declaration;
use crate::report::pysa::class::get_class_fields;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::override_graph::WholeProgramReversedOverrideGraph;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::scope::get_scope_parent;
use crate::report::pysa::slow_fun_monitor::slow_fun_monitor_scope;
use crate::report::pysa::step_logger::StepLogger;
use crate::report::pysa::types::PysaType;
use crate::report::pysa::types::is_callable_like;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

/// Represents a unique identifier for a function **within a module**.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FunctionId {
    /// Function declared with a `def` statement.
    Function { location: PysaLocation },
    /// Implicit function containing all top-level statements.
    ModuleTopLevel,
    /// Implicit function containing the class body.
    ClassTopLevel { class_id: ClassId },
    /// Function-like class field that is not a `def` statement.
    ClassField { class_id: ClassId, name: Name },
    /// Decorated target, which represents an artificial function containing all
    /// decorators of a function, inlined as an expression.
    /// For e.g, `@foo` on `def bar()` -> `return foo(bar)`
    FunctionDecoratedTarget { location: PysaLocation },
}

impl FunctionId {
    fn serialize_to_string(&self) -> String {
        match self {
            FunctionId::Function { location } => format!("F:{}", location.as_key()),
            FunctionId::ModuleTopLevel => "MTL".to_owned(),
            FunctionId::ClassTopLevel { class_id } => format!("CTL:{}", class_id.to_int()),
            FunctionId::ClassField { class_id, name } => {
                format!("CF:{}:{}", class_id.to_int(), name)
            }
            FunctionId::FunctionDecoratedTarget { location } => {
                format!("FDT:{}", location.as_key())
            }
        }
    }
}

impl Serialize for FunctionId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.serialize_to_string())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionRef {
    pub module_id: ModuleId,
    pub module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    pub function_id: FunctionId,
    pub function_name: Name, // For debugging purposes only. Reader should use the function id.
}

impl FunctionRef {
    pub fn from_decorated_function(function: &DecoratedFunction, context: &ModuleContext) -> Self {
        assert_decorated_function_in_context(function, context);
        assert!(should_export_decorated_function(function, context));
        let name = function.metadata().kind.function_name().into_owned();
        let display_range = context.module_info.display_range(function.id_range());
        FunctionRef {
            module_id: context.module_id,
            module_name: context.module_info.name(),
            function_id: FunctionId::Function {
                location: PysaLocation::new(display_range),
            },
            function_name: name,
        }
    }

    pub fn get_decorated_target(self) -> Option<Self> {
        match self {
            FunctionRef {
                module_id,
                module_name,
                function_id: FunctionId::Function { location },
                function_name,
            } => Some(FunctionRef {
                module_id,
                module_name,
                function_id: FunctionId::FunctionDecoratedTarget { location },
                function_name,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum FunctionParameter {
    PosOnly {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<Name>,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    Pos {
        name: Name,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    VarArg {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<Name>,
        annotation: PysaType,
    },
    KwOnly {
        name: Name,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    Kwargs {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<Name>,
        annotation: PysaType,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum FunctionParameters {
    List(Vec<FunctionParameter>),
    Ellipsis,
    ParamSpec,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionSignature {
    pub parameters: FunctionParameters,
    pub return_annotation: PysaType,
}

/// Represents information about a function definition, collected in a pre-analysis step.
/// See `FunctionDefinition` for the type exported to Pysa. This only store memory-efficient information.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionBaseDefinition {
    pub name: Name,
    pub parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_overload: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_staticmethod: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_classmethod: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_property_getter: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_property_setter: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_stub: bool,
    /// Is this defined with a `def` statement.
    /// False for synthesized methods (e.g, dataclass `__init__`) or `foo: Callable[..]` attributes.
    #[serde(skip_serializing_if = "Clone::clone")]
    pub is_def_statement: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// If this is a method, record the class it is defined in.
    pub defining_class: Option<ClassRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// If the method directly overrides a method in a parent class, we record that class.
    /// This is used for building overriding graphs.
    pub overridden_base_method: Option<FunctionRef>,
}

impl FunctionBaseDefinition {
    pub fn is_method(&self) -> bool {
        self.defining_class.is_some()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionDefinition {
    #[serde(flatten)]
    pub base: FunctionBaseDefinition,
    pub undecorated_signatures: Vec<FunctionSignature>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub captured_variables: Vec<CapturedVariable>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub decorator_callees: HashMap<PysaLocation, Vec<Target<FunctionRef>>>,
}

impl FunctionDefinition {
    #[cfg(test)]
    pub fn with_is_staticmethod(mut self, is_staticmethod: bool) -> Self {
        self.base.is_staticmethod = is_staticmethod;
        self
    }

    #[cfg(test)]
    pub fn with_is_classmethod(mut self, is_classmethod: bool) -> Self {
        self.base.is_classmethod = is_classmethod;
        self
    }

    #[cfg(test)]
    pub fn with_is_stub(mut self, is_stub: bool) -> Self {
        self.base.is_stub = is_stub;
        self
    }

    #[cfg(test)]
    pub fn with_is_property_getter(mut self, is_property_getter: bool) -> Self {
        self.base.is_property_getter = is_property_getter;
        self
    }

    #[cfg(test)]
    pub fn with_is_property_setter(mut self, is_property_setter: bool) -> Self {
        self.base.is_property_setter = is_property_setter;
        self
    }

    #[cfg(test)]
    pub fn with_defining_class(mut self, defining_class: ClassRef) -> Self {
        self.base.defining_class = Some(defining_class);
        self
    }

    #[cfg(test)]
    pub fn with_is_def_statement(mut self, is_def_statement: bool) -> Self {
        self.base.is_def_statement = is_def_statement;
        self
    }

    #[cfg(test)]
    pub fn with_decorator_callees(
        mut self,
        decorator_callees: HashMap<PysaLocation, Vec<Target<FunctionRef>>>,
    ) -> Self {
        self.decorator_callees = decorator_callees;
        self
    }

    #[cfg(test)]
    pub fn with_overridden_base_method(mut self, overridden_base_method: FunctionRef) -> Self {
        self.base.overridden_base_method = Some(overridden_base_method);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleFunctionDefinitions<GenericFunctionDefinition>(
    HashMap<FunctionId, GenericFunctionDefinition>,
);

impl<GenericFunctionDefinition> ModuleFunctionDefinitions<GenericFunctionDefinition> {
    pub fn new() -> Self {
        ModuleFunctionDefinitions(HashMap::new())
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = (&FunctionId, &GenericFunctionDefinition)> {
        self.0.iter()
    }
}

pub struct WholeProgramFunctionDefinitions<FunctionDefinition>(
    dashmap::ReadOnlyView<ModuleId, ModuleFunctionDefinitions<FunctionDefinition>>,
);

impl<GenericFunctionDefinition> WholeProgramFunctionDefinitions<GenericFunctionDefinition> {
    pub fn get<'a>(
        &'a self,
        module_id: ModuleId,
        function_id: &FunctionId,
    ) -> Option<&'a GenericFunctionDefinition> {
        self.0
            .get(&module_id)
            .and_then(|functions| functions.0.get(function_id))
    }

    pub fn get_for_module(
        &self,
        module_id: ModuleId,
    ) -> Option<&ModuleFunctionDefinitions<GenericFunctionDefinition>> {
        self.0.get(&module_id)
    }
}

fn export_function_parameter(param: &Param, context: &ModuleContext) -> FunctionParameter {
    match param {
        Param::PosOnly(name, ty, required) => FunctionParameter::PosOnly {
            name: name.clone(),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::Pos(name, ty, required) => FunctionParameter::Pos {
            name: name.clone(),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::VarArg(name, ty) => FunctionParameter::VarArg {
            name: name.clone(),
            annotation: PysaType::from_type(ty, context),
        },
        Param::KwOnly(name, ty, required) => FunctionParameter::KwOnly {
            name: name.clone(),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::Kwargs(name, ty) => FunctionParameter::Kwargs {
            name: name.clone(),
            annotation: PysaType::from_type(ty, context),
        },
    }
}

fn export_function_parameters(params: &Params, context: &ModuleContext) -> FunctionParameters {
    match params {
        Params::List(params) => FunctionParameters::List(
            params
                .items()
                .iter()
                .map(|param| export_function_parameter(param, context))
                .collect::<Vec<_>>(),
        ),
        Params::Ellipsis | Params::Materialization => FunctionParameters::Ellipsis,
        Params::ParamSpec(_, _) => FunctionParameters::ParamSpec,
    }
}

// For many function implementations, we need to pass the module context where the function is defined.
fn assert_decorated_function_in_context(function: &DecoratedFunction, context: &ModuleContext) {
    match &function.undecorated.metadata.kind {
        FunctionKind::Def(func_id) => {
            assert_eq!(func_id.module, context.module_info);
        }
        _ => (),
    }
}

pub fn should_export_decorated_function(
    function: &DecoratedFunction,
    context: &ModuleContext,
) -> bool {
    assert_decorated_function_in_context(function, context);
    // We only want to export one function when we have an @overload chain.
    // If the function has no successor (function in the same scope with the same name), then we should export it.
    // If the function has successors, but is not an overload, then we should export it. It probably means the successor is a redefinition.
    let has_successor = context.bindings.get(function.idx).successor.is_some();
    !has_successor || !function.is_overload()
}

// From any decorated function, returns the function that should be exported.
// Requires `context` to be the module context of the decorated function.
pub fn get_exported_decorated_function(
    key_decorated_function: Idx<KeyDecoratedFunction>,
    context: &ModuleContext,
) -> DecoratedFunction {
    // Follow the successor chain to find either the last function or a function that is not an overload.
    let mut last_decorated_function = key_decorated_function;
    loop {
        let binding_decorated_function = context.bindings.get(last_decorated_function);

        let undecorated_function = context
            .answers
            .get_idx(binding_decorated_function.undecorated_idx)
            .unwrap();
        if !undecorated_function.metadata.flags.is_overload {
            break;
        }

        let successor = binding_decorated_function.successor;
        if let Some(successor) = successor {
            last_decorated_function = successor;
        } else {
            break;
        }
    }

    DecoratedFunction::from_bindings_answers(
        last_decorated_function,
        &context.bindings,
        &context.answers,
    )
}

fn export_function_signature(function: &Callable, context: &ModuleContext) -> FunctionSignature {
    FunctionSignature {
        parameters: export_function_parameters(&function.params, context),
        return_annotation: PysaType::from_type(&function.ret, context),
    }
}

fn export_overload_signatures(
    overload: &Overload,
    context: &ModuleContext,
) -> Vec<FunctionSignature> {
    overload
        .signatures
        .iter()
        .map(|overload_type| match overload_type {
            pyrefly_types::types::OverloadType::Function(f) => f,
            pyrefly_types::types::OverloadType::Forall(pyrefly_types::types::Forall {
                body,
                ..
            }) => body,
        })
        .map(|function| export_function_signature(&function.signature, context))
        .collect::<Vec<_>>()
}

fn export_signatures_from_type(ty: &Type, context: &ModuleContext) -> Vec<FunctionSignature> {
    match ty {
        Type::Function(f) => vec![export_function_signature(&f.signature, context)],
        Type::Callable(callable) => {
            vec![export_function_signature(callable, context)]
        }
        Type::BoundMethod(bound_method) => match &bound_method.func {
            BoundMethodType::Function(f) => {
                vec![export_function_signature(&f.signature, context)]
            }
            BoundMethodType::Forall(forall) => {
                vec![export_function_signature(&forall.body.signature, context)]
            }
            BoundMethodType::Overload(overload) => export_overload_signatures(overload, context),
        },
        Type::Overload(overload) => export_overload_signatures(overload, context),
        Type::Union(box Union { members: union, .. }) => union
            .iter()
            .flat_map(|ty| export_signatures_from_type(ty, context))
            .collect::<Vec<_>>(),
        _ => vec![],
    }
}

/// Represents a function definition in pyrefly.
#[derive(Debug, Clone)]
pub enum FunctionNode {
    // Function defined with a `def` statement.
    DecoratedFunction(DecoratedFunction),
    // Function-like class field that is not a `def` statement.
    // For instance, `foo: Callable[.., int]` or a synthesized `__init__` method.
    ClassField {
        class: Class,
        name: Name,
        field: Arc<ClassField>,
    },
}

impl FunctionNode {
    // For many function implementations, we need to pass the module context where the function is defined.
    fn assert_in_context(&self, context: &ModuleContext) {
        match self {
            FunctionNode::DecoratedFunction(function) => {
                assert_decorated_function_in_context(function, context)
            }
            FunctionNode::ClassField { class, .. } => {
                assert_eq!(class.module(), &context.module_info);
            }
        }
    }

    pub fn should_export(&self, context: &ModuleContext) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => {
                should_export_decorated_function(function, context)
            }
            FunctionNode::ClassField { class, name, field } => {
                is_callable_like(&field.ty())
                    && match get_class_field_declaration(class, name, context) {
                        // Class field is a `def` statement.
                        Some(BindingClassField {
                            definition: ClassFieldDefinition::MethodLike { .. },
                            ..
                        }) => false,
                        _ => true,
                    }
            }
        }
    }

    pub fn name(&self) -> Cow<'_, Name> {
        match self {
            FunctionNode::DecoratedFunction(function) => function.metadata().kind.function_name(),
            FunctionNode::ClassField { name, .. } => Cow::Borrowed(name),
        }
    }

    // Return the function type, considering decorators and overloads.
    fn get_decorated_type(&self, context: &ModuleContext) -> Type {
        self.assert_in_context(context);
        match self {
            FunctionNode::DecoratedFunction(function) => {
                let definition_binding = Key::Definition(function.undecorated.identifier);
                let idx = context.bindings.key_to_idx(&definition_binding);
                context.answers.get_idx(idx).unwrap().arc_clone_ty()
            }
            FunctionNode::ClassField { field, .. } => field.ty(),
        }
    }

    fn get_undecorated_signatures(&self, context: &ModuleContext) -> Vec<FunctionSignature> {
        match self {
            FunctionNode::DecoratedFunction(function) => {
                // We need the list of raw parameters, ignoring decorators.
                // For overloads, we need the list of all overloads, not just the current one.
                // To get it, we check if `get_function_type` returns `Type::Overload`.
                let decorated_type = self.get_decorated_type(context);
                match decorated_type {
                    Type::Overload(overload) => export_overload_signatures(&overload, context),
                    _ => {
                        let return_binding = Key::ReturnType(function.undecorated.identifier);
                        let idx = context.bindings.key_to_idx(&return_binding);
                        let undecorated_return_type =
                            context.answers.get_idx(idx).unwrap().arc_clone_ty();
                        vec![FunctionSignature {
                            parameters: FunctionParameters::List(
                                function
                                    .undecorated
                                    .params
                                    .iter()
                                    .map(|param| export_function_parameter(param, context))
                                    .collect::<Vec<_>>(),
                            ),
                            return_annotation: PysaType::from_type(
                                &undecorated_return_type,
                                context,
                            ),
                        }]
                    }
                }
            }
            FunctionNode::ClassField { field, .. } => {
                export_signatures_from_type(&field.ty(), context)
            }
        }
    }

    pub fn exported_function_from_class_field(
        class: &Class,
        field_name: &Name,
        class_field: Arc<ClassField>,
        context: &ModuleContext,
    ) -> Option<Self> {
        let function_node = match get_class_field_declaration(class, field_name, context) {
            // Class field is a `def` statement.
            Some(BindingClassField {
                definition: ClassFieldDefinition::MethodLike { definition, .. },
                ..
            }) => {
                let binding = context.bindings.get(*definition);
                if let Binding::Function(key_decorated_function, _, _) = binding {
                    let exported_function =
                        get_exported_decorated_function(*key_decorated_function, context);
                    Some(FunctionNode::DecoratedFunction(exported_function))
                } else {
                    // TODO(T225700656): Handle special case
                    None
                }
            }
            _ => Some(FunctionNode::ClassField {
                class: class.dupe(),
                name: field_name.clone(),
                field: class_field.dupe(),
            }),
        }?;

        if function_node.should_export(context) {
            Some(function_node)
        } else {
            None
        }
    }

    pub fn exported_function_from_definition_item_with_docstring<'a>(
        item: &FindDefinitionItemWithDocstring,
        context: &ModuleContext<'a>,
    ) -> Option<(Self, ModuleContext<'a>)> {
        let handle = Handle::new(
            item.module.name(),
            item.module.path().dupe(),
            context.handle.sys_info().dupe(),
        );
        let context =
            ModuleContext::create(handle, context.transaction, context.module_ids).unwrap();
        let key_decorated_function =
            KeyDecoratedFunction(ShortIdentifier::from_text_range(item.definition_range));
        context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&key_decorated_function))
            .map(|idx| get_exported_decorated_function(idx, &context))
            .map(|exported_function| (FunctionNode::DecoratedFunction(exported_function), context))
    }

    pub fn as_function_ref(&self, context: &ModuleContext) -> FunctionRef {
        self.assert_in_context(context);
        assert!(self.should_export(context));

        match self {
            FunctionNode::DecoratedFunction(function) => {
                FunctionRef::from_decorated_function(function, context)
            }
            FunctionNode::ClassField { class, name, .. } => FunctionRef {
                module_id: context.module_id,
                module_name: class.module().name(),
                function_id: FunctionId::ClassField {
                    class_id: ClassId::from_class(class),
                    name: name.clone(),
                },
                function_name: name.clone(),
            },
        }
    }

    fn get_scope_parent(&self, context: &ModuleContext) -> ScopeParent {
        self.assert_in_context(context);
        match self {
            FunctionNode::DecoratedFunction(function) => match &function.undecorated.defining_cls {
                Some(cls) => ScopeParent::Class {
                    location: PysaLocation::new(
                        context.module_info.display_range(cls.qname().range()),
                    ),
                },
                None => get_scope_parent(&context.ast, &context.module_info, function.id_range()),
            },
            FunctionNode::ClassField { class, .. } => ScopeParent::Class {
                location: PysaLocation::new(
                    context.module_info.display_range(class.qname().range()),
                ),
            },
        }
    }

    fn is_def_statement(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(_) => true,
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_overload(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function.metadata().flags.is_overload,
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_staticmethod(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function.metadata().flags.is_staticmethod,
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_classmethod(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function.metadata().flags.is_classmethod,
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_property_getter(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function
                .metadata()
                .flags
                .property_metadata
                .as_ref()
                .is_some_and(|meta| matches!(meta.role, PropertyRole::Getter)),
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_property_setter(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function
                .metadata()
                .flags
                .property_metadata
                .as_ref()
                .is_some_and(|meta| matches!(meta.role, PropertyRole::Setter)),
            FunctionNode::ClassField { .. } => false,
        }
    }

    fn is_stub(&self) -> bool {
        match self {
            FunctionNode::DecoratedFunction(function) => function.is_stub(),
            FunctionNode::ClassField { .. } => false,
        }
    }

    pub fn defining_cls(&self) -> Option<&Class> {
        match self {
            FunctionNode::DecoratedFunction(function) => function.defining_cls(),
            FunctionNode::ClassField { class, .. } => Some(class),
        }
    }

    fn get_define_stmt<'a>(&self, context: &'a ModuleContext) -> Option<&'a StmtFunctionDef> {
        self.assert_in_context(context);
        match self {
            FunctionNode::DecoratedFunction(function) => {
                let range = function.id_range();
                Ast::locate_node(&context.ast, range.start())
                    .iter()
                    .find_map(|node| match node {
                        AnyNodeRef::StmtFunctionDef(stmt) if stmt.name.range == range => {
                            Some(*stmt)
                        }
                        _ => None,
                    })
            }
            FunctionNode::ClassField { .. } => None,
        }
    }

    fn get_decorator_callees(
        &self,
        function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
        context: &ModuleContext,
    ) -> HashMap<PysaLocation, Vec<Target<FunctionRef>>> {
        if let Some(function_def) = self.get_define_stmt(context) {
            resolve_decorator_callees(
                &function_def.decorator_list,
                function_base_definitions,
                context,
            )
        } else {
            HashMap::new()
        }
    }
}

/// Return all functions defined with a `def` statement.
pub fn get_all_decorated_functions(
    context: &ModuleContext,
) -> impl Iterator<Item = DecoratedFunction> {
    context.bindings.keys::<KeyDecoratedFunction>().map(|idx| {
        DecoratedFunction::from_bindings_answers(idx, &context.bindings, &context.answers)
    })
}

/// Return all function defined in the module.
pub fn get_all_functions(context: &ModuleContext) -> impl Iterator<Item = FunctionNode> {
    let decorated_functions = context.bindings.keys::<KeyDecoratedFunction>().map(|idx| {
        FunctionNode::DecoratedFunction(DecoratedFunction::from_bindings_answers(
            idx,
            &context.bindings,
            &context.answers,
        ))
    });
    let class_fields = get_all_classes(context)
        .flat_map(|class| {
            get_class_fields(&class, context)
                .filter(|(_, field)| is_callable_like(&field.ty()))
                .filter(|(name, _)| {
                    match get_class_field_declaration(&class, name, context) {
                        // Class field is a `def` statement.
                        Some(BindingClassField {
                            definition: ClassFieldDefinition::MethodLike { .. },
                            ..
                        }) => false,
                        _ => true,
                    }
                })
                .map(|(name, field)| (class.clone(), Cow::Owned(name.into_owned()), field))
                // Required by borrow checker.
                // This is fine because the amount of class fields should be small.
                .collect::<Vec<_>>()
        })
        .map(|(class, name, field)| FunctionNode::ClassField {
            class,
            name: Name::clone(&name),
            field,
        });
    decorated_functions.chain(class_fields)
}

pub fn export_all_functions(
    reversed_override_graph: &WholeProgramReversedOverrideGraph,
    context: &ModuleContext,
) -> ModuleFunctionDefinitions<FunctionBaseDefinition> {
    let mut function_base_definitions = ModuleFunctionDefinitions::new();

    for function in get_all_functions(context) {
        if !function.should_export(context) {
            continue;
        }

        let current_function = function.as_function_ref(context);
        let parent = function.get_scope_parent(context);
        assert!(
            function_base_definitions
                .0
                .insert(
                    current_function.function_id.clone(),
                    FunctionBaseDefinition {
                        name: current_function.function_name.clone(),
                        parent,
                        is_overload: function.is_overload(),
                        is_staticmethod: function.is_staticmethod(),
                        is_classmethod: function.is_classmethod(),
                        is_property_getter: function.is_property_getter(),
                        is_property_setter: function.is_property_setter(),
                        is_stub: function.is_stub(),
                        is_def_statement: function.is_def_statement(),
                        defining_class: function
                            .defining_cls()
                            .map(|class| ClassRef::from_class(class, context.module_ids)),
                        overridden_base_method: reversed_override_graph
                            .get(&current_function)
                            .cloned(),
                    }
                )
                .is_none(),
            "Found function definitions with the same location"
        );
    }

    function_base_definitions
}

pub fn export_function_definitions(
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    captured_variables: &HashMap<FunctionRef, Vec<CapturedVariable>>,
    context: &ModuleContext,
) -> ModuleFunctionDefinitions<FunctionDefinition> {
    let mut function_definitions = ModuleFunctionDefinitions::new();
    let function_base_definitions_for_module = function_base_definitions
        .get_for_module(context.module_id)
        .unwrap();

    for function in get_all_functions(context) {
        if !function.should_export(context) {
            continue;
        }
        let current_function = function.as_function_ref(context);
        let function_base_definition = function_base_definitions_for_module
            .0
            .get(&current_function.function_id)
            .unwrap();
        let undecorated_signatures = function.get_undecorated_signatures(context);

        let captured_variables = captured_variables
            .get(&current_function)
            .cloned()
            .unwrap_or_default();

        let decorator_callees = function.get_decorator_callees(function_base_definitions, context);

        assert!(
            function_definitions
                .0
                .insert(
                    current_function.function_id.clone(),
                    FunctionDefinition {
                        base: function_base_definition.to_owned(),
                        undecorated_signatures,
                        captured_variables,
                        decorator_callees,
                    },
                )
                .is_none(),
            "Found multiple function definitions with the same function id"
        );
    }

    function_definitions
}

pub fn collect_function_base_definitions(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
    reversed_override_graph: &WholeProgramReversedOverrideGraph,
) -> WholeProgramFunctionDefinitions<FunctionBaseDefinition> {
    let step = StepLogger::start(
        "Indexing function definitions",
        "Indexed function definitions",
    );

    let base_definitions = dashmap::DashMap::new();

    ThreadPool::new().install(|| {
        slow_fun_monitor_scope(|slow_function_monitor| {
            handles.par_iter().for_each(|handle| {
                let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();
                let context =
                    ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                let base_definitions_for_module = slow_function_monitor.monitor_function(
                    || export_all_functions(reversed_override_graph, &context),
                    format!(
                        "Indexing function definitions for {}",
                        handle.module().as_str(),
                    ),
                    /* max_time_in_seconds */ 4,
                );
                base_definitions.insert(module_id, base_definitions_for_module);
            });
        })
    });

    step.finish();
    WholeProgramFunctionDefinitions(base_definitions.into_read_only())
}
