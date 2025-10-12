/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::ops::Not;

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::types::Overload;
use pyrefly_types::types::Type;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::prelude::*;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use serde::Serialize;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecoratedFunction;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::resolve_decorator_callees;
use crate::report::pysa::captured_variable::CapturedVariable;
use crate::report::pysa::captured_variable::ModuleCapturedVariables;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::override_graph::WholeProgramReversedOverrideGraph;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::scope::get_scope_parent;
use crate::report::pysa::step_logger::StepLogger;
use crate::report::pysa::types::PysaType;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

/// Represents a unique identifier for a function **within a module**.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FunctionId {
    Function { location: PysaLocation },
    ModuleTopLevel,
    ClassTopLevel { class_id: ClassId },
}

impl FunctionId {
    fn serialize_to_string(&self) -> String {
        match self {
            FunctionId::Function { location } => format!("F:{}", location.as_key()),
            FunctionId::ModuleTopLevel => "MTL".to_owned(),
            FunctionId::ClassTopLevel { class_id } => format!("CTL:{}", class_id.to_int()),
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub struct FunctionRef {
    pub module_id: ModuleId,
    pub module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    pub function_id: FunctionId,
    pub function_name: Name, // For debugging purposes only. Reader should use the function id.
}

// For many function implementations, we need to pass the module context where the function is defined.
fn assert_function_in_context(function: &DecoratedFunction, context: &ModuleContext) {
    match &function.undecorated.metadata.kind {
        FunctionKind::Def(func_id) => {
            assert_eq!(func_id.module, context.module_info.name());
        }
        _ => (),
    }
}

impl FunctionRef {
    pub fn from_decorated_function(function: &DecoratedFunction, context: &ModuleContext) -> Self {
        assert_function_in_context(function, context);
        assert!(should_export_function(function, context));
        let name = function.metadata().kind.as_func_id().func;
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

    pub fn from_find_definition_item_with_docstring(
        item: &FindDefinitionItemWithDocstring,
        function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
        context: &ModuleContext,
    ) -> Option<Self> {
        // TODO: For overloads, return the last definition instead of the one from go-to-definitions.
        let display_range = item.module.display_range(item.definition_range);
        let function_id = FunctionId::Function {
            location: PysaLocation::new(display_range),
        };
        let module_id = context
            .module_ids
            .get(ModuleKey::from_module(&item.module))
            .unwrap();
        function_base_definitions
            .get(module_id, &function_id)
            .map(|function_base_definition| FunctionRef {
                module_id,
                module_name: item.module.name(),
                function_id: function_id.clone(),
                function_name: function_base_definition.name.clone(),
            })
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

/// Only store memory-efficient information from `FunctionDefinition`
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
    pub fn with_decorator_callees(
        mut self,
        decorator_callees: HashMap<PysaLocation, Vec<Target<FunctionRef>>>,
    ) -> Self {
        self.decorator_callees = decorator_callees;
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

    pub fn get_for_module<'a>(
        &'a self,
        module_id: ModuleId,
    ) -> Option<&'a ModuleFunctionDefinitions<GenericFunctionDefinition>> {
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
                .collect(),
        ),
        Params::Ellipsis | Params::Materialization => FunctionParameters::Ellipsis,
        Params::ParamSpec(_, _) => FunctionParameters::ParamSpec,
    }
}

fn export_function_signature(function: &Callable, context: &ModuleContext) -> FunctionSignature {
    FunctionSignature {
        parameters: export_function_parameters(&function.params, context),
        return_annotation: PysaType::from_type(&function.ret, context),
    }
}

pub fn get_all_functions(context: &ModuleContext) -> impl Iterator<Item = DecoratedFunction> {
    context.bindings.keys::<KeyDecoratedFunction>().map(|idx| {
        DecoratedFunction::from_bindings_answers(idx, &context.bindings, &context.answers)
    })
}

// Return the function type, considering decorators and overloads.
fn get_function_type(function: &DecoratedFunction, context: &ModuleContext) -> Type {
    assert_function_in_context(function, context);
    let definition_binding = Key::Definition(function.undecorated.identifier);
    let idx = context.bindings.key_to_idx(&definition_binding);
    context.answers.get_idx(idx).unwrap().arc_clone_ty()
}

fn get_undecorated_return_type(function: &DecoratedFunction, context: &ModuleContext) -> Type {
    let return_binding = Key::ReturnType(function.undecorated.identifier);
    let idx = context.bindings.key_to_idx(&return_binding);
    context.answers.get_idx(idx).unwrap().arc_clone_ty()
}

pub fn should_export_function(function: &DecoratedFunction, context: &ModuleContext) -> bool {
    assert_function_in_context(function, context);
    // We only want to export one function when we have an @overload chain.
    // If the function has no successor (function in the same scope with the same name), then we should export it.
    // If the function has successors, but is not an overload, then we should export it. It probably means the successor is a redefinition.
    let has_successor = context.bindings.get(function.idx).successor.is_some();
    !has_successor || !function.is_overload()
}

fn get_undecorated_signatures(
    function: &DecoratedFunction,
    context: &ModuleContext,
) -> Vec<FunctionSignature> {
    // We need the list of raw parameters, ignoring decorators.
    // For overloads, we need the list of all overloads, not just the current one.
    // To get it, we check if `get_function_type` returns `Type::Overload`.
    let decorated_type = get_function_type(function, context);
    match decorated_type {
        Type::Overload(Overload { signatures, .. }) => signatures
            .iter()
            .map(|overload_type| match overload_type {
                pyrefly_types::types::OverloadType::Function(f) => f,
                pyrefly_types::types::OverloadType::Forall(pyrefly_types::types::Forall {
                    body,
                    ..
                }) => body,
            })
            .map(|function| export_function_signature(&function.signature, context))
            .collect::<Vec<_>>(),
        _ => vec![FunctionSignature {
            parameters: FunctionParameters::List(
                function
                    .undecorated
                    .params
                    .iter()
                    .map(|param| export_function_parameter(param, context))
                    .collect(),
            ),
            return_annotation: PysaType::from_type(
                &get_undecorated_return_type(function, context),
                context,
            ),
        }],
    }
}

pub fn export_all_functions(
    reversed_override_graph: &WholeProgramReversedOverrideGraph,
    context: &ModuleContext,
) -> ModuleFunctionDefinitions<FunctionBaseDefinition> {
    let mut function_base_definitions = ModuleFunctionDefinitions::new();

    for function in get_all_functions(context) {
        if !should_export_function(&function, context) {
            continue;
        }

        let current_function = FunctionRef::from_decorated_function(&function, context);
        let parent = get_scope_parent(&context.ast, &context.module_info, function.id_range());
        assert!(
            function_base_definitions
                .0
                .insert(
                    current_function.function_id.clone(),
                    FunctionBaseDefinition {
                        name: current_function.function_name.clone(),
                        parent,
                        is_overload: function.metadata().flags.is_overload,
                        is_staticmethod: function.metadata().flags.is_staticmethod,
                        is_classmethod: function.metadata().flags.is_classmethod,
                        is_property_getter: function.metadata().flags.is_property_getter,
                        is_property_setter: function
                            .metadata()
                            .flags
                            .is_property_setter_with_getter
                            .is_some(),
                        is_stub: function.is_stub(),
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

fn find_definition_ast<'a>(
    function: &DecoratedFunction,
    context: &'a ModuleContext<'a>,
) -> Option<&'a StmtFunctionDef> {
    let range = function.id_range();
    Ast::locate_node(&context.ast, range.start())
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::StmtFunctionDef(stmt) if stmt.name.range == range => Some(*stmt),
            _ => None,
        })
}

fn get_decorator_callees(
    function: &DecoratedFunction,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    context: &ModuleContext,
) -> HashMap<PysaLocation, Vec<Target<FunctionRef>>> {
    if let Some(function_def) = find_definition_ast(function, context) {
        resolve_decorator_callees(
            &function_def.decorator_list,
            function_base_definitions,
            context,
        )
    } else {
        HashMap::new()
    }
}

pub fn export_function_definitions(
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    captured_variables: &ModuleCapturedVariables,
    context: &ModuleContext,
) -> ModuleFunctionDefinitions<FunctionDefinition> {
    let mut function_definitions = ModuleFunctionDefinitions::new();
    let function_base_definitions_for_module = function_base_definitions
        .get_for_module(context.module_id)
        .unwrap();

    for function in get_all_functions(context) {
        if !should_export_function(&function, context) {
            continue;
        }
        let current_function = FunctionRef::from_decorated_function(&function, context);
        let function_base_definition = function_base_definitions_for_module
            .0
            .get(&current_function.function_id)
            .unwrap();
        let undecorated_signatures = get_undecorated_signatures(&function, context);

        let captured_variables = captured_variables
            .get(&current_function)
            .map(|set| set.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        let decorator_callees =
            get_decorator_callees(&function, function_base_definitions, context);

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
        handles.par_iter().for_each(|handle| {
            let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();
            let base_definitions_for_module = export_all_functions(
                reversed_override_graph,
                &ModuleContext::create(handle.clone(), transaction, module_ids).unwrap(),
            );
            base_definitions.insert(module_id, base_definitions_for_module);
        });
    });

    step.finish();
    WholeProgramFunctionDefinitions(base_definitions.into_read_only())
}
