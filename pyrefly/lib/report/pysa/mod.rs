/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod call_graph;
mod override_graph;

use core::panic;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Not;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use dashmap::DashMap;
use dupe::Dupe;
use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
#[cfg(test)]
use pyrefly_types::class::ClassType;
use pyrefly_types::types::Overload;
use pyrefly_types::types::Type;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::thread_pool::ThreadPool;
use pyrefly_util::visit::Visit;
use rayon::prelude::*;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use starlark_map::Hashed;
use tracing::info;

use crate::alt::answers::Answers;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::module::typeshed::typeshed;
use crate::report::pysa::override_graph::OverrideGraph;
use crate::report::pysa::override_graph::create_reversed_override_graph_for_module;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;
use crate::types::display::TypeDisplayContext;
use crate::types::stdlib::Stdlib;

/// Represents a unique identifier for a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ModuleId(u32);

/// Represents a unique identifier for a class, inside a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ClassId(u32);

impl ClassId {
    pub fn from_class(class: &Class) -> ClassId {
        ClassId(class.index().0)
    }

    #[cfg(test)]
    pub fn from_int(id: u32) -> ClassId {
        ClassId(id)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PysaLocation(pub DisplayRange);

pub fn location_key(range: &DisplayRange) -> String {
    format!(
        "{}:{}-{}:{}",
        range.start.line, range.start.column, range.end.line, range.end.column
    )
}

impl std::fmt::Debug for PysaLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PysaLocation({})", location_key(&self.0))
    }
}

impl Serialize for PysaLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&location_key(&self.0))
    }
}

/// Represents a unique identifier for a function, inside a module
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FunctionId {
    Function {
        location: PysaLocation,
    },
    ModuleTopLevel,
    #[expect(dead_code)]
    ClassTopLevel {
        class_id: ClassId,
    },
}

impl FunctionId {
    fn serialize_to_string(&self) -> String {
        match self {
            FunctionId::Function { location } => format!("F:{}", location_key(&location.0)),
            FunctionId::ModuleTopLevel => "MTL".to_owned(),
            FunctionId::ClassTopLevel { class_id } => format!("CTL:{}", class_id.0),
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

#[derive(Debug, Clone, Serialize)]
struct PysaProjectModule {
    module_id: ModuleId,
    module_name: ModuleName,        // e.g, `foo.bar`
    source_path: ModulePathDetails, // Path to the source code
    info_path: Option<PathBuf>,     // Path to the PysaModuleFile
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_test: bool, // Uses a set of heuristics to determine if the module is a test file.
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_interface: bool, // Is this a .pyi file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_init: bool, // Is this a __init__.py(i) file?
}

/// Format of the index file `pyrefly.pysa.json`
#[derive(Debug, Clone, Serialize)]
struct PysaProjectFile {
    format_version: u32,
    modules: HashMap<ModuleId, PysaProjectModule>,
    builtin_module_id: ModuleId,
    object_class_id: ClassId,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ScopeParent {
    Function { location: PysaLocation },
    Class { location: PysaLocation },
    TopLevel,
}

// List of class names that a type refers to, after stripping Optional and Awaitable.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassNamesFromType {
    class_names: Vec<ClassRef>,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_coroutine: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_optional: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_readonly: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    unbound_type_variable: bool,
    // Is there an element (after stripping) that isn't a class name?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_exhaustive: bool,
}

/// Information needed from Pysa about a type.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PysaType {
    // Pretty string representation of the type. Usually meant for the user.
    string: String,

    // Whether the type is a bool/int/float/enum, after stripping Optional and Awaitable.
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_bool: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_int: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_float: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_enum: bool,

    #[serde(skip_serializing_if = "ClassNamesFromType::skip_serializing")]
    class_names: ClassNamesFromType,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum FunctionParameter {
    PosOnly {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    Pos {
        name: String,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    VarArg {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        annotation: PysaType,
    },
    KwOnly {
        name: String,
        annotation: PysaType,
        #[serde(skip_serializing_if = "<&bool>::not")]
        required: bool,
    },
    Kwargs {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClassRef {
    pub module_id: ModuleId,
    pub module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    pub class_id: ClassId,
    pub class_name: String, // For debugging purposes only. Reader should use the class id.
}

impl ClassRef {
    pub fn from_class(class: &Class, module_ids: &ModuleIds) -> ClassRef {
        ClassRef {
            module_id: module_ids
                .get(ModuleKey::from_module(class.module()))
                .unwrap(),
            module_name: class.module_name(),
            class_id: ClassId::from_class(class),
            class_name: class.qname().id().to_string(),
        }
    }
}

/// Only store memory-efficient information from `FunctionDefinition`
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionBaseDefinition {
    pub name: String,
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
    pub overridden_base_method: Option<DefinitionRef>,
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
    DashMap<ModuleId, ModuleFunctionDefinitions<FunctionDefinition>>,
);

impl<GenericFunctionDefinition> WholeProgramFunctionDefinitions<GenericFunctionDefinition> {
    pub fn new() -> WholeProgramFunctionDefinitions<GenericFunctionDefinition> {
        WholeProgramFunctionDefinitions(DashMap::new())
    }

    pub fn get_and_map<T, F>(
        &self,
        module_id: ModuleId,
        function_id: &FunctionId,
        f: F,
    ) -> Option<T>
    where
        F: FnOnce(&GenericFunctionDefinition) -> T,
    {
        // We cannot return a &GenericFunctionDefinition since the reference is only valid
        // while we are holding a lock on the hash map.
        // Instead, we allow to call a function f that will and copy the information we need.
        self.0
            .get(&module_id)
            .and_then(|functions| functions.0.get(function_id).map(f))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub struct DefinitionRef {
    module_id: ModuleId,
    pub(crate) module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    function_id: FunctionId,
    pub(crate) identifier: String, // For debugging purposes only
}

impl DefinitionRef {
    fn from_decorated_function(function: &DecoratedFunction, context: &ModuleContext) -> Self {
        let name = function.metadata().kind.as_func_id().func;
        let display_range = context.module_info.display_range(function.id_range());
        DefinitionRef {
            module_id: context
                .module_ids
                .get(ModuleKey::from_module(&context.module_info))
                .unwrap(),
            module_name: context.module_info.name(),
            function_id: FunctionId::Function {
                location: PysaLocation(display_range),
            },
            identifier: name.to_string(),
        }
    }

    fn from_find_definition_item_with_docstring(
        item: &FindDefinitionItemWithDocstring,
        function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
        context: &ModuleContext,
    ) -> Option<Self> {
        // TODO: For overloads, return the last definition instead of the one from go-to-definitions.
        let display_range = item.module.display_range(item.definition_range);
        let function_id = FunctionId::Function {
            location: PysaLocation(display_range),
        };
        let module_id = context
            .module_ids
            .get(ModuleKey::from_module(&item.module))
            .unwrap();
        function_base_definitions.get_and_map(module_id, &function_id, |function_base_definition| {
            DefinitionRef {
                module_id,
                module_name: item.module.name(),
                function_id: function_id.clone(),
                identifier: function_base_definition.name.clone(),
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum PysaClassMro {
    Resolved(Vec<ClassRef>),
    Cyclic,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PysaClassField {
    #[serde(rename = "type")]
    pub type_: PysaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_annotation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<PysaLocation>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GlobalVariable {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<PysaType>,
    pub location: PysaLocation,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassDefinition {
    pub class_id: ClassId,
    pub name: String,
    pub bases: Vec<ClassRef>,
    pub mro: PysaClassMro,
    pub parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_synthesized: bool, // True if this class was synthesized (e.g., from namedtuple), false if from actual `class X:` statement
    pub fields: HashMap<String, PysaClassField>,
}

impl ClassDefinition {
    #[cfg(test)]
    pub fn with_bases(mut self, bases: Vec<ClassRef>) -> Self {
        self.bases = bases;
        self
    }

    #[cfg(test)]
    pub fn with_mro(mut self, mro: PysaClassMro) -> Self {
        self.mro = mro;
        self
    }

    #[cfg(test)]
    pub fn with_fields(mut self, fields: HashMap<String, PysaClassField>) -> Self {
        self.fields = fields;
        self
    }
}

/// Format of a module file `my.module:id.json`
/// Represents all the information Pysa needs about a given module.
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleFile {
    format_version: u32,
    module_id: ModuleId,
    module_name: ModuleName,
    source_path: ModulePathDetails,
    type_of_expression: HashMap<PysaLocation, PysaType>,
    function_definitions: ModuleFunctionDefinitions<FunctionDefinition>,
    class_definitions: HashMap<PysaLocation, ClassDefinition>,
    global_variables: HashMap<String, GlobalVariable>,
}

impl PysaModuleFile {
    #[cfg(test)]
    pub fn global_variables(&self) -> &HashMap<String, GlobalVariable> {
        &self.global_variables
    }
}

/// Represents what makes a module unique
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleKey {
    name: ModuleName,
    path: ModulePath,
}

impl ModuleKey {
    pub fn from_handle(handle: &Handle) -> ModuleKey {
        ModuleKey {
            name: handle.module(),
            path: handle.path().dupe(),
        }
    }

    pub fn from_module(module: &Module) -> ModuleKey {
        ModuleKey {
            name: module.name(),
            path: module.path().dupe(),
        }
    }
}

pub struct ModuleIds(HashMap<ModuleKey, ModuleId>);

impl ModuleIds {
    /// Multiple python files can map to the same module name (e.g, `foo.bar`).
    /// This creates a unique and deterministic identifier for each handle.
    pub fn new(handles: &[Handle]) -> ModuleIds {
        let mut modules = handles
            .iter()
            .map(ModuleKey::from_handle)
            .collect::<Vec<_>>();
        modules.sort();

        let mut result = HashMap::new();
        let mut current_id = 1;
        for module in modules {
            assert!(
                result.insert(module, ModuleId(current_id)).is_none(),
                "Found multiple handles with the same module name and path"
            );
            current_id += 1;
        }
        ModuleIds(result)
    }

    pub fn get(&self, key: ModuleKey) -> Option<ModuleId> {
        self.0.get(&key).copied()
    }
}

pub struct ModuleContext<'a> {
    pub handle: &'a Handle,
    pub transaction: &'a Transaction<'a>,
    pub bindings: Bindings,
    pub answers: Arc<Answers>,
    pub stdlib: Arc<Stdlib>,
    pub ast: Arc<ModModule>,
    pub module_info: Module,
    pub module_id: ModuleId,
    pub module_ids: &'a ModuleIds,
}

impl ModuleContext<'_> {
    pub fn create<'a>(
        handle: &'a Handle,
        transaction: &'a Transaction<'a>,
        module_ids: &'a ModuleIds,
    ) -> Option<ModuleContext<'a>> {
        Some(ModuleContext {
            handle,
            transaction,
            bindings: transaction.get_bindings(handle)?,
            answers: transaction.get_answers(handle)?,
            stdlib: transaction.get_stdlib(handle),
            ast: transaction.get_ast(handle)?,
            module_info: transaction.get_module_info(handle)?,
            module_id: module_ids.get(ModuleKey::from_handle(handle))?,
            module_ids,
        })
    }
}

fn string_for_type(type_: &Type) -> String {
    let mut ctx = TypeDisplayContext::new(&[type_]);
    ctx.always_display_module_name_except_builtins();
    ctx.display(type_).to_string()
}

fn strip_self_type(mut ty: Type) -> Type {
    ty.transform_mut(&mut |t| {
        if let Type::SelfType(cls) = t {
            *t = Type::ClassType(cls.clone());
        }
    });
    ty
}

fn has_superclass(class: &Class, want: &Class, context: &ModuleContext) -> bool {
    context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
            solver.type_order().has_superclass(class, want)
        })
        .unwrap()
}

impl ClassNamesFromType {
    pub fn from_class(class: &Class, context: &ModuleContext) -> ClassNamesFromType {
        ClassNamesFromType {
            class_names: vec![ClassRef::from_class(class, context.module_ids)],
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive: true,
        }
    }

    #[cfg(test)]
    pub fn from_classes(class_names: Vec<ClassRef>, is_exhaustive: bool) -> ClassNamesFromType {
        ClassNamesFromType {
            class_names,
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive,
        }
    }

    pub fn not_a_class() -> ClassNamesFromType {
        ClassNamesFromType {
            class_names: vec![],
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive: false,
        }
    }

    fn skip_serializing(&self) -> bool {
        self.class_names.is_empty()
    }

    pub fn with_strip_optional(mut self, stripped_optional: bool) -> ClassNamesFromType {
        self.stripped_optional = stripped_optional;
        self
    }

    pub fn with_strip_coroutine(mut self, stripped_coroutine: bool) -> ClassNamesFromType {
        self.stripped_coroutine = stripped_coroutine;
        self
    }

    fn join_with(mut self, other: ClassNamesFromType) -> ClassNamesFromType {
        self.class_names.extend(other.class_names);
        self.stripped_coroutine |= other.stripped_coroutine;
        self.stripped_optional |= other.stripped_optional;
        self.stripped_readonly |= other.stripped_readonly;
        self.unbound_type_variable |= other.unbound_type_variable;
        self.is_exhaustive &= other.is_exhaustive;
        self
    }

    fn sort_and_dedup(mut self) -> ClassNamesFromType {
        self.class_names.sort();
        self.class_names.dedup();
        self
    }
}

fn strip_optional(type_: &Type) -> Option<&Type> {
    match type_ {
        Type::Union(elements) if elements.len() == 2 && elements[0] == Type::None => {
            Some(&elements[1])
        }
        Type::Union(elements) if elements.len() == 2 && elements[1] == Type::None => {
            Some(&elements[0])
        }
        _ => None,
    }
}

fn strip_awaitable<'a>(type_: &'a Type, context: &ModuleContext) -> Option<&'a Type> {
    match type_ {
        Type::ClassType(class_type)
            if class_type.class_object() == context.stdlib.awaitable_object()
                && class_type.targs().as_slice().len() == 1 =>
        {
            Some(&class_type.targs().as_slice()[0])
        }
        _ => None,
    }
}

fn strip_coroutine<'a>(type_: &'a Type, context: &ModuleContext) -> Option<&'a Type> {
    match type_ {
        Type::ClassType(class_type)
            if class_type.class_object() == context.stdlib.coroutine_object()
                && class_type.targs().as_slice().len() >= 3 =>
        {
            Some(&class_type.targs().as_slice()[2])
        }
        _ => None,
    }
}

fn is_scalar_type(get: &Type, want: &Class, context: &ModuleContext) -> bool {
    if let Some(inner) = strip_optional(get) {
        return is_scalar_type(inner, want, context);
    }
    if let Some(inner) = strip_awaitable(get, context) {
        return is_scalar_type(inner, want, context);
    }
    if let Some(inner) = strip_coroutine(get, context) {
        return is_scalar_type(inner, want, context);
    }
    match get {
        Type::ClassType(class_type) => has_superclass(class_type.class_object(), want, context),
        Type::TypeAlias(alias) => is_scalar_type(&alias.as_type(), want, context),
        _ => false,
    }
}

fn get_classes_of_type(type_: &Type, context: &ModuleContext) -> ClassNamesFromType {
    if let Some(inner) = strip_optional(type_) {
        return get_classes_of_type(inner, context).with_strip_optional(true);
    }
    if let Some(inner) = strip_awaitable(type_, context) {
        return get_classes_of_type(inner, context).with_strip_coroutine(true);
    }
    if let Some(inner) = strip_coroutine(type_, context) {
        return get_classes_of_type(inner, context).with_strip_coroutine(true);
    }
    // No need to strip ReadOnly[], it is already stripped by pyrefly.
    match type_ {
        Type::ClassType(class_type) => {
            ClassNamesFromType::from_class(class_type.class_object(), context)
        }
        Type::Tuple(_) => ClassNamesFromType::from_class(context.stdlib.tuple_object(), context),
        Type::Union(elements) if !elements.is_empty() => elements
            .iter()
            .map(|inner| get_classes_of_type(inner, context))
            .reduce(|acc, next| acc.join_with(next))
            .unwrap()
            .sort_and_dedup(),
        Type::TypeAlias(alias) => get_classes_of_type(&alias.as_type(), context),
        _ => ClassNamesFromType::not_a_class(),
    }
}

impl PysaType {
    #[cfg(test)]
    pub fn new(string: String, class_names: ClassNamesFromType) -> PysaType {
        PysaType {
            string,
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names,
        }
    }

    #[cfg(test)]
    pub fn with_is_bool(mut self, is_bool: bool) -> PysaType {
        self.is_bool = is_bool;
        self
    }

    #[cfg(test)]
    pub fn with_is_int(mut self, is_int: bool) -> PysaType {
        self.is_int = is_int;
        self
    }

    #[cfg(test)]
    pub fn with_is_float(mut self, is_float: bool) -> PysaType {
        self.is_float = is_float;
        self
    }

    #[cfg(test)]
    pub fn with_is_enum(mut self, is_enum: bool) -> PysaType {
        self.is_enum = is_enum;
        self
    }

    pub fn from_type(type_: &Type, context: &ModuleContext) -> PysaType {
        // Promote `Literal[..]` into `str` or `int`.
        let type_ = type_.clone().promote_literals(&context.stdlib);
        let type_ = strip_self_type(type_);

        let string = string_for_type(&type_);

        PysaType {
            string,
            is_bool: is_scalar_type(&type_, context.stdlib.bool().class_object(), context),
            is_int: is_scalar_type(&type_, context.stdlib.int().class_object(), context),
            is_float: is_scalar_type(&type_, context.stdlib.float().class_object(), context),
            is_enum: is_scalar_type(&type_, context.stdlib.enum_class().class_object(), context),
            class_names: get_classes_of_type(&type_, context),
        }
    }

    #[cfg(test)]
    pub fn from_class_type(class_type: &ClassType, context: &ModuleContext) -> PysaType {
        PysaType::from_type(&Type::ClassType(class_type.clone()), context)
    }

    #[cfg(test)]
    pub fn from_class(class: &Class, context: &ModuleContext) -> PysaType {
        PysaType::from_type(
            &Type::ClassType(ClassType::new(class.dupe(), Default::default())),
            context,
        )
    }

    #[cfg(test)]
    pub fn any_implicit() -> PysaType {
        PysaType {
            string: "Unknown".to_owned(),
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names: ClassNamesFromType::not_a_class(),
        }
    }

    #[cfg(test)]
    pub fn none() -> PysaType {
        PysaType {
            string: "None".to_owned(),
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names: ClassNamesFromType::not_a_class(),
        }
    }
}

struct VisitorContext<'a> {
    module_context: &'a ModuleContext<'a>,
    type_of_expression: &'a mut HashMap<PysaLocation, PysaType>,
    global_variables: &'a mut HashMap<String, GlobalVariable>,
}

fn export_function_parameter(param: &Param, context: &ModuleContext) -> FunctionParameter {
    match param {
        Param::PosOnly(name, ty, required) => FunctionParameter::PosOnly {
            name: name.as_ref().map(|n| n.to_string()),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::Pos(name, ty, required) => FunctionParameter::Pos {
            name: name.to_string(),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::VarArg(name, ty) => FunctionParameter::VarArg {
            name: name.as_ref().map(|n| n.to_string()),
            annotation: PysaType::from_type(ty, context),
        },
        Param::KwOnly(name, ty, required) => FunctionParameter::KwOnly {
            name: name.to_string(),
            annotation: PysaType::from_type(ty, context),
            required: matches!(required, pyrefly_types::callable::Required::Required),
        },
        Param::Kwargs(name, ty) => FunctionParameter::Kwargs {
            name: name.as_ref().map(|n| n.to_string()),
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
        Params::Ellipsis => FunctionParameters::Ellipsis,
        Params::ParamSpec(_, _) => FunctionParameters::ParamSpec,
    }
}

fn export_function_signature(function: &Callable, context: &ModuleContext) -> FunctionSignature {
    FunctionSignature {
        parameters: export_function_parameters(&function.params, context),
        return_annotation: PysaType::from_type(&function.ret, context),
    }
}

fn visit_expression(e: &Expr, context: &mut VisitorContext) {
    let range = e.range();

    // If the expression has a type, export it.
    if let Some(type_) = context.module_context.answers.get_type_trace(range) {
        let display_range = context.module_context.module_info.display_range(range);

        assert!(
            context
                .type_of_expression
                .insert(
                    PysaLocation(display_range),
                    PysaType::from_type(&type_, context.module_context)
                )
                .is_none(),
            "Found expressions with the same location"
        );
    }

    e.recurse(&mut |e| visit_expression(e, context));
}

fn visit_assign_target(target: &Expr, is_top_level: bool, context: &mut VisitorContext) {
    if !is_top_level {
        return;
    }

    Ast::expr_lvalue(target, &mut |global: &ExprName| {
        let type_ = context
            .module_context
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&Key::Definition(ShortIdentifier::expr_name(
                global,
            ))))
            .and_then(|idx| context.module_context.answers.get_idx(idx));
        if let Some(type_) = type_.as_ref()
            && type_.ty().is_type_variable()
        {
            // Don't export type variable globals.
            return;
        }
        let location = PysaLocation(
            context
                .module_context
                .module_info
                .display_range(global.range()),
        );
        context
            .global_variables
            .entry(global.id.to_string())
            .or_insert(GlobalVariable {
                type_: type_.map(|type_| PysaType::from_type(type_.ty(), context.module_context)),
                location,
            });
    });
}

fn visit_statement(stmt: &Stmt, is_top_level: bool, context: &mut VisitorContext) {
    match stmt {
        Stmt::FunctionDef(function_def) => {
            visit_expressions(
                function_def
                    .decorator_list
                    .iter()
                    .map(|decorator| &decorator.expression),
                context,
            );
            visit_expressions(
                function_def
                    .parameters
                    .posonlyargs
                    .iter()
                    .filter_map(|argument| argument.default.as_deref()),
                context,
            );
            visit_expressions(
                function_def
                    .parameters
                    .args
                    .iter()
                    .filter_map(|argument| argument.default.as_deref()),
                context,
            );
            visit_expressions(
                function_def
                    .parameters
                    .kwonlyargs
                    .iter()
                    .filter_map(|argument| argument.default.as_deref()),
                context,
            );
            visit_statements(
                function_def.body.iter(),
                /* is_top_level */ false,
                context,
            );
        }
        Stmt::ClassDef(class_def) => {
            visit_expressions(
                class_def
                    .decorator_list
                    .iter()
                    .map(|decorator| &decorator.expression),
                context,
            );
            if let Some(arguments) = &class_def.arguments {
                visit_expressions(arguments.args.iter(), context);
                visit_expressions(
                    arguments.keywords.iter().map(|keyword| &keyword.value),
                    context,
                );
            }
            visit_statements(
                class_def.body.iter(),
                /* is_top_level */ false,
                context,
            );
        }
        Stmt::Expr(e) => {
            visit_expression(&e.value, context);
        }
        Stmt::Assign(assign) => {
            stmt.visit(&mut |e| visit_expression(e, context));
            for t in &assign.targets {
                visit_assign_target(t, is_top_level, context);
            }
        }
        Stmt::AnnAssign(assign) => {
            stmt.visit(&mut |e| visit_expression(e, context));
            visit_assign_target(&assign.target, is_top_level, context);
        }
        Stmt::AugAssign(assign) => {
            stmt.visit(&mut |e| visit_expression(e, context));
            visit_assign_target(&assign.target, is_top_level, context);
        }
        Stmt::Return(_) | Stmt::Delete(_) | Stmt::Raise(_) => {
            // Statements that only contains expressions, use Visit<Expr>
            stmt.visit(&mut |e| visit_expression(e, context));
        }
        Stmt::For(for_stmt) => {
            visit_expression(&for_stmt.iter, context);
            visit_expression(&for_stmt.target, context);
            visit_statements(for_stmt.body.iter(), is_top_level, context);
            visit_statements(for_stmt.orelse.iter(), is_top_level, context);
        }
        Stmt::While(while_stmt) => {
            visit_expression(&while_stmt.test, context);
            visit_statements(while_stmt.body.iter(), is_top_level, context);
            visit_statements(while_stmt.orelse.iter(), is_top_level, context);
        }
        Stmt::If(if_stmt) => {
            visit_expression(&if_stmt.test, context);
            visit_statements(if_stmt.body.iter(), is_top_level, context);
            for elif_else_clause in &if_stmt.elif_else_clauses {
                if let Some(test) = &elif_else_clause.test {
                    visit_expression(test, context);
                }
                visit_statements(elif_else_clause.body.iter(), is_top_level, context);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                visit_expression(&item.context_expr, context);
                visit_expressions(item.optional_vars.iter().map(|x| &**x), context);
            }
            visit_statements(with_stmt.body.iter(), is_top_level, context);
        }
        Stmt::Match(match_stmt) => {
            visit_expression(&match_stmt.subject, context);
            for case in &match_stmt.cases {
                if let Some(guard) = &case.guard {
                    visit_expression(guard, context);
                }
                visit_statements(case.body.iter(), is_top_level, context);
            }
        }
        Stmt::Try(try_stmt) => {
            visit_statements(try_stmt.body.iter(), is_top_level, context);
            visit_statements(try_stmt.orelse.iter(), is_top_level, context);
            visit_statements(try_stmt.finalbody.iter(), is_top_level, context);
            for ruff_python_ast::ExceptHandler::ExceptHandler(except_handler) in &try_stmt.handlers
            {
                if let Some(annotation) = &except_handler.type_ {
                    visit_expression(annotation, context);
                }
                visit_statements(except_handler.body.iter(), is_top_level, context);
            }
        }
        Stmt::Assert(assert_stmt) => {
            visit_expression(&assert_stmt.test, context);
            if let Some(msg) = &assert_stmt.msg {
                visit_expression(msg, context);
            }
        }
        Stmt::TypeAlias(_)
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

fn visit_expressions<'a>(
    expressions: impl Iterator<Item = &'a Expr>,
    context: &mut VisitorContext,
) {
    for expr in expressions {
        visit_expression(expr, context);
    }
}

fn visit_statements<'a>(
    statements: impl Iterator<Item = &'a Stmt>,
    is_top_level: bool,
    context: &mut VisitorContext,
) {
    for stmt in statements {
        visit_statement(stmt, is_top_level, context);
    }
}

fn get_scope_parent(ast: &ModModule, module_info: &Module, range: TextRange) -> ScopeParent {
    Ast::locate_node(ast, range.start())
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::Identifier(id) if id.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) if class_def.name.range() == range => None,
            AnyNodeRef::StmtFunctionDef(fun_def) if fun_def.name.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) => Some(ScopeParent::Class {
                location: PysaLocation(module_info.display_range(class_def.name.range())),
            }),
            AnyNodeRef::StmtFunctionDef(fun_def) => Some(ScopeParent::Function {
                location: PysaLocation(module_info.display_range(fun_def.name.range())),
            }),
            _ => None,
        })
        .unwrap_or(ScopeParent::TopLevel)
}

fn get_all_functions(
    bindings: &Bindings,
    answers: &Answers,
) -> impl Iterator<Item = DecoratedFunction> {
    bindings
        .keys::<KeyDecoratedFunction>()
        .map(|idx| DecoratedFunction::from_bindings_answers(idx, bindings, answers))
}

// Return the function type, considering decorators and overloads.
fn get_function_type(function: &DecoratedFunction, context: &ModuleContext) -> Type {
    let definition_binding = Key::Definition(function.undecorated.identifier);
    let idx = context.bindings.key_to_idx(&definition_binding);
    context.answers.get_idx(idx).unwrap().arc_clone_ty()
}

fn get_undecorated_return_type(function: &DecoratedFunction, context: &ModuleContext) -> Type {
    let return_binding = Key::ReturnType(function.undecorated.identifier);
    let idx = context.bindings.key_to_idx(&return_binding);
    context.answers.get_idx(idx).unwrap().arc_clone_ty()
}

fn should_export_function(function: &DecoratedFunction, context: &ModuleContext) -> bool {
    // We only want to export one function when we have an @overload chain.
    // If the function has no successor (function in the same scope with the same name), then we should export it.
    // If the function has successors, but is not an overload, then we should export it. It probably means the successor is a redefinition.
    let has_successor = context.bindings.get(function.idx).successor.is_some();
    !has_successor || !function.is_overload()
}

pub struct ModuleReversedOverrideGraph(HashMap<DefinitionRef, DefinitionRef>);

impl ModuleReversedOverrideGraph {}

pub struct WholeProgramReversedOverrideGraph(DashMap<DefinitionRef, DefinitionRef>);

impl WholeProgramReversedOverrideGraph {
    pub fn new() -> WholeProgramReversedOverrideGraph {
        WholeProgramReversedOverrideGraph(DashMap::new())
    }
}

fn get_undecorated_signatures(
    function: DecoratedFunction,
    context: &ModuleContext,
) -> Vec<FunctionSignature> {
    // We need the list of raw parameters, ignoring decorators.
    // For overloads, we need the list of all overloads, not just the current one.
    // To get it, we check if `get_function_type` returns `Type::Overload`.
    let decorated_type = get_function_type(&function, context);
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
                &get_undecorated_return_type(&function, context),
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

    for function in get_all_functions(&context.bindings, &context.answers) {
        if !should_export_function(&function, context) {
            continue;
        }

        let current_function = DefinitionRef::from_decorated_function(&function, context);
        let parent = get_scope_parent(&context.ast, &context.module_info, function.id_range());
        assert!(
            function_base_definitions
                .0
                .insert(
                    current_function.function_id.clone(),
                    FunctionBaseDefinition {
                        name: current_function.identifier.clone(),
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
                            .0
                            .get(&current_function)
                            .map(|r| r.clone()),
                    }
                )
                .is_none(),
            "Found function definitions with the same location"
        );
    }

    function_base_definitions
}

fn get_all_classes(bindings: &Bindings, answers: &Answers) -> impl Iterator<Item = Class> {
    bindings
        .keys::<KeyClass>()
        .map(|idx| answers.get_idx(idx).unwrap().0.dupe().unwrap())
}

fn get_class_field(
    class: &Class,
    field: &Name,
    context: &ModuleContext,
) -> Option<Arc<ClassField>> {
    context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
            solver.get_field_from_current_class_only(class, field)
        })
        .unwrap()
}

fn get_class_field_declaration<'a>(
    class: &'a Class,
    field: &'a Name,
    context: &'a ModuleContext,
) -> Option<&'a BindingClassField> {
    let key_class_field = KeyClassField(class.index(), field.clone());
    // We use `key_to_idx_hashed_opt` below because the key might not be valid (could be a synthesized field).
    context
        .bindings
        .key_to_idx_hashed_opt(Hashed::new(&key_class_field))
        .map(|idx| context.bindings.get(idx))
}

fn get_class_mro(class: &Class, bindings: &Bindings, answers: &Answers) -> Arc<ClassMro> {
    answers
        .get_idx(bindings.key_to_idx(&KeyClassMro(class.index())))
        .unwrap()
}

fn export_class_fields(class: &Class, context: &ModuleContext) -> HashMap<String, PysaClassField> {
    class
        .fields()
        .filter_map(|name| get_class_field(class, name, context).map(|field| (name, field)))
        .filter_map(|(name, field)| {
            let field_binding = get_class_field_declaration(class, name, context);

            let explicit_annotation = match field_binding {
                Some(BindingClassField {
                    definition: ClassFieldDefinition::DeclaredByAnnotation { annotation },
                    ..
                }) => Some(*annotation),
                Some(BindingClassField {
                    definition: ClassFieldDefinition::AssignedInBody { annotation, .. },
                    ..
                }) => *annotation,
                Some(BindingClassField {
                    definition: ClassFieldDefinition::DefinedInMethod { annotation, .. },
                    ..
                }) => *annotation,
                _ => None,
            }
            .map(|idx| context.bindings.idx_to_key(idx))
            .and_then(|key_annotation| match key_annotation {
                // We want to export the annotation as it is in the source code.
                // We cannot use the answer for `key_annotation` (which wraps a `Type`),
                // because it contains a normalized type where some elements have
                // been stripped out (most notably, `typing.Annotated`).
                KeyAnnotation::Annotation(identifier) => {
                    // `Ast::locate_node` returns all covering AST nodes, from innermost to outermost.
                    // The innermost will be the Name node, so we need the second node.
                    match Ast::locate_node(&context.ast, identifier.range().start()).get(1) {
                        Some(AnyNodeRef::StmtAnnAssign(assign)) => Some(
                            context
                                .module_info
                                .code_at(assign.annotation.range())
                                .to_owned(),
                        ),
                        _ => None,
                    }
                }
                KeyAnnotation::AttrAnnotation(range) => {
                    Some(context.module_info.code_at(*range).to_owned())
                }
                _ => None,
            });

            match field_binding {
                Some(BindingClassField {
                    definition: ClassFieldDefinition::MethodLike { .. },
                    ..
                }) => {
                    // Exclude fields that are functions definitons, because they are already exported in `function_definitions`.
                    None
                }
                Some(BindingClassField { range, .. }) => Some((
                    name.to_string(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: Some(PysaLocation(context.module_info.display_range(*range))),
                    },
                )),
                _ => Some((
                    name.to_string(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: None,
                    },
                )),
            }
        })
        .collect()
}

pub fn export_all_classes(context: &ModuleContext) -> HashMap<PysaLocation, ClassDefinition> {
    let mut class_definitions = HashMap::new();

    for class_idx in context.bindings.keys::<KeyClass>() {
        let class = context
            .answers
            .get_idx(class_idx)
            .unwrap()
            .0
            .dupe()
            .unwrap();
        let display_range = context.module_info.display_range(class.qname().range());
        let class_index = class.index();
        let parent = get_scope_parent(&context.ast, &context.module_info, class.qname().range());
        let metadata = context
            .answers
            .get_idx(context.bindings.key_to_idx(&KeyClassMetadata(class_index)))
            .unwrap();

        let is_synthesized = match context.bindings.get(class_idx) {
            BindingClass::FunctionalClassDef(_, _, _, _) => true,
            BindingClass::ClassDef(_) => false,
        };

        let fields = export_class_fields(&class, context);

        let bases = metadata
            .base_class_objects()
            .iter()
            .map(|base_class| ClassRef::from_class(base_class, context.module_ids))
            .collect::<Vec<_>>();

        let mro = match &*get_class_mro(&class, &context.bindings, &context.answers) {
            ClassMro::Resolved(mro) => PysaClassMro::Resolved(
                mro.iter()
                    .map(|class_type| {
                        ClassRef::from_class(class_type.class_object(), context.module_ids)
                    })
                    .collect(),
            ),
            ClassMro::Cyclic => PysaClassMro::Cyclic,
        };

        let class_definition = ClassDefinition {
            class_id: ClassId::from_class(&class),
            name: class.qname().id().to_string(),
            parent,
            bases,
            mro,
            is_synthesized,
            fields,
        };

        assert!(
            class_definitions
                .insert(PysaLocation(display_range), class_definition)
                .is_none(),
            "Found class definitions with the same location"
        );
    }

    class_definitions
}

fn is_unittest_module(bindings: &Bindings, answers: &Answers) -> bool {
    get_all_classes(bindings, answers).any(|class| {
        match &*get_class_mro(&class, bindings, answers) {
            ClassMro::Resolved(mro) => mro
                .iter()
                .any(|base| base.has_qname("unittest.case", "TestCase")),
            ClassMro::Cyclic => false,
        }
    })
}

fn is_pytest_module(bindings: &Bindings, answers: &Answers, ast: &ModModule) -> bool {
    fn has_pytest_prefix(name: &Name) -> bool {
        name == "pytest" || name.starts_with("pytest.")
    }
    fn imports_pytest(ast: &ModModule) -> bool {
        ast.body.iter().any(|stmt| match stmt {
            Stmt::Import(import_stmt) => import_stmt
                .names
                .iter()
                .any(|alias| has_pytest_prefix(&alias.name.id)),
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(module),
                ..
            }) => has_pytest_prefix(&module.id),
            _ => false,
        })
    }
    fn has_test_function(bindings: &Bindings, answers: &Answers) -> bool {
        get_all_functions(bindings, answers).any(|function| {
            function
                .metadata()
                .kind
                .as_func_id()
                .func
                .starts_with("test_")
        })
    }
    imports_pytest(ast) && has_test_function(bindings, answers)
}

/// Returns true if the module is considered a test module for Pysa.
/// In that case, we won't analyze the module at all.
///
/// We currently use the following heuristics:
/// - If a class inherits from `unittest.TestCase`, we assume this is a test file.
/// - If `pytest` is imported and at least one function starts with `test_`, we assume this is a test file.
pub fn is_test_module(context: &ModuleContext) -> bool {
    is_unittest_module(&context.bindings, &context.answers)
        || is_pytest_module(&context.bindings, &context.answers, &context.ast)
}

pub fn add_undecorated_signatures(
    function_base_definitions: &ModuleFunctionDefinitions<FunctionBaseDefinition>,
    context: &ModuleContext,
) -> ModuleFunctionDefinitions<FunctionDefinition> {
    let mut function_definitions = HashMap::new();
    for function in get_all_functions(&context.bindings, &context.answers) {
        let current_function = DefinitionRef::from_decorated_function(&function, context);
        if let Some(function_base_definition) = function_base_definitions
            .0
            .get(&current_function.function_id)
        {
            assert!(
                function_definitions
                    .insert(
                        current_function.function_id.clone(),
                        FunctionDefinition {
                            base: function_base_definition.to_owned(),
                            undecorated_signatures: get_undecorated_signatures(function, context),
                        },
                    )
                    .is_none(),
                "Found undecorated signatures for the same function"
            );
        }
    }
    ModuleFunctionDefinitions(function_definitions)
}

pub fn get_module_file(
    context: &ModuleContext,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
) -> PysaModuleFile {
    let mut type_of_expression = HashMap::new();
    let mut global_variables = HashMap::new();

    for stmt in &context.ast.body {
        visit_statement(
            stmt,
            /* is_top_level */ true,
            &mut VisitorContext {
                module_context: context,
                type_of_expression: &mut type_of_expression,
                global_variables: &mut global_variables,
            },
        );
    }

    let function_base_definitions_for_module =
        function_base_definitions.0.get(&context.module_id).unwrap();
    let function_definitions =
        add_undecorated_signatures(&function_base_definitions_for_module, context);
    let class_definitions = export_all_classes(context);

    PysaModuleFile {
        format_version: 1,
        module_id: context.module_id,
        module_name: context.module_info.name(),
        source_path: context.module_info.path().details().clone(),
        type_of_expression,
        function_definitions,
        class_definitions,
        global_variables,
    }
}

pub fn collect_function_base_definitions(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
    reversed_override_graph: &WholeProgramReversedOverrideGraph,
) -> WholeProgramFunctionDefinitions<FunctionBaseDefinition> {
    let base_definitions: WholeProgramFunctionDefinitions<FunctionBaseDefinition> =
        WholeProgramFunctionDefinitions::new();

    ThreadPool::new().install(|| {
        handles.par_iter().for_each(|handle| {
            let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();
            let base_definitions_for_module = export_all_functions(
                reversed_override_graph,
                &ModuleContext::create(handle, transaction, module_ids).unwrap(),
            );
            base_definitions
                .0
                .insert(module_id, base_definitions_for_module);
        });
    });

    base_definitions
}

pub fn build_reversed_override_graph(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> WholeProgramReversedOverrideGraph {
    let reversed_override_graph = WholeProgramReversedOverrideGraph::new();

    ThreadPool::new().install(|| {
        handles.par_iter().for_each(|handle| {
            let context = ModuleContext::create(handle, transaction, module_ids).unwrap();
            for (key, value) in create_reversed_override_graph_for_module(&context).0 {
                reversed_override_graph.0.insert(key, value);
            }
        });
    });

    reversed_override_graph
}

pub fn write_results(results_directory: &Path, transaction: &Transaction) -> anyhow::Result<()> {
    let start = Instant::now();
    info!("Writing results to `{}`", results_directory.display());
    fs_anyhow::create_dir_all(results_directory)?;
    fs_anyhow::create_dir_all(&results_directory.join("modules"))?;

    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);
    let mut project_modules = HashMap::new();

    let mut module_info_tasks = Vec::new();
    for handle in &handles {
        let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();

        // Path where we will store the information on the module.
        let info_path = match handle.path().details() {
            ModulePathDetails::Namespace(_) => {
                // Indicates a directory that contains a `__init__.py` file.
                None
            }
            _ => {
                Some(PathBuf::from(format!(
                    "{}:{}.json",
                    // Filename must be less than 255 bytes
                    String::from_iter(
                        handle
                            .module()
                            .to_string()
                            .chars()
                            .filter(|c| c.is_ascii())
                            .take(220)
                    ),
                    module_id.0
                )))
            }
        };

        assert!(
            project_modules
                .insert(
                    module_id,
                    PysaProjectModule {
                        module_id,
                        module_name: handle.module(),
                        source_path: handle.path().details().clone(),
                        info_path: info_path.clone(),
                        is_test: false,
                        is_interface: handle.path().is_interface(),
                        is_init: handle.path().is_init(),
                    }
                )
                .is_none(),
            "Found multiple handles with the same module id"
        );

        if let Some(info_path) = info_path {
            module_info_tasks.push((handle, module_id, info_path));
        }
    }

    let project_modules = Arc::new(Mutex::new(project_modules));

    let reversed_override_graph = build_reversed_override_graph(&handles, transaction, &module_ids);
    let function_base_definitions = collect_function_base_definitions(
        &handles,
        transaction,
        &module_ids,
        &reversed_override_graph,
    );

    let _override_graph = OverrideGraph::from_reversed(&reversed_override_graph);

    // Retrieve and dump information about each module, in parallel.
    ThreadPool::new().install(|| -> anyhow::Result<()> {
        module_info_tasks.into_par_iter().try_for_each(
            |(handle, module_id, info_path)| -> anyhow::Result<()> {
                let writer = BufWriter::new(File::create(
                    results_directory.join("modules").join(info_path),
                )?);
                let context = ModuleContext::create(handle, transaction, &module_ids).unwrap();
                serde_json::to_writer(
                    writer,
                    &get_module_file(&context, &function_base_definitions),
                )?;

                if is_test_module(&context) {
                    project_modules
                        .lock()
                        .unwrap()
                        .get_mut(&module_id)
                        .unwrap()
                        .is_test = true;
                }

                Ok(())
            },
        )
    })?;

    // Dump all typeshed files, so we can parse them.
    let typeshed = typeshed()?;
    for typeshed_module in typeshed.modules() {
        let module_path = typeshed.find(typeshed_module).unwrap();
        let relative_path = match module_path.details() {
            ModulePathDetails::BundledTypeshed(path) => path,
            _ => panic!("unexpected module path for typeshed module"),
        };
        let content = typeshed.load(relative_path).unwrap();
        let target_path = results_directory.join("typeshed").join(relative_path);
        fs_anyhow::create_dir_all(target_path.parent().unwrap())?;
        fs_anyhow::write(&target_path, content.as_bytes())?;
    }

    let builtin_module = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "builtins")
        .exactly_one()
        .expect("expected exactly one builtins module");
    let object_class_id = ClassId::from_class(
        transaction
            .get_stdlib(builtin_module)
            .object()
            .class_object(),
    );

    let project_modules = Arc::into_inner(project_modules)
        .unwrap()
        .into_inner()
        .unwrap();

    let writer = BufWriter::new(File::create(results_directory.join("pyrefly.pysa.json"))?);
    serde_json::to_writer(
        writer,
        &PysaProjectFile {
            format_version: 1,
            modules: project_modules,
            builtin_module_id: module_ids
                .get(ModuleKey::from_handle(builtin_module))
                .unwrap(),
            object_class_id,
        },
    )?;

    let elapsed = start.elapsed();
    info!(
        "Wrote results to `{}` in {:.2}s",
        results_directory.display(),
        elapsed.as_secs_f32()
    );

    Ok(())
}
