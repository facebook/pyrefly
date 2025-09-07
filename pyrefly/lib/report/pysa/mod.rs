/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Not;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::types::Overload;
use pyrefly_types::types::Type;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::thread_pool::ThreadPool;
use pyrefly_util::visit::Visit;
use rayon::prelude::*;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use tracing::debug;
use tracing::info;

use crate::alt::answers::Answers;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::BindingClass;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::module::module_info::ModuleInfo;
use crate::module::typeshed::typeshed;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::types::display::TypeDisplayContext;
use crate::types::stdlib::Stdlib;

/// Represents a unique identifier for a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
struct ModuleId(u32);

/// Represents a unique identifier for a class, inside a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
struct ClassId(u32);

impl ClassId {
    fn from_class(class: &Class) -> ClassId {
        ClassId(class.index().0)
    }
}

#[derive(Debug, Clone, Serialize)]
struct PysaProjectModule {
    module_id: ModuleId,
    module_name: String,            // e.g, `foo.bar`
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

#[derive(Debug, Clone, Serialize)]
enum ScopeParent {
    Function { location: String },
    Class { location: String },
    TopLevel,
}

/// Information needed from Pysa about a type.
#[derive(Debug, Clone, Serialize)]
struct PysaType {
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

    // The list of classes that this type refers to, after stripping Optional and Awaitable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    class_names: Vec<ClassRef>,
}

#[derive(Debug, Clone, Serialize)]
enum FunctionParameter {
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

#[derive(Debug, Clone, Serialize)]
enum FunctionParameters {
    List(Vec<FunctionParameter>),
    Ellipsis,
    ParamSpec,
}

#[derive(Debug, Clone, Serialize)]
struct FunctionSignature {
    parameters: FunctionParameters,
    return_annotation: PysaType,
}

#[derive(Debug, Clone, Serialize)]
struct ClassRef {
    module_id: ModuleId,
    module_name: String, // For debugging purposes only. Reader should use the module id.
    class_id: ClassId,
    class_name: String, // For debugging purposes only. Reader should use the class id.
}

impl ClassRef {
    fn from_class(class: &Class, module_ids: &ModuleIds) -> ClassRef {
        ClassRef {
            module_id: module_ids
                .get(ModuleKey::from_module(class.module()))
                .unwrap(),
            module_name: class.module_name().to_string(),
            class_id: ClassId::from_class(class),
            class_name: class.qname().id().to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct FunctionDefinition {
    name: String,
    parent: ScopeParent,
    undecorated_signatures: Vec<FunctionSignature>,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_overload: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_staticmethod: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_classmethod: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_property_getter: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_property_setter: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_stub: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// If this is a method, record the class it is defined in.
    defining_class: Option<ClassRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// If the method directly overrides a method in a parent class, we record that class.
    /// This is used for building overriding graphs.
    overridden_base_class: Option<ClassRef>,
}

#[derive(Debug, Clone, Serialize)]
struct DefinitionRef {
    module_id: ModuleId,
    module_name: String, // For debugging purposes only. Reader should use the module id.
    location: String,
    identifier: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClassDefinition {
    class_id: ClassId,
    name: String,
    bases: Vec<ClassRef>,
    parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_synthesized: bool, // True if this class was synthesized (e.g., from namedtuple), false if from actual `class X:` statement
    fields: Vec<String>,
}

/// Format of a module file `my.module:id.json`
/// Represents all the information Pysa needs about a given module.
#[derive(Debug, Clone, Serialize)]
struct PysaModuleFile {
    format_version: u32,
    module_id: ModuleId,
    module_name: String,
    source_path: ModulePathDetails,
    type_of_expression: HashMap<String, PysaType>,
    goto_definitions_of_expression: HashMap<String, Vec<DefinitionRef>>,
    function_definitions: HashMap<String, FunctionDefinition>,
    class_definitions: HashMap<String, ClassDefinition>,
    global_variables: HashSet<String>,
}

/// Represents what makes a module unique
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ModuleKey {
    name: ModuleName,
    path: ModulePath,
}

impl ModuleKey {
    fn from_handle(handle: &Handle) -> ModuleKey {
        ModuleKey {
            name: handle.module(),
            path: handle.path().clone(),
        }
    }

    fn from_module_info(module_info: &ModuleInfo) -> ModuleKey {
        ModuleKey {
            name: module_info.name(),
            path: module_info.path().clone(),
        }
    }

    fn from_module(module: &Module) -> ModuleKey {
        ModuleKey {
            name: module.name(),
            path: module.path().clone(),
        }
    }
}

struct ModuleIds(HashMap<ModuleKey, ModuleId>);

impl ModuleIds {
    /// Multiple python files can map to the same module name (e.g, `foo.bar`).
    /// This creates a unique and deterministic identifier for each handle.
    fn new(handles: &[Handle]) -> ModuleIds {
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

    fn get(&self, key: ModuleKey) -> Option<ModuleId> {
        self.0.get(&key).copied()
    }
}

fn location_key(range: &DisplayRange) -> String {
    format!(
        "{}:{}-{}:{}",
        range.start.line, range.start.column, range.end.line, range.end.column
    )
}

struct ModuleContext<'a> {
    handle: &'a Handle,
    transaction: &'a Transaction<'a>,
    bindings: &'a Bindings,
    answers: &'a Answers,
    stdlib: &'a Stdlib,
    module_info: &'a ModuleInfo,
    module_ids: &'a ModuleIds,
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

fn get_classes_of_type(type_: &Type, context: &ModuleContext) -> Vec<Class> {
    if let Some(inner) = strip_optional(type_) {
        return get_classes_of_type(inner, context);
    }
    if let Some(inner) = strip_awaitable(type_, context) {
        return get_classes_of_type(inner, context);
    }
    if let Some(inner) = strip_coroutine(type_, context) {
        return get_classes_of_type(inner, context);
    }
    match type_ {
        Type::ClassType(class_type) => vec![class_type.class_object().clone()],
        Type::Union(elements) => elements
            .iter()
            .flat_map(|inner| get_classes_of_type(inner, context))
            .collect(),
        Type::TypeAlias(alias) => get_classes_of_type(&alias.as_type(), context),
        _ => Vec::new(),
    }
}

impl PysaType {
    fn from_type(type_: &Type, context: &ModuleContext) -> PysaType {
        // Promote `Literal[..]` into `str` or `int`.
        let type_ = type_.clone().promote_literals(context.stdlib);
        let type_ = strip_self_type(type_);

        let string = string_for_type(&type_);

        PysaType {
            string,
            is_bool: is_scalar_type(&type_, context.stdlib.bool().class_object(), context),
            is_int: is_scalar_type(&type_, context.stdlib.int().class_object(), context),
            is_float: is_scalar_type(&type_, context.stdlib.float().class_object(), context),
            is_enum: is_scalar_type(&type_, context.stdlib.enum_class().class_object(), context),
            class_names: {
                let mut classes = get_classes_of_type(&type_, context);
                classes.sort();
                classes.dedup();
                classes
                    .into_iter()
                    .map(|class_type| ClassRef::from_class(&class_type, context.module_ids))
                    .collect()
            },
        }
    }
}

struct VisitorContext<'a> {
    module_context: &'a ModuleContext<'a>,
    type_of_expression: &'a mut HashMap<String, PysaType>,
    definitions_of_expression: &'a mut HashMap<String, Vec<DefinitionRef>>,
    global_variables: &'a mut HashSet<String>,
}

fn add_expression_definitions(
    range: &DisplayRange,
    definitions: Vec<FindDefinitionItemWithDocstring>,
    identifier: &str,
    context: &mut VisitorContext,
) {
    let callees = definitions
        .iter()
        .filter(|definition| {
            matches!(
                definition.metadata,
                DefinitionMetadata::Variable(Some(SymbolKind::Function | SymbolKind::Class))
                    | DefinitionMetadata::Attribute(_)
            )
        })
        .filter_map(|definition| {
            let module_info = &definition.module;
            let display_range = module_info.display_range(definition.definition_range);
            match context
                .module_context
                .module_ids
                .get(ModuleKey::from_module_info(module_info))
            {
                Some(module_id) => Some(DefinitionRef {
                    module_id,
                    module_name: module_info.name().to_string(),
                    location: location_key(&display_range),
                    identifier: identifier.to_owned(),
                }),
                None => {
                    debug!(
                        "Module {} was not type checked, ignoring.",
                        module_info.name()
                    );
                    None
                }
            }
        })
        .collect::<Vec<_>>();

    if callees.is_empty() {
        return;
    }

    assert!(
        context
            .definitions_of_expression
            .insert(location_key(range), callees)
            .is_none(),
        "Found expressions with the same location"
    );
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
                    location_key(&display_range),
                    PysaType::from_type(&type_, context.module_context)
                )
                .is_none(),
            "Found expressions with the same location"
        );
    }

    // For some AST nodes, try to find the definitions.
    match e {
        Expr::Name(name)
            if matches!(
                name.ctx,
                ExprContext::Load | ExprContext::Del | ExprContext::Invalid
            ) =>
        {
            let identifier = Ast::expr_name_identifier(name.clone());
            let display_range = context.module_context.module_info.display_range(range);

            let definitions = context
                .module_context
                .transaction
                .find_definition_for_name_use(
                    context.module_context.handle,
                    &identifier,
                    &FindPreference::default(),
                )
                .map_or(vec![], |d| vec![d]);

            add_expression_definitions(&display_range, definitions, name.id.as_str(), context);
        }
        Expr::Attribute(attribute) => {
            let display_range = context.module_context.module_info.display_range(range);
            let definitions = context
                .module_context
                .transaction
                .find_definition_for_attribute(
                    context.module_context.handle,
                    attribute.value.range(),
                    &attribute.attr,
                    &FindPreference::default(),
                );
            add_expression_definitions(
                &display_range,
                definitions,
                attribute.attr.as_str(),
                context,
            );
        }
        _ => {}
    };

    e.recurse(&mut |e| visit_expression(e, context));
}

fn visit_assign_target(target: &Expr, is_top_level: bool, context: &mut VisitorContext) {
    if !is_top_level {
        return;
    }

    Ast::expr_lvalue(target, &mut |global: &ExprName| {
        context.global_variables.insert(global.id.to_string());
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
                location: location_key(&module_info.display_range(class_def.name.range())),
            }),
            AnyNodeRef::StmtFunctionDef(fun_def) => Some(ScopeParent::Function {
                location: location_key(&module_info.display_range(fun_def.name.range())),
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
    let definition_binding = Key::Definition(function.undecorated.identifier.clone());
    let idx = context.bindings.key_to_idx(&definition_binding);
    context.answers.get_idx(idx).unwrap().arc_clone_ty()
}

fn get_undecorated_return_type(function: &DecoratedFunction, context: &ModuleContext) -> Type {
    let return_binding = Key::ReturnType(function.undecorated.identifier.clone());
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

fn get_super_class_member(
    class: &Class,
    field: &Name,
    context: &ModuleContext,
) -> Option<ClassRef> {
    context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
            solver.get_super_class_member(class, None, field)
        })
        .unwrap()
        .map(|member| ClassRef::from_class(&member.defining_class, context.module_ids))
}

fn export_all_functions(
    ast: &ModModule,
    context: &ModuleContext,
) -> HashMap<String, FunctionDefinition> {
    let mut function_definitions = HashMap::new();

    for function in get_all_functions(context.bindings, context.answers) {
        if !should_export_function(&function, context) {
            continue;
        }

        // We need the list of raw parameters, ignoring decorators.
        // For overloads, we need the list of all overloads, not just the current one.
        // To get it, we check if `get_function_type` returns `Type::Overload`.
        let decorated_type = get_function_type(&function, context);
        let undecorated_signatures = match decorated_type {
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
        };

        let display_range = context.module_info.display_range(function.id_range());
        let name = function.metadata().kind.as_func_id().func;
        let parent = get_scope_parent(ast, context.module_info, function.id_range());
        let overridden_base_class = function
            .defining_cls()
            .and_then(|class| get_super_class_member(class, &name, context));
        assert!(
            function_definitions
                .insert(
                    location_key(&display_range),
                    FunctionDefinition {
                        name: name.to_string(),
                        parent,
                        undecorated_signatures,
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
                        overridden_base_class,
                    }
                )
                .is_none(),
            "Found function definitions with the same location"
        );
    }

    function_definitions
}

fn get_all_classes(bindings: &Bindings, answers: &Answers) -> impl Iterator<Item = Class> {
    bindings
        .keys::<KeyClass>()
        .map(|idx| answers.get_idx(idx).unwrap().0.clone().unwrap())
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

fn export_all_classes(
    ast: &ModModule,
    context: &ModuleContext,
) -> HashMap<String, ClassDefinition> {
    let mut class_definitions = HashMap::new();

    for class_idx in context.bindings.keys::<KeyClass>() {
        let class = context
            .answers
            .get_idx(class_idx)
            .unwrap()
            .0
            .clone()
            .unwrap();
        let display_range = context.module_info.display_range(class.qname().range());
        let class_index = class.index();
        let parent = get_scope_parent(ast, context.module_info, class.qname().range());
        let metadata = context
            .answers
            .get_idx(context.bindings.key_to_idx(&KeyClassMetadata(class_index)))
            .unwrap();

        let is_synthesized = match context.bindings.get(class_idx) {
            BindingClass::FunctionalClassDef(_, _, _) => true,
            BindingClass::ClassDef(_) => false,
        };

        let fields = class
            .fields()
            .filter_map(|field| {
                match get_class_field(&class, field, context) {
                    // We want to exclude fields that are function definitions,
                    // since those are exported in `definitions_of_expression`.
                    // There is no easy way to know if a field matches a `def ..`
                    // statement, so just use the type and explicit annotation
                    // as a heuristic for now.
                    Some(class_field)
                        if class_field.ty().is_function_type()
                            && !class_field.has_explicit_annotation() =>
                    {
                        None // This is a method.
                    }
                    Some(_) => Some(field.to_string()),
                    _ => None,
                }
            })
            .collect();

        let class_definition = ClassDefinition {
            class_id: ClassId::from_class(&class),
            name: class.qname().id().to_string(),
            parent,
            bases: metadata
                .base_class_objects()
                .iter()
                .map(|base_class| ClassRef::from_class(base_class, context.module_ids))
                .collect::<Vec<_>>(),
            is_synthesized,
            fields,
        };

        assert!(
            class_definitions
                .insert(location_key(&display_range), class_definition)
                .is_none(),
            "Found class definitions with the same location"
        );
    }

    class_definitions
}

fn is_unittest_module(bindings: &Bindings, answers: &Answers) -> bool {
    get_all_classes(bindings, answers).any(|class| {
        match &*answers
            .get_idx(bindings.key_to_idx(&KeyClassMro(class.index())))
            .unwrap()
        {
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
fn is_test_module(handle: &Handle, transaction: &Transaction) -> bool {
    let bindings = &transaction.get_bindings(handle).unwrap();
    let answers = &*transaction.get_answers(handle).unwrap();
    let ast = &*transaction.get_ast(handle).unwrap();

    is_unittest_module(bindings, answers) || is_pytest_module(bindings, answers, ast)
}

fn get_module_file(
    handle: &Handle,
    module_id: ModuleId,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> PysaModuleFile {
    let module_info = &transaction.get_module_info(handle).unwrap();

    let ast = &*transaction.get_ast(handle).unwrap();
    let bindings = &transaction.get_bindings(handle).unwrap();
    let answers = &*transaction.get_answers(handle).unwrap();
    let stdlib = &*transaction.get_stdlib(handle);

    let mut type_of_expression = HashMap::new();
    let mut definitions_of_expression = HashMap::new();
    let mut global_variables = HashSet::new();
    let context = ModuleContext {
        handle,
        transaction,
        bindings,
        answers,
        stdlib,
        module_info,
        module_ids,
    };

    for stmt in &ast.body {
        visit_statement(
            stmt,
            /* is_top_level */ true,
            &mut VisitorContext {
                module_context: &context,
                type_of_expression: &mut type_of_expression,
                definitions_of_expression: &mut definitions_of_expression,
                global_variables: &mut global_variables,
            },
        );
    }

    let function_definitions = export_all_functions(ast, &context);
    let class_definitions = export_all_classes(ast, &context);

    PysaModuleFile {
        format_version: 1,
        module_id,
        module_name: module_info.name().to_string(),
        source_path: module_info.path().details().clone(),
        type_of_expression,
        goto_definitions_of_expression: definitions_of_expression,
        function_definitions,
        class_definitions,
        global_variables,
    }
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
                        module_name: handle.module().to_string(),
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

    // Retrieve and dump information about each module, in parallel.
    ThreadPool::new().install(|| -> anyhow::Result<()> {
        module_info_tasks.into_par_iter().try_for_each(
            |(handle, module_id, info_path)| -> anyhow::Result<()> {
                let writer = BufWriter::new(File::create(
                    results_directory.join("modules").join(info_path),
                )?);
                serde_json::to_writer(
                    writer,
                    &get_module_file(handle, module_id, transaction, &module_ids),
                )?;

                if is_test_module(handle, transaction) {
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
