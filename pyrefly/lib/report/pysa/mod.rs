/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Not;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use itertools::Itertools;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::class::Class;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::visit::Visit;
use rayon::prelude::*;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
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
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::BindingClass;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::module::module_info::ModuleInfo;
use crate::module::typeshed::typeshed;
use crate::state::handle::Handle;
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

#[derive(Debug, Clone, Serialize)]
struct FunctionDefinition {
    name: String,
    parent: ScopeParent,
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
}

#[derive(Debug, Clone, Serialize)]
struct DefinitionRef {
    module_id: ModuleId,
    module_name: String, // For debugging purposes only. Reader should use the module id.
    location: String,
    identifier: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClassRef {
    module_id: ModuleId,
    module_name: String, // For debugging purposes only. Reader should use the module id.
    class_id: ClassId,
    class_name: String, // For debugging purposes only. Reader should use the class id.
}

#[derive(Debug, Clone, Serialize)]
struct ClassDefinition {
    class_id: ClassId,
    name: String,
    bases: Vec<ClassRef>,
    parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_synthesized: bool, // True if this class was synthesized (e.g., from namedtuple), false if from actual `class X:` statement
}

/// Format of a module file `my.module:id.json`
/// Represents all the information Pysa needs about a given module.
#[derive(Debug, Clone, Serialize)]
struct PysaModuleFile {
    format_version: u32,
    module_id: ModuleId,
    module_name: String,
    source_path: ModulePathDetails,
    type_of_expression: HashMap<String, String>,
    goto_definitions_of_expression: HashMap<String, Vec<DefinitionRef>>,
    function_definitions: HashMap<String, FunctionDefinition>,
    class_definitions: HashMap<String, ClassDefinition>,
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

struct VisitorContext<'a> {
    handle: &'a Handle,
    module_ids: &'a ModuleIds,
    module_info: &'a ModuleInfo,
    answers: &'a Answers,
    stdlib: &'a Stdlib,
    transaction: &'a Transaction<'a>,
    type_of_expression: &'a mut HashMap<String, String>,
    definitions_of_expression: &'a mut HashMap<String, Vec<DefinitionRef>>,
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

fn visit_expression(e: &Expr, context: &mut VisitorContext) {
    let range = e.range();

    // If the expression has a type, export it.
    if let Some(type_) = context.answers.get_type_trace(range) {
        // Promote `Literal[..]` into `str` or `int`.
        let type_ = (*type_).clone().promote_literals(context.stdlib);

        let display_range = context.module_info.display_range(range);

        let mut ctx = TypeDisplayContext::new(&[&type_]);
        ctx.always_display_module_name();

        assert!(
            context
                .type_of_expression
                .insert(
                    location_key(&display_range),
                    ctx.display(&type_).to_string(),
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
            let display_range = context.module_info.display_range(range);

            let definitions = context
                .transaction
                .find_definition_for_name_use(
                    context.handle,
                    &identifier,
                    &FindPreference::default(),
                )
                .map_or(vec![], |d| vec![d]);

            add_expression_definitions(&display_range, definitions, name.id.as_str(), context);
        }
        Expr::Attribute(attribute) => {
            let display_range = context.module_info.display_range(range);
            let definitions = context.transaction.find_definition_for_attribute(
                context.handle,
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

fn visit_statement(stmt: &Stmt, context: &mut VisitorContext) {
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
            visit_statements(function_def.body.iter(), context);
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
            visit_statements(class_def.body.iter(), context);
        }
        Stmt::Expr(e) => {
            visit_expression(&e.value, context);
        }
        Stmt::Return(_)
        | Stmt::Delete(_)
        | Stmt::Assign(_)
        | Stmt::AugAssign(_)
        | Stmt::AnnAssign(_)
        | Stmt::Raise(_) => {
            // Statements that only contains expressions, use Visit<Expr>
            stmt.visit(&mut |e| visit_expression(e, context));
        }
        Stmt::For(for_stmt) => {
            visit_expression(&for_stmt.iter, context);
            visit_expression(&for_stmt.target, context);
            visit_statements(for_stmt.body.iter(), context);
            visit_statements(for_stmt.orelse.iter(), context);
        }
        Stmt::While(while_stmt) => {
            visit_expression(&while_stmt.test, context);
            visit_statements(while_stmt.body.iter(), context);
            visit_statements(while_stmt.orelse.iter(), context);
        }
        Stmt::If(if_stmt) => {
            visit_expression(&if_stmt.test, context);
            visit_statements(if_stmt.body.iter(), context);
            for elif_else_clause in &if_stmt.elif_else_clauses {
                if let Some(test) = &elif_else_clause.test {
                    visit_expression(test, context);
                }
                visit_statements(elif_else_clause.body.iter(), context);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                visit_expression(&item.context_expr, context);
                visit_expressions(item.optional_vars.iter().map(|x| &**x), context);
            }
            visit_statements(with_stmt.body.iter(), context);
        }
        Stmt::Match(match_stmt) => {
            visit_expression(&match_stmt.subject, context);
            for case in &match_stmt.cases {
                if let Some(guard) = &case.guard {
                    visit_expression(guard, context);
                }
                visit_statements(case.body.iter(), context);
            }
        }
        Stmt::Try(try_stmt) => {
            visit_statements(try_stmt.body.iter(), context);
            visit_statements(try_stmt.orelse.iter(), context);
            visit_statements(try_stmt.finalbody.iter(), context);
            for ruff_python_ast::ExceptHandler::ExceptHandler(except_handler) in &try_stmt.handlers
            {
                if let Some(annotation) = &except_handler.type_ {
                    visit_expression(annotation, context);
                }
                visit_statements(except_handler.body.iter(), context);
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

fn visit_statements<'a>(statements: impl Iterator<Item = &'a Stmt>, context: &mut VisitorContext) {
    for stmt in statements {
        visit_statement(stmt, context);
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

fn export_all_functions(
    ast: &ModModule,
    module_info: &Module,
    bindings: &Bindings,
    answers: &Answers,
) -> HashMap<String, FunctionDefinition> {
    let mut function_definitions = HashMap::new();

    for function in get_all_functions(bindings, answers) {
        let display_range = module_info.display_range(function.id_range());
        let name = function.metadata().kind.as_func_id().func.to_string();
        let parent = get_scope_parent(ast, module_info, function.id_range());
        assert!(
            function_definitions
                .insert(
                    location_key(&display_range),
                    FunctionDefinition {
                        name,
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

fn export_all_classes(
    ast: &ModModule,
    module_info: &Module,
    bindings: &Bindings,
    answers: &Answers,
    module_ids: &ModuleIds,
) -> HashMap<String, ClassDefinition> {
    let mut class_definitions = HashMap::new();

    for class_idx in bindings.keys::<KeyClass>() {
        let class = answers.get_idx(class_idx).unwrap().0.clone().unwrap();
        let display_range = module_info.display_range(class.qname().range());
        let class_index = class.index();
        let parent = get_scope_parent(ast, module_info, class.qname().range());
        let metadata = answers
            .get_idx(bindings.key_to_idx(&KeyClassMetadata(class_index)))
            .unwrap();

        let is_synthesized = match bindings.get(class_idx) {
            BindingClass::FunctionalClassDef(_, _, _) => true,
            BindingClass::ClassDef(_) => false,
        };

        let class_definition = ClassDefinition {
            class_id: ClassId::from_class(&class),
            name: class.qname().id().to_string(),
            parent,
            bases: metadata
                .base_class_objects()
                .iter()
                .map(|base_class| ClassRef {
                    module_id: module_ids
                        .get(ModuleKey::from_module(base_class.module()))
                        .unwrap(),
                    module_name: base_class.module_name().to_string(),
                    class_id: ClassId::from_class(base_class),
                    class_name: base_class.qname().id().to_string(),
                })
                .collect::<Vec<_>>(),
            is_synthesized,
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

    for stmt in &ast.body {
        visit_statement(
            stmt,
            &mut VisitorContext {
                handle,
                module_ids,
                module_info,
                answers,
                stdlib,
                transaction,
                type_of_expression: &mut type_of_expression,
                definitions_of_expression: &mut definitions_of_expression,
            },
        );
    }

    let function_definitions = export_all_functions(ast, module_info, bindings, answers);
    let class_definitions = export_all_classes(ast, module_info, bindings, answers, module_ids);

    PysaModuleFile {
        format_version: 1,
        module_id,
        module_name: module_info.name().to_string(),
        source_path: module_info.path().details().clone(),
        type_of_expression,
        goto_definitions_of_expression: definitions_of_expression,
        function_definitions,
        class_definitions,
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
    )?;

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
