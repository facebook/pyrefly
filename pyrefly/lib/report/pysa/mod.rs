/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::visit::Visit;
use rayon::prelude::*;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use serde::Serialize;
use tracing::debug;
use tracing::info;

use crate::alt::answers::Answers;
use crate::module::module_info::ModuleInfo;
use crate::module::typeshed::typeshed;
use crate::state::handle::Handle;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;
use crate::types::display::TypeDisplayContext;
use crate::types::stdlib::Stdlib;

/// Represents a unique identifier for a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
struct ModuleId(u32);

#[derive(Debug, Clone, Serialize)]
struct PysaProjectModule {
    module_id: ModuleId,
    module_name: String,            // e.g, `foo.bar`
    source_path: ModulePathDetails, // Path to the source code
    info_path: Option<PathBuf>,     // Path to the PysaModuleFile
}

/// Format of the index file `pyrefly.pysa.json`
#[derive(Debug, Clone, Serialize)]
struct PysaProjectFile {
    format_version: u32,
    modules: HashMap<ModuleId, PysaProjectModule>,
}

#[derive(Debug, Clone, Serialize)]
struct DefinitionRef {
    module_id: ModuleId,
    module_name: String, // For debugging purposes only. Reader should use the module id.
    location: String,
    identifier: String,
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

    fn from_info(module_info: &ModuleInfo) -> ModuleKey {
        ModuleKey {
            name: module_info.name(),
            path: module_info.path().clone(),
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
            let module_info = &definition.location.module;
            let display_range = module_info.display_range(definition.location.range);
            match context.module_ids.get(ModuleKey::from_info(module_info)) {
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
                .find_definition_for_name_use(context.handle, &identifier, true)
                .map_or(vec![], |d| vec![d]);

            add_expression_definitions(&display_range, definitions, name.id.as_str(), context);
        }
        Expr::Attribute(attribute) => {
            let display_range = context.module_info.display_range(range);
            let definitions = context.transaction.find_definition_for_attribute(
                context.handle,
                attribute.value.range(),
                &attribute.attr,
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

fn get_module_file(
    handle: &Handle,
    module_id: ModuleId,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> PysaModuleFile {
    let module_info = &transaction.get_module_info(handle).unwrap();

    let ast = &*transaction.get_ast(handle).unwrap();
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

    PysaModuleFile {
        format_version: 1,
        module_id,
        module_name: module_info.name().to_string(),
        source_path: module_info.path().details().clone(),
        type_of_expression,
        goto_definitions_of_expression: definitions_of_expression,
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
                Some(results_directory.join("modules").join(format!(
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
                    }
                )
                .is_none(),
            "Found multiple handles with the same module id"
        );

        if let Some(info_path) = info_path {
            module_info_tasks.push((handle, module_id, info_path));
        }
    }

    // Dump information about each module, in parallel.
    module_info_tasks.into_par_iter().try_for_each(
        |(handle, module_id, info_path)| -> anyhow::Result<()> {
            let writer = BufWriter::new(File::create(info_path)?);
            serde_json::to_writer(
                writer,
                &get_module_file(handle, module_id, transaction, &module_ids),
            )?;
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

    let writer = BufWriter::new(File::create(results_directory.join("pyrefly.pysa.json"))?);
    serde_json::to_writer(
        writer,
        &PysaProjectFile {
            format_version: 1,
            modules: project_modules,
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
