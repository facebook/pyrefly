/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Detection and rendering of Run and Test CodeLens commands.

use lsp_types::CodeLens;
use lsp_types::Command;
use lsp_types::Range;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::telemetry::EmptyResponseReason;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::TextRange;
use serde_json::Value;

use super::super::Server;
use crate::state::state::Transaction;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunnableKind {
    Run,
    Test,
}

#[derive(Clone, Debug)]
struct RunnableEntry {
    range: TextRange,
    kind: RunnableKind,
    test_name: Option<String>,
    class_name: Option<String>,
    is_unittest: bool,
}

/// Collect and render all enabled runnable lenses for one Python file.
pub(super) fn code_lenses(
    server: &Server,
    transaction: &Transaction<'_>,
    handle: &Handle,
    uri: &Url,
) -> Result<Vec<CodeLens>, EmptyResponseReason> {
    let path = server
        .path_for_uri(uri)
        .ok_or(EmptyResponseReason::NoFilePath)?;
    let runnable_code_lens = server
        .workspaces
        .get_with(path.clone(), |(_, workspace)| workspace.runnable_code_lens);
    let maybe_cell_idx = server.maybe_get_code_cell_index(uri);
    let info = transaction
        .get_module_info(handle)
        .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
    let entries = runnable_entries(transaction, handle, uri, runnable_code_lens)
        .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
    let config = server.state.config_finder().python_file(
        ModuleNameWithKind::guaranteed(ModuleName::unknown()),
        &ModulePath::filesystem(path.clone()),
    );
    let cwd = config
        .source
        .root()
        .map(std::path::Path::to_path_buf)
        .or_else(|| {
            server
                .workspaces
                .get_with(path, |(workspace_root, _)| workspace_root.cloned())
        })
        .map(|path| path.to_string_lossy().into_owned());

    let mut lenses = Vec::new();
    for entry in entries {
        if info.to_cell_for_lsp(entry.range.start()) != maybe_cell_idx {
            continue;
        }
        let range = info.to_lsp_range(entry.range);
        lenses.push(entry_to_lsp_code_lens(uri, range, entry, cwd.as_deref()));
    }
    Ok(lenses)
}

/// Render a semantic runnable entry as an executable LSP command.
fn entry_to_lsp_code_lens(
    uri: &Url,
    range: Range,
    entry: RunnableEntry,
    cwd: Option<&str>,
) -> CodeLens {
    let (title, command, arguments) = match entry.kind {
        RunnableKind::Run => {
            let mut args = serde_json::Map::new();
            args.insert("uri".to_owned(), serde_json::json!(uri.to_string()));
            if let Some(cwd) = cwd {
                args.insert("cwd".to_owned(), serde_json::json!(cwd));
            }
            ("Run", "pyrefly.runMain", Some(vec![Value::Object(args)]))
        }
        RunnableKind::Test => {
            let mut args = serde_json::Map::new();
            args.insert("uri".to_owned(), serde_json::json!(uri.to_string()));
            if let Some(cwd) = cwd {
                args.insert("cwd".to_owned(), serde_json::json!(cwd));
            }
            args.insert(
                "position".to_owned(),
                serde_json::json!({
                    "line": range.start.line,
                    "character": range.start.character,
                }),
            );
            if let Some(test_name) = entry.test_name {
                args.insert("testName".to_owned(), serde_json::json!(test_name));
            }
            if let Some(class_name) = entry.class_name {
                args.insert("className".to_owned(), serde_json::json!(class_name));
            }
            args.insert(
                "isUnittest".to_owned(),
                serde_json::json!(entry.is_unittest),
            );
            ("Test", "pyrefly.runTest", Some(vec![Value::Object(args)]))
        }
    };

    CodeLens {
        range,
        command: Some(Command {
            title: title.to_owned(),
            command: command.to_owned(),
            arguments,
            tooltip: None,
        }),
        data: None,
    }
}

/// Extract runnable entries from the file AST without constructing LSP values.
fn runnable_entries(
    transaction: &Transaction<'_>,
    handle: &Handle,
    uri: &Url,
    runnable_code_lens: bool,
) -> Option<Vec<RunnableEntry>> {
    if !runnable_code_lens || uri.path().ends_with(".pyi") || uri.path().ends_with(".ipynb") {
        return Some(Vec::new());
    }
    let ast = transaction.get_ast(handle)?;
    let mut entries = Vec::new();
    collect_module_entries(&ast.body, &mut entries);
    Some(entries)
}

fn collect_module_entries(stmts: &[Stmt], entries: &mut Vec<RunnableEntry>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(func) => {
                maybe_push_test(entries, func.name.as_str(), func.name.range, None, false);
            }
            Stmt::ClassDef(class_def) => {
                let is_unittest = is_unittest_class(class_def);
                if is_test_class(class_def, is_unittest) {
                    entries.push(RunnableEntry {
                        range: class_def.name.range,
                        kind: RunnableKind::Test,
                        test_name: None,
                        class_name: Some(class_def.name.as_str().to_owned()),
                        is_unittest,
                    });
                }
                collect_class_entries(
                    &class_def.body,
                    entries,
                    class_def.name.as_str(),
                    is_unittest,
                );
            }
            Stmt::If(stmt_if) if Ast::is_main_guard(&stmt_if.test) => {
                entries.push(RunnableEntry {
                    range: stmt_if.range,
                    kind: RunnableKind::Run,
                    test_name: None,
                    class_name: None,
                    is_unittest: false,
                });
            }
            _ => {}
        }
    }
}

fn collect_class_entries(
    stmts: &[Stmt],
    entries: &mut Vec<RunnableEntry>,
    class_name: &str,
    is_unittest: bool,
) {
    for stmt in stmts {
        if let Stmt::FunctionDef(func) = stmt {
            maybe_push_test(
                entries,
                func.name.as_str(),
                func.name.range,
                Some(class_name.to_owned()),
                is_unittest,
            );
        }
    }
}

fn maybe_push_test(
    entries: &mut Vec<RunnableEntry>,
    name: &str,
    range: TextRange,
    class_name: Option<String>,
    is_unittest: bool,
) {
    if is_test_name(name) {
        entries.push(RunnableEntry {
            range,
            kind: RunnableKind::Test,
            test_name: Some(name.to_owned()),
            class_name,
            is_unittest,
        });
    }
}

fn is_test_name(name: &str) -> bool {
    name.starts_with("test_")
}

fn is_test_class(class_def: &StmtClassDef, is_unittest: bool) -> bool {
    class_def.name.as_str().starts_with("Test") || is_unittest
}

fn is_unittest_class(class_def: &StmtClassDef) -> bool {
    class_def.bases().iter().any(is_unittest_base)
}

fn is_unittest_base(base: &Expr) -> bool {
    match base {
        Expr::Name(name) => name.id.as_str().ends_with("TestCase"),
        Expr::Attribute(ExprAttribute { attr, .. }) => attr.id.as_str().ends_with("TestCase"),
        _ => false,
    }
}
