/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeLens;
use lsp_types::Command;
use lsp_types::Location;
use lsp_types::Range;
use lsp_types::Uri;
use lsp_types::VersionedTextDocumentIdentifier;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::TextRange;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::state::state::Transaction;

/// A document batch for override lenses. Ranges use the specified editor version.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverrideCodeLensParams {
    pub text_document: VersionedTextDocumentIdentifier,
    pub ranges: Vec<Range>,
    /// The client handles `pyrefly/overrideCodeLens/refresh` notifications.
    #[serde(default)]
    pub refresh_support: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OverrideCodeLensTarget {
    #[serde(flatten)]
    pub location: Location,
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OverrideCodeLensTargets {
    pub parents: Vec<OverrideCodeLensTarget>,
    pub implementations: Vec<OverrideCodeLensTarget>,
}

/// Invalidates override lenses after the server updates its analysis.
pub enum OverrideCodeLensRefreshNotification {}

impl lsp_types::Notification for OverrideCodeLensRefreshNotification {
    type Params = Value;
    const METHOD: lsp_types::LspNotificationMethod<'static> =
        lsp_types::LspNotificationMethod::Custom("pyrefly/overrideCodeLens/refresh");
    const MESSAGE_DIRECTION: lsp_types::MessageDirection =
        lsp_types::MessageDirection::ServerToClient;
}

pub enum OverrideCodeLensRequest {}

impl lsp_types::Request for OverrideCodeLensRequest {
    type Params = OverrideCodeLensParams;
    type Result = Vec<OverrideCodeLensTargets>;
    const METHOD: lsp_types::LspRequestMethod<'static> =
        lsp_types::LspRequestMethod::Custom("pyrefly/textDocument/overrideCodeLens");
    const MESSAGE_DIRECTION: lsp_types::MessageDirection =
        lsp_types::MessageDirection::ClientToServer;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeLensKind {
    Run,
    Test,
}

#[derive(Clone, Debug)]
pub struct CodeLensEntry {
    pub range: TextRange,
    pub kind: CodeLensKind,
    pub test_name: Option<String>,
    pub class_name: Option<String>,
    pub is_unittest: bool,
}

pub fn runnable_lsp_code_lens(
    uri: &Uri,
    range: Range,
    entry: CodeLensEntry,
    cwd: Option<&str>,
) -> CodeLens {
    let (title, command, arguments) = match entry.kind {
        CodeLensKind::Run => {
            let mut args = serde_json::Map::new();
            args.insert("uri".to_owned(), serde_json::json!(uri.to_string()));
            if let Some(cwd) = cwd {
                args.insert("cwd".to_owned(), serde_json::json!(cwd));
            }
            ("Run", "pyrefly.runMain", Some(vec![Value::Object(args)]))
        }
        CodeLensKind::Test => {
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

impl<'a> Transaction<'a> {
    pub fn runnable_code_lens_entries(
        &self,
        handle: &Handle,
        uri: &Uri,
        runnable_code_lens: bool,
    ) -> Option<Vec<CodeLensEntry>> {
        if !runnable_code_lens || uri.path().ends_with(".pyi") || uri.path().ends_with(".ipynb") {
            return Some(Vec::new());
        }
        let ast = self.get_ast(handle)?;
        let mut entries = Vec::new();
        collect_module_entries(&ast.body, &mut entries);
        Some(entries)
    }
}

fn collect_module_entries(stmts: &[Stmt], entries: &mut Vec<CodeLensEntry>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(func) => {
                maybe_push_test(entries, func.name.as_str(), func.name.range, None, false);
            }
            Stmt::ClassDef(class_def) => {
                let is_unittest = is_unittest_class(class_def);
                if is_test_class(class_def, is_unittest) {
                    entries.push(CodeLensEntry {
                        range: class_def.name.range,
                        kind: CodeLensKind::Test,
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
                entries.push(CodeLensEntry {
                    range: stmt_if.range,
                    kind: CodeLensKind::Run,
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
    entries: &mut Vec<CodeLensEntry>,
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
    entries: &mut Vec<CodeLensEntry>,
    name: &str,
    range: TextRange,
    class_name: Option<String>,
    is_unittest: bool,
) {
    if is_test_name(name) {
        entries.push(CodeLensEntry {
            range,
            kind: CodeLensKind::Test,
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
    if class_def.name.as_str().starts_with("Test") {
        return true;
    }
    is_unittest
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
