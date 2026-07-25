/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_server::ErrorCode;
use lsp_server::RequestId;
use lsp_types::CodeLens;
use lsp_types::CodeLensParams;
use lsp_types::Command;
use lsp_types::DocumentSymbol;
use lsp_types::Location;
use lsp_types::Range;
use lsp_types::SymbolKind;
use lsp_types::Url;
use lsp_types::request::CodeLensRequest;
use lsp_types::request::Request as _;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::telemetry::ActivityKey;
use pyrefly_util::telemetry::EmptyResponseReason;
use pyrefly_util::telemetry::TelemetryEventKind;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::TextRange;
use serde_json::Value;
use starlark_map::small_set::SmallSet;
use tracing::info;

use super::Server;
use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::lsp::new_response;
use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Response;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodeLensKind {
    Run,
    Test,
}

#[derive(Clone, Debug)]
struct CodeLensEntry {
    range: TextRange,
    kind: CodeLensKind,
    test_name: Option<String>,
    class_name: Option<String>,
    is_unittest: bool,
}

#[derive(Debug)]
struct CodeLensTarget {
    range: Range,
    definition: FindDefinitionItemWithDocstring,
}

impl Server {
    /// Build runnable lenses and resolve reference counts on the references queue.
    pub(super) fn code_lens<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        params: CodeLensParams,
        activity_key: Option<ActivityKey>,
    ) -> Result<(), EmptyResponseReason> {
        let uri = &params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            self.send_response(new_response(request_id, Ok(Some(Vec::<CodeLens>::new()))));
            return Ok(());
        }

        let handle = self.make_handle_if_enabled(uri, Some(CodeLensRequest::METHOD))?;
        let runnable_lenses = self.runnable_code_lenses(transaction, &handle, &params)?;
        let targets = if self.indexing_mode == IndexingMode::None {
            Vec::new()
        } else {
            self.reference_code_lens_targets(transaction, &handle, uri)?
        };
        if targets.is_empty() {
            self.send_response(new_response(request_id, Ok(Some(runnable_lenses))));
            return Ok(());
        }

        let path_remapper = self.path_remapper.clone();
        let source_uri = uri.clone();
        self.find_reference_queue.queue_task(
            TelemetryEventKind::FindFromDefinition,
            Box::new(move |server, _telemetry, telemetry_event| {
                telemetry_event.set_activity_key(activity_key);
                let mut transaction = server.state.cancellable_transaction();
                server
                    .cancellation_handles
                    .lock()
                    .insert(request_id.clone(), transaction.get_cancellation_handle());
                server.validate_in_memory_for_transaction(
                    transaction.as_mut(),
                    telemetry_event,
                    None,
                );

                let reference_lenses = match resolve_reference_code_lenses(
                    &mut transaction,
                    &handle,
                    &source_uri,
                    targets,
                    path_remapper.as_ref(),
                ) {
                    Ok(lenses) => lenses,
                    Err(Cancelled) => {
                        let message = format!("Request {request_id} is canceled");
                        info!("{message}");
                        server.connection.send(Message::Response(Response::new_err(
                            request_id,
                            ErrorCode::RequestCanceled as i32,
                            message,
                        )));
                        return;
                    }
                };

                let mut lenses = runnable_lenses;
                lenses.extend(reference_lenses);
                server.cancellation_handles.lock().remove(&request_id);
                server.connection.send(Message::Response(new_response(
                    request_id,
                    Ok(Some(lenses)),
                )));
            }),
        );
        Ok(())
    }

    fn reference_code_lens_targets(
        &self,
        transaction: &Transaction<'_>,
        handle: &Handle,
        uri: &Url,
    ) -> Result<Vec<CodeLensTarget>, EmptyResponseReason> {
        let module_info = transaction
            .get_module_info(handle)
            .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
        let symbols = transaction
            .symbols(handle, None)
            .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
        let mut symbol_defs = Vec::new();
        collect_reference_symbols(&symbols, &mut symbol_defs);

        let mut seen = SmallSet::new();
        let mut targets = Vec::new();
        for symbol in symbol_defs {
            let position = self.from_lsp_position(uri, &module_info, symbol.selection_range.start);
            let definition = match transaction.find_definition(
                handle,
                position,
                FindPreference {
                    import_behavior: ImportBehavior::StopAtRenamedImports,
                    ..Default::default()
                },
            ) {
                Ok(definitions) => definitions.into_vec().swap_remove(0),
                Err(_) => continue,
            };
            let key = (definition.module.path().dupe(), definition.definition_range);
            if seen.insert(key) {
                targets.push(CodeLensTarget {
                    range: symbol.selection_range,
                    definition,
                });
            }
        }
        Ok(targets)
    }

    fn runnable_code_lenses(
        &self,
        transaction: &Transaction<'_>,
        handle: &Handle,
        params: &CodeLensParams,
    ) -> Result<Vec<CodeLens>, EmptyResponseReason> {
        let uri = &params.text_document.uri;
        let path = self
            .path_for_uri(uri)
            .ok_or(EmptyResponseReason::NoFilePath)?;
        let runnable_code_lens = self
            .workspaces
            .get_with(path.clone(), |(_, workspace)| workspace.runnable_code_lens);
        let maybe_cell_idx = self.maybe_get_code_cell_index(uri);
        let info = transaction
            .get_module_info(handle)
            .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
        let entries = transaction
            .runnable_code_lens_entries(handle, uri, runnable_code_lens)
            .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
        let config = self.state.config_finder().python_file(
            ModuleNameWithKind::guaranteed(ModuleName::unknown()),
            &ModulePath::filesystem(path.clone()),
        );
        let cwd = config
            .source
            .root_from_file()
            .map(std::path::Path::to_path_buf)
            .or_else(|| {
                self.workspaces
                    .get_with(path, |(workspace_root, _)| workspace_root.cloned())
            })
            .map(|path| path.to_string_lossy().into_owned());

        let mut lenses = Vec::new();
        for entry in entries {
            if info.to_cell_for_lsp(entry.range.start()) != maybe_cell_idx {
                continue;
            }
            let range = info.to_lsp_range(entry.range);
            lenses.push(runnable_lsp_code_lens(uri, range, entry, cwd.as_deref()));
        }
        Ok(lenses)
    }
}

fn collect_reference_symbols<'a>(symbols: &'a [DocumentSymbol], out: &mut Vec<&'a DocumentSymbol>) {
    for symbol in symbols {
        if matches!(
            symbol.kind,
            SymbolKind::CLASS | SymbolKind::FUNCTION | SymbolKind::METHOD
        ) {
            out.push(symbol);
        }
        if let Some(children) = symbol.children.as_deref() {
            collect_reference_symbols(children, out);
        }
    }
}

fn resolve_reference_code_lenses(
    transaction: &mut CancellableTransaction<'_>,
    handle: &Handle,
    source_uri: &Url,
    targets: Vec<CodeLensTarget>,
    path_remapper: Option<&PathRemapper>,
) -> Result<Vec<CodeLens>, Cancelled> {
    let mut lenses = Vec::with_capacity(targets.len());
    for target in targets {
        let local_results = transaction.find_global_references_from_definition(
            *handle.sys_info(),
            target.definition.metadata,
            TextRangeWithModule::new(
                target.definition.module.clone(),
                target.definition.definition_range,
            ),
            false,
        )?;

        let mut locations = Vec::new();
        for (info, ranges) in local_results {
            if let Some(uri) = module_info_to_uri(&info, path_remapper) {
                for range in ranges {
                    locations.push(Location {
                        uri: uri.clone(),
                        range: info.to_lsp_range(range),
                    });
                }
            }
        }

        let reference_count = locations.len();
        let title = if reference_count == 1 {
            "1 reference".to_owned()
        } else {
            format!("{reference_count} references")
        };
        lenses.push(CodeLens {
            range: target.range,
            command: Some(Command {
                title,
                command: "editor.action.showReferences".to_owned(),
                arguments: Some(vec![
                    serde_json::to_value(source_uri).expect("URI should serialize for code lens"),
                    serde_json::to_value(target.range.start)
                        .expect("Position should serialize for code lens"),
                    serde_json::to_value(&locations)
                        .expect("Locations should serialize for code lens"),
                ]),
                tooltip: None,
            }),
            data: None,
        });
    }
    Ok(lenses)
}

fn runnable_lsp_code_lens(
    uri: &Url,
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
    fn runnable_code_lens_entries(
        &self,
        handle: &Handle,
        uri: &Url,
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
