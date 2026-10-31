/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use lsp_types::ClientCapabilities;
use lsp_types::CodeAction;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionResponse;
use lsp_types::Command;
use lsp_types::DeleteFile;
use lsp_types::DeleteFileOptions;
use lsp_types::DocumentChange;
use lsp_types::Location;
use lsp_types::ResourceOperationKind;
use lsp_types::Uri;
use lsp_types::WorkspaceEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde_json::json;

use crate::lsp::non_wasm::module_helpers::handle_from_module_path;
use crate::state::lsp::FindPreference;
use crate::state::state::State;
use crate::state::state::Transaction;

fn supports_workspace_edit_document_changes(capabilities: &ClientCapabilities) -> bool {
    capabilities
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.workspace_edit.as_ref())
        .and_then(|workspace_edit| workspace_edit.document_changes)
        .unwrap_or(false)
}

fn supports_workspace_edit_resource_ops(
    capabilities: &ClientCapabilities,
    required: &[ResourceOperationKind],
) -> bool {
    let supported = capabilities
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.workspace_edit.as_ref())
        .and_then(|workspace_edit| workspace_edit.resource_operations.as_ref());
    required
        .iter()
        .all(|kind| supported.is_some_and(|ops| ops.contains(kind)))
}

/// Builds safe-delete refactor actions for a file.
pub(crate) fn safe_delete_file_code_actions(
    capabilities: &ClientCapabilities,
    supports_find_usages: bool,
    state: &State,
    transaction: &Transaction<'_>,
    uri: &Uri,
    get_open_handle: impl Fn(&Path) -> Option<Handle>,
    to_lsp_location: impl Fn(&TextRangeWithModule) -> Option<Location>,
) -> Option<Vec<CodeActionResponse>> {
    if !supports_workspace_edit_document_changes(capabilities) {
        return None;
    }
    if !supports_workspace_edit_resource_ops(capabilities, &[ResourceOperationKind::Delete]) {
        return None;
    }
    let path = uri.to_file_path().ok()?;
    if !path.is_file() {
        return None;
    }
    if !PYTHON_EXTENSIONS
        .iter()
        .any(|ext| path.extension().and_then(|e| e.to_str()) == Some(*ext))
    {
        return None;
    }
    let file_name = path.file_name()?.to_string_lossy().to_string();
    let handle = handle_from_module_path(state, ModulePath::filesystem(path.clone()));
    let module_name = handle.module();
    if module_name == ModuleName::unknown() {
        return None;
    }
    let rdeps = transaction.get_rdeps(&handle);
    if rdeps.is_empty() {
        return Some(vec![delete_file_code_action(
            format!("Safe delete file `{file_name}`"),
            uri,
        )]);
    }
    let mut actions = Vec::new();
    if supports_find_usages {
        let usages = collect_import_usages(transaction, rdeps, module_name, get_open_handle);
        let locations: Vec<_> = usages.iter().filter_map(to_lsp_location).collect();
        if !locations.is_empty() {
            actions.push(find_usages_code_action(&file_name, uri, locations));
        }
    }
    actions.push(delete_file_code_action(
        format!("Delete file `{file_name}` anyway"),
        uri,
    ));
    Some(actions)
}

fn delete_file_code_action(title: String, uri: &Uri) -> CodeActionResponse {
    let operation = DocumentChange::DeleteFile(DeleteFile {
        uri: uri.clone(),
        options: Some(DeleteFileOptions {
            recursive: Some(false),
            ignore_if_not_exists: Some(true),
        }),
        annotation_id: None,
    });
    CodeActionResponse::CodeAction(CodeAction {
        title,
        kind: Some(CodeActionKind::Custom("refactor.delete".into())),
        edit: Some(WorkspaceEdit {
            document_changes: Some(vec![operation]),
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn find_usages_code_action(
    file_name: &str,
    uri: &Uri,
    locations: Vec<Location>,
) -> CodeActionResponse {
    let title = format!("Find usages of file `{file_name}`");
    CodeActionResponse::CodeAction(CodeAction {
        title: title.clone(),
        kind: Some(CodeActionKind::Custom("refactor.delete".into())),
        command: Some(Command {
            title,
            tooltip: None,
            command: "pyrefly.findFileUsages".to_owned(),
            arguments: Some(vec![json!({
                "uri": uri,
                "locations": locations,
            })]),
        }),
        ..Default::default()
    })
}

fn collect_import_usages(
    transaction: &Transaction<'_>,
    rdeps: HashSet<Handle>,
    target_module: ModuleName,
    get_open_handle: impl Fn(&Path) -> Option<Handle>,
) -> Vec<TextRangeWithModule> {
    let mut seen = HashSet::new();
    let mut usages = Vec::new();
    for rdep_handle in rdeps {
        // An open buffer can remove its dependency while the indexed disk file retains it.
        let rdep_handle = get_open_handle(rdep_handle.path().as_path()).unwrap_or(rdep_handle);
        if !seen.insert(rdep_handle.path().as_path().to_owned()) {
            continue;
        }
        let Some(module_info) = transaction.get_module_info(&rdep_handle) else {
            continue;
        };
        if !matches!(
            module_info.path().details(),
            ModulePathDetails::FileSystem(_) | ModulePathDetails::Memory(_)
        ) {
            continue;
        }
        // Only indexed closed files without a retained AST need parsing.
        let ast = transaction.get_ast(&rdep_handle).unwrap_or_else(|| {
            Arc::new(Ast::parse(module_info.contents(), module_info.source_type()).0)
        });
        let mut visitor = ImportUsageVisitor {
            transaction,
            importer: &rdep_handle,
            target_module,
            current_module: module_info.name(),
            is_init: module_info.path().is_init(),
            ranges: Vec::new(),
        };
        for stmt in &ast.body {
            visitor.visit_stmt(stmt);
        }
        if visitor.ranges.is_empty() {
            continue;
        }
        for range in visitor.ranges {
            usages.push(TextRangeWithModule::new(module_info.clone(), range));
        }
    }
    usages
}

struct ImportUsageVisitor<'a> {
    transaction: &'a Transaction<'a>,
    importer: &'a Handle,
    target_module: ModuleName,
    current_module: ModuleName,
    is_init: bool,
    ranges: Vec<TextRange>,
}

impl ImportUsageVisitor<'_> {
    fn record_import(&mut self, imported_module: ModuleName, range: TextRange) -> bool {
        if module_matches_target(imported_module, self.target_module) {
            self.ranges.push(range);
            return true;
        }
        false
    }

    fn visit_import(&mut self, import: &StmtImport) {
        for alias in &import.names {
            let imported_module = ModuleName::from_name(&alias.name.id);
            if self.record_import(imported_module, import.range()) {
                return;
            }
        }
    }

    fn visit_import_from(&mut self, import_from: &StmtImportFrom) {
        let base = self.current_module.new_maybe_relative(
            self.is_init,
            import_from.level,
            import_from.module.as_ref().map(|id| &id.id),
        );
        let Some(base) = base else {
            return;
        };
        if self.record_import(base, import_from.range()) {
            return;
        }
        for alias in &import_from.names {
            let refers_to_target = self
                .transaction
                .resolve_named_import(
                    self.importer,
                    base,
                    alias.name.id.clone(),
                    FindPreference::default(),
                )
                .is_some_and(|(handle, _, _)| {
                    module_matches_target(handle.module(), self.target_module)
                });
            if refers_to_target {
                self.ranges.push(import_from.range());
                return;
            }
        }
    }
}

impl Visitor<'_> for ImportUsageVisitor<'_> {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Import(import) => self.visit_import(import),
            Stmt::ImportFrom(import_from) => self.visit_import_from(import_from),
            _ => walk_stmt(self, stmt),
        }
    }
}

fn module_matches_target(module: ModuleName, target: ModuleName) -> bool {
    if module == target {
        return true;
    }
    let module_str = module.as_str();
    let target_str = target.as_str();
    module_str.starts_with(target_str)
        && module_str.len() > target_str.len()
        && module_str.as_bytes().get(target_str.len()) == Some(&b'.')
}

#[cfg(test)]
mod tests {
    use std::fs;

    use lsp_types::ClientCapabilities;
    use lsp_types::CodeActionResponse;
    use lsp_types::Uri;
    use serde_json::json;
    use tempfile::tempdir;

    use super::safe_delete_file_code_actions;
    use crate::test::util::TestEnv;

    #[test]
    fn test_safe_delete_file_skips_locations_for_unsupported_clients() {
        let root = tempdir().unwrap();
        let target_path = root.path().join("target.py");
        let consumer_path = root.path().join("consumer.py");
        fs::write(&target_path, "value = 1\n").unwrap();
        fs::write(&consumer_path, "import target\n").unwrap();
        let mut env = TestEnv::new();
        env.add_real_path("target", target_path.clone());
        env.add_real_path("consumer", consumer_path);
        let (state, handle) = env.to_state();
        let transaction = state.transaction();
        assert!(!transaction.get_rdeps(&handle("target")).is_empty());
        let capabilities: ClientCapabilities = serde_json::from_value(json!({
            "workspace": {
                "workspaceEdit": {
                    "documentChanges": true,
                    "resourceOperations": ["delete"]
                }
            }
        }))
        .unwrap();
        let actions = safe_delete_file_code_actions(
            &capabilities,
            false,
            &state,
            &transaction,
            &Uri::from_file_path(target_path).unwrap(),
            |_| panic!("unsupported clients must not inspect importer buffers"),
            |_| panic!("unsupported clients must not convert locations"),
        )
        .unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], CodeActionResponse::CodeAction(action)
            if action.title == "Delete file `target.py` anyway"));
    }
}
