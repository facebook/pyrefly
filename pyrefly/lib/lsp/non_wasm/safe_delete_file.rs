/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;

use lsp_types::ClientCapabilities;
use lsp_types::CodeAction;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::DeleteFile;
use lsp_types::DeleteFileOptions;
use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::ResourceOp;
use lsp_types::ResourceOperationKind;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_stmt;

use crate::lsp::non_wasm::module_helpers::handle_from_module_path;
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

/// Builds a safe-delete refactor action for a file if no import usages are found.
pub(crate) fn safe_delete_file_code_action(
    capabilities: &ClientCapabilities,
    state: &State,
    transaction: &Transaction<'_>,
    uri: &Url,
) -> Option<CodeActionOrCommand> {
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
    if has_import_usages(transaction, &handle, module_name, &path) {
        return None;
    }
    let operation = DocumentChangeOperation::Op(ResourceOp::Delete(DeleteFile {
        uri: uri.clone(),
        options: Some(DeleteFileOptions {
            recursive: Some(false),
            ignore_if_not_exists: Some(true),
            annotation_id: None,
        }),
    }));
    Some(CodeActionOrCommand::CodeAction(CodeAction {
        title: format!("Safe delete file `{file_name}`"),
        kind: Some(CodeActionKind::new("refactor.delete")),
        edit: Some(WorkspaceEdit {
            document_changes: Some(DocumentChanges::Operations(vec![operation])),
            ..Default::default()
        }),
        ..Default::default()
    }))
}

fn has_import_usages(
    transaction: &Transaction<'_>,
    handle: &Handle,
    target_module: ModuleName,
    target_path: &std::path::Path,
) -> bool {
    let rdeps = transaction.get_transitive_rdeps(handle.clone());
    let mut seen = HashSet::new();
    for rdep_handle in rdeps {
        if !seen.insert(rdep_handle.path().as_path().to_owned()) {
            continue;
        }
        let Some(module_info) = transaction.get_module_info(&rdep_handle) else {
            continue;
        };
        if module_info.path().as_path() == target_path {
            continue;
        }
        if !matches!(
            module_info.path().details(),
            ModulePathDetails::FileSystem(_) | ModulePathDetails::Memory(_)
        ) {
            continue;
        }
        let ast = Ast::parse(module_info.contents(), module_info.source_type()).0;
        let mut visitor = ImportUsageVisitor {
            target_module,
            current_module: module_info.name(),
            is_init: module_info.path().is_init(),
            found: false,
        };
        for stmt in &ast.body {
            visitor.visit_stmt(stmt);
            if visitor.found {
                return true;
            }
        }
    }
    false
}

struct ImportUsageVisitor {
    target_module: ModuleName,
    current_module: ModuleName,
    is_init: bool,
    found: bool,
}

impl ImportUsageVisitor {
    fn record_import(&mut self, imported_module: ModuleName) {
        if module_matches_target(imported_module, self.target_module) {
            self.found = true;
        }
    }

    fn visit_import(&mut self, import: &StmtImport) {
        for alias in &import.names {
            let imported_module = ModuleName::from_name(&alias.name.id);
            self.record_import(imported_module);
            if self.found {
                return;
            }
        }
    }

    fn visit_import_from(&mut self, import_from: &StmtImportFrom) {
        if self.found {
            return;
        }
        let base = self.current_module.new_maybe_relative(
            self.is_init,
            import_from.level,
            import_from.module.as_ref().map(|id| &id.id),
        );
        let Some(base) = base else {
            return;
        };
        self.record_import(base);
        if self.found {
            return;
        }
        for alias in &import_from.names {
            let imported_module = append_module(base, &alias.name.id);
            self.record_import(imported_module);
            if self.found {
                return;
            }
        }
    }
}

impl Visitor<'_> for ImportUsageVisitor {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.found {
            return;
        }
        match stmt {
            Stmt::Import(import) => self.visit_import(import),
            Stmt::ImportFrom(import_from) => self.visit_import_from(import_from),
            _ => walk_stmt(self, stmt),
        }
    }
}

fn append_module(base: ModuleName, suffix: &Name) -> ModuleName {
    if base.as_str().is_empty() {
        ModuleName::from_name(suffix)
    } else {
        base.append(suffix)
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
