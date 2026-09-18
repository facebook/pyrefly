/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;

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

/// Builds a safe-delete refactor action for a file that nothing imports.
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
    if transaction.is_depended_on_by_anything(&handle) {
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
