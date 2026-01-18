/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use lsp_types::ClientCapabilities;
use lsp_types::CodeAction;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::DeleteFile;
use lsp_types::DeleteFileOptions;
use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::RenameFile;
use lsp_types::ResourceOp;
use lsp_types::ResourceOperationKind;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;

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

fn package_dir_is_empty(dir: &Path, init_file: &OsStr) -> bool {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name == init_file || name == OsStr::new("__pycache__") {
            continue;
        }
        return false;
    }
    true
}

pub(crate) fn convert_module_package_code_actions(
    capabilities: &ClientCapabilities,
    uri: &Url,
) -> Vec<CodeActionOrCommand> {
    if !supports_workspace_edit_document_changes(capabilities) {
        return Vec::new();
    }
    let path = match uri.to_file_path() {
        Ok(path) => path,
        Err(_) => return Vec::new(),
    };
    let extension = match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext @ "py") | Some(ext @ "pyi") => ext,
        _ => return Vec::new(),
    };
    if !path.is_file() {
        return Vec::new();
    }
    let Some(file_stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return Vec::new();
    };
    let mut actions = Vec::new();
    if file_stem == "__init__" {
        if !supports_workspace_edit_resource_ops(
            capabilities,
            &[ResourceOperationKind::Rename, ResourceOperationKind::Delete],
        ) {
            return actions;
        }
        let Some(package_dir) = path.parent() else {
            return actions;
        };
        let Some(package_name) = package_dir.file_name().and_then(|name| name.to_str()) else {
            return actions;
        };
        let Some(parent_dir) = package_dir.parent() else {
            return actions;
        };
        let new_path = parent_dir.join(format!("{package_name}.{extension}"));
        if new_path.exists() {
            return actions;
        }
        let Some(init_name) = path.file_name() else {
            return actions;
        };
        if !package_dir_is_empty(package_dir, init_name) {
            return actions;
        }
        let old_uri = match Url::from_file_path(&path) {
            Ok(uri) => uri,
            Err(_) => return actions,
        };
        let new_uri = match Url::from_file_path(&new_path) {
            Ok(uri) => uri,
            Err(_) => return actions,
        };
        let package_uri = match Url::from_file_path(package_dir) {
            Ok(uri) => uri,
            Err(_) => return actions,
        };
        let operations = vec![
            DocumentChangeOperation::Op(ResourceOp::Rename(RenameFile {
                old_uri,
                new_uri,
                options: None,
                annotation_id: None,
            })),
            DocumentChangeOperation::Op(ResourceOp::Delete(DeleteFile {
                uri: package_uri,
                options: Some(DeleteFileOptions {
                    recursive: Some(true),
                    ignore_if_not_exists: Some(true),
                    annotation_id: None,
                }),
            })),
        ];
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Convert package to module".to_owned(),
            kind: Some(CodeActionKind::new("refactor.move")),
            edit: Some(WorkspaceEdit {
                document_changes: Some(DocumentChanges::Operations(operations)),
                ..Default::default()
            }),
            ..Default::default()
        }));
    } else {
        if !supports_workspace_edit_resource_ops(capabilities, &[ResourceOperationKind::Rename]) {
            return actions;
        }
        let Some(parent_dir) = path.parent() else {
            return actions;
        };
        let package_dir = parent_dir.join(file_stem);
        if package_dir.exists() {
            return actions;
        }
        let new_path = package_dir.join(format!("__init__.{extension}"));
        if new_path.exists() {
            return actions;
        }
        let old_uri = match Url::from_file_path(&path) {
            Ok(uri) => uri,
            Err(_) => return actions,
        };
        let new_uri = match Url::from_file_path(&new_path) {
            Ok(uri) => uri,
            Err(_) => return actions,
        };
        let operations = vec![DocumentChangeOperation::Op(ResourceOp::Rename(
            RenameFile {
                old_uri,
                new_uri,
                options: None,
                annotation_id: None,
            },
        ))];
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Convert module to package".to_owned(),
            kind: Some(CodeActionKind::new("refactor.move")),
            edit: Some(WorkspaceEdit {
                document_changes: Some(DocumentChanges::Operations(operations)),
                ..Default::default()
            }),
            ..Default::default()
        }));
    }
    actions
}
