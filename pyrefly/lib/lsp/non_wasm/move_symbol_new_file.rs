/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use dupe::Dupe;
use itertools::Itertools;
use lsp_types::ClientCapabilities;
use lsp_types::CodeAction;
use lsp_types::CodeActionDisabled;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::CreateFile;
use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::OptionalVersionedTextDocumentIdentifier;
use lsp_types::Position;
use lsp_types::Range;
use lsp_types::ResourceOp;
use lsp_types::ResourceOperationKind;
use lsp_types::TextDocumentEdit;
use lsp_types::TextEdit;
use lsp_types::TextEditOrAnnotatedOrSnippet;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use ruff_text_size::TextRange;

use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::lsp::non_wasm::module_helpers::path_to_uri;
use crate::state::lsp::ImportFormat;
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

fn supports_code_action_disabled(capabilities: &ClientCapabilities) -> bool {
    capabilities
        .text_document
        .as_ref()
        .and_then(|text_document| text_document.code_action.as_ref())
        .and_then(|code_action| code_action.disabled_support)
        .unwrap_or(false)
}

pub(crate) fn move_symbol_to_new_file_code_action(
    capabilities: &ClientCapabilities,
    transaction: &Transaction<'_>,
    handle: &Handle,
    uri: &Url,
    selection: TextRange,
    import_format: ImportFormat,
    path_remapper: Option<&PathRemapper>,
) -> Option<CodeActionOrCommand> {
    if !supports_workspace_edit_document_changes(capabilities) {
        return None;
    }
    if !supports_workspace_edit_resource_ops(capabilities, &[ResourceOperationKind::Create]) {
        return None;
    }

    let path = uri.to_file_path().ok()?;
    let extension = path.extension().and_then(|ext| ext.to_str())?;
    if !PYTHON_EXTENSIONS.contains(&extension) {
        return None;
    }

    let context = transaction.module_member_move_context(handle, selection)?;
    let new_path = path
        .parent()?
        .join(format!("{}.{}", context.member_name, extension));
    if new_path == path {
        return None;
    }
    if new_path.exists() {
        // The target file already exists, so we cannot create it. Rather than silently
        // dropping the action, surface it as disabled with a reason so the user
        // understands why it is unavailable (only when the client advertises
        // `disabledSupport`; otherwise omit it to avoid showing a no-op action).
        if !supports_code_action_disabled(capabilities) {
            return None;
        }
        return Some(CodeActionOrCommand::CodeAction(CodeAction {
            title: format!("Move `{}` to new file", context.member_name),
            kind: Some(CodeActionKind::new("refactor.move")),
            disabled: Some(CodeActionDisabled {
                reason: format!(
                    "Cannot move: {}.{} already exists",
                    context.member_name, extension
                ),
            }),
            ..Default::default()
        }));
    }

    let config = transaction
        .config_finder()
        .python_file(handle.module_kind(), context.module_info.path());
    let new_module_name = ModuleName::from_path(
        &new_path,
        config.search_path().chain(
            config
                .fallback_search_path
                .for_directory(new_path.parent())
                .iter(),
        ),
        &config.extra_file_extensions,
    )?;
    let target_handle = Handle::new(
        new_module_name,
        ModulePath::filesystem(new_path.clone()),
        handle.sys_info().dupe(),
    );

    let edits =
        transaction.module_member_move_edits(handle, &context, &target_handle, import_format)?;

    let new_uri = path_to_uri(&new_path, path_remapper)?;
    let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();
    for (module, range, new_text) in edits {
        let Some(edit_uri) = module_info_to_uri(&module, path_remapper) else {
            continue;
        };
        changes.entry(edit_uri).or_default().push(TextEdit {
            range: module.to_lsp_range(range),
            new_text,
        });
    }

    let mut operations = vec![
        DocumentChangeOperation::Op(ResourceOp::Create(CreateFile {
            uri: new_uri.clone(),
            options: None,
            annotation_id: None,
        })),
        DocumentChangeOperation::Edit(TextDocumentEdit {
            text_document: OptionalVersionedTextDocumentIdentifier {
                uri: new_uri,
                version: None,
            },
            edits: vec![TextEditOrAnnotatedOrSnippet::TextEdit(TextEdit {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 0),
                },
                new_text: context.member_text,
            })],
        }),
    ];

    for (uri, mut text_edits) in changes
        .into_iter()
        .sorted_by(|a, b| a.0.as_str().cmp(b.0.as_str()))
    {
        text_edits.sort_by(|a, b| {
            (
                a.range.start.line,
                a.range.start.character,
                a.range.end.line,
                a.range.end.character,
            )
                .cmp(&(
                    b.range.start.line,
                    b.range.start.character,
                    b.range.end.line,
                    b.range.end.character,
                ))
        });
        operations.push(DocumentChangeOperation::Edit(TextDocumentEdit {
            text_document: OptionalVersionedTextDocumentIdentifier { uri, version: None },
            edits: text_edits
                .into_iter()
                .map(TextEditOrAnnotatedOrSnippet::TextEdit)
                .collect(),
        }));
    }

    Some(CodeActionOrCommand::CodeAction(CodeAction {
        title: format!("Move `{}` to new file", context.member_name),
        kind: Some(CodeActionKind::new("refactor.move")),
        edit: Some(WorkspaceEdit {
            document_changes: Some(DocumentChanges::Operations(operations)),
            ..Default::default()
        }),
        ..Default::default()
    }))
}
