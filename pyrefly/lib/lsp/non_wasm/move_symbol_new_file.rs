/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use dupe::Dupe;
use lsp_types::ClientCapabilities;
use lsp_types::CodeAction;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::CreateFile;
use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::OneOf;
use lsp_types::OptionalVersionedTextDocumentIdentifier;
use lsp_types::Position;
use lsp_types::Range;
use lsp_types::ResourceOp;
use lsp_types::ResourceOperationKind;
use lsp_types::TextDocumentEdit;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::absolutize::Absolutize as _;
use ruff_text_size::TextRange;

use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
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

fn path_to_uri(path: &std::path::Path, remapper: Option<&PathRemapper>) -> Option<Url> {
    let final_path = remapper
        .map(|remap| remap(path).into_owned())
        .unwrap_or_else(|| path.to_path_buf());
    let abs_path = final_path.absolutize();
    Url::from_file_path(abs_path).ok()
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
    if new_path == path || new_path.exists() {
        return None;
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

    let mut edits = transaction.module_member_source_move_edits(
        handle,
        &context,
        &target_handle,
        import_format,
    )?;
    edits.extend(transaction.module_member_consumer_import_updates(
        handle,
        &context.module_info,
        &context.member_name,
        &target_handle,
        import_format,
    ));

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
            edits: vec![OneOf::Left(TextEdit {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 0),
                },
                new_text: context.member_text,
            })],
        }),
    ];

    let mut sorted_changes: Vec<(Url, Vec<TextEdit>)> = changes.into_iter().collect();
    sorted_changes.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    for (uri, mut text_edits) in sorted_changes {
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
            edits: text_edits.into_iter().map(OneOf::Left).collect(),
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
