/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeActionOrCommand;
use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::ResourceOp;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

fn init_with_create_support(root_path: &std::path::Path) -> (LspInteraction, Url) {
    let scope_uri = Url::from_file_path(root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            capabilities: Some(json!({
                "workspace": {
                    "workspaceEdit": {
                        "documentChanges": true,
                        "resourceOperations": ["create"]
                    }
                }
            })),
            ..Default::default()
        })
        .unwrap();
    (interaction, scope_uri)
}

fn has_edit(ops: &[DocumentChangeOperation], uri: &Url, expected_text: &str) -> bool {
    ops.iter().any(|op| {
        let DocumentChangeOperation::Edit(edit) = op else {
            return false;
        };
        edit.text_document.uri == *uri
            && edit.edits.iter().any(|edit| match edit {
                lsp_types::OneOf::Left(TextEdit { new_text, .. }) => new_text == expected_text,
                lsp_types::OneOf::Right(_) => false,
            })
    })
}

#[test]
fn test_move_symbol_to_new_file_code_action() {
    let root = get_test_files_root();
    let root_path = root.path().join("move_symbol_to_new_file");
    let (interaction, _scope_uri) = init_with_create_support(&root_path);

    let source_path = root_path.join("source.py");
    let source_uri = Url::from_file_path(&source_path).unwrap();
    let consumer_uri = Url::from_file_path(root_path.join("consumer.py")).unwrap();
    let new_uri = Url::from_file_path(root_path.join("test.py")).unwrap();

    interaction.client.did_open("source.py");
    interaction.client.did_open("consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": source_uri },
            "range": {
                "start": { "line": 0, "character": 4 },
                "end": { "line": 0, "character": 4 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionOrCommand>>| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                if code_action.title != "Move `test` to new file" {
                    return false;
                }
                let Some(edit) = &code_action.edit else {
                    return false;
                };
                let Some(DocumentChanges::Operations(ops)) = &edit.document_changes else {
                    return false;
                };
                if ops.len() != 4 {
                    return false;
                }
                let has_create = ops.iter().any(|op| match op {
                    DocumentChangeOperation::Op(ResourceOp::Create(create)) => {
                        create.uri == new_uri
                    }
                    _ => false,
                });
                has_create
                    && has_edit(ops, &new_uri, "def test(x, y):\n    return x + y\n")
                    && has_edit(ops, &source_uri, "from test import test\n")
                    && has_edit(ops, &consumer_uri, "from test import test\n")
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
