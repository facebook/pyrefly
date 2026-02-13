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
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn init_with_delete_support(root_path: &std::path::Path) -> (LspInteraction, Url) {
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
                        "resourceOperations": ["delete"]
                    }
                }
            })),
            ..Default::default()
        })
        .unwrap();
    (interaction, scope_uri)
}

#[test]
fn test_safe_delete_file_unused() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path);

    let file = "unused.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                if code_action.title != "Safe delete file `unused.py`" {
                    return false;
                }
                let Some(edit) = &code_action.edit else {
                    return false;
                };
                let Some(DocumentChanges::Operations(ops)) = &edit.document_changes else {
                    return false;
                };
                if ops.len() != 1 {
                    return false;
                }
                match &ops[0] {
                    DocumentChangeOperation::Op(ResourceOp::Delete(delete)) => delete.uri == uri,
                    _ => false,
                }
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_rejects_usages() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path);

    let file = "target.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);
    interaction.client.did_open("consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return true;
            };
            actions.iter().all(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return true;
                };
                code_action.title != "Safe delete file `target.py`"
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
