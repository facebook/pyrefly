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

fn init_with_workspace_edit_support(root_path: &std::path::Path) -> (LspInteraction, Url) {
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
                        "resourceOperations": ["rename", "delete"]
                    }
                }
            })),
            ..Default::default()
        })
        .unwrap();
    (interaction, scope_uri)
}

#[test]
fn test_convert_module_to_package_code_action() {
    let root = get_test_files_root();
    let root_path = root.path().join("convert_module_package");
    let (interaction, _scope_uri) = init_with_workspace_edit_support(&root_path);

    let file = "foo.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);

    let expected_old = Url::from_file_path(&file_path).unwrap();
    let expected_new = Url::from_file_path(root_path.join("foo/__init__.py")).unwrap();

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
                if code_action.title != "Convert module to package" {
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
                    DocumentChangeOperation::Op(ResourceOp::Rename(rename)) => {
                        rename.old_uri == expected_old && rename.new_uri == expected_new
                    }
                    _ => false,
                }
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_convert_package_to_module_code_action() {
    let root = get_test_files_root();
    let root_path = root.path().join("convert_module_package");
    let (interaction, _scope_uri) = init_with_workspace_edit_support(&root_path);

    let file = "empty_pkg/__init__.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);

    let expected_old = Url::from_file_path(&file_path).unwrap();
    let expected_new = Url::from_file_path(root_path.join("empty_pkg.py")).unwrap();
    let expected_delete = Url::from_file_path(root_path.join("empty_pkg")).unwrap();

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
                if code_action.title != "Convert package to module" {
                    return false;
                }
                let Some(edit) = &code_action.edit else {
                    return false;
                };
                let Some(DocumentChanges::Operations(ops)) = &edit.document_changes else {
                    return false;
                };
                if ops.len() != 2 {
                    return false;
                }
                let rename_ok = match &ops[0] {
                    DocumentChangeOperation::Op(ResourceOp::Rename(rename)) => {
                        rename.old_uri == expected_old && rename.new_uri == expected_new
                    }
                    _ => false,
                };
                let delete_ok = match &ops[1] {
                    DocumentChangeOperation::Op(ResourceOp::Delete(delete)) => {
                        delete.uri == expected_delete
                    }
                    _ => false,
                };
                rename_ok && delete_ok
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
