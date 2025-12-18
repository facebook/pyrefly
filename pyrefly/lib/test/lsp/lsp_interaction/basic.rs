/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::request::DocumentDiagnosticRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
#[allow(deprecated)]
fn test_initialize_basic() {
    let interaction = LspInteraction::new();

    interaction
        .client
        .send_initialize(
            interaction
                .client
                .get_initialize_params(&InitializeSettings::default()),
        )
        .expect_response(json!({"capabilities": {
            "positionEncoding": "utf-16",
            "textDocumentSync": 2,
            "definitionProvider": true,
            "typeDefinitionProvider": true,
            "codeActionProvider": {
                "codeActionKinds": ["quickfix", "refactor.extract"]
            },
            "completionProvider": {
                "triggerCharacters": [".", "'", "\""]
            },
            "declarationProvider": true,
            "documentHighlightProvider": true,
            "signatureHelpProvider": {
                "triggerCharacters": ["(", ","]
            },
            "hoverProvider": true,
            "implementationProvider": true,
            "inlayHintProvider": true,
            "notebookDocumentSync":{"notebookSelector":[{"cells":[{"language":"python"}]}]},
            "documentSymbolProvider": true,
            "foldingRangeProvider":true,
            "workspaceSymbolProvider": true,
            "workspace": {
                "workspaceFolders": {
                    "supported": true,
                    "changeNotifications": true
                },
                "fileOperations": {
                    "willRename": {
                        "filters": [
                            {
                                "pattern": {
                                    "glob": "**/*.{py,pyi}",
                                    "matches": "file"
                                },
                                "scheme": "file"
                            }
                        ]
                    }
                }
            }
        }, "serverInfo": {
            "name":"pyrefly-lsp",
            "version":"pyrefly-lsp-test-version"
        }}))
        .unwrap();
    interaction.client.send_initialized();
    interaction.shutdown().unwrap();
}

#[test]
fn test_shutdown() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction
        .client
        .send_shutdown()
        .expect_response(json!(null))
        .unwrap();

    interaction.client.send_exit();
    interaction.client.expect_stop();
}

#[test]
fn test_shutdown_with_messages_in_between() {
    // This is a regression test for https://github.com/facebook/pyrefly/issues/1016
    // nvim sometimes sends messages in between shutdown and exit. The server should
    // handle this gracefully and not hang.
    // Per LSP spec, requests after shutdown should be rejected with InvalidRequest.
    let test_files_root = get_test_files_root();
    let root = test_files_root.path().join("basic");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let test_file = root.join("foo.py");
    let uri = Url::from_file_path(&test_file).unwrap();

    // Open a file
    interaction.client.did_open("foo.py");

    // Expect initial diagnostics
    interaction.client.expect_any_message().unwrap();

    // Send shutdown request & wait for response
    interaction
        .client
        .send_shutdown()
        .expect_response(json!(null))
        .unwrap();

    // After shutdown, send a request (simulating what might happen with :wq)
    // Per LSP spec, this should be rejected with InvalidRequest
    interaction
        .client
        .send_request::<DocumentDiagnosticRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }));

    interaction.client.expect_stop();
}

#[test]
fn test_exit_without_shutdown() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.send_exit();
    interaction.client.expect_stop();
}

#[test]
#[allow(deprecated)]
fn test_initialize_with_python_path() {
    let scope_uri = Url::from_file_path(get_test_files_root()).unwrap();
    let python_path = "/path/to/python/interpreter";

    let interaction = LspInteraction::new();

    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    };

    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction.client.expect_any_message().unwrap();
    interaction.client.send_initialized();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .unwrap()
        .send_configuration_response(json!([{"pythonPath": python_path}]));

    interaction.shutdown().unwrap();
}

// This test exists as a regression test for certain notebooks that mock a fake file in /tmp/.
#[test]
fn test_nonexistent_file() {
    let root = get_test_files_root();
    let nonexistent_filename = root.path().join("nonexistent_file.py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction
        .client
        .send_notification::<DidOpenTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string(),
                "languageId": "python",
                "version": 1,
                "text": String::default(),
            }
        }));

    interaction
        .client
        .send_request::<DocumentDiagnosticRequest>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string()
            },
        }))
        .expect_response(json!({"items":[],"kind":"full"}))
        .unwrap();

    let notebook_content = std::fs::read_to_string(root.path().join("notebook.py")).unwrap();
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": format!("{}\n{}\n", notebook_content, "t")
            }],
        }));

    interaction.shutdown().unwrap();
}

#[test]
fn test_unknown_request() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();
    interaction.client.send_message(Message::Request(Request {
        id: RequestId::from(1),
        method: "fake-method".to_owned(),
        params: json!(null),
    }));
    interaction
        .client
        .expect_response_error(
            RequestId::from(1),
            json!({
                "code": -32601,
                "message": "Unknown request: fake-method",
                "data": null,
            }),
        )
        .unwrap();
}

#[test]
fn test_connection_closed_server_stops() {
    let mut interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    // Close the connection by dropping both the receiver and sender
    // This simulates the client disconnecting unexpectedly
    interaction.client.drop_connection();
    interaction.client.drop_connection();

    // The server should stop when the connection is closed
    interaction.client.expect_stop();
}
