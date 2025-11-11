/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
#[allow(deprecated)]
fn test_initialize_basic() {
    let mut interaction = LspInteraction::new();

    interaction.server.send_initialize(
        interaction
            .server
            .get_initialize_params(&InitializeSettings::default()),
    );
    interaction
        .client
        .expect_message(Message::Response(Response {
            id: RequestId::from(1),
            result: Some(serde_json::json!({"capabilities": {
                "positionEncoding": "utf-16",
                "textDocumentSync": 2,
                "definitionProvider": true,
                "typeDefinitionProvider": true,
                "codeActionProvider": {
                    "codeActionKinds": ["quickfix"]
                },
                "completionProvider": {
                    "triggerCharacters": ["."]
                },
                "documentHighlightProvider": true,
                "signatureHelpProvider": {
                    "triggerCharacters": ["(", ","]
                },
                "hoverProvider": true,
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
            }})),
            error: None,
        }));
    interaction.server.send_initialized();
    interaction.shutdown();
}

#[test]
fn test_shutdown() {
    let mut interaction = LspInteraction::new();
    interaction.initialize(InitializeSettings::default());

    interaction.server.send_shutdown(RequestId::from(2));

    interaction
        .client
        .expect_message(Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!(null)),
            error: None,
        }));

    interaction.server.send_exit();
    interaction.server.expect_stop();
}

#[test]
fn test_exit_without_shutdown() {
    let mut interaction = LspInteraction::new();
    interaction.initialize(InitializeSettings::default());

    interaction.server.send_exit();
    interaction.server.expect_stop();
}

#[test]
#[allow(deprecated)]
fn test_initialize_with_python_path() {
    let scope_uri = Url::from_file_path(get_test_files_root()).unwrap();
    let python_path = "/path/to/python/interpreter";

    let mut interaction = LspInteraction::new();

    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    };

    interaction
        .server
        .send_initialize(interaction.server.get_initialize_params(&settings));
    interaction.client.expect_any_message();
    interaction.server.send_initialized();

    interaction
        .client
        .expect_configuration_request(1, Some(vec![&scope_uri]));
    interaction.server.send_configuration_response(
        1,
        serde_json::json!([{"pythonPath": python_path}, {"pythonPath": python_path}]),
    );

    interaction.shutdown();
}

// This test exists as a regression test for certain notebooks that mock a fake file in /tmp/.
#[test]
fn test_nonexistent_file() {
    let root = get_test_files_root();
    let nonexistent_filename = root.path().join("nonexistent_file.py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didOpen".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string(),
                    "languageId": "python",
                    "version": 1,
                    "text": String::default(),
                }
            }),
        }));

    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/diagnostic".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string()
            },
        }),
    }));

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items":[],"kind":"full"})),
        error: None,
    });

    let notebook_content = std::fs::read_to_string(root.path().join("notebook.py")).unwrap();
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&nonexistent_filename).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "text": format!("{}\n{}\n", notebook_content, "t")
                }],
            }),
        }));

    interaction.shutdown();
}

#[test]
fn test_unknown_request() {
    let mut interaction = LspInteraction::new();
    interaction.initialize(InitializeSettings::default());
    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(1),
        method: "fake-method".to_owned(),
        params: serde_json::json!(null),
    }));
    interaction
        .client
        .expect_message(Message::Response(Response {
            id: RequestId::from(1),
            result: None,
            error: Some(ResponseError {
                code: -32601,
                message: "Unknown request: fake-method".to_owned(),
                data: None,
            }),
        }));
}
