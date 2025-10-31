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
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_edits_while_recheck() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");
    let path = root.path().join("basic/foo.py");
    // In this test, we trigger didSave and didChange to try to exercise the behavior
    // where we have concurrent in-memory recheck and on-disk recheck.
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didSave".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 1,
                    "text": std::fs::read_to_string(path.clone()).unwrap()
                }
            }),
        }));

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [
                    {"text": format!("{}\n\nextra_stuff", std::fs::read_to_string(&path).unwrap())}
                ],
            }),
        }));

    interaction.server.definition("foo.py", 6, 18);

    interaction
        .client
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9);

    interaction.shutdown();
}

#[test]
fn test_file_watcher() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());

    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        file_watch: true,
        ..Default::default()
    });

    interaction.client.expect_request(Request {
        id: RequestId::from(1),
        method: "client/registerCapability".to_owned(),
        params: serde_json::json!({
            "registrations": [{
                "id": "FILEWATCHER",
                "method": "workspace/didChangeWatchedFiles",
                "registerOptions": {
                    "watchers": [
                        {"globPattern": root.path().join("**/*.py").to_str().unwrap(), "kind": 7},
                        {"globPattern": root.path().join("**/*.pyi").to_str().unwrap(), "kind": 7},
                        {"globPattern": root.path().join("**/*.ipynb").to_str().unwrap(), "kind": 7},
                        {"globPattern": root.path().join("**/pyrefly.toml"), "kind": 7},
                        {"globPattern": root.path().join("**/pyproject.toml"), "kind": 7}
                    ]
                }
            }]
        }),
    });

    interaction.server.send_message(Message::Response(Response {
        id: RequestId::from(1),
        result: None,
        error: None,
    }));

    interaction.shutdown();
}
