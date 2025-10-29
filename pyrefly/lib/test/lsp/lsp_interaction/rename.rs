/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_prepare_rename() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");

    let path = root.path().join("basic/foo.py");
    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/prepareRename".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": 6,
                "character": 16
            }
        }),
    }));

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "start": {"line": 6, "character": 16},
            "end": {"line": 6, "character": 19},
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_rename() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(Some(
            serde_json::json!([{ "indexing_mode": "lazy_blocking" }]),
        )),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("bar.py");
    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");

    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/rename".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&bar).unwrap().to_string()
            },
            "position": {
                "line": 10,
                "character": 1
            },
            "newName": "Baz"
        }),
    }));

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "changes": {
                Url::from_file_path(&foo).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":16},"end":{"line":6,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":8,"character":0},"end":{"line":8,"character":3}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":9,"character":4},"end":{"line":9,"character":7}}
                    },
                ],
                Url::from_file_path(&various_imports).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"line":5,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":26},"end":{"line":5,"character":29}}
                    },
                ],
                Url::from_file_path(&with_synthetic_bindings).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"character":19,"line":5}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":4},"end":{"character":7,"line":10}}
                    },
                ],
                Url::from_file_path(&bar).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":6},"end":{"character":9,"line":6}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":0},"end":{"character":3,"line":10}}
                    },
                ]
            }
        })),
        error: None,
    });

    interaction.shutdown();
}
