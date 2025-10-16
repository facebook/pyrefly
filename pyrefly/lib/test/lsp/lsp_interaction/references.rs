/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_references() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");
    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(2),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 10,
                    "character": 1
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6,"character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(3),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 6,
                    "character": 7
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(4),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 6,
                    "character": 17
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(4),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(5),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 7,
                    "character": 0
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(5),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":5,"character":23},"end":{"line":5,"character":24}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"line":7,"character":1}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(bar.clone()).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "text": format!("\n\n{}", std::fs::read_to_string(bar.clone()).unwrap())
                }],
            }),
        }));

    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(6),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 6,
                    "character": 17
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(6),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction
        .server
        .send_message(Message::Request(lsp_server::Request {
            id: RequestId::from(7),
            method: "textDocument/references".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 8,
                    "character": 7
                },
                "context": {
                    "includeDeclaration": true
                },
            }),
        }));

    interaction.client.expect_response(Response {
        id: RequestId::from(7),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":8,"character":6},"end":{"character":9,"line":8}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":12,"character":0},"end":{"character":3,"line":12}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}
