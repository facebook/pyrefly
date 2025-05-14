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
use lsp_types::ConfigurationItem;
use lsp_types::ConfigurationParams;
use lsp_types::Url;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use lsp_types::request::WorkspaceConfiguration;
use tempfile::TempDir;

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction_util::TestCase;
use crate::test::lsp::lsp_interaction_util::build_did_open_notification;
use crate::test::lsp::lsp_interaction_util::get_test_files_root;
use crate::test::lsp::lsp_interaction_util::run_test_lsp;

#[test]
fn test_initialize_basic() {
    run_test_lsp(TestCase::default());
}

#[test]
fn test_initialize_with_python_path() {
    let scope_uri = Url::from_file_path(get_test_files_root()).unwrap();
    let python_path = "/path/to/python/interpreter";
    let id = RequestId::from(1);
    run_test_lsp(TestCase {
        messages_from_language_client: vec![Message::Response(Response {
            id: id.clone(),
            result: Some(
                serde_json::json!([{"pythonPath": python_path}, {"pythonPath": python_path}]),
            ),
            error: None,
        })],
        expected_messages_from_language_server: vec![Message::Request(Request {
            id,
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: serde_json::json!(ConfigurationParams {
                items: Vec::from([
                    ConfigurationItem {
                        scope_uri: Some(scope_uri.clone()),
                        section: Some("python".to_owned()),
                    },
                    ConfigurationItem {
                        scope_uri: None,
                        section: Some("python".to_owned()),
                    }
                ]),
            }),
        })],
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

fn test_go_to_def(root: &TempDir, workspace_folders: Option<Vec<(String, Url)>>) {
    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(root.path().join("foo.py"))),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/definition".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 6,
                        "character": 16
                    }
                }),
            }),
            Message::from(Request {
                id: RequestId::from(3),
                method: "textDocument/definition".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 8,
                        "character": 9
                    }
                }),
            }),
            Message::from(Request {
                id: RequestId::from(4),
                method: "textDocument/definition".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 9,
                        "character": 7
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!({
                    "uri": Url::from_file_path(root.path().join("bar.py")).unwrap().to_string(),
                    "range": {
                        "start": {
                            "line": 6,
                            "character": 6
                        },
                        "end": {
                            "line": 6,
                            "character": 9
                        }
                    }
                })),
                error: None,
            }),
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "uri": Url::from_file_path(root.path().join("bar.py")).unwrap().to_string(),
                    "range": {
                        "start": {
                            "line": 7,
                            "character": 4
                        },
                        "end": {
                            "line": 7,
                            "character": 7
                        }
                    }
                })),
                error: None,
            }),
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "uri": Url::from_file_path(root.path().join("bar.py")).unwrap().to_string(),
                    "range": {
                        "start": {
                            "line": 6,
                            "character": 6
                        },
                        "end": {
                            "line": 6,
                            "character": 9
                        }
                    }
                })),
                error: None,
            }),
        ],
        workspace_folders,
        ..Default::default()
    });
}

#[test]
fn test_go_to_def_single_root() {
    let root = get_test_files_root();
    test_go_to_def(
        &root,
        Some(vec![(
            "test".to_owned(),
            Url::from_file_path(root.path()).unwrap(),
        )]),
    );
}

#[test]
fn test_go_to_def_no_root() {
    let root = get_test_files_root();
    test_go_to_def(&root, Some(vec![]));
}

#[test]
fn test_go_to_def_no_root_uses_upwards_search() {
    let root = get_test_files_root();
    test_go_to_def(&root, Some(vec![]));
}

#[test]
fn test_go_to_def_no_folder_capability() {
    let root = get_test_files_root();
    test_go_to_def(&root, None);
}

#[test]
fn definition_in_builtins_enabled() {
    let root = get_test_files_root();
    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(
                root.path().join("imports_builtins.py"),
            )),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/definition".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("imports_builtins.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 7,
                        "character": 7
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({
                "range":{"end":{"character":4,"line":378},"start":{"character":0,"line":378}},"uri":format!("contentsasuri://$$MATCH_EVERYTHING$$")})),
            error: None,
        })],
        contents_as_uri: true,
        ..Default::default()
    });
}

#[test]
fn definition_in_builtins_disabled() {
    let root = get_test_files_root();
    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(
                root.path().join("imports_builtins.py"),
            )),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/definition".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("imports_builtins.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 7,
                        "character": 7
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!([])),
            error: None,
        })],
        contents_as_uri: false,
        ..Default::default()
    });
}

#[test]
fn test_hover() {
    let root = get_test_files_root();

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(root.path().join("foo.py"))),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/hover".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 6,
                        "character": 16
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({"contents": {
                "kind": "markdown",
                "value": "```python\ntype[Bar]\n```",
            }})),
            error: None,
        })],
        ..Default::default()
    });
}

#[test]
fn test_completion() {
    let root = get_test_files_root();

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(root.path().join("foo.py"))),
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string(),
                        "languageId": "python",
                        "version": 2
                    },
                    "contentChanges": [{
                        "text": format!("{}\n{}", std::fs::read_to_string(root.path().join("foo.py")).unwrap(), "Ba")
                    }],
                }),
            }),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 11,
                        "character": 1
                    }
                }),
            }),
            Message::from(Request {
                id: RequestId::from(3),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 11,
                        "character": 2
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(
                    serde_json::json!({"isIncomplete":false,"items":[{"detail":"type[Bar]","kind":6,"label":"Bar","sortText":"0"}]}),
                ),
                error: None,
            }),
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(
                    serde_json::json!({"isIncomplete":false,"items":[{"detail":"type[Bar]","kind":6,"label":"Bar","sortText":"0"}]}),
                ),
                error: None,
            }),
        ],
        ..Default::default()
    });
}

#[test]
fn test_references() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut test_messages = Vec::new();
    let mut expected_responses = Vec::new();
    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");
    test_messages.push(Message::from(build_did_open_notification(bar.clone())));

    // Find reference from a reference location in the same in-memory file
    test_messages.push(Message::from(Request {
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

    expected_responses.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
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
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    }));

    // Find reference from a definition location in the same in-memory file
    test_messages.push(Message::from(Request {
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

    expected_responses.push(Message::Response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([
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
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    }));

    test_messages.push(Message::from(build_did_open_notification(foo.clone())));

    // Find reference from a reference location in a different file
    test_messages.push(Message::from(Request {
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

    expected_responses.push(Message::Response(Response {
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
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
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
        ])),
        error: None,
    }));

    // Change the definition file in memory.
    // However, find ref still reports the stale result based on the filesystem content.
    test_messages.push(Message::from(Notification {
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
    test_messages.push(Message::from(Request {
        id: RequestId::from(5),
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
    expected_responses.push(Message::Response(Response {
        id: RequestId::from(5),
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
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
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
        ])),
        error: None,
    }));

    // When we do a find-ref in an in-memory file with changed content,
    // it will cause us to fail to find references in other files.
    test_messages.push(Message::from(Request {
        id: RequestId::from(6),
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

    expected_responses.push(Message::Response(Response {
        id: RequestId::from(6),
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
    }));

    run_test_lsp(TestCase {
        messages_from_language_client: test_messages,
        expected_messages_from_language_server: expected_responses,
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });
}

#[test]
fn test_did_change_configuration() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut messages_from_language_client = Vec::new();
    messages_from_language_client.push(Message::Notification(Notification {
        method: DidChangeConfiguration::METHOD.to_owned(),
        params: serde_json::json!({"settings": {}}),
    }));
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(1),
        result: Some(serde_json::json!([{}])),
        error: None,
    }));
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([{}])),
        error: None,
    }));
    let params = serde_json::json!(ConfigurationParams {
        items: Vec::from([
            ConfigurationItem {
                scope_uri: Some(scope_uri.clone()),
                section: Some("python".to_owned()),
            },
            ConfigurationItem {
                scope_uri: None,
                section: Some("python".to_owned()),
            }
        ]),
    });
    let expected_messages_from_language_server = vec![
        Message::Request(Request {
            id: RequestId::from(1),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: params.clone(),
        }),
        Message::Request(Request {
            id: RequestId::from(2),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params,
        }),
    ];
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_disable_language_services() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let file_path = test_files_root.path().join("foo.py");
    let mut messages_from_language_client = Vec::new();
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(1),
        result: Some(serde_json::json!([{}])),
        error: None,
    }));
    messages_from_language_client.push(Message::from(build_did_open_notification(
        file_path.clone(),
    )));
    let go_to_definition_params = serde_json::json!({
        "textDocument": {
            "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
        },
        "position": {
            "line": 6,
            "character": 16
        }
    });
    messages_from_language_client.push(Message::from(Request {
        id: RequestId::from(2),
        method: "textDocument/definition".to_owned(),
        params: go_to_definition_params.clone(),
    }));
    messages_from_language_client.push(Message::Notification(Notification {
        method: DidChangeConfiguration::METHOD.to_owned(),
        params: serde_json::json!({"settings": {}}),
    }));
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([{"pyrefly": {"disableLanguageServices": true}}, {"pyrefly": {"disableLanguageServices": true}}])),
        error: None,
    }));
    messages_from_language_client.push(Message::from(Request {
        id: RequestId::from(3),
        method: "textDocument/definition".to_owned(),
        params: go_to_definition_params.clone(),
    }));
    let mut expected_messages_from_language_server = Vec::new();
    let configuration_params = serde_json::json!(ConfigurationParams {
        items: Vec::from([
            ConfigurationItem {
                scope_uri: Some(scope_uri.clone()),
                section: Some("python".to_owned()),
            },
            ConfigurationItem {
                scope_uri: None,
                section: Some("python".to_owned()),
            }
        ]),
    });
    expected_messages_from_language_server.push(Message::Request(Request {
        id: RequestId::from(1),
        method: WorkspaceConfiguration::METHOD.to_owned(),
        params: configuration_params.clone(),
    }));
    expected_messages_from_language_server.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(test_files_root.path().join("bar.py")).unwrap().to_string(),
            "range": {
                "start": {
                    "line": 6,
                    "character": 6
                },
                "end": {
                    "line": 6,
                    "character": 9
                }
            }
        })),
        error: None,
    }));
    expected_messages_from_language_server.push(Message::Request(Request {
        id: RequestId::from(2),
        method: WorkspaceConfiguration::METHOD.to_owned(),
        params: configuration_params.clone(),
    }));
    expected_messages_from_language_server.push(Message::Response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([])),
        error: None,
    }));
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_disable_language_services_default_workspace() {
    let test_files_root = get_test_files_root();
    let file_path = test_files_root.path().join("foo.py");
    let mut messages_from_language_client = Vec::new();
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(1),
        result: Some(serde_json::json!([{}])),
        error: None,
    }));
    messages_from_language_client.push(Message::from(build_did_open_notification(
        file_path.clone(),
    )));
    let go_to_definition_params = serde_json::json!({
        "textDocument": {
            "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
        },
        "position": {
            "line": 6,
            "character": 16
        }
    });
    messages_from_language_client.push(Message::from(Request {
        id: RequestId::from(2),
        method: "textDocument/definition".to_owned(),
        params: go_to_definition_params.clone(),
    }));
    messages_from_language_client.push(Message::Notification(Notification {
        method: DidChangeConfiguration::METHOD.to_owned(),
        params: serde_json::json!({"settings": {}}),
    }));
    messages_from_language_client.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([{"pyrefly": {"disableLanguageServices": true}}, {"pyrefly": {"disableLanguageServices": true}}])),
        error: None,
    }));
    messages_from_language_client.push(Message::from(Request {
        id: RequestId::from(3),
        method: "textDocument/definition".to_owned(),
        params: go_to_definition_params.clone(),
    }));
    let mut expected_messages_from_language_server = Vec::new();
    let configuration_params = serde_json::json!(ConfigurationParams {
        items: Vec::from([ConfigurationItem {
            scope_uri: None,
            section: Some("python".to_owned()),
        }]),
    });
    expected_messages_from_language_server.push(Message::Request(Request {
        id: RequestId::from(1),
        method: WorkspaceConfiguration::METHOD.to_owned(),
        params: configuration_params.clone(),
    }));
    expected_messages_from_language_server.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(test_files_root.path().join("bar.py")).unwrap().to_string(),
            "range": {
                "start": {
                    "line": 6,
                    "character": 6
                },
                "end": {
                    "line": 6,
                    "character": 9
                }
            }
        })),
        error: None,
    }));
    expected_messages_from_language_server.push(Message::Request(Request {
        id: RequestId::from(2),
        method: WorkspaceConfiguration::METHOD.to_owned(),
        params: configuration_params.clone(),
    }));
    expected_messages_from_language_server.push(Message::Response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([])),
        error: None,
    }));
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_edits_while_recheck() {
    let mut test_messages = Vec::new();
    let mut expected_responses = Vec::new();
    let root = get_test_files_root();

    let path = root.path().join("foo.py");
    test_messages.push(Message::from(build_did_open_notification(path.clone())));
    // In this test, we trigger didSave and didChange to try to exercise the behavior
    // where we have concurrent in-memory recheck and on-disk recheck.
    test_messages.push(Message::from(Notification {
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
    test_messages.push(Message::from(Notification {
        method: "textDocument/didChange".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [
                {"text": format!("{}\n\nextra_stuff", std::fs::read_to_string(path).unwrap())}
            ],
        }),
    }));

    test_messages.push(Message::from(Request {
        id: RequestId::from(2),
        method: "textDocument/definition".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
            },
            "position": {
                "line": 6,
                "character": 18
            }
        }),
    }));

    expected_responses.push(Message::Response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(root.path().join("bar.py")).unwrap().to_string(),
            "range": {
                "start": {
                    "line": 6,
                    "character": 6
                },
                "end": {
                    "line": 6,
                    "character": 9
                }
            }
        })),
        error: None,
    }));

    run_test_lsp(TestCase {
        messages_from_language_client: test_messages,
        expected_messages_from_language_server: expected_responses,
        ..Default::default()
    });
}

#[test]
fn test_file_watcher() {
    let root = get_test_files_root();
    run_test_lsp(TestCase {
        messages_from_language_client: vec![Message::Response(Response {
            id: RequestId::from(1),
            result: None,
            error: None,
        })],
        expected_messages_from_language_server: vec![Message::Request(Request {
            id: RequestId::from(1),
            method: "client/registerCapability".to_owned(),
            params: serde_json::json!({
            "registrations": [{"id": "FILEWATCHER", "method": "workspace/didChangeWatchedFiles", "registerOptions": {"watchers": [
                {"globPattern": root.path().join("**/*.py").to_str().unwrap(), "kind": 7},
                {"globPattern": root.path().join("**/*.pyi").to_str().unwrap(), "kind": 7},
                {"globPattern": root.path().join("**/pyrefly.toml"), "kind": 7},
                {"globPattern": root.path().join("**/pyproject.toml"), "kind": 7}
            ]}}]}),
        })],
        workspace_folders: Some(vec![(
            "test".to_owned(),
            Url::from_file_path(root).unwrap(),
        )]),
        file_watch: true,
        ..Default::default()
    });
}

#[test]
fn test_did_change_workspace_folder() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!([{}])),
                error: None,
            }),
            Message::Notification(Notification {
                method: DidChangeWorkspaceFolders::METHOD.to_owned(),
                params: serde_json::json!({
                    "event": {
                    "added": [{"uri": Url::from_file_path(&root).unwrap(), "name": "test"}],
                    "removed": [],
                    }
                }),
            }),
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!([{}, {}])),
                error: None,
            }),
        ],
        expected_messages_from_language_server: vec![
            Message::Request(Request {
                id: RequestId::from(1),
                method: WorkspaceConfiguration::METHOD.to_owned(),
                params: serde_json::json!(ConfigurationParams {
                    items: Vec::from([ConfigurationItem {
                        scope_uri: None,
                        section: Some("python".to_owned()),
                    }]),
                }),
            }),
            Message::Request(Request {
                id: RequestId::from(2),
                method: WorkspaceConfiguration::METHOD.to_owned(),
                params: serde_json::json!(ConfigurationParams {
                    items: Vec::from([
                        ConfigurationItem {
                            scope_uri: Some(scope_uri.clone()),
                            section: Some("python".to_owned()),
                        },
                        ConfigurationItem {
                            scope_uri: None,
                            section: Some("python".to_owned()),
                        }
                    ]),
                }),
            }),
        ],
        configuration: true,
        ..Default::default()
    });
}

fn get_diagnostics_result() -> serde_json::Value {
    serde_json::json!({"items": [
            {"code":"bad-argument-type","message":"`+` is not supported between `Literal[1]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter with type `int` in function `int.__add__`",
            "range":{"end":{"character":6,"line":5},"start":{"character":0,"line":5}},"severity":1,"source":"Pyrefly"}],"kind":"full"
    })
}

#[test]
fn test_diagnostics_default_workspace() {
    let root = get_test_files_root();
    let file_path = root.path().join("type_errors.py");
    let messages_from_language_client = vec![
        Message::from(build_did_open_notification(file_path.clone())),
        Message::from(Request {
            id: RequestId::from(1),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
    ];

    let expected_messages_from_language_server = vec![Message::Response(Response {
        id: RequestId::from(1),
        result: Some(get_diagnostics_result()),
        error: None,
    })];

    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        ..Default::default()
    });
}

#[test]
fn test_diagnostics_in_workspace() {
    let root = get_test_files_root();
    let file_path = root.path().join("type_errors.py");
    let messages_from_language_client = vec![
        Message::from(build_did_open_notification(file_path.clone())),
        Message::from(Request {
            id: RequestId::from(1),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
    ];

    let expected_messages_from_language_server = vec![Message::Response(Response {
        id: RequestId::from(1),
        result: Some(get_diagnostics_result()),
        error: None,
    })];

    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        workspace_folders: Some(vec![(
            "test".to_owned(),
            Url::from_file_path(root).unwrap(),
        )]),
        ..Default::default()
    });
}

#[test]
fn test_disable_type_errors_language_services_still_work() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let file_path = test_files_root.path().join("foo.py");
    let messages_from_language_client = vec![
        Message::Response(Response {
            id: RequestId::from(1),
            result: Some(
                serde_json::json!([{"pyrefly": {"disableTypeErrors": true}}, {"pyrefly": {"disableTypeErrors": true}}]),
            ),
            error: None,
        }),
        Message::from(build_did_open_notification(file_path.clone())),
        Message::from(Request {
            id: RequestId::from(2),
            method: "textDocument/hover".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
                },
                "position": {
                    "line": 6,
                    "character": 17
                }
            }),
        }),
    ];
    let expected_messages_from_language_server = vec![
        Message::Request(Request {
            id: RequestId::from(1),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: serde_json::json!(ConfigurationParams {
                items: Vec::from([
                    ConfigurationItem {
                        scope_uri: Some(scope_uri.clone()),
                        section: Some("python".to_owned()),
                    },
                    ConfigurationItem {
                        scope_uri: None,
                        section: Some("python".to_owned()),
                    }
                ]),
            }),
        }),
        Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({
                "contents": {"kind":"markdown","value":"```python\ntype[Bar]\n```"}
            })),
            error: None,
        }),
    ];
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_disable_type_errors_workspace_folder() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let file_path = test_files_root.path().join("type_errors.py");
    let configuration_request_params = serde_json::json!(ConfigurationParams {
        items: Vec::from([
            ConfigurationItem {
                scope_uri: Some(scope_uri.clone()),
                section: Some("python".to_owned()),
            },
            ConfigurationItem {
                scope_uri: None,
                section: Some("python".to_owned()),
            }
        ]),
    });

    let messages_from_language_client = vec![
        Message::from(build_did_open_notification(file_path.clone())),
        Message::Response(Response {
            id: RequestId::from(1),
            result: Some(serde_json::json!([])),
            error: None,
        }),
        Message::from(Request {
            id: RequestId::from(2),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
        Message::Notification(Notification {
            method: DidChangeConfiguration::METHOD.to_owned(),
            params: serde_json::json!({"settings": {}}),
        }),
        Message::Response(Response {
            id: RequestId::from(2),
            result: Some(
                serde_json::json!([{"pyrefly": {"disableTypeErrors": true}}, {"pyrefly": {"disableTypeErrors": true}}]),
            ),
            error: None,
        }),
        Message::from(Request {
            id: RequestId::from(3),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
    ];
    let expected_messages_from_language_server = vec![
        Message::Request(Request {
            id: RequestId::from(1),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: configuration_request_params.clone(),
        }),
        Message::Response(Response {
            id: RequestId::from(2),
            result: Some(get_diagnostics_result()),
            error: None,
        }),
        Message::Request(Request {
            id: RequestId::from(2),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: configuration_request_params,
        }),
        Message::Response(Response {
            id: RequestId::from(3),
            result: Some(serde_json::json!({"items": [], "kind": "full"})),
            error: None,
        }),
    ];
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_disable_type_errors_default_workspace() {
    let test_files_root = get_test_files_root();
    let file_path = test_files_root.path().join("type_errors.py");
    let messages_from_language_client = vec![
        Message::Response(Response {
            id: RequestId::from(1),
            result: Some(serde_json::json!([])),
            error: None,
        }),
        Message::from(build_did_open_notification(file_path.clone())),
        Message::from(Request {
            id: RequestId::from(2),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
        Message::Notification(Notification {
            method: DidChangeConfiguration::METHOD.to_owned(),
            params: serde_json::json!({"settings": {}}),
        }),
        Message::Response(Response {
            id: RequestId::from(2),
            result: Some(
                serde_json::json!([{"pyrefly": {"disableTypeErrors": true}}, {"pyrefly": {"disableTypeErrors": true}}]),
            ),
            error: None,
        }),
        Message::from(Request {
            id: RequestId::from(3),
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(file_path.clone()).unwrap().to_string()
            }}),
        }),
    ];
    let expected_messages_from_language_server = vec![
        Message::Request(Request {
            id: RequestId::from(1),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: serde_json::json!(ConfigurationParams {
                items: Vec::from([ConfigurationItem {
                    scope_uri: None,
                    section: Some("python".to_owned()),
                }]),
            }),
        }),
        Message::Response(Response {
            id: RequestId::from(2),
            result: Some(get_diagnostics_result()),
            error: None,
        }),
        Message::Request(Request {
            id: RequestId::from(2),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: serde_json::json!(ConfigurationParams {
                items: Vec::from([ConfigurationItem {
                    scope_uri: None,
                    section: Some("python".to_owned()),
                }]),
            }),
        }),
        Message::Response(Response {
            id: RequestId::from(3),
            result: Some(serde_json::json!({"items": [], "kind": "full"})),
            error: None,
        }),
    ];
    run_test_lsp(TestCase {
        messages_from_language_client,
        expected_messages_from_language_server,
        configuration: true,
        ..Default::default()
    });
}
