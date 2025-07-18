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

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::build_did_open_notification;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::run_test_lsp;

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
                        "range": {
                            "start": {"line": 10, "character": 0},
                            "end": {"line": 12, "character": 0}
                        },
                        "text": format!("\n{}\n", "Ba")
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
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string(),
                        "languageId": "python",
                        "version": 2
                    },
                    "contentChanges": [{
                        "text": format!("{}\n{}", std::fs::read_to_string(root.path().join("foo.py")).unwrap(), "Sequenc")
                    }],
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
fn test_completion_with_autoimport() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(root_path.join("foo.py"))),
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root_path.join("foo.py")).unwrap().to_string(),
                        "languageId": "python",
                        "version": 2
                    },
                    "contentChanges": [{
                        "text": format!("{}\n{}", std::fs::read_to_string(root_path.join("foo.py")).unwrap(), "this_is_a_very_long_function_name_so_we_can")
                    }],
                }),
            }),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root_path.join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 11,
                        "character": 43
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({
                "isIncomplete":false,
                "items":[
                    {"detail":"type[Bar]","kind":6,"label":"Bar","sortText":"0"},
                    {
                        "additionalTextEdits":[{
                            "newText":"from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n",
                            "range":{"end":{"character":0,"line":5},"start":{"character":0,"line":5}}
                        }],
                        "detail":"from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n",
                        "kind":3,
                        "label":"this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search",
                        "sortText":"3"
                    },
                ]
            })),
            error: None,
        })],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });
}

#[test]
fn test_module_completion() {
    let root = get_test_files_root();
    let foo = root.path().join("tests_requiring_config").join("foo.py");

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(foo.clone())),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(foo).unwrap().to_string()
                    },
                    "position": {
                        "line": 5,
                        "character": 10
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(
                serde_json::json!({"isIncomplete":false,"items":[{"detail":"bar","kind":9,"label":"bar","sortText":"0"}]}),
            ),
            error: None,
        })],
        ..Default::default()
    });
}

// TODO: Handle relative import (via ModuleName::new_maybe_relative)
#[test]
fn test_relative_module_completion() {
    let root = get_test_files_root();
    let foo = root.path().join("relative_test").join("relative_import.py");

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(foo.clone())),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(foo).unwrap().to_string()
                    },
                    "position": {
                        "line": 5,
                        "character": 10
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({"isIncomplete":false,"items":[]})),
            error: None,
        })],
        ..Default::default()
    });
}

#[test]
fn test_empty_filepath_file_completion() {
    let root = get_test_files_root();
    let empty_filename = root.path().join("empty_file.py");

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(Notification {
                method: "textDocument/didOpen".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(&empty_filename).unwrap().to_string(),
                        "languageId": "python",
                        "version": 1,
                        "text": String::default(),
                    }
                }),
            }),
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(&empty_filename).unwrap().to_string(),
                        "languageId": "python",
                        "version": 2
                    },
                    "contentChanges": [{
                        "text": format!("{}\n{}\n", std::fs::read_to_string(root.path().join("notebook.py")).unwrap(), "t")
                    }],
                }),
            }),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(&empty_filename).unwrap().to_string()
                    },
                    "position": {
                        "line": 9,
                        "character": 1
                    }
                }),
            }),
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(&empty_filename).unwrap().to_string(),
                        "languageId": "python",
                        "version": 3
                    },
                    "contentChanges": [{
                        "text": format!("{}\n{}", std::fs::read_to_string(root.path().join("notebook.py")).unwrap(), "t")
                    }],
                }),
            }),
            Message::from(Request {
                id: RequestId::from(3),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(&empty_filename).unwrap().to_string()
                    },
                    "position": {
                        "line": 10,
                        "character": 1
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(
                    serde_json::json!({"isIncomplete":false,"items":[{"detail":"(a: int, b: int, c: str) -> int","kind":3,"label":"tear","sortText":"0"}]}),
                ),
                error: None,
            }),
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(
                    serde_json::json!({"isIncomplete":false,"items":[{"detail":"(a: int, b: int, c: str) -> int","kind":3,"label":"tear","sortText":"0"}]}),
                ),
                error: None,
            }),
        ],
        ..Default::default()
    });
}
