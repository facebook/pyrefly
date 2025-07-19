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

use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::build_did_open_notification;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::run_test_lsp;

#[test]
fn test_hover_basic() {
    let root = get_test_files_root();
    let request_file_name = root.path().join("bar.py");
    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(
                root.path().join(request_file_name.clone()),
            )),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/hover".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root.path().join(request_file_name)).unwrap().to_string()
                    },
                    "position": {
                        "line": 7,
                        "character": 5
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!({
                "contents": {
                    "kind": "markdown",
                    "value": "```python\n(variable) foo: Literal[3]\n```",
                }
            })),
            error: None,
        })],
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
                "value": "```python\n(class) Bar: type[Bar]\n```\n---\nGo to [Bar](".to_owned()
                    + Url::from_file_path(root.path().join("bar.py")).unwrap().as_str()
                    + "#L7,7)",
            }})),
            error: None,
        })],
        ..Default::default()
    });
}
