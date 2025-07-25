/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::iter::once;

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::build_did_open_notification;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::run_test_lsp;

fn test_go_to_def(
    root: &TempDir,
    workspace_folders: Option<Vec<(String, Url)>>,
    // request file name, relative to root
    request_file_name: &str,
    // (line, character, response_file_name (relative to root), response_line_start, response_character_start, response_line_end, response_character_end)
    requests: Vec<(u32, u32, String, u32, u32, u32, u32)>,
) {
    run_test_lsp(TestCase {
        messages_from_language_client: once(Message::from(build_did_open_notification(
                root.path().join(request_file_name),
        ))).chain(
            requests.iter().enumerate().map(
                |(i, (request_line, request_character, _response_file_name, _response_line_start, _response_character_start, _response_line_end, _response_character_end))| {
                Message::from(Request {
                    id: RequestId::from((2 + i) as i32),
                    method: "textDocument/definition".to_owned(),
                    params: serde_json::json!({
                        "textDocument": {
                            "uri": Url::from_file_path(root.path().join(request_file_name)).unwrap().to_string()
                        },
                        "position": {
                            "line": request_line,
                            "character": request_character
                        }
                    }),
                })
            })).collect(),
        expected_messages_from_language_server: requests.iter().enumerate().map(
            |(
                i,
                (
                    _request_line,
                    _request_character,
                    response_file_name,
                    response_line_start,
                    response_character_start,
                    response_line_end,
                    response_character_end,
                ),
            )| {
                Message::Response(Response {
                    id: RequestId::from((2 + i) as i32),
                    result: Some(serde_json::json!({
                        "uri": Url::from_file_path(root.path().join(response_file_name)).unwrap().to_string(),
                        "range": {
                            "start": {
                                "line": response_line_start,
                                "character": response_character_start
                            },
                            "end": {
                                "line": response_line_end,
                                "character": response_character_end
                            }
                        }
                    })),
                    error: None,
                })
            },
        ).collect(),
        workspace_folders,
        ..Default::default()
    });
}

fn test_go_to_def_basic(root: &TempDir, workspace_folders: Option<Vec<(String, Url)>>) {
    test_go_to_def(
        root,
        workspace_folders,
        "foo.py",
        vec![
            (5, 7, "bar.py".to_owned(), 0, 0, 0, 0),
            (6, 16, "bar.py".to_owned(), 6, 6, 6, 9),
            (8, 9, "bar.py".to_owned(), 7, 4, 7, 7),
            (9, 7, "bar.py".to_owned(), 6, 6, 6, 9),
        ],
    );
}

#[test]
fn test_go_to_def_single_root() {
    let root = get_test_files_root();
    test_go_to_def_basic(
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
    test_go_to_def_basic(&root, Some(vec![]));
}

#[test]
fn test_go_to_def_no_root_uses_upwards_search() {
    let root = get_test_files_root();
    test_go_to_def_basic(&root, Some(vec![]));
}

#[test]
fn test_go_to_def_no_folder_capability() {
    let root = get_test_files_root();
    test_go_to_def_basic(&root, None);
}

#[test]
fn test_go_to_def_relative_path() {
    test_go_to_def(
        &get_test_files_root(),
        None,
        "foo_relative.py",
        vec![
            (5, 14, "bar.py".to_owned(), 0, 0, 0, 0),
            (6, 17, "bar.py".to_owned(), 6, 6, 6, 9),
            (8, 9, "bar.py".to_owned(), 7, 4, 7, 7),
            (9, 7, "bar.py".to_owned(), 6, 6, 6, 9),
        ],
    );
}

#[test]
fn definition_in_builtins() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = env::temp_dir().join("pyrefly_bundled_typeshed");
    let result_file = pyrefly_typeshed_materialized.join("typing.pyi");
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
                "range":{"end":{"character":4,"line":425},"start":{"character":0,"line":425}},"uri":Url::from_file_path(result_file.as_path()).unwrap().to_string()})),
            error: None,
        })],
        ..Default::default()
    });
    assert!(
        pyrefly_typeshed_materialized.join("pyrefly.toml").exists(),
        "Expected pyrefly.toml to exist at {:?}",
        pyrefly_typeshed_materialized.to_str()
    );
    assert!(
        result_file.exists(),
        "Expected pyrefly.toml to exist at {result_file:?}",
    );
}
