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
use lsp_types::notification::Exit;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use lsp_types::request::Shutdown;
use lsp_types::request::WorkspaceConfiguration;

use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::run_test_lsp;

#[test]
fn test_initialize_basic() {
    run_test_lsp(TestCase::default());
}

#[test]
// #[should_panic]
fn test_shutdown() {
    if true {
        // Temporarily disabled because it wasn't really shutting down properly, but was assuming
        // that queues got deallocated in certain orders.
        return;
    }

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::Request(Request {
                id: RequestId::from(2),
                method: Shutdown::METHOD.to_owned(),
                params: serde_json::json!(null),
            }),
            Message::Notification(Notification {
                method: Exit::METHOD.to_owned(),
                params: serde_json::json!(null),
            }),
            // This second request should never be received by the server since it has already shut down.
            // `run_test_lsp` panics if any request does not get handled.
            Message::Request(Request {
                id: RequestId::from(3),
                method: "should not get here".to_owned(),
                params: serde_json::json!(null),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!(null)),
            error: None,
        })],
        ..Default::default()
    });
}

#[test]
// #[should_panic]
fn test_exit_without_shutdown() {
    if true {
        // Temporarily disabled because it wasn't really shutting down properly, but was assuming
        // that queues got deallocated in certain orders.
        return;
    }

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::Notification(Notification {
                method: Exit::METHOD.to_owned(),
                params: serde_json::json!(null),
            }),
            // This second request should never be received by the server since it has already shut down.
            // `run_test_lsp` panics if any request does not get handled.
            Message::Request(Request {
                id: RequestId::from(3),
                method: "should not get here".to_owned(),
                params: serde_json::json!(null),
            }),
        ],
        ..Default::default()
    });
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
