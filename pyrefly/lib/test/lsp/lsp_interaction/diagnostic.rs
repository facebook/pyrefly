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
use crate::test::lsp::lsp_interaction::object_model::ValidationResult;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_cycle_class() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("cycle_class/foo.py");
    interaction.server.diagnostic("cycle_class/foo.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_unexpected_keyword_range() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.server.did_open("unexpected_keyword.py");
    interaction.server.diagnostic("unexpected_keyword.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [
                {
                    "code": "unexpected-keyword",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#unexpected-keyword"
                    },
                    "message": "Unexpected keyword argument `foo` in function `test`",
                    "range": {
                        "end": {"character": 8, "line": 10},
                        "start": {"character": 5, "line": 10}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_error_documentation_links() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.server.did_open("error_docs_test.py");
    interaction.server.diagnostic("error_docs_test.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 11, "line": 9},
                        "start": {"character": 9, "line": 9}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                },
                {
                    "code": "bad-context-manager",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-context-manager"
                    },
                    "message": "Cannot use `A` as a context manager\n  Object of class `A` has no attribute `__enter__`",
                    "range": {
                        "end": {"character": 8, "line": 17},
                        "start": {"character": 5, "line": 17}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                },
                {
                    "code": "bad-context-manager",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-context-manager"
                    },
                    "message": "Cannot use `A` as a context manager\n  Object of class `A` has no attribute `__exit__`",
                    "range": {
                        "end": {"character": 8, "line": 17},
                        "start": {"character": 5, "line": 17}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                },
                {
                    "code": "missing-attribute",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#missing-attribute"
                    },
                    "message": "Object of class `object` has no attribute `nonexistent_method`",
                    "range": {
                        "end": {"character": 22, "line": 22},
                        "start": {"character": 0, "line": 22}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_unreachable_branch_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}},
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]),
    );

    interaction.server.did_open("unreachable_branch.py");
    interaction.server.diagnostic("unreachable_branch.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [
                {
                    "code": "unreachable-code",
                    "message": "This code is unreachable for the current configuration",
                    "range": {
                        "end": {"character": 12, "line": 6},
                        "start": {"character": 4, "line": 6}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_version_support_publish_diagnostics() {
    let test_files_root = get_test_files_root();
    let root = test_files_root.path().to_path_buf();
    let mut file = root.clone();
    file.push("text_document.py");
    let uri = Url::from_file_path(file).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root);
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        capabilities: Some(serde_json::json!({
            "textDocument": {
                "publishDiagnostics": {
                    "versionSupport": true,
                },
            },
        })),
        ..Default::default()
    });

    let gen_validator = |expected_version: i64| {
        let actual_uri = uri.as_str();
        move |msg: &Message| {
            let Message::Notification(Notification { method, params }) = msg else {
                return ValidationResult::Skip;
            };
            let Some(uri_val) = params.get("uri") else {
                return ValidationResult::Skip;
            };
            let Some(expected_uri) = uri_val.as_str() else {
                return ValidationResult::Skip;
            };
            if expected_uri == actual_uri && method == "textDocument/publishDiagnostics" {
                if let Some(actual_version) = params.get("version") {
                    if let Some(actual_version) = actual_version.as_i64() {
                        assert!(
                            actual_version <= expected_version,
                            "expected version: {}, actual version: {}",
                            expected_version,
                            actual_version
                        );
                        return match actual_version.cmp(&expected_version) {
                            std::cmp::Ordering::Less => ValidationResult::Skip,
                            std::cmp::Ordering::Equal => ValidationResult::Pass,
                            std::cmp::Ordering::Greater => ValidationResult::Fail,
                        };
                    }
                }
            }
            ValidationResult::Skip
        }
    };

    interaction.server.did_open("text_document.py");

    let version = 1;
    interaction.client.expect_message_helper(
        gen_validator(version),
        &format!(
            "publishDiagnostics notification with version {} for file: {}",
            version,
            uri.as_str()
        ),
    );

    interaction.server.did_change("text_document.py", "a = b");

    let version = 2;
    interaction.client.expect_message_helper(
        gen_validator(version),
        &format!(
            "publishDiagnostics notification with version {} for file: {}",
            version,
            uri.as_str()
        ),
    );

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didClose".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": uri.as_str(),
                    "languageId": "python",
                    "version": 3
                },
            }),
        }));

    let version = 3;
    interaction.client.expect_message_helper(
        gen_validator(version),
        &format!(
            "publishDiagnostics notification with version {} for file: {}",
            version,
            uri.as_str()
        ),
    );

    interaction.shutdown();
}
