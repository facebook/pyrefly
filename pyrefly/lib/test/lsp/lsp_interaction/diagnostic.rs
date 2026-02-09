/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_types::DocumentDiagnosticReportResult;
use lsp_types::Url;
use lsp_types::notification::Notification as _;
use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::Initialize;
use lsp_types::request::Request as _;
use lsp_types::request::WorkspaceConfiguration;
use pyrefly_util::stdlib::register_stdlib_paths;
use serde_json::Value;
use serde_json::json;

use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Notification;
use crate::lsp::non_wasm::protocol::Request;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::object_model::LspMessageError;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn require_markdown_initialize(interaction: &LspInteraction) {
    let settings = InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    };
    let mut params = interaction.client.get_initialize_params(&settings);
    params["capabilities"]["textDocument"]["diagnostic"]["markupMessageSupport"] = json!(true);
    interaction.client.send_message(Message::Request(Request {
        id: RequestId::from(1),
        method: Initialize::METHOD.to_owned(),
        params,
        activity_key: None,
    }));
    interaction
        .client
        .expect_any_message()
        .expect("Failed to receive initialize response");
    interaction.client.send_initialized();
    if let Some(settings) = settings.configuration {
        interaction
            .client
            .expect_any_message()
            .expect("Failed to receive configuration request");
        interaction.client.send_response::<WorkspaceConfiguration>(
            RequestId::from(1),
            settings.unwrap_or(json!([])),
        );
    }
}

#[test]
fn test_show_syntax_errors_without_config() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("syntax_errors.py");

    interaction
        .client
        .diagnostic("syntax_errors.py")
        .expect_response(json!({"items": [{"code":"parse-error","codeDescription":{"href":"https://pyrefly.org/en/docs/error-kinds/#parse-error"},"message":"Parse error: Expected an indented block after `if` statement","range":{"end":{"character":1,"line":9},"start":{"character":0,"line":9}},"severity":1,"source":"Pyrefly"}], "kind": "full"}))
        .expect("Failed to receive expected response");
}

#[test]
fn test_diagnostics_markdown_messages() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    require_markdown_initialize(&interaction);

    interaction.client.did_open("syntax_errors.py");

    interaction
        .client
        .expect_message("publishDiagnostics with markdown message", |msg| {
            let Message::Notification(notification) = msg else {
                return None;
            };
            if notification.method != PublishDiagnostics::METHOD {
                return None;
            }
            let uri = notification
                .params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !uri.ends_with("syntax_errors.py") {
                return None;
            }
            let diagnostics = notification
                .params
                .get("diagnostics")
                .and_then(Value::as_array)
                .ok_or_else(|| LspMessageError::Custom {
                    description: "Missing diagnostics array".to_owned(),
                });
            let diagnostics = match diagnostics {
                Ok(diagnostics) => diagnostics,
                Err(err) => return Some(Err(err)),
            };
            let Some(diagnostic) = diagnostics.first() else {
                return Some(Err(LspMessageError::Custom {
                    description: "Expected at least one diagnostic".to_owned(),
                }));
            };
            let message = diagnostic
                .get("message")
                .and_then(Value::as_object)
                .ok_or_else(|| LspMessageError::Custom {
                    description: "Expected markdown message object".to_owned(),
                });
            let message = match message {
                Ok(message) => message,
                Err(err) => return Some(Err(err)),
            };
            let kind = message.get("kind").and_then(Value::as_str);
            let value = message.get("value").and_then(Value::as_str);
            if kind != Some("markdown")
                || value != Some("Parse error: Expected an indented block after `if` statement")
            {
                return Some(Err(LspMessageError::Custom {
                    description: format!(
                        "Unexpected markdown message: kind={kind:?} value={value:?}"
                    ),
                }));
            }
            Some(Ok(()))
        })
        .expect("Failed to receive markdown publishDiagnostics message");

    interaction
        .client
        .diagnostic("syntax_errors.py")
        .expect_response(json!({
            "items": [{
                "code":"parse-error",
                "codeDescription":{"href":"https://pyrefly.org/en/docs/error-kinds/#parse-error"},
                "message":{"kind":"markdown","value":"Parse error: Expected an indented block after `if` statement"},
                "range":{"end":{"character":1,"line":9},"start":{"character":0,"line":9}},
                "severity":1,
                "source":"Pyrefly"
            }],
            "kind": "full"
        }))
        .expect("Failed to receive markdown diagnostic response");

    interaction.shutdown().unwrap();
}
#[test]
fn test_stream_diagnostics_after_save() {
    let root = get_test_files_root();
    let root_path = root.path().join("streaming");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "streaming".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();
    let d_path = root_path.join("d.py");
    let b_path = root_path.join("b.py");
    let b_contents = std::fs::read_to_string(&b_path).unwrap();
    interaction.client.did_open("d.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for d");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for d");
    interaction.client.did_open("b.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(b_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for b");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for b");
    let new_contents = b_contents.replace("1", "''");
    interaction.client.edit_file("b.py", &new_contents);
    // Streamed diagnostics
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 1)
        .expect("Failed to receive streamed diagnostics for d");
    // Diagnostics sent again after recheck finishes
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 1)
        .expect("Failed to receive transaction complete diagnostics for d");
    interaction.shutdown().unwrap();
}

#[test]
fn test_stream_diagnostics_no_flicker_after_undo_edit() {
    let root = get_test_files_root();
    let root_path = root.path().join("streaming");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "streaming".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();
    let d_path = root_path.join("d.py");
    let b_path = root_path.join("b.py");
    interaction.client.did_open("d.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for d");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for d");
    interaction.client.did_open("b.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(b_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for b");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for b");
    // Delete contents of b.py & save
    interaction.do_not_commit_next_recheck();
    let b_contents = std::fs::read_to_string(&b_path).unwrap();
    let new_contents = b_contents.replace("1", "''");
    interaction.client.edit_file("b.py", &new_contents);
    // Streamed diagnostic for first recheck
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 1)
        .expect("Failed to receive streamed diagnostics for first edit");
    // While first transaction is suspended, immediately restore contents of b.py & save
    interaction.client.edit_file("b.py", &b_contents);
    // When the transaction completes, it will take the newly saved state of b and the errors should converge
    interaction.continue_recheck();
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count_between(d_path.clone(), 0, 1)
        .expect("Failed to receive transaction complete diagnostics for first edit");
    // Diagnostics for second recheck
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 0)
        .expect("Failed to receive diagnostics for second edit");
    interaction.shutdown().unwrap();
}

/// Test opening a file while a recheck for another file is happening.
/// Start with only b open, then open file d while a recheck for b is happening.
#[test]
#[ignore] // TODO: flaky
fn test_open_file_during_recheck() {
    let root = get_test_files_root();
    let root_path = root.path().join("streaming");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "streaming".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();
    let d_path = root_path.join("d.py");
    let b_path = root_path.join("b.py");
    let b_contents = std::fs::read_to_string(&b_path).unwrap();
    // Open only b initially
    interaction.client.did_open("b.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(b_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for b");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for b");
    // Trigger a recheck by modifying and saving b
    interaction.do_not_commit_next_recheck();
    let new_contents = b_contents.replace("1", "''");
    interaction.client.edit_file("b.py", &new_contents);
    // Streamed diagnostic for first recheck
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(b_path.clone(), 0)
        .expect("Failed to receive streamed diagnostics for first edit");
    // While recheck is blocked, open file d
    interaction.client.did_open("d.py");
    // Expect initial diagnostic for d to show no errors since it's based on old state
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 0)
        .expect("Failed to receive diagnostics for d after opening during recheck");
    // After recheck completes, error count reflects new state
    interaction.continue_recheck();
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 1)
        .expect("Failed to receive transaction complete diagnostics for second edit");

    interaction.shutdown().unwrap();
}

/// Test editing a file (didChange without saving) while a recheck for another file is happening.
/// Start with b and d open, then edit file d while a recheck for b is happening.
#[test]
fn test_edit_file_during_recheck() {
    let root = get_test_files_root();
    let root_path = root.path().join("streaming");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "streaming".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();
    let d_path = root_path.join("d.py");
    let b_path = root_path.join("b.py");
    let b_contents = std::fs::read_to_string(&b_path).unwrap();
    // Open both b and d initially
    interaction.client.did_open("b.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(b_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for b");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for b");
    interaction.client.did_open("d.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 0)
        .expect("Failed to receive initial diagnostics for d");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for d");
    // Set flag to prevent recheck from committing
    interaction.do_not_commit_next_recheck();
    // Trigger a recheck by modifying and saving b
    let new_contents = b_contents.replace("1", "''");
    interaction.client.edit_file("b.py", &new_contents);
    // Streamed diagnostic for first recheck
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(d_path.clone(), 1)
        .expect("Failed to receive streamed diagnostics for first edit");
    // While recheck is blocked, edit file d without saving
    let d_contents = std::fs::read_to_string(&d_path).unwrap();
    let edited_d_contents = format!("{}\nY: int = ''\nZ: int = ''", d_contents);
    interaction.client.did_change("d.py", &edited_d_contents);
    // Streamed errors are replaced w/ diagnostics based on old state + edit
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 2)
        .expect("Failed to receive streamed diagnostics for first edit");
    // After recheck completes, error count reflects new state + edit
    interaction.continue_recheck();
    interaction
        .client
        .expect_publish_diagnostics_must_have_error_count(d_path.clone(), 3)
        .expect("Failed to receive diagnostics for d after editing during recheck");
    interaction.shutdown().unwrap();
}

#[test]
fn test_cycle_class() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("cycle_class/foo.py");

    interaction
        .client
        .diagnostic("cycle_class/foo.py")
        .expect_response(json!({
            "items": [],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unexpected_keyword_range() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.client.did_open("unexpected_keyword.py");

    interaction
        .client
        .diagnostic("unexpected_keyword.py")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_error_documentation_links() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.client.did_open("error_docs_test.py");

    interaction
        .client
        .diagnostic("error_docs_test.py")
        .expect_response(json!({
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
        })).unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unreachable_branch_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unreachable_branch.py");

    interaction
        .client
        .diagnostic("unreachable_branch.py")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unused_parameter_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unused_parameter/example.py");

    interaction
        .client
        .diagnostic("unused_parameter/example.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "unused-parameter",
                    "message": "Parameter `unused_arg` is unused",
                    "range": {
                        "start": {"line": 6, "character": 21},
                        "end": {"line": 6, "character": 31}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unused_parameter_no_report() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unused_parameter/no_report.py");
    interaction
        .client
        .diagnostic("unused_parameter/no_report.py")
        .expect_response(json!({
            "items": [],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unused_import_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unused_import/example.py");

    interaction
        .client
        .diagnostic("unused_import/example.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "unused-import",
                    "message": "Import `os` is unused",
                    "range": {
                        "start": {"line": 6, "character": 7},
                        "end": {"line": 6, "character": 9}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_unused_from_import_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unused_import/from_import.py");

    interaction
        .client
        .diagnostic("unused_import/from_import.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "unused-import",
                    "message": "Import `Dict` is unused",
                    "range": {
                        "start": {"line": 6, "character": 19},
                        "end": {"line": 6, "character": 23}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_diagnostic_import_used_in_all() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("unused_import_all/__init__.py");
    interaction
        .client
        .diagnostic("unused_import_all/__init__.py")
        .expect_response(json!({
            "items": [],
            "kind": "full"
        }))
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_unused_variable_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]));

    interaction.client.did_open("unused_variable/example.py");
    interaction
        .client
        .diagnostic("unused_variable/example.py")
        .expect_response(json!({
                    "items": [
                        {
                            "code": "unused-variable",
                            "message": "Variable `unused_var` is unused",
                            "range": {
                                "start": {"line": 7, "character": 4},
                                "end": {"line": 7, "character": 14}
                            },
                            "severity": 4,
                            "source": "Pyrefly",
                            "tags": [1]
                        }
                    ],
                    "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[cfg(unix)]
#[test]
fn test_publish_diagnostics_preserves_symlink_uri() {
    use std::os::unix::fs::symlink;

    use lsp_types::Url;

    let test_files_root = get_test_files_root();
    let symlink_name = "type_errors_symlink.py";
    let symlink_target = test_files_root.path().join("type_errors.py");
    let symlink_path = test_files_root.path().join(symlink_name);
    symlink(&symlink_target, &symlink_path).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open(symlink_name);
    interaction
        .client
        .expect_publish_diagnostics_uri(&Url::from_file_path(&symlink_path).unwrap(), 1)
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_shows_stdlib_type_errors_with_force_on() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    register_stdlib_paths(vec![
        test_files_root
            .path()
            .join("filtering_stdlib_errors/usr/lib/python3.12"),
    ]);

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    let stdlib_filepath = "filtering_stdlib_errors/usr/lib/python3.12/stdlib_file.py";

    interaction.client.did_open(stdlib_filepath);

    interaction
        .client
        .diagnostic(stdlib_filepath)
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_shows_stdlib_errors_for_multiple_versions_and_paths_with_force_on() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    register_stdlib_paths(vec![
        test_files_root
            .path()
            .join("filtering_stdlib_errors/usr/lib/python3.12"),
    ]);

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction
        .client
        .did_open("filtering_stdlib_errors/usr/local/lib/python3.12/stdlib_file.py");

    interaction
        .client
        .diagnostic("filtering_stdlib_errors/usr/local/lib/python3.12/stdlib_file.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    register_stdlib_paths(vec![
        test_files_root
            .path()
            .join("filtering_stdlib_errors/usr/lib/python3.8"),
    ]);

    interaction
        .client
        .did_open("filtering_stdlib_errors/usr/local/lib/python3.8/stdlib_file.py");

    interaction
        .client
        .diagnostic("filtering_stdlib_errors/usr/local/lib/python3.8/stdlib_file.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction
        .client
        .did_open("filtering_stdlib_errors/usr/lib/python3.12/stdlib_file.py");

    interaction
        .client
        .diagnostic("filtering_stdlib_errors/usr/lib/python3.12/stdlib_file.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    register_stdlib_paths(vec![
        test_files_root
            .path()
            .join("filtering_stdlib_errors/usr/lib64/python3.12"),
    ]);

    interaction
        .client
        .did_open("filtering_stdlib_errors/usr/lib64/python3.12/stdlib_file.py");

    interaction
        .client
        .diagnostic("filtering_stdlib_errors/usr/lib64/python3.12/stdlib_file.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_does_not_filter_out_stdlib_errors_with_default_displaytypeerrors() {
    let test_files_root = get_test_files_root();

    register_stdlib_paths(vec![
        test_files_root
            .path()
            .join("filtering_stdlib_errors_with_default/usr/lib/python3.12"),
    ]);

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "default"}}]));

    let stdlib_filepath = "filtering_stdlib_errors_with_default/usr/lib/python3.12/stdlib_file.py";

    interaction.client.did_open(stdlib_filepath);

    interaction
        .client
        .diagnostic(stdlib_filepath)
        .expect_response(json!({
            "items": [],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_shows_stdlib_errors_when_explicitly_included_in_project_includes() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "default"}}]));

    let stdlib_filepath = "stdlib_with_explicit_includes/usr/lib/python3.12/stdlib_file.py";

    interaction.client.did_open(stdlib_filepath);

    interaction
        .client
        .diagnostic(stdlib_filepath)
        .expect_response(json!({
            "items": [
                {
                    "code": "bad-assignment",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                    },
                    "message": "`Literal['1']` is not assignable to `int`",
                    "range": {
                        "end": {"character": 12, "line": 5},
                        "start": {"character": 9, "line": 5}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_publish_diagnostics_version_numbers_only_go_up() {
    let test_files_root = get_test_files_root();
    let root = test_files_root.path();
    let file = root.join("text_document.py");
    let uri = Url::from_file_path(file).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let create_version_validator = |expected_version: i64| {
        let actual_uri = uri.as_str();
        move |msg: Message| match msg {
            Message::Notification(Notification {
                method,
                params,
                activity_key: None,
            }) if let Some((expected_uri, actual_version)) = params
                .get("uri")
                .and_then(|uri| uri.as_str())
                .zip(params.get("version").and_then(|version| version.as_i64()))
                && expected_uri == actual_uri
                && method == "textDocument/publishDiagnostics" =>
            {
                assert!(
                    actual_version == expected_version,
                    "expected version: {}, actual version: {}",
                    expected_version,
                    actual_version
                );
                (actual_version == expected_version).then_some(Ok(()))
            }
            _ => None,
        }
    };

    interaction.client.did_open("text_document.py");

    let version = 1;
    interaction
        .client
        .expect_message(
            &format!(
                "publishDiagnostics notification with version {} for file: {}",
                version,
                uri.as_str()
            ),
            create_version_validator(version),
        )
        .unwrap();

    interaction.client.did_change("text_document.py", "a = b");

    let version = 2;
    interaction
        .client
        .expect_message(
            &format!(
                "publishDiagnostics notification with version {} for file: {}",
                version,
                uri.as_str()
            ),
            create_version_validator(version),
        )
        .unwrap();

    interaction
        .client
        .send_message(Message::Notification(Notification {
            method: "textDocument/didClose".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": uri.as_str(),
                    "languageId": "python",
                    "version": 3
                },
            }),
            activity_key: None,
        }));

    let version = 3;
    interaction
        .client
        .expect_message(
            &format!(
                "publishDiagnostics notification with version {} for file: {}",
                version,
                uri.as_str()
            ),
            create_version_validator(version),
        )
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_missing_source_for_stubs_diagnostic() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction
        .client
        .did_open("missing_source_for_stubs/test.py");

    interaction
        .client
        .diagnostic("missing_source_for_stubs/test.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "missing-source-for-stubs",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#missing-source-for-stubs"
                    },
                    "message": "Stubs for `whatthepatch` are bundled with Pyrefly but the source files for the package are not found.",
                    "range": {
                        "start": {"line": 5, "character": 7},
                        "end": {"line": 5, "character": 19}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_missing_source_with_config_diagnostic_has_errors() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("missing_source_with_config/test.py");

    interaction
        .client
        .diagnostic("missing_source_with_config/test.py")
        .expect_response_with(|response| {
            if let DocumentDiagnosticReportResult::Report(report) = response
                && let lsp_types::DocumentDiagnosticReport::Full(full) = report
            {
                let items = &full.full_document_diagnostic_report.items;
                if items.len() != 1 {
                    return false;
                }
                let item = &items[0];
                return item.code
                    == Some(lsp_types::NumberOrString::String(
                        "missing-import".to_owned(),
                    ))
                    && item
                        .message
                        .starts_with("Cannot find module `whatthepatch`")
                    && item.range.start.line == 5
                    && item.range.start.character == 7
                    && item.range.end.line == 5
                    && item.range.end.character == 19
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR);
            }
            false
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_untyped_import_diagnostic_does_not_show_non_recommended_packages() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction
        .client
        .did_open("untyped_import_with_source/test.py");

    interaction
        .client
        .diagnostic("untyped_import_with_source/test.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "unused-import",
                    "message": "Import `boto3` is unused",
                    "range": {
                        "start": {"line": 5, "character": 7},
                        "end": {"line": 5, "character": 12}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_untyped_import_diagnostic_shows_error_for_recommended_packages() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .unwrap()
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.client.did_open("untyped_import_django/test.py");

    interaction
        .client
        .diagnostic("untyped_import_django/test.py")
        .expect_response(json!({
            "items": [
                {
                    "code": "untyped-import",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#untyped-import"
                    },
                    "message": "Cannot find type stubs for module `django`\n  Hint: install the `django-stubs` package",
                    "range": {
                        "start": {"line": 5, "character": 7},
                        "end": {"line": 5, "character": 13}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                },
                {
                    "code": "unused-import",
                    "message": "Import `django` is unused",
                    "range": {
                        "start": {"line": 5, "character": 7},
                        "end": {"line": 5, "character": 13}
                    },
                    "severity": 4,
                    "source": "Pyrefly",
                    "tags": [1]
                }
            ],
            "kind": "full"
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
