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
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::notification::Notification as _;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_did_change_configuration() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction
        .server
        .send_configuration_response(2, serde_json::json!([{}]));

    interaction.shutdown();
}

#[test]
fn test_disable_language_services() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");
    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
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
    });

    interaction.server.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"disableLanguageServices": true}}, {"pyrefly": {"disableLanguageServices": true}}]));

    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_disable_language_services_default_workspace() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");
    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
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
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"disableLanguageServices": true}}, {"pyrefly": {"disableLanguageServices": true}}]));

    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!([])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_did_change_workspace_folder() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: DidChangeWorkspaceFolders::METHOD.to_owned(),
            params: serde_json::json!({
                "event": {
                "added": [{"uri": Url::from_file_path(&root).unwrap(), "name": "test"}],
                "removed": [],
                }
            }),
        }));

    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction
        .server
        .send_configuration_response(2, serde_json::json!([{}, {}]));

    interaction.shutdown();
}

fn get_diagnostics_result() -> serde_json::Value {
    serde_json::json!({"items": [
            {"code":"unsupported-operation","message":"`+` is not supported between `Literal[1]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            "range":{"end":{"character":6,"line":5},"start":{"character":0,"line":5}},"severity":1,"source":"Pyrefly"}],"kind":"full"
    })
}

#[test]
fn test_disable_type_errors_language_services_still_work() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(Some(serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-off"}}, {"pyrefly": {"displayTypeErrors": "force-off"}}]))),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");

    interaction.server.hover("foo.py", 6, 17);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "contents": {
                "kind":"markdown",
                "value":"```python\n(class) Bar: type[Bar]\n```\n\nGo to [Bar](".to_owned()
                    + Url::from_file_path(test_files_root.path().join("bar.py")).unwrap().as_str()
                    + "#L7,7)"
            }
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_disable_type_errors_workspace_folder() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-off"}}, {"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(3, Some(&scope_uri));
    interaction.server.send_configuration_response(3, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(4),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_disable_type_errors_default_workspace() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-off"}}, {"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(3, None);
    interaction.server.send_configuration_response(3, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(4),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_disable_type_errors_config() {
    let root = get_test_files_root();
    let test_files_root = root.path().join("disable_type_error_in_config");
    let scope_uri = Url::from_file_path(test_files_root.as_path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}

/// If we failed to parse pylance configs, we would fail to apply the `disableTypeErrors` settings.
/// This test ensures that we don't fail to apply `disableTypeErrors`.
#[test]
fn test_parse_pylance_configs() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pyrefly": {"displayTypeErrors": "force-off"},
                "analysis": {
                    "diagnosticMode": "workspace",
                    "importFormat": "relative",
                    "inlayHints": {
                        "callArgumentNames": "on",
                        "functionReturnTypes": true,
                        "pytestParameters": true,
                        "variableTypes": true
                    },
                }
            },
        ]),
    );
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_diagnostics_default_workspace() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_diagnostics_default_workspace_with_config() {
    let test_root = get_test_files_root();
    let root = test_root.path().join("tests_requiring_config");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.clone());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-off"}}, {"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_diagnostics_in_workspace() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("type_errors.py");

    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.server.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(2, Some(&scope_uri));
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}
