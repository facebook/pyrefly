/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(unix)]
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::notification::Notification as _;
use pyrefly_util::fs_anyhow::write;

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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
    interaction
        .server
        .send_configuration_response(2, serde_json::json!([{}]));

    interaction.shutdown();
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_pythonpath_change() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");

    // Create a mock Python interpreter script that returns the environment info
    // This simulates what a real Python interpreter would return when queried with the env script
    let python_script = format!(
        r#"#!/bin/bash
if [[ "$1" == "-c" && "$2" == *"import json, sys"* ]]; then
    cat << 'EOF'
{{"python_platform": "linux", "python_version": "3.12.0", "site_package_path": ["{site_packages}"]}}
EOF
else
    echo "Mock python interpreter - args: $@" >&2
    exit 1
fi
"#,
        site_packages = custom_interpreter_path
            .join("bin/site-packages")
            .to_str()
            .unwrap()
    );

    let interpreter_suffix = if cfg!(windows) { ".exe" } else { "" };
    let python_path = custom_interpreter_path.join(format!("bin/python{interpreter_suffix}"));
    write(&python_path, python_script).unwrap();
    let mut perms = fs::metadata(&python_path).unwrap().permissions();
    perms.set_mode(0o755); // rwxr-xr-x
    fs::set_permissions(&python_path, perms).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction.server.did_open("custom_interpreter/src/foo.py");
    // Prior to the config taking effect, there should be 1 diagnostic showing an import error
    interaction.client.expect_publish_diagnostics_error_count(
        test_files_root.path().join("custom_interpreter/src/foo.py"),
        1,
    );
    interaction
        .server
        .definition("custom_interpreter/src/foo.py", 5, 31);
    // The definition response is in the same file
    interaction.client.expect_definition_response_from_root(
        "custom_interpreter/src/foo.py",
        5,
        26,
        5,
        37,
    );

    interaction.server.did_change_configuration();
    interaction.client.expect_request(Request {
        id: RequestId::from(2),
        method: "workspace/configuration".to_owned(),
        params: serde_json::json!({"items":[{"section":"python"}]}),
    });
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pythonPath": python_path.to_str().unwrap()
            }
        ]),
    );
    // After the new config takes effect, publish diagnostics should have 0 errors
    interaction.client.expect_publish_diagnostics_error_count(
        test_files_root.path().join("custom_interpreter/src/foo.py"),
        0,
    );
    // The definition can now be found in site-packages
    interaction
        .server
        .definition("custom_interpreter/src/foo.py", 5, 31);
    interaction.client.expect_definition_response_from_root(
        "custom_interpreter/bin/site-packages/custom_module.py",
        6,
        6,
        6,
        17,
    );

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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
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
        .expect_configuration_request(3, Some(vec![&scope_uri]));
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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
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
        .expect_configuration_request(2, Some(vec![&scope_uri]));
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction.server.diagnostic("type_errors.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction.shutdown();
}
