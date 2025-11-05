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
#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;

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

#[cfg(unix)]
fn setup_dummy_interpreter(custom_interpreter_path: &Path) -> PathBuf {
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

    let interpreter_path = custom_interpreter_path.join("bin/python");
    write(&interpreter_path, python_script).unwrap();
    let mut perms = fs::metadata(&interpreter_path).unwrap().permissions();
    perms.set_mode(0o755); // rwxr-xr-x
    fs::set_permissions(&interpreter_path, perms).unwrap();

    interpreter_path
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_pythonpath_change() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");
    let bad_interpreter_root = test_files_root.path().join("bad_interpreter_bin");

    // Interpreter path that should be able to find an expected import in site packages
    let interpreter_path = setup_dummy_interpreter(&custom_interpreter_path);
    // Interpreter path that should *not* be able to find an expected import in site packages.
    // This is more to make sure that the test in
    // [`test_workspace_pythonpath_ignored_when_set_in_config_file`] works correctly by proving
    // in this test that we will fail to find an import using this interpreter.
    let bad_interpreter_path = setup_dummy_interpreter(&bad_interpreter_root);

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
                "pythonPath": interpreter_path.to_str().unwrap()
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

    // Try setting the interpreter back to a bad interpreter, and make sure it fails
    // successfully
    interaction.server.did_change_configuration();
    interaction.client.expect_request(Request {
        id: RequestId::from(3),
        method: "workspace/configuration".to_owned(),
        params: serde_json::json!({"items":[{"section":"python"}]}),
    });
    interaction.server.send_configuration_response(
        3,
        serde_json::json!([
            {
                "pythonPath": bad_interpreter_path.to_str().unwrap()
            }
        ]),
    );
    // After the bad config takes effect, publish diagnostics should have 1 error
    interaction.client.expect_publish_diagnostics_error_count(
        test_files_root.path().join("custom_interpreter/src/foo.py"),
        1,
    );
    // The definition should not be found in site-packages
    interaction
        .server
        .definition("custom_interpreter/src/foo.py", 5, 31);
    interaction.client.expect_definition_response_from_root(
        "custom_interpreter/src/foo.py",
        5,
        26,
        5,
        37,
    );

    interaction.shutdown();
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_workspace_pythonpath_ignored_when_set_in_config_file() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter_config");
    let bad_interpreter_root = test_files_root.path().join("bad_interpreter_bin");

    // Interpreter path that should be able to find an expected import in site packages.
    // This is set in a pyrefly.toml, so we don't actually need to use the value here, but it
    // still needs to be set up.
    let _ = setup_dummy_interpreter(&custom_interpreter_path);
    // Interpreter path that should *not* be able to find an expected import in site packages.
    // We try to pass this in but make sure we still use the interpreter set in the config,
    // which is proven when the import is able to be found. The [`test_pythonpath_change`] test
    // above proves that setting this interpreter will fail to find anything.
    let bad_interpreter_path = setup_dummy_interpreter(&bad_interpreter_root);

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction
        .server
        .did_open("custom_interpreter_config/src/foo.py");
    // Prior to the config taking effect, things should work with the interpreter in the provided
    // config
    interaction.client.expect_publish_diagnostics_error_count(
        test_files_root
            .path()
            .join("custom_interpreter_config/src/foo.py"),
        0,
    );
    interaction
        .server
        .definition("custom_interpreter_config/src/foo.py", 5, 31);
    // The definition response is in the same file
    interaction.client.expect_definition_response_from_root(
        "custom_interpreter_config/bin/site-packages/custom_module.py",
        6,
        6,
        6,
        17,
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
                "pythonPath": bad_interpreter_path.to_str().unwrap()
            }
        ]),
    );
    // After the new config takes effect, results should stay the same
    interaction.client.expect_publish_diagnostics_error_count(
        test_files_root
            .path()
            .join("custom_interpreter_config/src/foo.py"),
        0,
    );
    // The definition can still be found in site-packages
    interaction
        .server
        .definition("custom_interpreter_config/src/foo.py", 5, 31);
    interaction.client.expect_definition_response_from_root(
        "custom_interpreter_config/bin/site-packages/custom_module.py",
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
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
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
            "uri": Url::from_file_path(root_path.join("bar.py")).unwrap().to_string(),
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
    let root_path = test_files_root.path().join("basic");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");
    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(root_path.join("bar.py")).unwrap().to_string(),
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
fn test_disable_specific_language_services_via_analysis_config() {
    let test_files_root = get_test_files_root();
    let this_test_root = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(this_test_root.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(this_test_root.to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");

    // Test hover works initially
    interaction.server.hover("foo.py", 6, 17);
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "contents": {
                "kind":"markdown",
                "value":"```python\n(class) Bar: type[Bar]\n```\n\nGo to [Bar](".to_owned()
                    + Url::from_file_path(this_test_root.join("bar.py")).unwrap().as_str()
                    + "#L7,7)"
            }
        })),
        error: None,
    });

    // Test definition works initially
    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(this_test_root.join("bar.py")).unwrap().to_string(),
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

    // Change configuration to disable only hover (mimicking pyrefly.analysis.disabledLanguageServices)
    interaction.server.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(2, Some(vec![&scope_uri]));
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pyrefly": {
                    "analysis": {
                        "disabledLanguageServices": {
                            "hover": true,
                        }
                    }
                }
            },
            {
                "pyrefly": {
                    "analysis": {
                        "disabledLanguageServices": {
                            "hover": true,
                        }
                    }
                }
            }
        ]),
    );

    // Hover should now be disabled
    interaction.server.hover("foo.py", 6, 17);
    interaction.client.expect_response(Response {
        id: RequestId::from(4),
        result: Some(serde_json::json!({"contents": []})),
        error: None,
    });

    // But definition should still work
    interaction.server.definition("foo.py", 6, 16);
    interaction.client.expect_response(Response {
        id: RequestId::from(5),
        result: Some(serde_json::json!({
            "uri": Url::from_file_path(this_test_root.join("bar.py")).unwrap().to_string(),
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
            {"code":"unsupported-operation","codeDescription":{"href":"https://pyrefly.org/en/docs/error-kinds/#unsupported-operation"},"message":"`+` is not supported between `Literal[1]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            "range":{"end":{"character":6,"line":5},"start":{"character":0,"line":5}},"severity":1,"source":"Pyrefly"}],"kind":"full"
    })
}

#[test]
fn test_disable_type_errors_language_services_still_work() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
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
                    + Url::from_file_path(root_path.join("bar.py")).unwrap().as_str()
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

#[test]
fn test_diagnostics_file_not_in_includes() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction
        .server
        .did_open("diagnostics_file_not_in_includes/type_errors_exclude.py");
    interaction
        .server
        .did_open("diagnostics_file_not_in_includes/type_errors_include.py");

    interaction
        .server
        .diagnostic("diagnostics_file_not_in_includes/type_errors_include.py");

    // prove that it works for a project included
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction
        .server
        .diagnostic("diagnostics_file_not_in_includes/type_errors_exclude.py");

    // prove that it ignores a file not in project includes
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_diagnostics_file_in_excludes() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction
        .server
        .did_open("diagnostics_file_in_excludes/type_errors_exclude.py");
    interaction
        .server
        .did_open("diagnostics_file_in_excludes/type_errors_include.py");

    interaction
        .server
        .diagnostic("diagnostics_file_in_excludes/type_errors_include.py");

    // prove that it works for a project included
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(get_diagnostics_result()),
        error: None,
    });

    interaction
        .server
        .diagnostic("diagnostics_file_in_excludes/type_errors_exclude.py");

    // prove that it ignores a file not in project includes
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.shutdown();
}
