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

use lsp_types::DocumentDiagnosticReport;
use lsp_types::DocumentDiagnosticReportResult;
use lsp_types::Url;
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::request::WorkspaceConfiguration;
use pyrefly_util::fs_anyhow::write;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_did_change_configuration() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{}]));

    interaction.shutdown().expect("Failed to shutdown");
}

#[cfg(unix)]
fn setup_dummy_interpreter(custom_interpreter_path: &Path) -> PathBuf {
    // Create a mock Python interpreter script that returns the environment info
    // This simulates what a real Python interpreter would return when queried with the env script
    let python_script = format!(
        r#"#!/usr/bin/env bash
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
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("custom_interpreter/src/foo.py");
    // Prior to the config taking effect, there should be 1 diagnostic showing an import error
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            1,
        )
        .expect("Failed to receive publish diagnostics");

    // The definition response is in the same file
    interaction
        .client
        .definition("custom_interpreter/src/foo.py", 5, 31)
        .expect_definition_response_from_root("custom_interpreter/src/foo.py", 5, 26, 5, 37)
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .expect("")
        .send_configuration_response(json!([
            {
                "pythonPath": interpreter_path.to_str().unwrap()
            }
        ]));
    // After the new config takes effect, publish diagnostics should have 0 errors
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    // The definition can now be found in site-packages
    interaction
        .client
        .definition("custom_interpreter/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter/bin/site-packages/custom_module.py",
            6,
            6,
            6,
            17,
        )
        .unwrap();

    // Try setting the interpreter back to a bad interpreter, and make sure it fails
    // successfully
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .expect("")
        .send_configuration_response(json!([
            {
                "pythonPath": bad_interpreter_path.to_str().unwrap()
            }
        ]));
    // After the bad config takes effect, publish diagnostics should have 1 error
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            1,
        )
        .expect("Failed to receive publish diagnostics");
    // The definition should not be found in site-packages
    interaction
        .client
        .definition("custom_interpreter/src/foo.py", 5, 31)
        .expect_definition_response_from_root("custom_interpreter/src/foo.py", 5, 26, 5, 37)
        .unwrap();

    interaction.shutdown().expect("Failed to shutdown");
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
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction
        .client
        .did_open("custom_interpreter_config/src/foo.py");
    // Prior to the config taking effect, things should work with the interpreter in the provided
    // config
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root
                .path()
                .join("custom_interpreter_config/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    // The definition response is in the same file
    interaction
        .client
        .definition("custom_interpreter_config/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter_config/bin/site-packages/custom_module.py",
            6,
            6,
            6,
            17,
        )
        .unwrap();

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .expect("")
        .send_configuration_response(json!([
            {
                "pythonPath": bad_interpreter_path.to_str().unwrap()
            }
        ]));
    // After the new config takes effect, results should stay the same
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root
                .path()
                .join("custom_interpreter_config/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    // The definition can still be found in site-packages
    interaction
        .client
        .definition("custom_interpreter_config/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter_config/bin/site-packages/custom_module.py",
            6,
            6,
            6,
            17,
        )
        .unwrap();

    interaction.shutdown().expect("Failed to shutdown");
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_interpreter_change_removes_type_errors() {
    let test_files_root = get_test_files_root();
    let good_interpreter_path =
        setup_dummy_interpreter(&test_files_root.path().join("custom_interpreter"));
    let bad_interpreter_path = setup_dummy_interpreter(
        &test_files_root
            .path()
            .join("interpreter_with_no_site_packages"),
    );

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

    interaction.client.did_open("custom_interpreter/src/foo.py");
    // Without any interpreter configured, there should be 1 import error
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            1,
        )
        .unwrap();
    // Configure broken interpreter with empty site-packages - should still have 1 import error
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .unwrap()
        .send_configuration_response(json!([
            {
                "pythonPath": bad_interpreter_path.to_str().unwrap()
            }
        ]));
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            1,
        )
        .unwrap();

    // Switch to good interpreter with site-packages
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .unwrap()
        .send_configuration_response(json!([
            {
                "pythonPath": good_interpreter_path.to_str().unwrap()
            }
        ]));

    // After switching to good interpreter, the error should be resolved to 0.
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            0,
        )
        .unwrap();

    interaction.shutdown().unwrap();
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_interpreter_change_changes_existing_type_errors() {
    let test_files_root = get_test_files_root();
    let interpreter_path = setup_dummy_interpreter(
        &test_files_root
            .path()
            .join("interpreter_with_no_site_packages"),
    );

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

    interaction.client.did_open("custom_interpreter/src/foo.py");
    // Without any interpreter configured, there should be 1 import error
    interaction
        .client
        .expect_publish_diagnostics_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            1,
        )
        .unwrap();
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_request::<WorkspaceConfiguration>(json!({"items":[{"section":"python"}]}))
        .unwrap()
        .send_configuration_response(json!([
            {
                "pythonPath": interpreter_path.to_str().unwrap()
            }
        ]));
    interaction
        .client
        .expect_publish_diagnostics_message_contains(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            "interpreter_with_no_site_packages",
        )
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_disable_language_services() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"disableLanguageServices": true}}]));

    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!([]))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_language_services_default_workspace() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"disableLanguageServices": true}}]));

    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!([]))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_specific_language_services_via_analysis_config() {
    let test_files_root = get_test_files_root();
    let this_test_root = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(this_test_root.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(this_test_root.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    // Test hover works initially
    interaction
        .client
        .hover("foo.py", 6, 17)
        .expect_response(json!({
            "contents": {
                "kind":"markdown",
                "value":"```python\n(class) Bar: type[Bar]\n```\n\nGo to [Bar](".to_owned()
                    + Url::from_file_path(this_test_root.join("bar.py")).unwrap().as_str()
                    + "#L7,7)"
            }
        }))
        .unwrap();

    // Test definition works initially
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!({
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
        }))
        .unwrap();

    // Change configuration to disable only hover (using pyrefly.disabledLanguageServices)
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([
            {
                "pyrefly": {
                    "disabledLanguageServices": {
                        "hover": true,
                    }
                }
            }
        ]));

    // Hover should now be disabled
    interaction
        .client
        .hover("foo.py", 6, 17)
        .expect_response(json!({"contents": []}))
        .expect("Failed to receive expected response");

    // But definition should still work
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!({
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
        }))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_did_change_workspace_folder() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction
        .client
        .send_notification::<DidChangeWorkspaceFolders>(json!({
            "event": {
            "added": [{"uri": Url::from_file_path(&root).unwrap(), "name": "test"}],
            "removed": [],
            }
        }));

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{}]));

    interaction.shutdown().expect("Failed to shutdown");
}

fn get_diagnostics_result() -> serde_json::Value {
    json!({"items": [
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
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    interaction
        .client
        .hover("foo.py", 6, 17)
        .expect_response(json!({
            "contents": {
                "kind":"markdown",
                "value":"```python\n(class) Bar: type[Bar]\n```\n\nGo to [Bar](".to_owned()
                    + Url::from_file_path(root_path.join("bar.py")).unwrap().as_str()
                    + "#L7,7)"
            }
        }))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_type_errors_workspace_folder() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_type_errors_default_workspace() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_type_errors_config() {
    let root = get_test_files_root();
    let test_files_root = root.path().join("disable_type_error_in_config");
    let scope_uri = Url::from_file_path(test_files_root.as_path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

/// If we failed to parse pylance configs, we would fail to apply the `disableTypeErrors` settings.
/// This test ensures that we don't fail to apply `disableTypeErrors`.
#[test]
fn test_parse_pylance_configs() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([
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
        ]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_diagnostics_default_workspace() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_diagnostics_default_workspace_with_config() {
    let test_root = get_test_files_root();
    let root = test_root.path().join("tests_requiring_config");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_diagnostics_in_workspace() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_diagnostics_file_not_in_includes() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction
        .client
        .did_open("diagnostics_file_not_in_includes/type_errors_exclude.py");
    interaction
        .client
        .did_open("diagnostics_file_not_in_includes/type_errors_include.py");

    // prove that it works for a project included
    interaction
        .client
        .diagnostic("diagnostics_file_not_in_includes/type_errors_include.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    // prove that it ignores a file not in project includes
    interaction
        .client
        .diagnostic("diagnostics_file_not_in_includes/type_errors_exclude.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_diagnostics_file_in_excludes() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction
        .client
        .did_open("diagnostics_file_in_excludes/type_errors_exclude.py");
    interaction
        .client
        .did_open("diagnostics_file_in_excludes/type_errors_include.py");

    // prove that it works for a project included
    interaction
        .client
        .diagnostic("diagnostics_file_in_excludes/type_errors_include.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    // prove that it ignores a file not in project includes
    interaction
        .client
        .diagnostic("diagnostics_file_in_excludes/type_errors_exclude.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_initialization_options_respected() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());

    // Pass configuration via initialization_options instead of waiting for workspace/configuration
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            initialization_options: Some(json!({
                "pyrefly": {
                    "disableLanguageServices": true
                }
            })),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file and immediately test that language services are disabled
    // This proves that initialization_options were respected without needing
    // to wait for workspace/configuration request/response
    // Should return empty array because language services are disabled from initialization_options
    interaction.client.did_open("foo.py");
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!([]))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_initialization_options_without_workspace_folders() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());

    // Pass configuration via initialization_options for a client that doesn't support workspace folders
    // This should apply configuration to the default workspace
    interaction
        .initialize(InitializeSettings {
            workspace_folders: None,
            initialization_options: Some(json!({
                "pyrefly": {
                    "disableLanguageServices": true
                }
            })),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!([]))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_error_missing_imports_mode() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("error_missing_imports_mode");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "error-missing-imports"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("test_file.py");

    interaction
        .client
        .diagnostic("test_file.py")
        .expect_response_with(|response| {
            if let DocumentDiagnosticReportResult::Report(DocumentDiagnosticReport::Full(
                full_report,
            )) = response
            {
                let items = &full_report.full_document_diagnostic_report.items;

                let has_missing_import = items.iter().any(|item| {
                    item.code.as_ref().and_then(|c| match c {
                        lsp_types::NumberOrString::String(s) => Some(s.as_str()),
                        _ => None,
                    }) == Some("missing-import")
                });

                let has_bad_assignment = items.iter().any(|item| {
                    item.code.as_ref().and_then(|c| match c {
                        lsp_types::NumberOrString::String(s) => Some(s.as_str()),
                        _ => None,
                    }) == Some("bad-assignment")
                });

                has_missing_import && !has_bad_assignment
            } else {
                false
            }
        })
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}
