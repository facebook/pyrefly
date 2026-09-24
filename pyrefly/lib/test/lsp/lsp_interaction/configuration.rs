/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::request::WorkspaceConfiguration;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;
#[cfg(unix)]
use crate::test::python_env::TestVenv;

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

#[test]
fn test_invalid_workspace_configuration_response_does_not_crash() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    };

    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction
        .client
        .expect_any_message()
        .expect("Failed to initialize");
    interaction.client.send_initialized();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_unchecked_response(json!("not-a-list"));

    interaction.shutdown().expect("Failed to shutdown");
}

#[cfg(unix)]
#[test]
fn test_workspace_discovers_project_venv() {
    let test_files_root = get_test_files_root();
    let project_root = test_files_root.path().join("custom_interpreter");
    TestVenv::synthetic(project_root.join(".venv"))
        .add_site_package_module(
            &project_root
                .join("explicit_interpreter/lib/python3.12/site-packages/custom_module.py"),
        )
        .create_mock_interpreter();

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            initialization_options: Some(json!({
                "pyrefly": {"streamDiagnostics": false},
            })),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("custom_interpreter/src/foo.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(project_root.join("src/foo.py"), 0)
        .expect("Failed to receive publish diagnostics");
    interaction
        .client
        .definition("custom_interpreter/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter/.venv/lib/python3.12/site-packages/custom_module.py",
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
fn test_pythonpath_change() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");

    // The import below resolves only via the `pythonPath` config exercised
    // later in this test.
    let interpreter_path = TestVenv::mock_interpreter_excluded_from_discovery(
        custom_interpreter_path.join("explicit_interpreter"),
    );

    // This interpreter's site-packages doesn't contain `custom_module.py`, so it
    // should *not* resolve the import below.
    // `test_workspace_pythonpath_ignored_when_set_in_config_file` relies on a
    // `bad_interpreter_bin`-rooted `pythonPath` behaving the same way: passing
    // it alongside a config-set interpreter only sees 0 errors because the
    // config interpreter (not the bad one) is actually used.
    let bad_interpreter_path =
        TestVenv::mock_interpreter(test_files_root.path().join("bad_interpreter_bin"));

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            initialization_options: Some(json!({
                "pyrefly": {"streamDiagnostics": false},
            })),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("custom_interpreter/src/foo.py");
    // Prior to the config taking effect, there should be 1 diagnostic showing an import error
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
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
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root.path().join("custom_interpreter/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    // The definition can now be found in site-packages
    interaction
        .client
        .definition("custom_interpreter/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter/explicit_interpreter/lib/python3.12/site-packages/custom_module.py",
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
        .expect_publish_diagnostics_eventual_error_count(
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

    // Reachable only via the `python-interpreter-path` set in the fixture's
    // `pyrefly.toml`.
    TestVenv::mock_interpreter_excluded_from_discovery(
        custom_interpreter_path.join("explicit_interpreter"),
    );

    // This interpreter's site-packages doesn't contain `custom_module.py`.
    // `test_pythonpath_change` proves that using it as `pythonPath` fails to
    // resolve the import; passing it here alongside the config's own
    // interpreter and still seeing 0 errors below proves the config
    // interpreter takes precedence over an explicit `pythonPath`.
    let bad_interpreter_path =
        TestVenv::mock_interpreter(test_files_root.path().join("bad_interpreter_bin"));

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
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("custom_interpreter_config/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    interaction
        .client
        .definition("custom_interpreter_config/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter_config/explicit_interpreter/lib/python3.12/site-packages/custom_module.py",
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
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("custom_interpreter_config/src/foo.py"),
            0,
        )
        .expect("Failed to receive publish diagnostics");
    interaction
        .client
        .definition("custom_interpreter_config/src/foo.py", 5, 31)
        .expect_definition_response_from_root(
            "custom_interpreter_config/explicit_interpreter/lib/python3.12/site-packages/custom_module.py",
            6,
            6,
            6,
            17,
        )
        .unwrap();

    interaction.shutdown().expect("Failed to shutdown");
}

// A config with `skip-interpreter-query = true` opts out of interpreter queries
// entirely: a client-provided `pythonPath` must not be applied, even though it
// would resolve the import. Regression test for the LSP eagerly querying (and
// applying) the client interpreter despite the opt-out.
// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_skip_interpreter_query_ignores_lsp_pythonpath() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");
    // This interpreter *would* resolve `custom_module` if it were applied, so the
    // test distinguishes "pythonPath applied" (0 errors) from "pythonPath ignored
    // because of `skip-interpreter-query`" (1 error).
    let good_interpreter_path = TestVenv::mock_interpreter_excluded_from_discovery(
        custom_interpreter_path.join("explicit_interpreter"),
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

    interaction
        .client
        .did_open("skip_interpreter_config/src/foo.py");
    // `skip-interpreter-query = true` with no `site-package-path` in the config means
    // the import cannot be resolved.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("skip_interpreter_config/src/foo.py"),
            1,
        )
        .unwrap();

    // Even though this interpreter would resolve the import, the config opted out of
    // interpreter queries, so the import error must persist.
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
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("skip_interpreter_config/src/foo.py"),
            1,
        )
        .unwrap();

    interaction.shutdown().unwrap();
}

// A client-provided `pythonPath` fills in what the config left unset; it must not
// discard what the config set explicitly. Regression test for the LSP replacing the
// whole Python environment with the interpreter's, which dropped an explicit
// `python-version` and silently type checked against the interpreter's version.
// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_config_python_version_survives_lsp_pythonpath() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");
    // The import below resolves only via the `pythonPath` config applied
    // later in this test. This interpreter reports 3.12.0, disagreeing with
    // the `python-version = "3.9"` in the fixture's config. The fixture's
    // only error sits behind a `sys.version_info >= (3, 10)` guard, so it is
    // reported iff the configured version was discarded in favor of the
    // interpreter's.
    let interpreter_path = TestVenv::mock_interpreter_excluded_from_discovery(
        custom_interpreter_path.join("explicit_interpreter"),
    );

    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            initialization_options: Some(json!({
                "pyrefly": {"streamDiagnostics": false},
            })),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("python_version_config/src/foo.py");
    // Both the unresolved import and the `typing.override` error that the
    // configured 3.9 produces before any interpreter is applied.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("python_version_config/src/foo.py"),
            2,
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
    // The interpreter resolves the import, proving it was applied. The
    // `typing.override` error remains, so `python-version` is still the
    // configured 3.9; had it been overwritten with the interpreter's 3.12 that
    // error would have disappeared too, leaving no diagnostics.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(
            test_files_root
                .path()
                .join("python_version_config/src/foo.py"),
            1,
        )
        .unwrap();

    interaction.shutdown().unwrap();
}

// Only run this test on unix since windows has no way to mock a .exe without compiling something
// (we call python with python.exe)
#[cfg(unix)]
#[test]
fn test_interpreter_change_removes_type_errors() {
    let test_files_root = get_test_files_root();
    let custom_interpreter_path = test_files_root.path().join("custom_interpreter");
    // The import below resolves only via the `pythonPath` config exercised
    // later in this test.
    let good_interpreter_path = TestVenv::mock_interpreter_excluded_from_discovery(
        custom_interpreter_path.join("explicit_interpreter"),
    );

    // A missing (not merely empty) `site-packages` directory, matching the
    // fixture name: `custom_module` must still fail to resolve.
    let bad_interpreter_path = TestVenv::synthetic_without_site_packages(
        test_files_root
            .path()
            .join("interpreter_with_no_site_packages"),
    )
    .create_mock_interpreter();

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
        .expect_publish_diagnostics_eventual_error_count(
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
        .expect_publish_diagnostics_eventual_error_count(
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
        .expect_publish_diagnostics_eventual_error_count(
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
    // A missing (not merely empty) `site-packages` directory, matching the
    // fixture name.
    let interpreter_path = TestVenv::synthetic_without_site_packages(
        test_files_root
            .path()
            .join("interpreter_with_no_site_packages"),
    )
    .create_mock_interpreter();

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
        .expect_publish_diagnostics_eventual_error_count(
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
        .expect_publish_diagnostics_eventual_message_contains(
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
        .expect_response(json!(null))
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
        .expect_response(json!(null))
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
        .expect_hover_response_with_markup(|value| {
            value.is_some_and(|text| {
                text.contains("(class) Bar: def Bar() -> Bar: ...")
                    && text.contains(
                        Url::from_file_path(this_test_root.join("bar.py"))
                            .unwrap()
                            .as_str(),
                    )
            })
        })
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
        .expect_response(json!(null))
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
        .expect_hover_response_with_markup(|value| {
            value.is_some_and(|text| {
                text.contains("(class) Bar: def Bar() -> Bar: ...")
                    && text.contains(
                        Url::from_file_path(root_path.join("bar.py"))
                            .unwrap()
                            .as_str(),
                    )
            })
        })
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

/// `displayTypeErrors` is the legacy IDE setting; it's deprecated in
/// favor of `typeCheckingMode` + `disableTypeErrors`. The legacy values
/// map onto the new model as:
/// - `force-off` → workspace `disableTypeErrors = true` (kill switch)
/// - `force-on` → `typeCheckingMode = "default"` (Default preset)
///
/// This test pins both dynamic transitions: empty (Basic, errors
/// silenced) → `force-on` (Default, errors visible) → `force-off`
/// (kill switch, errors hidden). The recheck after each
/// `did_change_configuration` is async, so the test waits on streamed
/// `publishDiagnostics` notifications (push) rather than firing a
/// synchronous `diagnostic` pull that would race the cache
/// invalidation.
#[test]
fn test_disable_type_errors_workspace_folder() {
    let test_files_root = get_test_files_root();
    let scope_uri = Url::from_file_path(test_files_root.path()).unwrap();
    let type_errors_path = test_files_root.path().join("type_errors.py");
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

    // Initial empty configuration → resolver picks `Basic` preset,
    // which silences `unsupported-operation` (the only error in
    // `type_errors.py`).
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive initial empty diagnostics");

    // Switch to `force-on` → maps to `typeCheckingMode = "default"`,
    // which routes through the resolver's config-cache invalidation.
    // Wait for the recheck-driven publish (1 error: unsupported-operation).
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 1)
        .expect("Failed to receive force-on diagnostics");

    // Switch to `force-off` → maps to `disableTypeErrors = true`
    // (workspace kill switch). The kill switch silences every
    // diagnostic; wait for the publish that drops the count back to 0.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive force-off diagnostics");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_disable_type_errors_default_workspace() {
    let test_files_root = get_test_files_root();
    let type_errors_path = test_files_root.path().join("type_errors.py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    // Initial empty configuration → Basic preset → silenced.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive initial empty diagnostics");

    // `force-on` → Default preset → 1 error visible.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 1)
        .expect("Failed to receive force-on diagnostics");

    // `force-off` → kill switch → suppressed.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-off"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive force-off diagnostics");

    interaction.shutdown().expect("Failed to shutdown");
}

/// `disable-type-errors-in-ide = true` in `pyrefly.toml` suppresses
/// IDE diagnostics for files in the project. Legacy `displayTypeErrors
/// = "force-on"` does NOT pierce this flag — `disableTypeErrors` is a
/// clean two-state boolean and the project's committed config wins.
/// This test pins that contract so a future change can't silently
/// re-introduce a force-show override.
#[test]
fn test_disable_type_errors_in_config_wins_over_force_on() {
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

    // Initial: in-config disable suppresses errors.
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    interaction.client.did_change_configuration();

    // After legacy `force-on`: still suppressed. The legacy mapping
    // sets `typeCheckingMode = "default"` (which doesn't apply because
    // the project has a real config) and is a no-op on
    // `disableTypeErrors`. The in-config disable wins.
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .diagnostic("type_errors.py")
        .expect_response(json!({"items": [], "kind": "full"}))
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
                    "diagnosticMode": "openFilesOnly",
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

/// Dynamic switch from an empty configuration (Basic preset, errors
/// silenced) to `displayTypeErrors = "force-on"` (legacy mapping →
/// `typeCheckingMode = "default"`, errors visible) without an explicit
/// workspace folder. The recheck after `did_change_configuration` is
/// async, so the test waits on streamed `publishDiagnostics` rather
/// than firing a synchronous `diagnostic` pull that would race the
/// cache invalidation.
#[test]
fn test_diagnostics_default_workspace() {
    let root = get_test_files_root();
    let type_errors_path = root.path().join("type_errors.py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("type_errors.py");

    // Empty configuration → Basic preset silences `unsupported-operation`.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive initial empty diagnostics");

    // `force-on` → Default preset → 1 error visible.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(None)
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 1)
        .expect("Failed to receive force-on diagnostics");

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

/// Dynamic switch from an empty configuration (Basic preset, errors
/// silenced) to `displayTypeErrors = "force-on"` (legacy mapping →
/// `typeCheckingMode = "default"`, errors visible) inside an explicit
/// workspace folder. The recheck after `did_change_configuration` is
/// async, so the test waits on streamed `publishDiagnostics` rather
/// than firing a synchronous `diagnostic` pull that would race the
/// cache invalidation.
#[test]
fn test_diagnostics_in_workspace() {
    let root = get_test_files_root();
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let type_errors_path = root.path().join("type_errors.py");
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

    // Empty configuration → Basic preset silences `unsupported-operation`.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 0)
        .expect("Failed to receive initial empty diagnostics");

    // `force-on` → Default preset → 1 error visible.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]));
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(type_errors_path.clone(), 1)
        .expect("Failed to receive force-on diagnostics");

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

/// `pyrefly.extraProjectExcludes` lets a client push its own notion of excluded
/// directories down to the server without writing a `pyrefly.toml`. It is
/// additive: the config file's `project-excludes` still applies.
#[test]
fn test_client_project_excludes() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("client_project_excludes");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Url::from_file_path(&root_path).unwrap(),
            )]),
            // The glob is relative to the workspace folder, mirroring how an
            // editor reports its excluded content roots.
            initialization_options: Some(json!({
                "pyrefly": {
                    "displayTypeErrors": "force-on",
                    "extraProjectExcludes": ["generated"]
                }
            })),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("included.py");
    interaction.client.did_open("excluded_by_config.py");
    interaction
        .client
        .did_open("generated/excluded_by_client.py");

    interaction
        .client
        .diagnostic("included.py")
        .expect_response(get_diagnostics_result())
        .expect("Failed to receive expected response");

    interaction
        .client
        .diagnostic("generated/excluded_by_client.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .expect("Failed to receive expected response");

    // The client's excludes are appended to the config's, not substituted for
    // them, so a file the project itself excluded stays excluded.
    interaction
        .client
        .diagnostic("excluded_by_config.py")
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
        .expect_response(json!(null))
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
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_fallback_search_path_heuristics_nested() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root = test_files_root
        .path()
        .join("fallback_search_path_heuristics_nested");
    interaction.set_root(root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Url::from_file_path(&root).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("src/main.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(root.join("src/main.py"), 0)
        .unwrap();
    interaction.shutdown().unwrap();
}
