/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Notification;
use lsp_types::DocumentDiagnosticReportResult;
use lsp_types::Url;
use pyrefly_config::environment::environment::PythonEnvironment;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

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

    PythonEnvironment::get_interpreter_stdlib_path()
        .write()
        .insert(
            test_files_root
                .path()
                .join("filtering_stdlib_errors/usr/lib/python3.12"),
        );

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

    PythonEnvironment::get_interpreter_stdlib_path()
        .write()
        .insert(
            test_files_root
                .path()
                .join("filtering_stdlib_errors/usr/lib/python3.12"),
        );

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

    PythonEnvironment::get_interpreter_stdlib_path()
        .write()
        .insert(
            test_files_root
                .path()
                .join("filtering_stdlib_errors/usr/lib/python3.8"),
        );

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

    PythonEnvironment::get_interpreter_stdlib_path()
        .write()
        .insert(
            test_files_root
                .path()
                .join("filtering_stdlib_errors/usr/lib64/python3.12"),
        );

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

    PythonEnvironment::get_interpreter_stdlib_path()
        .write()
        .insert(
            test_files_root
                .path()
                .join("filtering_stdlib_errors_with_default/usr/lib/python3.12"),
        );

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
            Message::Notification(Notification { method, params })
                if let Some((expected_uri, actual_version)) = params
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
                (actual_version == expected_version).then_some(())
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
                        .starts_with("Could not find import of `whatthepatch`")
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
fn test_untyped_import_diagnostic() {
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
                    "code": "untyped-import",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#untyped-import"
                    },
                    "message": "Missing type stubs for `boto3`\n  Hint: install the `boto3-stubs` package",
                    "range": {
                        "start": {"line": 5, "character": 7},
                        "end": {"line": 5, "character": 12}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                },
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
