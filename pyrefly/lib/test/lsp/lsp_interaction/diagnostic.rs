/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
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
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction.server.did_open(symlink_name);
    interaction.client.expect_publish_diagnostics_exact_uri(
        Url::from_file_path(&symlink_path).unwrap().as_str(),
        1,
    );

    interaction.shutdown();
}
