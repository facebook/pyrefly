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
                        "end": {"character": 12, "line": 1},
                        "start": {"character": 4, "line": 1}
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
