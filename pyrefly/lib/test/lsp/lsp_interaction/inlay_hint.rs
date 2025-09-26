/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_inlay_hint_default_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("inlay_hint_test.py");
    interaction
        .server
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!([
            {
                "label":" -> tuple[Literal[1], Literal[2]]",
                "position":{"character":21,"line":6},
                "textEdits":[{
                    "newText":" -> tuple[Literal[1], Literal[2]]",
                    "range":{"end":{"character":21,"line":6},"start":{"character":21,"line":6}}
                }]
            },
            {
                "label":": tuple[Literal[1], Literal[2]]",
                "position":{"character":6,"line":11},
                "textEdits":[{
                    "newText":": tuple[Literal[1], Literal[2]]",
                    "range":{"end":{"character":6,"line":11},"start":{"character":6,"line":11}}
                }]
            },
            {
                "label":" -> Literal[0]",
                "position":{"character":15,"line":14},
                "textEdits":[{
                    "newText":" -> Literal[0]",
                    "range":{"end":{"character":15,"line":14},"start":{"character":15,"line":14}}
                }]
            }
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_all() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(serde_json::json!([{
            "analysis": {
                "inlayHints": {
                    "callArgumentNames": "all",
                    "functionReturnTypes": false,
                    "pytestParameters": false,
                    "variableTypes": false
                },
            }
        }]))),
        ..Default::default()
    });

    interaction.server.did_open("inlay_hint_test.py");

    interaction
        .server
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!([])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_variables() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(serde_json::json!([{
            "analysis": {
                "inlayHints": {
                    "variableTypes": false
                },
            }
        }]))),
        ..Default::default()
    });

    interaction.server.did_open("inlay_hint_test.py");

    interaction
        .server
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!([{
            "label":" -> tuple[Literal[1], Literal[2]]",
            "position":{"character":21,"line":6},
            "textEdits":[{
                "newText":" -> tuple[Literal[1], Literal[2]]",
                "range":{"end":{"character":21,"line":6},"start":{"character":21,"line":6}}
            }]
        },
        {
            "label":" -> Literal[0]",
            "position":{"character":15,"line":14},
            "textEdits":[{
                "newText":" -> Literal[0]",
                "range":{"end":{"character":15,"line":14},"start":{"character":15,"line":14}}
            }]
        }])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_returns() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(serde_json::json!([{
            "analysis": {
                "inlayHints": {
                    "functionReturnTypes": false
                },
            }
        }]))),
        ..Default::default()
    });

    interaction.server.did_open("inlay_hint_test.py");

    interaction
        .server
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!([{
            "label":": tuple[Literal[1], Literal[2]]",
            "position":{"character":6,"line":11},
            "textEdits":[{
                "newText":": tuple[Literal[1], Literal[2]]",
                "range":{"end":{"character":6,"line":11},"start":{"character":6,"line":11}}
            }]
        }])),
        error: None,
    });

    interaction.shutdown();
}
