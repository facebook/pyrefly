/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_inlay_hint_default_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response(json!([
            {
                "label":[
                    {"value":" -> "},
                    {"value":"tuple"},
                    {"value":"["},
                    {"value":"Literal"},
                    {"value":"["},
                    {"value":"1"},
                    {"value":"]"},
                    {"value":", "},
                    {"value":"Literal"},
                    {"value":"["},
                    {"value":"2"},
                    {"value":"]"},
                    {"value":"]"}
                ],
                "position":{"character":21,"line":6},
                "textEdits":[{
                    "newText":" -> tuple[Literal[1], Literal[2]]",
                    "range":{"end":{"character":21,"line":6},"start":{"character":21,"line":6}}
                }]
            },
            {
                "label":[
                    {"value":": "},
                    {"value":"tuple"},
                    {"value":"["},
                    {"value":"Literal"},
                    {"value":"["},
                    {"value":"1"},
                    {"value":"]"},
                    {"value":", "},
                    {"value":"Literal"},
                    {"value":"["},
                    {"value":"2"},
                    {"value":"]"},
                    {"value":"]"}
                ],
                "position":{"character":6,"line":11},
                "textEdits":[{
                    "newText":": tuple[Literal[1], Literal[2]]",
                    "range":{"end":{"character":6,"line":11},"start":{"character":6,"line":11}}
                }]
            },
            {
                "label":[
                    {"value":" -> "},
                    {"value":"Literal"},
                    {"value":"["},
                    {"value":"0"},
                    {"value":"]"}
                ],
                "position":{"character":15,"line":14},
                "textEdits":[{
                    "newText":" -> Literal[0]",
                    "range":{"end":{"character":15,"line":14},"start":{"character":15,"line":14}}
                }]
            }
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_default_and_pyrefly_analysis() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
                "pyrefly":{"analysis": {}},
                "analysis": {
                    "inlayHints": {
                        "callArgumentNames": "off",
                        "functionReturnTypes": false,
                        "pytestParameters": false,
                        "variableTypes": false
                    },
                }
            }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response(json!([]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_disable_all() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
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
        })
        .unwrap();

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response(json!([]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_disable_variables() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
                "analysis": {
                    "inlayHints": {
                        "variableTypes": false
                    },
                }
            }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response(json!([{
            "label":[
                {"value":" -> "},
                {"value":"tuple"},
                {"value":"["},
                {"value":"Literal"},
                {"value":"["},
                {"value":"1"},
                {"value":"]"},
                {"value":", "},
                {"value":"Literal"},
                {"value":"["},
                {"value":"2"},
                {"value":"]"},
                {"value":"]"}
            ],
            "position":{"character":21,"line":6},
            "textEdits":[{
                "newText":" -> tuple[Literal[1], Literal[2]]",
                "range":{"end":{"character":21,"line":6},"start":{"character":21,"line":6}}
            }]
        },
        {
            "label":[
                {"value":" -> "},
                {"value":"Literal"},
                {"value":"["},
                {"value":"0"},
                {"value":"]"}
            ],
            "position":{"character":15,"line":14},
            "textEdits":[{
                "newText":" -> Literal[0]",
                "range":{"end":{"character":15,"line":14},"start":{"character":15,"line":14}}
            }]
        }]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_disable_returns() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
                "analysis": {
                    "inlayHints": {
                        "functionReturnTypes": false
                    },
                }
            }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response(json!([{
            "label":[
                {"value":": "},
                {"value":"tuple"},
                {"value":"["},
                {"value":"Literal"},
                {"value":"["},
                {"value":"1"},
                {"value":"]"},
                {"value":", "},
                {"value":"Literal"},
                {"value":"["},
                {"value":"2"},
                {"value":"]"},
                {"value":"]"}
            ],
            "position":{"character":6,"line":11},
            "textEdits":[{
                "newText":": tuple[Literal[1], Literal[2]]",
                "range":{"end":{"character":6,"line":11},"start":{"character":6,"line":11}}
            }]
        }]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_labels_support_goto_type_definition() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("type_def_inlay_hint_test.py");

    // Expect LabelParts with location information for clickable type hints
    interaction
        .client
        .inlay_hint("type_def_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };

            // Should have hints for the function return type and variable type
            if hints.len() != 2 {
                return false;
            }

            // Check that the hints have label parts (not simple strings)
            for hint in hints {
                match &hint.label {
                    lsp_types::InlayHintLabel::LabelParts(parts) => {
                        if parts.is_empty() {
                            return false;
                        }

                        // Check that at least one label part has a location
                        // (The first part is typically the prefix like " -> " with no location,
                        // while the type name part has the location)
                        if !parts.iter().any(|part| part.location.is_some()) {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
