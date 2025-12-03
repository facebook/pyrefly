/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::ExpectedInlayHint;
use crate::test::lsp::lsp_interaction::util::ExpectedTextEdit;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::inlay_hints_match_expected;

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

    let expected = [
        ExpectedInlayHint {
            labels: &[
                " -> ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2", "]", "]",
            ],
            position: (6, 21),
            text_edit: ExpectedTextEdit {
                new_text: " -> tuple[Literal[1], Literal[2]]",
                range_start: (6, 21),
                range_end: (6, 21),
            },
        },
        ExpectedInlayHint {
            labels: &[
                ": ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2", "]", "]",
            ],
            position: (11, 6),
            text_edit: ExpectedTextEdit {
                new_text: ": tuple[Literal[1], Literal[2]]",
                range_start: (11, 6),
                range_end: (11, 6),
            },
        },
        ExpectedInlayHint {
            labels: &[" -> ", "Literal", "[", "0", "]"],
            position: (14, 15),
            text_edit: ExpectedTextEdit {
                new_text: " -> Literal[0]",
                range_start: (14, 15),
                range_end: (14, 15),
            },
        },
    ];

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| inlay_hints_match_expected(result, &expected))
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
        .expect_response_with(|result| {
            let expected = [
                ExpectedInlayHint {
                    labels: &[
                        " -> ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2",
                        "]", "]",
                    ],
                    position: (6, 21),
                    text_edit: ExpectedTextEdit {
                        new_text: " -> tuple[Literal[1], Literal[2]]",
                        range_start: (6, 21),
                        range_end: (6, 21),
                    },
                },
                ExpectedInlayHint {
                    labels: &[" -> ", "Literal", "[", "0", "]"],
                    position: (14, 15),
                    text_edit: ExpectedTextEdit {
                        new_text: " -> Literal[0]",
                        range_start: (14, 15),
                        range_end: (14, 15),
                    },
                },
            ];
            inlay_hints_match_expected(result, &expected)
        })
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
        .expect_response_with(|result| {
            let expected = [ExpectedInlayHint {
                labels: &[
                    ": ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2", "]",
                    "]",
                ],
                position: (11, 6),
                text_edit: ExpectedTextEdit {
                    new_text: ": tuple[Literal[1], Literal[2]]",
                    range_start: (11, 6),
                    range_end: (11, 6),
                },
            }];
            inlay_hints_match_expected(result, &expected)
        })
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

#[test]
fn test_inlay_hint_typing_literals_have_locations() {
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
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };

            hints.iter().any(|hint| match &hint.label {
                lsp_types::InlayHintLabel::LabelParts(parts) => parts.iter().any(|part| {
                    part.value == "Literal"
                        && part
                            .location
                            .as_ref()
                            .map(|loc| loc.uri.path().contains("typing.pyi"))
                            .unwrap_or(false)
                }),
                _ => false,
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
