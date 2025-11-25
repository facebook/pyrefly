/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;
use lsp_types::Position as LspPosition;
use lsp_types::Range;
use lsp_types::Url;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn expect_inlay_labels(response: Option<Vec<InlayHint>>, expected: &[(&str, u32, u32)]) -> bool {
    match response {
        Some(hints) => {
            if hints.len() != expected.len() {
                return false;
            }
            hints
                .iter()
                .zip(expected.iter())
                .all(|(hint, (label, line, character))| {
                    if hint.position.line != *line || hint.position.character != *character {
                        return false;
                    }
                    let rendered: String = match &hint.label {
                        InlayHintLabel::String(text) => text.clone(),
                        InlayHintLabel::LabelParts(parts) => {
                            parts.iter().map(|part| part.value.as_str()).collect()
                        }
                    };
                    if rendered != *label {
                        return false;
                    }
                    match &hint.text_edits {
                        Some(edits) => edits.iter().any(|edit| edit.new_text.as_str() == *label),
                        None => false,
                    }
                })
        }
        None => expected.is_empty(),
    }
}

#[test]
fn test_inlay_hint_default_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|response| {
            expect_inlay_labels(
                response,
                &[
                    (" -> tuple[Literal[1], Literal[2]]", 6, 21),
                    (": tuple[Literal[1], Literal[2]]", 11, 6),
                    (" -> Literal[0]", 14, 15),
                ],
            )
        });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_default_and_pyrefly_analysis() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
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
    });

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|response| expect_inlay_labels(response, &[]));

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_all() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
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
    });

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|response| expect_inlay_labels(response, &[]));

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_variables() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(json!([{
            "analysis": {
                "inlayHints": {
                    "variableTypes": false
                },
            }
        }]))),
        ..Default::default()
    });

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|response| {
            expect_inlay_labels(
                response,
                &[
                    (" -> tuple[Literal[1], Literal[2]]", 6, 21),
                    (" -> Literal[0]", 14, 15),
                ],
            )
        });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_disable_returns() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(json!([{
            "analysis": {
                "inlayHints": {
                    "functionReturnTypes": false
                },
            }
        }]))),
        ..Default::default()
    });

    interaction.client.did_open("inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|response| {
            expect_inlay_labels(response, &[(": tuple[Literal[1], Literal[2]]", 11, 6)])
        });

    interaction.shutdown();
}

#[test]
fn test_inlay_hint_labels_support_goto_type_definition() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    let expected_uri =
        Url::from_file_path(root.path().join("type_def_inlay_hint_test.py")).unwrap();

    interaction.client.did_open("type_def_inlay_hint_test.py");

    // Expect LabelParts with location information for clickable type hints
    interaction
        .client
        .inlay_hint("type_def_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|maybe_hints| {
            let Some(hints) = maybe_hints else {
                return false;
            };
            if hints.len() != 2 {
                return false;
            }

            let expected_range = Range::new(LspPosition::new(6, 6), LspPosition::new(6, 13));

            let check_hint = |hint: &lsp_types::InlayHint, prefix: &str| match &hint.label {
                InlayHintLabel::LabelParts(parts) => {
                    if parts.is_empty() {
                        return false;
                    }
                    if parts.first().unwrap().value != prefix {
                        return false;
                    }
                    parts.iter().any(|part| {
                        if let Some(location) = &part.location {
                            location.uri == expected_uri && location.range == expected_range
                        } else {
                            false
                        }
                    })
                }
                _ => false,
            };

            check_hint(&hints[0], " -> ") && check_hint(&hints[1], ": ")
        });

    interaction.shutdown();
}
