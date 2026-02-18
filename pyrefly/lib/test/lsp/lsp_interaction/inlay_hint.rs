/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::check_inlay_hint_label_values;
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
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 3 {
                return false;
            }

            let hint0 = &hints[0];
            if hint0.position.line != 6 || hint0.position.character != 21 {
                return false;
            }
            if !check_inlay_hint_label_values(
                hint0,
                &[
                    (" -> ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            ) {
                return false;
            }

            let hint1 = &hints[1];
            if hint1.position.line != 11 || hint1.position.character != 6 {
                return false;
            }
            if !check_inlay_hint_label_values(
                hint1,
                &[
                    (": ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            ) {
                return false;
            }

            let hint2 = &hints[2];
            if hint2.position.line != 14 || hint2.position.character != 15 {
                return false;
            }
            check_inlay_hint_label_values(
                hint2,
                &[
                    (" -> ", false),
                    ("Literal", true),
                    ("[", false),
                    ("0", false),
                    ("]", false),
                ],
            )
        })
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
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 2 {
                return false;
            }

            let hint0 = &hints[0];
            if hint0.position.line != 6 || hint0.position.character != 21 {
                return false;
            }
            if !check_inlay_hint_label_values(
                hint0,
                &[
                    (" -> ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            ) {
                return false;
            }

            let hint1 = &hints[1];
            if hint1.position.line != 14 || hint1.position.character != 15 {
                return false;
            }
            check_inlay_hint_label_values(
                hint1,
                &[
                    (" -> ", false),
                    ("Literal", true),
                    ("[", false),
                    ("0", false),
                    ("]", false),
                ],
            )
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
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 11 || hint.position.character != 6 {
                return false;
            }
            check_inlay_hint_label_values(
                hint,
                &[
                    (": ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            )
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
fn test_inlay_hint_tuple_type_has_location() {
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
            if hints.len() != 3 {
                return false;
            }

            let hint0 = &hints[0];
            if !check_inlay_hint_label_values(
                hint0,
                &[
                    (" -> ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            ) {
                return false;
            }

            let hint1 = &hints[1];
            check_inlay_hint_label_values(
                hint1,
                &[
                    (": ", false),
                    ("tuple", true),
                    ("[", false),
                    ("Literal", true),
                    ("[", false),
                    ("1", false),
                    ("]", false),
                    (", ", false),
                    ("Literal", true),
                    ("[", false),
                    ("2", false),
                    ("]", false),
                    ("]", false),
                ],
            )
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_typevar_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("typevar_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("typevar_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 10 || hint.position.character != 14 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("TypeVar", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_typevartuple_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("typevartuple_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("typevartuple_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 10 || hint.position.character != 14 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("TypeVarTuple", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_paramspec_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("paramspec_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("paramspec_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 10 || hint.position.character != 14 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("ParamSpec", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_class_based_typed_dict_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("typed_dict_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("typed_dict_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 13 || hint.position.character != 24 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("MyTypedDict", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_anonymous_typed_dict_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("anonymous_typed_dict_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("anonymous_typed_dict_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 6 || hint.position.character != 34 {
                return false;
            }
            check_inlay_hint_label_values(
                hint,
                &[
                    (" -> ", false),
                    ("dict", true),
                    ("[", false),
                    ("str", true),
                    (", ", false),
                    ("int", true),
                    (" | ", false),
                    ("str", true),
                    ("]", false),
                ],
            )
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_never_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("never_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("never_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 6 || hint.position.character != 19 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("Never", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_literal_string_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("literal_string_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("literal_string_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 8 || hint.position.character != 40 {
                return false;
            }
            check_inlay_hint_label_values(hint, &[(" -> ", false), ("LiteralString", true)])
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_inlay_hint_type_guard_has_location() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("type_guard_inlay_hint_test.py");

    interaction
        .client
        .inlay_hint("type_guard_inlay_hint_test.py", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }

            let hint = &hints[0];
            if hint.position.line != 12 || hint.position.character != 7 {
                return false;
            }
            check_inlay_hint_label_values(
                hint,
                &[
                    (": ", false),
                    ("(", false),
                    ("val", false),
                    (": ", false),
                    ("object", true),
                    (") -> ", false),
                    ("TypeGuard", true),
                    ("[", false),
                    ("str", true),
                    ("]", false),
                ],
            )
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
