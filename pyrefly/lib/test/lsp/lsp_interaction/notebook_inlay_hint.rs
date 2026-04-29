/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::check_inlay_hint_label_values;
use crate::util::get_test_files_root;

#[test]
fn test_inlay_hints() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook(
        "notebook.ipynb",
        vec![
            "def no_return_annot():\n    _ = (1, 2)  # no inlay hint here\n    return (1, 2)",
            "result = no_return_annot()",
            "async def foo():\n    return 0",
        ],
    );

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell1", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }
            let hint = &hints[0];
            if hint.position.line != 0 || hint.position.character != 21 {
                return false;
            }
            if !check_inlay_hint_label_values(
                hint,
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
            let text_edits = match &hint.text_edits {
                Some(edits) if edits.len() == 1 => &edits[0],
                _ => return false,
            };
            text_edits.new_text == " -> tuple[Literal[1], Literal[2]]"
                && text_edits.range.start.line == 0
                && text_edits.range.start.character == 21
                && text_edits.range.end.line == 0
                && text_edits.range.end.character == 21
        })
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell2", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }
            let hint = &hints[0];
            if hint.position.line != 0 || hint.position.character != 6 {
                return false;
            }
            if !check_inlay_hint_label_values(
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
            ) {
                return false;
            }
            let text_edits = match &hint.text_edits {
                Some(edits) if edits.len() == 1 => &edits[0],
                _ => return false,
            };
            text_edits.new_text == ": tuple[Literal[1], Literal[2]]"
                && text_edits.range.start.line == 0
                && text_edits.range.start.character == 6
                && text_edits.range.end.line == 0
                && text_edits.range.end.character == 6
        })
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell3", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            if hints.len() != 1 {
                return false;
            }
            let hint = &hints[0];
            if hint.position.line != 0 || hint.position.character != 15 {
                return false;
            }
            if !check_inlay_hint_label_values(
                hint,
                &[
                    (" -> ", false),
                    ("Literal", true),
                    ("[", false),
                    ("0", false),
                    ("]", false),
                ],
            ) {
                return false;
            }
            let text_edits = match &hint.text_edits {
                Some(edits) if edits.len() == 1 => &edits[0],
                _ => return false,
            };
            text_edits.new_text == " -> Literal[0]"
                && text_edits.range.start.line == 0
                && text_edits.range.start.character == 15
                && text_edits.range.end.line == 0
                && text_edits.range.end.character == 15
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
