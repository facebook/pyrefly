/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_lsp_test::object_model::CellKind;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::check_inlay_hint_label_values;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_inlay_hints() {
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
            check_inlay_hint_label_values(
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
            )
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
            check_inlay_hint_label_values(
                hint,
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

/// Regression test: a markdown cell preceding a code cell must not suppress the
/// code cell's inlay hints. The cell filter compares the code-cell index (which
/// skips markdown cells) against `to_cell_for_lsp`, so the latter must also
/// return a code-cell index rather than the absolute all-cells index.
#[test]
fn test_inlay_hints_with_preceding_markdown_cell() {
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
        .unwrap();
    // cell1: code, cell2: markdown, cell3: code (the cell under test).
    interaction.open_notebook_with_kinds(
        "notebook.ipynb",
        vec![
            (CellKind::Code, "def make():\n    return (1, 2)"),
            (CellKind::Markdown, "# Title"),
            (CellKind::Code, "result = make()"),
        ],
    );

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell3", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let hints = match result {
                Some(hints) => hints,
                None => return false,
            };
            // The `result = make()` assignment must get a variable-type hint
            // right after `result` (column 6), despite the preceding markdown cell.
            hints.len() == 1 && hints[0].position.line == 0 && hints[0].position.character == 6
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
