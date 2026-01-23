/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::ExpectedInlayHint;
use crate::test::lsp::lsp_interaction::util::ExpectedTextEdit;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::inlay_hints_match_expected;

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
            let expected = [ExpectedInlayHint {
                labels: &[
                    " -> ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2", "]",
                    "]",
                ],
                position: (0, 21),
                text_edit: ExpectedTextEdit {
                    new_text: " -> tuple[Literal[1], Literal[2]]",
                    range_start: (0, 21),
                    range_end: (0, 21),
                },
            }];
            inlay_hints_match_expected(result, &expected)
        })
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell2", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let expected = [ExpectedInlayHint {
                labels: &[
                    ": ", "tuple", "[", "Literal", "[", "1", "]", ", ", "Literal", "[", "2", "]",
                    "]",
                ],
                position: (0, 6),
                text_edit: ExpectedTextEdit {
                    new_text: ": tuple[Literal[1], Literal[2]]",
                    range_start: (0, 6),
                    range_end: (0, 6),
                },
            }];
            inlay_hints_match_expected(result, &expected)
        })
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell3", 0, 0, 100, 0)
        .expect_response_with(|result| {
            let expected = [ExpectedInlayHint {
                labels: &[" -> ", "Literal", "[", "0", "]"],
                position: (0, 15),
                text_edit: ExpectedTextEdit {
                    new_text: " -> Literal[0]",
                    range_start: (0, 15),
                    range_end: (0, 15),
                },
            }];
            inlay_hints_match_expected(result, &expected)
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
