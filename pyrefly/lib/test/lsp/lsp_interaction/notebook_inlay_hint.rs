/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;

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
fn test_inlay_hints() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
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
        .expect_response_with(|response| {
            expect_inlay_labels(response, &[(" -> tuple[Literal[1], Literal[2]]", 0, 21)])
        });

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell2", 0, 0, 100, 0)
        .expect_response_with(|response| {
            expect_inlay_labels(response, &[(": tuple[Literal[1], Literal[2]]", 0, 6)])
        });

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell3", 0, 0, 100, 0)
        .expect_response_with(|response| {
            expect_inlay_labels(response, &[(" -> Literal[0]", 0, 15)])
        });
    interaction.shutdown();
}
