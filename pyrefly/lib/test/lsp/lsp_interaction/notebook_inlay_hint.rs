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
        .expect_response(json!([{
            "label": [
                {"value": " -> "},
                {"value": "tuple"},
                {"value": "["},
                {"value": "Literal"},
                {"value": "["},
                {"value": "1"},
                {"value": "]"},
                {"value": ", "},
                {"value": "Literal"},
                {"value": "["},
                {"value": "2"},
                {"value": "]"},
                {"value": "]"}
            ],
            "position": {"character": 21, "line": 0},
            "textEdits": [{
                "newText": " -> tuple[Literal[1], Literal[2]]",
                "range": {"end": {"character": 21, "line": 0}, "start": {"character": 21, "line": 0}}
            }]
        }]))
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell2", 0, 0, 100, 0)
        .expect_response(json!([{
            "label": [
                {"value": ": "},
                {"value": "tuple"},
                {"value": "["},
                {"value": "Literal"},
                {"value": "["},
                {"value": "1"},
                {"value": "]"},
                {"value": ", "},
                {"value": "Literal"},
                {"value": "["},
                {"value": "2"},
                {"value": "]"},
                {"value": "]"}
            ],
            "position": {"character": 6, "line": 0},
            "textEdits": [{
                "newText": ": tuple[Literal[1], Literal[2]]",
                "range": {"end": {"character": 6, "line": 0}, "start": {"character": 6, "line": 0}}
            }]
        }]))
        .unwrap();

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell3", 0, 0, 100, 0)
        .expect_response(json!([{
            "label": [
                {"value": " -> "},
                {"value": "Literal"},
                {"value": "["},
                {"value": "0"},
                {"value": "]"}
            ],
            "position": {"character": 15, "line": 0},
            "textEdits": [{
                "newText": " -> Literal[0]",
                "range": {"end": {"character": 15, "line": 0}, "start": {"character": 15, "line": 0}}
            }]
        }]))
        .unwrap();
    interaction.shutdown().unwrap();
}
