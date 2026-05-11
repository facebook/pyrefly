/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_notebook_prepare_rename() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    // Cell 1: "x = 1", Cell 2: "y = x"
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    // Prepare rename on "x" at (0, 0) in cell1 should return the range of "x"
    interaction
        .prepare_rename_cell("notebook.ipynb", "cell1", 0, 0)
        .expect_response(json!({
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        }))
        .unwrap();

    // Prepare rename on "x" at (0, 4) in cell2 should return the range of "x"
    interaction
        .prepare_rename_cell("notebook.ipynb", "cell2", 0, 4)
        .expect_response(json!({
            "start": {"line": 0, "character": 4},
            "end": {"line": 0, "character": 5},
        }))
        .unwrap();

    // Prepare rename on "y" at (0, 0) in cell2 should return the range of "y"
    interaction
        .prepare_rename_cell("notebook.ipynb", "cell2", 0, 0)
        .expect_response(json!({
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        }))
        .unwrap();

    // Prepare rename on whitespace (0, 2) in cell1 should return null
    interaction
        .prepare_rename_cell("notebook.ipynb", "cell1", 0, 2)
        .expect_response(json!(null))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_rename() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    // Cell 1: "x = 1", Cell 2: "y = x"
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    // Rename "x" to "z" should rename x in both cells
    interaction
        .rename_cell("notebook.ipynb", "cell1", 0, 0, "z")
        .expect_response(json!({
            "changes": {
                cell1_uri.to_string(): [
                    {
                        "newText": "z",
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}
                    },
                ],
                cell2_uri.to_string(): [
                    {
                        "newText": "z",
                        "range": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 5}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
