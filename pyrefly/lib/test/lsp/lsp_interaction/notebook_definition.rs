/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::expect_definition_points_to_symbol;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_definition_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    // fake configuration so we never find system python
    // todo(kylei): better solution for this
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{"pythonPath": "/fake/python/path"}]))),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["from typing import List"]);

    // Jump to definition of "List"
    // Check that the response points to List in typing.pyi
    interaction
        .definition_cell("notebook.ipynb", "cell1", 0, 20)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "typing.pyi", "List =")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_definition_cross_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    // Jump to definition of "x" in the second cell
    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .definition_cell("notebook.ipynb", "cell2", 0, 4)
        .expect_response(json!({
            "uri": cell1_uri,
            "range": {
                "start": {
                    "line": 0,
                    "character": 0
                },
                "end": {
                    "line": 0,
                    "character": 1
                }
            }
        }))
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_definition_same_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x = 1\ny = x"]);

    // Jump to definition of "x" on the second line
    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .definition_cell("notebook.ipynb", "cell1", 1, 4)
        .expect_response(json!({
            "uri": cell1_uri,
            "range": {
                "start": {
                    "line": 0,
                    "character": 0
                },
                "end": {
                    "line": 0,
                    "character": 1
                }
            }
        }))
        .unwrap();
    interaction.shutdown().unwrap();
}
