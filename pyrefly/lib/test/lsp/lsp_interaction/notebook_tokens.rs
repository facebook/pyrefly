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
fn test_semantic_tokens_full() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "x2 = 1"]);

    interaction
        .semantic_tokens_cell("notebook.ipynb", "cell1")
        .expect_response(json!({"data":[0,0,1,8,0]}))
        .unwrap();

    interaction
        .semantic_tokens_cell("notebook.ipynb", "cell2")
        .expect_response(json!({"data":[0,0,2,8,0]}))
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_semantic_tokens_ranged() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x = 1\nx2 = 1"]);

    // Request ranged semantic tokens for just the first line in the first cell
    // It should be the same as the cell1 semantic tokens in test_semantic_tokens_full
    interaction
        .semantic_tokens_ranged_cell("notebook.ipynb", "cell1", 0, 0, 1, 0)
        .expect_response(json!({"data":[0,0,1,8,0]}))
        .unwrap();
    interaction.shutdown().unwrap();
}
