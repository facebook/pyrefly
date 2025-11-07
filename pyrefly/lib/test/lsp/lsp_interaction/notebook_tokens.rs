/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_semantic_tokens_full() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "x2 = 1"]);

    interaction.semantic_tokens_cell("notebook.ipynb", "cell1");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"data":[0,0,1,8,0]})),
        error: None,
    });

    interaction.semantic_tokens_cell("notebook.ipynb", "cell2");
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"data":[0,0,2,8,0]})),
        error: None,
    });
    interaction.shutdown();
}

#[test]
fn test_semantic_tokens_ranged() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["x = 1\nx2 = 1"]);

    // Request ranged semantic tokens for just the first line in the first cell
    // It should be the same as the cell1 semantic tokens in test_semantic_tokens_full
    interaction.semantic_tokens_ranged_cell("notebook.ipynb", "cell1", 0, 0, 1, 0);
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"data":[0,0,1,8,0]})),
        error: None,
    });
    interaction.shutdown();
}
