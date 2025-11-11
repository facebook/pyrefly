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
fn test_notebook_references() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["x = 1\ny = x"]);

    interaction.references_cell("notebook.ipynb", "cell1", 0, 0, true);
    // TODO: references always returns empty for notebooks atm
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([])),
        error: None,
    });
    interaction.shutdown();
}
