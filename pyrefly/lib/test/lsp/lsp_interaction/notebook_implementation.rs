/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_notebook_implementation() {
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
        vec!["class Base:\n    def method(self) -> None: ...\n\nclass Child(Base):\n    def method(self) -> None: ..."],
    );

    // Go to implementations of Base.method (line 1, col 8)
    // Should find Child.method (line 4, col 8)
    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .implementation_cell("notebook.ipynb", "cell1", 1, 8)
        .expect_response_with(|response| {
            // Implementation response should contain Child.method in the same cell
            if let Some(lsp_types::GotoDefinitionResponse::Array(locations)) = response {
                locations.iter().any(|loc| {
                    loc.uri == cell1_uri
                        && loc.range.start.line == 4
                        && loc.range.start.character == 8
                        && loc.range.end.character == 14
                })
            } else if let Some(lsp_types::GotoDefinitionResponse::Scalar(loc)) = response {
                loc.uri == cell1_uri
                    && loc.range.start.line == 4
                    && loc.range.start.character == 8
                    && loc.range.end.character == 14
            } else {
                false
            }
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
