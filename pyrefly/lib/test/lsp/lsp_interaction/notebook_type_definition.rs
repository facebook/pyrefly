/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::expect_definition_points_to_symbol;
use crate::util::get_test_files_root;

#[test]
fn test_notebook_type_definition() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x: list[int] = [1, 2, 3]"]);

    // Go to type definition of "x" should point to `list` in builtins
    interaction
        .type_definition_cell("notebook.ipynb", "cell1", 0, 0)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class list")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
