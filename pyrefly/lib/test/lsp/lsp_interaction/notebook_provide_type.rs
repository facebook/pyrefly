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
fn test_notebook_provide_type() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["x = 1"]);

    // provide_type on "x" should return its type
    interaction
        .provide_type_cell("notebook.ipynb", "cell1", 0, 0)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "typing.Literal[1]",
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
