/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

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

#[test]
fn test_notebook_provide_type_second_cell() {
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
        vec!["foo = \"bar\"\nfoo", "x = 1\nbar = x"],
    );

    // `bar` is at cell2 line 1; its type is the value of `x`.
    interaction
        .provide_type_cell("notebook.ipynb", "cell2", 1, 0)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "typing.Literal[1]",
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
