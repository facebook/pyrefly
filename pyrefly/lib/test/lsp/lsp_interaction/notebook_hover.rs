/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::bundled_typeshed_path;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_hover_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // Open notebook with a single cell containing "x = 3"
    interaction.open_notebook("notebook.ipynb", vec!["x = 3"]);

    // Hover over the "x"
    interaction
        .hover_cell("notebook.ipynb", "cell1", 0, 0)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "```python\n(variable) x: Literal[3]\n```",
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_hover_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // Open notebook with a single cell containing "from typing import List"
    interaction.open_notebook("notebook.ipynb", vec!["from typing import List"]);

    // Hover over "List"
    let expected_path = bundled_typeshed_path().join("builtins.pyi");
    let expected_url = Url::from_file_path(&expected_path).unwrap();

    interaction
        .hover_cell("notebook.ipynb", "cell1", 0, 20)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": format!("```python\n(class) List: type[list]\n```\n\nGo to [list]({}#L3351,7)", expected_url.as_str()),
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
