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

fn new_notebook_interaction() -> LspInteraction {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction
}

fn expect_cell_content_hover(interaction: &LspInteraction) {
    interaction
        .hover_cell("notebook.ipynb", "cell1", 0, 0)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "```python\n(variable) x: Literal[3]\n```",
            }
        }))
        .unwrap();
}

#[test]
fn test_notebook_hover_basic() {
    let interaction = new_notebook_interaction();
    // Open notebook with a single cell containing "x = 3"
    interaction.open_notebook("notebook.ipynb", vec!["x = 3"]);

    // Hover over the "x"
    expect_cell_content_hover(&interaction);
    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_hover_after_kernel_change() {
    let interaction = new_notebook_interaction();
    // Open notebook with a single cell containing "x = 3"
    interaction.open_notebook("notebook.ipynb", vec!["x = 3"]);

    // Stimulate kernel-switch by sending metadata-only didChange event
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        json!({
        "metadata": {
        "language_info": {"name": "python"},
        }
            }),
    );

    // Hover over the "x" after kernel-switch
    expect_cell_content_hover(&interaction);

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
    interaction
        .hover_cell("notebook.ipynb", "cell1", 0, 20)
        .expect_response_with(|response| {
            if let Some(hover) = response
                && let lsp_types::HoverContents::Markup(content) = &hover.contents
            {
                let value = &content.value;
                return value.contains("(class) List:")
                    && value.contains("def __init__")
                    && value.contains("list[Unknown]")
                    && value.contains("Go to [list](")
                    && value.contains("builtins.pyi#L");
            }
            false
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
