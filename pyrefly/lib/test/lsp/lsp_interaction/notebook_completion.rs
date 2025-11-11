/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["abcdef = 1", ""]);
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        serde_json::json!({
            "cells": {
                "textContent": [{
                    "document": {
                        "uri": interaction.cell_uri("notebook.ipynb", "cell2"),
                        "version": 2
                    },
                    "changes": [{
                        "text": "abc"
                    }]
                }]
            }
        }),
    );
    interaction.completion_cell("notebook.ipynb", "cell2", 0, 2);
    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                return items_array.iter().any(|item| {
                    if let Some(label) = item.get("label")
                        && let Some(label_str) = label.as_str()
                    {
                        label_str == "abcdef"
                    } else {
                        false
                    }
                });
            }
            false
        },
        "Expected completion response with 'abcdef' in items",
    );
    interaction.shutdown();
}
