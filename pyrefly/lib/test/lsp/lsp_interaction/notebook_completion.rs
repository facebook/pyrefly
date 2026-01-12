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
fn test_notebook_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["abcdef = 1", ""]);
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        json!({
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
    interaction
        .completion_cell("notebook.ipynb", "cell2", 0, 2)
        .expect_completion_response_with(|list| {
            list.items.iter().any(|item| item.label == "abcdef")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
