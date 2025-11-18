/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_signature_help_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.open_notebook(
        "notebook.ipynb",
        vec!["def f(a: str, b: int) -> None: ...\nf("],
    );

    interaction.signature_help_cell("notebook.ipynb", "cell1", 1, 2);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(json!({
            "signatures": [{
                "label": "def f(\n    a: str,\n    b: int\n) -> None: ...",
                "parameters": [
                    {"label": "a: str"},
                    {"label": "b: int"}
                ],
                "activeParameter": 0
            }],
            "activeSignature": 0,
            "activeParameter": 0
        })),
        error: None,
    });

    // Provide the first argument & check signature help works for the second one
    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        json!({
            "cells": {
                "textContent": [{
                    "document": {
                        "uri": cell_uri,
                        "version": 2
                    },
                    "changes": [{
                        "text": "def f(a: str, b: int) -> None: ...\nf(\"hello\", "
                    }]
                }]
            }
        }),
    );

    interaction.signature_help_cell("notebook.ipynb", "cell1", 1, 11);

    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(json!({
            "signatures": [{
                "label": "def f(\n    a: str,\n    b: int\n) -> None: ...",
                "parameters": [
                    {"label": "a: str"},
                    {"label": "b: int"}
                ],
                "activeParameter": 1
            }],
            "activeSignature": 0,
            "activeParameter": 1
        })),
        error: None,
    });

    interaction.shutdown();
}
