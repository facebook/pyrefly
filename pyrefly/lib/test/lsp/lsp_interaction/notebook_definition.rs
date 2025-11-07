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
fn test_notebook_definition_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["from typing import List"]);

    // Jump to definition of "List"
    interaction.definition_cell("notebook.ipynb", "cell1", 0, 20);

    // Check that the response uri ends with "typing.py"
    interaction.client.expect_response_with(
        |response| {
            response.result.as_ref().is_some_and(|r| {
                r.get("uri")
                    .is_some_and(|uri| uri.as_str().is_some_and(|x| x.ends_with("typing.py")))
            })
        },
        "expected definition in typing.pyi",
    );
    interaction.shutdown();
}

#[test]
fn test_notebook_definition_cross_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    // Jump to definition of "x" in the second cell
    interaction.definition_cell("notebook.ipynb", "cell2", 0, 4);
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": interaction.cell_uri("notebook.ipynb", "cell1"),
            "range": {
                "start": {
                    "line": 0,
                    "character": 0
                },
                "end": {
                    "line": 0,
                    "character": 1
                }
            }
        })),
        error: None,
    });
    interaction.shutdown();
}

#[test]
fn test_notebook_definition_same_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["x = 1\ny = x"]);

    // Jump to definition of "x" on the second line
    interaction.definition_cell("notebook.ipynb", "cell1", 1, 4);
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "uri": interaction.cell_uri("notebook.ipynb", "cell1"),
            "range": {
                "start": {
                    "line": 0,
                    "character": 0
                },
                "end": {
                    "line": 0,
                    "character": 1
                }
            }
        })),
        error: None,
    });
    interaction.shutdown();
}
