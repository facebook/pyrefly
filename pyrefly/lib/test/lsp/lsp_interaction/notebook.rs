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
use crate::test::lsp::lsp_interaction::util::open_notebook;

#[test]
fn test_notebook_did_open() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    // Send did open notification
    open_notebook(
        "notebook.ipynb",
        &mut interaction,
        &root,
        vec!["x: int = 1", "y: str = \"foo\"", "z: str = ''\nz = 1"],
    );

    // Cell 1 doesn't have any errors
    interaction
        .server
        .diagnostic_for_cell("notebook.ipynb#cell1");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });
    // Cell 3 has an error
    interaction
        .server
        .diagnostic_for_cell("notebook.ipynb#cell3");
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({
            "items": [{
                "code": "bad-assignment",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                },
                "message": "`Literal[1]` is not assignable to variable `z` with type `str`",
                "range": {
                    "start": {"line": 1, "character": 4},
                    "end": {"line": 1, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}
