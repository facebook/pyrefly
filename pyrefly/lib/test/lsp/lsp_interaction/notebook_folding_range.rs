/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::FoldingRange;
use lsp_types::request::FoldingRangeRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_notebook_folding_range() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    // Cell 1: class with a method (2 folding ranges: class body, method body)
    // Cell 2: standalone function (1 folding range: function body)
    interaction.open_notebook(
        "notebook.ipynb",
        vec![
            "class Foo:\n    def bar(self) -> None:\n        pass",
            "def baz() -> int:\n    return 42",
        ],
    );

    // Request folding ranges for cell1 — should only contain ranges from cell1
    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .client
        .send_request::<FoldingRangeRequest>(json!({
            "textDocument": { "uri": cell1_uri }
        }))
        .expect_response_with(|response: Option<Vec<FoldingRange>>| {
            let ranges = response.expect("cell1 should have folding ranges");
            // Class Foo folds from line 0, method bar folds from line 1
            assert_eq!(ranges.len(), 2, "cell1 should have 2 folding ranges");
            assert_eq!(ranges[0].start_line, 0, "class Foo should start at line 0");
            assert_eq!(ranges[1].start_line, 1, "method bar should start at line 1");
            // All ranges should be cell-relative (no lines beyond cell1's content)
            for range in &ranges {
                assert!(
                    range.end_line <= 2,
                    "folding range end_line {} exceeds cell1 bounds",
                    range.end_line
                );
            }
            true
        })
        .unwrap();

    // Request folding ranges for cell2 — should only contain baz's range
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");
    interaction
        .client
        .send_request::<FoldingRangeRequest>(json!({
            "textDocument": { "uri": cell2_uri }
        }))
        .expect_response_with(|response: Option<Vec<FoldingRange>>| {
            let ranges = response.expect("cell2 should have folding ranges");
            assert_eq!(ranges.len(), 1, "cell2 should have 1 folding range");
            assert_eq!(
                ranges[0].start_line, 0,
                "baz should start at line 0 (cell-relative)"
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
