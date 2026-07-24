/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Position;
use lsp_types::Range;
use lsp_types::SelectionRange;
use lsp_types::Url;
use lsp_types::request::SelectionRangeRequest;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn selection_range_includes_enclosing_scopes() {
    let root = TempDir::new().unwrap();
    std::fs::write(
        root.path().join("test.py"),
        "\
def outer():
    if enabled:
        value = transform(source)
    return value
result = outer()
",
    )
    .unwrap();

    let root_uri = Url::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), root_uri)]),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("test.py");

    let uri = Url::from_file_path(root.path().join("test.py")).unwrap();
    interaction
        .client
        .send_request::<SelectionRangeRequest>(json!({
            "textDocument": {"uri": uri},
            "positions": [{"line": 2, "character": 28}],
        }))
        .expect_response_with(|response: Option<Vec<SelectionRange>>| {
            let Some(mut selection) = response.and_then(|ranges| ranges.into_iter().next()) else {
                return false;
            };
            let mut ranges = Vec::new();
            loop {
                ranges.push(selection.range);
                let Some(parent) = selection.parent else {
                    break;
                };
                selection = *parent;
            }

            let expected_scopes = [
                Range::new(Position::new(2, 8), Position::new(2, 33)),
                Range::new(Position::new(1, 4), Position::new(2, 33)),
                Range::new(Position::new(0, 0), Position::new(3, 16)),
                Range::new(Position::new(0, 0), Position::new(5, 0)),
            ];
            expected_scopes
                .iter()
                .try_fold(0, |start, expected| {
                    ranges[start..]
                        .iter()
                        .position(|range| range == expected)
                        .map(|offset| start + offset + 1)
                })
                .is_some()
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
