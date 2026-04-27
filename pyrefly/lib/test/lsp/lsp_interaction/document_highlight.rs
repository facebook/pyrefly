/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::request::DocumentHighlightRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn document_highlight_includes_read_write_kind() {
    let root = get_test_files_root();
    let path = root.path().join("document_highlight.py");
    std::fs::write(&path, "x = 1\ny = x\n").unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();
    interaction.client.did_open("document_highlight.py");

    interaction
        .client
        .send_request::<DocumentHighlightRequest>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": 1,
                "character": 4
            }
        }))
        .expect_response(json!([
            {
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": 0, "character": 1 }
                },
                "kind": 3
            },
            {
                "range": {
                    "start": { "line": 1, "character": 4 },
                    "end": { "line": 1, "character": 5 }
                },
                "kind": 2
            }
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}
