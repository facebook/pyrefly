/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_text_document_did_change() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("text_document.py");

    let filepath = root.path().join("text_document.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 7, "character": 0}
                },
                "text": format!("{}\n", "rint(\"another change\")")
            }],
        }));

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 3
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 6, "character": 0}
                },
                "text": "p"
            }],
        }));

    interaction
        .client
        .diagnostic("text_document.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_text_document_did_change_unicode() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("utf.py");

    let utf_filepath = root.path().join("utf.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&utf_filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "range": {
                    "start": { "line": 7, "character": 8 },
                    "end": { "line": 8, "character": 2 }
                },
                "rangeLength": 3,
                "text": ""
            }],
        }));

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&utf_filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 3
            },
            "contentChanges": [{
                "range": {
                    "start": { "line": 7, "character": 8 },
                    "end": { "line": 7, "character": 8 }
                },
                "rangeLength": 0,
                "text": format!("\n{}", "print(\"")
            }],
        }));

    interaction
        .client
        .diagnostic("utf.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}
