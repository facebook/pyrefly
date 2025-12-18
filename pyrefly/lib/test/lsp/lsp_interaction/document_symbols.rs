/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbolResponse;
use lsp_types::Url;
use lsp_types::request::DocumentSymbolRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_document_symbols_underscore_prefix() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("prefixed_with_underscore");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    // Open the file with underscore prefix
    interaction.client.did_open("_private.py");

    // Construct the URI for the document symbol request
    let path = test_root.join("_private.py");
    let uri = Url::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = match response {
                Some(DocumentSymbolResponse::Nested(s)) => s,
                _ => return false,
            };

            // Verify the symbols are present
            let has_function = symbols
                .iter()
                .any(|s| s.name == "my_function" && s.kind == lsp_types::SymbolKind::FUNCTION);

            let has_class = symbols
                .iter()
                .any(|s| s.name == "MyClass" && s.kind == lsp_types::SymbolKind::CLASS);

            has_function && has_class
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_document_symbols_normal_file() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("prefixed_with_underscore");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    // Open the normal file (without underscore prefix)
    interaction.client.did_open("normal.py");

    // Construct the URI for the document symbol request
    let path = test_root.join("normal.py");
    let uri = Url::from_file_path(&path).unwrap();

    // Request document symbols - should return symbols for normal files
    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = match response {
                Some(DocumentSymbolResponse::Nested(s)) => s,
                _ => return false,
            };

            // Verify we got symbols back (not empty)
            if symbols.is_empty() {
                return false;
            }

            // Check for the function and class
            let has_function = symbols
                .iter()
                .any(|s| s.name == "normal_function" && s.kind == lsp_types::SymbolKind::FUNCTION);

            let has_class = symbols
                .iter()
                .any(|s| s.name == "NormalClass" && s.kind == lsp_types::SymbolKind::CLASS);

            has_function && has_class
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
