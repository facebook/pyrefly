/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbolRequest;
use lsp_types::DocumentSymbolResponse;
use lsp_types::Uri;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

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
    let uri = Uri::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = match response {
                Some(DocumentSymbolResponse::DocumentSymbolList(s)) => s,
                _ => return false,
            };

            // Verify the symbols are present
            let has_function = symbols
                .iter()
                .any(|s| s.name == "my_function" && s.kind == lsp_types::SymbolKind::Function);

            let has_class = symbols
                .iter()
                .any(|s| s.name == "MyClass" && s.kind == lsp_types::SymbolKind::Class);

            has_function && has_class
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// An empty `disabledLanguageServices` object (which VS Code materializes from
/// its `{}` default and returns whenever the server pulls config) must not
/// disable document symbols, since nothing is actually disabled.
#[test]
fn test_document_symbols_with_empty_disabled_services() {
    let root = get_test_files_root();
    let test_root = root.path().join("prefixed_with_underscore");
    let scope_uri = Uri::from_file_path(test_root.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"disabledLanguageServices": {}}}]));

    interaction.client.did_open("normal.py");

    let path = test_root.join("normal.py");
    let uri = Uri::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = match response {
                Some(DocumentSymbolResponse::DocumentSymbolList(s)) => s,
                _ => return false,
            };
            symbols
                .iter()
                .any(|s| s.name == "normal_function" && s.kind == lsp_types::SymbolKind::Function)
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
    let uri = Uri::from_file_path(&path).unwrap();

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
                Some(DocumentSymbolResponse::DocumentSymbolList(s)) => s,
                _ => return false,
            };

            // Verify we got symbols back (not empty)
            if symbols.is_empty() {
                return false;
            }

            // Check for the function and class
            let has_function = symbols
                .iter()
                .any(|s| s.name == "normal_function" && s.kind == lsp_types::SymbolKind::Function);

            let class_symbol = symbols
                .iter()
                .find(|s| s.name == "NormalClass" && s.kind == lsp_types::SymbolKind::Class);

            let has_class_and_method = match class_symbol {
                Some(c) => c.children.as_ref().is_some_and(|children| {
                    children.iter().any(|s| {
                        s.name == "normal_method" && s.kind == lsp_types::SymbolKind::Method
                    })
                }),
                None => false,
            };

            has_function && has_class_and_method
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// VS Code's Outline asks for document symbols once per document version, so a
/// request cancelled because a configuration change arrived behind it leaves the
/// Outline empty until the next edit.
#[test]
fn test_document_symbols_not_canceled_by_subsequent_configuration_change() {
    let root = get_test_files_root();
    let test_root = root.path().join("basic");
    pyrefly_util::fs_anyhow::write(&test_root.join("symbols.py"), "def f():\n    pass\n")
        .expect("Failed to write the test module");
    let scope_uri = Uri::from_file_path(&test_root).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_root.clone());
    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        configuration: Some(None),
        ..Default::default()
    };
    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction
        .client
        .expect_any_message()
        .expect("Failed to initialize");
    interaction.client.send_initialized();
    let configuration = interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request");

    // The server holds these until the initial settings commit, then replays them
    // in order, so the symbol request is dequeued with a mutation behind it.
    interaction.client.did_open("symbols.py");
    let uri = Uri::from_file_path(test_root.join("symbols.py")).unwrap();
    let symbols = interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": { "uri": uri.to_string() },
        }));
    interaction.client.did_change_configuration();
    configuration.send_configuration_response(json!([{}, {}]));

    symbols
        .expect_response_with(|response: Option<DocumentSymbolResponse>| match response {
            Some(DocumentSymbolResponse::DocumentSymbolList(s)) => s.iter().any(|s| s.name == "f"),
            _ => false,
        })
        .expect("The symbol request should be answered, not canceled");

    interaction.shutdown().expect("Failed to shutdown");
}
