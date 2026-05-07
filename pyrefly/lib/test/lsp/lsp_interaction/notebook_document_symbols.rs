/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbol;
use lsp_types::DocumentSymbolResponse;
use lsp_types::SymbolKind;
use lsp_types::request::DocumentSymbolRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

/// Unwrap a document symbol response into the nested (hierarchical) form.
fn unwrap_nested(response: Option<DocumentSymbolResponse>) -> Vec<DocumentSymbol> {
    match response {
        Some(DocumentSymbolResponse::Nested(symbols)) => symbols,
        other => panic!("expected Nested document symbols, got {other:?}"),
    }
}

#[test]
fn test_notebook_document_symbols() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook(
        "notebook.ipynb",
        vec![
            "class Foo:\n    def bar(self) -> None: ...",
            "def baz() -> int:\n    return 42\n\nx: str = \"hello\"",
        ],
    );

    // Request symbols for cell1 — should contain only Foo (with child method bar)
    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": { "uri": cell1_uri }
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = unwrap_nested(response);
            assert_eq!(symbols.len(), 1, "cell1 should have exactly 1 symbol");
            assert_eq!(symbols[0].name, "Foo");
            assert_eq!(symbols[0].kind, SymbolKind::CLASS);
            let children = symbols[0]
                .children
                .as_ref()
                .expect("Foo should have children");
            assert_eq!(children.len(), 1, "Foo should have exactly 1 child");
            assert_eq!(children[0].name, "bar");
            assert_eq!(children[0].kind, SymbolKind::METHOD);
            true
        })
        .unwrap();

    // Request symbols for cell2 — should contain only baz and x, not Foo
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");
    interaction
        .client
        .send_request::<DocumentSymbolRequest>(json!({
            "textDocument": { "uri": cell2_uri }
        }))
        .expect_response_with(|response: Option<DocumentSymbolResponse>| {
            let symbols = unwrap_nested(response);
            assert_eq!(symbols.len(), 2, "cell2 should have exactly 2 symbols");
            assert_eq!(symbols[0].name, "baz");
            assert_eq!(symbols[0].kind, SymbolKind::FUNCTION);
            assert_eq!(symbols[1].name, "x");
            assert_eq!(symbols[1].kind, SymbolKind::VARIABLE);
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
