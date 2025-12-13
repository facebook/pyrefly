/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SemanticTokensResult;
use lsp_types::Url;
use lsp_types::request::SemanticTokensFullRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

#[test]
fn test_semantic_tokens_full_includes_imports() {
    // Regression test for https://github.com/facebook/pyrefly/issues/1811
    // Before the fix, semantic tokens full requests wouldn't include import bindings
    // because the server was initialized with Require::indexing() which doesn't load bindings.
    // The fix in 704baf232bf8bb3e60ed79b6a9a2cd95b92d3759 ensures semantic_tokens_full
    // upgrades to Require::Everything.
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-imports").unwrap();
    let text = r#"from json import decoder
from typing import Literal

x = decoder
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": uri.to_string() }
        }))
        .expect_response(json!({
            "data": [0,5,4,0,0,0,12,7,0,0,1,5,6,0,0,0,14,7,2,512,2,0,1,0,0,0,4,7,0,0]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
