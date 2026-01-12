/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SemanticTokensResult;
use lsp_types::Url;
use lsp_types::request::Completion;
use lsp_types::request::SemanticTokensFullRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

#[test]
fn test_semantic_tokens_for_unsaved_file() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-1").unwrap();
    let text = r#"def foo():
    return 1

foo()
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": uri.to_string() }
        }))
        .expect_response_with(|response| match response {
            Some(SemanticTokensResult::Tokens(xs)) => !xs.data.is_empty(),
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_for_unsaved_file() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-2").unwrap();
    let text = r#"import math
math.
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    interaction
        .client
        .send_request::<Completion>(json!({
            "textDocument": {"uri": uri.to_string()},
            "position": {"line": 1, "character": 5}
        }))
        .expect_completion_response_with(|list| !list.items.is_empty())
        .unwrap();

    interaction.shutdown().unwrap();
}
