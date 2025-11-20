/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_types::Url;
use lsp_types::request::Completion;
use lsp_types::request::SemanticTokensFullRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

#[test]
fn test_semantic_tokens_for_unsaved_file() {
    let mut interaction = LspInteraction::new();
    interaction.initialize(InitializeSettings::default());

    let uri = Url::parse("untitled:Untitled-1").unwrap();
    let text = r#"def foo():
    return 1

foo()
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(
            RequestId::from(2),
            json!({
                "textDocument": { "uri": uri.to_string() }
            }),
        );

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(data) = result.get("data")
                && let Some(data_array) = data.as_array()
            {
                return !data_array.is_empty();
            }
            false
        },
        "Expected semantic tokens data for unsaved file",
    );

    interaction.shutdown();
}

#[test]
fn test_completion_for_unsaved_file() {
    let mut interaction = LspInteraction::new();
    interaction.initialize(InitializeSettings::default());

    let uri = Url::parse("untitled:Untitled-2").unwrap();
    let text = r#"import math
math.
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    interaction.client.send_request::<Completion>(
        RequestId::from(2),
        json!({
            "textDocument": {"uri": uri.to_string()},
            "position": {"line": 1, "character": 5}
        }),
    );

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result {
                if let Some(items) = result.get("items").and_then(|items| items.as_array()) {
                    return !items.is_empty();
                }
                if let Some(array) = result.as_array() {
                    return !array.is_empty();
                }
            }
            false
        },
        "Expected completion items for unsaved file",
    );

    interaction.shutdown();
}
