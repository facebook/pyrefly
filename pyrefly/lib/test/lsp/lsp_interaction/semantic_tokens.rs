/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::SemanticTokens;
use lsp_types::SemanticTokensLegend;
use lsp_types::SemanticTokensResult;
use lsp_types::Url;
use lsp_types::request::SemanticTokensFullRequest;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// Decodes an LSP delta-encoded semantic token stream into `(type_name, token_text)` pairs.
///
/// The LSP semantic token format encodes positions as deltas relative to the
/// previous token.  This helper resolves those deltas and slices the source text
/// so callers can simply filter and assert on plain strings.
fn decode_semantic_tokens(
    tokens: &SemanticTokens,
    legend: &SemanticTokensLegend,
    source: &str,
) -> Vec<(String, String)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut line = 0u32;
    let mut col = 0u32;
    let mut result = Vec::new();
    for token in &tokens.data {
        line += token.delta_line;
        col = if token.delta_line == 0 {
            col + token.delta_start
        } else {
            token.delta_start
        };
        let Some(token_type) = legend.token_types.get(token.token_type as usize) else {
            continue;
        };
        let Some(&line_text) = lines.get(line as usize) else {
            continue;
        };
        let start = col as usize;
        let end = start + token.length as usize;
        let Some(text) = line_text.get(start..end) else {
            continue;
        };
        result.push((token_type.as_str().to_owned(), text.to_owned()));
    }
    result
}

#[test]
fn semantic_tokens_import_submodule_alias() {
    let root = get_test_files_root();
    let root_path = root.path().join("nested_package_imports");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let main_path = root_path.join("main.py");
    let main_text = fs::read_to_string(&main_path).unwrap();
    let main_uri = Url::from_file_path(&main_path).unwrap();

    interaction.client.did_open("main.py");

    let legend = SemanticTokensLegends::lsp_semantic_token_legends();
    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": main_uri.to_string() }
        }))
        .expect_response_with(|response| match response {
            Some(SemanticTokensResult::Tokens(tokens)) => {
                let tokens = decode_semantic_tokens(&tokens, &legend, &main_text);
                let pkg_count = tokens
                    .iter()
                    .filter(|(ty, text)| ty == "namespace" && text == "pkg")
                    .count();
                let sub_count = tokens
                    .iter()
                    .filter(|(ty, text)| ty == "namespace" && text == "sub")
                    .count();
                pkg_count == 1 && sub_count == 2
            }
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn semantic_tokens_format_specifier() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-1").unwrap();
    let text = r#"import logging
logger = logging.getLogger()
logger.info("Hello %s %d", "world", 123)
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    let legend = SemanticTokensLegends::lsp_semantic_token_legends();
    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": uri.to_string() }
        }))
        .expect_response_with(move |response| match response {
            Some(SemanticTokensResult::Tokens(tokens)) => {
                let specifiers: Vec<_> = decode_semantic_tokens(&tokens, &legend, text)
                    .into_iter()
                    .filter(|(ty, _)| ty == "formatSpecifier")
                    .map(|(_, text)| text)
                    .collect();
                specifiers == ["%s", "%d"]
            }
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// `%%` is an escaped literal percent in Python printf-style formatting and must
/// never be highlighted as a format specifier.  `%%%d` should produce exactly one
/// token (`%d`) because the leading `%%` is the escape, not a specifier.
#[test]
fn semantic_tokens_format_specifier_escaped_percent() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-2").unwrap();
    // "100%% done, item %d" — `%%` must produce no token; `%d` must produce one.
    let text = r#"import logging
logger = logging.getLogger()
logger.info("100%% done, item %d", 42)
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    let legend = SemanticTokensLegends::lsp_semantic_token_legends();
    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": uri.to_string() }
        }))
        .expect_response_with(move |response| match response {
            Some(SemanticTokensResult::Tokens(tokens)) => {
                let specifiers: Vec<_> = decode_semantic_tokens(&tokens, &legend, text)
                    .into_iter()
                    .filter(|(ty, _)| ty == "formatSpecifier")
                    .map(|(_, text)| text)
                    .collect();
                // `%%` must not appear; only `%d` is a real specifier.
                specifiers == ["%d"]
            }
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn semantic_tokens_format_specifier_printf() {
    let interaction = LspInteraction::new();
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let uri = Url::parse("untitled:Untitled-3").unwrap();
    // String modulo operator (printf-style) formatting
    let text = r#"
msg = "Hello %s, you have %d messages" % ("Alice", 5)
"#;
    interaction.client.did_open_uri(&uri, "python", text);

    let legend = SemanticTokensLegends::lsp_semantic_token_legends();
    interaction
        .client
        .send_request::<SemanticTokensFullRequest>(json!({
            "textDocument": { "uri": uri.to_string() }
        }))
        .expect_response_with(move |response| match response {
            Some(SemanticTokensResult::Tokens(tokens)) => {
                let specifiers: Vec<_> = decode_semantic_tokens(&tokens, &legend, text)
                    .into_iter()
                    .filter(|(ty, _)| ty == "formatSpecifier")
                    .map(|(_, text)| text)
                    .collect();
                specifiers == ["%s", "%d"]
            }
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
