/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::SemanticTokensResult;
use lsp_types::Url;
use lsp_types::request::SemanticTokensFullRequest;
use serde_json::json;

use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

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
                let mut line = 0u32;
                let mut col = 0u32;
                let mut pkg_tokens = 0;
                let mut sub_tokens = 0;
                let lines: Vec<&str> = main_text.lines().collect();
                for token in tokens.data {
                    let delta_line = token.delta_line;
                    let delta_start = token.delta_start;
                    let length = token.length;
                    let token_type = token.token_type;

                    line += delta_line;
                    col = if delta_line == 0 {
                        col + delta_start
                    } else {
                        delta_start
                    };

                    let line_text = match lines.get(line as usize) {
                        Some(line_text) => *line_text,
                        None => continue,
                    };
                    let start = col as usize;
                    let end = start + length as usize;
                    let text = match line_text.get(start..end) {
                        Some(text) => text,
                        None => continue,
                    };
                    let token_type = legend
                        .token_types
                        .get(token_type as usize)
                        .map(|token_type| token_type.as_str())
                        .unwrap_or_default();

                    if text == "pkg" && token_type == "namespace" {
                        pkg_tokens += 1;
                    }
                    if text == "sub" && token_type == "namespace" {
                        sub_tokens += 1;
                    }
                }
                pkg_tokens == 1 && sub_tokens == 2
            }
            _ => false,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
