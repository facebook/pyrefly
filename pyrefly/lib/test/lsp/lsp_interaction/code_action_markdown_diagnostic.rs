/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

/// A `textDocument/codeAction` request whose `context.diagnostics` contain a diagnostic
/// with a markdown message (added in LSP 3.18) must deserialize successfully.
#[test]
fn test_code_action_with_markdown_diagnostic_parses() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("syntax_errors.py");

    let uri = Url::from_file_path(test_files_root.path().join("syntax_errors.py")).unwrap();

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": {
                "diagnostics": [
                    {
                        "code": "unknown-name",
                        "codeDescription": {
                            "href": "https://pyrefly.org/en/docs/error-kinds/#unknown-name"
                        },
                        "data": "committing-transaction",
                        "message": {
                            "kind": "markdown",
                            "value": "Could not find name `l`"
                        },
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 1 }
                        },
                        "severity": 1,
                        "source": "Pyrefly"
                    }
                ],
                "triggerKind": 2,
                "only": ["quickfix"]
            }
        }))
        // The request parses and the server returns a (possibly empty)
        // response rather than an InvalidParams error.
        .expect_response_with(|_response| true)
        .expect("code action request with a markdown diagnostic should parse");

    interaction.shutdown().unwrap();
}
