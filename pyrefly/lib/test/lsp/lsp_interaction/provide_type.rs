/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_provide_type_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("bar.py");
    interaction.server.provide_type("bar.py", 7, 5);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": [{
                "kind": "plaintext",
                "value": "typing.Literal[3]",
            }]
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_provide_type() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");
    interaction.server.provide_type("foo.py", 6, 16);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": [{
                "kind": "plaintext",
                "value": "type[bar.Bar]".to_owned()
            }]
        })),
        error: None,
    });

    interaction.shutdown();
}
