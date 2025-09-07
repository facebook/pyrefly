/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_unexpected_keyword_range() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_change_configuration();

    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(2, serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}, {"pyrefly": {"displayTypeErrors": "force-on"}}]));

    interaction.server.did_open("unexpected_keyword.py");
    interaction.server.diagnostic("unexpected_keyword.py");

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [
                {
                    "code": "unexpected-keyword",
                    "message": "Unexpected keyword argument `foo` in function `test`",
                    "range": {
                        "end": {"character": 8, "line": 10},
                        "start": {"character": 5, "line": 10}
                    },
                    "severity": 1,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}
