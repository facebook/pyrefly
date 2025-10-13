/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_will_rename_files() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    let foo = "tests_requiring_config/foo.py";
    let bar = "tests_requiring_config/bar.py";
    interaction.server.did_open(foo);
    interaction.server.did_open(bar);

    let bar_path = root.path().join(bar);
    let baz_path = root.path().join("tests_requiring_config/baz.py");
    let foo_path = root.path().join(foo);

    // Send will_rename_files request to rename bar.py to baz.py
    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "workspace/willRenameFiles".to_owned(),
        params: serde_json::json!({
            "files": [{
                "oldUri": Url::from_file_path(&bar_path).unwrap().to_string(),
                "newUri": Url::from_file_path(&baz_path).unwrap().to_string()
            }]
        }),
    }));

    // Expect a response with edits to update imports in foo.py
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "changes": {
                Url::from_file_path(&foo_path).unwrap().to_string(): [
                    {
                        "newText": "baz",
                        "range": {
                            "start": {"line": 5, "character": 7},
                            "end": {"line": 5, "character": 10}
                        }
                    },
                    {
                        "newText": "baz",
                        "range": {
                            "start": {"line": 6, "character": 5},
                            "end": {"line": 6, "character": 8}
                        }
                    }
                ]
            }
        })),
        error: None,
    });

    interaction.shutdown();
}
