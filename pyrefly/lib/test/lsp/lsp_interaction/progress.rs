/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::ProgressParams;
use lsp_types::ProgressParamsValue;
use lsp_types::WorkDoneProgress;
use lsp_types::notification::Notification as _;
use lsp_types::notification::Progress;
use lsp_types::request::Request as _;
use lsp_types::request::WorkDoneProgressCreate;
use serde_json::json;

use crate::lsp::non_wasm::protocol::Message;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_work_done_progress_notifications() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            capabilities: Some(json!({
                "window": {"workDoneProgress": true}
            })),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("foo.py");
    interaction.client.did_save("foo.py");

    let (request_id, token) = interaction
        .client
        .expect_message("workDoneProgress/create request", |msg| {
            if let Message::Request(request) = msg
                && request.method == WorkDoneProgressCreate::METHOD
            {
                let params: lsp_types::WorkDoneProgressCreateParams =
                    serde_json::from_value(request.params).unwrap();
                Some(Ok((request.id, params.token)))
            } else {
                None
            }
        })
        .unwrap();

    interaction
        .client
        .send_response::<WorkDoneProgressCreate>(request_id, json!(null));

    interaction
        .client
        .expect_message("$/progress begin", |msg| {
            if let Message::Notification(notification) = msg
                && notification.method == Progress::METHOD
            {
                let params: ProgressParams = serde_json::from_value(notification.params).unwrap();
                if params.token == token {
                    match params.value {
                        ProgressParamsValue::WorkDone(WorkDoneProgress::Begin(_)) => Some(Ok(())),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
