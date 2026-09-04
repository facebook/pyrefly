/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Coverage for the build-system half of the status-bar payload.
//!
//! The regression these guard against is noise: a source database rebuild is
//! queued on every `didOpen`, so a careless implementation would report build
//! state — and push a refresh notification — to the overwhelming majority of
//! projects, which have no build system configured at all.

use lsp_types::Url;
use lsp_types::notification::Notification as _;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::Message;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::json;

use crate::lsp::non_wasm::type_error_display_status::TypeErrorDisplayStatusChangedNotification;
use crate::lsp::non_wasm::type_error_display_status::TypeErrorDisplayStatusRequest;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// A project with no `[build-system]` must neither report a build status nor
/// push a refresh notification.
///
/// `build_system_blocking` makes the (no-op) source database rebuild run inline
/// during `didOpen` rather than on the queue thread. That is what makes the
/// absence assertion deterministic: anything the rebuild would have sent is
/// necessarily ahead of the status response in the message stream, so reaching
/// that response proves nothing was sent.
#[test]
fn test_no_build_system_is_silent() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::None,
            workspace_indexing_limit: 0,
            build_system_blocking: true,
        },
        ..Default::default()
    });
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            initialization_options: Some(json!({
                "pyrefly": {
                    "typeErrorDisplayStatusVersion": "v2",
                    "pushTypeErrorDisplayStatus": true,
                }
            })),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("foo.py");

    let uri = Url::from_file_path(root_path.join("foo.py")).unwrap();
    let id = interaction
        .client
        .send_request::<TypeErrorDisplayStatusRequest>(json!({ "uri": uri }))
        .id()
        .clone();

    interaction
        .client
        .expect_message("typeErrorDisplayStatus response", |msg| match msg {
            Message::Notification(n)
                if n.method == TypeErrorDisplayStatusChangedNotification::METHOD =>
            {
                panic!(
                    "server pushed a status refresh for a project with no build system; \
                     every non-build-system user would get this on each didOpen"
                )
            }
            Message::Response(r) if r.id == id => {
                let result = r.result.unwrap();
                assert_eq!(
                    result.get("buildSystem"),
                    Some(&serde_json::Value::Null),
                    "expected no build-system status, got: {result}"
                );
                Some(Ok(()))
            }
            _ => None,
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
