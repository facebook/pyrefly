/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for workspace folder handling, particularly edge cases with non-file URIs.

use lsp_types::Url;
use lsp_types::notification::DidChangeWorkspaceFolders;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// Test that the LSP server doesn't crash when receiving workspace folder change
/// notifications with non-file URIs (like vscode-remote://, ssh://, untitled:).
#[test]
fn test_workspace_folder_change_with_non_file_uri_does_not_crash() {
    let test_files_root = get_test_files_root();
    let root = test_files_root.path().join("basic");
    let root_uri = Url::from_file_path(&root).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root.clone());

    // Initialize with a file:// workspace folder and workspace folder support
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), root_uri.clone())]),
            ..Default::default()
        })
        .unwrap();

    // Send a workspace/didChangeWorkspaceFolders notification with a non-file URI.
    // This simulates what happens when a user opens a remote workspace (VS Code Remote SSH,
    // Containers, WSL, etc.) alongside a local workspace.
    let remote_uri = "vscode-remote://ssh-remote+myserver/home/user/project";
    interaction
        .client
        .send_notification::<DidChangeWorkspaceFolders>(json!({
            "event": {
                "added": [{
                    "uri": remote_uri,
                    "name": "remote-project"
                }],
                "removed": []
            }
        }));

    // Send a simple request to verify the server is still alive and responding.
    // If the server crashed due to the non-file URI, this would timeout or fail.
    interaction.client.did_open("foo.py");

    // Wait for diagnostics to confirm the server processed the request
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(root.join("foo.py"), 0)
        .unwrap();

    // Also test removal with non-file URI doesn't crash
    interaction
        .client
        .send_notification::<DidChangeWorkspaceFolders>(json!({
            "event": {
                "added": [],
                "removed": [{
                    "uri": remote_uri,
                    "name": "remote-project"
                }]
            }
        }));

    // Verify server is still responding after the removal
    interaction.client.did_close("foo.py");

    interaction.shutdown().unwrap();
}
