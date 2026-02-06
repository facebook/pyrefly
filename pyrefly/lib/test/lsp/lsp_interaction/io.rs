/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidSaveTextDocument;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_edits_while_recheck() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");
    let path = root.path().join("basic/foo.py");
    // In this test, we trigger didSave and didChange to try to exercise the behavior
    // where we have concurrent in-memory recheck and on-disk recheck.
    interaction
        .client
        .send_notification::<DidSaveTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 1,
                "text": std::fs::read_to_string(path.clone()).unwrap()
            }
        }));

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [
                {"text": format!("{}\n\nextra_stuff", std::fs::read_to_string(&path).unwrap())}
            ],
        }));

    interaction
        .client
        .definition("foo.py", 6, 18)
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9)
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Test that file watcher registration works during initialization.
/// The initialize() function handles file watcher registration automatically
/// when file_watch: true, so we just verify that initialization succeeds.
#[test]
fn test_file_watcher() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());

    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    // File watcher registration is now handled by initialize() when file_watch: true.
    // Additional file watcher behavior tests are in file_watcher.rs.

    interaction.shutdown().unwrap();
}
