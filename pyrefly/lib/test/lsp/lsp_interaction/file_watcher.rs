/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;

use lsp_types::RegistrationParams;
use lsp_types::Url;
use lsp_types::request::RegisterCapability;
use lsp_types::request::Request as _;
use serde::Deserialize;
use tempfile::TempDir;

use crate::lsp::non_wasm::protocol::Message;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::object_model::LspMessageError;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

pub fn expect_watched_files(
    interaction: &LspInteraction,
) -> Result<HashSet<String>, LspMessageError> {
    let params: RegistrationParams = interaction.client.expect_message(
        &format!("Request {}", RegisterCapability::METHOD),
        |msg| {
            if let Message::Request(x) = msg
                && x.method == RegisterCapability::METHOD
            {
                Some(Ok(serde_json::from_value(x.params).unwrap()))
            } else {
                None
            }
        },
    )?;
    assert!(params.registrations.iter().any(|x| x.id == "FILEWATCHER"));
    #[derive(Deserialize)]
    struct Pattern {
        #[serde(rename = "globPattern")]
        glob_pattern: String,
    }
    #[derive(Deserialize)]
    struct Options {
        watchers: Vec<Pattern>,
    }
    let patterns = params
        .registrations
        .into_iter()
        .filter_map(|r| r.register_options)
        .filter_map(|o| serde_json::from_value::<Options>(o).ok())
        .flat_map(|o| o.watchers)
        .map(|w| w.glob_pattern)
        .collect();
    Ok(patterns)
}

/// Initialize a test interaction with file watcher enabled.
/// Returns the TempDir (to keep it alive) and the interaction after consuming
/// the initial file watcher registration.
fn setup_file_watcher_test() -> (TempDir, LspInteraction) {
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

    (root, interaction)
}

/// Test that file watcher registration happens even when no specific patterns are watched.
/// This ensures the server always registers file watchers after initialization.
#[test]
fn test_file_watcher_registered_on_initialization() {
    let (_root, interaction) = setup_file_watcher_test();

    interaction.shutdown().unwrap();
}

/// Test that incremental pattern additions only send register (no unregister first)
/// when the change is small enough.
#[test]
fn test_incremental_pattern_addition() {
    let (_root, interaction) = setup_file_watcher_test();

    // Opening a new file with a new extension shouldn't trigger full re-watch
    // Just an incremental register for new patterns
    interaction.client.did_open("text_document.py");
    let text_document_watched = expect_watched_files(&interaction).unwrap();

    interaction
        .client
        .did_open("imports_builtins/imports_builtins.py");

    // We only watch new files, even though some similar files should be watched.
    let builtins_watched = expect_watched_files(&interaction).unwrap();
    assert!(text_document_watched.is_disjoint(&builtins_watched));

    interaction
        .client
        .did_open("imports_builtins/site-packages/typing.py");

    // Opening a new file with an already opened config watches no new files.
    let new_builtins_watched = expect_watched_files(&interaction).unwrap();
    assert!(new_builtins_watched.is_empty());

    // The test passes if shutdown succeeds without seeing unregister requests
    interaction.shutdown().unwrap();
}
