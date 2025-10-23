/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// Initialize a test interaction with file watcher enabled.
/// Returns the interaction after consuming the initial file watcher registration.
fn setup_file_watcher_test() -> LspInteraction {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());

    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
        file_watch: true,
        ..Default::default()
    });

    // Consume the initial registration
    interaction.client.expect_file_watcher_register();

    // Acknowledge the registration
    interaction.server.send_message(Message::Response(Response {
        id: RequestId::from(1),
        result: None,
        error: None,
    }));

    // Open a file to start the test
    interaction.server.did_open("text_document.py");

    interaction
}

/// Test that file modifications (metadata-only changes) do NOT trigger watcher re-registration
#[test]
fn test_file_modification_no_reregister() {
    let interaction = setup_file_watcher_test();

    // Send a file change notification (modification only)
    interaction.server.file_modified("text_document.py");

    // Give the server time to process - if it were going to re-register, it would do so quickly
    std::thread::sleep(std::time::Duration::from_millis(500));

    // The test passes if shutdown succeeds without seeing any registration requests
    // The shutdown will consume remaining messages and any unexpected registration
    // would cause the test to hang or fail
    interaction.shutdown();
}

/// Test that file creation DOES trigger watcher re-registration
#[test]
fn test_file_creation_triggers_reregister() {
    let interaction = setup_file_watcher_test();

    // Send a file creation notification
    interaction.server.file_created("new_file.py");

    // Expect unregister then register
    interaction.client.expect_file_watcher_unregister();
    interaction.client.expect_file_watcher_register();

    interaction.shutdown();
}

/// Test that file deletion DOES trigger watcher re-registration
#[test]
fn test_file_deletion_triggers_reregister() {
    let interaction = setup_file_watcher_test();

    // Send a file deletion notification
    interaction.server.file_deleted("text_document.py");

    // Expect unregister then register
    interaction.client.expect_file_watcher_unregister();
    interaction.client.expect_file_watcher_register();

    interaction.shutdown();
}

/// Test that config file changes DOES trigger watcher re-registration
#[test]
fn test_config_file_change_triggers_reregister() {
    let interaction = setup_file_watcher_test();

    // Send a config file change notification
    interaction.server.file_modified("pyrefly.toml");

    // Expect unregister then register
    interaction.client.expect_file_watcher_unregister();
    interaction.client.expect_file_watcher_register();

    interaction.shutdown();
}

/// Test that multiple file modifications (batch) do NOT trigger watcher re-registration
#[test]
fn test_multiple_file_modifications_no_reregister() {
    let interaction = setup_file_watcher_test();

    // Send multiple file change notifications (modifications only)
    interaction.server.file_modified("text_document.py");
    interaction.server.file_modified("utf.py");

    // Give the server time to process - if it were going to re-register, it would do so quickly
    std::thread::sleep(std::time::Duration::from_millis(500));

    // The test passes if shutdown succeeds without seeing any registration requests
    interaction.shutdown();
}

/// Test that mixed events (modification + creation) DOES trigger watcher re-registration
#[test]
fn test_mixed_events_triggers_reregister() {
    let interaction = setup_file_watcher_test();

    // Send mixed file change notifications (modification + creation)
    interaction.server.file_modified("text_document.py");
    interaction.server.file_created("new_file.py");

    // Expect unregister then register (because of creation)
    interaction.client.expect_file_watcher_unregister();
    interaction.client.expect_file_watcher_register();

    interaction.shutdown();
}
