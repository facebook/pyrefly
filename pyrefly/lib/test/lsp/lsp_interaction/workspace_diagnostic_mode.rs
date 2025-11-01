/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// Test that workspace mode shows errors from unopened files via publishDiagnostics
#[test]
fn test_workspace_mode_shows_unopened_file_errors() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("workspace_diagnostic_mode");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![(
            "workspace_diagnostic_mode".to_owned(),
            scope_uri.clone(),
        )]),
        configuration: Some(Some(serde_json::json!([{
            "pyrefly": {"displayTypeErrors": "force-on"},
            "analysis": {"diagnosticMode": "workspace"}
        }]))),
        ..Default::default()
    });

    // Open file_with_error.py (has 2 errors)
    // In workspace mode, this should trigger publishDiagnostics for ALL workspace files with errors
    interaction.server.did_open("file_with_error.py");

    // The opened file should show its 2 errors
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("file_with_error.py"), 2);

    // The UNOPENED file (opened_file.py) should ALSO show its 1 error via publishDiagnostics
    // This is the key test - workspace mode publishes diagnostics for unopened files!
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("opened_file.py"), 1);

    interaction.shutdown();
}

/// Test that openFilesOnly mode only publishes diagnostics for open files
#[test]
fn test_open_files_only_mode_filters_correctly() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("workspace_diagnostic_mode");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![(
            "workspace_diagnostic_mode".to_owned(),
            scope_uri.clone(),
        )]),
        configuration: Some(Some(serde_json::json!([{
            "pyrefly": {"displayTypeErrors": "force-on"},
            "analysis": {"diagnosticMode": "openFilesOnly"}
        }]))),
        ..Default::default()
    });

    // Open file with errors
    interaction.server.did_open("file_with_error.py");

    // Should show errors for the opened file
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("file_with_error.py"), 2);

    interaction.shutdown();
}

/// Test default behavior (should be openFilesOnly)
#[test]
fn test_default_mode_is_open_files_only() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("workspace_diagnostic_mode");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![(
            "workspace_diagnostic_mode".to_owned(),
            scope_uri.clone(),
        )]),
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    // Open file with errors
    interaction.server.did_open("file_with_error.py");

    // Default mode is openFilesOnly, should show errors for opened file only
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("file_with_error.py"), 2);

    interaction.shutdown();
}

/// Test that workspace mode filters out errors from stdlib/dependencies
#[test]
fn test_workspace_mode_excludes_files_outside_workspace() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("workspace_diagnostic_mode");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![(
            "workspace_diagnostic_mode".to_owned(),
            scope_uri.clone(),
        )]),
        configuration: Some(Some(serde_json::json!([{
            "pyrefly": {"displayTypeErrors": "force-on"},
            "analysis": {"diagnosticMode": "workspace"}
        }]))),
        ..Default::default()
    });

    // Open file with errors
    interaction.server.did_open("file_with_error.py");

    // Should show errors for workspace files only, not stdlib/dependencies
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("file_with_error.py"), 2);

    // Unopened file errors should also be published
    interaction
        .client
        .expect_publish_diagnostics_error_count(root_path.join("opened_file.py"), 1);

    interaction.shutdown();
}
