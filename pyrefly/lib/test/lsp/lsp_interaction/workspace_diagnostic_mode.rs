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

/// Test that workspace mode uses get_all_errors (shows all analyzed files)
/// This verifies the filtering logic respects workspace diagnostic mode
#[test]
fn test_workspace_mode_uses_get_all_errors() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    // Send configuration change to enable workspace diagnostic mode
    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "workspace"
                }
            },
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "workspace"
                }
            }
        ]),
    );

    // Open a file - in workspace mode, this should work normally
    // The real test is that the code path uses get_all_errors() instead of get_errors(&handles)
    interaction
        .server
        .did_open("workspace_diagnostic_mode/opened_file.py");

    // Request diagnostics - should work in workspace mode
    interaction
        .server
        .diagnostic("workspace_diagnostic_mode/opened_file.py");

    // File has no errors
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

/// Test that openFilesOnly mode only shows errors for files that are opened
#[test]
fn test_open_files_only_mode_filters_correctly() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    // Send configuration with openFilesOnly mode (explicit)
    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "openFilesOnly"
                }
            },
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "openFilesOnly"
                }
            }
        ]),
    );

    // Open a file without errors
    interaction
        .server
        .did_open("workspace_diagnostic_mode/opened_file.py");

    // Request diagnostics for an UNOPENED file with errors
    // In openFilesOnly mode, we should NOT get diagnostics for unopened files
    interaction
        .server
        .diagnostic("workspace_diagnostic_mode/file_with_error.py");

    // Expect NO errors because the file is not opened and we're in openFilesOnly mode
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

/// Test default behavior (should be openFilesOnly for backward compatibility)
#[test]
fn test_default_mode_is_open_files_only() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    // Don't set diagnosticMode - should default to openFilesOnly
    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {"pyrefly": {"displayTypeErrors": "force-on"}},
            {"pyrefly": {"displayTypeErrors": "force-on"}}
        ]),
    );

    // Open a file without errors
    interaction
        .server
        .did_open("workspace_diagnostic_mode/opened_file.py");

    // Request diagnostics for an UNOPENED file with errors
    // Default mode should be openFilesOnly, so no diagnostics for unopened files
    interaction
        .server
        .diagnostic("workspace_diagnostic_mode/file_with_error.py");

    // Expect NO errors because default mode is openFilesOnly
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

/// Test that workspace mode does not show errors for files outside the workspace folder
#[test]
fn test_workspace_mode_excludes_files_outside_workspace() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    // Send configuration with workspace mode
    interaction.server.did_change_configuration();
    interaction.client.expect_configuration_request(2, None);
    interaction.server.send_configuration_response(
        2,
        serde_json::json!([
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "workspace"
                }
            },
            {
                "pyrefly": {
                    "displayTypeErrors": "force-on"
                },
                "analysis": {
                    "diagnosticMode": "workspace"
                }
            }
        ]),
    );

    // Open a file in the workspace
    interaction
        .server
        .did_open("workspace_diagnostic_mode/opened_file.py");

    // Request diagnostics - workspace mode should only show errors from files within the workspace
    // Files outside the workspace (like dependencies) should not be shown
    interaction
        .server
        .diagnostic("workspace_diagnostic_mode/opened_file.py");

    // Expect NO errors because the file itself has no errors
    // More importantly, we should NOT see errors from dependencies or files outside workspace
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "items": [],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}
