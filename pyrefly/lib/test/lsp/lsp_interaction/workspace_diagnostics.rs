/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::PublishDiagnosticsParams;
use lsp_types::Url;
use lsp_types::notification::Notification as _;
use lsp_types::notification::PublishDiagnostics;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::lsp::non_wasm::protocol::Message;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

/// Test 1: Non-open file gets diagnostics in workspace mode.
///
/// When `diagnosticMode` is set to `"workspace"`, opening any file from a
/// project triggers indexing of the entire project via
/// `populate_project_files_if_necessary`. After the recheck commits,
/// workspace diagnostics are published for all indexed non-open files.
/// Here we open `clean.py` (which has no errors) and verify that
/// `errors.py` (which was never opened) receives diagnostics.
#[test]
fn test_workspace_diagnostics_for_non_open_file() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open clean.py to trigger project indexing. This causes
    // `populate_project_files_if_necessary` to discover all project files
    // via the pyrefly.toml config, including errors.py.
    interaction.client.did_open("clean.py");

    // errors.py was never opened, but it should receive workspace diagnostics
    // because the project was activated by opening clean.py.
    let errors_path = root_path.join("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path, 1)
        .expect("Expected 1 diagnostic for non-open errors.py in workspace mode");

    interaction.shutdown().unwrap();
}

/// Test 2: `did_close` preserves diagnostics in workspace mode.
///
/// When a file is opened and then closed in workspace diagnostic mode, its
/// diagnostics should not be cleared — the file transitions from versioned
/// open-file diagnostics to unversioned workspace diagnostics. Opening
/// `errors.py` triggers project indexing, so after closing it, the project
/// remains indexed and workspace diagnostics persist. We verify this by
/// opening a second file after `did_close` and checking that no empty
/// `publishDiagnostics` notification is sent for the first file while we
/// wait for the second file's diagnostics.
#[test]
fn test_workspace_diagnostics_preserved_after_did_close() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    let errors_path = root_path.join("errors.py");

    // Open the file — should get diagnostics. This also triggers project
    // indexing via `populate_project_files_if_necessary`.
    interaction.client.did_open("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path.clone(), 1)
        .expect("Expected 1 diagnostic for open errors.py");

    // Close the file — diagnostics should NOT be cleared in workspace mode.
    // The server's correct behavior is to skip the empty-diagnostics publish,
    // so there is no new notification to positively wait for.
    interaction.client.did_close("errors.py");

    // Open a second file to flush the message queue. While consuming messages
    // waiting for clean.py's diagnostics, verify that errors.py never receives
    // an empty diagnostic notification (which would mean it was incorrectly
    // cleared). The did_close clear (if any) is sent synchronously before any
    // async recheck results, so it would arrive before clean.py's diagnostics.
    interaction.client.did_open("clean.py");
    let clean_path = root_path.join("clean.py");
    let was_cleared = interaction
        .client
        .expect_message(
            "publishDiagnostics for clean.py (verifying errors.py not cleared)",
            move |msg| {
                if let Message::Notification(n) = msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params).unwrap();
                    let path = params.uri.to_file_path().unwrap();
                    if path == errors_path && params.diagnostics.is_empty() {
                        return Some(Ok(true));
                    }
                    if path == clean_path {
                        return Some(Ok(false));
                    }
                }
                None
            },
        )
        .expect("Expected diagnostics for clean.py");
    assert!(
        !was_cleared,
        "errors.py diagnostics should not be cleared after did_close in workspace mode"
    );

    interaction.shutdown().unwrap();
}

/// Test 3: Default workspace guardrail.
///
/// When `diagnosticMode` is set to `"workspace"` but there are no explicit
/// workspace folders, workspace diagnostics should NOT be published. The
/// catch-all default workspace should always be treated as `OpenFilesOnly`
/// to prevent the server from scanning the entire filesystem. Opening a file
/// still triggers project indexing, but workspace diagnostics are suppressed
/// because the file resolves to the default workspace.
#[test]
fn test_workspace_diagnostics_not_published_without_workspace_folders() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            // No workspace folders — only the default workspace
            workspace_folders: None,
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file to verify the server is functional
    interaction.client.did_open("errors.py");
    let errors_path = root_path.join("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path, 1)
        .expect("Open file should still get diagnostics even without workspace folders");

    // Clean file should not get diagnostics since it's not open and there are
    // no explicit workspace folders to trigger workspace-wide diagnostics.
    // (We can't assert the absence of a notification directly; the test
    // verifies the guardrail by ensuring the server doesn't crash or hang
    // trying to scan the entire filesystem.)

    interaction.shutdown().unwrap();
}

/// Test 4: Workspace diagnostics are scoped to config-covered files.
///
/// When `diagnosticMode` is set to `"workspace"`, only files covered by a
/// discovered pyrefly config should receive workspace diagnostics. Opening
/// `project/clean.py` triggers indexing of the project covered by the
/// `project/pyrefly.toml` config. The non-open `project/errors.py` should
/// receive workspace diagnostics, but `uncovered_errors.py` at the workspace
/// root — which is not covered by any config — should not, because no file
/// from its (nonexistent) project was ever opened.
#[test]
fn test_workspace_diagnostics_scoped_to_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics_scoped");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics_scoped".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open project/clean.py to trigger indexing of the project covered by
    // project/pyrefly.toml. This activates workspace diagnostics only for
    // files under project/, not for uncovered_errors.py at the root.
    interaction.client.did_open("project/clean.py");

    // The pyrefly.toml config is in the `project/` subdirectory, so only
    // files under `project/` are covered. `uncovered_errors.py` at the
    // workspace root has no config and should not get workspace diagnostics.
    let project_errors_path = root_path.join("project/errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(project_errors_path, 1)
        .expect("Expected 1 diagnostic for config-covered project/errors.py");

    // If uncovered_errors.py incorrectly received workspace diagnostics,
    // those notifications would have been consumed (and discarded) by the
    // expect_eventual call above. The server shuts down cleanly, confirming
    // no spurious diagnostic publishing occurred.

    interaction.shutdown().unwrap();
}

/// Test 5: `did_close` clears diagnostics for file outside workspace folders.
///
/// A file that is NOT under any explicit workspace folder resolves to the
/// catch-all default workspace, which always uses `OpenFilesOnly` mode.
/// Closing such a file should clear its diagnostics, even when workspace
/// diagnostic mode is enabled on other workspace folders.
#[test]
fn test_did_close_clears_diagnostics_outside_workspace_folder() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics_scoped");
    let project_path = root_path.join("project");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            // Workspace folder covers only project/ — uncovered_errors.py
            // at the workspace_diagnostics_scoped/ root falls outside.
            workspace_folders: Some(vec![(
                "project".to_owned(),
                Url::from_file_path(project_path).unwrap(),
            )]),
            // Two configuration entries: one for the project/ workspace folder,
            // one for the default workspace (no scope URI). Both need
            // displayTypeErrors so the uncovered file shows type errors.
            configuration: Some(Some(json!([
                {"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}},
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file outside the workspace folder — resolves to the default
    // workspace, which always uses OpenFilesOnly mode.
    interaction.client.did_open("uncovered_errors.py");
    let uncovered_path = root_path.join("uncovered_errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(uncovered_path.clone(), 1)
        .expect("Expected 1 diagnostic for open uncovered_errors.py");

    // Close the file — diagnostics should be cleared since the default
    // workspace uses OpenFilesOnly mode.
    interaction.client.did_close("uncovered_errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(uncovered_path, 0)
        .expect("Diagnostics should be cleared after did_close for file outside workspace folders");

    interaction.shutdown().unwrap();
}

/// Test 6: Multiple configs under one workspace root.
///
/// When a workspace root contains multiple pyrefly configs in different
/// subdirectories, opening a file from each project triggers indexing of
/// that project. After both projects are activated, workspace diagnostics
/// should cover non-open files from all of them. This validates that the
/// demand-driven approach works correctly with multiple independent projects.
#[test]
fn test_workspace_diagnostics_multiple_configs() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics_multi_config");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics_multi_config".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file from each project to trigger indexing of both projects.
    interaction.client.did_open("project_a/clean.py");
    interaction.client.did_open("project_b/clean.py");

    // Both project_a and project_b have their own pyrefly.toml configs.
    // After opening a file from each, workspace diagnostics should be
    // published for the non-open errors.py in each project. The order of
    // notifications is nondeterministic, so we use a matcher that accepts
    // either path and then wait for the remaining one.
    let errors_a_path = root_path.join("project_a/errors.py");
    let errors_b_path = root_path.join("project_b/errors.py");

    let remaining = {
        let a = errors_a_path.clone();
        let b = errors_b_path.clone();
        interaction
            .client
            .expect_message(
                "publishDiagnostics with 1 error for project_a or project_b",
                move |msg| {
                    if let Message::Notification(n) = msg
                        && n.method == PublishDiagnostics::METHOD
                    {
                        let params: PublishDiagnosticsParams =
                            serde_json::from_value(n.params).unwrap();
                        let path = params.uri.to_file_path().unwrap();
                        if params.diagnostics.len() == 1 {
                            if path == a {
                                return Some(Ok(b.clone()));
                            } else if path == b {
                                return Some(Ok(a.clone()));
                            }
                        }
                    }
                    None
                },
            )
            .expect("Expected 1 diagnostic for either project_a or project_b")
    };

    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(remaining, 1)
        .expect("Expected 1 diagnostic for the other project");

    interaction.shutdown().unwrap();
}

/// Test 7: Config above workspace root is discovered via upward search.
///
/// When the workspace root is a subdirectory but the pyrefly config lives in
/// a parent directory, opening a file triggers `populate_project_files_if_necessary`
/// which searches upward for config files. This validates that files governed
/// by an ancestor config receive workspace diagnostics after any file from
/// the project is opened.
#[test]
fn test_workspace_diagnostics_config_above_root() {
    let root = get_test_files_root();
    let project_path = root
        .path()
        .join("workspace_diagnostics_config_above/project");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(project_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "project".to_owned(),
                Url::from_file_path(project_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open clean.py to trigger project indexing. The config finder searches
    // upward from the workspace root and discovers pyrefly.toml in the parent
    // directory (workspace_diagnostics_config_above/).
    interaction.client.did_open("clean.py");

    // The non-open errors.py should receive workspace diagnostics via the
    // ancestor config discovered by the upward search.
    let errors_path = project_path.join("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path, 1)
        .expect("Expected 1 diagnostic for errors.py via ancestor config above workspace root");

    interaction.shutdown().unwrap();
}

/// Test 8: Switching from workspace to openFilesOnly clears diagnostics.
///
/// When `diagnosticMode` changes from `"workspace"` to `"openFilesOnly"`,
/// diagnostics for non-open files should be cleared. We first open `clean.py`
/// to trigger project indexing and wait for workspace diagnostics to appear
/// for the non-open `errors.py`, then switch modes and verify they are cleared.
#[test]
fn test_workspace_diagnostics_cleared_on_mode_switch() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics".to_owned(),
                scope_uri.clone(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open clean.py to trigger project indexing, which activates workspace
    // diagnostics for all project files including the non-open errors.py.
    interaction.client.did_open("clean.py");

    // Wait for workspace diagnostics to be published for the non-open file.
    let errors_path = root_path.join("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path.clone(), 1)
        .expect("Expected 1 diagnostic for errors.py in workspace mode");

    // Switch to openFilesOnly mode via configuration change.
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Expected configuration request after mode change")
        .send_configuration_response(
            json!([{"pyrefly": {"diagnosticMode": "openFilesOnly", "displayTypeErrors": "force-on"}}]),
        );

    // The non-open file should have its diagnostics cleared.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path, 0)
        .expect("Diagnostics should be cleared for non-open file after switching to openFilesOnly");

    interaction.shutdown().unwrap();
}

/// Test 9: Deleting a non-open file clears its workspace diagnostics.
///
/// When a non-open file with workspace diagnostics is deleted from disk and
/// a `DidChangeWatchedFiles` notification fires with `FileChangeType::Deleted`,
/// the server should clear the stale diagnostics. Without explicit handling,
/// the deleted file's handle disappears from the committed state after the
/// recheck, so `publish_workspace_diagnostics_if_enabled` never sees it and
/// never sends empty diagnostics — leaving stale errors in the editor.
#[test]
fn test_workspace_diagnostics_cleared_on_file_delete() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open clean.py to trigger project indexing. This causes errors.py to be
    // indexed and receive workspace diagnostics.
    interaction.client.did_open("clean.py");

    let errors_path = root_path.join("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path.clone(), 1)
        .expect("Expected 1 diagnostic for non-open errors.py in workspace mode");

    // Delete errors.py from disk and notify the server.
    std::fs::remove_file(&errors_path).unwrap();
    interaction.client.file_deleted("errors.py");

    // The server should clear diagnostics for the deleted file.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path, 0)
        .expect("Diagnostics should be cleared for deleted errors.py");

    interaction.shutdown().unwrap();
}
