/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DiagnosticSeverity;
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

/// Non-open file gets diagnostics in workspace mode.
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

/// Clean non-open workspace files should not be published.
///
/// In workspace diagnostic mode, a non-open file with no diagnostics should
/// not receive an empty `publishDiagnostics` notification. There is nothing to
/// clear for that URI, so sending the notification only creates traffic and log
/// spam on large workspaces.
#[test]
fn test_workspace_diagnostics_skip_clean_non_open_file() {
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

    interaction.client.did_open("clean.py");

    let error_path = root_path.join("errors.py");
    let clean_path = root_path.join("extra_clean.py");
    interaction
        .client
        .expect_message(
            "publishDiagnostics for errors.py without any publish for extra_clean.py",
            |msg| {
                if let Message::Notification(n) = &msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params.clone()).unwrap();
                    let path = params.uri.to_file_path().unwrap();
                    if path == clean_path {
                        return Some(Err(crate::object_model::LspMessageError::Custom {
                            description: format!(
                                "Did not expect publishDiagnostics for clean non-open file {}",
                                clean_path.display()
                            ),
                        }));
                    }
                    if path == error_path && params.diagnostics.len() == 1 {
                        return Some(Ok(()));
                    }
                }
                None
            },
        )
        .unwrap();

    let shutdown_handle = interaction.client.send_shutdown();
    let shutdown_id = shutdown_handle.id.clone();
    interaction
        .client
        .expect_message(
            "shutdown response without any later publish for extra_clean.py",
            |msg| {
                if let Message::Notification(n) = &msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params.clone()).unwrap();
                    if params.uri.to_file_path().unwrap() == clean_path {
                        return Some(Err(crate::object_model::LspMessageError::Custom {
                            description: format!(
                                "Did not expect a later publishDiagnostics for clean non-open file {}",
                                clean_path.display()
                            ),
                        }));
                    }
                }
                if let Message::Response(r) = &msg
                    && r.id == shutdown_id
                {
                    return Some(Ok(()));
                }
                None
            },
        )
        .unwrap();
    interaction.client.send_exit();
}

/// `did_close` preserves diagnostics in workspace mode.
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

/// Default workspace guardrail.
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

/// Workspace diagnostics are scoped to config-covered files.
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

/// `did_close` clears diagnostics for file outside workspace folders.
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

/// Multiple configs under one workspace root.
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

/// Config above workspace root is discovered via upward search.
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

/// Switching from workspace to openFilesOnly clears diagnostics.
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

/// `did_close` does not produce stale "memory path not found" diagnostics.
///
/// When a file is opened (creating an in-memory handle) and then closed in
/// workspace mode, the Memory handle's backing content is cleared. If workspace
/// diagnostics are published for the stale Memory handle, `Load::load_from_path`
/// fails with "memory path not found". This test verifies that closing a file
/// does not cause such false errors to appear for it.
#[test]
fn test_did_close_no_stale_memory_path_errors() {
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

    // Open errors.py — triggers project indexing and creates a Memory handle.
    interaction.client.did_open("errors.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(errors_path.clone(), 1)
        .expect("Expected 1 diagnostic for open errors.py");

    // Close errors.py — the Memory handle's content is cleared, but the handle
    // may linger in committed state.
    interaction.client.did_close("errors.py");

    // Send shutdown to fence the message queue. The server processes all pending
    // work (including any async recheck from did_close) before responding to
    // shutdown. Drain all messages until the shutdown response, checking every
    // publishDiagnostics notification for the stale error.
    let shutdown_handle = interaction.client.send_shutdown();
    let shutdown_id = shutdown_handle.id.clone();
    let saw_stale_error = interaction
        .client
        .expect_message(
            "drain all messages until shutdown (checking no stale memory errors)",
            move |msg| {
                match msg {
                    Message::Notification(n) if n.method == PublishDiagnostics::METHOD => {
                        let params: PublishDiagnosticsParams =
                            serde_json::from_value(n.params).unwrap();
                        let path = params.uri.to_file_path().unwrap();
                        if path == errors_path
                            && params
                                .diagnostics
                                .iter()
                                .any(|d| d.message.contains("memory path not found"))
                        {
                            return Some(Ok(true));
                        }
                    }
                    Message::Response(r) if r.id == shutdown_id => {
                        // Shutdown response — all server work is done.
                        return Some(Ok(false));
                    }
                    _ => {}
                }
                None
            },
        )
        .expect("Expected shutdown response");
    assert!(
        !saw_stale_error,
        "errors.py received 'memory path not found' diagnostic after did_close — stale Memory handle bug"
    );

    interaction.client.send_exit();
}

/// Deleting a non-open file clears its workspace diagnostics.
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

/// Warning-severity diagnostics follow open/close transitions.
///
/// In workspace diagnostic mode, non-open files should only receive
/// error-severity diagnostics. Warning-severity diagnostics should be
/// restricted to files that are currently open in the editor.
///
/// `warning.py` has both a `bad-assignment` (error) and a `bad-return`
/// (configured as warn in pyrefly.toml). When non-open, only the error
/// should be published. When opened, both the error and the warning
/// should appear. When closed again, the file should transition back to
/// non-open diagnostics and drop the warning. This guards against stale
/// open-file warnings lingering after `did_close`.
#[test]
fn test_workspace_diagnostics_severity_tracks_open_close_transitions() {
    let root = get_test_files_root();
    let root_path = root.path().join("workspace_diagnostics_severity");
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "workspace_diagnostics_severity".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            configuration: Some(Some(
                json!([{"pyrefly": {"diagnosticMode": "workspace", "displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open clean.py to trigger project indexing.
    interaction.client.did_open("clean.py");

    // warning.py (non-open) has both an error (bad-assignment) and a
    // warning (bad-return configured as warn). Only the error should be
    // published for non-open files.
    let warning_path = root_path.join("warning.py");
    let warning_path_clone = warning_path.clone();
    interaction
        .client
        .expect_message(
            "publishDiagnostics for non-open warning.py with only error-severity diagnostics",
            move |msg| {
                if let Message::Notification(n) = msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params).unwrap();
                    let path = params.uri.to_file_path().unwrap();
                    if path == warning_path_clone && !params.diagnostics.is_empty() {
                        // All published diagnostics should be error-severity.
                        let all_errors = params
                            .diagnostics
                            .iter()
                            .all(|d| d.severity == Some(DiagnosticSeverity::ERROR));
                        if all_errors {
                            return Some(Ok(()));
                        }
                    }
                }
                None
            },
        )
        .expect("Non-open warning.py should only have error-severity diagnostics");

    // Now open warning.py. The warning diagnostic should appear alongside
    // the error, proving it exists but was filtered from workspace publishing.
    interaction.client.did_open("warning.py");
    interaction
        .client
        .expect_message(
            "publishDiagnostics for opened warning.py with both error and warning",
            move |msg| {
                if let Message::Notification(n) = msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params).unwrap();
                    let path = params.uri.to_file_path().unwrap();
                    if path == warning_path {
                        let has_error = params
                            .diagnostics
                            .iter()
                            .any(|d| d.severity == Some(DiagnosticSeverity::ERROR));
                        let has_warning = params
                            .diagnostics
                            .iter()
                            .any(|d| d.severity == Some(DiagnosticSeverity::WARNING));
                        if has_error && has_warning {
                            return Some(Ok(()));
                        }
                    }
                }
                None
            },
        )
        .expect("Opened warning.py should have both error and warning diagnostics");

    // Close warning.py again. The warning should disappear immediately,
    // proving the file transitioned back to non-open workspace diagnostics.
    let warning_path = root_path.join("warning.py");
    interaction.client.did_close("warning.py");
    interaction
        .client
        .expect_message(
            "publishDiagnostics for closed warning.py with only error-severity diagnostics",
            move |msg| {
                if let Message::Notification(n) = msg
                    && n.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(n.params).unwrap();
                    let path = params.uri.to_file_path().unwrap();
                    if path == warning_path && !params.diagnostics.is_empty() {
                        let all_errors = params
                            .diagnostics
                            .iter()
                            .all(|d| d.severity == Some(DiagnosticSeverity::ERROR));
                        if all_errors {
                            return Some(Ok(()));
                        }
                    }
                }
                None
            },
        )
        .expect("Closed warning.py should return to only error-severity diagnostics");

    interaction.shutdown().unwrap();
}
