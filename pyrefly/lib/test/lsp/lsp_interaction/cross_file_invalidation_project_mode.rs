/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentDiagnosticReport;
use lsp_types::DocumentDiagnosticReportResult;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// In-editor `didChange` edits propagate cross-file. After `bar.py` defines
/// `FizzBuzz`, the dependent `foo.py` retains only its unused-import diagnostic.
///
/// Tracking: https://github.com/zed-extensions/pyrefly/issues/19
#[test]
fn test_cross_file_invalidation_in_project_mode() {
    let root = get_test_files_root();
    let root_path = root.path().join("cross_file_invalidation_project_mode");
    let foo_path = root_path.join("foo.py");
    let bar_path = root_path.join("bar.py");
    let bar_contents = std::fs::read_to_string(&bar_path).expect("read bar.py");

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("initialize");

    // foo.py imports a name bar.py doesn't yet define: 2 diagnostics
    // (missing-module-attribute + unused-import).
    interaction.client.did_open("foo.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(foo_path, 2)
        .expect("foo.py should publish 2 diagnostics (missing FizzBuzz + unused import)");

    interaction.client.did_open("bar.py");

    // In-memory edit only: no disk write, didSave, or didChangeWatchedFiles.
    let new_bar_contents = format!("{}\nclass FizzBuzz: pass\n", bar_contents.trim_end());
    interaction.client.did_change("bar.py", &new_bar_contents);

    interaction
        .client
        .diagnostic("foo.py")
        .expect_response_with(|response| {
            let DocumentDiagnosticReportResult::Report(DocumentDiagnosticReport::Full(report)) =
                response
            else {
                return false;
            };
            report.full_document_diagnostic_report.items.len() == 1
        })
        .expect("foo.py should drop to 1 diagnostic after an in-editor didChange");

    interaction.shutdown().unwrap();
}

/// Non-strict LSP clients (VS Code, Helix) send `didSave` plus
/// `workspace/didChangeWatchedFiles` regardless of the server's declared
/// capabilities. The watched-file notification surfaces the on-disk write
/// to pyrefly's filesystem-handle rdep graph, so the dependent `foo.py`'s
/// stale diagnostic clears.
#[test]
fn test_cross_file_invalidation_in_project_mode_with_watched_files() {
    let root = get_test_files_root();
    let root_path = root.path().join("cross_file_invalidation_project_mode");
    let foo_path = root_path.join("foo.py");
    let bar_path = root_path.join("bar.py");
    let bar_contents = std::fs::read_to_string(&bar_path).expect("read bar.py");

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            file_watch: true,
            ..Default::default()
        })
        .expect("initialize");

    interaction.client.did_open("foo.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(foo_path.clone(), 2)
        .expect("foo.py should publish 2 diagnostics (missing FizzBuzz + unused import)");

    interaction.client.did_open("bar.py");

    // Full VSCode-style edit: in-memory change AND on-disk write AND save
    // notification AND watched-files notification. This routes pyrefly
    // through the filesystem-handle invalidation path.
    let new_bar_contents = format!("{}\nclass FizzBuzz: pass\n", bar_contents.trim_end());
    interaction.client.did_change("bar.py", &new_bar_contents);
    std::fs::write(&bar_path, &new_bar_contents).expect("write bar.py");
    interaction.client.file_modified("bar.py");
    interaction.client.did_save("bar.py");

    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(foo_path.clone(), 1)
        .expect("foo.py should drop to 1 diagnostic after watched-file notification");

    interaction.shutdown().unwrap();
}

/// `didSave` alone (without `workspace/didChangeWatchedFiles`) drives
/// cross-file invalidation in project-config mode. This is the path
/// spec-conformant clients like Zed exercise once pyrefly declares
/// `save` in its `TextDocumentSyncOptions` (clients otherwise gate
/// `didSave` delivery per LSP spec).
#[test]
fn test_cross_file_invalidation_in_project_mode_with_did_save_only() {
    let root = get_test_files_root();
    let root_path = root.path().join("cross_file_invalidation_project_mode");
    let foo_path = root_path.join("foo.py");
    let bar_path = root_path.join("bar.py");
    let bar_contents = std::fs::read_to_string(&bar_path).expect("read bar.py");

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .expect("initialize");

    interaction.client.did_open("foo.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(foo_path.clone(), 2)
        .expect("foo.py should publish 2 diagnostics");

    interaction.client.did_open("bar.py");

    // In-memory edit + on-disk write + didSave; NO didChangeWatchedFiles.
    let new_bar_contents = format!("{}\nclass FizzBuzz: pass\n", bar_contents.trim_end());
    interaction.client.did_change("bar.py", &new_bar_contents);
    std::fs::write(&bar_path, &new_bar_contents).expect("write bar.py");
    interaction.client.did_save("bar.py");

    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(foo_path.clone(), 1)
        .expect("foo.py should drop to 1 diagnostic after did_save propagates");

    interaction.shutdown().unwrap();
}
