/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for TSP getSnapshot request

use lsp_server::RequestId;
use lsp_server::Response;
use tempfile::TempDir;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;

#[test]
fn test_tsp_get_snapshot() {
    // Test retrieval of TSP snapshot version
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("test.py");

    let test_content = r#"# Simple test file
print("Hello, World!")
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    // Create a pyproject.toml to make this a recognized Python project
    let pyproject_content = r#"[build-system]
requires = ["setuptools>=45", "setuptools-scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "1.0.0"
"#;
    std::fs::write(temp_dir.path().join("pyproject.toml"), pyproject_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Open the test file
    tsp.server.did_open("test.py");

    // Wait for any diagnostics/RecheckFinished events
    tsp.client.expect_any_message();

    // Get snapshot
    tsp.server.get_snapshot();

    // Expect snapshot response with integer (should increment after RecheckFinished from indexing)
    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!(1)),
        error: None,
    });

    tsp.shutdown();
}

#[test]
fn test_tsp_snapshot_updates_on_file_change() {
    // Test that DidChangeWatchedFiles events trigger async recheck and update snapshots
    // With the recheck queue thread running, async tasks should execute and generate RecheckFinished events
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("changing_test.py");

    let initial_content = r#"# Initial content
x = 1
"#;

    std::fs::write(&test_file_path, initial_content).unwrap();

    // Create a pyproject.toml to make this a recognized Python project
    let pyproject_content = r#"[build-system]
requires = ["setuptools>=45", "setuptools-scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "1.0.0"
"#;
    std::fs::write(temp_dir.path().join("pyproject.toml"), pyproject_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Open the test file
    tsp.server.did_open("changing_test.py");

    // Wait for any diagnostics/RecheckFinished events
    tsp.client.expect_any_message();

    // Get initial snapshot
    tsp.server.get_snapshot();

    // Expect first snapshot response
    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!(1)),
        error: None,
    });

    // Modify the file to trigger a state change
    let updated_content = r#"# Updated content
x = 2
y = "hello"
"#;

    std::fs::write(&test_file_path, updated_content).unwrap();

    // Simulate the LSP DidChangeWatchedFiles notification for the file change
    tsp.server
        .did_change_watched_files("changing_test.py", "changed");

    // Wait for the async RecheckFinished event to be processed
    tsp.client.expect_any_message();

    // Get snapshot after async recheck completes
    tsp.server.get_snapshot();

    // Expect snapshot to be incremented to 2 after RecheckFinished from file change
    tsp.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!(2)), // Should be 2 after RecheckFinished from file change
        error: None,
    });

    tsp.shutdown();
}

#[test]
fn test_tsp_snapshot_updates_on_did_change() {
    // Test that didChange events cause snapshot to update
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("change_test.py");

    let initial_content = r#"# Initial content
x = 1
"#;

    std::fs::write(&test_file_path, initial_content).unwrap();

    // Create a pyproject.toml to make this a recognized Python project
    let pyproject_content = r#"[build-system]
requires = ["setuptools>=45", "setuptools-scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "1.0.0"
"#;
    std::fs::write(temp_dir.path().join("pyproject.toml"), pyproject_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Open the test file
    tsp.server.did_open("change_test.py");

    // Wait for any diagnostics/RecheckFinished events from opening
    tsp.client.expect_any_message();

    // Get initial snapshot
    tsp.server.get_snapshot();

    // Expect first snapshot response
    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!(1)),
        error: None,
    });

    // Send a didChange notification with updated content
    let changed_content = r#"# Changed content
x = 2
y = 'updated'
"#;
    tsp.server.did_change("change_test.py", changed_content, 2);

    // Wait for any RecheckFinished events triggered by the change
    tsp.client.expect_any_message();

    // Get updated snapshot
    tsp.server.get_snapshot();

    // Expect second snapshot response - should be incremented due to didChange
    tsp.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!(2)),
        error: None,
    });

    tsp.shutdown();
}
