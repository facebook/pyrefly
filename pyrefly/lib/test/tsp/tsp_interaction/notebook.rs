/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for TSP requests with `vscode-notebook-cell:` URIs.
//!
//! These tests verify that the TSP server handles notebook cell URIs by
//! resolving them to their parent notebook's filesystem path, rather than
//! crashing with "URI must use the file:// scheme".

use tempfile::TempDir;
use tsp_types::TypeKind;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;
use crate::test::tsp::tsp_interaction::object_model::write_pyproject;

/// Set up a project with a notebook containing the given cells.
/// Returns (tsp, cell1_uri_string, snapshot).
fn setup_notebook_project(cell_contents: Vec<&str>) -> (TspInteraction, String, i32) {
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.open_notebook("notebook.ipynb", cell_contents);
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);
    let cell_uri = tsp.server.cell_uri("notebook.ipynb", "cell1").to_string();

    (tsp, cell_uri, snapshot)
}

// =======================================================================
// getComputedType with notebook cell URI
// =======================================================================

#[test]
fn test_get_computed_type_notebook_cell() {
    let (mut tsp, cell_uri, snapshot) = setup_notebook_project(vec!["x = 42\n"]);

    tsp.server.get_computed_type(&cell_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success for notebook cell URI, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type result");

    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .expect("Expected kind field");
    assert_eq!(kind, TypeKind::Class as u64, "Expected Class kind for int");

    tsp.shutdown();
}

// =======================================================================
// getDeclaredType with notebook cell URI
// =======================================================================

#[test]
fn test_get_declared_type_notebook_cell() {
    let (mut tsp, cell_uri, snapshot) = setup_notebook_project(vec!["x: int = 42\n"]);

    tsp.server.get_declared_type(&cell_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success for notebook cell URI, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type result");

    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .expect("Expected kind field");
    assert_eq!(kind, TypeKind::Class as u64, "Expected Class kind for int");

    tsp.shutdown();
}

// =======================================================================
// getExpectedType with notebook cell URI
// =======================================================================

#[test]
fn test_get_expected_type_notebook_cell() {
    let (mut tsp, cell_uri, snapshot) = setup_notebook_project(vec!["y: str = \"hello\"\n"]);

    tsp.server.get_expected_type(&cell_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success for notebook cell URI, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type result");

    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .expect("Expected kind field");
    assert_eq!(kind, TypeKind::Class as u64, "Expected Class kind for str");

    tsp.shutdown();
}

// =======================================================================
// resolveImport with notebook cell URI
// =======================================================================

#[test]
fn test_resolve_import_notebook_cell_absolute() {
    // Resolve `import os` from a notebook cell — should return the typeshed path.
    let (mut tsp, cell_uri, snapshot) = setup_notebook_project(vec!["import os\n"]);

    tsp.server
        .resolve_import(&cell_uri, vec!["os"], 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success for notebook cell URI, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(
        result.is_string(),
        "Expected string URI for 'os', got: {result}"
    );
    let uri_str = result.as_str().unwrap();
    assert!(
        uri_str.contains("os"),
        "Expected resolved URI to contain 'os', got: {uri_str}"
    );

    tsp.shutdown();
}

// =======================================================================
// getPythonSearchPaths with notebook cell URI
// =======================================================================

#[test]
fn test_get_python_search_paths_notebook_cell() {
    // Search paths from a notebook cell should resolve to the notebook's
    // project context, not crash or return an empty list.
    let (mut tsp, cell_uri, snapshot) = setup_notebook_project(vec!["x = 1\n"]);

    tsp.server.get_python_search_paths(&cell_uri, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success for notebook cell URI, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    let paths = result.as_array().expect("Expected array of search paths");
    // Should include at least the typeshed path
    assert!(
        !paths.is_empty(),
        "Expected non-empty search paths for notebook cell"
    );

    tsp.shutdown();
}

// =======================================================================
// Malformed URI still produces an error (not a crash)
// =======================================================================

#[test]
fn test_get_computed_type_malformed_uri_returns_error() {
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Initialize bumps the request counter to 1, so getSnapshot is request 2.
    let snapshot = get_current_snapshot(&mut tsp, 2);

    // A truly malformed URI (not parseable) should return an error response,
    // not crash.
    tsp.server.get_computed_type("not a uri", 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_some(),
        "Expected error for malformed URI, got success: {:?}",
        resp.result
    );

    tsp.shutdown();
}
