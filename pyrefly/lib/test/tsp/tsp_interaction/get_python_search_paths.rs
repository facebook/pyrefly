/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `typeServer/getPythonSearchPaths` TSP request.

use lsp_server::RequestId;
use lsp_types::Url;
use tempfile::TempDir;

use crate::lsp::non_wasm::protocol::Response;
use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;
use crate::test::tsp::tsp_interaction::object_model::write_pyproject;

#[test]
fn test_get_python_search_paths_returns_array() {
    // Verify that getPythonSearchPaths returns a non-empty array of URI strings.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let test_file = temp_dir.path().join("main.py");
    std::fs::write(&test_file, "x = 1\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let from_uri = Url::from_file_path(&test_file).unwrap().to_string();
    tsp.server.get_python_search_paths(&from_uri, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    let paths: Vec<String> = serde_json::from_value(result).expect("Expected array of strings");
    assert!(
        !paths.is_empty(),
        "Expected at least one search path (import root or site-packages)"
    );
    // Every entry should be a valid file:// URI.
    for p in &paths {
        assert!(p.starts_with("file://"), "Expected file:// URI, got: {p}");
    }

    tsp.shutdown();
}

#[test]
fn test_get_python_search_paths_contains_project_root() {
    // With a pyproject.toml the project root directory should appear
    // in the search paths (as the import root).
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let test_file = temp_dir.path().join("main.py");
    std::fs::write(&test_file, "x = 1\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let from_uri = Url::from_file_path(&test_file).unwrap().to_string();
    tsp.server.get_python_search_paths(&from_uri, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    let paths: Vec<String> = serde_json::from_value(result).expect("Expected array of strings");

    // The canonical project root should appear among the search paths.
    let canonical_root = temp_dir.path().canonicalize().unwrap();
    let root_uri = Url::from_file_path(&canonical_root).unwrap().to_string();
    assert!(
        paths.iter().any(|p| p == &root_uri),
        "Expected search paths to contain project root {root_uri}, got: {paths:?}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_python_search_paths_stale_snapshot() {
    // A stale snapshot should return a ServerCancelled error.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let test_file = temp_dir.path().join("main.py");
    std::fs::write(&test_file, "x = 1\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    let from_uri = Url::from_file_path(&test_file).unwrap().to_string();
    // Use snapshot=9999 which is definitely stale.
    tsp.server.get_python_search_paths(&from_uri, 9999);

    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: None,
        error: Some(lsp_server::ResponseError {
            code: lsp_server::ErrorCode::ServerCancelled as i32,
            message: "Snapshot outdated: client sent 9999, server is at 0".to_owned(),
            data: None,
        }),
    });

    tsp.shutdown();
}

#[test]
fn test_get_python_search_paths_invalid_uri() {
    // An invalid URI should return an InvalidParams error.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Snapshot 0 is current (no events processed yet).
    tsp.server.get_python_search_paths("not-a-valid-uri", 0);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(resp.error.is_some(), "Expected error response");
    let err = resp.error.unwrap();
    assert_eq!(err.code, lsp_server::ErrorCode::InvalidParams as i32);

    tsp.shutdown();
}

#[test]
fn test_get_python_search_paths_src_layout() {
    // When source files live under a `src/` subdirectory, the import root
    // heuristic should detect `src/` and include it in the search paths.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let src_dir = temp_dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    let test_file = src_dir.join("main.py");
    std::fs::write(&test_file, "x = 1\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("src/main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let from_uri = Url::from_file_path(&test_file).unwrap().to_string();
    tsp.server.get_python_search_paths(&from_uri, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    let paths: Vec<String> = serde_json::from_value(result).expect("Expected array of strings");

    // The canonical src/ directory should appear among the search paths.
    let canonical_src = src_dir.canonicalize().unwrap();
    let src_uri = Url::from_file_path(&canonical_src).unwrap().to_string();
    assert!(
        paths.iter().any(|p| p == &src_uri),
        "Expected search paths to contain src/ directory {src_uri}, got: {paths:?}"
    );

    tsp.shutdown();
}
