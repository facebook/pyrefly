/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `typeServer/resolveImport` TSP request.

use lsp_server::RequestId;
use lsp_types::Url;
use tempfile::TempDir;

use crate::lsp::non_wasm::protocol::Response;
use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;
use crate::test::tsp::tsp_interaction::object_model::write_pyproject;

#[test]
fn test_resolve_import_absolute_stdlib() {
    // Resolve `import os` from a project file — should return the typeshed path.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let test_file = temp_dir.path().join("main.py");
    std::fs::write(&test_file, "import os\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let source_uri = Url::from_file_path(&test_file).unwrap().to_string();
    tsp.server
        .resolve_import(&source_uri, vec!["os"], 0, snapshot);

    // expect_response skips any interleaved notifications.
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
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
        "Expected URI to contain 'os', got: {uri_str}"
    );

    tsp.shutdown();
}

#[test]
fn test_resolve_import_local_module() {
    // Resolve an import to a local project file.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    std::fs::write(temp_dir.path().join("mymodule.py"), "x = 42\n").unwrap();

    let main_path = temp_dir.path().join("main.py");
    std::fs::write(&main_path, "import mymodule\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let source_uri = Url::from_file_path(&main_path).unwrap().to_string();
    tsp.server
        .resolve_import(&source_uri, vec!["mymodule"], 0, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(result.is_string(), "Expected string URI, got: {result}");
    let uri_str = result.as_str().unwrap();
    assert!(
        uri_str.contains("mymodule"),
        "Expected URI to contain 'mymodule', got: {uri_str}"
    );

    tsp.shutdown();
}

#[test]
fn test_resolve_import_nonexistent_module() {
    // Attempting to resolve a module that doesn't exist should return null.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let main_path = temp_dir.path().join("main.py");
    std::fs::write(&main_path, "# empty\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);

    let source_uri = Url::from_file_path(&main_path).unwrap().to_string();
    tsp.server
        .resolve_import(&source_uri, vec!["nonexistent_module_xyz"], 0, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(
        result.is_null(),
        "Expected null for nonexistent module, got: {result}"
    );

    tsp.shutdown();
}

#[test]
fn test_resolve_import_stale_snapshot() {
    // Sending a stale snapshot should yield a ServerCancelled error.
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let main_path = temp_dir.path().join("main.py");
    std::fs::write(&main_path, "# empty\n").unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let source_uri = Url::from_file_path(&main_path).unwrap().to_string();
    // Use snapshot=9999 which is definitely stale.
    tsp.server.resolve_import(&source_uri, vec!["os"], 0, 9999);

    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: None,
        error: Some(lsp_server::ResponseError {
            code: lsp_server::ErrorCode::ServerCancelled as i32,
            message: "Snapshot outdated: client sent 9999, server is at 1".to_owned(),
            data: None,
        }),
    });

    tsp.shutdown();
}
