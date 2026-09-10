/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type queries against files the client never opened.
//!
//! Pylance queries modules it has not sent a `didOpen` for, so these go through
//! the path that solves a module and commits it rather than reading one the
//! editor is already holding in memory.

use lsp_types::Url;
use tempfile::TempDir;
use tsp_types::TypeKind;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::util::get_test_files_root;

/// A project whose open file imports a module the client never opens.
/// Returns the interaction, the directory (kept alive for the test), the
/// unopened module's URI, and the current snapshot.
fn setup_with_unopened_import() -> (TspInteraction, TempDir, String, i32) {
    let test_files = get_test_files_root();
    let root = test_files.path().join("tsp_unopened_files");

    let mut tsp = TspInteraction::new();
    tsp.set_root(root.clone());
    tsp.initialize(Default::default());
    tsp.server.did_open("main.py");
    tsp.client.expect_notification("typeServer/snapshotChanged");

    let snapshot = current_snapshot(&mut tsp);
    let uri = Url::from_file_path(root.join("helper.py"))
        .unwrap()
        .to_string();
    (tsp, test_files, uri, snapshot)
}

/// The server's current snapshot. Unlike the shared helper this does not assert
/// a request id, so tests can ask more than once.
fn current_snapshot(tsp: &mut TspInteraction) -> i32 {
    tsp.server.get_snapshot();
    let resp = tsp.client.receive_response_skip_notifications();
    serde_json::from_value(resp.result.expect("getSnapshot should succeed")).unwrap()
}

fn computed_type(
    tsp: &mut TspInteraction,
    uri: &str,
    line: u32,
    character: u32,
    snapshot: i32,
) -> serde_json::Value {
    tsp.server.get_computed_type(uri, line, character, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    resp.result.expect("Expected a result")
}

/// The name of the class a type result resolves to, e.g. `int`.
fn declaration_name(result: &serde_json::Value) -> &str {
    result
        .get("declaration")
        .and_then(|d| d.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or_else(|| panic!("Expected declaration.name in: {result}"))
}

#[test]
fn test_get_declared_type_on_unopened_import() {
    let (mut tsp, _dir, uri, snapshot) = setup_with_unopened_import();

    tsp.server.get_declared_type(&uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();

    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected a result");
    assert!(
        !result.is_null(),
        "Expected a type for an unopened module, got null"
    );
    let kind = result.get("kind").and_then(|v| v.as_u64());
    assert_eq!(kind, Some(TypeKind::Class as u64), "got: {result}");

    tsp.shutdown();
}

#[test]
fn test_get_expected_type_on_unopened_import() {
    let (mut tsp, _dir, uri, snapshot) = setup_with_unopened_import();

    tsp.server.get_expected_type(&uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();

    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    assert!(
        !resp.result.expect("Expected a result").is_null(),
        "Expected a type for an unopened module, got null"
    );

    tsp.shutdown();
}

#[test]
fn test_unopened_file_query_sees_change_on_disk() {
    // The first query solves the module and commits it. A later edit has to
    // invalidate that, or every subsequent query answers from the stale solve.
    let (mut tsp, test_files, uri, snapshot) = setup_with_unopened_import();
    let root = test_files.path().join("tsp_unopened_files");

    let before = computed_type(&mut tsp, &uri, 0, 0, snapshot);
    assert_eq!(declaration_name(&before), "int", "got: {before}");

    std::fs::write(root.join("helper.py"), "value: str = \"s\"\n").unwrap();
    tsp.server.did_change_watched_files("helper.py", "changed");
    tsp.client.expect_notification("typeServer/snapshotChanged");

    let snapshot = current_snapshot(&mut tsp);
    let after = computed_type(&mut tsp, &uri, 0, 0, snapshot);
    assert_eq!(
        declaration_name(&after),
        "str",
        "the committed solve should have been invalidated, got: {after}"
    );

    tsp.shutdown();
}

#[test]
fn test_opened_file_query_uses_in_memory_contents_after_unopened_query() {
    let (mut tsp, _test_files, uri, snapshot) = setup_with_unopened_import();

    let unopened = computed_type(&mut tsp, &uri, 0, 0, snapshot);
    assert_eq!(declaration_name(&unopened), "int", "got: {unopened}");

    tsp.server.did_open("helper.py");
    tsp.server
        .did_change("helper.py", "value: str = \"memory\"\n", 2);
    tsp.client.expect_notification("typeServer/snapshotChanged");

    let snapshot = current_snapshot(&mut tsp);
    let opened = computed_type(&mut tsp, &uri, 0, 0, snapshot);
    assert_eq!(
        declaration_name(&opened),
        "str",
        "the query should use the open file's in-memory contents, got: {opened}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_on_path_not_in_state() {
    // A path the server has never seen and that does not exist on disk is
    // answered, not dropped or turned into an error.
    let (mut tsp, test_files, _uri, snapshot) = setup_with_unopened_import();

    let missing = test_files
        .path()
        .join("tsp_unopened_files/does_not_exist.py");
    let uri = Url::from_file_path(&missing).unwrap().to_string();
    tsp.server.get_computed_type(&uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();

    assert!(
        resp.error.is_none(),
        "An unknown path should not be an error: {:?}",
        resp.error
    );
    assert!(
        resp.result.expect("Expected a result").is_null(),
        "An unknown path has no type"
    );

    tsp.shutdown();
}
