/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for TSP snapshotChanged notification

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::Duration;
use std::time::Instant;

use lsp_server::ErrorCode;
use lsp_types::Uri;
use tempfile::TempDir;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;
use crate::test::tsp::tsp_interaction::object_model::write_pyproject;

#[test]
fn test_tsp_snapshot_changed_notification_on_recheck() {
    // After opening a file and triggering an initial recheck, the client
    // should receive a snapshotChanged notification.
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("test.py");
    std::fs::write(&test_file_path, "x = 1\n").unwrap();

    let pyproject = "[project]\nname = \"test-project\"\nversion = \"1.0.0\"\n";
    std::fs::write(temp_dir.path().join("pyproject.toml"), pyproject).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Opening a file eventually triggers RecheckFinished which increments the
    // snapshot and sends snapshotChanged.
    tsp.server.did_open("test.py");

    let params = tsp.client.expect_notification("typeServer/snapshotChanged");
    // The notification params should contain old and new snapshot values.
    let old_snapshot = params["old"].as_i64().expect("old should be an integer");
    let new_snapshot = params["new"].as_i64().expect("new should be an integer");
    // Snapshot starts at 0 in TspServer::new and this is the first event that
    // triggers an increment, so old must be 0.
    assert_eq!(old_snapshot, 0, "old snapshot should be 0 for first change");
    assert!(
        new_snapshot > 0,
        "new snapshot should be positive after recheck"
    );

    tsp.shutdown();
}

#[test]
fn test_tsp_snapshot_changed_notification_on_did_change() {
    // A didChange notification should also trigger snapshotChanged.
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("change.py");
    std::fs::write(&test_file_path, "x = 1\n").unwrap();

    let pyproject = "[project]\nname = \"test-project\"\nversion = \"1.0.0\"\n";
    std::fs::write(temp_dir.path().join("pyproject.toml"), pyproject).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("change.py");

    // Consume the first snapshotChanged from the open/recheck
    tsp.client.expect_notification("typeServer/snapshotChanged");

    // Now send a didChange and expect another snapshotChanged
    tsp.server.did_change("change.py", "x = 2\n", 2);

    let params = tsp.client.expect_notification("typeServer/snapshotChanged");
    let new_snapshot = params["new"].as_i64().expect("new should be an integer");
    assert!(
        new_snapshot > 1,
        "new snapshot should be > 1 after second change"
    );

    tsp.shutdown();
}

/// The name of the computed type of `x` in `x = ...` in `main.py`, or the
/// error code of the response.
fn computed_type_of_x(tsp: &mut TspInteraction, uri: &str, snapshot: i32) -> Result<String, i32> {
    tsp.server.get_computed_type(uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    match (resp.error, resp.result) {
        (Some(error), _) => Err(error.code),
        (None, Some(result)) => Ok(result["declaration"]["name"]
            .as_str()
            .unwrap_or_else(|| panic!("Expected declaration.name in: {result}"))
            .to_owned()),
        (None, None) => panic!("getComputedType returned neither a result nor an error"),
    }
}

/// A close without a save changes the contents of `main.py` from the unsaved
/// text to the text on disk, so the close moves the snapshot.
#[test]
fn test_tsp_close_without_save_moves_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());
    std::fs::write(temp_dir.path().join("main.py"), "x = 1\n").unwrap();
    let uri = Uri::from_file_path(temp_dir.path().join("main.py"))
        .unwrap()
        .to_string();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());
    tsp.server.did_open("main.py");
    tsp.client.expect_notification("typeServer/snapshotChanged");
    tsp.server.did_change("main.py", "x = \"s\"\n", 2);
    tsp.client.expect_notification("typeServer/snapshotChanged");
    let snapshot = get_current_snapshot(&mut tsp, 2);

    assert_eq!(
        computed_type_of_x(&mut tsp, &uri, snapshot),
        Ok("str".to_owned())
    );
    tsp.server.did_close("main.py");
    let params = tsp.client.expect_notification("typeServer/snapshotChanged");
    assert_eq!(params["old"], snapshot);
    assert_eq!(
        computed_type_of_x(&mut tsp, &uri, snapshot),
        Err(ErrorCode::ServerCancelled as i32),
        "the close should move the snapshot"
    );

    tsp.shutdown();
}

/// A recheck commits on another thread before the event loop learns of it.
/// A query in between must not read the new state under the old snapshot.
#[test]
fn test_tsp_recheck_commit_moves_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());
    std::fs::write(temp_dir.path().join("dep.py"), "value = 1\n").unwrap();
    std::fs::write(
        temp_dir.path().join("main.py"),
        "from dep import value\nnumber = value\n",
    )
    .unwrap();
    let uri = Uri::from_file_path(temp_dir.path().join("main.py"))
        .unwrap()
        .to_string();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());
    tsp.server.did_open("main.py");
    tsp.client.expect_notification("typeServer/snapshotChanged");

    std::fs::write(temp_dir.path().join("dep.py"), "value = \"s\"\n").unwrap();
    tsp.server.did_change_watched_files("dep.py", "changed");
    let mut answers: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !answers.values().any(|names| names.contains("str")) {
        assert!(Instant::now() < deadline, "main.py never saw the change");
        tsp.server.get_snapshot();
        let snapshot = serde_json::from_value(
            tsp.client
                .receive_response_skip_notifications()
                .result
                .unwrap(),
        )
        .unwrap();
        tsp.server.get_computed_type(&uri, 1, 9, snapshot);
        let resp = tsp.client.receive_response_skip_notifications();
        match (resp.error, resp.result) {
            (Some(error), _) if error.code == ErrorCode::ServerCancelled as i32 => {}
            (None, Some(result)) => {
                answers
                    .entry(snapshot)
                    .or_default()
                    .insert(result["declaration"]["name"].as_str().unwrap().to_owned());
            }
            (error, result) => panic!("getComputedType failed: {error:?}, {result:?}"),
        }
    }
    assert!(
        answers.values().all(|names| names.len() == 1),
        "each snapshot should give one answer: {answers:?}"
    );

    tsp.shutdown();
}
