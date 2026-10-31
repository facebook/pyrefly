/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for `pyrefly tsp --config`.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use lsp_types::Uri;
use tempfile::TempDir;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;

/// Lays out a project whose `import libmod` resolves differently under each config:
///
/// ```text
/// project/pyrefly.toml              search-path = ["discovered_libs"]
/// project/discovered_libs/libmod.py
/// project/main.py                   import libmod
/// explicit/pyrefly.toml             search-path = ["../explicit_libs"]
/// explicit_libs/libmod.py
/// ```
///
/// `../explicit_libs` only exists relative to `explicit/`, so resolving there also
/// shows that the explicit config's relative paths are anchored at its own directory.
/// Returns the project directory and the explicit config path.
fn write_projects(root: &Path) -> (PathBuf, PathBuf) {
    let project = root.join("project");
    fs::create_dir_all(project.join("discovered_libs")).unwrap();
    fs::write(
        project.join("pyrefly.toml"),
        "search-path = [\"discovered_libs\"]\n",
    )
    .unwrap();
    fs::write(project.join("discovered_libs/libmod.py"), "x = 1\n").unwrap();
    fs::write(project.join("main.py"), "import libmod\n").unwrap();

    let explicit = root.join("explicit");
    fs::create_dir_all(&explicit).unwrap();
    fs::create_dir_all(root.join("explicit_libs")).unwrap();
    let explicit_config = explicit.join("pyrefly.toml");
    fs::write(&explicit_config, "search-path = [\"../explicit_libs\"]\n").unwrap();
    fs::write(root.join("explicit_libs/libmod.py"), "x = 1\n").unwrap();

    (project, explicit_config)
}

/// Starts a TSP server with the given `--config`, and returns where `import libmod`
/// in `project/main.py` resolves.
fn resolve_libmod(project: &Path, config: Option<PathBuf>) -> String {
    let mut tsp = TspInteraction::with_config(config);
    tsp.set_root(project.to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);
    let source_uri = Uri::from_file_path(project.join("main.py"))
        .unwrap()
        .to_string();
    tsp.server
        .resolve_import(&source_uri, vec!["libmod"], 0, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    let uri = result
        .as_str()
        .unwrap_or_else(|| panic!("Expected `libmod` to resolve, got: {result}"))
        .to_owned();

    tsp.shutdown();
    uri
}

#[test]
fn test_explicit_config_overrides_discovered_config() {
    let temp_dir = TempDir::new().unwrap();
    let (project, explicit_config) = write_projects(temp_dir.path());

    let uri = resolve_libmod(&project, Some(explicit_config));
    assert!(
        uri.ends_with("/explicit_libs/libmod.py"),
        "Expected `libmod` from the explicit config's search path, got: {uri}"
    );
}

#[test]
fn test_without_explicit_config_uses_discovered_config() {
    let temp_dir = TempDir::new().unwrap();
    let (project, _) = write_projects(temp_dir.path());

    let uri = resolve_libmod(&project, None);
    assert!(
        uri.ends_with("/project/discovered_libs/libmod.py"),
        "Expected `libmod` from the discovered config's search path, got: {uri}"
    );
}
