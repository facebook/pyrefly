/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use lsp_types::GotoDefinitionResponse;
use lsp_types::Location;
use pyrefly_util::fs_anyhow;
use tempfile::TempDir;

use crate::module::bundled::BundledStub;
use crate::module::typeshed::typeshed;

pub fn get_test_files_root() -> TempDir {
    let mut source_files =
        std::env::current_dir().expect("std:env::current_dir() unavailable for test");
    let test_files_path = std::env::var("TEST_FILES_PATH")
        .expect("TEST_FILES_PATH env var not set: cargo or buck should set this automatically");
    source_files.push(test_files_path);

    // We copy all files over to a separate temp directory so we are consistent between Cargo and Buck.
    // In particular, given the current directory, Cargo is likely to find a pyproject.toml, but Buck won't.
    let t = TempDir::with_prefix("pyrefly_lsp_test").unwrap();
    copy_dir_recursively(&source_files, t.path());

    t
}

pub fn bundled_typeshed_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(typeshed().unwrap().get_path_name());
    path
}

fn copy_dir_recursively(src: &Path, dst: &Path) {
    if !dst.exists() {
        std::fs::create_dir_all(dst).unwrap();
    }

    for entry in fs_anyhow::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let file_type = entry.file_type().unwrap();
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursively(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap();
        }
    }
}

/// Validates that a goto definition response points to the expected symbol.
/// This is resilient to typeshed changes as it validates behavior (correct symbol)
/// rather than exact position (line/column numbers).
///
/// Reads the content at the returned location and verifies it contains the expected symbol.
pub fn expect_definition_points_to_symbol(
    response: Option<&GotoDefinitionResponse>,
    expected_file_pattern: &str,
    expected_symbol: &str,
) -> bool {
    let response = match response {
        Some(r) => r,
        None => return false,
    };

    let locations: &[Location] = match response {
        GotoDefinitionResponse::Scalar(loc) => std::slice::from_ref(loc),
        GotoDefinitionResponse::Array(locs) => locs,
        GotoDefinitionResponse::Link(_) => return false, // Not expected in our tests
    };

    // Check if any location matches our criteria
    locations.iter().any(|location| {
        // Verify file path contains expected pattern
        let path = match location.uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return false,
        };

        if !path.to_string_lossy().contains(expected_file_pattern) {
            return false;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return false,
        };

        let line_content = match content.lines().nth(location.range.start.line as usize) {
            Some(line) => line,
            None => return false,
        };

        line_content.contains(expected_symbol)
    })
}

/// Helper to read line content at a specific location.
/// Useful for multi-target tests that need to check multiple responses.
pub fn line_at_location(location: &Location) -> Option<String> {
    let path = location.uri.to_file_path().ok()?;
    let content = fs::read_to_string(&path).ok()?;
    content
        .lines()
        .nth(location.range.start.line as usize)
        .map(|s| s.to_owned())
}
