/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Build script that generates #[test] functions for each test_*.md file.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn get_test_dir() -> PathBuf {
    match env::var_os("LAZINESS_TEST_PATH") {
        Some(root) => PathBuf::from(root),
        // Cargo: relative to the crate root (pyrefly/pyrefly/)
        None => PathBuf::from("test_laziness"),
    }
}

fn get_output_path() -> PathBuf {
    match env::var_os("OUT") {
        // Buck genrule: OUT is the output directory
        Some(path) => PathBuf::from(path),
        // Cargo: OUT_DIR is set by cargo
        None => PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set")),
    }
}

fn main() {
    let test_dir = get_test_dir();
    let output_dir = get_output_path();
    let output_file = output_dir.join("laziness_tests_generated.rs");

    // Tell cargo to rerun if test files change.
    println!("cargo::rerun-if-changed={}", test_dir.display());

    let mut test_names: Vec<String> = fs::read_dir(&test_dir)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", test_dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "md")
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("test_"))
            {
                let name = path.file_stem()?.to_string_lossy().into_owned();
                Some(name)
            } else {
                None
            }
        })
        .collect();
    test_names.sort();

    let mut code = String::new();
    for name in &test_names {
        code.push_str(&format!("laziness_test!({name});\n"));
    }

    // Write the file, creating parent dirs if needed.
    if let Some(parent) = Path::new(&output_file).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&output_file, code)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", output_file.display()));
}
