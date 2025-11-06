/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_util::fs_anyhow;
use tempfile::TempDir;

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
