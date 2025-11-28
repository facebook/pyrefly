/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;
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

pub struct ExpectedTextEdit<'a> {
    pub new_text: &'a str,
    pub range_start: (u32, u32),
    pub range_end: (u32, u32),
}

pub struct ExpectedInlayHint<'a> {
    pub labels: &'a [&'a str],
    pub position: (u32, u32),
    pub text_edit: ExpectedTextEdit<'a>,
}

pub fn inlay_hints_match_expected(
    result: Option<Vec<InlayHint>>,
    expected: &[ExpectedInlayHint<'_>],
) -> bool {
    let hints = match result {
        Some(hints) => hints,
        None => return false,
    };
    if hints.len() != expected.len() {
        return false;
    }

    for (hint, expected_hint) in hints.iter().zip(expected.iter()) {
        let position = hint.position;
        if position.line != expected_hint.position.0
            || position.character != expected_hint.position.1
        {
            return false;
        }

        let parts = match &hint.label {
            InlayHintLabel::LabelParts(parts) => parts,
            _ => return false,
        };
        if parts.len() != expected_hint.labels.len() {
            return false;
        }
        for (part, expected_value) in parts.iter().zip(expected_hint.labels.iter()) {
            if part.value != *expected_value {
                return false;
            }
            if *expected_value == "Literal" && part.location.is_none() {
                return false;
            }
        }

        let text_edits = match &hint.text_edits {
            Some(edits) if edits.len() == 1 => &edits[0],
            _ => return false,
        };
        if text_edits.new_text != expected_hint.text_edit.new_text {
            return false;
        }
        let start = &expected_hint.text_edit.range_start;
        let end = &expected_hint.text_edit.range_end;
        if text_edits.range.start.line != start.0
            || text_edits.range.start.character != start.1
            || text_edits.range.end.line != end.0
            || text_edits.range.end.character != end.1
        {
            return false;
        }
    }
    true
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
