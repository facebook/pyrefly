/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use lsp_types::TextEdit;
use lsp_types::Url;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::lsp::rename::find_word_occurrences;
use crate::state::state::CancellableTransaction;

pub(crate) fn append_comment_and_string_occurrences(
    transaction: &CancellableTransaction<'_>,
    workspace_root: Option<&Path>,
    old_name: &str,
    references: &mut Vec<(ModuleInfo, Vec<TextRange>)>,
) {
    let tx = transaction.as_ref();
    let mut merged = std::mem::take(references);

    for candidate in tx.handles() {
        let Some(module_info) = tx.get_module_info(&candidate) else {
            continue;
        };
        if let Some(root) = workspace_root
            && !candidate.path().as_path().starts_with(root)
        {
            continue;
        }
        if !PYTHON_EXTENSIONS.iter().any(|ext| {
            candidate
                .path()
                .as_path()
                .extension()
                .and_then(|e| e.to_str())
                == Some(*ext)
        }) {
            continue;
        }
        if tx.is_third_party_module(&module_info, &candidate)
            && !tx.is_source_file(&module_info, &candidate)
        {
            continue;
        }
        let comment_ranges = tx.text_occurrences_in_comments_and_strings(&module_info, old_name);
        if comment_ranges.is_empty() {
            continue;
        }
        if let Some((_, ranges)) = merged
            .iter_mut()
            .find(|(existing, _)| existing.path() == module_info.path())
        {
            ranges.extend(comment_ranges);
        } else {
            merged.push((module_info, comment_ranges));
        }
    }

    *references = merged;
}

pub(crate) fn text_occurrence_edits_in_workspace(
    workspace_root: Option<&Path>,
    old_name: &str,
    new_name: &str,
) -> HashMap<Url, Vec<TextEdit>> {
    let Some(root) = workspace_root else {
        return HashMap::new();
    };

    let mut changes = HashMap::new();
    for path in workspace_file_paths(root) {
        if PYTHON_EXTENSIONS
            .iter()
            .any(|ext| path.extension().and_then(|e| e.to_str()) == Some(*ext))
        {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        if bytes.contains(&0) {
            continue;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        let ranges = find_word_occurrences(&text, old_name);
        if ranges.is_empty() {
            continue;
        }
        let buffer = LinedBuffer::new(Arc::new(text));
        let edits = ranges
            .into_iter()
            .map(|range| TextEdit {
                range: buffer.to_lsp_range(range, None),
                new_text: new_name.to_owned(),
            })
            .collect::<Vec<_>>();
        if let Ok(uri) = Url::from_file_path(&path) {
            changes.insert(uri, edits);
        }
    }

    changes
}

fn workspace_file_paths(root: &Path) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };

            if file_type.is_symlink() {
                continue;
            }

            if file_type.is_dir() {
                if should_skip_dir(&path) {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file() {
                files.push(path);
            }
        }
    }

    files
}

fn should_skip_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };

    if name.starts_with('.') {
        return true;
    }

    matches!(
        name,
        "__pycache__" | "venv" | ".venv" | "node_modules" | "target" | "dist" | "build"
    )
}
