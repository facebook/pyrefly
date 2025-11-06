/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::str::FromStr;

use lsp_server::Message;
use lsp_server::Notification;
use lsp_types::Url;
use pyrefly_util::fs_anyhow;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

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

/// Opens a notebook document with the given cell contents.
/// Each string in `cell_contents` becomes a separate code cell in the notebook.
pub fn open_notebook(
    file_name: &str,
    interaction: &mut LspInteraction,
    root: &TempDir,
    cell_contents: Vec<&str>,
) {
    let notebook_path = root.path().to_path_buf().join(file_name);
    let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();

    let mut cells = Vec::new();
    let mut cell_text_documents = Vec::new();

    for (i, text) in cell_contents.iter().enumerate() {
        let cell_uri = Url::from_str(&format!(
            "vscode-notebook-cell://{}#cell{}",
            root.path().to_path_buf().join(file_name).to_string_lossy(),
            i + 1
        ))
        .unwrap()
        .to_string();

        cells.push(serde_json::json!({
            "kind": 2,
            "document": cell_uri,
        }));

        cell_text_documents.push(serde_json::json!({
            "uri": cell_uri,
            "languageId": "python",
            "version": 1,
            "text": *text
        }));
    }

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "notebookDocument/didOpen".to_owned(),
            params: serde_json::json!({
                "notebookDocument": {
                    "uri": notebook_uri,
                    "notebookType": "jupyter-notebook",
                    "version": 1,
                    "metadata": {
                        "language_info": {
                            "name": "python"
                        }
                    },
                    "cells": cells
                },
                "cellTextDocuments": cell_text_documents
            }),
        }));
}
