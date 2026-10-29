/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Open documents shared by native and WebAssembly hosts.
//! URI resolution, version ordering policy, and rechecking belong to the host.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use lsp_types::TextDocumentContentChangeEvent;
use pyrefly_util::lock::RwLock;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::SourceLocation;

use crate::state::load::LspFile;

#[derive(Clone)]
pub struct OpenFile {
    pub contents: Arc<LspFile>,
    pub version: i32,
}

#[derive(Default)]
pub struct OpenFiles(RwLock<HashMap<PathBuf, OpenFile>>);

impl OpenFiles {
    pub fn open(&self, path: PathBuf, version: i32, contents: Arc<LspFile>) {
        self.0.write().insert(path, OpenFile { contents, version });
    }

    pub fn close(&self, path: &Path) -> Option<OpenFile> {
        self.close_if(path, |_| true)
    }

    /// Leave the document intact when the host rejects a close notification.
    pub fn close_if(
        &self,
        path: &Path,
        should_close: impl FnOnce(&OpenFile) -> bool,
    ) -> Option<OpenFile> {
        let mut files = self.0.write();
        if !should_close(files.get(path)?) {
            return None;
        }
        files.remove(path)
    }

    pub fn get(&self, path: &Path) -> Option<Arc<LspFile>> {
        self.0
            .read()
            .get(path)
            .map(|file| Arc::clone(&file.contents))
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.0.read().contains_key(path)
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().is_empty()
    }

    pub fn paths(&self) -> Vec<PathBuf> {
        self.0.read().keys().cloned().collect()
    }

    pub fn versions(&self) -> HashMap<PathBuf, i32> {
        self.0
            .read()
            .iter()
            .map(|(path, file)| (path.clone(), file.version))
            .collect()
    }

    /// Copy contents and versions from a single read view.
    pub fn snapshot(&self) -> HashMap<PathBuf, OpenFile> {
        self.0.read().clone()
    }

    /// Replace contents and version together only when `f` succeeds.
    /// The callback holds the write lock and must not access this collection.
    pub fn update(
        &self,
        path: &Path,
        version: i32,
        f: impl FnOnce(&OpenFile) -> anyhow::Result<LspFile>,
    ) -> anyhow::Result<()> {
        let mut files = self.0.write();
        let file = files
            .get_mut(path)
            .ok_or_else(|| anyhow!("File not found in open_files: {}", path.display()))?;
        let contents = f(file)?;
        *file = OpenFile {
            contents: Arc::new(contents),
            version,
        };
        Ok(())
    }
}

impl LspFile {
    /// Apply text edits while preserving the language ID; notebooks require their own updates.
    pub fn apply_text_changes(
        &self,
        changes: Vec<TextDocumentContentChangeEvent>,
    ) -> anyhow::Result<Self> {
        match self {
            Self::Source {
                contents,
                language_id,
            } => Ok(Self::Source {
                contents: Arc::new(apply_change_events(contents, changes)),
                language_id: Arc::clone(language_id),
            }),
            Self::Notebook(_) => Err(anyhow!("Expected text file, but got notebook")),
        }
    }
}

pub fn apply_change_events(original: &str, changes: Vec<TextDocumentContentChangeEvent>) -> String {
    /// Convert lsp_types::Position to usize index for a given text.
    fn position_to_usize(
        position: lsp_types::Position,
        index: &LineIndex,
        source_text: &str,
    ) -> usize {
        let source_location = SourceLocation {
            line: OneIndexed::from_zero_indexed(position.line as usize),
            character_offset: OneIndexed::from_zero_indexed(position.character as usize),
        };
        let text_size = index.offset(
            source_location,
            source_text,
            ruff_source_file::PositionEncoding::Utf16,
        );
        text_size.to_usize()
    }

    let mut result = original.to_owned();
    for change in changes {
        let (range, text) = match change {
            TextDocumentContentChangeEvent::TextDocumentContentChangePartial(partial) => {
                (Some(partial.range), partial.text)
            }
            TextDocumentContentChangeEvent::TextDocumentContentChangeWholeDocument(whole) => {
                (None, whole.text)
            }
        };
        // If no range is given, we can full text replace.
        match range {
            None => result = text,
            Some(range) => {
                let index = LineIndex::from_source_text(&result);
                let start = position_to_usize(range.start, &index, &result);
                let end = position_to_usize(range.end, &index, &result);
                result.replace_range(start..end, &text);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::slice;
    use std::sync::Barrier;
    use std::thread;

    use super::*;
    use crate::lsp::wasm::notebook::NotebookDocument;
    use crate::state::notebook::LspNotebook;

    fn source(text: &str) -> LspFile {
        LspFile::from_source(text.to_owned(), "python".to_owned())
    }

    #[test]
    fn whole_document_changes_are_applied_in_order() {
        let changes = serde_json::from_value(serde_json::json!([
            { "text": "first\n" },
            { "text": "second\r\n" }
        ]))
        .unwrap();
        assert_eq!(apply_change_events("original\n", changes), "second\r\n");
        assert_eq!(
            apply_change_events("unchanged\n", Vec::new()),
            "unchanged\n"
        );
    }

    #[test]
    fn sequential_utf16_changes_preserve_crlf() {
        let changes = serde_json::from_value(serde_json::json!([
            {
                "range": { "start": { "line": 0, "character": 5 }, "end": { "line": 0, "character": 7 } },
                "text": "hello"
            },
            {
                "range": { "start": { "line": 1, "character": 4 }, "end": { "line": 1, "character": 5 } },
                "text": "2"
            },
            {
                "range": { "start": { "line": 0, "character": 5 }, "end": { "line": 0, "character": 10 } },
                "text": "ok"
            }
        ]))
        .unwrap();
        assert_eq!(
            apply_change_events("x = '\u{1f600}'\r\ny = 1\r\n", changes),
            "x = 'ok'\r\ny = 2\r\n"
        );
    }

    #[test]
    fn open_reopen_and_close_keep_contents_and_version_together() {
        let files = OpenFiles::default();
        let path = PathBuf::from("main.py");
        assert!(files.is_empty());
        assert!(files.get(&path).is_none());
        assert!(files.close(&path).is_none());
        files.open(path.clone(), 7, Arc::new(source("old")));
        assert!(files.contains(&path));
        assert_eq!(files.paths().as_slice(), slice::from_ref(&path));
        files.open(path.clone(), 1, Arc::new(source("new")));
        assert_eq!(files.get(&path).unwrap().get_string(), "new");
        assert_eq!(files.versions()[&path], 1);
        let closed = files.close(&path).unwrap();
        assert_eq!(closed.contents.get_string(), "new");
        assert_eq!(closed.version, 1);
        assert!(files.is_empty());
        assert!(files.versions().is_empty());
    }

    #[test]
    fn successful_text_changes_preserve_language_and_accept_host_versions() {
        let files = OpenFiles::default();
        let path = PathBuf::from("main.py");
        let old = Arc::new(source("x = 1\n"));
        files.open(path.clone(), 3, Arc::clone(&old));
        let snapshot = files.snapshot();
        let changes = serde_json::from_value(serde_json::json!([{ "text": "x = 2\n" }])).unwrap();
        files
            .update(&path, 4, |file| file.contents.apply_text_changes(changes))
            .unwrap();
        let file = files.snapshot().remove(&path).unwrap();
        assert_eq!(file.version, 4);
        assert_eq!(file.contents.get_string(), "x = 2\n");
        assert!(
            matches!(&*file.contents, LspFile::Source { language_id, .. } if language_id.as_ref() == "python")
        );
        assert!(Arc::ptr_eq(&snapshot[&path].contents, &old));
        assert_eq!(snapshot[&path].version, 3);
        files
            .update(&path, 1, |file| {
                file.contents.apply_text_changes(Vec::new())
            })
            .unwrap();
        assert_eq!(files.versions()[&path], 1);
    }

    #[test]
    fn failed_updates_preserve_both_fields_and_missing_files_do_not_call_the_callback() {
        let files = OpenFiles::default();
        let path = PathBuf::from("main.py");
        let old = Arc::new(source("unchanged"));
        files.open(path.clone(), 2, Arc::clone(&old));
        let error = files
            .update(&path, 3, |_| Err(anyhow!("Rejected update")))
            .unwrap_err();
        assert_eq!(error.to_string(), "Rejected update");
        assert!(Arc::ptr_eq(&files.get(&path).unwrap(), &old));
        assert_eq!(files.versions()[&path], 2);
        let called = Cell::new(false);
        let missing = PathBuf::from("missing.py");
        assert!(
            files
                .update(&missing, 3, |_| {
                    called.set(true);
                    Ok(source("must not be inserted"))
                })
                .is_err()
        );
        assert!(!called.get());
        assert!(!files.contains(&missing));
        assert_eq!(files.versions().len(), 1);
    }

    #[test]
    fn rejected_close_keeps_the_complete_record() {
        let files = OpenFiles::default();
        let path = PathBuf::from("main.py");
        let old = Arc::new(source("unchanged"));
        files.open(path.clone(), 2, Arc::clone(&old));
        assert!(files.close_if(&path, |_| false).is_none());
        assert!(Arc::ptr_eq(&files.get(&path).unwrap(), &old));
        assert_eq!(files.versions()[&path], 2);
    }

    #[test]
    fn notebook_text_changes_fail_without_publishing_a_version() {
        let notebook: NotebookDocument = serde_json::from_value(serde_json::json!({
            "uri": "file:///workspace/notebook.ipynb",
            "notebookType": "jupyter-notebook",
            "version": 1,
            "cells": []
        }))
        .unwrap();
        let contents = Arc::new(LspFile::Notebook(Arc::new(LspNotebook::new(
            notebook.clone().to_ruff_notebook(&HashMap::new()).unwrap(),
            notebook,
        ))));
        let files = OpenFiles::default();
        let path = PathBuf::from("notebook.ipynb");
        files.open(path.clone(), 1, Arc::clone(&contents));
        let error = files
            .update(&path, 2, |file| {
                file.contents.apply_text_changes(Vec::new())
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "Expected text file, but got notebook");
        assert!(Arc::ptr_eq(&files.get(&path).unwrap(), &contents));
        assert_eq!(files.versions()[&path], 1);
    }

    #[test]
    fn concurrent_snapshots_never_mix_contents_and_versions() {
        let files = OpenFiles::default();
        let path = PathBuf::from("main.py");
        files.open(path.clone(), 0, Arc::new(source("0")));
        let start = Barrier::new(4);
        thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                for version in 1..=100 {
                    files
                        .update(&path, version, |old| {
                            assert_eq!(old.contents.get_string(), old.version.to_string());
                            Ok(source(&version.to_string()))
                        })
                        .unwrap();
                }
            });
            for _ in 0..3 {
                scope.spawn(|| {
                    start.wait();
                    for _ in 0..100 {
                        let snapshot = files.snapshot();
                        let file = &snapshot[&path];
                        assert_eq!(file.contents.get_string(), file.version.to_string());
                    }
                });
            }
        });
        assert_eq!(files.versions()[&path], 100);
    }
}
