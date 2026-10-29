/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Document buffers and text edits shared by native and portable hosts.
//! URI resolution, version ordering policy, and rechecking belong to the host.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use lsp_types::TextDocumentContentChangeEvent;
use lsp_types::VersionedTextDocumentIdentifier;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::SourceLocation;

use crate::state::load::LspFile;

pub type OpenFiles = RwLock<HashMap<PathBuf, Arc<LspFile>>>;
pub type DocumentVersions = Mutex<HashMap<PathBuf, i32>>;

/// Record an opened document and the version supplied by its host.
pub fn open_document(
    files: &OpenFiles,
    versions: &DocumentVersions,
    path: PathBuf,
    version: i32,
    contents: Arc<LspFile>,
) {
    versions.lock().insert(path.clone(), version);
    files.write().insert(path, contents);
}

/// Apply text changes, publishing their version only after the mutation succeeds.
pub fn change_document(
    files: &OpenFiles,
    versions: &DocumentVersions,
    path: &Path,
    document: &VersionedTextDocumentIdentifier,
    changes: Vec<TextDocumentContentChangeEvent>,
) -> anyhow::Result<()> {
    let mut lock = files.write();
    let Some(original) = lock.get_mut(path) else {
        return Err(anyhow!("File not found in open_files: {}", path.display()));
    };
    let language_id = match original.as_ref() {
        LspFile::Source { language_id, .. } => Arc::clone(language_id),
        LspFile::Notebook(_) => {
            return Err(anyhow!(
                "Expected text file for {}, but got notebook",
                document.text_document_identifier.uri
            ));
        }
    };
    *original = Arc::new(LspFile::Source {
        contents: Arc::new(apply_change_events(original.get_string(), changes)),
        language_id,
    });
    drop(lock);
    versions.lock().insert(path.to_owned(), document.version);
    Ok(())
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
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use lsp_types::TextDocumentIdentifier;
    use lsp_types::Uri;
    use lsp_types::VersionedTextDocumentIdentifier;

    use super::*;
    use crate::lsp::wasm::notebook::NotebookDocument;
    use crate::state::notebook::LspNotebook;

    fn document(version: i32) -> VersionedTextDocumentIdentifier {
        VersionedTextDocumentIdentifier {
            text_document_identifier: TextDocumentIdentifier {
                uri: Uri::parse("file:///workspace/main.py").unwrap(),
            },
            version,
        }
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
    fn changes_preserve_language_and_publish_the_supplied_version() {
        let path = PathBuf::from("main.py");
        let files = OpenFiles::default();
        let versions = DocumentVersions::default();
        open_document(
            &files,
            &versions,
            path.clone(),
            3,
            Arc::new(LspFile::from_source(
                "x = 1\n".to_owned(),
                "python".to_owned(),
            )),
        );
        let changes = serde_json::from_value(serde_json::json!([{ "text": "x = 2\n" }])).unwrap();
        change_document(&files, &versions, &path, &document(4), changes).unwrap();
        let file = files.read()[&path].clone();
        assert_eq!(file.get_string(), "x = 2\n");
        assert!(
            matches!(&*file, LspFile::Source { language_id, .. } if language_id.as_ref() == "python")
        );
        assert_eq!(versions.lock()[&path], 4);
        change_document(&files, &versions, &path, &document(1), Vec::new()).unwrap();
        assert_eq!(versions.lock()[&path], 1);
    }

    #[test]
    fn reopening_replaces_contents_and_version() {
        let path = PathBuf::from("main.py");
        let files = OpenFiles::default();
        let versions = DocumentVersions::default();
        for (version, text) in [(7, "old"), (1, "new")] {
            open_document(
                &files,
                &versions,
                path.clone(),
                version,
                Arc::new(LspFile::from_source(text.to_owned(), "python".to_owned())),
            );
        }
        assert_eq!(files.read()[&path].get_string(), "new");
        assert_eq!(versions.lock()[&path], 1);
    }

    #[test]
    fn missing_file_change_does_not_publish_a_version() {
        let path = PathBuf::from("main.py");
        let files = OpenFiles::default();
        let versions = DocumentVersions::default();
        versions.lock().insert(path.clone(), 2);
        let error =
            change_document(&files, &versions, &path, &document(3), Vec::new()).unwrap_err();
        assert_eq!(error.to_string(), "File not found in open_files: main.py");
        assert!(files.read().is_empty());
        assert_eq!(versions.lock()[&path], 2);
    }

    #[test]
    fn text_changes_reject_notebooks_without_mutating_them() {
        let path = PathBuf::from("main.py");
        let notebook = NotebookDocument {
            uri: Uri::parse("file:///workspace/notebook.ipynb").unwrap(),
            notebook_type: "jupyter-notebook".to_owned(),
            version: 1,
            metadata: None,
            cells: Vec::new(),
        };
        let contents = Arc::new(LspFile::Notebook(Arc::new(LspNotebook::new(
            notebook.clone().to_ruff_notebook(&HashMap::new()).unwrap(),
            notebook,
        ))));
        let files = OpenFiles::default();
        let versions = DocumentVersions::default();
        open_document(&files, &versions, path.clone(), 1, contents.clone());
        let error =
            change_document(&files, &versions, &path, &document(2), Vec::new()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Expected text file for file:///workspace/main.py, but got notebook"
        );
        assert!(Arc::ptr_eq(&files.read()[&path], &contents));
        assert_eq!(versions.lock()[&path], 1);
    }
}
