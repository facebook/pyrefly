/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use lsp_types::Url;
use pyrefly_util::lock::RwLock;

const VIRTUAL_DOCUMENT_ROOT: &str = "__pyrefly_virtual__";

/// Tracks mappings between URIs and generated on-disk-like paths for files that
/// do not yet exist on disk (e.g. untitled buffers).
pub struct UnsavedFileTracker {
    open_file_uris: RwLock<HashMap<PathBuf, Url>>,
    uri_to_path: RwLock<HashMap<Url, PathBuf>>,
    virtual_document_counter: AtomicU32,
}

impl Default for UnsavedFileTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsavedFileTracker {
    pub fn new() -> Self {
        Self {
            open_file_uris: RwLock::new(HashMap::new()),
            uri_to_path: RwLock::new(HashMap::new()),
            virtual_document_counter: AtomicU32::new(0),
        }
    }

    pub fn ensure_path_for_open(&self, uri: &Url, language_id: &str) -> PathBuf {
        if let Some(existing) = self.uri_to_path.read().get(uri).cloned() {
            return existing;
        }

        let counter = self.virtual_document_counter.fetch_add(1, Ordering::SeqCst);
        let mut path = PathBuf::from("/");
        path.push(VIRTUAL_DOCUMENT_ROOT);
        path.push(Self::sanitize_virtual_component(uri.scheme()));

        let raw_candidate = uri.path().trim_matches('/');
        let candidate = if raw_candidate.is_empty() {
            uri.host_str().unwrap_or("")
        } else {
            raw_candidate
        };
        let candidate_path = Path::new(candidate);
        let stem = candidate_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(candidate);
        let sanitized_stem = Self::sanitize_virtual_component(stem);
        let mut file_name = if sanitized_stem.is_empty() {
            format!("document-{counter}")
        } else {
            format!("{sanitized_stem}-{counter}")
        };
        if let Some(ext) = candidate_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Self::sanitize_virtual_component)
        {
            if !ext.is_empty() {
                file_name.push('.');
                file_name.push_str(&ext);
            }
        } else if matches!(language_id, "python") {
            file_name.push_str(".py");
        }
        path.push(file_name);

        self.remember_uri_path(uri, &path);
        path
    }

    pub fn path_for_uri(&self, uri: &Url) -> Option<PathBuf> {
        self.uri_to_path.read().get(uri).cloned()
    }

    pub fn forget_uri_path(&self, uri: &Url) -> Option<PathBuf> {
        let removed = self.uri_to_path.write().remove(uri);
        if let Some(path) = &removed {
            self.open_file_uris.write().remove(path);
        }
        removed
    }

    fn remember_uri_path(&self, uri: &Url, path: &Path) {
        let path_buf = path.to_path_buf();
        self.open_file_uris
            .write()
            .insert(path_buf.clone(), uri.clone());
        self.uri_to_path.write().insert(uri.clone(), path_buf);
    }

    fn sanitize_virtual_component(component: &str) -> String {
        let mut sanitized = component
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        if sanitized.is_empty() {
            sanitized = "untitled".to_owned();
        }
        sanitized
    }
}
