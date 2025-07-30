/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use lsp_types::FileChangeType;
use notify::EventKind;

#[derive(Debug, Clone, Default)]
pub struct CategorizedEvents {
    pub created: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub unknown: Vec<PathBuf>,
}

impl CategorizedEvents {
    pub fn new_notify(events: Vec<notify::Event>) -> Self {
        let mut res = Self::default();
        for event in events {
            match event.kind {
                EventKind::Create(_) => res.created.extend(event.paths),
                EventKind::Modify(_) => res.modified.extend(event.paths),
                EventKind::Remove(_) => res.removed.extend(event.paths),
                EventKind::Any => res.unknown.extend(event.paths),
                EventKind::Access(_) | EventKind::Other => {}
            }
        }
        res
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_lsp(events: Vec<lsp_types::FileEvent>) -> CategorizedEvents {
        let mut res = CategorizedEvents::default();
        for event in events {
            if let Ok(path) = event.uri.to_file_path() {
                match event.typ {
                    FileChangeType::CREATED => res.created.push(path),
                    FileChangeType::CHANGED => res.modified.push(path),
                    FileChangeType::DELETED => res.removed.push(path),
                    _ => res.unknown.push(path),
                }
            }
        }
        res
    }

    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.modified.is_empty()
            && self.removed.is_empty()
            && self.unknown.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PathBuf> {
        self.created
            .iter()
            .chain(self.modified.iter())
            .chain(self.removed.iter())
            .chain(self.unknown.iter())
    }
}
