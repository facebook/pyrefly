/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! In-memory MRU (most recently used) tracking for completion items.
//!
//! This mirrors Pyright's behavior: the MRU list is process-local and not
//! persisted to disk. It resets when the language server restarts.

use std::collections::VecDeque;

const DEFAULT_MAX_ENTRIES: usize = 128;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CompletionMruEntry {
    pub label: String,
    pub auto_import_text: String,
}

#[derive(Clone, Debug)]
pub struct CompletionMru {
    entries: VecDeque<CompletionMruEntry>,
    max_entries: usize,
}

impl CompletionMru {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
        }
    }

    pub fn record(&mut self, label: &str, auto_import_text: &str) {
        let needle = CompletionMruEntry {
            label: label.to_owned(),
            auto_import_text: auto_import_text.to_owned(),
        };
        if let Some(index) = self.entries.iter().position(|entry| entry == &needle) {
            self.entries.remove(index);
        }
        self.entries.push_front(needle);
        if self.entries.len() > self.max_entries {
            self.entries.truncate(self.max_entries);
        }
    }

    pub fn index_for(&self, label: &str, auto_import_text: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.label == label && entry.auto_import_text == auto_import_text)
    }
}

impl Default for CompletionMru {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }
}
