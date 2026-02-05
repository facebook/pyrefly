/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Workspace-scoped MRU (most recently used) completion tracking.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use pyrefly_util::fs_anyhow;
use serde::Deserialize;
use serde::Serialize;

const DEFAULT_MAX_ENTRIES: usize = 2000;
const DEFAULT_MAX_AGE_DAYS: u64 = 30;
const MRU_DIR_NAME: &str = ".pyrefly";
const MRU_FILE_NAME: &str = "completion_mru.json";

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct MruEntry {
    #[serde(default)]
    last_used: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct MruFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    entries: HashMap<String, MruEntry>,
}

/// Workspace-scoped MRU tracking for completion items.
#[derive(Clone, Debug)]
pub struct CompletionMru {
    entries: HashMap<String, MruEntry>,
    max_entries: usize,
    max_age_days: u64,
}

impl CompletionMru {
    pub fn new(max_entries: usize, max_age_days: u64) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            max_age_days,
        }
    }

    pub fn record(&mut self, key: &str, now_epoch_secs: u64) {
        self.entries.entry(key.to_owned()).or_default().last_used = now_epoch_secs;
    }

    pub fn last_used(&self, key: &str) -> Option<u64> {
        self.entries.get(key).map(|entry| entry.last_used)
    }

    pub fn to_file(&self) -> MruFile {
        MruFile {
            version: 1,
            entries: self.entries.clone(),
        }
    }

    pub fn merge_from_file(&mut self, file: MruFile) {
        for (key, entry) in file.entries {
            let slot = self.entries.entry(key).or_default();
            slot.last_used = slot.last_used.max(entry.last_used);
        }
    }

    pub fn prune(&mut self, now_epoch_secs: u64) {
        self.prune_age(now_epoch_secs);
        self.prune_count();
    }

    fn prune_age(&mut self, now_epoch_secs: u64) {
        let max_age_secs = self.max_age_days.saturating_mul(24 * 60 * 60);
        if max_age_secs == 0 {
            return;
        }
        self.entries
            .retain(|_, entry| now_epoch_secs.saturating_sub(entry.last_used) <= max_age_secs);
    }

    fn prune_count(&mut self) {
        if self.entries.len() <= self.max_entries {
            return;
        }
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.last_used))
            .collect();
        entries.sort_by(|(_, a), (_, b)| b.cmp(a));

        let keep: HashSet<String> = entries
            .into_iter()
            .take(self.max_entries)
            .map(|(key, _)| key)
            .collect();
        self.entries.retain(|key, _| keep.contains(key));
    }
}

pub fn default_mru() -> CompletionMru {
    CompletionMru::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_AGE_DAYS)
}

pub fn workspace_mru_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(MRU_DIR_NAME).join(MRU_FILE_NAME)
}

pub fn load_from_path(path: &Path, max_entries: usize, max_age_days: u64) -> CompletionMru {
    let mut mru = CompletionMru::new(max_entries, max_age_days);
    let Ok(contents) = fs_anyhow::read_to_string(path) else {
        return mru;
    };
    let Ok(file) = serde_json::from_str::<MruFile>(&contents) else {
        return mru;
    };
    mru.merge_from_file(file);
    mru
}

pub fn load_from_path_default(path: &Path) -> CompletionMru {
    load_from_path(path, DEFAULT_MAX_ENTRIES, DEFAULT_MAX_AGE_DAYS)
}

pub fn save_to_path(mru: &CompletionMru, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs_anyhow::create_dir_all(parent)?;
    }
    let file = mru.to_file();
    let contents = serde_json::to_vec_pretty(&file)?;
    fs_anyhow::write(path, contents)?;
    Ok(())
}
