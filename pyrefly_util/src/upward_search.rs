/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe;
use path_absolutize::Absolutize;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;

use crate::lock::RwLock;

/// A cached way to search the file system upwards for specific files.
pub struct UpwardSearch<T> {
    /// The filenames we are looking for.
    filenames: Vec<OsString>,
    /// The cached state, with previously found entries.
    state: RwLock<SmallMap<PathBuf, Option<T>>>,
    /// Given a config file that exists on disk, load it.
    load: Box<dyn Fn(&Path) -> T + Send + Sync>,
}

impl<T: Dupe + Debug> UpwardSearch<T> {
    /// Given the files you want to find, and a function to load a file
    /// (which must have one of those filenames), create a cached searcher.
    pub fn new(
        filenames: Vec<OsString>,
        load: impl Fn(&Path) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            filenames,
            state: Default::default(),
            load: Box::new(load),
        }
    }

    /// Clear all cached data.
    pub fn clear(&self) {
        self.state.write().clear();
    }

    /// Get the config file associated with a directory.
    pub fn directory(&self, dir: &Path) -> Option<T> {
        self.directory_absolute(dir.absolutize().as_deref().unwrap_or(dir))
    }

    /// Given an already-absolute directory, return the value `T` for it, or `None` if there is not one.
    pub fn directory_absolute(&self, dir: &Path) -> Option<T> {
        // This function iterates through ancestors 3 times, to ensure we don't hold the lock while doing IO.
        // 1. Find the nearest result in the cache (read lock).
        // 2. Find the nearest config on disk and load it (no lock, doing IO).
        // 3. Fill in all the results (write lock).
        const FULL_PATH: usize = 100; // If we want to hit everything in the path

        let mut cache_answer = None; // The (index, result) further up
        let mut found_answer = None;

        let lock = self.state.read();
        for (i, x) in dir.ancestors().enumerate() {
            if let Some(res) = lock.get(x) {
                if i == 0 {
                    // We found a perfect hit
                    return res.dupe();
                } else {
                    cache_answer = Some((i, res.dupe()));
                    break;
                }
            }
        }
        drop(lock);

        // We didn't find a perfect hit, so now search the actual path
        let mut buffer = dir.to_owned();
        'outer: for i in 0..cache_answer.as_ref().map_or(FULL_PATH, |x| x.0) {
            for stem in &self.filenames {
                buffer.push(stem);
                if buffer.exists() {
                    let c = Some((self.load)(&buffer));
                    found_answer = Some((i + 1, c));
                    break 'outer;
                }
                buffer.pop();
            }
            if !buffer.pop() {
                break;
            }
        }

        let (applicable, result) = found_answer
            .or(cache_answer)
            .unwrap_or_else(|| (FULL_PATH, None));

        let mut lock = self.state.write();
        for (i, x) in dir.ancestors().take(applicable).enumerate() {
            match lock.entry(x.to_owned()) {
                Entry::Occupied(e) => {
                    if i == 0 {
                        // Race condition and someone else won on the leaf, so use their result
                        return e.get().dupe();
                    } else {
                        // Someone else filled in some of our prefix, so ours just applies further down
                        return result;
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(result.dupe());
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use starlark_map::small_set::SmallSet;
    use tempfile::TempDir;

    use super::*;
    use crate::lock::Mutex;

    const NONE: &str = "__none__";

    /// Given the files in a directory, and a series of questions, we expect these config files to be loaded
    fn assert(files: &[&str], ask: &[(&str, &str)]) {
        let root = TempDir::new().unwrap();
        for file in files {
            let file_path = root.path().join(file);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&file_path, "").unwrap();
        }

        let loaded = Arc::new(Mutex::new(SmallSet::new()));
        let loaded2 = loaded.dupe();
        let root_path = root.path().to_string_lossy().into_owned();
        let finder = UpwardSearch::new(vec![OsString::from("pyrefly.toml")], move |file| {
            let file = file.to_string_lossy();
            let file = file
                .strip_prefix(&root_path)
                .unwrap()
                .strip_suffix("pyrefly.toml")
                .unwrap()
                .trim_matches(['/', '\\']);
            assert!(
                loaded2.lock().insert(file.to_owned()),
                "Should only load each once"
            );
            Arc::new(file.to_owned())
        });

        for (ask, expect) in ask {
            let res = finder.directory_absolute(&root.path().join(ask));
            assert_eq!(res.as_ref().map_or(NONE, |x| x.as_str()), *expect);
        }
    }

    #[test]
    fn test_config_finder() {
        assert(
            &["hello/pyrefly.toml"],
            &[
                ("hello", "hello"),
                ("hello", "hello"),
                ("", NONE),
                ("other", NONE),
            ],
        );
        assert(
            &["foo/pyrefly.toml", "pyrefly.toml"],
            &[
                ("foo", "foo"),
                ("bar", ""),
                ("foo/bar", "foo"),
                ("bar/magic", ""),
            ],
        );
    }
}
