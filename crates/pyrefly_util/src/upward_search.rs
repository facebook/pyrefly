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
use std::sync::Arc;

use dupe::Dupe;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;

use crate::absolutize::Absolutize as _;
use crate::lock::RwLock;

/// A single filegroup used in an upward search, paired with a predicate on whether
/// the loaded result should be used or the search should continue. This is especially
/// useful when you have one set of filenames that would be preferred, but require
/// introspection on whether the result is actually useful.
///
/// For example, whether a `pyproject.toml` should be considered a configuration file
/// (has a `[tool.pyrefly]` section) or marker file (is probably the root of a project).
pub struct FileGroup<T> {
    /// The filenames we are looking for. Multi-component paths are supported, such as
    /// `.git/info/exclude`.
    filenames: Vec<OsString>,
    /// A predicate denoting wehterh the loaded result from this filegroup should be returned
    /// by the [`UpwardSearch`].
    predicate: Box<dyn Fn(&T) -> bool + Send + Sync>,
}

impl<T> FileGroup<T> {
    /// Creates a new `FileGroup`, searching for the given filenames, and only using
    /// the loaded result if `predicate` returns true.
    pub fn new(
        filenames: Vec<OsString>,
        predicate: impl Fn(&T) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            filenames,
            predicate: Box::new(predicate),
        }
    }

    /// Creates a simple instance of `FileGroup`, whose predicate always returns true.
    pub fn new_simple(filenames: Vec<OsString>) -> Self {
        Self {
            filenames,
            predicate: Box::new(|_| true),
        }
    }
}

/// A cached way to search the file system upwards for specific files.
pub struct UpwardSearch<T> {
    /// The filenames we are looking for. Multi-component paths are supported, such as
    /// `.git/info/exclude`.
    /// Each [`FileGroup`] represents filetypes we should perform
    /// a full upward search for before starting to search for the next group.
    /// If the predicate in a filegroup returns false, the returned result is discarded,
    /// and the search continues.
    filegroups: Vec<FileGroup<T>>,
    /// The cached state, with previously found entries.
    state: RwLock<SmallMap<PathBuf, (Option<T>, Arc<PathBuf>)>>,
    /// Given a config file that exists on disk, load it.
    load: Box<dyn Fn(&Path) -> T + Send + Sync>,
}

impl<T: Dupe + Debug> UpwardSearch<T> {
    /// Given the files you want to find, and a function to load a file
    /// (which must have one of those filenames), create a cached searcher.
    /// This only performs a single upward search, with all files at the same
    /// group priority.
    /// The predicate used always returns true, which means the first file
    /// found is always returned.
    pub fn new(
        filenames: Vec<OsString>,
        load: impl Fn(&Path) -> T + Send + Sync + 'static,
    ) -> Self {
        Self::new_grouped(vec![FileGroup::new_simple(filenames)], load)
    }

    /// Given the files you want to find, and a function to load a file
    /// (which must have one of those filenames), create a cached searcher.
    /// Each [`FileGroup`] represents a set of files at the same priority level, which we will
    /// perform a full upward search for before moving onto the next group for a match.
    /// The associated function determines if the loaded `T` should be used, or if
    /// the upward search should continue/move onto later search groups.
    pub fn new_grouped(
        filegroups: Vec<FileGroup<T>>,
        load: impl Fn(&Path) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            filegroups,
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
        self.directory_absolute(&dir.absolutize())
    }

    /// Given an already-absolute directory, return the value `T` for it, or `None` if there is not one.
    pub fn directory_absolute(&self, dir: &Path) -> Option<T> {
        // This function iterates through ancestors 3 times, to ensure we don't hold the lock while doing IO.
        // 1. Find the nearest result in the cache (read lock).
        // 2. Find the nearest config on disk and load it (no lock, doing IO).
        // 3. Fill in all the results (write lock).
        const FULL_PATH: usize = 100; // If we want to hit everything in the path

        let mut cache_answer = None; // The (index, result, path) further up
        let mut found_answer = None;

        let lock = self.state.read();
        for (i, x) in dir.ancestors().enumerate() {
            if let Some((res, path)) = lock.get(x) {
                if i == 0 {
                    // We found a perfect hit
                    return res.dupe();
                } else {
                    cache_answer = Some((i, res.dupe(), path.dupe()));
                    break;
                }
            }
        }
        drop(lock);

        // We didn't find a perfect hit, so now search the actual path
        'outer: for filegroup in &self.filegroups {
            let mut buffer = dir.to_owned();
            for i in 0..cache_answer.as_ref().map_or(FULL_PATH, |x| x.0) {
                for stem in &filegroup.filenames {
                    let stem_length = PathBuf::from(stem).components().count();
                    buffer.push(stem);
                    if buffer.exists() {
                        let c = (self.load)(&buffer);
                        if (filegroup.predicate)(&c) {
                            found_answer = Some((i + 1, Some(c), Arc::new(buffer)));
                            break 'outer;
                        }
                    }
                    for _ in 0..stem_length {
                        buffer.pop();
                    }
                }
                if !buffer.pop() {
                    break;
                }
            }
            if let Some((i, Some(c), cache_path)) = &cache_answer
                && filegroup.filenames.iter().any(|n| cache_path.ends_with(n))
                && (filegroup.predicate)(c)
            {
                found_answer = Some((i + 1, Some(c.dupe()), cache_path.dupe()));
                break 'outer;
            }
        }

        let (applicable, result, path) = found_answer
            .or(cache_answer)
            .unwrap_or_else(|| (FULL_PATH, None, Arc::new(PathBuf::from("/"))));

        let mut lock = self.state.write();
        for (i, x) in dir.ancestors().take(applicable).enumerate() {
            match lock.entry(x.to_owned()) {
                Entry::Occupied(e) => {
                    if i == 0 {
                        // Race condition and someone else won on the leaf, so use their result
                        return e.get().0.dupe();
                    } else {
                        // Someone else filled in some of our prefix, so ours just applies further down
                        return result;
                    }
                }
                Entry::Vacant(e) => {
                    e.insert((result.dupe(), path.dupe()));
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

    #[test]
    fn test_multi_component_path() {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("a/b/c")).unwrap();
        fs::write(root.path().join("a/b/c/d.test"), "").unwrap();
        fs::create_dir_all(root.path().join("a/e/f/g")).unwrap();

        let upward_search = UpwardSearch::new(vec![OsString::from("a/b/c/d.test")], |p| {
            Arc::new(p.to_path_buf())
        });

        assert_eq!(
            &*upward_search
                .directory_absolute(&root.path().join("a/e/f/g"))
                .unwrap(),
            &root.path().join("a/b/c/d.test")
        );
    }

    #[test]
    fn test_grouped_path() {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("a/b/c/d")).unwrap();
        fs::write(root.path().join("a/b/c/d/mypy.ini"), "").unwrap();
        fs::write(root.path().join("a/b/c/pyrightconfig.json"), "").unwrap();
        fs::write(root.path().join("a/b/pyproject.toml"), "").unwrap();
        fs::write(root.path().join("a/pyrefly.toml"), "").unwrap();

        let upward_search = |trim_front| {
            let groups = [
                vec![OsString::from("pyrefly.toml")],
                vec![OsString::from("pyproject.toml")],
                vec![
                    OsString::from("pyrightconfig.json"),
                    OsString::from("mypy.ini"),
                ],
            ];
            UpwardSearch::new_grouped(
                groups
                    .into_iter()
                    .skip(trim_front)
                    .map(FileGroup::new_simple)
                    .collect(),
                |p| Arc::new(p.to_path_buf()),
            )
        };

        assert_eq!(
            **upward_search(0)
                .directory_absolute(&root.path().join("a/b/c/d"))
                .unwrap(),
            root.path().join("a/pyrefly.toml")
        );
        assert_eq!(
            **upward_search(1)
                .directory_absolute(&root.path().join("a/b/c/d"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml")
        );
        assert_eq!(
            **upward_search(2)
                .directory_absolute(&root.path().join("a/b/c/d"))
                .unwrap(),
            root.path().join("a/b/c/d/mypy.ini")
        );
        assert_eq!(
            **upward_search(0)
                .directory_absolute(&root.path().join("a/b/c"))
                .unwrap(),
            root.path().join("a/pyrefly.toml")
        );
        assert_eq!(
            **upward_search(1)
                .directory_absolute(&root.path().join("a/b/c"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml")
        );
        assert_eq!(
            **upward_search(2)
                .directory_absolute(&root.path().join("a/b/c"))
                .unwrap(),
            root.path().join("a/b/c/pyrightconfig.json")
        );
        assert_eq!(
            **upward_search(0)
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/pyrefly.toml")
        );
        assert_eq!(
            **upward_search(1)
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml")
        );
        assert_eq!(
            upward_search(2).directory_absolute(&root.path().join("a/b")),
            None,
        );
    }

    #[test]
    fn test_predicate_path() {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("a/b")).unwrap();
        fs::write(root.path().join("a/b/pyproject.toml"), "").unwrap();
        fs::write(root.path().join("a/pyrefly.toml"), "").unwrap();
        fs::write(root.path().join("pyproject.toml"), "[tool.pyrefly]").unwrap();

        let pred = |b| -> Box<dyn Fn(&Arc<PathBuf>) -> bool + Send + Sync + 'static> {
            Box::new(move |_| b)
        };

        let upward_search = |p| {
            let groups = vec![
                FileGroup::new(
                    vec![
                        OsString::from("pyrefly.toml"),
                        OsString::from("pyproject.toml"),
                    ],
                    p,
                ),
                FileGroup::new(
                    vec![OsString::from("mypy.ini"), OsString::from("pyproject.toml")],
                    pred(true),
                ),
            ];
            UpwardSearch::new_grouped(groups, |p| Arc::new(p.to_path_buf()))
        };

        fn matches_file<'a>(f: &'a str) -> Box<dyn Fn(&Arc<PathBuf>) -> bool + Send + Sync + 'a> {
            Box::new(move |p| p.file_name().unwrap().to_str().unwrap() == f)
        }

        fn not_empty() -> Box<dyn Fn(&Arc<PathBuf>) -> bool + Send + Sync + 'static> {
            Box::new(|p| !fs::read_to_string(&**p).unwrap().is_empty())
        }

        // tests if simple behavior is the same
        assert_eq!(
            **upward_search(pred(true))
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml"),
        );
        assert_eq!(
            **upward_search(pred(true))
                .directory_absolute(&root.path().join("a"))
                .unwrap(),
            root.path().join("a/pyrefly.toml"),
        );
        assert_eq!(
            **upward_search(pred(true))
                .directory_absolute(root.path())
                .unwrap(),
            root.path().join("pyproject.toml"),
        );

        // tests that we only select files if they match the filename predicate
        assert_eq!(
            **upward_search(matches_file("pyproject.toml"))
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml"),
        );
        assert_eq!(
            **upward_search(matches_file("pyproject.toml"))
                .directory_absolute(&root.path().join("a"))
                .unwrap(),
            root.path().join("pyproject.toml"),
        );
        assert_eq!(
            **upward_search(matches_file("pyrefly.toml"))
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/pyrefly.toml"),
        );
        // tests that we fall back to the 'always return true' group if nothing is found in the
        // first predicate group
        assert_eq!(
            **upward_search(matches_file("does_not_exist"))
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml"),
        );

        // tests that searching for a non-empty file will skip over empty files
        assert_eq!(
            **upward_search(not_empty())
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("pyproject.toml"),
        );
        // tests that removing the file with contents will fall back to the second group
        fs::remove_file(root.path().join("pyproject.toml")).unwrap();
        assert_eq!(
            **upward_search(not_empty())
                .directory_absolute(&root.path().join("a/b"))
                .unwrap(),
            root.path().join("a/b/pyproject.toml"),
        );
        assert!(
            upward_search(not_empty())
                .directory_absolute(&root.path().join("a"))
                .is_none()
        );
    }

    #[test]
    fn test_cached_value_takes_priority_if_group_priority() {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("b")).unwrap();
        fs::write(root.path().join("b/pyproject.toml"), "").unwrap();
        fs::write(root.path().join("pyrefly.toml"), "").unwrap();

        let pred = |b| -> Box<dyn Fn(&Arc<PathBuf>) -> bool + Send + Sync + 'static> {
            Box::new(move |_| b)
        };
        let groups = vec![
            FileGroup::new(vec![OsString::from("pyrefly.toml")], pred(true)),
            FileGroup::new(vec![OsString::from("pyproject.toml")], pred(true)),
        ];
        let upward_search = UpwardSearch::new_grouped(groups, |p| Arc::new(p.to_path_buf()));

        assert_eq!(
            *upward_search.directory(root.path()).unwrap(),
            root.path().join("pyrefly.toml")
        );

        // we should still find pyrefly.toml, since it has a higher group priority,
        // even if pyrefly.toml is cached for a higher directory's result
        assert_eq!(
            *upward_search.directory(&root.path().join("b")).unwrap(),
            root.path().join("pyrefly.toml"),
        );
    }
}
