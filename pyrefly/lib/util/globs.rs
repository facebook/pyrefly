/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_set::SmallSet;

use crate::util::fs_anyhow;
use crate::util::listing::FileList;
use crate::util::prelude::SliceExt;
use crate::util::prelude::VecExt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, Default)]
pub struct Glob(PathBuf);

impl Glob {
    pub fn new(pattern: String) -> Self {
        Self(PathBuf::from(pattern))
    }

    pub fn new_with_root(root: &Path, pattern: String) -> Self {
        if root == Path::new("") || root == Path::new(".") {
            return Self::new(pattern);
        }

        Self(Self::pattern_relative_to_root(root, Path::new(&pattern)))
    }

    pub fn from_root(self, root: &Path) -> Self {
        Self(Self::pattern_relative_to_root(root, &self.0))
    }

    fn contains_asterisk(part: &OsStr) -> bool {
        part.as_encoded_bytes().contains(&b'*')
    }

    fn pattern_relative_to_root(root: &Path, pattern: &Path) -> PathBuf {
        let mut relative_path = root.to_path_buf();
        for comp in pattern.components() {
            match comp {
                prefix @ Component::Prefix(_) => {
                    // we'll only hit this if we're on windows -- this will fully replace `relative_path`
                    relative_path.push(prefix);
                    match root.components().next() {
                        // if we're on a different drive, give up because we can only hope
                        // this relative path is correct
                        Some(root_prefix @ Component::Prefix(_)) if prefix != root_prefix => (),
                        // otherwise, push the root, and prepare to push the rest of the path
                        _ => relative_path.push(root),
                    }
                }
                rest => relative_path.push(rest),
            }
        }

        relative_path
    }

    fn get_glob_root(&self) -> PathBuf {
        let mut path = PathBuf::new();

        // we need to add any path prefix and root items (there should be at most one of each,
        // and prefix only exists on windows) to the root we're building
        self.0
            .components()
            .take_while(|comp| {
                match comp {
                    // this should be alright to do, since a prefix will always come before a root,
                    // which will always come before the rest of the path
                    Component::Prefix(_)
                    | Component::RootDir
                    | Component::CurDir
                    | Component::ParentDir => true,
                    Component::Normal(part) => !Self::contains_asterisk(part),
                }
            })
            .for_each(|comp| path.push(comp));
        if path.extension().is_some() {
            path.pop();
        }
        path
    }

    fn is_python_extension(ext: Option<&OsStr>) -> bool {
        ext.is_some_and(|e| e == "py" || e == "pyi")
    }

    fn resolve_dir(path: &Path, results: &mut Vec<PathBuf>) -> anyhow::Result<()> {
        for entry in fs_anyhow::read_dir(path)? {
            let entry = entry
                .with_context(|| format!("When iterating over directory `{}`", path.display()))?;
            let path = entry.path();
            if path.is_dir() {
                Self::resolve_dir(&path, results)?;
            } else if Self::is_python_extension(path.extension()) {
                results.push(path);
            }
        }
        Ok(())
    }

    fn resolve_pattern(pattern: &str) -> anyhow::Result<Vec<PathBuf>> {
        let mut result = Vec::new();
        let paths = glob::glob(pattern)?;
        for path in paths {
            let path = path?;
            if path.is_dir() {
                Self::resolve_dir(&path, &mut result)?;
            } else if Self::is_python_extension(path.extension()) {
                result.push(path);
            }
        }
        Ok(result)
    }

    /// Returns true if the given file matches any of the contained globs.
    /// We always attempt to append `**` if a pattern ends in `/` in case
    /// the pattern is meant to be a directory wildcard.
    pub fn matches(&self, file: &Path) -> anyhow::Result<bool> {
        let pattern_path = &self.0;
        let mut pattern_str = pattern_path.to_string_lossy().to_string();
        if pattern_str.ends_with("/") {
            pattern_str.push_str("**");
        }
        let pattern = glob::Pattern::new(&pattern_str)
            .with_context(|| format!("When resolving pattern `{pattern_str}`"))?;
        if pattern.matches_path(file) {
            return Ok(true);
        }
        Ok(false)
    }
}

impl Display for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl FileList for Glob {
    fn files(&self) -> anyhow::Result<Vec<PathBuf>> {
        let pattern = &self.0;
        let pattern_str = pattern.to_string_lossy().to_string();
        let result = Self::resolve_pattern(&pattern_str)
            .with_context(|| format!("When resolving pattern `{pattern_str}`"))?;
        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "No files matched pattern `{}`",
                pattern_str
            ));
        }
        Ok(result)
    }

    fn covers(&self, path: &Path) -> bool {
        self.matches(path).unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, Default)]
pub struct Globs(Vec<Glob>);

impl Globs {
    /// Create a new `Globs` from the given patterns. If you want them to be relative
    /// to a root, please use `Globs::new_with_root()` instead.
    pub fn new(patterns: Vec<String>) -> Self {
        Self(patterns.into_map(Glob::new))
    }

    /// Create a new `Globs`, rewriting all patterns to be relative to `root`.
    pub fn new_with_root(root: &Path, patterns: Vec<String>) -> Self {
        if root == Path::new("") || root == Path::new(".") {
            return Self::new(patterns);
        }

        Self::rewrite_with_root(root, patterns.into_map(Glob::new))
    }

    fn rewrite_with_root(root: &Path, patterns: Vec<Glob>) -> Self {
        Self(
            patterns
                .into_iter()
                .map(|pattern| pattern.from_root(root))
                .collect(),
        )
    }

    pub fn from_root(self, root: &Path) -> Self {
        // TODO(connernilsen): store root as part of globs to make it easier to rewrite later on
        Self::rewrite_with_root(root, self.0)
    }

    /// Given a glob pattern, return the directories that can contain files that match the pattern.
    pub fn roots(&self) -> Vec<PathBuf> {
        self.0.map(|s| s.get_glob_root())
    }

    /// Returns true if the given file matches any of the contained globs.
    /// We always attempt to append `**` if a pattern ends in `/` in case
    /// the pattern is meant to be a directory wildcard.
    fn matches(&self, file: &Path) -> anyhow::Result<bool> {
        for pattern in &self.0 {
            if pattern.matches(file)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Display for Globs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|g| g.to_string()).join(", "))
    }
}

impl FileList for Globs {
    fn files(&self) -> anyhow::Result<Vec<PathBuf>> {
        let mut result = SmallSet::new();
        for pattern in &self.0 {
            result.extend(pattern.files()?);
        }
        Ok(result.into_iter().collect())
    }

    fn covers(&self, path: &Path) -> bool {
        self.matches(path).unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct FilteredGlobs {
    includes: Globs,
    excludes: Globs,
}

impl FilteredGlobs {
    pub fn new(includes: Globs, excludes: Globs) -> Self {
        Self { includes, excludes }
    }

    /// Given a glob pattern, return the directories that can contain files that match the pattern.
    pub fn roots(&self) -> Vec<PathBuf> {
        self.includes.roots()
    }
}

impl FileList for FilteredGlobs {
    fn files(&self) -> anyhow::Result<Vec<PathBuf>> {
        let mut result = Vec::new();
        for file in self.includes.files()? {
            if !self.excludes.matches(&file)? {
                result.push(file);
            }
        }
        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "All found `project_includes` files were filtered by `project_excludes` patterns.\n`project_includes`: {}\n`project_excludes`: {}",
                self.includes.0.iter().map(|p| p.to_string()).join(", "),
                self.excludes.0.iter().map(|p| p.to_string()).join(", "),
            ));
        }
        Ok(result)
    }

    fn covers(&self, path: &Path) -> bool {
        self.includes.covers(path) && !self.excludes.covers(path)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use path_absolutize::Absolutize;

    use super::*;
    use crate::test::util::TestPath;

    #[test]
    fn test_roots() {
        fn f(pattern: &str, root: &str) {
            let globs = Globs::new(vec![pattern.to_owned()]);
            assert_eq!(
                globs.roots(),
                vec![PathBuf::from(root)],
                "Glob parsing failed for pattern {}",
                pattern
            );
        }

        f("project/**/files", "project");
        f("**/files", "");
        f("pattern", "pattern");
        f("pattern.txt", "");
        f("a/b", "a/b");
        f("a/b/c.txt", "a/b");
        f("a/b*/c", "a");
        f("a/b/*.txt", "a/b");
        f("/**", "/");
        f("/absolute/path/**/files", "/absolute/path");

        if cfg!(windows) {
            // These all use the \ separator, which only works on Windows.
            f(r"C:\\windows\project\**\files", r"C:\\windows\project");
            f(r"c:\windows\project\**\files", r"c:\windows\project");
            f(r"\windows\project\**\files", r"\windows\project");
            f(r"c:project\**\files", "c:project");
            f(r"project\**\files", "project");
            f(r"**\files", "");
            f("pattern", "pattern");
            f("pattern.txt", "");
            f(r"a\b", r"a\b");
            f(r"a\b\c.txt", r"a\b");
            f(r"a\b*\c", "a");
            f(r"a\b\*.txt", r"a\b");
        }
    }

    #[test]
    fn test_contains_asterisk() {
        assert!(!Glob::contains_asterisk(&OsString::from("")));
        assert!(Glob::contains_asterisk(&OsString::from("*")));
        assert!(Glob::contains_asterisk(&OsString::from("*a")));
        assert!(Glob::contains_asterisk(&OsString::from("a*")));
        assert!(!Glob::contains_asterisk(&OsString::from("abcd")));
        assert!(Glob::contains_asterisk(&OsString::from("**")));
        assert!(Glob::contains_asterisk(&OsString::from("asdf*fdsa")));
    }

    #[test]
    fn test_globs_relative_to_root() {
        let mut inputs: Vec<&str> = vec![
            "project/**/files",
            "**/files",
            "pattern",
            "pattern.txt",
            "a/b",
            "a/b/c.txt",
            "a/b*/c",
            "a/b/*.txt",
            "/**",
            "/absolute/path/**/files",
        ];
        if cfg!(windows) {
            inputs.extend([r"c:\absolute\path\**", r"c:relative\path\**"]);
        }
        let inputs: Vec<String> = inputs.into_iter().map(String::from).collect();

        let f = |root: &str, expected: Vec<&str>, windows_extras: Vec<&str>| {
            let mut expected: Vec<PathBuf> = expected.into_iter().map(PathBuf::from).collect();
            let inputs = inputs.clone();
            let root = root.to_owned();

            // because windows will construct all paths with `\` instead of `/`, let's make things simple by
            // switching everything over to that ourselves
            if cfg!(windows) {
                expected.extend(
                    windows_extras
                        .into_iter()
                        .map(PathBuf::from)
                        .collect::<Vec<PathBuf>>(),
                );
            }
            let globs: Vec<PathBuf> = Globs::new_with_root(Path::new(&root), inputs)
                .0
                .into_iter()
                .map(|g| g.0)
                .collect();
            assert_eq!(globs, expected, "with root {:?}", root);
        };

        f(
            "",
            vec![
                "project/**/files",
                "**/files",
                "pattern",
                "pattern.txt",
                "a/b",
                "a/b/c.txt",
                "a/b*/c",
                "a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![r"c:\absolute\path\**", r"c:relative\path\**"],
        );
        f(
            ".",
            vec![
                "project/**/files",
                "**/files",
                "pattern",
                "pattern.txt",
                "a/b",
                "a/b/c.txt",
                "a/b*/c",
                "a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![r"c:\absolute\path\**", r"c:relative\path\**"],
        );
        f(
            "..",
            vec![
                "../project/**/files",
                "../**/files",
                "../pattern",
                "../pattern.txt",
                "../a/b",
                "../a/b/c.txt",
                "../a/b*/c",
                "../a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![r"c:\absolute\path\**", r"c:..\relative\path\**"],
        );
        f(
            "no/trailing/slash",
            vec![
                "no/trailing/slash/project/**/files",
                "no/trailing/slash/**/files",
                "no/trailing/slash/pattern",
                "no/trailing/slash/pattern.txt",
                "no/trailing/slash/a/b",
                "no/trailing/slash/a/b/c.txt",
                "no/trailing/slash/a/b*/c",
                "no/trailing/slash/a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![
                r"c:\absolute\path\**",
                r"c:no\trailing\slash\relative\path\**",
            ],
        );
        f(
            "relative/path/to/",
            vec![
                "relative/path/to/project/**/files",
                "relative/path/to/**/files",
                "relative/path/to/pattern",
                "relative/path/to/pattern.txt",
                "relative/path/to/a/b",
                "relative/path/to/a/b/c.txt",
                "relative/path/to/a/b*/c",
                "relative/path/to/a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![
                r"c:\absolute\path\**",
                r"c:relative\path\to\relative\path\**",
            ],
        );
        f(
            "/my/path/to",
            vec![
                "/my/path/to/project/**/files",
                "/my/path/to/**/files",
                "/my/path/to/pattern",
                "/my/path/to/pattern.txt",
                "/my/path/to/a/b",
                "/my/path/to/a/b/c.txt",
                "/my/path/to/a/b*/c",
                "/my/path/to/a/b/*.txt",
                "/**",
                "/absolute/path/**/files",
            ],
            vec![r"c:\absolute\path\**", r"c:\my\path\to\relative\path\**"],
        );
        if cfg!(windows) {
            f(
                r"c:\my\path\to",
                vec![
                    r"c:\my\path\to\project\**\files",
                    r"c:\my\path\to\**\files",
                    r"c:\my\path\to\pattern",
                    r"c:\my\path\to\pattern.txt",
                    r"c:\my\path\to\a\b",
                    r"c:\my\path\to\a\b\c.txt",
                    r"c:\my\path\to\a\b*\c",
                    r"c:\my\path\to\a\b\*.txt",
                    r"c:\**",
                    r"c:\absolute\path\**\files",
                ],
                vec![r"c:\absolute\path\**", r"c:\my\path\to\relative\path\**"],
            );
            f(
                r"c:my\path\to",
                vec![
                    r"c:my\path\to\project\**\files",
                    r"c:my\path\to\**\files",
                    r"c:my\path\to\pattern",
                    r"c:my\path\to\pattern.txt",
                    r"c:my\path\to\a\b",
                    r"c:my\path\to\a\b\c.txt",
                    r"c:my\path\to\a\b*\c",
                    r"c:my\path\to\a\b\*.txt",
                    r"c:\**",
                    r"c:\absolute\path\**\files",
                ],
                vec![r"c:\absolute\path\**", r"c:my\path\to\relative\path\**"],
            );
            f(
                r"d:\my\path\to",
                vec![
                    r"d:\my\path\to\project\**\files",
                    r"d:\my\path\to\**\files",
                    r"d:\my\path\to\pattern",
                    r"d:\my\path\to\pattern.txt",
                    r"d:\my\path\to\a\b",
                    r"d:\my\path\to\a\b\c.txt",
                    r"d:\my\path\to\a\b*\c",
                    r"d:\my\path\to\a\b\*.txt",
                    r"d:\**",
                    r"d:\absolute\path\**\files",
                ],
                vec![r"c:\absolute\path\**", r"c:relative\path\**"],
            );
            f(
                r"d:my\path\to",
                vec![
                    r"d:my\path\to\project\**\files",
                    r"d:my\path\to\**\files",
                    r"d:my\path\to\pattern",
                    r"d:my\path\to\pattern.txt",
                    r"d:my\path\to\a\b",
                    r"d:my\path\to\a\b\c.txt",
                    r"d:my\path\to\a\b*\c",
                    r"d:my\path\to\a\b\*.txt",
                    r"d:\**",
                    r"d:\absolute\path\**\files",
                ],
                vec![r"c:\absolute\path\**", r"c:relative\path\**"],
            );
        }
    }

    #[test]
    fn test_is_python_extension() {
        assert!(!Glob::is_python_extension(None));
        assert!(!Glob::is_python_extension(Some(OsStr::new("hello world!"))));
        assert!(Glob::is_python_extension(Some(OsStr::new("py"))));
        assert!(Glob::is_python_extension(Some(OsStr::new("pyi"))));
    }

    #[test]
    fn test_path_matches_default_exclude_glob() {
        let patterns = Globs::new(vec![
            "**/__pycache__/**".to_owned(),
            "**/.[!/.]*".to_owned(),
        ]);

        assert!(patterns.matches(Path::new("__pycache__/")).unwrap());
        assert!(
            patterns
                .matches(Path::new("__pycache__/some/cached/file.pyc"))
                .unwrap()
        );
        assert!(patterns.matches(Path::new("path/to/__pycache__/")).unwrap());
        assert!(patterns.matches(Path::new(".hidden")).unwrap());
        assert!(patterns.matches(Path::new("path/to/.hidden")).unwrap());
        assert!(!patterns.matches(Path::new("./test")).unwrap());
        assert!(!patterns.matches(Path::new("../test")).unwrap());
        assert!(!patterns.matches(Path::new("a/.")).unwrap());
        assert!(!patterns.matches(Path::new("a/..")).unwrap());
        assert!(!patterns.matches(Path::new("a/./")).unwrap());
        assert!(!patterns.matches(Path::new("a/../")).unwrap());
        assert!(!patterns.matches(Path::new("a/./test")).unwrap());
        assert!(!patterns.matches(Path::new("a/../test")).unwrap());
        assert!(patterns.matches(Path::new("a/.a/")).unwrap());
        assert!(patterns.matches(Path::new("a/.ab/")).unwrap());
        assert!(patterns.matches(Path::new("a/.a/")).unwrap());
        assert!(patterns.matches(Path::new("a/.ab/")).unwrap());
        assert!(!patterns.matches(Path::new("just/a/regular.file")).unwrap());
        assert!(!patterns.matches(Path::new("file/with/a.dot")).unwrap());
        assert!(
            !Globs::new(vec!["**/__pycache__".to_owned()])
                .matches(Path::new("__pycache__/some/file.pyc"))
                .unwrap()
        );
        assert!(
            Globs::new(vec!["**/__pycache__/".to_owned()])
                .matches(Path::new("__pycache__/some/file.pyc"))
                .unwrap()
        );
        assert!(
            !Globs::new(vec!["**/__pycache__".to_owned()])
                .matches(Path::new("__pycache__/"))
                .unwrap()
        );
        assert!(
            Globs::new(vec!["**/__pycache__".to_owned()])
                .matches(Path::new("__pycache__"))
                .unwrap()
        );
        assert!(
            Globs::new(vec!["**/__pycache__/".to_owned()])
                .matches(Path::new("__pycache__/"))
                .unwrap()
        );
        assert!(
            !Globs::new(vec!["**/__pycache__/".to_owned()])
                .matches(Path::new("__pycache__"))
                .unwrap()
        );
        assert!(
            !Globs::new(vec!["**/__pycache__/**".to_owned()])
                .matches(Path::new("__pycache__"))
                .unwrap()
        );
    }

    #[test]
    fn test_globs_match_file() {
        fn glob_matches(pattern: &str, equal: bool) {
            let file_to_match = Path::new("path/to/my/file.py").absolutize().unwrap();
            let root = std::env::current_dir().unwrap();
            let root = root.absolutize().unwrap();
            let glob = Glob::new_with_root(&root, pattern.to_owned());
            assert!(
                glob.matches(file_to_match.as_ref()).unwrap() == equal,
                "glob `{}` failed (`{}` expanded, `{}` file)",
                pattern,
                glob,
                file_to_match.display(),
            );
        }

        // TODO(connernilsen): all of these should be uncommented and pass

        // glob_matches("path/to", true);
        // glob_matches("path/to/", true);
        glob_matches("path/to/m", false);
        glob_matches("path/to/m*", true);

        glob_matches("path/to/my/file.py", true);
        glob_matches("path/to/my/file.py/", true);
        glob_matches("path/to/my/file.py/this_is_weird.py", false);
        glob_matches("path/to/my/file.pyi", false);
        glob_matches("path/to/my/file", false);
        glob_matches("path/to/my/file*", true);
        glob_matches("path/to/my/f*", true);
        glob_matches("path/to/my/*e*", true);

        // glob_matches("", true);
        // glob_matches("..", true);
        // glob_matches(".", true);
        // glob_matches("path/to/./my", true);
        glob_matches("*", true);
        glob_matches("**", true);
        glob_matches("**/*", true);
        glob_matches("**/*.py", true);
        glob_matches("**/*.pyi", false);
    }

    #[test]
    fn test_globbing_on_project() {
        use std::path::StripPrefixError;

        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "a",
                    vec![
                        TestPath::file("b.py"),
                        TestPath::dir(
                            "c",
                            vec![
                                TestPath::file("d.py"),
                                TestPath::file("e.pyi"),
                                TestPath::file("f.not_py"),
                            ],
                        ),
                        TestPath::file(".dotfile.py"),
                        TestPath::dir(
                            "__pycache__",
                            vec![TestPath::file("g.py"), TestPath::file("h.pyi")],
                        ),
                    ],
                ),
                TestPath::dir(
                    // another c directory
                    "c",
                    vec![
                        TestPath::file("i"),
                        TestPath::dir("j", vec![TestPath::file("k.py")]),
                    ],
                ),
                TestPath::file("l.py"),
                TestPath::dir("also_has_l", vec![TestPath::file("m.py")]),
            ],
        );

        let glob_files_match = |pattern: &str, expected: &[&str]| -> anyhow::Result<()> {
            let glob_files = Glob::new_with_root(root, pattern.to_owned()).files()?;
            let mut glob_files = glob_files
                .iter()
                .map(|p| p.strip_prefix(root))
                .collect::<Result<Vec<&Path>, StripPrefixError>>()
                .unwrap();
            glob_files.sort();
            glob_files.dedup();

            let mut expected = expected.iter().map(Path::new).collect::<Vec<&Path>>();
            expected.sort();
            expected.dedup();

            assert_eq!(
                glob_files, expected,
                "failed to match with pattern `{pattern}`"
            );

            Ok(())
        };

        let all_valid_files = &[
            "a/b.py",
            "a/c/d.py",
            "a/c/e.pyi",
            "a/.dotfile.py",
            "a/__pycache__/g.py",
            "a/__pycache__/h.pyi",
            "c/j/k.py",
            "l.py",
            "also_has_l/m.py",
        ];

        // TODO(connernilsen): all below files/tests should be uncommented
        glob_files_match("", all_valid_files).unwrap();
        glob_files_match(".", all_valid_files).unwrap();
        glob_files_match(
            "**",
            // should be all_valid_files
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/.dotfile.py",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
                "c/j/k.py",
                // "l.py",
                "also_has_l/m.py",
            ],
        )
        .unwrap();
        glob_files_match("**/*", all_valid_files).unwrap();

        glob_files_match(
            "**/*.py",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/.dotfile.py",
                "a/__pycache__/g.py",
                "c/j/k.py",
                "l.py",
                "also_has_l/m.py",
            ],
        )
        .unwrap();
        glob_files_match("**/*.pyi", &["a/c/e.pyi", "a/__pycache__/h.pyi"]).unwrap();
        glob_files_match("**/*.py?", &["a/c/e.pyi", "a/__pycache__/h.pyi"]).unwrap();
        glob_files_match("**/*.py*", all_valid_files).unwrap();

        // this one may be unexpected, since the glob pattern should only match `l.py`,  but we
        // have `resolve_dir` to handle searching this anyway.
        // in our case, it will probably be fine, since we technically are 'matching' the
        // directories there, and it can be further tuned with `project_excludes`.
        glob_files_match("*", all_valid_files).unwrap();

        glob_files_match(
            "**/a",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/.dotfile.py",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
            ],
        )
        .unwrap();
        glob_files_match(
            "**/a/**",
            &[
                // "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                // "a/.dotfile.py",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
            ],
        )
        .unwrap();
        glob_files_match(
            "**/a/*",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
                "a/.dotfile.py",
            ],
        )
        .unwrap();

        glob_files_match("**/c", &["a/c/d.py", "a/c/e.pyi", "c/j/k.py"]).unwrap();
        glob_files_match(
            "**/c/**",
            &[
                // "a/c/d.py",
                // "a/c/e.pyi",
                "c/j/k.py",
            ],
        )
        .unwrap();
        glob_files_match("a/c", &["a/c/d.py", "a/c/e.pyi"]).unwrap();

        assert!(glob_files_match("l", &[]).is_err());
        glob_files_match("*l", &["also_has_l/m.py"]).unwrap();
        glob_files_match("*l*", &["l.py", "also_has_l/m.py"]).unwrap();
        glob_files_match(
            "?",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/.dotfile.py",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
                "c/j/k.py",
            ],
        )
        .unwrap();
    }
}
