/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::path::Component;
use std::path::MAIN_SEPARATOR;
use std::path::MAIN_SEPARATOR_STR;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use bstr::ByteSlice;
use glob::Pattern;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use serde::de;
use serde::de::Visitor;
use starlark_map::small_set::SmallSet;
use tracing::debug;

use crate::absolutize::Absolutize as _;
use crate::fs_anyhow;
use crate::prelude::SliceExt;
use crate::prelude::VecExt;

#[derive(Debug, Clone, Eq, Default)]

/// A glob pattern for matching files.
///
/// Only matches Python files (.py, .pyi, .pyw) and automatically excludes:
/// - Files that don't have .py, .pyi, or .pyw extensions
/// - Files whose names start with '.' (dot files)
pub struct Glob(Pattern);

impl Glob {
    /// Create a new `Glob`, but do not do absolutizing (since we don't want to do
    /// that until rewriting with a root)
    pub fn new(mut pattern: String) -> anyhow::Result<Self> {
        if pattern.ends_with("**") {
            pattern.push_str(&format!("{MAIN_SEPARATOR_STR}*"));
        } else if pattern.ends_with("**/") || pattern.ends_with(r"**\") {
            pattern.push('*');
        }
        Ok(Self(Pattern::new(&pattern).with_context(|| {
            format!("While constructing glob pattern from {pattern}")
        })?))
    }

    /// Create a new `Glob`, with the pattern relative to `root`.
    /// `root` should be an absolute path.
    pub fn new_with_root(root: &Path, pattern: String) -> anyhow::Result<Self> {
        Ok(Self::new(pattern)?.from_root(root))
    }

    /// Rewrite the current `Glob` relative to `root`.
    /// `root` should be an absolute path.
    pub fn from_root(self, root: &Path) -> Self {
        Self(Self::pattern_relative_to_root(root, &self.0))
    }

    fn contains_glob_char(part: &OsStr) -> bool {
        let bytes = part.as_encoded_bytes();
        bytes.contains(&b'*') || bytes.contains(&b'?') || bytes.contains(&b'[')
    }

    fn pattern_relative_to_root(root: &Path, pattern: &Pattern) -> Pattern {
        let from_root = Path::new(pattern.as_str())
            .absolutize_from(Path::new(&Pattern::escape(root.to_string_lossy().as_ref())));
        // the unwrap is okay because the previous `pattern` worked already
        // and the escaped `root` shouldn't realistically fail
        Pattern::new(&from_root.to_string_lossy())
            .with_context(|| "Got invalid Glob pattern relative to root that was assumed to be safe. Please report this to the Pyrefly maintainers on our GitHub.").unwrap()
    }

    fn get_glob_root(&self) -> PathBuf {
        let mut path = PathBuf::new();

        // we need to add any path prefix and root items (there should be at most one of each,
        // and prefix only exists on windows) to the root we're building
        self.as_path()
            .components()
            .take_while(|comp| {
                match comp {
                    // this should be alright to do, since a prefix will always come before a root,
                    // which will always come before the rest of the path
                    Component::Prefix(_)
                    | Component::RootDir
                    | Component::CurDir
                    | Component::ParentDir => true,
                    Component::Normal(part) => !Self::contains_glob_char(part),
                }
            })
            .for_each(|comp| path.push(comp));
        if path.extension().is_some() {
            path.pop();
        }
        path
    }

    pub fn as_path(&self) -> &Path {
        Path::new(self.0.as_str())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    fn is_python_extension(ext: Option<&OsStr>) -> bool {
        ext.is_some_and(|e| e == "py" || e == "pyi" || e == "pyw")
    }

    /// Returns true if the given file should be included in results.
    /// Filters out non-Python files and dot files.
    fn should_include_file(path: &Path) -> bool {
        // Check if it's a Python file
        if !Self::is_python_extension(path.extension()) {
            return false;
        }

        // Check if it's a dot file
        if let Some(file_name) = path.file_name().and_then(OsStr::to_str)
            && file_name.starts_with('.')
        {
            return false;
        }

        true
    }

    fn resolve_path(
        path: PathBuf,
        results: &mut Vec<PathBuf>,
        filter: &Globs,
    ) -> anyhow::Result<()> {
        if filter.matches(&path) {
            return Ok(());
        }
        if path.is_dir() {
            Self::resolve_dir(&path, results, filter)?;
        } else if Self::should_include_file(&path) {
            results.push(path);
        }
        Ok(())
    }

    fn resolve_dir(path: &Path, results: &mut Vec<PathBuf>, filter: &Globs) -> anyhow::Result<()> {
        for entry in fs_anyhow::read_dir(path)? {
            let entry = entry
                .with_context(|| format!("When iterating over directory `{}`", path.display()))?;
            let path = entry.path();
            Self::resolve_path(path, results, filter)?;
        }
        Ok(())
    }

    fn resolve_pattern(pattern: &str, filter: &Globs) -> anyhow::Result<Vec<PathBuf>> {
        let mut result = Vec::new();
        let paths = glob::glob(pattern)?;
        for path in paths {
            let path = path?;
            Self::resolve_path(path, &mut result, filter)?;
        }
        Ok(result)
    }

    /// Returns true if the given file matches any of the contained globs.
    /// We always attempt to append `**` in case
    /// the pattern is meant to be a directory wildcard.
    pub fn matches(&self, file: &Path) -> bool {
        if self.0.matches_path(file) {
            return true;
        }

        // if we could match before, see if it's because of some matching semantics
        // around the glob library we're using, where the end MUST be a wildcard
        let pattern_path = &self.0;
        let mut pattern_str = pattern_path.as_str().to_owned();
        if !pattern_str.ends_with(['/', '\\']) {
            pattern_str.push(MAIN_SEPARATOR);
        }
        pattern_str.push_str("**");

        // don't return an error if we fail to construct a glob here, since it's something
        // we automatically attempted and failed at. We should ignore failure here, since
        // we attempted to do this automatically, and the pattern we're constructing should be valid
        // (i.e. the previous pattern we constructed should have failed before we get to here).
        glob::Pattern::new(&pattern_str).is_ok_and(|pattern| pattern.matches_path(file))
    }
}

impl Display for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.as_str())
    }
}

impl<'de> Deserialize<'de> for Glob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct GlobVisitor;

        impl<'de> Visitor<'de> for GlobVisitor {
            type Value = Glob;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("glob")
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                match Glob::new(value) {
                    Ok(ok) => Ok(ok),
                    Err(error) => Err(E::custom(
                        format!("Failed to deserialize as Glob: {error}",),
                    )),
                }
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                self.visit_string(v.to_owned())
            }
        }

        deserializer.deserialize_string(GlobVisitor)
    }
}

impl Serialize for Glob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl Hash for Glob {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_path().hash(state);
    }
}

impl PartialEq for Glob {
    fn eq(&self, other: &Self) -> bool {
        // we want to use path equality, since we don't want to have to worry about
        // platform-dependent path separators
        self.as_path() == other.as_path()
    }
}

impl Glob {
    fn files(&self, filter: &Globs) -> anyhow::Result<Vec<PathBuf>> {
        let pattern = &self.0;
        if filter.matches(self.as_path()) {
            return Err(anyhow::anyhow!(
                "Pattern {} is matched by `project-excludes`.\n`project-excludes`: {}",
                pattern.as_str(),
                filter.0.iter().map(|p| p.to_string()).join(", "),
            ));
        }
        let pattern_str = pattern.as_str().to_owned();
        let result = Self::resolve_pattern(&pattern_str, filter)
            .with_context(|| format!("When resolving pattern `{pattern_str}`"))?;
        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "No Python files matched pattern `{}`",
                pattern_str
            ));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, Default)]
pub struct Globs(Vec<Glob>);

impl Globs {
    pub fn empty() -> Self {
        Self(vec![])
    }

    /// Create a new `Globs` from the given patterns. If you want them to be relative
    /// to a root, please use `Globs::new_with_root()` instead.
    pub fn new(patterns: Vec<String>) -> anyhow::Result<Self> {
        Ok(Self(patterns.into_try_map(Glob::new)?))
    }

    /// Create a new `Globs`, rewriting all patterns to be relative to `root`.
    /// `root` should be an absolute path.
    pub fn new_with_root(root: &Path, patterns: Vec<String>) -> anyhow::Result<Self> {
        Ok(Self::rewrite_with_root(
            root,
            patterns.into_try_map(Glob::new)?,
        ))
    }

    fn rewrite_with_root(root: &Path, patterns: Vec<Glob>) -> Self {
        Self(patterns.into_map(|pattern| pattern.from_root(root)))
    }

    /// Rewrite the existing `Globs` to be relative to `root`.
    /// `root` should be an absolute path.
    pub fn from_root(self, root: &Path) -> Self {
        // TODO(connernilsen): store root as part of globs to make it easier to rewrite later on
        Self::rewrite_with_root(root, self.0)
    }

    /// Given a glob pattern, return the directories that can contain files that match the pattern.
    pub fn roots(&self) -> Vec<PathBuf> {
        let mut res = self.0.map(|s| s.get_glob_root());
        res.sort();
        res.dedup();
        // We could dedup more in future, if there is `/foo` and `/foo/bar` then the second is redundant.
        res
    }

    /// Returns true if the given file matches any of the contained globs.
    /// We always attempt to append `**` if a pattern ends in `/` in case
    /// the pattern is meant to be a directory wildcard.
    fn matches(&self, file: &Path) -> bool {
        self.0.iter().any(|pattern| pattern.matches(file))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn append(&mut self, patterns: &[Glob]) {
        self.0.extend_from_slice(patterns);
    }
}

impl Display for Globs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|g| g.to_string()).join(", "))
    }
}

/// If `eden` is likely to be available, we can resolve the globs faster.
/// For a 100K file project, with a warm disk, non-Eden = 1.6s, Eden = 1.1s.
/// For a cold disk, Eden is likely to win by a much larger margin.
/// Currently `eden` is only likely available inside Meta.
const USE_EDEN: bool = cfg!(fbcode_build);

impl Globs {
    pub fn files_eden(&self, filter: &Globs) -> anyhow::Result<Vec<PathBuf>> {
        fn hg_root() -> anyhow::Result<PathBuf> {
            let output = Command::new("hg")
                .arg("root")
                .output()
                .context("Failed to run `hg root`")?;
            if !output.status.success() {
                return Err(anyhow::anyhow!(
                    "Failed to run `hg root`, stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            Ok(PathBuf::from(std::str::from_utf8(
                output.stdout.trim_ascii(),
            )?))
        }

        fn eden_glob(root: PathBuf, patterns: Vec<&Path>) -> anyhow::Result<Vec<PathBuf>> {
            let mut command = Command::new("eden");
            command.arg("glob");
            command.args(patterns);
            command.current_dir(&root);
            let output = command.output().context("Failed to run `eden glob`")?;
            if !output.status.success() {
                // Last line of stderr of `eden glob` is usually a good indicator of what happened
                let stderr_text = String::from_utf8_lossy(&output.stderr);
                return Err(
                    anyhow::anyhow!("{}", stderr_text.lines().last().unwrap_or(""))
                        .context("Failure when running `eden glob`"),
                );
            }
            let mut result: Vec<PathBuf> = Vec::new();
            for line in output.stdout.lines() {
                let path = line.to_path().with_context(|| {
                    format!(
                        "Failed to convert line `{}` into a valid path",
                        line.to_str_lossy()
                    )
                })?;
                Glob::resolve_path(root.join(path), &mut result, &Globs::empty())?;
            }
            Ok(result)
        }

        let root = hg_root()?;
        let globs = self.0.try_map(|g| g.as_path().strip_prefix(&root))?;
        let mut result = eden_glob(root, globs)?;
        result.retain(|p| !filter.matches(p));
        Ok(result)
    }

    fn filtered_files(&self, filter: &Globs) -> anyhow::Result<Vec<PathBuf>> {
        if USE_EDEN {
            match self.files_eden(filter) {
                Ok(files) if files.is_empty() => {
                    return Err(anyhow::anyhow!(
                        "No Python files matched pattern(s) {}",
                        self.0.map(|p| format!("`{}`", p.as_str())).join(", "),
                    ));
                }
                Ok(files) => return Ok(files),
                Err(e) => debug!("Failed to use `eden` for glob: {e:#}"),
            }
        }

        let mut result = SmallSet::new();
        for pattern in &self.0 {
            result.extend(pattern.files(filter)?);
        }
        Ok(result.into_iter().collect())
    }

    pub fn files(&self) -> anyhow::Result<Vec<PathBuf>> {
        self.filtered_files(&Globs::empty())
    }

    pub fn covers(&self, path: &Path) -> bool {
        self.matches(path)
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

    pub fn files(&self) -> anyhow::Result<Vec<PathBuf>> {
        self.includes.filtered_files(&self.excludes)
    }

    pub fn covers(&self, path: &Path) -> bool {
        self.includes.covers(path) && !self.excludes.covers(path)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use super::*;
    use crate::test_path::TestPath;

    #[test]
    fn test_roots() {
        fn f(pattern: &str, root: &str) {
            let globs = Globs::new(vec![pattern.to_owned()]).unwrap();
            assert_eq!(
                globs.roots(),
                vec![PathBuf::from(root)],
                "Glob parsing failed for pattern {pattern}",
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
    fn test_contains_glob_char() {
        assert!(!Glob::contains_glob_char(&OsString::from("")));
        assert!(Glob::contains_glob_char(&OsString::from("*")));
        assert!(Glob::contains_glob_char(&OsString::from("*a")));
        assert!(Glob::contains_glob_char(&OsString::from("a*")));
        assert!(!Glob::contains_glob_char(&OsString::from("abcd")));
        assert!(Glob::contains_glob_char(&OsString::from("**")));
        assert!(Glob::contains_glob_char(&OsString::from("asdf*fdsa")));
        assert!(Glob::contains_glob_char(&OsString::from("?")));
        assert!(Glob::contains_glob_char(&OsString::from("?a")));
        assert!(Glob::contains_glob_char(&OsString::from("a?")));
        assert!(Glob::contains_glob_char(&OsString::from("asdf?fdsa")));
        assert!(Glob::contains_glob_char(&OsString::from("[")));
        assert!(Glob::contains_glob_char(&OsString::from("[ab]")));
        assert!(Glob::contains_glob_char(&OsString::from("a[]")));
        assert!(Glob::contains_glob_char(&OsString::from("asdf[abcd]fdsa")));
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
            "/**/",
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

            // windows has drives, so add tests for that when applicable
            if cfg!(windows) {
                expected.extend(
                    windows_extras
                        .into_iter()
                        .map(PathBuf::from)
                        .collect::<Vec<PathBuf>>(),
                );
            }
            let globs: Vec<PathBuf> = Globs::new_with_root(Path::new(&root), inputs)
                .unwrap()
                .0
                .into_iter()
                .map(|g| g.as_path().to_path_buf())
                .collect();
            assert_eq!(globs, expected, "with root {root:?}");
        };

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
                "/**/*",
                "/**/*",
                "/absolute/path/**/files",
            ],
            vec![
                r"c:\absolute\path\**\*",
                r"c:\my\path\to\relative\path\**\*",
            ],
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
                    r"c:\**\*",
                    r"c:\**\*",
                    r"c:\absolute\path\**\files",
                ],
                vec![
                    r"c:\absolute\path\**\*",
                    r"c:\my\path\to\relative\path\**\*",
                ],
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
                    r"d:\**\*",
                    r"d:\**\*",
                    r"d:\absolute\path\**\files",
                ],
                vec![
                    r"c:\absolute\path\**\*",
                    r"c:\my\path\to\relative\path\**\*",
                ],
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
        ])
        .unwrap();

        assert!(patterns.matches(Path::new("__pycache__/")));
        assert!(patterns.matches(Path::new("__pycache__/some/cached/file.pyc")));
        assert!(patterns.matches(Path::new("path/to/__pycache__/")));
        assert!(patterns.matches(Path::new(".hidden")));
        assert!(patterns.matches(Path::new("path/to/.hidden")));
        assert!(!patterns.matches(Path::new("./test")));
        assert!(!patterns.matches(Path::new("../test")));
        assert!(!patterns.matches(Path::new("a/.")));
        assert!(!patterns.matches(Path::new("a/..")));
        assert!(!patterns.matches(Path::new("a/./")));
        assert!(!patterns.matches(Path::new("a/../")));
        assert!(!patterns.matches(Path::new("a/./test")));
        assert!(!patterns.matches(Path::new("a/../test")));
        assert!(patterns.matches(Path::new("a/.a/")));
        assert!(patterns.matches(Path::new("a/.ab/")));
        assert!(patterns.matches(Path::new("a/.a/")));
        assert!(patterns.matches(Path::new("a/.ab/")));
        assert!(!patterns.matches(Path::new("just/a/regular.file")));
        assert!(!patterns.matches(Path::new("file/with/a.dot")));
        assert!(
            Globs::new(vec!["**/__pycache__".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__/some/file.pyc"))
        );
        assert!(
            Globs::new(vec!["**/__pycache__/".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__/some/file.pyc"))
        );
        assert!(
            Globs::new(vec!["**/__pycache__".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__/"))
        );
        assert!(
            Globs::new(vec!["**/__pycache__".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__"))
        );
        assert!(
            Globs::new(vec!["**/__pycache__/".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__/"))
        );
        assert!(
            !Globs::new(vec!["**/__pycache__/".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__"))
        );
        assert!(
            !Globs::new(vec!["**/__pycache__/**".to_owned()])
                .unwrap()
                .matches(Path::new("__pycache__"))
        );
    }

    #[test]
    fn test_globs_match_file() {
        fn glob_matches(pattern: &str, equal: bool) {
            let root = std::env::current_dir().unwrap();
            let root = root.absolutize();
            let escaped_root = Pattern::escape(root.to_string_lossy().as_ref());
            let escaped_root = Path::new(&escaped_root);

            let file_to_match = escaped_root.join("path/to/my/file.py");

            let glob = Glob::new_with_root(&root, pattern.to_owned()).unwrap();
            assert!(
                glob.matches(file_to_match.as_ref()) == equal,
                "glob `{}` failed (`{}` expanded, `{}` file)",
                pattern,
                glob,
                file_to_match.display(),
            );
        }

        glob_matches("path/to", true);
        glob_matches("path/to/", true);
        glob_matches("path/to/my", true);
        glob_matches("path/to/my/", true);
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

        glob_matches("", true);
        glob_matches("..", true);
        glob_matches("../**", true);
        glob_matches(".", true);
        glob_matches("./**", true);
        glob_matches("path/to/./my", true);
        glob_matches("path/to/./my/**", true);
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
            let glob_files = Glob::new_with_root(root, pattern.to_owned())
                .unwrap()
                .files(&Globs::empty())?;
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
            "a/__pycache__/g.py",
            "a/__pycache__/h.pyi",
            "c/j/k.py",
            "l.py",
            "also_has_l/m.py",
        ];

        glob_files_match("", all_valid_files).unwrap();
        glob_files_match(".", all_valid_files).unwrap();
        glob_files_match("**", all_valid_files).unwrap();
        glob_files_match("**/*", all_valid_files).unwrap();

        glob_files_match(
            "**/*.py",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/__pycache__/g.py",
                "c/j/k.py",
                "l.py",
                "also_has_l/m.py",
            ],
        )
        .unwrap();
        glob_files_match("**/*.pyi", &["a/c/e.pyi", "a/__pycache__/h.pyi"]).unwrap();
        glob_files_match("**/*.py?", &["a/c/e.pyi", "a/__pycache__/h.pyi"]).unwrap();
        glob_files_match(
            "**/*.py*",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
                "c/j/k.py",
                "l.py",
                "also_has_l/m.py",
            ],
        )
        .unwrap();
        glob_files_match("**/*py*", all_valid_files).unwrap();

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
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
            ],
        )
        .unwrap();
        glob_files_match(
            "**/a/",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
            ],
        )
        .unwrap();
        glob_files_match(
            "**/a/**",
            &[
                "a/b.py",
                "a/c/d.py",
                "a/c/e.pyi",
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
            ],
        )
        .unwrap();

        glob_files_match("**/c", &["a/c/d.py", "a/c/e.pyi", "c/j/k.py"]).unwrap();
        glob_files_match("**/c/", &["a/c/d.py", "a/c/e.pyi", "c/j/k.py"]).unwrap();
        glob_files_match("**/c/**", &["a/c/d.py", "a/c/e.pyi", "c/j/k.py"]).unwrap();
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
                "a/__pycache__/g.py",
                "a/__pycache__/h.pyi",
                "c/j/k.py",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_dot_file_exclusion() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "a",
                    vec![
                        TestPath::file("b.py"),
                        TestPath::file(".dotfile.py"),
                        TestPath::dir(
                            "c",
                            vec![TestPath::file("d.py"), TestPath::file(".hidden.py")],
                        ),
                    ],
                ),
                TestPath::file(".top_level_dot.py"),
            ],
        );

        // Helper function to assert that a glob pattern returns no files
        let assert_empty_glob = |pattern_str: &str, description: &str| {
            let found_files = Glob::new_with_root(root, pattern_str.to_owned())
                .unwrap()
                .files(&Globs::empty())
                .unwrap_or_else(|_| Vec::new());
            assert!(
                found_files.is_empty(),
                "{description} should be excluded, found: {found_files:?}",
            );
        };

        // Test explicit dot file exclusion
        assert_empty_glob("a/.dotfile.py", "Direct dot file path");
        assert_empty_glob("**/.dotfile.py", "Recursive dot file pattern");
        assert_empty_glob("**/.*.py", "Dot file wildcard");
        assert_empty_glob(".top_level_dot.py", "Top-level dot file");
        assert_empty_glob("a/c/.hidden.py", "Nested dot file");

        // Verify that normal files are still found
        let normal_files = Glob::new_with_root(root, "**/*.py".to_owned())
            .unwrap()
            .files(&Globs::empty())
            .unwrap();
        assert!(
            !normal_files.is_empty(),
            "Normal Python files should still be found"
        );

        // Ensure no dot files are in the results
        for file in &normal_files {
            if let Some(file_name) = file.file_name().and_then(|n| n.to_str()) {
                assert!(
                    !file_name.starts_with('.'),
                    "Found dot file in results: {file:?}",
                );
            }
        }
    }

    #[test]
    fn test_glob_filter_files() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "a",
                vec![TestPath::file("b.py"), TestPath::file("c.py")],
            )],
        );

        let pattern = root.join("**").to_string_lossy().to_string();

        let mut sorted_globs = Glob::resolve_pattern(&pattern, &Globs::empty()).unwrap();
        sorted_globs.sort();
        assert_eq!(sorted_globs, vec![root.join("a/b.py"), root.join("a/c.py")]);
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap()
            )
            .unwrap(),
            Vec::<PathBuf>::new()
        );
        assert!(
            Glob::new(pattern.clone())
                .unwrap()
                .files(&Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap())
                .is_err()
        );
        // double check that <path>/** will also match <path>
        assert!(
            Glob::new(root.to_string_lossy().to_string())
                .unwrap()
                .files(&Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap())
                .is_err()
        );
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &Globs::new(vec![root.join("a/c.py").to_string_lossy().to_string()]).unwrap()
            )
            .unwrap(),
            vec![root.join("a/b.py")],
        );
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &Globs::new(vec![root.join("a").to_string_lossy().to_string()]).unwrap()
            )
            .unwrap(),
            Vec::<PathBuf>::new()
        );
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &Globs::new(vec![root.join("a/b*").to_string_lossy().to_string()]).unwrap()
            )
            .unwrap(),
            vec![root.join("a/c.py")],
        );
    }
}
