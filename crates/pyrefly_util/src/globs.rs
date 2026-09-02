/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::thread::available_parallelism;

use anyhow::Context;
use glob::Pattern;
use ignore::Match;
use ignore::WalkBuilder;
use ignore::WalkState;
use ignore::gitignore::Gitignore;
use ignore::gitignore::GitignoreBuilder;
use ignore::types::Types;
use ignore::types::TypesBuilder;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use serde::de;
use serde::de::Visitor;
use tracing::warn;

use crate::absolutize::Absolutize as _;
use crate::includes::Includes;
use crate::lock::Mutex;
use crate::prelude::SliceExt;
use crate::prelude::VecExt;
use crate::upward_search::UpwardSearch;

static IGNORE_FILES_SEARCH: LazyLock<Vec<UpwardSearch<Arc<(PathBuf, PathBuf)>>>> =
    LazyLock::new(|| {
        [".gitignore", ".ignore", ".git/info/exclude"]
            .iter()
            .map(|f| {
                UpwardSearch::new(vec![OsString::from(f)], |p| {
                    let mut ignore_root = p.to_path_buf();
                    ignore_root.pop();
                    if *f == ".git/info/exclude" {
                        ignore_root.pop();
                        ignore_root.pop();
                    }
                    Arc::new((p.to_path_buf(), ignore_root))
                })
            })
            .collect::<Vec<_>>()
    });

const PYTHON_FILE_EXTENSIONS: &[&str] = &["py", "pyi", "pyw", "ipynb"];

// Match Ruff's cap: walking is filesystem-bound, so spawning one thread per CPU
// can overwhelm EdenFS on large devservers without improving latency.
const MAX_PARALLEL_WALK_THREADS: usize = 12;

#[derive(Clone, Eq)]
/// A glob pattern for matching files.
/// Patterns must use `/` as the path separator for cross-platform consistency.
///
/// Only matches Python files (.py, .pyi, .pyw) and automatically excludes:
/// - Files that don't have .py, .pyi, or .pyw extensions
/// - Files whose names start with '.' (dot files)
pub struct Glob {
    /// The (already-normalized) glob pattern. Doubles as the source string via
    /// [`Glob::as_str`].
    pattern: Pattern,
    /// Precompiled `<pattern>/**` variant, so a glob naming a directory matches
    /// the files beneath it. Precomputed so [`Glob::matches`] stays allocation-
    /// and parse-free on the discovery hot path. `None` when `pattern` already
    /// ends in a `**` globstar and so matches its descendants on its own, or if
    /// this best-effort variant cannot be parsed.
    dir_pattern: Option<Pattern>,
}

impl Glob {
    /// Create a new `Glob`, but do not do absolutizing (since we don't want to do
    /// that until rewriting with a root).
    /// Patterns must use `/` as the path separator for cross-platform consistency.
    pub fn new(pattern: String) -> anyhow::Result<Self> {
        Ok(Self::from_base_pattern(Self::base_pattern(pattern)?))
    }

    fn base_pattern(mut pattern: String) -> anyhow::Result<Pattern> {
        if pattern.ends_with("**") {
            pattern.push_str("/*");
        } else if pattern.ends_with("**/") {
            pattern.push('*');
        }
        Pattern::new(&pattern)
            .with_context(|| format!("While constructing glob pattern from {pattern}"))
    }

    /// Build a `Glob` from an already-validated base [`Pattern`], precompiling the
    /// `<pattern>/**` variant used to match files beneath a directory pattern.
    fn from_base_pattern(pattern: Pattern) -> Self {
        // A pattern ending in `**/*` (the normalized form of a trailing `**`)
        // already matches every descendant, so the `<pattern>/**` variant would
        // be redundant: `P/**/*` matches a superset of `P/**/*/**`.
        let dir_pattern = if pattern.as_str().ends_with("**/*") {
            None
        } else {
            let mut dir = pattern.as_str().to_owned();
            if !dir.ends_with('/') {
                dir.push('/');
            }
            dir.push_str("**");
            Pattern::new(&dir).ok()
        };
        Self {
            pattern,
            dir_pattern,
        }
    }

    /// Create a new `Glob`, with the pattern relative to `root`.
    /// `root` should be an absolute path.
    pub fn new_with_root(root: &Path, pattern: String) -> anyhow::Result<Self> {
        Ok(Self::from_base_pattern(Self::pattern_relative_to_root(
            root,
            &Self::base_pattern(pattern)?,
        )))
    }

    /// Rewrite the current `Glob` relative to `root`.
    /// `root` should be an absolute path.
    pub fn from_root(self, root: &Path) -> Self {
        Self::from_base_pattern(Self::pattern_relative_to_root(root, &self.pattern))
    }

    fn contains_glob_char(part: &OsStr) -> bool {
        let bytes = part.as_encoded_bytes();
        bytes.contains(&b'*') || bytes.contains(&b'?') || bytes.contains(&b'[')
    }

    /// Returns true if this pattern contains no glob wildcards, i.e. the user
    /// named a concrete path rather than a pattern. Says nothing about whether
    /// the path exists on disk.
    fn has_no_wildcards(&self) -> bool {
        self.as_path().components().all(|comp| match comp {
            Component::Normal(part) => !Self::contains_glob_char(part),
            _ => true,
        })
    }

    /// Returns true if this pattern is "explicit" - i.e., it has no wildcards
    /// and directly specifies a file path that exists.
    fn is_explicit_file_pattern(&self) -> bool {
        self.has_no_wildcards() && self.as_path().is_file()
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
        // A trailing component with an extension is a file name, not part of the
        // directory prefix -- unless it is a real directory that happens to have a
        // dot in its name (e.g. `my.project/`), which the walk must keep as a root.
        if path.extension().is_some() && !path.is_dir() {
            path.pop();
        }
        path
    }

    pub fn as_path(&self) -> &Path {
        Path::new(self.pattern.as_str())
    }

    pub fn as_str(&self) -> &str {
        self.pattern.as_str()
    }

    fn is_python_extension(ext: Option<&OsStr>) -> bool {
        ext.and_then(OsStr::to_str)
            .is_some_and(|ext| PYTHON_FILE_EXTENSIONS.contains(&ext))
    }

    /// Returns true if the given file should be included in results.
    /// Filters out non-Python files and dot files.
    ///
    /// If `is_explicit` is true, the extension check is skipped, allowing
    /// files without Python extensions to be included (for explicitly specified files).
    /// Dot files are always excluded regardless of the `is_explicit` flag.
    fn should_include_file(path: &Path, is_explicit: bool) -> bool {
        // Check if it's a Python file (skip for explicitly specified files)
        if !is_explicit && !Self::is_python_extension(path.extension()) {
            return false;
        }

        // Check if it's a dot file (always excluded)
        if let Some(file_name) = path.file_name().and_then(OsStr::to_str)
            && file_name.starts_with('.')
        {
            return false;
        }

        true
    }

    #[cfg(test)]
    fn resolve_pattern(pattern: &str, filter: &GlobFilter) -> anyhow::Result<Vec<PathBuf>> {
        Ok(Globs::new(vec![pattern.to_owned()])?
            .filtered_files_iter(filter, None)?
            .collect())
    }

    /// Returns true if the given file matches this glob. The precompiled
    /// `<pattern>/**` variant lets a glob naming a directory match the files
    /// beneath it.
    pub fn matches(&self, file: &Path) -> bool {
        self.pattern.matches_path(file)
            || self
                .dir_pattern
                .as_ref()
                .is_some_and(|dir_pattern| dir_pattern.matches_path(file))
    }
}

impl Debug for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern.as_str())
    }
}

impl Display for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern.as_str())
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
    #[cfg(test)]
    fn files(&self, filter: &GlobFilter, limit: Option<usize>) -> anyhow::Result<Vec<PathBuf>> {
        Ok(Globs(vec![self.clone()])
            .filtered_files_iter(filter, limit)?
            .collect())
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
    /// Directory-style matching is handled by each [`Glob`]'s precomputed
    /// `<pattern>/**` variant.
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

    pub fn globs(&self) -> &[Glob] {
        &self.0
    }
}

impl Display for Globs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|g| g.to_string()).join(", "))
    }
}

impl Globs {
    /// Build the `ignore` file-type filter restricting the walk to Python files
    /// (and notebooks), so the walker never hands us unrelated files.
    fn python_file_types() -> anyhow::Result<Types> {
        let mut builder = TypesBuilder::new();
        for extension in PYTHON_FILE_EXTENSIONS {
            builder
                .add("pyreflypython", &format!("*.{extension}"))
                .with_context(|| {
                    format!("While constructing file type filter for extension `{extension}`")
                })?;
        }
        builder.select("pyreflypython");
        builder
            .build()
            .context("While constructing Python file type filter")
    }

    fn merged_files_iter(
        mut explicit_files: Vec<PathBuf>,
        mut walked_files: Vec<PathBuf>,
    ) -> impl Iterator<Item = PathBuf> {
        explicit_files.sort();
        walked_files.sort();

        let mut explicit_files = explicit_files.into_iter().peekable();
        let mut walked_files = walked_files.into_iter().peekable();
        std::iter::from_fn(move || {
            let path = match (explicit_files.peek(), walked_files.peek()) {
                (Some(explicit), Some(walked)) if explicit <= walked => explicit_files.next(),
                (Some(_), Some(_)) => walked_files.next(),
                (Some(_), None) => explicit_files.next(),
                (None, Some(_)) => walked_files.next(),
                (None, None) => return None,
            }
            .expect("peeked path should be available");
            while explicit_files.peek() == Some(&path) {
                explicit_files.next();
            }
            while walked_files.peek() == Some(&path) {
                walked_files.next();
            }
            Some(path)
        })
    }

    fn non_nested_roots(mut roots: Vec<PathBuf>) -> Vec<PathBuf> {
        roots.sort();
        roots.dedup();
        // `Path::starts_with` does not treat `.` as a prefix of relative paths
        // like `tests`, but walking `.` already covers every nested relative root.
        if roots.first().is_some_and(|root| root == Path::new(".")) {
            return vec![PathBuf::from(".")];
        }

        let mut non_nested_roots: Vec<PathBuf> = Vec::new();
        for root in roots {
            if !non_nested_roots
                .iter()
                .any(|existing| root.starts_with(existing))
            {
                non_nested_roots.push(root);
            }
        }
        non_nested_roots
    }

    /// The directories to walk for the given (non-explicit) patterns: the
    /// non-wildcard prefix of each pattern, with nested roots removed so the same
    /// subtree is never walked twice.
    fn sorted_walk_roots(patterns: &[&Glob]) -> Vec<PathBuf> {
        let roots = patterns
            .iter()
            .filter_map(|pattern| {
                if pattern.is_explicit_file_pattern() {
                    return None;
                }
                if pattern.has_no_wildcards() {
                    return pattern
                        .as_path()
                        .is_dir()
                        .then(|| pattern.as_path().to_path_buf());
                }
                let root = pattern.get_glob_root();
                let root = if root.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    root
                };
                root.is_dir().then_some(root)
            })
            .collect::<Vec<_>>();
        Self::non_nested_roots(roots)
    }

    /// Construct an `ignore` walker over `roots`, restricted to Python files and
    /// with `filter` installed as a `filter_entry` predicate so excluded, hidden,
    /// and gitignored directories are pruned during traversal. Shared by the
    /// sorted and parallel walkers; returns `None` when there are no roots.
    fn walk_builder(roots: &[PathBuf], filter: &GlobFilter) -> anyhow::Result<Option<WalkBuilder>> {
        let Some((first_root, rest_roots)) = roots.split_first() else {
            return Ok(None);
        };

        let mut builder = WalkBuilder::new(first_root);
        for root in rest_roots {
            builder.add(root);
        }
        // We apply ignore files and excludes ourselves via `GlobFilter`, so turn
        // off the walker's built-in filtering and use it only to enumerate files.
        builder.standard_filters(false);
        builder.hidden(false);
        builder.follow_links(true);
        builder.types(Self::python_file_types()?);

        // `filter_entry` requires a `Fn + Send + Sync + 'static` predicate for both
        // the serial and parallel walkers, so move an owned filter clone in.
        let walk_filter = filter.for_walk();
        builder.filter_entry(move |entry| {
            let is_dir = entry
                .file_type()
                .is_some_and(|file_type| file_type.is_dir());
            !walk_filter.is_excluded_with_file_type(entry.path(), is_dir)
        });

        Ok(Some(builder))
    }

    fn should_skip_path_error(filter: &GlobFilter, path: &Path) -> bool {
        filter.is_excluded_with_file_type(path, false)
            || filter.is_excluded_with_file_type(path, true)
    }

    fn should_skip_walk_error(
        filter: &GlobFilter,
        error: &ignore::Error,
        path: Option<&Path>,
    ) -> bool {
        match error {
            ignore::Error::Partial(errors) => errors
                .iter()
                .all(|error| Self::should_skip_walk_error(filter, error, path)),
            ignore::Error::WithPath { path, err } => {
                Self::should_skip_walk_error(filter, err, Some(path))
            }
            ignore::Error::WithLineNumber { err, .. } | ignore::Error::WithDepth { err, .. } => {
                Self::should_skip_walk_error(filter, err, path)
            }
            ignore::Error::Loop { child, .. } => Self::should_skip_path_error(filter, child),
            ignore::Error::Io(_)
            | ignore::Error::Glob { .. }
            | ignore::Error::UnrecognizedFileType(_)
            | ignore::Error::InvalidDefinition => {
                path.is_some_and(|path| Self::should_skip_path_error(filter, path))
            }
        }
    }

    fn broken_symlink_walk_error(
        error: &ignore::Error,
        path: Option<&Path>,
    ) -> Option<(PathBuf, PathBuf)> {
        match error {
            ignore::Error::Partial(errors) => errors
                .iter()
                .find_map(|error| Self::broken_symlink_walk_error(error, path)),
            ignore::Error::WithPath { path, err } => {
                Self::broken_symlink_walk_error(err, Some(path))
            }
            ignore::Error::WithLineNumber { err, .. } | ignore::Error::WithDepth { err, .. } => {
                Self::broken_symlink_walk_error(err, path)
            }
            ignore::Error::Io(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let path = path?;
                let metadata = std::fs::symlink_metadata(path).ok()?;
                if metadata.file_type().is_symlink() {
                    Some((path.to_path_buf(), std::fs::read_link(path).ok()?))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn walk_error(error: ignore::Error) -> anyhow::Error {
        if let Some((path, target)) = Self::broken_symlink_walk_error(&error, None) {
            anyhow::Error::new(error).context(format!(
                "Broken symlink when walking project files: `{}` points to {:?}",
                path.display(),
                target,
            ))
        } else {
            anyhow::Error::new(error).context("When walking project files")
        }
    }

    /// Single-threaded, lexicographically-sorted walk used when a `limit` is set
    /// (workspace indexing), so a truncated result is deterministic.
    fn walk_files_sorted(
        &self,
        roots: &[PathBuf],
        filter: &GlobFilter,
        limit: usize,
    ) -> anyhow::Result<Vec<PathBuf>> {
        let Some(mut builder) = Self::walk_builder(roots, filter)? else {
            return Ok(Vec::new());
        };
        builder.sort_by_file_path(|a, b| a.cmp(b));

        let mut result = Vec::new();
        for entry in builder.build() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if Self::should_skip_walk_error(filter, &error, None) => continue,
                Err(error) => return Err(Self::walk_error(error)),
            };
            if entry
                .file_type()
                .is_some_and(|file_type| file_type.is_dir())
            {
                continue;
            }
            let path = entry.into_path();
            if self.matches(&path) && Glob::should_include_file(&path, false) {
                result.push(path);
                if result.len() >= limit {
                    break;
                }
            }
        }
        Ok(result)
    }

    /// Parallel walk used for unlimited discovery (full checks).
    fn walk_files_parallel(
        &self,
        roots: &[PathBuf],
        filter: &GlobFilter,
    ) -> anyhow::Result<Vec<PathBuf>> {
        let Some(mut builder) = Self::walk_builder(roots, filter)? else {
            return Ok(Vec::new());
        };

        builder.threads(
            available_parallelism()
                .map_or(1, NonZeroUsize::get)
                .min(MAX_PARALLEL_WALK_THREADS),
        );

        struct WalkedFiles<'a> {
            result: &'a Mutex<Vec<PathBuf>>,
            local_files: Vec<PathBuf>,
        }

        impl Drop for WalkedFiles<'_> {
            fn drop(&mut self) {
                if !self.local_files.is_empty() {
                    self.result.lock().append(&mut self.local_files);
                }
            }
        }

        let result = Mutex::new(Vec::new());
        let walk_error = Mutex::new(None);
        builder.build_parallel().run(|| {
            let mut walked_files = WalkedFiles {
                result: &result,
                local_files: Vec::new(),
            };
            let walk_error = &walk_error;
            Box::new(move |entry| match entry {
                Ok(entry) => {
                    if entry
                        .file_type()
                        .is_some_and(|file_type| file_type.is_dir())
                    {
                        return WalkState::Continue;
                    }
                    let path = entry.into_path();
                    if self.matches(&path) && Glob::should_include_file(&path, false) {
                        walked_files.local_files.push(path);
                    }
                    WalkState::Continue
                }
                Err(error) => {
                    if Self::should_skip_walk_error(filter, &error, None) {
                        return WalkState::Continue;
                    }
                    let error = Self::walk_error(error);
                    let mut walk_error = walk_error.lock();
                    if walk_error.is_none() {
                        *walk_error = Some(error);
                    }
                    WalkState::Quit
                }
            })
        });

        if let Some(error) = walk_error.into_inner() {
            return Err(error);
        }
        Ok(result.into_inner())
    }

    fn no_matched_files_error(&self) -> anyhow::Error {
        if self.0.is_empty() {
            return anyhow::anyhow!("There are no patterns to match Python files.");
        }
        if self.0.len() == 1 {
            let pattern = &self.0[0];
            let pattern_str = pattern.as_str();
            // A lone concrete path (no wildcards) that resolved to nothing does
            // not exist on disk -- a clearer error than "no files matched" for a
            // path the user named directly.
            if pattern.has_no_wildcards() && !pattern.as_path().exists() {
                return anyhow::anyhow!("Path `{}` does not exist", pattern_str);
            }
            return anyhow::anyhow!("No Python files matched pattern `{}`", pattern_str);
        }
        anyhow::anyhow!(
            "No Python files matched patterns {}",
            self.0
                .iter()
                .map(|p| format!("`{}`", p.as_str()))
                .join(", "),
        )
    }

    fn filtered_files_iter(
        &self,
        filter: &GlobFilter,
        limit: Option<usize>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf>>> {
        if limit == Some(0) {
            return Ok(Box::new(std::iter::empty()));
        }
        let mut active_patterns = Vec::new();
        let mut explicit_files = Vec::new();
        for pattern in &self.0 {
            // Skip include patterns that are themselves excluded (e.g. the
            // default `**/*.ipynb` include when the user sets
            // `project-excludes = ["**/*.ipynb"]`). Warn so the user knows
            // this pattern is not being used, but don't abort discovery.
            if filter.is_excluded(pattern.as_path()) {
                warn!(
                    "Skipping include pattern `{}` because it is matched by \
                     `project-excludes` or an ignore file.\n{}",
                    pattern.as_str(),
                    filter,
                );
                continue;
            }
            active_patterns.push(pattern);
            // An explicit concrete file is included directly, bypassing the
            // extension filter that wildcard matches must satisfy.
            if pattern.is_explicit_file_pattern()
                && Glob::should_include_file(pattern.as_path(), true)
            {
                explicit_files.push(pattern.as_path().to_path_buf());
            }
        }

        let roots = Self::sorted_walk_roots(&active_patterns);
        let walked_files = if let Some(limit) = limit {
            self.walk_files_sorted(&roots, filter, limit)?
        } else {
            self.walk_files_parallel(&roots, filter)?
        };
        if explicit_files.is_empty() && walked_files.is_empty() {
            return Err(self.no_matched_files_error());
        }
        Ok(Box::new(
            Self::merged_files_iter(explicit_files, walked_files).take(limit.unwrap_or(usize::MAX)),
        ))
    }

    pub fn files_iter(&self) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf>>> {
        self.filtered_files_iter(&GlobFilter::empty(), None)
    }

    pub fn covers(&self, path: &Path) -> bool {
        self.matches(path)
    }
}

/// A struct which allows filtering by matching a high-priority [`Globs`] of excludes
/// and several ignore files. The first positive (ignore) or negative (allowlist)
/// match that's found from the following order is what's used.
/// 1. `excludes`: user-provided paths, either from a config or CLI.
/// 2. `.gitignore`: if one exists from an upward search from `root`, the first
///    positive or negative match (`!`) is used
/// 3. `.ignore`: if it exists, behaves similar to `.gitignore`
/// 4. `.git/info/excludes`: if it exists, behaves similar to `.gitignore`
#[derive(Debug)]
pub struct GlobFilter {
    excludes: Globs,
    ignores: Vec<Gitignore>,
    ignore_paths: Vec<PathBuf>,
    errors: Vec<anyhow::Error>,
    hidden_dir_filter: HiddenDirFilter,
}

/// Controls whether paths with hidden directory components (names starting
/// with `.`, excluding `.` and `..`) are excluded during file filtering.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HiddenDirFilter {
    /// No filtering of hidden directories.
    Disabled,
    /// Exclude paths containing any hidden directory component.
    All,
    /// Exclude paths with hidden directory components relative to any of the
    /// given roots. Hidden ancestors above each root are allowed, so that
    /// projects living under hidden directories (e.g.
    /// `~/.codex/worktrees/XXX/project/`) are not falsely excluded.
    RelativeTo(Vec<PathBuf>),
}

impl Display for GlobFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`project-excludes`: {}, ignore files [{}]",
            self.excludes,
            self.ignore_paths.iter().map(|p| p.display()).join(", ")
        )?;
        Ok(())
    }
}

impl PartialEq for GlobFilter {
    fn eq(&self, other: &Self) -> bool {
        self.excludes == other.excludes
            && self.ignore_paths == other.ignore_paths
            && self.hidden_dir_filter == other.hidden_dir_filter
    }
}

impl Eq for GlobFilter {}

impl Hash for GlobFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.excludes.hash(state);
        self.ignore_paths.hash(state);
        self.hidden_dir_filter.hash(state);
    }
}

impl GlobFilter {
    /// Create a new `GlobFilter` with the given `Globs` as highest-priority excludes.
    /// If `ignore_file_search_start` is provided, it is where the upward search for
    /// ignore files will originate from. Typically, this should be your project root.
    /// `hidden_dir_filter` controls whether paths with hidden directory components
    /// (starting with `.`) are excluded.
    pub fn new(
        excludes: Globs,
        ignorefile_search_start: Option<&Path>,
        hidden_dir_filter: HiddenDirFilter,
    ) -> Self {
        let (ignores, errors, ignore_paths) = if let Some(root) = ignorefile_search_start {
            Self::ignore_files(root)
        } else {
            (vec![], vec![], vec![])
        };

        Self {
            excludes,
            ignores,
            ignore_paths,
            errors,
            hidden_dir_filter,
        }
    }

    pub fn empty() -> Self {
        Self {
            excludes: Globs::empty(),
            ignores: vec![],
            ignore_paths: vec![],
            errors: vec![],
            hidden_dir_filter: HiddenDirFilter::Disabled,
        }
    }

    pub fn ignore_files(root: &Path) -> (Vec<Gitignore>, Vec<anyhow::Error>, Vec<PathBuf>) {
        let found_ignores = IGNORE_FILES_SEARCH
            .iter()
            .filter_map(|s| s.directory_absolute(root));
        let mut errors = vec![];
        let mut ignores = vec![];
        let mut ignore_paths = vec![];
        for item in found_ignores {
            let (ignore_file, ignore_root) = &*item;
            let mut builder = GitignoreBuilder::new(ignore_root);
            if let Some(error) = builder.add(ignore_file) {
                errors.push(error.into());
            }
            match builder.build() {
                Ok(ignore) => ignores.push(ignore),
                Err(error) => errors.push(error.into()),
            }
            ignore_paths.push(ignore_file.to_owned());
        }
        (ignores, errors, ignore_paths)
    }

    /// Returns true if `path` contains a hidden directory component (starting
    /// with `.`, excluding `.` and `..`). When `root` is provided, only
    /// components below `root` are checked so that hidden ancestors above the
    /// project root are allowed. For a directory the final component is checked
    /// too, so the walker can prune a hidden directory before descending into it.
    fn has_hidden_component(path: &Path, root: Option<&Path>, is_dir: bool) -> bool {
        let relative = match root {
            Some(root) => path.strip_prefix(root).unwrap_or(path),
            None => path,
        };
        let components: Vec<_> = relative.components().collect();
        let dir_component_count = if is_dir {
            components.len()
        } else {
            components.len().saturating_sub(1)
        };
        let dir_components = &components[..dir_component_count];
        dir_components.iter().any(|c| {
            if let Component::Normal(s) = c {
                s.as_encoded_bytes().first() == Some(&b'.')
            } else {
                false
            }
        })
    }

    /// A copy of this filter for use inside a walk closure. `errors` are dropped
    /// because `anyhow::Error` is not `Clone`; walk-time errors are surfaced
    /// separately by the walk itself.
    fn for_walk(&self) -> Self {
        Self {
            excludes: self.excludes.clone(),
            ignores: self.ignores.clone(),
            ignore_paths: self.ignore_paths.clone(),
            errors: Vec::new(),
            hidden_dir_filter: self.hidden_dir_filter.clone(),
        }
    }

    // Does this path match (either positively or negatively), the `excludes` or ignore
    // files found.
    pub fn is_excluded(&self, path: &Path) -> bool {
        self.is_excluded_with_file_type(path, path.is_dir())
    }

    /// Like [`Self::is_excluded`], but with the file type supplied by the caller.
    /// The walker already knows whether each entry is a directory, so it passes
    /// that in to avoid an extra `stat`.
    fn is_excluded_with_file_type(&self, path: &Path, is_dir: bool) -> bool {
        if self.excludes.matches(path) {
            return true;
        }

        match &self.hidden_dir_filter {
            HiddenDirFilter::Disabled => {}
            HiddenDirFilter::All => {
                if Self::has_hidden_component(path, None, is_dir) {
                    return true;
                }
            }
            HiddenDirFilter::RelativeTo(roots) => {
                // Find the most specific root that is a prefix of this path.
                // If none match, fall back to checking all components.
                let root = roots
                    .iter()
                    .filter(|r| path.starts_with(r))
                    .max_by_key(|r| r.as_os_str().len());
                if Self::has_hidden_component(path, root.map(|r| r.as_path()), is_dir) {
                    return true;
                }
            }
        }

        for ignore in &self.ignores {
            let ignore_root = ignore.path();
            if path.starts_with(ignore_root) {
                match ignore.matched_path_or_any_parents(path, is_dir) {
                    Match::None => (),
                    Match::Whitelist(_) => return false,
                    Match::Ignore(_) => return true,
                }
            }
        }
        false
    }

    /// Get the errors from this glob, replacing them with an empty list.
    pub fn errors(&mut self) -> Vec<anyhow::Error> {
        std::mem::take(&mut self.errors)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FilteredGlobs {
    includes: Globs,
    filter: GlobFilter,
}

impl FilteredGlobs {
    /// Build a new `FilteredGlobs` from the given `includes` and `excludes`.
    /// If an `ignorefile_search_start` is provided, it is the path from which we will
    /// perform an upward search for applicable ignore files, which will be used when
    /// filtering out files from our glob search. `hidden_dir_filter` controls whether
    /// paths with hidden directory components are excluded.
    pub fn new(
        includes: Globs,
        excludes: Globs,
        ignorefile_search_start: Option<&Path>,
        hidden_dir_filter: HiddenDirFilter,
    ) -> Self {
        Self {
            includes,
            filter: GlobFilter::new(excludes, ignorefile_search_start, hidden_dir_filter),
        }
    }
}

impl Includes for FilteredGlobs {
    /// Given a glob pattern, return the directories that can contain files that match the pattern.
    fn roots(&self) -> Vec<PathBuf> {
        self.includes.roots()
    }

    fn files_iter(&self) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf> + '_>> {
        self.includes.filtered_files_iter(&self.filter, None)
    }

    fn covers(&self, path: &Path) -> bool {
        self.covers_ignoring_excludes(path) && !self.filter.is_excluded(path)
    }

    fn covers_ignoring_excludes(&self, path: &Path) -> bool {
        self.includes.covers(path)
    }

    fn errors(&mut self) -> Vec<anyhow::Error> {
        self.filter.errors()
    }
}

impl FilteredGlobs {
    /// Same as `files_iter`, but with an upper limit on the number of files returned.
    /// This is useful for indexing of workspaces, where we don't want to index too many files
    /// when the user decides to open VSCode at the root of the filesystem.
    pub fn files_iter_with_limit(
        &self,
        limit: usize,
    ) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf> + '_>> {
        self.includes.filtered_files_iter(&self.filter, Some(limit))
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
        let inputs: Vec<&str> = vec![
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
        let inputs: Vec<String> = inputs.into_iter().map(String::from).collect();

        let f = |root: &str, expected: Vec<&str>| {
            let expected: Vec<PathBuf> = expected.into_iter().map(PathBuf::from).collect();
            let inputs = inputs.clone();
            let root = root.to_owned();

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
        );
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
            let glob_files = Globs::new_with_root(root, vec![pattern.to_owned()])
                .unwrap()
                .files_iter()?
                .collect::<Vec<_>>();
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
                .files(&GlobFilter::empty(), None)
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
            .files(&GlobFilter::empty(), None)
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

        let mut sorted_globs = Glob::resolve_pattern(&pattern, &GlobFilter::empty()).unwrap();
        sorted_globs.sort();
        assert_eq!(sorted_globs, vec![root.join("a/b.py"), root.join("a/c.py")]);
        assert!(
            Glob::resolve_pattern(
                &pattern,
                &GlobFilter::new(
                    Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap(),
                    None,
                    HiddenDirFilter::Disabled,
                ),
            )
            .is_err()
        );
        assert!(
            Glob::new(pattern.clone())
                .unwrap()
                .files(
                    &GlobFilter::new(
                        Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap(),
                        None,
                        HiddenDirFilter::Disabled,
                    ),
                    None
                )
                .is_err()
        );
        // double check that <path>/** will also match <path>
        assert!(
            Globs::new(vec![root.to_string_lossy().to_string()])
                .unwrap()
                .filtered_files_iter(
                    &GlobFilter::new(
                        Globs::new(vec![root.join("**").to_string_lossy().to_string()]).unwrap(),
                        None,
                        HiddenDirFilter::Disabled,
                    ),
                    None
                )
                .is_err()
        );
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &GlobFilter::new(
                    Globs::new(vec![root.join("a/c.py").to_string_lossy().to_string()]).unwrap(),
                    None,
                    HiddenDirFilter::Disabled,
                )
            )
            .unwrap(),
            vec![root.join("a/b.py")],
        );
        assert!(
            Glob::resolve_pattern(
                &pattern,
                &GlobFilter::new(
                    Globs::new(vec![root.join("a").to_string_lossy().to_string()]).unwrap(),
                    None,
                    HiddenDirFilter::Disabled,
                )
            )
            .is_err()
        );
        assert_eq!(
            Glob::resolve_pattern(
                &pattern,
                &GlobFilter::new(
                    Globs::new(vec![root.join("a/b*").to_string_lossy().to_string()]).unwrap(),
                    None,
                    HiddenDirFilter::Disabled,
                ),
            )
            .unwrap(),
            vec![root.join("a/c.py")],
        );
    }

    /// Tests that hidden directory ancestors above the project root do not
    /// cause files inside the project to be excluded.
    #[test]
    fn test_hidden_dir_ancestor_does_not_exclude_project() {
        let tempdir = tempfile::tempdir().unwrap();
        let base = tempdir.path();

        // Simulate a project inside a hidden ancestor directory:
        //   base/.hidden_ancestor/project/src/foo.py
        //   base/.hidden_ancestor/project/.venv/lib.py
        TestPath::setup_test_directory(
            base,
            vec![TestPath::dir(
                ".hidden_ancestor",
                vec![TestPath::dir(
                    "project",
                    vec![
                        TestPath::dir("src", vec![TestPath::file("foo.py")]),
                        TestPath::dir(".venv", vec![TestPath::file("lib.py")]),
                    ],
                )],
            )],
        );

        let project_root = base.join(".hidden_ancestor/project");
        let filter = GlobFilter::new(
            Globs::empty(),
            Some(&project_root),
            HiddenDirFilter::RelativeTo(vec![project_root.clone()]),
        );

        // A file inside the project should NOT be excluded, even though
        // .hidden_ancestor is a hidden dir in the absolute path.
        assert!(
            !filter.is_excluded(&project_root.join("src/foo.py")),
            "file inside project with hidden ancestor should not be excluded"
        );

        // A file inside a hidden dir *within* the project should still be excluded.
        assert!(
            filter.is_excluded(&project_root.join(".venv/lib.py")),
            "file inside hidden dir within project should be excluded"
        );
    }

    /// Tests hidden-dir filtering with `HiddenDirFilter::All` (checks all components).
    #[test]
    fn test_skip_hidden_dirs_no_root() {
        let filter = GlobFilter::new(Globs::empty(), None, HiddenDirFilter::All);

        assert!(
            filter.is_excluded(Path::new("/home/user/.venv/lib.py")),
            "hidden dir component should be excluded"
        );
        assert!(
            !filter.is_excluded(Path::new("/home/user/project/lib.py")),
            "path with no hidden dirs should not be excluded"
        );
        // `.` and `..` are Component::CurDir/ParentDir, not Component::Normal
        assert!(
            !filter.is_excluded(Path::new("/home/user/./lib.py")),
            "current-dir component should not trigger exclusion"
        );
        assert!(
            !filter.is_excluded(Path::new("/home/user/../lib.py")),
            "parent-dir component should not trigger exclusion"
        );
    }

    #[test]
    fn test_globfilter_finds_ignorefiles() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file_with_contents(
                    ".gitignore",
                    "**/gitignore_exclude\n!**/gitignore_include/**",
                ),
                TestPath::dir(
                    ".git",
                    vec![TestPath::dir(
                        "info",
                        vec![TestPath::file_with_contents(
                            "exclude",
                            "**/gitexclude_exclude",
                        )],
                    )],
                ),
                TestPath::dir(
                    "project",
                    vec![
                        TestPath::file("pyrefly.toml"),
                        TestPath::file_with_contents(
                            ".ignore",
                            // added gitignore_include here to show that .gitignore's allowlist,
                            // which will be found first will be preferred over anything later
                            "**/gitignore_include/**\nignore_exclude",
                        ),
                    ],
                ),
            ],
        );
        let filter = GlobFilter::new(
            Globs::empty(),
            Some(&root.join("project")),
            HiddenDirFilter::Disabled,
        );

        assert_eq!(
            filter.ignore_paths,
            vec![
                root.join(".gitignore"),
                root.join("project/.ignore"),
                root.join(".git/info/exclude"),
            ],
        );
        assert_eq!(filter.errors.len(), 0);
        assert_eq!(filter.ignores.len(), 3);
    }

    #[test]
    fn test_gitignore_globfilter() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file_with_contents(
                    ".gitignore",
                    "**/*.gitignore_exclude\n!**/include/**",
                ),
                TestPath::dir(
                    ".git",
                    vec![TestPath::dir(
                        "info",
                        vec![TestPath::file_with_contents("exclude", "**/*.gitexclude")],
                    )],
                ),
                TestPath::dir(
                    "project",
                    vec![
                        TestPath::file("pyrefly.toml"),
                        TestPath::file_with_contents(
                            ".ignore",
                            // added gitignore_include here to show that .gitignore's allowlist,
                            // which will be found first will be preferred over anything later
                            "**/include/**\n**/*.ignore_exclude",
                        ),
                    ],
                ),
            ],
        );

        let project_root = root.join("project");
        let filter = GlobFilter::new(
            Globs::new_with_root(&project_root, vec!["exclude_glob/**".to_owned()]).unwrap(),
            Some(&project_root),
            HiddenDirFilter::Disabled,
        );

        // do non-excluded files get excluded
        assert!(!filter.is_excluded(&project_root.join("my_file.py")));

        // test exclude globs
        assert!(filter.is_excluded(&project_root.join("exclude_glob/my_file.py")));

        // test `.gitignore`
        assert!(filter.is_excluded(&project_root.join("my_file.gitignore_exclude")));
        // Even though this is included in `.ignore`'s excludes, `.gitignore` takes priority,
        // which allowlists it
        assert!(!filter.is_excluded(&project_root.join("include/test.gitignore_exclude")));

        // test `.ignore`
        assert!(filter.is_excluded(&project_root.join("test/my_file.ignore_exclude")));

        // test `.git/info/exclude`
        assert!(filter.is_excluded(&project_root.join("my_file.gitexclude")));
    }

    #[test]
    fn test_is_excluded_on_file_outside_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "project",
                vec![TestPath::file_with_contents(".ignore", "*.py")],
            )],
        );
        let project_root = root.join("project");
        let filter = GlobFilter::new(
            Globs::empty(),
            Some(&project_root),
            HiddenDirFilter::Disabled,
        );
        assert!(!filter.is_excluded(&root.join("my_file.py")));
    }

    #[test]
    fn test_explicitly_specified_files_without_extension() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file_with_contents("myscript", "#!/usr/bin/env python3\nprint('hello')"),
                TestPath::file_with_contents("another_script", "import sys\nprint(sys.version)"),
                TestPath::dir(
                    "scripts",
                    vec![
                        TestPath::file_with_contents("tool", "# Python script\nprint('tool')"),
                        TestPath::file("regular.py"),
                    ],
                ),
            ],
        );

        // Test single file without extension
        let files = Globs::new_with_root(root, vec!["myscript".to_owned()])
            .unwrap()
            .files_iter()
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], root.join("myscript"));

        // Test multiple files without extensions
        let files = Globs::new_with_root(
            root,
            vec!["myscript".to_owned(), "another_script".to_owned()],
        )
        .unwrap()
        .files_iter()
        .unwrap()
        .collect::<Vec<_>>();
        assert_eq!(files.len(), 2);

        // Test file in subdirectory without extension
        let files = Globs::new_with_root(root, vec!["scripts/tool".to_owned()])
            .unwrap()
            .files_iter()
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], root.join("scripts/tool"));

        // Test that glob patterns still filter by extension (wildcards should still require .py extension)
        let files = Globs::new_with_root(root, vec!["*".to_owned()])
            .unwrap()
            .files_iter()
            .unwrap()
            .collect::<Vec<_>>();
        // Should not include files without extensions when using wildcard
        assert!(!files.contains(&root.join("myscript")));
        assert!(!files.contains(&root.join("another_script")));
    }

    #[test]
    fn test_excluded_directory_is_pruned_from_walk() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("excluded", vec![TestPath::file("bad.py")]),
                TestPath::dir("src", vec![TestPath::file("ok.py")]),
            ],
        );

        let filtered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::new_with_root(root, vec!["excluded".to_owned()]).unwrap(),
            None,
            HiddenDirFilter::Disabled,
        );

        assert_eq!(
            filtered.files_iter().unwrap().collect::<Vec<_>>(),
            vec![root.join("src/ok.py")]
        );
    }

    #[test]
    fn test_hidden_directory_filter_is_respected_by_walk() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(".venv", vec![TestPath::file("bad.py")]),
                TestPath::dir("src", vec![TestPath::file("ok.py")]),
            ],
        );

        let unfiltered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::empty(),
            None,
            HiddenDirFilter::Disabled,
        );
        let mut files = unfiltered.files_iter().unwrap().collect::<Vec<_>>();
        files.sort();
        assert_eq!(
            files,
            vec![root.join(".venv/bad.py"), root.join("src/ok.py")]
        );

        let hidden_filtered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::empty(),
            None,
            HiddenDirFilter::RelativeTo(vec![root.to_path_buf()]),
        );

        assert_eq!(
            hidden_filtered.files_iter().unwrap().collect::<Vec<_>>(),
            vec![root.join("src/ok.py")]
        );
    }

    /// A directory whose name contains a dot (so it looks like it has a file
    /// extension) must still be used as a walk root rather than being truncated.
    #[test]
    fn test_walk_root_preserves_directory_with_extension() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("project.with.dot");
        std::fs::create_dir(&root).unwrap();
        TestPath::setup_test_directory(&root, vec![TestPath::file("ok.py")]);

        let includes = Globs::new_with_root(&root, vec!["**".to_owned()]).unwrap();

        assert_eq!(includes.roots(), vec![root]);
    }

    #[test]
    fn test_dot_walk_root_contains_nested_relative_roots() {
        assert_eq!(
            Globs::non_nested_roots(vec![PathBuf::from("."), PathBuf::from("tests")]),
            vec![PathBuf::from(".")]
        );
    }

    /// A limited walk is single-threaded and lexicographically sorted, so the
    /// truncated result is deterministic.
    #[test]
    fn test_files_iter_with_limit_is_dfs_sorted() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("z.py"),
                TestPath::dir("b", vec![TestPath::file("one.py")]),
                TestPath::dir(
                    "a",
                    vec![TestPath::file("two.py"), TestPath::file("one.py")],
                ),
            ],
        );

        let filtered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::empty(),
            None,
            HiddenDirFilter::Disabled,
        );
        let files = filtered
            .files_iter_with_limit(3)
            .unwrap()
            .collect::<Vec<_>>();
        let files = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            files,
            vec![
                Path::new("a/one.py"),
                Path::new("a/two.py"),
                Path::new("b/one.py"),
            ],
        );
    }

    #[test]
    fn test_files_iter_with_limit_zero_returns_no_files() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("a.py")]);

        let filtered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::empty(),
            None,
            HiddenDirFilter::Disabled,
        );

        assert!(
            filtered
                .files_iter_with_limit(0)
                .unwrap()
                .collect::<Vec<_>>()
                .is_empty()
        );
    }

    #[test]
    fn test_walk_error_classification() {
        fn io_error(kind: std::io::ErrorKind) -> ignore::Error {
            ignore::Error::Io(std::io::Error::from(kind))
        }

        fn missing_path(path: &str) -> ignore::Error {
            path_error(path, io_error(std::io::ErrorKind::NotFound))
        }

        fn path_error(path: &str, err: ignore::Error) -> ignore::Error {
            ignore::Error::WithPath {
                path: PathBuf::from(path),
                err: Box::new(err),
            }
        }

        fn missing_at_depth(path: &str) -> ignore::Error {
            ignore::Error::WithDepth {
                depth: 1,
                err: Box::new(missing_path(path)),
            }
        }

        fn missing_on_line(path: &str) -> ignore::Error {
            ignore::Error::WithLineNumber {
                line: 1,
                err: Box::new(missing_path(path)),
            }
        }

        let filter = GlobFilter::new(Globs::empty(), None, HiddenDirFilter::Disabled);
        let excluded_filter = GlobFilter::new(
            Globs::new(vec!["excluded/**".to_owned()]).unwrap(),
            None,
            HiddenDirFilter::Disabled,
        );
        let skip = |error: &ignore::Error| Globs::should_skip_walk_error(&filter, error, None);

        let all_excluded_errors = ignore::Error::Partial(vec![
            missing_path("excluded/.eslintignore"),
            path_error(
                "excluded/denied.py",
                io_error(std::io::ErrorKind::PermissionDenied),
            ),
        ]);
        let symlink_loop = ignore::Error::Loop {
            ancestor: PathBuf::from("root"),
            child: PathBuf::from("root/loop"),
        };
        let loop_with_excluded_source = ignore::Error::Loop {
            ancestor: PathBuf::from("root"),
            child: PathBuf::from("excluded/loop"),
        };
        let loop_with_excluded_destination = ignore::Error::Loop {
            ancestor: PathBuf::from("excluded"),
            child: PathBuf::from("root/loop"),
        };

        assert!(!skip(&io_error(std::io::ErrorKind::NotFound)));
        assert!(!skip(&missing_path(".eslintignore")));
        assert!(!skip(&missing_path("generated.txt")));
        assert!(!skip(&missing_at_depth(".eslintignore")));
        assert!(!skip(&missing_on_line("generated.txt")));
        assert!(!skip(&symlink_loop));
        assert!(Globs::should_skip_walk_error(
            &excluded_filter,
            &missing_path("excluded/broken.py"),
            None,
        ));
        assert!(Globs::should_skip_walk_error(
            &excluded_filter,
            &path_error(
                "excluded/denied.py",
                io_error(std::io::ErrorKind::PermissionDenied),
            ),
            None,
        ));
        assert!(Globs::should_skip_walk_error(
            &excluded_filter,
            &path_error(
                "excluded/glob",
                ignore::Error::Glob {
                    glob: Some("[".to_owned()),
                    err: "invalid range pattern".to_owned(),
                },
            ),
            None,
        ));
        assert!(Globs::should_skip_walk_error(
            &excluded_filter,
            &loop_with_excluded_source,
            None,
        ));
        assert!(!Globs::should_skip_walk_error(
            &excluded_filter,
            &loop_with_excluded_destination,
            None,
        ));
        assert!(Globs::should_skip_walk_error(
            &excluded_filter,
            &all_excluded_errors,
            None,
        ));

        assert!(!skip(&missing_path("missing.py")));
        assert!(!skip(&missing_path(".hidden.py")));
        assert!(!skip(&missing_path("missing")));
        assert!(!skip(&ignore::Error::Io(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied
        ))));
        assert!(!skip(&ignore::Error::Glob {
            glob: Some("[".to_owned()),
            err: "invalid range pattern".to_owned(),
        }));
        assert!(!skip(&ignore::Error::UnrecognizedFileType(
            "python".to_owned()
        )));
        assert!(!skip(&ignore::Error::InvalidDefinition));
        let mixed_error = ignore::Error::Partial(vec![
            missing_path("excluded/.eslintignore"),
            missing_path("missing.py"),
        ]);
        assert!(!Globs::should_skip_walk_error(
            &excluded_filter,
            &mixed_error,
            None,
        ));
        let mixed_error = ignore::Error::Partial(vec![
            missing_path("excluded/.eslintignore"),
            io_error(std::io::ErrorKind::PermissionDenied),
        ]);
        assert!(!Globs::should_skip_walk_error(
            &excluded_filter,
            &mixed_error,
            None,
        ));
    }

    #[cfg(unix)]
    #[test]
    fn test_dangling_symlink_errors() {
        fn check(link_name: &str, target: &str) {
            let tempdir = tempfile::tempdir().unwrap();
            let root = tempdir.path();
            TestPath::setup_test_directory(root, vec![TestPath::file("ok.py")]);
            std::os::unix::fs::symlink(target, root.join(link_name)).unwrap();

            let filtered = FilteredGlobs::new(
                Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
                Globs::empty(),
                None,
                HiddenDirFilter::Disabled,
            );

            let err = filtered.files_iter().err().unwrap();
            assert!(
                format!("{err:#}").contains("Broken symlink when walking project files"),
                "got: {err:#}",
            );
            assert!(format!("{err:#}").contains(link_name), "got: {err:#}");
            assert!(
                format!("{err:#}").contains(&format!("{:?}", PathBuf::from(target))),
                "got: {err:#}",
            );
            let err = filtered.files_iter_with_limit(10).err().unwrap();
            assert!(
                format!("{err:#}").contains("Broken symlink when walking project files"),
                "got: {err:#}",
            );
            assert!(format!("{err:#}").contains(link_name), "got: {err:#}");
            assert!(
                format!("{err:#}").contains(&format!("{:?}", PathBuf::from(target))),
                "got: {err:#}",
            );
        }

        check(".eslintignore", "missing-target");
        check("generated.txt", "missing-target");
        check("broken.py", "missing-target\n");
        check("package", "missing-target");
    }

    /// An excluded dangling symlink is not part of the Python file set and
    /// should not fail traversal even if its name looks like Python.
    #[cfg(unix)]
    #[test]
    fn test_excluded_dangling_python_symlink_is_skipped() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("ok.py")]);
        std::os::unix::fs::symlink("missing-target", root.join("excluded.py")).unwrap();

        let filtered = FilteredGlobs::new(
            Globs::new_with_root(root, vec!["**".to_owned()]).unwrap(),
            Globs::new_with_root(root, vec!["excluded.py".to_owned()]).unwrap(),
            None,
            HiddenDirFilter::Disabled,
        );

        assert_eq!(
            filtered.files_iter().unwrap().collect::<Vec<_>>(),
            vec![root.join("ok.py")]
        );
        assert_eq!(
            filtered
                .files_iter_with_limit(10)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![root.join("ok.py")]
        );
    }

    /// Regression test: setting `project-excludes = ["**/*.ipynb"]` should not
    /// prevent `.py` files from being discovered. The default project-includes
    /// contains both `**/*.py*` and `**/*.ipynb`. When the exclude pattern
    /// `**/*.ipynb` matches the include pattern `**/*.ipynb`, that include is
    /// skipped, but the remaining `**/*.py*` include should still find .py files.
    #[test]
    fn test_project_excludes_ipynb_does_not_break_py_discovery() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("main.py"),
                TestPath::file("helper.py"),
                TestPath::file("notebook.ipynb"),
                TestPath::dir(
                    "subdir",
                    vec![
                        TestPath::file("module.py"),
                        TestPath::file("analysis.ipynb"),
                    ],
                ),
            ],
        );

        // Reproduce the real config scenario: default project-includes has both
        // `**/*.py*` and `**/*.ipynb`, and the user sets project-excludes to
        // `["**/*.ipynb"]`.
        let includes =
            Globs::new_with_root(root, vec!["**/*.py*".to_owned(), "**/*.ipynb".to_owned()])
                .unwrap();
        let excludes = Globs::new_with_root(root, vec!["**/*.ipynb".to_owned()]).unwrap();
        let filtered = FilteredGlobs::new(includes, excludes, None, HiddenDirFilter::Disabled);

        let mut files = filtered
            .files_iter()
            .expect(
                "file discovery should succeed: the excluded **/*.ipynb include \
                 pattern should be skipped, not abort the entire search",
            )
            .collect::<Vec<_>>();
        files.sort();
        let files: Vec<&Path> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap())
            .collect();
        // Only .py files should be returned; .ipynb files are excluded.
        assert_eq!(
            files,
            vec![
                Path::new("helper.py"),
                Path::new("main.py"),
                Path::new("subdir/module.py"),
            ],
        );
    }

    /// A lone non-existent concrete path now reports a clear "does not exist"
    /// error instead of the generic "no files matched". A non-existent path
    /// alongside a real one is still silently skipped, so tools that pass a mix
    /// of paths keep working. https://github.com/facebook/pyrefly/issues/3647
    #[test]
    fn test_explicit_nonexistent_path() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("real.py")]);

        // A lone non-existent concrete path errors with a clear message.
        let err = Globs::new_with_root(root, vec!["does_not_exist.py".to_owned()])
            .unwrap()
            .filtered_files_iter(&GlobFilter::empty(), None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("does not exist"), "got: {err}");

        // A non-existent path alongside a real one is silently skipped.
        let files = Globs::new_with_root(
            root,
            vec!["real.py".to_owned(), "does_not_exist.py".to_owned()],
        )
        .unwrap()
        .filtered_files_iter(&GlobFilter::empty(), None)
        .unwrap()
        .collect::<Vec<_>>();
        assert_eq!(files, vec![root.join("real.py")]);
    }
}
