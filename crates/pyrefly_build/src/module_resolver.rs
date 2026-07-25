/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::fmt::Debug;
use std::io::Read;
use std::iter;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::locked_map::LockedMap;
use pyrefly_util::timer::Timer;
use regex::Regex;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

const PKGUTIL_DETECTION_MAX_BYTES: usize = 4096;

pub trait ModuleResolutionObserver {
    fn observe_stat(&self, elapsed_ns: u64);
    fn observe_read(&self, elapsed_ns: u64);
}

fn timed_stat(observer: Option<&dyn ModuleResolutionObserver>, f: impl FnOnce() -> bool) -> bool {
    match observer {
        None => f(),
        Some(observer) => {
            let start = Timer::start();
            let result = f();
            observer.observe_stat(start.elapsed_nanos());
            result
        }
    }
}

/// Matches the `__path__ = ...extend_path(...` assignment used by pkgutil-style
/// legacy namespace packages, in any of its common spellings:
///   __path__ = extend_path(__path__, __name__)
///   __path__ = pkgutil.extend_path(__path__, __name__)
///   __path__ = __import__('pkgutil').extend_path(__path__, __name__)
///
/// The pattern is anchored at the start of a (possibly indented) line: some
/// packages guard `extend_path` inside a conditional. It disallows `#` and
/// newlines between `=` and `extend_path`, avoiding comments and accidental
/// multi-line spans; `\b` avoids identifiers like `_extend_path`.
///
/// Known limitations: a call split across physical lines will not match, and
/// the same text inside a triple-quoted string would still match. Those cases
/// are rare enough that a bounded regex pass is preferable to parsing every
/// `__init__.py` encountered during discovery.
static PKGUTIL_EXTEND_PATH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*__path__\s*=\s*[^#\n]*\bextend_path\s*\(")
        .expect("PKGUTIL_EXTEND_PATH_PATTERN regex should be valid")
});

fn is_pkgutil_namespace(init_path: &Path, observer: Option<&dyn ModuleResolutionObserver>) -> bool {
    let start = observer.map(|_| Timer::start());
    let Ok(mut file) = std::fs::File::open(init_path) else {
        return false;
    };
    let mut buf = [0u8; PKGUTIL_DETECTION_MAX_BYTES];
    let mut total = 0;
    while total < buf.len() {
        match file.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(_) => return false,
        }
    }
    if let Some(observer) = observer {
        observer.observe_read(start.unwrap().elapsed_nanos());
    }
    let contents = String::from_utf8_lossy(&buf[..total]);
    PKGUTIL_EXTEND_PATH_PATTERN.is_match(&contents)
}

/// Cache of directory listings to avoid repeated stat() calls during module resolution.
///
/// Each directory is read at most once. Entries store
/// the file type alongside the name, which normally comes from dirent `d_type`;
/// symlinks are followed with `metadata()` to preserve `Path::is_dir()` behavior.
/// The cache is never invalidated, so callers should scope it to a stable
/// resolution transaction or replace it when file changes are observed.
#[derive(Default)]
pub struct DirEntryCache {
    /// Cached directory listings: maps a directory to its entries (name -> is_dir).
    entry_cache: LockedMap<PathBuf, Option<Arc<SmallMap<OsString, bool>>>>,
    /// Cached `pkgutil.extend_path` namespace-package check, keyed by `__init__` path.
    pkgutil_cache: LockedMap<PathBuf, bool>,
}

impl Debug for DirEntryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirEntryCache").finish_non_exhaustive()
    }
}

impl DirEntryCache {
    pub fn new() -> Self {
        Self {
            entry_cache: LockedMap::new(),
            pkgutil_cache: LockedMap::new(),
        }
    }

    /// Cached form of [`is_pkgutil_namespace`], keyed by `__init__` path.
    fn is_pkgutil_namespace(
        &self,
        init_path: &Path,
        observer: Option<&dyn ModuleResolutionObserver>,
    ) -> bool {
        let key = init_path.to_path_buf();
        if let Some(cached) = self.pkgutil_cache.get(&key) {
            return *cached;
        }
        let result = is_pkgutil_namespace(init_path, observer);
        self.pkgutil_cache.insert(key, result);
        result
    }

    pub fn file_exists(&self, path: &Path) -> bool {
        match (path.parent(), path.file_name()) {
            (Some(parent), Some(name)) => self
                .get_entries(parent)
                .is_some_and(|entries| matches!(entries.get(name), Some(false))),
            _ => path.exists(),
        }
    }

    pub fn dir_exists(&self, dir: &Path) -> bool {
        match (dir.parent(), dir.file_name()) {
            (Some(parent), Some(name)) => {
                if let Some(entries) = self.get_entries(parent) {
                    return matches!(entries.get(name), Some(true));
                }
                self.get_entries(dir).is_some()
            }
            _ => self.get_entries(dir).is_some(),
        }
    }

    fn get_entries(&self, dir: &Path) -> Option<Arc<SmallMap<OsString, bool>>> {
        let key = dir.to_path_buf();
        if let Some(cached) = self.entry_cache.get(&key) {
            return cached.clone();
        }
        let listing = Self::read_dir_entries(dir);
        self.entry_cache.insert(key.clone(), listing);
        self.entry_cache.get(&key).and_then(|v| v.clone())
    }

    fn read_dir_entries(dir: &Path) -> Option<Arc<SmallMap<OsString, bool>>> {
        std::fs::read_dir(dir).ok().map(|entries| {
            Arc::new(
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let is_dir = e.file_type().is_ok_and(|ft| {
                            if ft.is_symlink() {
                                std::fs::metadata(e.path()).is_ok_and(|m| m.file_type().is_dir())
                            } else {
                                ft.is_dir()
                            }
                        });
                        (e.file_name(), is_dir)
                    })
                    .collect(),
            )
        })
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FindResult {
    /// Found a single-file .pyi module. The path must not point to an __init__ file.
    SingleFilePyiModule(PathBuf),
    /// Found a single-file .py module. The path must not point to an __init__ file.
    SingleFilePyModule(PathBuf),
    /// Found a regular package. The first field is its `__init__` file, and the
    /// second is the package directory used as the sole root for submodule lookup.
    RegularPackage(PathBuf, PathBuf),
    /// Found a legacy namespace package: a package whose `__init__` calls
    /// `pkgutil.extend_path`. The first field is the winning `__init__`; the
    /// roots accumulate same-named directories across the search path.
    LegacyNamespacePackage(PathBuf, Vec1<PathBuf>),
    /// Found an implicit namespace package with one or more same-named directories.
    ImplicitNamespacePackage(Vec1<PathBuf>),
    /// Found compiled Python artifacts. These have no source path usable by Pyrefly.
    CompiledModule(PathBuf),
}

impl FindResult {
    fn single_file(path: PathBuf, ext: &str) -> Self {
        if ext == "pyi" {
            Self::SingleFilePyiModule(path)
        } else {
            Self::SingleFilePyModule(path)
        }
    }

    fn style(&self) -> Option<ModuleStyle> {
        match self {
            Self::SingleFilePyiModule(_) => Some(ModuleStyle::Interface),
            Self::SingleFilePyModule(_) => Some(ModuleStyle::Executable),
            _ => None,
        }
    }

    fn best_result(a: FindResult, b: FindResult) -> Self {
        match (&a, &b) {
            // RegularPackage and LegacyNamespacePackage share the top tier: both
            // resolve to a concrete `__init__`. Tying them lets the prefer-`a`
            // rule preserve sys.path order when fallback roots are folded in.
            (FindResult::RegularPackage(..), _) | (FindResult::LegacyNamespacePackage(..), _) => a,
            (_, FindResult::RegularPackage(..)) | (_, FindResult::LegacyNamespacePackage(..)) => b,
            (FindResult::SingleFilePyiModule(_), _) => a,
            (_, FindResult::SingleFilePyiModule(_)) => b,
            (FindResult::SingleFilePyModule(_), _) => a,
            (_, FindResult::SingleFilePyModule(_)) => b,
            (FindResult::CompiledModule(_), _) => a,
            (_, FindResult::CompiledModule(_)) => b,
            (FindResult::ImplicitNamespacePackage(_), _) => a,
        }
    }

    pub fn module_path(self) -> Option<ModulePath> {
        match self {
            FindResult::SingleFilePyiModule(path)
            | FindResult::SingleFilePyModule(path)
            | FindResult::RegularPackage(path, _)
            | FindResult::LegacyNamespacePackage(path, _) => Some(ModulePath::filesystem(path)),
            FindResult::ImplicitNamespacePackage(roots) => {
                Some(ModulePath::namespace(roots.first().clone()))
            }
            FindResult::CompiledModule(_) => None,
        }
    }
}

pub fn package_has_py_typed(
    module: ModuleName,
    result: &FindResult,
    dir_cache: &DirEntryCache,
) -> bool {
    let depth = module.components().len().saturating_sub(1);
    let mut package_root = match result {
        FindResult::RegularPackage(_, dir) => dir.as_path(),
        FindResult::LegacyNamespacePackage(init_path, _) => {
            let Some(dir) = init_path.parent() else {
                return false;
            };
            dir
        }
        FindResult::SingleFilePyModule(path)
        | FindResult::SingleFilePyiModule(path)
        | FindResult::CompiledModule(path) => {
            if depth == 0 {
                return false;
            }
            path.as_path()
        }
        FindResult::ImplicitNamespacePackage(_) => return false,
    };

    for _ in 0..depth {
        let Some(parent) = package_root.parent() else {
            return false;
        };
        package_root = parent;
    }

    dir_cache.file_exists(&package_root.join("py.typed"))
}

fn find_one_part_in_root(
    name: &str,
    root: &Path,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    observer: Option<&dyn ModuleResolutionObserver>,
) -> Option<FindResult> {
    let candidate_dir = root.join(name);
    let candidate_init_suffixes = if style_filter.is_some_and(|s| s == ModuleStyle::Executable) {
        ["__init__.py", "__init__.pyi"]
    } else {
        ["__init__.pyi", "__init__.py"]
    };
    let dir_exists = timed_stat(observer, || dir_cache.dir_exists(&candidate_dir));

    if dir_exists {
        for candidate_init_suffix in candidate_init_suffixes {
            let init_path = candidate_dir.join(candidate_init_suffix);
            if timed_stat(observer, || dir_cache.file_exists(&init_path)) {
                if dir_cache.is_pkgutil_namespace(&init_path, observer) {
                    return Some(FindResult::LegacyNamespacePackage(
                        init_path,
                        Vec1::new(candidate_dir),
                    ));
                }
                return Some(FindResult::RegularPackage(init_path, candidate_dir));
            } else if let Some(v) = phantom_paths.as_deref_mut() {
                v.push(init_path);
            }
        }
    } else if let Some(v) = phantom_paths.as_deref_mut() {
        for candidate_init_suffix in candidate_init_suffixes {
            v.push(candidate_dir.join(candidate_init_suffix));
        }
    }

    for candidate_file_suffix in ["pyi", "py"] {
        let candidate_path = root.join(format!("{name}.{candidate_file_suffix}"));
        if timed_stat(observer, || dir_cache.file_exists(&candidate_path)) {
            let result = FindResult::single_file(candidate_path.clone(), candidate_file_suffix);
            if let Some(filter) = style_filter {
                if let Some(style) = result.style()
                    && style == filter
                {
                    return Some(result);
                }
            } else {
                return Some(result);
            }
        } else if let Some(v) = phantom_paths.as_deref_mut() {
            v.push(candidate_path);
        }
    }

    for candidate_compiled_suffix in COMPILED_FILE_SUFFIXES {
        let candidate_path = root.join(format!("{name}.{candidate_compiled_suffix}"));
        if timed_stat(observer, || dir_cache.file_exists(&candidate_path)) {
            let result = FindResult::CompiledModule(candidate_path);
            if let Some(filter) = style_filter {
                match filter {
                    ModuleStyle::Executable => return Some(result),
                    ModuleStyle::Interface => continue,
                }
            }
            return Some(result);
        } else if let Some(v) = phantom_paths.as_deref_mut() {
            v.push(candidate_path);
        }
    }

    if dir_exists {
        Some(FindResult::ImplicitNamespacePackage(Vec1::new(
            candidate_dir,
        )))
    } else {
        if let Some(v) = phantom_paths.as_deref_mut() {
            v.push(candidate_dir);
        }
        None
    }
}

enum NamespaceAccumulator {
    Implicit(Vec1<PathBuf>),
    Legacy(PathBuf, Vec1<PathBuf>),
}

impl NamespaceAccumulator {
    fn into_find_result(self) -> FindResult {
        match self {
            NamespaceAccumulator::Implicit(roots) => FindResult::ImplicitNamespacePackage(roots),
            NamespaceAccumulator::Legacy(init, roots) => {
                FindResult::LegacyNamespacePackage(init, roots)
            }
        }
    }
}

fn find_one_part<'a>(
    name: &str,
    mut roots: impl Iterator<Item = &'a PathBuf>,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    observer: Option<&dyn ModuleResolutionObserver>,
) -> Option<(FindResult, Vec<PathBuf>)> {
    if name == "__pycache__" {
        return None;
    }

    let mut acc: Option<NamespaceAccumulator> = None;

    while let Some(root) = roots.next() {
        match find_one_part_in_root(name, root, style_filter, phantom_paths, dir_cache, observer) {
            None => (),
            Some(FindResult::ImplicitNamespacePackage(pkg)) => {
                let namespace_dir = pkg.into_vec().remove(0);
                match &mut acc {
                    None => acc = Some(NamespaceAccumulator::Implicit(Vec1::new(namespace_dir))),
                    Some(NamespaceAccumulator::Implicit(roots)) => roots.push(namespace_dir),
                    Some(NamespaceAccumulator::Legacy(_, roots)) => roots.push(namespace_dir),
                }
            }
            Some(FindResult::LegacyNamespacePackage(init_path, init_roots)) => {
                debug_assert_eq!(init_roots.len(), 1);
                let init_dir = init_roots.into_vec().remove(0);
                match &mut acc {
                    None => {
                        acc = Some(NamespaceAccumulator::Legacy(init_path, Vec1::new(init_dir)));
                    }
                    Some(NamespaceAccumulator::Legacy(_, roots)) => roots.push(init_dir),
                    Some(NamespaceAccumulator::Implicit(_)) => {
                        let prior = match acc.take() {
                            Some(NamespaceAccumulator::Implicit(rs)) => rs.into_vec(),
                            _ => unreachable!(),
                        };
                        let mut combined = Vec1::new(init_dir);
                        combined.extend(prior);
                        acc = Some(NamespaceAccumulator::Legacy(init_path, combined));
                    }
                }
            }
            Some(FindResult::RegularPackage(init_path, init_dir)) => match &mut acc {
                None | Some(NamespaceAccumulator::Implicit(_)) => {
                    return Some((
                        FindResult::RegularPackage(init_path, init_dir),
                        roots.cloned().collect(),
                    ));
                }
                Some(NamespaceAccumulator::Legacy(_, roots)) => roots.push(init_dir),
            },
            Some(result) if acc.is_none() => {
                return Some((result, roots.cloned().collect::<Vec<_>>()));
            }
            Some(_) => {}
        }
    }

    acc.map(|a| (a.into_find_result(), vec![]))
}

fn continue_find_module(
    start_result: FindResult,
    components_rest: &str,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    observer: Option<&dyn ModuleResolutionObserver>,
) -> Option<FindResult> {
    let mut current_result = Some(start_result);
    if components_rest.is_empty() {
        return current_result;
    }
    for part in components_rest.split('.') {
        match current_result {
            None => break,
            Some(FindResult::SingleFilePyiModule(_))
            | Some(FindResult::SingleFilePyModule(_))
            | Some(FindResult::CompiledModule(_)) => {
                current_result = None;
                break;
            }
            Some(FindResult::RegularPackage(_, next_root)) => {
                current_result = find_one_part(
                    part,
                    iter::once(&next_root),
                    style_filter,
                    phantom_paths,
                    dir_cache,
                    observer,
                )
                .map(|x| x.0);
            }
            Some(FindResult::LegacyNamespacePackage(_, next_roots))
            | Some(FindResult::ImplicitNamespacePackage(next_roots)) => {
                current_result = find_one_part(
                    part,
                    next_roots.iter(),
                    style_filter,
                    phantom_paths,
                    dir_cache,
                    observer,
                )
                .map(|x| x.0);
            }
        }
    }
    current_result
}

fn find_module_components<'a, I>(
    first: &str,
    components_rest: &str,
    include: I,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    observer: Option<&dyn ModuleResolutionObserver>,
) -> Option<FindResult>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let (first_component_result, fallback_search) = find_one_part(
        first,
        include.clone(),
        style_filter,
        phantom_paths,
        dir_cache,
        observer,
    )?;

    let current_result = continue_find_module(
        first_component_result,
        components_rest,
        style_filter,
        phantom_paths,
        dir_cache,
        observer,
    )?;

    match current_result {
        FindResult::SingleFilePyiModule(_)
        | FindResult::RegularPackage(..)
        | FindResult::LegacyNamespacePackage(..) => Some(current_result),
        _ => Some(
            fallback_search
                .into_iter()
                .filter_map(|s| {
                    Some(
                        find_one_part(
                            first,
                            [s].iter(),
                            style_filter,
                            &mut None,
                            dir_cache,
                            observer,
                        )?
                        .0,
                    )
                })
                .filter_map(|first| {
                    continue_find_module(
                        first.clone(),
                        components_rest,
                        style_filter,
                        &mut None,
                        dir_cache,
                        observer,
                    )
                })
                .fold(current_result, FindResult::best_result),
        ),
    }
}

#[derive(Debug, Default)]
pub struct ModuleSearchResult {
    pub stub_result: Option<FindResult>,
    pub normal_result: Option<FindResult>,
}

pub fn find_module_results<'a, I>(
    module: ModuleName,
    include: I,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    observer: Option<&dyn ModuleResolutionObserver>,
) -> ModuleSearchResult
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let module = module.as_str();
    let (first, rest) = module.split_once('.').unwrap_or((module, ""));
    let stub_first = format!("{first}-stubs");
    let stub_result = find_module_components(
        &stub_first,
        rest,
        include.clone(),
        style_filter,
        phantom_paths,
        dir_cache,
        observer,
    );
    let normal_result = find_module_components(
        first,
        rest,
        include,
        style_filter,
        phantom_paths,
        dir_cache,
        observer,
    );
    ModuleSearchResult {
        stub_result,
        normal_result,
    }
}

/// Filesystem-only module resolver for build-system integrations.
///
/// This intentionally exposes the shared package/stub/namespace lookup core
/// without checker-specific policy such as bundled third-party stubs, `py.typed`
/// diagnostics, import-ignore handling, or namespace fallback across separate
/// search phases.
#[derive(Debug)]
pub struct ModuleResolver {
    roots: Vec<PathBuf>,
    dir_cache: DirEntryCache,
}

impl ModuleResolver {
    pub fn new(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
            dir_cache: DirEntryCache::new(),
        }
    }

    pub fn resolve(&self, module: ModuleName, style: Option<ModuleStyle>) -> Option<ModulePath> {
        let result = find_module_results(
            module,
            self.roots.iter(),
            style,
            &mut None,
            &self.dir_cache,
            None,
        );
        match (result.normal_result, result.stub_result) {
            (_, Some(stub_result)) => stub_result.module_path(),
            (Some(normal_result), None) => normal_result.module_path(),
            (None, None) => None,
        }
    }
}

fn find_one_part_prefix<'a>(
    prefix: &str,
    roots: impl Iterator<Item = &'a PathBuf>,
) -> Vec<(FindResult, ModuleName)> {
    let mut results = Vec::new();
    let mut namespace_roots: SmallMap<ModuleName, Vec<PathBuf>> = SmallMap::new();

    for root in roots {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                let file_name = path.file_name().and_then(|n| n.to_str());

                if let Some(name) = file_name
                    && name.starts_with(prefix)
                {
                    if path.is_dir() {
                        for candidate_init_suffix in ["__init__.pyi", "__init__.py"] {
                            let init_path = path.join(candidate_init_suffix);
                            if init_path.is_file() {
                                let result = if is_pkgutil_namespace(&init_path, None) {
                                    FindResult::LegacyNamespacePackage(
                                        init_path,
                                        Vec1::new(path.clone()),
                                    )
                                } else {
                                    FindResult::RegularPackage(init_path, path.clone())
                                };
                                results.push((result, ModuleName::from_str(name)));
                                break;
                            }
                        }

                        if !results.iter().any(|r| match r {
                            (FindResult::RegularPackage(_, p), _) => *p == path,
                            (FindResult::LegacyNamespacePackage(_, ps), _) => ps.first() == &path,
                            _ => false,
                        }) {
                            namespace_roots
                                .entry(ModuleName::from_str(name))
                                .or_default()
                                .push(path.clone());
                        }
                    } else if let Some((stem, ext)) = name.rsplit_once('.')
                        && path.is_file()
                        && !["__init__", "__main__"].contains(&stem)
                    {
                        if ["pyi", "py"].contains(&ext) {
                            results.push((
                                FindResult::single_file(path.clone(), ext),
                                ModuleName::from_str(stem),
                            ));
                        } else if COMPILED_FILE_SUFFIXES.contains(&ext) {
                            results.push((
                                FindResult::CompiledModule(path.clone()),
                                ModuleName::from_str(stem),
                            ));
                        }
                    }
                }
            }
        }
    }

    for (name, roots) in namespace_roots {
        if let Ok(namespace_roots) = Vec1::try_from_vec(roots) {
            results.push((FindResult::ImplicitNamespacePackage(namespace_roots), name));
        }
    }

    results
}

pub fn find_module_prefixes<'a>(
    prefix: ModuleName,
    include: impl Iterator<Item = &'a PathBuf>,
) -> Vec<ModuleName> {
    let dir_cache = DirEntryCache::new();
    let components = prefix.components();
    // Empty prefixes are used by completions to list top-level modules.
    let first = components.first().map_or("", |first| first.as_str());
    let rest = if components.is_empty() {
        &[]
    } else {
        &components[1..]
    };
    let mut results = Vec::new();
    if rest.is_empty() {
        results = find_one_part_prefix(first, include)
    } else {
        let mut current_result =
            find_one_part(first, include, None, &mut None, &dir_cache, None).map(|x| x.0);
        for (i, part) in rest.iter().enumerate() {
            let is_last = i == rest.len() - 1;
            let part = part.as_str();
            match current_result {
                None => break,
                Some(
                    FindResult::SingleFilePyiModule(_)
                    | FindResult::SingleFilePyModule(_)
                    | FindResult::CompiledModule(_),
                ) => break,
                Some(FindResult::RegularPackage(_, next_root)) => {
                    if is_last {
                        results = find_one_part_prefix(part, iter::once(&next_root));
                        break;
                    } else {
                        current_result = find_one_part(
                            part,
                            iter::once(&next_root),
                            None,
                            &mut None,
                            &dir_cache,
                            None,
                        )
                        .map(|x| x.0);
                    }
                }
                Some(FindResult::LegacyNamespacePackage(_, next_roots))
                | Some(FindResult::ImplicitNamespacePackage(next_roots)) => {
                    if is_last {
                        results = find_one_part_prefix(part, next_roots.iter());
                        break;
                    } else {
                        current_result = find_one_part(
                            part,
                            next_roots.iter(),
                            None,
                            &mut None,
                            &dir_cache,
                            None,
                        )
                        .map(|x| x.0);
                    }
                }
            }
        }
    }
    results.iter().map(|(_, name)| *name).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_util::test_path::TestPath;

    use super::*;

    #[test]
    fn test_is_pkgutil_namespace_detection() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        let init1 = root.join("init1.py");
        std::fs::write(
            &init1,
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(is_pkgutil_namespace(&init1, None));

        let init2 = root.join("init2.py");
        std::fs::write(
            &init2,
            "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(is_pkgutil_namespace(&init2, None));

        let init3 = root.join("init3.py");
        std::fs::write(
            &init3,
            "import pkgutil\n__path__ = pkgutil.extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(is_pkgutil_namespace(&init3, None));

        let init4 = root.join("init4.py");
        std::fs::write(&init4, "from . import foo\n__all__ = ['foo']\n").unwrap();
        assert!(!is_pkgutil_namespace(&init4, None));

        let init5 = root.join("init5.py");
        std::fs::write(&init5, "").unwrap();
        assert!(!is_pkgutil_namespace(&init5, None));

        let init_comment = root.join("init_comment.py");
        std::fs::write(
            &init_comment,
            "# __path__ = pkgutil.extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(!is_pkgutil_namespace(&init_comment, None));

        let init_inline_comment = root.join("init_inline_comment.py");
        std::fs::write(
            &init_inline_comment,
            "__path__ = pkgutil. # commented out\n    extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(!is_pkgutil_namespace(&init_inline_comment, None));

        let init_suffix = root.join("init_suffix.py");
        std::fs::write(
            &init_suffix,
            "__path__ = mymod._extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(!is_pkgutil_namespace(&init_suffix, None));

        let init_multiline = root.join("init_multiline.py");
        std::fs::write(
            &init_multiline,
            "__path__ = (\n    pkgutil.extend_path(__path__, __name__)\n)\n",
        )
        .unwrap();
        assert!(!is_pkgutil_namespace(&init_multiline, None));

        let init_trailing_comment = root.join("init_trailing_comment.py");
        std::fs::write(
            &init_trailing_comment,
            "__path__ = pkgutil.extend_path(__path__, __name__)  # legacy ns\n",
        )
        .unwrap();
        assert!(is_pkgutil_namespace(&init_trailing_comment, None));

        let init_indented = root.join("init_indented.py");
        std::fs::write(
            &init_indented,
            "if True:\n    __path__ = pkgutil.extend_path(__path__, __name__)\n",
        )
        .unwrap();
        assert!(is_pkgutil_namespace(&init_indented, None));

        let init_truncated = root.join("init_truncated.py");
        let mut padding = String::with_capacity(PKGUTIL_DETECTION_MAX_BYTES + 128);
        for _ in 0..(PKGUTIL_DETECTION_MAX_BYTES / 4) {
            padding.push_str("# x\n");
        }
        padding.push_str("__path__ = pkgutil.extend_path(__path__, __name__)\n");
        std::fs::write(&init_truncated, &padding).unwrap();
        assert!(!is_pkgutil_namespace(&init_truncated, None));
    }

    #[test]
    fn test_find_precedence_in_all_roots() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "foo",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("baz.py"),
                        TestPath::dir(
                            "compiled",
                            vec![TestPath::file("__init__.py"), TestPath::file("a.pyc")],
                        ),
                        TestPath::dir("namespace", vec![]),
                    ],
                ),
                TestPath::dir(
                    "bar",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("baz.pyi"),
                        TestPath::dir(
                            "compiled",
                            vec![TestPath::file("__init__.py"), TestPath::file("a.py")],
                        ),
                        TestPath::file("namespace.py"),
                    ],
                ),
            ],
        );
        let roots = [root.join("foo"), root.join("bar")];

        assert_eq!(
            find_one_part(
                "baz",
                roots.iter(),
                None,
                &mut None,
                &DirEntryCache::new(),
                None
            ),
            Some((
                FindResult::SingleFilePyModule(root.join("foo/baz.py")),
                vec![root.join("bar")]
            ))
        );
        assert_eq!(
            continue_find_module(
                FindResult::SingleFilePyiModule(root.join("foo/baz.py")),
                "",
                None,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            Some(FindResult::SingleFilePyiModule(root.join("foo/baz.py")))
        );
        assert_eq!(
            find_module_components(
                "baz",
                "",
                roots.iter(),
                None,
                &mut None,
                &DirEntryCache::new(),
                None
            )
            .unwrap(),
            FindResult::SingleFilePyiModule(root.join("bar/baz.pyi")),
        );

        assert_eq!(
            find_one_part(
                "compiled",
                roots.iter(),
                None,
                &mut None,
                &DirEntryCache::new(),
                None
            ),
            Some((
                FindResult::RegularPackage(
                    root.join("foo/compiled/__init__.py"),
                    root.join("foo/compiled")
                ),
                vec![root.join("bar")]
            ))
        );
        assert_eq!(
            continue_find_module(
                FindResult::RegularPackage(
                    root.join("foo/compiled/__init__.py"),
                    root.join("foo/compiled")
                ),
                "a",
                None,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindResult::CompiledModule(root.join("foo/compiled/a.pyc"))
        );
        assert_eq!(
            find_module_components(
                "compiled",
                "a",
                roots.iter(),
                None,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindResult::SingleFilePyModule(root.join("bar/compiled/a.py"))
        );
    }

    #[test]
    fn test_find_module_prefixes_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("foo.py")]);
        assert_eq!(
            find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter()),
            vec![ModuleName::from_str("foo")]
        );
    }

    #[test]
    fn test_find_module_prefixes_ignores_init() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::file("foo.py"), TestPath::file("__init__.py")],
        );
        assert_eq!(
            find_module_prefixes(ModuleName::from_str(""), [root.to_path_buf()].iter()),
            vec![ModuleName::from_str("foo")]
        );
    }

    #[test]
    fn test_find_module_prefixes_nested_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir("baz", vec![TestPath::file("foo.py")])],
        );
        assert_eq!(
            find_module_prefixes(ModuleName::from_str("baz.fo"), [root.to_path_buf()].iter()),
            vec![ModuleName::from_str("foo")]
        );
    }

    #[test]
    fn test_find_module_prefixes_nested_regular_package() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "baz",
                vec![TestPath::dir("foo", vec![TestPath::file("__init__.py")])],
            )],
        );
        assert_eq!(
            find_module_prefixes(ModuleName::from_str("baz.fo"), [root.to_path_buf()].iter()),
            vec![ModuleName::from_str("foo")]
        );
    }

    #[test]
    fn test_find_module_prefixes_regular_package() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir("foo", vec![TestPath::file("__init__.py")])],
        );
        assert_eq!(
            find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter()),
            vec![ModuleName::from_str("foo")]
        );
    }

    #[test]
    fn test_find_module_prefixes_multiple_search_paths() {
        let root = tempfile::tempdir().unwrap();
        let root2 = tempfile::tempdir().unwrap();
        TestPath::setup_test_directory(
            root.path(),
            vec![TestPath::dir("foo", vec![TestPath::file("__init__.py")])],
        );
        TestPath::setup_test_directory(root2.path(), vec![TestPath::file("foo2.py")]);
        assert_eq!(
            find_module_prefixes(
                ModuleName::from_str("fo"),
                [root.path().to_path_buf(), root2.path().to_path_buf()].iter(),
            ),
            vec![ModuleName::from_str("foo"), ModuleName::from_str("foo2")]
        );
    }

    #[test]
    fn test_find_module_prefixes_nested_namespaces() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("foo", Vec::new()),
                TestPath::dir("foo2", Vec::new()),
            ],
        );
        let mut res = find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter());
        res.sort();
        assert_eq!(
            res,
            vec![ModuleName::from_str("foo"), ModuleName::from_str("foo2")]
        );
    }

    #[test]
    fn test_find_module_prefixes_namespaces() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("foo", Vec::new()),
                TestPath::dir("foo2", Vec::new()),
            ],
        );
        let mut res = find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter());
        res.sort();
        assert_eq!(
            res,
            vec![ModuleName::from_str("foo"), ModuleName::from_str("foo2")]
        );
    }

    #[test]
    fn test_find_one_part_with_pyc() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("nested_module.pyc"),
                TestPath::file("another_nested_module.py"),
                TestPath::file("cython_module.pyx"),
                TestPath::file("windows_dll.pyd"),
            ],
        );
        let result = find_one_part(
            "nested_module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("nested_module.pyc"))
        );
        let result = find_one_part(
            "cython_module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("cython_module.pyx"))
        );
        let result = find_one_part(
            "windows_dll",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("windows_dll.pyd"))
        );
        let result = find_one_part(
            "another_nested_module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::SingleFilePyModule(root.join("another_nested_module.py"))
        );
    }

    #[test]
    fn test_continue_find_module_with_pyc() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "subdir",
                vec![
                    TestPath::file("nested_module.pyc"),
                    TestPath::file("another_nested_module.py"),
                ],
            )],
        );
        let find_result = find_module_components(
            "subdir",
            "nested_module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap();
        assert!(matches!(find_result, FindResult::CompiledModule(_)));
        let module_path = find_module_components(
            "subdir",
            "another_nested_module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap();
        assert_eq!(
            module_path,
            FindResult::SingleFilePyModule(root.join("subdir/another_nested_module.py"))
        );
    }

    #[test]
    fn test_continue_find_module_signature() {
        let start_result =
            FindResult::RegularPackage(PathBuf::from("path/to/init.py"), PathBuf::from("path/to"));
        assert!(
            continue_find_module(
                start_result,
                "test_module",
                None,
                &mut None,
                &DirEntryCache::new(),
                None
            )
            .is_none()
        );
    }

    #[test]
    fn test_continue_find_module_with_pyc_no_source_ignored() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("module.pyc")]);
        let start_result = find_one_part(
            "module",
            [root.to_path_buf()].iter(),
            None,
            &mut None,
            &DirEntryCache::new(),
            None,
        )
        .unwrap()
        .0;
        assert!(matches!(
            continue_find_module(
                start_result,
                "",
                None,
                &mut None,
                &DirEntryCache::new(),
                None
            )
            .unwrap(),
            FindResult::CompiledModule(_)
        ));
    }

    #[test]
    fn test_dir_entry_cache_basic() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "foo",
                    vec![TestPath::file("bar.py"), TestPath::file("__init__.py")],
                ),
                TestPath::file("baz.py"),
            ],
        );

        let cache = DirEntryCache::new();

        assert!(cache.dir_exists(&root.join("foo")));
        assert!(!cache.dir_exists(&root.join("nonexistent")));
        assert!(cache.file_exists(&root.join("baz.py")));
        assert!(!cache.file_exists(&root.join("missing.py")));
        assert!(cache.file_exists(&root.join("foo").join("bar.py")));
        assert!(cache.file_exists(&root.join("foo").join("__init__.py")));
    }

    #[test]
    fn test_dir_entry_cache_file_exists_rejects_directories() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::dir("pkg.py", vec![])]);

        assert!(!DirEntryCache::new().file_exists(&root.join("pkg.py")));
    }

    #[test]
    fn test_dir_entry_cache_reuses_listing() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "pkg",
                vec![TestPath::file("a.py"), TestPath::file("b.py")],
            )],
        );

        let cache = DirEntryCache::new();
        let pkg = root.join("pkg");

        assert!(cache.file_exists(&pkg.join("a.py")));
        assert!(cache.file_exists(&pkg.join("b.py")));
        assert!(!cache.file_exists(&pkg.join("c.py")));
    }
}
