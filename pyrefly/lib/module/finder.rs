/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::atomic::Ordering;

pub use pyrefly_build::module_resolver::DirEntryCache;
use pyrefly_build::module_resolver::FindResult;
use pyrefly_build::module_resolver::ModuleResolutionObserver;
use pyrefly_build::module_resolver::find_module_prefixes;
use pyrefly_build::module_resolver::find_module_results;
use pyrefly_build::module_resolver::package_has_py_typed;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::locked_map::LockedMap;
use pyrefly_util::suggest::best_suggestion;
use ruff_python_ast::name::Name;
use vec1::Vec1;

/// Global cache for stdlib import suggestions.
/// Keyed by the missing module name, returns the suggested module name (if any).
static STDLIB_SUGGESTION_CACHE: LazyLock<LockedMap<ModuleName, Option<ModuleName>>> =
    LazyLock::new(LockedMap::new);

use crate::config::config::ConfigFile;
use crate::module::bundled::BundledStub;
use crate::module::third_party::get_bundled_third_party;
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;
use crate::state::loader::FindError;
use crate::state::loader::FindingOrError;
use crate::state::state::TransactionTimingCounters;

impl ModuleResolutionObserver for TransactionTimingCounters {
    fn observe_stat(&self, elapsed_ns: u64) {
        self.total_stat_count.fetch_add(1, Ordering::Relaxed);
        if elapsed_ns > 1_000_000 {
            self.slow_stat_count.fetch_add(1, Ordering::Relaxed);
            self.slow_stat_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }

    fn observe_read(&self, elapsed_ns: u64) {
        self.total_read_count.fetch_add(1, Ordering::Relaxed);
        if elapsed_ns > 1_000_000 {
            self.slow_read_count.fetch_add(1, Ordering::Relaxed);
            self.slow_read_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }
}

fn observer(timing: Option<&TransactionTimingCounters>) -> Option<&dyn ModuleResolutionObserver> {
    timing.map(|x| x as &dyn ModuleResolutionObserver)
}

fn find_result_module_path(result: FindResult) -> FindingOrError<ModulePath> {
    match result {
        FindResult::CompiledModule(_) => FindingOrError::Error(FindError::Ignored),
        result => FindingOrError::new_finding(
            result
                .module_path()
                .expect("non-compiled module search results should have module paths"),
        ),
    }
}

/// Determines whether to use a bundled stub based on the search results.
///
/// Returns `Some(result)` if we should use the bundled stub (possibly with an error attached).
/// Returns `None` if we should continue to the normal result handling - this happens when:
/// - No bundled stub was provided, OR
/// - A `-stubs` package was found (higher priority), OR
/// - Using a real config file with no source package installed
fn resolve_third_party_stub(
    module: ModuleName,
    stub_result: Option<&FindResult>,
    normal_result: Option<&FindResult>,
    bundled_stub: Option<FindingOrError<ModulePath>>,
    from_real_config_file: bool,
    dir_cache: &DirEntryCache,
) -> Option<FindingOrError<ModulePath>> {
    // This is the case where we do have a config file, the package is installed, but there are no stubs
    // available besides the bundled stubs. In this case
    // return the stub but with the error attached telling the user to install stubs.
    if let Some(ref bundled) = bundled_stub
        && from_real_config_file
        && let Some(normal_result) = normal_result
        && !package_has_py_typed(module, normal_result, dir_cache)
        && stub_result.is_none()
    {
        if let Some(pip_package) = recommended_stubs_package(module) {
            return Some(bundled.clone().with_error(FindError::UntypedImport(
                module,
                pip_package.to_string().into(),
            )));
        } else {
            // If we do not have a stub package that we recommend, just return the bundled stub without
            // the error
            return Some(bundled.clone());
        }
    }

    // If we do have a bundled stub and we also do not find a
    // higher priority stub from the site packages, then we should use the
    // bundled stub. However, if we also don't find the actual
    // package (normal_result), we should attach a MissingSource error.
    if let Some(bundled) = bundled_stub
        && stub_result.is_none()
    {
        if normal_result.is_none() {
            // If we have a real config file, don't return stubs when package is missing.
            // Return None to continue search, which will eventually hit NotFound error.
            if from_real_config_file {
                return None;
            } else {
                // Keep existing behavior for non-real config files
                return Some(bundled.with_error(FindError::MissingSourceForStubs(module)));
            }
        } else {
            // We have both typeshed third party stubs and the actual package
            return Some(bundled);
        }
    }

    None // No typeshed stub precedence applies, continue to normal handling
}

/// Combines stub and normal search results into a final FindingOrError.
///
/// When a namespace package is found, it is accumulated into `namespaces_found`
/// and `None` is returned to allow the search to continue in other paths.
fn combine_normal_and_stub_results(
    module: ModuleName,
    stub_result: Option<FindResult>,
    normal_result: Option<FindResult>,
    namespaces_found: &mut Vec<PathBuf>,
    dir_cache: &DirEntryCache,
) -> Option<FindingOrError<ModulePath>> {
    match (normal_result, stub_result) {
        (None, Some(stub_result)) => {
            Some(find_result_module_path(stub_result).with_error(FindError::MissingSource(module)))
        }
        (Some(_), Some(stub_result)) => Some(find_result_module_path(stub_result)),
        (Some(FindResult::ImplicitNamespacePackage(namespaces)), _) => {
            namespaces_found.append(&mut namespaces.into_vec());
            None
        }
        (Some(normal_result), None) => {
            if let Some(missing_stub_result) = recommended_stubs_package(module)
                && !package_has_py_typed(module, &normal_result, dir_cache)
            {
                Some(
                    find_result_module_path(normal_result).with_error(FindError::UntypedImport(
                        module,
                        missing_stub_result.as_str().to_owned().into(),
                    )),
                )
            } else {
                Some(find_result_module_path(normal_result))
            }
        }
        (None, _) => None,
    }
}

/// Search for the given [`ModuleName`] in the given `include`, which is
/// a list of paths denoting import roots. A [`FindError`] result indicates
/// searching should be discontinued because of a special condition, whereas
/// an `Ok(None)` indicates the module wasn't found here, but could be found in another
/// search location (`search_path`, `typeshed`, ...).
///
/// If the result is a [`FindResult::ImplicitNamespacePackage`], we instead add its entries to
/// `namespaces_found`, since this can be overridden by a higher-priority [`FindResult`]
/// variant later. It is the calling function's responsibility to recognize that
/// `namespaces_found` might hold the final result if no `Ok(Some(_))` values are
/// returned from this function.
///
/// If `style_filter` is provided, only modules matching that style will be returned.
/// Returns the first module found that matches the style, or `None` if no matching module is found.
///
/// If `phantom_paths` is provided, paths that were checked but did not exist will be added to it.
/// Note: `phantom_paths` and `style_filter` are mutually exclusive.
fn find_module<'a, I>(
    module: ModuleName,
    include: I,
    namespaces_found: &mut Vec<PathBuf>,
    style_filter: Option<ModuleStyle>,
    typeshed_third_party_stub: Option<FindingOrError<ModulePath>>,
    from_real_config_file: bool,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    timing: Option<&TransactionTimingCounters>,
) -> Option<FindingOrError<ModulePath>>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let results = find_module_results(
        module,
        include,
        style_filter,
        phantom_paths,
        dir_cache,
        observer(timing),
    );
    if let Some(result) = resolve_third_party_stub(
        module,
        results.stub_result.as_ref(),
        results.normal_result.as_ref(),
        typeshed_third_party_stub,
        from_real_config_file,
        dir_cache,
    ) {
        return Some(result);
    }
    combine_normal_and_stub_results(
        module,
        results.stub_result,
        results.normal_result,
        namespaces_found,
        dir_cache,
    )
}

/// Configerator file extensions that use the keyword-escaping convention.
/// When a path component matches a Python keyword (e.g. `if`), module names
/// escape it with a trailing underscore (`if_`). This convention is specific
/// to configerator repos and should not apply to other extra file extensions.
///
/// Kept consistent with `CONFIGERATOR_FILE_SUFFIX_EXCLUDE_THRIFT` in Pyright's
/// `configerator-file-system.ts`.
const CONFIGERATOR_EXTENSIONS: &[&str] = &["cinc", "cconf", "thrift-cvalidator", "ctest", "mcconf"];

/// If `component` has a trailing underscore and the base is a Python keyword
/// (e.g. `if_` → `if`), return the keyword. Otherwise return the component
/// unchanged. This handles configerator paths where directories or filename
/// segments are named with Python keywords, and the module path uses `if_`
/// because Python syntax forbids bare keywords as identifiers.
fn unescape_keyword(component: &str) -> &str {
    if let Some(base) = component.strip_suffix('_')
        && pyrefly_python::keywords::is_keyword(base)
    {
        return base;
    }
    component
}

/// Attempt to find a module that uses an extra file extension where dots in
/// filenames act as module separators.
///
/// Given module `a.b.c.cinc` and search roots, tries to find the file by
/// progressively collapsing the rightmost module components into a dotted
/// filename. The search order (from most to fewest directory components) is:
///   1. `<root>/a/b/c.cinc`
///   2. `<root>/a/b.c.cinc`
///   3. `<root>/a.b.c.cinc`
///
/// Files "closer" to the source directory (more directory components) take
/// precedence over files further away.
///
/// For configerator extensions (`cinc`, `cconf`, etc.), module components
/// matching Python keywords with a trailing underscore (e.g. `if_`) are
/// unescaped to their real names (e.g. `if`). This handles configerator repos
/// where path segments can be named with Python keywords. Non-configerator
/// extra extensions are not affected.
fn find_extra_extension_module<'a>(
    module: ModuleName,
    roots: impl Iterator<Item = &'a PathBuf>,
    extra_extensions: &[String],
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
) -> Option<FindingOrError<ModulePath>> {
    let components = module.components();
    let Ok(components) = Vec1::try_from_vec(components) else {
        return None;
    };
    if components.len() < 2 {
        return None;
    }
    let last = components.last();
    // The last component must be a recognized extra extension.
    if !extra_extensions.iter().any(|ext| ext == last.as_str()) {
        return None;
    }

    // Keyword unescaping (e.g. `if_` → `if`) only applies to configerator
    // extensions. Other extra file extensions don't use this convention.
    let should_unescape = CONFIGERATOR_EXTENSIONS.contains(&last.as_str());

    for root in roots {
        let mut dir = root.clone();
        // Push directory components for the first (most-directories) candidate.
        // The max dir_count is len-2 since the last component is the extension.
        for part in &components[..components.len() - 2] {
            let part_str = part.as_str();
            dir.push(if should_unescape {
                unescape_keyword(part_str)
            } else {
                part_str
            });
        }
        // Try splitting at each point: dir_count components form the directory
        // path and the remaining components form a dot-joined filename.
        // We start from the highest dir_count (most directories) for precedence,
        // so that "closer" files win.
        // e.g., for module `a.b.c.cinc`:
        //   dir_count=2: dir=root/a/b, filename=c.cinc
        //   dir_count=1: dir=root/a,   filename=b.c.cinc
        //   dir_count=0: dir=root,     filename=a.b.c.cinc
        for dir_count in (0..components.len() - 1).rev() {
            // Form the dotted filename from the remaining components.
            // Unescape Python keywords only for configerator extensions.
            let filename: String = components[dir_count..]
                .iter()
                .map(|p| {
                    if should_unescape {
                        unescape_keyword(p.as_str())
                    } else {
                        p.as_str()
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
            let candidate = dir.join(&filename);
            // Prefer .pyi stub files (e.g., foo.thrift.pyi) over raw files
            // (e.g., foo.thrift). This allows generated type stubs to provide
            // Python type information for non-Python file extensions.
            let pyi_candidate = dir.join(format!("{}.pyi", &filename));
            if pyi_candidate.is_file() {
                return Some(FindingOrError::new_finding(ModulePath::filesystem(
                    pyi_candidate,
                )));
            } else if let Some(v) = phantom_paths.as_deref_mut() {
                v.push(pyi_candidate);
            }
            if candidate.is_file() {
                return Some(FindingOrError::new_finding(ModulePath::filesystem(
                    candidate,
                )));
            } else if let Some(v) = phantom_paths.as_deref_mut() {
                v.push(candidate);
            }
            // Pop the last directory component for the next iteration.
            dir.pop();
        }
    }
    None
}

/// This function will find either third party typeshed stubs or other third party stubs
/// Here a decision is being made to prioritize typeshed stubs over other third party stubs that are bundled.
/// Since we run the typeshed update script with a more regular cadence, it is more likely that
/// these stubs will be more up to date.
fn find_third_party_stub(
    module: ModuleName,
    style_filter: Option<ModuleStyle>,
) -> Option<FindingOrError<ModulePath>> {
    let third_party_typeshed_stub = if matches!(style_filter, Some(ModuleStyle::Interface) | None) {
        typeshed_third_party().map_or_else(
            |err| {
                Some(FindingOrError::Error(FindError::missing_import(
                    err, module,
                )))
            },
            |ts| ts.find(module).map(FindingOrError::new_finding),
        )
    } else {
        None
    };

    if let Some(stub) = third_party_typeshed_stub {
        return Some(stub);
    }

    if matches!(style_filter, Some(ModuleStyle::Interface) | None) {
        get_bundled_third_party().map_or_else(
            |err| {
                Some(FindingOrError::Error(FindError::missing_import(
                    err, module,
                )))
            },
            |ts| ts.find(module).map(FindingOrError::new_finding),
        )
    } else {
        None
    }
}

// TODO(connernilsen): change things so that we return all entries that match for a given
// module name across all path components (search path, site package path, ...).
// Instead, at specific times (`find_module_components`, `find_module`, `find_import_filtered`),
// see if we have a result for a highest priority item (something that is a single file module
// matching our style_filter (if applicable) or regular package, and return that. Otherwise,
// keep searching, and if we get the end, look through everything we've found and select the
// best thing.
/// Attempt to find an import with [`ModuleName`] from the search components specified
/// in the config (including build system). Origin specifies the file we're importing from,
/// and should only be empty when importing from typeshed/builtins.
///
/// [`ModuleStyle`] specifies whether we prefer a `.py` or `.pyi` file. When provided,
/// - if the result is an init file, we will treat `style_filter` as a preference, meaning
///   if we find a match, we'll see if the preferred value exists, but return whatever we
///   find immediately.
/// - if our best result is a namespace, we return nothing. Anything else is always
///   preferable to a namespace.
/// - otherwise, we return the first value if it matches the `style_filter`. If nothing
///   matches, we return None, even if there were other results.
///
/// If `None` is returned when `style_filter.is_some()`, the import should be retried
/// with `style_filter.is_none()`, since we hard-filter a lot of values here.
pub fn find_import_internal(
    config: &ConfigFile,
    module: ModuleName,
    origin: Option<&ModulePath>,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    timing: Option<&TransactionTimingCounters>,
) -> FindingOrError<ModulePath> {
    let mut namespaces_found = vec![];
    let origin = origin.map(|p| p.as_path());
    let typeshed_third_party_result = find_third_party_stub(module, style_filter);
    let typeshed_third_party_stub = typeshed_third_party_result.clone();
    let from_real_config_file = config.from_real_config_file();

    if module != ModuleName::builtins() && config.replace_imports_with_any(origin, module) {
        FindingOrError::Error(FindError::Ignored)
    } else if let Some(build_system) = config.build_system.as_ref()
        && let Some(path) = find_module(
            module,
            build_system.search_path_prefix.iter(),
            &mut namespaces_found,
            style_filter,
            None,
            false,
            phantom_paths,
            dir_cache,
            timing,
        )
    {
        path
    } else if let Some(sourcedb) = config.source_db.as_ref()
        && let Some(path) = sourcedb.lookup(module, origin, style_filter)
    {
        FindingOrError::new_finding(path.clone())
    } else if let Some(path) = find_module(
        module,
        config.search_path(),
        &mut namespaces_found,
        style_filter,
        None,
        false,
        phantom_paths,
        dir_cache,
        timing,
    ) {
        path
    } else if let Some(custom_typeshed_path) = &config.typeshed_path
        && let Some(path) = find_module(
            module,
            std::iter::once(&custom_typeshed_path.join("stdlib")),
            &mut namespaces_found,
            style_filter,
            None,
            false,
            phantom_paths,
            dir_cache,
            timing,
        )
    {
        path
    } else if matches!(style_filter, Some(ModuleStyle::Interface) | None)
        && let Some(path) = typeshed().map_or_else(
            |err| {
                Some(FindingOrError::Error(FindError::missing_import(
                    err, module,
                )))
            },
            |ts| ts.find(module).map(FindingOrError::new_finding),
        )
    {
        path
    } else if !config.disable_search_path_heuristics
        && let Some(path) = find_module(
            module,
            config
                .fallback_search_path
                .for_directory(origin.and_then(|p| p.parent()))
                .iter(),
            &mut namespaces_found,
            style_filter,
            None,
            false,
            phantom_paths,
            dir_cache,
            timing,
        )
    {
        path
    } else if let Some(path) = find_module(
        module,
        config.site_package_path(),
        &mut namespaces_found,
        style_filter,
        typeshed_third_party_stub.clone(),
        from_real_config_file,
        phantom_paths,
        dir_cache,
        timing,
    ) {
        path
    } else if config.has_extra_file_extensions()
        && let Some(path) = find_extra_extension_module(
            module,
            config.search_path().chain(config.site_package_path()),
            &config.extra_file_extensions,
            phantom_paths,
        )
    {
        path
    } else if let Some(namespace) = namespaces_found.into_iter().next() &&
        // only use namespaces if style filter is none, since otherwise we might be
        // skipping a result that's more preferable, but excluded because of the style
        // filter
    style_filter.is_none()
    {
        FindingOrError::new_finding(ModulePath::namespace(namespace))
    } else if config.ignore_missing_imports(origin, module) {
        FindingOrError::Error(FindError::Ignored)
    } else {
        FindingOrError::Error(FindError::import_lookup_path(
            config.structured_import_lookup_path(origin),
            module,
            &config.source,
        ))
    }
}

/// Get the given [`ModuleName`] from this config's search and site package paths.
/// We take the [`Handle`] of the file we're searching for the module from to determine if
/// we should replace imports with `typing.Any` and to perform lookups within a
/// `SourceDatabase`.
/// Return `Err` when indicating the module could not be found.
pub fn find_import(
    config: &ConfigFile,
    module: ModuleName,
    origin: Option<&ModulePath>,
    mut phantom_paths: Option<&mut Vec<PathBuf>>,
    dir_cache: &DirEntryCache,
    timing: Option<&TransactionTimingCounters>,
) -> FindingOrError<ModulePath> {
    find_import_internal(
        config,
        module,
        origin,
        None,
        &mut phantom_paths,
        dir_cache,
        timing,
    )
}

pub fn find_import_filtered(
    config: &ConfigFile,
    module: ModuleName,
    origin: Option<&ModulePath>,
    style_filter: Option<ModuleStyle>,
    dir_cache: &DirEntryCache,
    timing: Option<&TransactionTimingCounters>,
) -> FindingOrError<ModulePath> {
    find_import_internal(
        config,
        module,
        origin,
        style_filter,
        &mut None,
        dir_cache,
        timing,
    )
}

/// Find all legitimate imports that start with `module`
pub fn find_import_prefixes(config: &ConfigFile, module: ModuleName) -> Vec<ModuleName> {
    let mut results = find_module_prefixes(
        module,
        config.search_path().chain(config.site_package_path()),
    );

    if let Ok(ts) = typeshed() {
        let module_str = module.as_str();
        let typeshed_modules = ts
            .modules()
            .filter(|m| module_str.is_empty() || m.as_str().starts_with(module_str));

        results.extend(typeshed_modules);
    }

    if !config.from_real_config_file()
        && let Ok(typeshed_third_party) = typeshed_third_party()
    {
        let module_str = module.as_str();
        let typeshed_modules = typeshed_third_party
            .modules()
            .filter(|m| module_str.is_empty() || m.as_str().starts_with(module_str));

        results.extend(typeshed_modules);
    }

    results
}

fn recommended_stubs_package(module: ModuleName) -> Option<ModuleName> {
    match module.first_component().as_str() {
        "django" => Some(ModuleName::from_str("django-stubs")),
        _ => {
            // If the module has stubs in typeshed, recommend types-<package>
            if let Ok(ts) = typeshed_third_party()
                && let Some(package_name) = ts.package_name(module)
            {
                Some(ModuleName::from_str(&format!("types-{}", package_name)))
            } else {
                None
            }
        }
    }
}

/// Suggest a similar stdlib module name for a mistyped import.
/// Uses Levenshtein distance to find the closest match from typeshed's stdlib modules.
/// Results are cached globally since typeshed doesn't change during a session.
pub fn suggest_stdlib_import(missing: ModuleName) -> Option<ModuleName> {
    *STDLIB_SUGGESTION_CACHE
        .ensure(&missing, || suggest_stdlib_import_uncached(missing))
        .0
}

fn suggest_stdlib_import_uncached(missing: ModuleName) -> Option<ModuleName> {
    let ts = typeshed().ok()?;
    let missing_str = missing.as_str();

    // For single-component module names, use best_suggestion with first component
    // For multi-component, we compare full module name strings
    let missing_name = Name::new(missing_str);

    // Collect all stdlib module names and find the best suggestion
    let candidates: Vec<Name> = ts.modules().map(|m| Name::new(m.as_str())).collect();

    best_suggestion(&missing_name, candidates.iter().map(|c| (c, 0)))
        .map(|suggestion| ModuleName::from_str(suggestion.as_str()))
}

#[cfg(test)]
mod tests {
    use pyrefly_config::config::ConfigSource;
    use pyrefly_config::environment::environment::PythonEnvironment;
    use pyrefly_config::environment::interpreters::Interpreters;
    use pyrefly_python::module_path::ModulePathDetails;
    use pyrefly_util::test_path::TestPath;

    use super::*;
    use crate::state::loader::Finding;

    #[test]
    fn test_find_module_simple() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::file("bar.py"),
                    TestPath::file("baz.pyi"),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar.py")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz.pyi")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None,
        );
    }

    #[test]
    fn test_find_module_records_timing_counters() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir("foo", vec![TestPath::file("__init__.py")])],
        );
        let timing = TransactionTimingCounters::default();

        assert_eq!(
            find_module(
                ModuleName::from_str("foo"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                Some(&timing),
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/__init__.py"))),
        );
        assert!(timing.total_stat_count.load(Ordering::Relaxed) > 0);
        assert!(timing.total_read_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_find_module_init() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                    TestPath::dir("baz", vec![TestPath::file("__init__.pyi")]),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz/__init__.pyi")))
        );
    }

    #[test]
    fn test_find_pyi_takes_precedence() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::file("bar.pyi"),
                    TestPath::file("bar.py"),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar.pyi")))
        );
    }

    #[test]
    fn test_find_init_takes_precedence() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::file("bar.py"),
                    TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
    }

    #[test]
    fn test_basic_namespace_package() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "first",
                    vec![
                        TestPath::dir("a", vec![]),
                        TestPath::dir("b", vec![TestPath::dir("c", vec![])]),
                        TestPath::dir("c", vec![TestPath::dir("d", vec![TestPath::file("e.py")])]),
                    ],
                ),
                TestPath::dir(
                    "second",
                    vec![
                        TestPath::dir("a", vec![]),
                        TestPath::dir("b", vec![TestPath::dir("c", vec![])]),
                        TestPath::dir("c", vec![TestPath::dir("d", vec![TestPath::file("e.py")])]),
                    ],
                ),
            ],
        );
        let search_roots = [root.join("first"), root.join("second")];
        let assert_namespace = |name, expected: &str| {
            let mut namespaces = vec![];
            assert_eq!(
                find_module(
                    ModuleName::from_str(name),
                    search_roots.iter(),
                    &mut namespaces,
                    None,
                    None,
                    false,
                    &mut None,
                    &DirEntryCache::new(),
                    None,
                ),
                None
            );
            assert_eq!(
                namespaces,
                vec![
                    root.join(format!("first/{expected}")),
                    root.join(format!("second/{expected}"))
                ]
            );
        };
        assert_namespace("a", "a");
        assert_namespace("b", "b");
        assert_namespace("c.d", "c/d");
        assert_eq!(
            find_module(
                ModuleName::from_str("c.d.e"),
                search_roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("first/c/d/e.py")))
        );
    }

    #[test]
    fn test_find_regular_package_zero_instances_found() {
        // When no root contains the package at all, find_module returns None.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::dir("search_root0", vec![])]);
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                [root.join("search_root0")].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_regular_package_one_instance_found() {
        // A regular package found in exactly one root resolves to its __init__.py.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "search_root0",
                vec![TestPath::dir(
                    "a",
                    vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                )],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                [root.join("search_root0")].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );
        // Submodule in the same root is also reachable.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                [root.join("search_root0")].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );
    }

    #[test]
    fn test_regular_package_short_circuits() {
        // A regular package (no extend_path) claims the package name exclusively.
        // The second root's `__init__.py` and its submodules are unreachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        let roots = [root.join("search_root0"), root.join("search_root1")];
        // `a` resolves to the first root's __init__.py.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );
        // `a.b` is reachable (in root0).
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );
        // `a.c` is NOT reachable: root0 is a regular package and claims `a` exclusively.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_regular_package_short_circuits_over_namespace() {
        // A regular package in root0 short-circuits even when root1 has only a namespace
        // directory. Submodules in root1 are not reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir("a", vec![TestPath::file("c.py")])],
                ),
            ],
        );
        let roots = [root.join("search_root0"), root.join("search_root1")];
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );
        // `a.c` is NOT reachable: root0's regular package owns `a` exclusively.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    const PKGUTIL_INIT: &str =
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n";

    #[test]
    fn test_legacy_namespace_package_basic() {
        // A legacy namespace package (extend_path in __init__.py) makes submodules
        // in all same-named directories across search roots reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        // Write the pkgutil boilerplate into both __init__.py files.
        std::fs::write(root.join("search_root0/a/__init__.py"), PKGUTIL_INIT).unwrap();
        std::fs::write(root.join("search_root1/a/__init__.py"), PKGUTIL_INIT).unwrap();
        let roots = [root.join("search_root0"), root.join("search_root1")];
        // `a` resolves to the first root's __init__.py.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );
        // `a.b` is reachable from root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );
        // `a.c` is also reachable: extend_path merges all same-named directories.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
    }

    #[test]
    fn test_implicit_namespace_then_regular_package() {
        // Verified against CPython 3.9: when root0 has an implicit namespace
        // dir (no __init__.py) and root1 has a regular package, the regular
        // package wins exclusively — a.__path__ = [root1/a], a.c is
        // reachable, a.b is not.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir("a", vec![TestPath::file("b.py")])],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        let roots = [root.join("search_root0"), root.join("search_root1")];

        // `a` resolves to root1's __init__.py (RegularPackage wins over INP).
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root1/a/__init__.py")
            ))
        );

        // `a.c` is reachable (root1's regular package owns `a`).
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );

        // `a.b` is NOT reachable: the regular package claims `a` exclusively.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_legacy_namespace_package_then_regular_package() {
        // Once `find_one_part` enters LegacyNamespacePackage (LNP) mode
        // (root0 has an extend_path __init__.py), a *regular* package in
        // a later root must NOT take over the resolution. The LNP keeps
        // the winning __init__ from root0, but root1's directory is absorbed
        // into the LNP's path — extend_path includes every same-named
        // directory on sys.path regardless of __init__.py content.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        // root0's __init__.py is an LegacyNamespacePackage; root1's is a plain regular package.
        std::fs::write(root.join("search_root0/a/__init__.py"), PKGUTIL_INIT).unwrap();
        std::fs::write(root.join("search_root1/a/__init__.py"), "").unwrap();
        let roots = [root.join("search_root0"), root.join("search_root1")];
        // `a` resolves to the LegacyNamespacePackage's __init__.py from root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );
        // `a.b` is reachable (root0).
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );
        // `a.c` is reachable: extend_path absorbs root1's directory.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
    }

    #[test]
    fn test_legacy_namespace_package_absorbs_regular_package_dir() {
        // Verified against CPython 3.9: when root0 has an LNP and root1 has
        // a regular package, extend_path includes every same-named directory
        // on sys.path regardless of __init__.py content, so both a.b and a.c
        // are reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        std::fs::write(root.join("search_root0/a/__init__.py"), PKGUTIL_INIT).unwrap();
        std::fs::write(root.join("search_root1/a/__init__.py"), "").unwrap();
        let roots = [root.join("search_root0"), root.join("search_root1")];

        // This is correct: `a` resolves to root0's LNP __init__.py.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );

        // This is correct: `a.b` is reachable from root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );

        // `a.c` is reachable: extend_path absorbs root1's dir.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
    }

    #[test]
    fn test_legacy_namespace_package_pyi_init() {
        // An `__init__.pyi` containing extend_path text is also classified
        // as LegacyNamespacePackage — the regex match is by content, not filename. This pins
        // current behavior: stub files don't execute at runtime, so this is
        // a slight overshoot, but the alternative (filename-based skip) would
        // miss the real-world case of inline-stubbed legacy namespace packages.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.pyi"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir("a", vec![TestPath::file("c.py")])],
                ),
            ],
        );
        std::fs::write(root.join("search_root0/a/__init__.pyi"), PKGUTIL_INIT).unwrap();
        let roots = [root.join("search_root0"), root.join("search_root1")];
        // `a` resolves to the .pyi file in root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.pyi")
            ))
        );
        // `a.b` is reachable from root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );
        // `a.c` from root1 is reachable: LegacyNamespacePackage absorbs the ImplicitNamespacePackage dir in root1.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
    }

    #[test]
    fn test_single_file_wins_over_dir_without_init() {
        // Verified against CPython 3.9: when both `a/b.py` and `a/b/c.py`
        // exist but `a/b/` has no `__init__.py`, `a.b` resolves to `b.py`
        // (a single-file module) and `a.b.c` is not importable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "root",
                vec![TestPath::dir(
                    "a",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("b.py"),
                        TestPath::dir("b", vec![TestPath::file("c.py")]),
                    ],
                )],
            )],
        );
        let roots = [root.join("root")];

        // `a.b` resolves to the single file `b.py`, not the directory.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("root/a/b.py")))
        );

        // `a.b.c` is not reachable: `b.py` is a module, not a package.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_package_init_wins_over_single_file() {
        // Verified against CPython 3.9: when both `a/b.py` and
        // `a/b/__init__.py` exist, the package wins — `a.b` resolves to
        // `__init__.py` and `a.b.c` is reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "root",
                vec![TestPath::dir(
                    "a",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("b.py"),
                        TestPath::dir(
                            "b",
                            vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                        ),
                    ],
                )],
            )],
        );
        let roots = [root.join("root")];

        // `a.b` resolves to the package's `__init__.py`, not `b.py`.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("root/a/b/__init__.py")))
        );

        // `a.b.c` is reachable via the package.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("root/a/b/c.py")))
        );
    }

    #[test]
    fn test_pkgutil_init_wins_over_single_file() {
        // Verified against CPython 3.9: when both `a/b.py` and
        // `a/b/__init__.py` (with extend_path) exist, the pkgutil package
        // wins — `a.b` resolves to `__init__.py` and `a.b.c` is reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "root",
                vec![TestPath::dir(
                    "a",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("b.py"),
                        TestPath::dir(
                            "b",
                            vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                        ),
                    ],
                )],
            )],
        );
        std::fs::write(root.join("root/a/b/__init__.py"), PKGUTIL_INIT).unwrap();
        let roots = [root.join("root")];

        // `a.b` resolves to the pkgutil package's `__init__.py`, not `b.py`.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("root/a/b/__init__.py")))
        );

        // `a.b.c` is reachable via the pkgutil package.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("root/a/b/c.py")))
        );
    }

    #[test]
    fn test_regular_package_then_legacy_namespace_package() {
        // Verified against CPython 3.9: when root0 has a regular package and
        // root1 has an LNP, the regular package wins exclusively — a.c from
        // root1 is NOT reachable.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("b.py")],
                    )],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir(
                        "a",
                        vec![TestPath::file("__init__.py"), TestPath::file("c.py")],
                    )],
                ),
            ],
        );
        std::fs::write(root.join("search_root0/a/__init__.py"), "").unwrap();
        std::fs::write(root.join("search_root1/a/__init__.py"), PKGUTIL_INIT).unwrap();
        let roots = [root.join("search_root0"), root.join("search_root1")];

        // `a` resolves to root0's regular package.
        assert_eq!(
            find_module(
                ModuleName::from_str("a"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root0/a/__init__.py")
            ))
        );

        // `a.b` is reachable from root0.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.b"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root0/a/b.py")))
        );

        // `a.c` is NOT reachable: regular package is exclusive.
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                roots.iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_namespace_package_no_early_return() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "search_root0",
                    vec![
                        TestPath::dir("a", vec![TestPath::file("b.py")]),
                        TestPath::dir("spp_priority", vec![TestPath::dir("d", vec![])]),
                    ],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir("a", vec![TestPath::file("c.py")])],
                ),
                TestPath::dir(
                    "site_package_path",
                    vec![TestPath::dir(
                        "spp_priority",
                        vec![TestPath::file("__init__.py"), TestPath::file("d.py")],
                    )],
                ),
            ],
        );

        let mut interpreters = Interpreters::default();
        interpreters.skip_interpreter_query = true;
        let mut config = ConfigFile {
            search_path_from_file: vec![root.join("search_root0"), root.join("search_root1")],
            interpreters,
            python_environment: PythonEnvironment {
                site_package_path: Some(vec![root.join("site_package_path")]),
                ..Default::default()
            },
            ..Default::default()
        };
        config.configure();
        assert_eq!(
            find_import_filtered(
                &config,
                ModuleName::from_str("a.c"),
                None,
                None,
                &DirEntryCache::new(),
                None
            ),
            // We will find `a.c` because `a` is a namespace package whose search roots
            // include both `search_root0/a/` and `search_root1/a/`.
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
        assert_eq!(
            find_import_filtered(
                &config,
                ModuleName::from_str("spp_priority"),
                None,
                None,
                &DirEntryCache::new(),
                None
            ),
            // We will find `spp_priority` in `site_package_path`, even though it's
            // in a later module find component, because we continue searching for
            // a better option when we find a namespace package
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("site_package_path/spp_priority/__init__.py")
            )),
        );
        // we would either take the `__init__.py` result or nothing when a `ModuleStyle` is
        // provided than a namespace package
        assert_eq!(
            find_import_filtered(
                &config,
                ModuleName::from_str("spp_priority.d"),
                None,
                None,
                &DirEntryCache::new(),
                None
            ),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("site_package_path/spp_priority/d.py")
            )),
        );
        assert_eq!(
            find_import_filtered(
                &config,
                ModuleName::from_str("spp_priority.d"),
                None,
                Some(ModuleStyle::Interface),
                &DirEntryCache::new(),
                None,
            ),
            // When applying a `ModuleStyle`, we don't find a result and force a find import
            // without a module style.
            FindingOrError::Error(FindError::import_lookup_path(
                config.structured_import_lookup_path(None),
                ModuleName::from_str("spp_priority.d"),
                &config.source,
            )),
        );
    }

    #[test]
    fn test_find_site_package_path_no_py_typed() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "foo",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                        TestPath::dir("baz", vec![TestPath::file("__init__.pyi")]),
                    ],
                ),
                TestPath::dir(
                    "foo-stubs",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                    ],
                ),
            ],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("foo-stubs/bar/__init__.py")
            )),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_site_package_path_no_stubs() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                    TestPath::dir("baz", vec![TestPath::file("__init__.pyi")]),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz/__init__.pyi")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_site_package_path_partial_py_typed() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "foo",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                        TestPath::dir("baz", vec![TestPath::file("__init__.pyi")]),
                    ],
                ),
                TestPath::dir(
                    "foo-stubs",
                    vec![TestPath::file_with_contents("py.typed", "partial\n")],
                ),
            ],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar/__init__.py"))),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz/__init__.pyi")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_site_package_path_no_stubs_with_py_typed() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::file("py.typed"),
                    TestPath::file("__init__.py"),
                    TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                    TestPath::dir("baz", vec![TestPath::file("__init__.pyi")]),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/bar/__init__.py"))),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_site_package_path_no_source() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "foo-stubs",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::dir("bar", vec![TestPath::file("__init__.py")]),
                    ],
                ),
                TestPath::dir("baz", vec![]),
                TestPath::dir(
                    "baz-stubs",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::dir("qux", vec![TestPath::file("__init__.py")]),
                    ],
                ),
            ],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::Finding(Finding {
                finding: ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
                error: Some(FindError::MissingSource(ModuleName::from_str("foo.bar"))),
            })
        );
        assert!(matches!(
            find_module(
                ModuleName::from_str("baz.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::Finding(Finding {
                finding: _,
                error: Some(FindError::MissingSource(_)),
            })
        ));
    }

    #[test]
    fn test_find_namespaces_with_nested_init_files() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "nampspace",
                    vec![TestPath::dir("a", vec![TestPath::file("__init__.py")])],
                ),
                TestPath::dir(
                    "namespace",
                    vec![
                        TestPath::dir("a", vec![TestPath::file("__init__.py")]),
                        TestPath::dir("b", vec![TestPath::file("__init__.py")]),
                    ],
                ),
            ],
        );

        let mut namespaces = vec![];
        assert_eq!(
            find_module(
                ModuleName::from_str("namespace"),
                [root.to_path_buf()].iter(),
                &mut namespaces,
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
        assert_eq!(namespaces, vec![root.join("namespace")]);
        assert_eq!(
            find_module(
                ModuleName::from_str("namespace.a"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("namespace/a/__init__.py")
            ))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("namespace.b"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("namespace/b/__init__.py")
            ))
        );
    }

    #[test]
    fn test_find_compiled_module() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("compiled_module.pyc")]);
        let find_compiled_result = find_module(
            ModuleName::from_str("compiled_module"),
            [root.to_path_buf()].iter(),
            &mut vec![],
            None,
            None,
            false,
            &mut None,
            &DirEntryCache::new(),
            None,
        );
        assert_eq!(
            find_compiled_result.unwrap(),
            FindingOrError::Error(FindError::Ignored)
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("compiled_module.nested"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
    }

    #[test]
    fn test_find_compiled_module_with_source() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::file("foo.py"), TestPath::file("foo.pyc")],
        );
        // Ensure that the source file takes precedence over the compiled file
        assert_eq!(
            find_module(
                ModuleName::from_str("foo"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("foo.py")))
        );
    }

    #[test]
    fn test_nested_imports_with_compiled_modules() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "subdir",
                vec![
                    TestPath::file("another_compiled_module.pyc"),
                    TestPath::file("nested_import.py"),
                ],
            )],
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("subdir.nested_import"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("subdir/nested_import.py")
            ))
        );
        let find_compiled_result = find_module(
            ModuleName::from_str("subdir.another_compiled_module"),
            [root.to_path_buf()].iter(),
            &mut vec![],
            None,
            None,
            false,
            &mut None,
            &DirEntryCache::new(),
            None,
        );
        assert_eq!(
            find_compiled_result.unwrap(),
            FindingOrError::Error(FindError::Ignored)
        );
    }

    #[test]
    fn test_find_module_filter_basic() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("bar.py"),
                TestPath::file("bar.pyi"),
                TestPath::dir(
                    "module",
                    vec![
                        TestPath::file("__init__.py"),
                        TestPath::file("__init__.pyi"),
                    ],
                ),
            ],
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("bar.py")))
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("module"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Interface),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("module/__init__.pyi")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("module"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("module/__init__.py")))
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Interface),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("bar.pyi")))
        );
    }

    #[test]
    fn test_find_module_with_filter_pyc_treated_as_executable() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::file("bar.pyc"), TestPath::file("bar.pyi")],
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::Error(FindError::Ignored)
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Interface),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("bar.pyi")))
        );
    }

    #[test]
    fn test_find_module_with_filter_init_pyi() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "baz",
                vec![TestPath::file("__init__.pyi"), TestPath::file("bar.py")],
            )],
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("baz.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("baz").join("bar.py")))
        );
    }

    #[test]
    fn test_find_module_with_style_filter_across_roots() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("search_root1", vec![TestPath::file("standalone.pyi")]),
                TestPath::dir(
                    "search_root2",
                    vec![
                        TestPath::file("standalone.py"),
                        TestPath::file("standalone2.py"),
                    ],
                ),
            ],
        );

        let search_roots = [root.join("search_root1"), root.join("search_root2")];

        assert_eq!(
            find_module(
                ModuleName::from_str("standalone"),
                search_roots.iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root2/standalone.py")
            ))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("standalone"),
                search_roots.iter(),
                &mut vec![],
                Some(ModuleStyle::Interface),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root1/standalone.pyi")
            ))
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("standalone2"),
                search_roots.iter(),
                &mut vec![],
                Some(ModuleStyle::Interface),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            ),
            None
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("standalone2"),
                search_roots.iter(),
                &mut vec![],
                Some(ModuleStyle::Executable),
                None,
                false,
                &mut None,
                &DirEntryCache::new(),
                None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("search_root2/standalone2.py")
            ))
        );
    }

    fn get_config(source: ConfigSource) -> ConfigFile {
        let mut interpreters = Interpreters::default();
        interpreters.skip_interpreter_query = true;
        let mut config = ConfigFile {
            interpreters,
            python_environment: PythonEnvironment {
                site_package_path: Some(vec![]),
                ..Default::default()
            },
            source,
            ..Default::default()
        };
        config.configure();
        config
    }

    #[test]
    fn test_find_import_uses_typeshed_third_party_without_config() {
        let mut config = get_config(ConfigSource::Synthetic);
        let config_root = std::env::current_dir().unwrap();
        config.rewrite_with_path_to_config(&config_root);

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );
        assert!(
            matches!(result, FindingOrError::Finding(_)),
            "Expected to find 'requests' from typeshed third party stubs without a config file, but got: {:?}",
            result
        );
    }

    #[test]
    fn test_find_import_returns_not_found_with_real_config_but_no_installed_package() {
        let mut config = get_config(ConfigSource::File("".into()));
        let config_root = std::env::current_dir().unwrap();
        config.rewrite_with_path_to_config(&config_root);

        assert!(config.from_real_config_file());

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );
        assert!(
            matches!(
                &result,
                FindingOrError::Error(FindError::MissingImport(_, _))
            ),
            "Expected NotFound error for 'requests' with real config file but package not installed, but got: {:?}",
            result
        );
    }

    #[test]
    fn test_find_import_uses_typeshed_third_party_with_marker_config() {
        let mut config = get_config(ConfigSource::Marker("".into()));
        let config_root = std::env::current_dir().unwrap();
        config.rewrite_with_path_to_config(&config_root);

        assert!(!config.from_real_config_file());
        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );
        assert!(
            matches!(result, FindingOrError::Finding(_)),
            "Expected to find 'requests' from typeshed third party stubs with a marker config file, but got: {:?}",
            result
        );
    }

    #[test]
    fn test_find_import_prefixes_handles_typeshed_third_party() {
        let mut config_synthetic = get_config(ConfigSource::Synthetic);
        let config_root = std::env::current_dir().unwrap();
        config_synthetic.rewrite_with_path_to_config(&config_root);

        let prefixes_synthetic = find_import_prefixes(&config_synthetic, ModuleName::from_str(""));
        let has_requests_synthetic = prefixes_synthetic.iter().any(|m| m.as_str() == "requests");
        assert!(
            has_requests_synthetic,
            "find_import_prefixes should include typeshed third party stubs without a real config file"
        );

        let mut config_file = get_config(ConfigSource::File("".into()));
        config_file.rewrite_with_path_to_config(&config_root);
        assert!(config_file.from_real_config_file());
        let prefixes_file = find_import_prefixes(&config_file, ModuleName::from_str(""));
        let has_requests_file = prefixes_file.iter().any(|m| m.as_str() == "requests");
        assert!(
            !has_requests_file,
            "find_import_prefixes should NOT include typeshed third party stubs with a real config file"
        );
    }

    #[test]
    fn test_real_config_file_with_third_party_stub_returns_not_found() {
        let config = get_config(ConfigSource::File("".into()));
        assert!(config.from_real_config_file());
        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        // Should return NotFound error when using real config and typeshed third party stubs exist but package is not installed
        let error = result.error().expect("Expected error to be present");
        assert!(
            matches!(error, FindError::MissingImport(_, _)),
            "Expected NotFound error with real config, got: {:?}",
            error
        );

        if let FindError::MissingImport(module, _) = error {
            assert_eq!(module, ModuleName::from_str("requests"));
        }
    }

    #[test]
    fn test_missing_stubs_error_not_created_without_real_config() {
        let config_synthetic = get_config(ConfigSource::Synthetic);
        let result_synthetic = find_import_filtered(
            &config_synthetic,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );
        assert!(
            matches!(result_synthetic, FindingOrError::Finding(_)),
            "Should find the module in typeshed third party with synthetic config, got: {:?}",
            result_synthetic
        );

        let config_marker = get_config(ConfigSource::Marker("".into()));
        let result_marker = find_import_filtered(
            &config_marker,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );
        assert!(
            matches!(result_marker, FindingOrError::Finding(_)),
            "Should find the module in typeshed third party with marker config, got: {:?}",
            result_marker
        );
    }

    #[test]
    fn test_typeshed_third_party_with_real_config_recommends_types_package() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with 'requests' installed but no stubs
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "requests",
                    vec![TestPath::file("__init__.py")],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::File("".into()));
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = &result {
            let error = finding
                .error
                .as_ref()
                .expect("Expected UntypedImport error");
            let FindError::UntypedImport(module, stubs_package) = error else {
                panic!("Expected UntypedImport error, got: {:?}", error);
            };
            assert_eq!(*module, ModuleName::from_str("requests"));
            assert_eq!(stubs_package.as_str(), "types-requests");
        } else {
            panic!(
                "Expected Finding with UntypedImport error, got: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_typeshed_third_party_with_real_config_and_py_typed_no_recommendation() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with typed 'requests' installed but no stubs
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "requests",
                    vec![TestPath::file("py.typed"), TestPath::file("__init__.py")],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::File("".into()));
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = &result {
            assert!(
                finding.error.is_none(),
                "Expected no UntypedImport error for typed package, got: {:?}",
                finding.error
            );
        } else {
            panic!("Expected Finding, got: {:?}", result);
        }
    }

    #[test]
    fn test_typeshed_third_party_with_real_config_and_py_typed_submodule_no_recommendation() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with typed 'requests' package and submodule
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "requests",
                    vec![
                        TestPath::file("py.typed"),
                        TestPath::file("__init__.py"),
                        TestPath::dir("api", vec![TestPath::file("__init__.py")]),
                    ],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::File("".into()));
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests.api"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = &result {
            assert!(
                finding.error.is_none(),
                "Expected no UntypedImport error for typed package submodule, got: {:?}",
                finding.error
            );
        } else {
            panic!("Expected Finding, got: {:?}", result);
        }
    }

    #[test]
    fn test_typeshed_third_party_with_real_config_and_py_typed_submodule_file_no_recommendation() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with typed 'requests' package and submodule file
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "requests",
                    vec![
                        TestPath::file("py.typed"),
                        TestPath::file("__init__.py"),
                        TestPath::file("api.py"),
                    ],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::File("".into()));
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests.api"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = &result {
            assert!(
                finding.error.is_none(),
                "Expected no UntypedImport error for typed package submodule file, got: {:?}",
                finding.error
            );
        } else {
            panic!("Expected Finding, got: {:?}", result);
        }
    }

    #[test]
    fn test_typeshed_third_party_recommends_correct_package_for_dateutil() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with 'dateutil' installed but no stubs
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "dateutil",
                    vec![TestPath::file("__init__.py")],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::File("".into()));
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        let result = find_import_filtered(
            &config,
            ModuleName::from_str("dateutil"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = &result {
            let error = finding
                .error
                .as_ref()
                .expect("Expected UntypedImport error");
            let FindError::UntypedImport(module, stubs_package) = error else {
                panic!("Expected UntypedImport error, got: {:?}", error);
            };
            assert_eq!(*module, ModuleName::from_str("dateutil"));
            assert_eq!(stubs_package.as_str(), "types-python-dateutil");
        } else {
            panic!(
                "Expected Finding with UntypedImport error, got: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_typeshed_third_party_no_with_no_package_returns_typeshed_source_error() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up empty site package directory (no actual 'requests' package)
        TestPath::setup_test_directory(root, vec![TestPath::dir("site_packages", vec![])]);

        let mut config = get_config(ConfigSource::Synthetic);
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        // 'requests' exists in typeshed third party stubs but not in our site_packages
        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        let FindError::MissingSourceForStubs(module) = result.error().unwrap() else {
            panic!("Expected MissingSourceForStubs error");
        };
        assert_eq!(module, ModuleName::from_str("requests"));
    }

    #[test]
    fn test_typeshed_third_party_with_package_does_not_return_source_error() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Set up site package directory with actual 'requests' package
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "site_packages",
                vec![TestPath::dir(
                    "requests",
                    vec![TestPath::file("__init__.py")],
                )],
            )],
        );

        let mut config = get_config(ConfigSource::Synthetic);
        config.python_environment.site_package_path = Some(vec![root.join("site_packages")]);
        config.configure();

        // 'requests' exists in both typeshed third party stubs AND site_packages
        let result = find_import_filtered(
            &config,
            ModuleName::from_str("requests"),
            None,
            None,
            &DirEntryCache::new(),
            None,
        );

        if let FindingOrError::Finding(finding) = result {
            assert!(matches!(
                finding.finding.details(),
                ModulePathDetails::BundledTypeshedThirdParty(_)
            ));
            // Should not have a MissingSource error since the package exists
            assert!(finding.error.is_none());
        } else {
            panic!("Expected to find typeshed stub, got: {:?}", result);
        }
    }

    #[test]
    fn test_find_third_party_stub_prioritizes_typeshed_over_bundled() {
        // 'requests' exists in typeshed third party stubs, so it should
        // return BundledTypeshedThirdParty (typeshed is prioritized over other bundled stubs)
        let result = find_third_party_stub(ModuleName::from_str("requests"), None);
        assert!(result.is_some(), "Should find 'requests' stub");

        if let Some(FindingOrError::Finding(finding)) = result {
            assert!(
                matches!(
                    finding.finding.details(),
                    ModulePathDetails::BundledTypeshedThirdParty(_)
                ),
                "Expected BundledTypeshedThirdParty for 'requests', got: {:?}",
                finding.finding.details()
            );
        } else {
            panic!("Expected Finding result for 'requests', got: {:?}", result);
        }
    }

    #[test]
    fn test_find_third_party_stub_returns_bundled_when_not_in_typeshed() {
        // 'conans' exists only in BundledThirdParty (from third_party/stubs/conans-stubs),
        // not in typeshed, so it should return BundledThirdParty
        let result = find_third_party_stub(ModuleName::from_str("conans"), None);
        assert!(result.is_some(), "Should find 'conans' stub");

        if let Some(FindingOrError::Finding(finding)) = result {
            assert!(
                matches!(
                    finding.finding.details(),
                    ModulePathDetails::BundledThirdParty(_)
                ),
                "Expected BundledThirdParty for 'conans', got: {:?}",
                finding.finding.details()
            );
        } else {
            panic!("Expected Finding result for 'conans', got: {:?}", result);
        }
    }

    #[test]
    fn test_suggest_stdlib_import() {
        // Test that we suggest 'math' for 'mathh' (one character typo)
        let suggestion = suggest_stdlib_import(ModuleName::from_str("mathh"));
        assert_eq!(
            suggestion,
            Some(ModuleName::from_str("math")),
            "Should suggest 'math' for 'mathh'"
        );

        // Test that we suggest 'os' for 'oss' (one character typo)
        let suggestion = suggest_stdlib_import(ModuleName::from_str("oss"));
        assert_eq!(
            suggestion,
            Some(ModuleName::from_str("os")),
            "Should suggest 'os' for 'oss'"
        );

        // Test that we suggest 'json' for 'jsn' (missing character)
        let suggestion = suggest_stdlib_import(ModuleName::from_str("jsn"));
        assert_eq!(
            suggestion,
            Some(ModuleName::from_str("json")),
            "Should suggest 'json' for 'jsn'"
        );

        // Test that we don't suggest for completely unrelated names
        let suggestion = suggest_stdlib_import(ModuleName::from_str("xyzabc123"));
        assert_eq!(
            suggestion, None,
            "Should not suggest for completely unrelated names"
        );
    }

    // -------------- Phantom Paths Tests --------------------

    fn get_config_with_search_path(search_path: Vec<PathBuf>) -> ConfigFile {
        let mut interpreters = Interpreters::default();
        interpreters.skip_interpreter_query = true;
        let mut config = ConfigFile {
            interpreters,
            python_environment: PythonEnvironment {
                site_package_path: Some(vec![]),
                ..Default::default()
            },
            search_path_from_file: search_path,
            source: ConfigSource::Synthetic,
            // Disable fallback search path heuristics to avoid extra phantom paths
            disable_search_path_heuristics: true,
            ..Default::default()
        };
        config.configure();
        config
    }

    #[test]
    fn test_phantom_paths_module_not_found() {
        // When a module is not found at all, all checked paths should be collected
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        // Create an empty directory structure
        TestPath::setup_test_directory(root, vec![]);

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let _result = find_import(
            &config,
            ModuleName::from_str("nonexistent"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        // find_import first checks for -stubs package, then the regular package
        // Should have checked paths for both nonexistent-stubs and nonexistent
        let expected_paths = vec![
            // -stubs package check
            root.join("nonexistent-stubs/__init__.pyi"),
            root.join("nonexistent-stubs/__init__.py"),
            root.join("nonexistent-stubs.pyi"),
            root.join("nonexistent-stubs.py"),
            root.join("nonexistent-stubs.pyc"),
            root.join("nonexistent-stubs.pyx"),
            root.join("nonexistent-stubs.pyd"),
            root.join("nonexistent-stubs"),
            // Regular package check
            root.join("nonexistent/__init__.pyi"),
            root.join("nonexistent/__init__.py"),
            root.join("nonexistent.pyi"),
            root.join("nonexistent.py"),
            root.join("nonexistent.pyc"),
            root.join("nonexistent.pyx"),
            root.join("nonexistent.pyd"),
            root.join("nonexistent"),
        ];

        assert_eq!(
            phantom_paths, expected_paths,
            "Should collect all checked paths when module not found"
        );
    }

    #[test]
    fn test_phantom_paths_regular_package_found_immediately() {
        // When a regular package is found with __init__.pyi, phantom paths are collected
        // for the -stubs package that was checked first
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "mypackage",
                vec![TestPath::file("__init__.pyi")],
            )],
        );

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mypackage"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find the package");
        // find_import checks -stubs first, so we have phantom paths from that check
        let expected = vec![
            root.join("mypackage-stubs/__init__.pyi"),
            root.join("mypackage-stubs/__init__.py"),
            root.join("mypackage-stubs.pyi"),
            root.join("mypackage-stubs.py"),
            root.join("mypackage-stubs.pyc"),
            root.join("mypackage-stubs.pyx"),
            root.join("mypackage-stubs.pyd"),
            root.join("mypackage-stubs"),
        ];
        assert_eq!(
            phantom_paths, expected,
            "Should collect -stubs phantom paths before finding regular package"
        );
    }

    #[test]
    fn test_phantom_paths_regular_package_found_with_init_py() {
        // When __init__.pyi doesn't exist but __init__.py does
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "mypackage",
                vec![TestPath::file("__init__.py")],
            )],
        );

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mypackage"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find the package");
        // Phantom paths from -stubs check + __init__.pyi before __init__.py
        let expected = vec![
            // -stubs check
            root.join("mypackage-stubs/__init__.pyi"),
            root.join("mypackage-stubs/__init__.py"),
            root.join("mypackage-stubs.pyi"),
            root.join("mypackage-stubs.py"),
            root.join("mypackage-stubs.pyc"),
            root.join("mypackage-stubs.pyx"),
            root.join("mypackage-stubs.pyd"),
            root.join("mypackage-stubs"),
            // Regular package - __init__.pyi checked before __init__.py
            root.join("mypackage/__init__.pyi"),
        ];
        assert_eq!(
            phantom_paths, expected,
            "Should collect -stubs and __init__.pyi phantom paths"
        );
    }

    #[test]
    fn test_phantom_paths_single_file_pyi_found() {
        // When a .pyi file is found (no init files exist)
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(root, vec![TestPath::file("mymodule.pyi")]);

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mymodule"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find the module");
        // Phantom paths from -stubs check + init files before .pyi
        let expected = vec![
            // -stubs check
            root.join("mymodule-stubs/__init__.pyi"),
            root.join("mymodule-stubs/__init__.py"),
            root.join("mymodule-stubs.pyi"),
            root.join("mymodule-stubs.py"),
            root.join("mymodule-stubs.pyc"),
            root.join("mymodule-stubs.pyx"),
            root.join("mymodule-stubs.pyd"),
            root.join("mymodule-stubs"),
            // Regular module - init files checked before .pyi
            root.join("mymodule/__init__.pyi"),
            root.join("mymodule/__init__.py"),
        ];
        assert_eq!(
            phantom_paths, expected,
            "Should collect -stubs and init phantom paths before finding .pyi"
        );
    }

    #[test]
    fn test_phantom_paths_single_file_py_found() {
        // When only a .py file exists (no .pyi)
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(root, vec![TestPath::file("mymodule.py")]);

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mymodule"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find the module");
        // Phantom paths from -stubs check + paths before .py
        let expected = vec![
            // -stubs check
            root.join("mymodule-stubs/__init__.pyi"),
            root.join("mymodule-stubs/__init__.py"),
            root.join("mymodule-stubs.pyi"),
            root.join("mymodule-stubs.py"),
            root.join("mymodule-stubs.pyc"),
            root.join("mymodule-stubs.pyx"),
            root.join("mymodule-stubs.pyd"),
            root.join("mymodule-stubs"),
            // Regular module - paths checked before .py
            root.join("mymodule/__init__.pyi"),
            root.join("mymodule/__init__.py"),
            root.join("mymodule.pyi"),
        ];
        assert_eq!(
            phantom_paths, expected,
            "Should collect all paths before finding .py"
        );
    }

    #[test]
    fn test_phantom_paths_compiled_module_found() {
        // When only a compiled module (.pyc) exists, it gets ignored for type checking
        // but we still collect phantom paths up to that point
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(root, vec![TestPath::file("mymodule.pyc")]);

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mymodule"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        // Compiled modules are ignored for type checking (no source/type info)
        assert!(
            result.finding().is_none(),
            "Compiled modules are ignored for type checking"
        );
        // Phantom paths from -stubs check + paths checked before finding .pyc
        let expected = vec![
            // -stubs check
            root.join("mymodule-stubs/__init__.pyi"),
            root.join("mymodule-stubs/__init__.py"),
            root.join("mymodule-stubs.pyi"),
            root.join("mymodule-stubs.py"),
            root.join("mymodule-stubs.pyc"),
            root.join("mymodule-stubs.pyx"),
            root.join("mymodule-stubs.pyd"),
            root.join("mymodule-stubs"),
            // Regular module - paths checked before .pyc
            root.join("mymodule/__init__.pyi"),
            root.join("mymodule/__init__.py"),
            root.join("mymodule.pyi"),
            root.join("mymodule.py"),
        ];
        assert_eq!(
            phantom_paths, expected,
            "Should collect all paths before compiled module"
        );
    }

    #[test]
    fn test_phantom_paths_multipart_module() {
        // Test phantom paths collection during multi-part module resolution
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "parent",
                vec![
                    TestPath::file("__init__.py"), // Has __init__.py, not .pyi
                    TestPath::file("child.py"),    // Has .py, not .pyi
                ],
            )],
        );

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("parent.child"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find parent.child");

        // Phantom paths from:
        // 1. parent-stubs check (all paths since not found)
        // 2. parent check (__init__.pyi before __init__.py)
        // 3. child check (__init__.pyi, __init__.py, .pyi before .py)
        let expected = vec![
            // parent-stubs check
            root.join("parent-stubs/__init__.pyi"),
            root.join("parent-stubs/__init__.py"),
            root.join("parent-stubs.pyi"),
            root.join("parent-stubs.py"),
            root.join("parent-stubs.pyc"),
            root.join("parent-stubs.pyx"),
            root.join("parent-stubs.pyd"),
            root.join("parent-stubs"),
            // parent check - __init__.pyi before __init__.py
            root.join("parent/__init__.pyi"),
            // child check within parent - paths before .py
            root.join("parent/child/__init__.pyi"),
            root.join("parent/child/__init__.py"),
            root.join("parent/child.pyi"),
        ];

        assert_eq!(
            phantom_paths, expected,
            "Should collect phantom paths from all levels of module resolution"
        );
    }

    #[test]
    fn test_phantom_paths_multiple_search_paths() {
        // Test that phantom paths are collected from all search paths until match is found
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("root1", vec![]), // Empty first root
                TestPath::dir(
                    "root2",
                    vec![TestPath::file("mymodule.py")], // Module in second root
                ),
            ],
        );

        let config = get_config_with_search_path(vec![root.join("root1"), root.join("root2")]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("mymodule"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(
            result.finding().is_some(),
            "Should find the module in second root"
        );

        // Phantom paths from:
        // 1. mymodule-stubs check across both roots
        // 2. mymodule check (root1 all, root2 until .py found)
        let expected = vec![
            // -stubs check in root1 (all paths)
            root.join("root1/mymodule-stubs/__init__.pyi"),
            root.join("root1/mymodule-stubs/__init__.py"),
            root.join("root1/mymodule-stubs.pyi"),
            root.join("root1/mymodule-stubs.py"),
            root.join("root1/mymodule-stubs.pyc"),
            root.join("root1/mymodule-stubs.pyx"),
            root.join("root1/mymodule-stubs.pyd"),
            root.join("root1/mymodule-stubs"),
            // -stubs check in root2 (all paths)
            root.join("root2/mymodule-stubs/__init__.pyi"),
            root.join("root2/mymodule-stubs/__init__.py"),
            root.join("root2/mymodule-stubs.pyi"),
            root.join("root2/mymodule-stubs.py"),
            root.join("root2/mymodule-stubs.pyc"),
            root.join("root2/mymodule-stubs.pyx"),
            root.join("root2/mymodule-stubs.pyd"),
            root.join("root2/mymodule-stubs"),
            // Regular check in root1 (all paths since not found)
            root.join("root1/mymodule/__init__.pyi"),
            root.join("root1/mymodule/__init__.py"),
            root.join("root1/mymodule.pyi"),
            root.join("root1/mymodule.py"),
            root.join("root1/mymodule.pyc"),
            root.join("root1/mymodule.pyx"),
            root.join("root1/mymodule.pyd"),
            root.join("root1/mymodule"),
            // Regular check in root2 (paths before .py found)
            root.join("root2/mymodule/__init__.pyi"),
            root.join("root2/mymodule/__init__.py"),
            root.join("root2/mymodule.pyi"),
        ];

        assert_eq!(
            phantom_paths, expected,
            "Should collect phantom paths from all search paths until match"
        );
    }

    #[test]
    fn test_phantom_paths_deep_nesting() {
        // Test phantom paths with deeply nested module resolution
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "a",
                vec![
                    TestPath::file("__init__.py"),
                    TestPath::dir(
                        "b",
                        vec![
                            TestPath::file("__init__.py"),
                            TestPath::dir(
                                "c",
                                vec![TestPath::file("__init__.py"), TestPath::file("d.py")],
                            ),
                        ],
                    ),
                ],
            )],
        );

        let config = get_config_with_search_path(vec![root.to_path_buf()]);
        let mut phantom_paths = vec![];

        let result = find_import(
            &config,
            ModuleName::from_str("a.b.c.d"),
            None,
            Some(&mut phantom_paths),
            &DirEntryCache::new(),
            None,
        );

        assert!(result.finding().is_some(), "Should find a.b.c.d");

        // Phantom paths from:
        // 1. a-stubs check (all paths)
        // 2. a check (__init__.pyi)
        // 3. b check (__init__.pyi)
        // 4. c check (__init__.pyi)
        // 5. d check (init files, .pyi before .py)
        let expected = vec![
            // a-stubs check
            root.join("a-stubs/__init__.pyi"),
            root.join("a-stubs/__init__.py"),
            root.join("a-stubs.pyi"),
            root.join("a-stubs.py"),
            root.join("a-stubs.pyc"),
            root.join("a-stubs.pyx"),
            root.join("a-stubs.pyd"),
            root.join("a-stubs"),
            // a: __init__.pyi before __init__.py
            root.join("a/__init__.pyi"),
            // b: __init__.pyi before __init__.py
            root.join("a/b/__init__.pyi"),
            // c: __init__.pyi before __init__.py
            root.join("a/b/c/__init__.pyi"),
            // d: init files and .pyi before .py
            root.join("a/b/c/d/__init__.pyi"),
            root.join("a/b/c/d/__init__.py"),
            root.join("a/b/c/d.pyi"),
        ];

        assert_eq!(
            phantom_paths, expected,
            "Should collect phantom paths at all nesting levels"
        );
    }

    #[test]
    fn test_find_extra_extension_module_simple() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // Create a directory with extra-extension files (.cinc and .cconf).
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "service",
                vec![
                    TestPath::file("config.cinc"),
                    TestPath::file("settings.cconf"),
                ],
            )],
        );
        let extra = vec!["cinc".to_owned(), "cconf".to_owned()];

        // `import service.config.cinc` should find `service/config.cinc`
        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("service.config.cinc"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("service/config.cinc")))
        );

        // `import service.settings.cconf` should find `service/settings.cconf`
        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("service.settings.cconf"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("service/settings.cconf")
            ))
        );

        // Module without an extra extension should return None.
        assert!(
            find_extra_extension_module(
                ModuleName::from_str("service.config.py"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .is_none()
        );

        // Module without extra extension as last component should return None.
        assert!(
            find_extra_extension_module(
                ModuleName::from_str("service.config"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .is_none()
        );
    }

    #[test]
    fn test_find_extra_extension_module_cinc_py_not_resolved() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // A file like `foo.cinc.py` has `.py` as its real extension, not `.cinc`.
        // The extra extension finder should not resolve it because `py` is not
        // in the extra extensions list.
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "service",
                vec![TestPath::file("config.cinc.py")],
            )],
        );
        let extra = vec!["cinc".to_owned(), "cconf".to_owned()];

        assert!(
            find_extra_extension_module(
                ModuleName::from_str("service.config.cinc.py"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .is_none()
        );
    }

    #[test]
    fn test_find_extra_extension_module_dotted_filename() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // Create a flat file with dots in its name.
        TestPath::setup_test_directory(root, vec![TestPath::file("service.config.cinc")]);
        let extra = vec!["cinc".to_owned()];

        // `import service.config.cinc` should find `service.config.cinc`
        // (the flat dotted filename) when no directory structure exists.
        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("service.config.cinc"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("service.config.cinc")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_precedence() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // When both `a/b.cinc` and `a.b.cinc` exist, prefer the one with more
        // directory components (closer to source).
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir("a", vec![TestPath::file("b.cinc")]),
                TestPath::file("a.b.cinc"),
            ],
        );
        let extra = vec!["cinc".to_owned()];

        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("a.b.cinc"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("a/b.cinc")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_phantom_paths() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // No files exist — all candidates should be recorded as phantom paths.
        let extra = vec!["cinc".to_owned()];
        let mut phantom_paths = Vec::new();
        let result = find_extra_extension_module(
            ModuleName::from_str("a.b.cinc"),
            [root.to_path_buf()].iter(),
            &extra,
            &mut Some(&mut phantom_paths),
        );
        assert!(result.is_none());
        // For module `a.b.cinc` with one root, the finder checks two candidates
        // plus their .pyi stubs:
        //   dir_count=1: root/a/b.cinc.pyi, root/a/b.cinc
        //   dir_count=0: root/a.b.cinc.pyi, root/a.b.cinc
        assert_eq!(
            phantom_paths,
            vec![
                root.join("a/b.cinc.pyi"),
                root.join("a/b.cinc"),
                root.join("a.b.cinc.pyi"),
                root.join("a.b.cinc"),
            ]
        );
    }

    #[test]
    fn test_find_extra_extension_module_pyi_stub_preferred() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // When both `a/b.thrift` and `a/b.thrift.pyi` exist, prefer the .pyi stub.
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "a",
                vec![TestPath::file("b.thrift"), TestPath::file("b.thrift.pyi")],
            )],
        );
        let extra = vec!["thrift".to_owned()];

        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("a.b.thrift"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("a/b.thrift.pyi")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_pyi_stub_only() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // When only `a/b.thrift.pyi` exists (no raw .thrift file), it should
        // still be found. This is the typical case for python_type_stubs/.
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir("a", vec![TestPath::file("b.thrift.pyi")])],
        );
        let extra = vec!["thrift".to_owned()];

        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("a.b.thrift"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("a/b.thrift.pyi")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_pyi_stub_across_roots() {
        let tempdir = tempfile::tempdir().unwrap();
        let root1 = tempdir.path().join("source");
        let root2 = tempdir.path().join("source/python_type_stubs");
        // Root 1 has the raw .thrift file, root 2 has the .pyi stub.
        // The stub in root 2 should be found because root 2 is searched first
        // (when listed first in the search path).
        TestPath::setup_test_directory(
            tempdir.path(),
            vec![TestPath::dir(
                "source",
                vec![
                    TestPath::dir("a", vec![TestPath::file("b.thrift")]),
                    TestPath::dir(
                        "python_type_stubs",
                        vec![TestPath::dir("a", vec![TestPath::file("b.thrift.pyi")])],
                    ),
                ],
            )],
        );
        let extra = vec!["thrift".to_owned()];

        // When python_type_stubs root is listed first, the .pyi stub is found.
        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("a.b.thrift"),
                [root2.clone(), root1.clone()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root2.join("a/b.thrift.pyi")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_keyword_escaping() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // Configerator directories can be named with Python keywords (e.g. `if`).
        // Since `if` is a Python keyword, the module name uses `if_` (with trailing
        // underscore). The finder should strip the underscore to find the real directory.
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "some",
                vec![TestPath::dir("if", vec![TestPath::file("config.cconf")])],
            )],
        );
        let extra = vec!["cconf".to_owned()];

        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("some.if_.config.cconf"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("some/if/config.cconf")))
        );
    }

    #[test]
    fn test_find_extra_extension_module_keyword_in_filename() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // When a Python keyword appears as part of the dotted filename portion
        // (not just directory), the trailing underscore should also be stripped.
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "rules",
                vec![TestPath::file("if.config.cconf")],
            )],
        );
        let extra = vec!["cconf".to_owned()];

        assert_eq!(
            find_extra_extension_module(
                ModuleName::from_str("rules.if_.config.cconf"),
                [root.to_path_buf()].iter(),
                &extra,
                &mut None,
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("rules/if.config.cconf")))
        );
    }
}
