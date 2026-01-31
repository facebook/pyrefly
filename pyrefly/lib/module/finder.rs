/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::locked_map::LockedMap;
use pyrefly_util::suggest::best_suggestion;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
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

#[derive(Debug, PartialEq, Clone)]
enum FindResult {
    /// Found a single-file .pyi module. The path must not point to an __init__ file.
    SingleFilePyiModule(PathBuf),
    /// Found a single-file .py module. The path must not point to an __init__ file.
    SingleFilePyModule(PathBuf),
    /// Found regular packages. First path must point to an __init__ file.
    /// Second path indicates where to continue search next. It should always point to the parent of the __init__ file.
    /// The ordering of packages should be the same as the order they're found
    /// in the `includes`.
    RegularPackage(PathBuf, PathBuf),
    /// Found a namespace package.
    /// The path component indicates where to continue search next. It may contain more than one directories as the namespace package
    /// may span across multiple search roots.
    NamespacePackage(Vec1<PathBuf>),
    /// Found a compiled Python file (.pyc, .pyx, .pyd). Represents some kind of
    /// compiled module, whether that's bytecode, C extension, or DLL.
    /// Compiled modules lack source and type info, and are
    /// treated as `typing.Any` to handle imports without type errors.
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

    /// Compares the given `FindResult`s, taking the variant with the highest priority,
    /// and preferring variant `a` (the 'earlier' variant). The contents of the variants
    /// are not compared.
    fn best_result(a: FindResult, b: FindResult) -> Self {
        match (&a, &b) {
            (FindResult::RegularPackage(..), _) => a,
            (_, FindResult::RegularPackage(..)) => b,
            (FindResult::SingleFilePyiModule(_), _) => a,
            (_, FindResult::SingleFilePyiModule(_)) => b,
            (FindResult::SingleFilePyModule(_), _) => a,
            (_, FindResult::SingleFilePyModule(_)) => b,
            (FindResult::CompiledModule(_), _) => a,
            (_, FindResult::CompiledModule(_)) => b,
            (FindResult::NamespacePackage(_), _) => a,
        }
    }

    /// Converts a `FindResult` into a [`ModulePath`], returning a [`FindError`] instead
    /// if the module is not reachable.
    fn module_path(self) -> FindingOrError<ModulePath> {
        match self {
            FindResult::SingleFilePyiModule(path)
            | FindResult::SingleFilePyModule(path)
            | FindResult::RegularPackage(path, _) => {
                FindingOrError::new_finding(ModulePath::filesystem(path))
            }
            FindResult::NamespacePackage(roots) => {
                // TODO(grievejia): Preserving all info in the list instead of dropping all but the first one.
                FindingOrError::new_finding(ModulePath::namespace(roots.first().clone()))
            }
            FindResult::CompiledModule(_) => FindingOrError::Error(FindError::Ignored),
        }
    }
}

/// In the given root, attempt to find a match for the given [`Name`].
///
/// If `style_filter` is provided, only results matching that style will be returned.
/// The function will check candidates in priority order and return the first match that satisfies the filter.
///
/// If `phantom_paths` is provided, paths that were checked but did not exist will be added to it.
/// Note: `phantom_paths` and `style_filter` are mutually exclusive.
fn find_one_part_in_root(
    name: &Name,
    root: &Path,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
) -> Option<FindResult> {
    let candidate_dir = root.join(name.as_str());
    // Do not filter by style filter here since __init__.pyi could potentially have .py files covered under it.
    // Instead, use `ModuleStyle` as a preference.
    let candidate_init_suffixes = if style_filter.is_some_and(|s| s == ModuleStyle::Executable) {
        ["__init__.py", "__init__.pyi"]
    } else {
        ["__init__.pyi", "__init__.py"]
    };

    // First check if `name` corresponds to a regular package.
    for candidate_init_suffix in candidate_init_suffixes {
        let init_path = candidate_dir.join(candidate_init_suffix);
        if init_path.exists() {
            return Some(FindResult::RegularPackage(init_path, candidate_dir));
        } else if let Some(v) = phantom_paths.as_deref_mut() {
            v.push(init_path);
        }
    }
    // Second check if `name` corresponds to a single-file module.
    for candidate_file_suffix in ["pyi", "py"] {
        let candidate_path = root.join(format!("{name}.{candidate_file_suffix}"));
        if candidate_path.exists() {
            let result = FindResult::single_file(candidate_path.clone(), candidate_file_suffix);
            if let Some(filter) = style_filter {
                if let Some(style) = result.style()
                    && style == filter
                {
                    return Some(result);
                }
                // else, continue the search
            } else {
                return Some(result);
            }
        } else if let Some(v) = phantom_paths.as_deref_mut() {
            v.push(candidate_path);
        }
    }

    // Check if `name` corresponds to a compiled module.
    for candidate_compiled_suffix in COMPILED_FILE_SUFFIXES {
        let candidate_path = root.join(format!("{name}.{candidate_compiled_suffix}"));
        if candidate_path.exists() {
            let result = FindResult::CompiledModule(candidate_path);
            if let Some(filter) = style_filter {
                // compiled files are considered executable
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
    // Finally check if `name` corresponds to a namespace package.
    if candidate_dir.is_dir() {
        let result = FindResult::NamespacePackage(Vec1::new(candidate_dir));
        // Namespace packages don't have a style in the same sense, so we return them regardless of filter
        return Some(result);
    } else if let Some(v) = phantom_paths.as_deref_mut() {
        v.push(candidate_dir);
    }
    None
}

/// Finds the first package (regular, single file, or namespace) in all search roots.
/// Returns None if no module is found. If `name` is `__pycache__`, we always
/// return `None`.
///
/// If `style_filter` is provided, only results matching that style will be returned.
/// The function will continue searching until it finds a result that matches the style.
///
/// If `phantom_paths` is provided, paths that were checked but did not exist will be added to it.
/// Note: `phantom_paths` and `style_filter` are mutually exclusive.
fn find_one_part<'a>(
    name: &Name,
    mut roots: impl Iterator<Item = &'a PathBuf>,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
) -> Option<(FindResult, Vec<PathBuf>)> {
    // skip looking in `__pycache__`, since those modules are not accessible
    if name == &Name::new_static("__pycache__") {
        return None;
    }
    let mut namespace_roots = Vec::new();
    while let Some(root) = roots.next() {
        match find_one_part_in_root(name, root, style_filter, phantom_paths) {
            None => (),
            Some(FindResult::NamespacePackage(package)) => {
                namespace_roots.push(package.first().clone())
            }
            Some(result) => return Some((result, roots.cloned().collect::<Vec<_>>())),
        }
    }
    match Vec1::try_from_vec(namespace_roots) {
        Err(_) => None,
        Ok(namespace_roots) => Some((FindResult::NamespacePackage(namespace_roots), vec![])),
    }
}

/// Finds the first package (regular, single file, or namespace) in search roots. Returns None if no module is found.
/// name: module name
/// roots: search roots
fn find_one_part_prefix<'a>(
    prefix: &Name,
    roots: impl Iterator<Item = &'a PathBuf>,
) -> Vec<(FindResult, ModuleName)> {
    let mut results = Vec::new();
    let mut namespace_roots: SmallMap<ModuleName, Vec<PathBuf>> = SmallMap::new();

    for root in roots {
        // List all entries in the root directory
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                let file_name = path.file_name().and_then(|n| n.to_str());

                if let Some(name) = file_name {
                    // Check if the name starts with the prefix
                    if name.starts_with(prefix.as_str()) {
                        // Check if it's a regular package
                        if path.is_dir() {
                            for candidate_init_suffix in ["__init__.pyi", "__init__.py"] {
                                let init_path = path.join(candidate_init_suffix);
                                if init_path.exists() {
                                    results.push((
                                        FindResult::RegularPackage(init_path, path.clone()),
                                        ModuleName::from_str(name),
                                    ));
                                    break;
                                }
                            }

                            if !results.iter().any(|r| match r {
                                (FindResult::RegularPackage(_, p), _) => p == &path,
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
    }

    // Add namespace packages to results
    for (name, roots) in namespace_roots {
        if let Ok(namespace_roots) = Vec1::try_from_vec(roots) {
            results.push((FindResult::NamespacePackage(namespace_roots), name));
        }
    }

    // todo: also return modulename so we know what to call this
    results
}

/// Find a module from a single package. Returns None if no module is found.
///
/// If `style_filter` is provided, only results matching that style will be returned.
/// The function will continue searching until it finds a result that matches the style.
///
/// If `phantom_paths` is provided, paths that were checked but did not exist will be added to it.
/// Note: `phantom_paths` and `style_filter` are mutually exclusive.
fn continue_find_module(
    start_result: FindResult,
    components_rest: &[Name],
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
) -> Option<FindResult> {
    let mut current_result = Some(start_result);
    for part in components_rest.iter() {
        match current_result {
            None => {
                // Nothing has been found in the previous round. No point keep looking.
                break;
            }
            Some(FindResult::SingleFilePyiModule(_))
            | Some(FindResult::SingleFilePyModule(_))
            | Some(FindResult::CompiledModule(_)) => {
                // We've already reached leaf nodes. Cannot keep searching
                current_result = None;
                break;
            }
            Some(FindResult::RegularPackage(_, next_root)) => {
                current_result =
                    find_one_part(part, [next_root].iter(), style_filter, phantom_paths)
                        .map(|x| x.0);
            }
            Some(FindResult::NamespacePackage(next_roots)) => {
                current_result =
                    find_one_part(part, next_roots.iter(), style_filter, phantom_paths)
                        .map(|x| x.0);
            }
        }
    }
    current_result
}

/// Attempt to find the given module from its first component (which might have
/// `-stubs` appended) and remaining components in the given `includes`.
/// If a result is found that might have a more preferable option later in the
/// includes, continue searching for it and return the best option.
///
/// If `style_filter` is provided, only modules matching that style will be returned.
/// The function will continue searching until it finds a module that matches the style.
///
/// If `phantom_paths` is provided, paths that were checked but did not exist will be added to it.
/// Note: `phantom_paths` and `style_filter` are mutually exclusive.
fn find_module_components<'a, I>(
    first: &Name,
    components_rest: &[Name],
    include: I,
    style_filter: Option<ModuleStyle>,
    phantom_paths: &mut Option<&mut Vec<PathBuf>>,
) -> Option<FindResult>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let (first_component_result, fallback_search) =
        find_one_part(first, include.clone(), style_filter, phantom_paths)?;

    let current_result = continue_find_module(
        first_component_result,
        components_rest,
        style_filter,
        phantom_paths,
    )?;

    match current_result {
        FindResult::SingleFilePyiModule(_) | FindResult::RegularPackage(..) => Some(current_result),
        _ => Some(
            fallback_search
                .into_iter()
                .filter_map(|s| Some(find_one_part(first, [s].iter(), style_filter, &mut None)?.0))
                .filter_map(|first| {
                    continue_find_module(first.clone(), components_rest, style_filter, &mut None)
                })
                .fold(current_result, FindResult::best_result),
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
) -> Option<FindingOrError<ModulePath>> {
    // This is the case where we do have a config file, the package is installed, but there are no stubs
    // available besides the bundled stubs. In this case
    // return the stub but with the error attached telling the user to install stubs.
    if let Some(ref bundled) = bundled_stub
        && from_real_config_file
        && normal_result.is_some()
        && stub_result.is_none()
    {
        if let Some(pip_package) = recommended_stubs_package(module) {
            return Some(bundled.clone().with_error(FindError::MissingStubs(
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
    // package (normal_result), we should attach a NoSource error.
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
                return Some(bundled.with_error(FindError::NoSourceForStubs(module)));
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
) -> Option<FindingOrError<ModulePath>> {
    match (normal_result, stub_result) {
        (None, Some(stub_result)) => Some(
            stub_result
                .module_path()
                .with_error(FindError::NoSource(module)),
        ),
        (Some(_), Some(stub_result)) => Some(stub_result.module_path()),
        (Some(FindResult::NamespacePackage(namespaces)), _) => {
            namespaces_found.append(&mut namespaces.into_vec());
            None
        }
        (Some(normal_result), None) => {
            if let Some(missing_stub_result) = recommended_stubs_package(module) {
                Some(
                    normal_result
                        .module_path()
                        .with_error(FindError::MissingStubs(
                            module,
                            missing_stub_result.as_str().to_owned().into(),
                        )),
                )
            } else {
                Some(normal_result.module_path())
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
/// If the result is a [`FindResult::NamespacePackage`], we instead add its entries to
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
) -> Option<FindingOrError<ModulePath>>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    match module.components().as_slice() {
        [] => None,
        [first, rest @ ..] => {
            // First try finding the module in `-stubs`.
            let stub_first = Name::new(format!("{first}-stubs"));
            let stub_result = find_module_components(
                &stub_first,
                rest,
                include.clone(),
                style_filter,
                phantom_paths,
            );

            // If we couldn't find it in a `-stubs` module or we want to check for missing stubs, look normally.
            let normal_result =
                find_module_components(first, rest, include.clone(), style_filter, phantom_paths);

            // Check if third-party stub should take precedence
            if let Some(result) = resolve_third_party_stub(
                module,
                stub_result.as_ref(),
                normal_result.as_ref(),
                typeshed_third_party_stub,
                from_real_config_file,
            ) {
                return Some(result);
            }

            combine_normal_and_stub_results(module, stub_result, normal_result, namespaces_found)
        }
    }
}

fn find_module_prefixes<'a>(
    prefix: ModuleName,
    include: impl Iterator<Item = &'a PathBuf>,
) -> Vec<ModuleName> {
    let components = prefix.components();
    let first = &components[0];
    let rest = &components[1..];
    let mut results = Vec::new();
    if rest.is_empty() {
        results = find_one_part_prefix(first, include)
    } else {
        let mut current_result = find_one_part(first, include, None, &mut None).map(|x| x.0);
        for (i, part) in rest.iter().enumerate() {
            let is_last = i == rest.len() - 1;
            match current_result {
                None => {
                    break;
                }
                Some(
                    FindResult::SingleFilePyiModule(_)
                    | FindResult::SingleFilePyModule(_)
                    | FindResult::CompiledModule(_),
                ) => {
                    break;
                }
                Some(FindResult::RegularPackage(_, next_root)) => {
                    if is_last {
                        results = find_one_part_prefix(part, iter::once(&next_root));
                        break;
                    } else {
                        current_result =
                            find_one_part(part, iter::once(&next_root), None, &mut None)
                                .map(|x| x.0);
                    }
                }
                Some(FindResult::NamespacePackage(next_roots)) => {
                    if is_last {
                        results = find_one_part_prefix(part, next_roots.iter());
                        break;
                    } else {
                        current_result =
                            find_one_part(part, next_roots.iter(), None, &mut None).map(|x| x.0);
                    }
                }
            }
        }
    }
    results.iter().map(|(_, name)| *name).collect::<Vec<_>>()
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
            |err| Some(FindingOrError::Error(FindError::not_found(err, module))),
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
            |err| Some(FindingOrError::Error(FindError::not_found(err, module))),
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
        )
    {
        path
    } else if matches!(style_filter, Some(ModuleStyle::Interface) | None)
        && let Some(path) = typeshed().map_or_else(
            |err| Some(FindingOrError::Error(FindError::not_found(err, module))),
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
    ) {
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
) -> FindingOrError<ModulePath> {
    find_import_internal(config, module, origin, None, &mut phantom_paths)
}

pub fn find_import_filtered(
    config: &ConfigFile,
    module: ModuleName,
    origin: Option<&ModulePath>,
    style_filter: Option<ModuleStyle>,
) -> FindingOrError<ModulePath> {
    find_import_internal(config, module, origin, style_filter, &mut None)
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
        _ => None,
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
            ),
            None,
        );
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
            )
            .unwrap(),
            FindingOrError::new_finding(ModulePath::filesystem(root.join("first/c/d/e.py")))
        );
    }

    #[test]
    fn test_find_regular_package_early_return() {
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
        assert_eq!(
            find_module(
                ModuleName::from_str("a.c"),
                [root.join("search_root0"), root.join("search_root1")].iter(),
                &mut vec![],
                None,
                None,
                false,
                &mut None,
            ),
            // We won't find `a.c` because when searching for package `a`, we've already
            // committed to `search_root0/a/` as the path to search next for `c`. And there's
            // no `c.py` in `search_root0/a/`.
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
            find_import_filtered(&config, ModuleName::from_str("a.c"), None, None),
            // We will find `a.c` because `a` is a namespace package whose search roots
            // include both `search_root0/a/` and `search_root1/a/`.
            FindingOrError::new_finding(ModulePath::filesystem(root.join("search_root1/a/c.py")))
        );
        assert_eq!(
            find_import_filtered(&config, ModuleName::from_str("spp_priority"), None, None),
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
            find_import_filtered(&config, ModuleName::from_str("spp_priority.d"), None, None),
            FindingOrError::new_finding(ModulePath::filesystem(
                root.join("site_package_path/spp_priority/d.py")
            )),
        );
        assert_eq!(
            find_import_filtered(
                &config,
                ModuleName::from_str("spp_priority.d"),
                None,
                Some(ModuleStyle::Interface)
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

        // pyi preferred over py
        assert_eq!(
            find_one_part(&Name::new("baz"), roots.iter(), None, &mut None),
            Some((
                FindResult::SingleFilePyModule(root.join("foo/baz.py")),
                vec![root.join("bar")]
            ))
        );
        assert_eq!(
            continue_find_module(
                FindResult::SingleFilePyiModule(root.join("foo/baz.py")),
                &[],
                None,
                &mut None,
            ),
            Some(FindResult::SingleFilePyiModule(root.join("foo/baz.py")))
        );
        assert_eq!(
            find_module_components(&Name::new("baz"), &[], roots.iter(), None, &mut None).unwrap(),
            FindResult::SingleFilePyiModule(root.join("bar/baz.pyi")),
        );

        // py preferred over pyc
        assert_eq!(
            find_one_part(&Name::new("compiled"), roots.iter(), None, &mut None),
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
                &[Name::new("a")],
                None,
                &mut None,
            )
            .unwrap(),
            FindResult::CompiledModule(root.join("foo/compiled/a.pyc"))
        );
        assert_eq!(
            find_module_components(
                &Name::new("compiled"),
                &[Name::new("a")],
                roots.iter(),
                None,
                &mut None,
            )
            .unwrap(),
            FindResult::SingleFilePyModule(root.join("bar/compiled/a.py"))
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
            )
            .unwrap(),
            FindingOrError::Finding(Finding {
                finding: ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
                error: Some(FindError::NoSource(ModuleName::from_str("foo.bar"))),
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
            )
            .unwrap(),
            FindingOrError::Finding(Finding {
                finding: _,
                error: Some(FindError::NoSource(_)),
            })
        ));
    }

    #[test]
    fn test_find_module_prefixes_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("foo.py")]);
        assert_eq!(
            find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter(),),
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
            find_module_prefixes(ModuleName::from_str(""), [root.to_path_buf()].iter(),),
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
            find_module_prefixes(ModuleName::from_str("baz.fo"), [root.to_path_buf()].iter(),),
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
            find_module_prefixes(ModuleName::from_str("baz.fo"), [root.to_path_buf()].iter(),),
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
            find_module_prefixes(ModuleName::from_str("fo"), [root.to_path_buf()].iter(),),
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
        );
        assert_eq!(
            find_compiled_result.unwrap(),
            FindingOrError::Error(FindError::Ignored)
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
            &Name::new("nested_module"),
            [root.to_path_buf()].iter(),
            None,
            &mut None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("nested_module.pyc"))
        );
        let result = find_one_part(
            &Name::new("cython_module"),
            [root.to_path_buf()].iter(),
            None,
            &mut None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("cython_module.pyx"))
        );
        let result = find_one_part(
            &Name::new("windows_dll"),
            [root.to_path_buf()].iter(),
            None,
            &mut None,
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("windows_dll.pyd"))
        );
        let result = find_one_part(
            &Name::new("another_nested_module"),
            [root.to_path_buf()].iter(),
            None,
            &mut None,
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
        let first = Name::new("subdir");
        let find_result = find_module_components(
            &first,
            &[Name::new("nested_module")],
            [root.to_path_buf()].iter(),
            None,
            &mut None,
        )
        .unwrap();
        assert_eq!(
            find_result.module_path(),
            FindingOrError::Error(FindError::Ignored)
        );
        let module_path = find_module_components(
            &first,
            &[Name::new("another_nested_module")],
            [root.to_path_buf()].iter(),
            None,
            &mut None,
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
        let components_rest = vec![Name::new("test_module")];
        assert!(continue_find_module(start_result, &components_rest, None, &mut None).is_none());
    }

    #[test]
    fn test_continue_find_module_with_pyc_no_source_ignored() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("module.pyc")]);
        let start_result = find_one_part(
            &Name::new("module"),
            [root.to_path_buf()].iter(),
            None,
            &mut None,
        )
        .unwrap()
        .0;
        assert!(matches!(
            continue_find_module(start_result, &[], None, &mut None).unwrap(),
            FindResult::CompiledModule(_)
        ));
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

        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);
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

        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);
        assert!(
            matches!(&result, FindingOrError::Error(FindError::NotFound(_, _))),
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
        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);
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
        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);

        // Should return NotFound error when using real config and typeshed third party stubs exist but package is not installed
        let error = result.error().expect("Expected error to be present");
        assert!(
            matches!(error, FindError::NotFound(_, _)),
            "Expected NotFound error with real config, got: {:?}",
            error
        );

        if let FindError::NotFound(module, _) = error {
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
        );
        assert!(
            matches!(result_synthetic, FindingOrError::Finding(_)),
            "Should find the module in typeshed third party with synthetic config, got: {:?}",
            result_synthetic
        );

        let config_marker = get_config(ConfigSource::Marker("".into()));
        let result_marker =
            find_import_filtered(&config_marker, ModuleName::from_str("requests"), None, None);
        assert!(
            matches!(result_marker, FindingOrError::Finding(_)),
            "Should find the module in typeshed third party with marker config, got: {:?}",
            result_marker
        );
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
        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);

        let FindError::NoSourceForStubs(module) = result.error().unwrap() else {
            panic!("Expected NoSourceForStubs error");
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
        let result = find_import_filtered(&config, ModuleName::from_str("requests"), None, None);

        if let FindingOrError::Finding(finding) = result {
            assert!(matches!(
                finding.finding.details(),
                ModulePathDetails::BundledTypeshedThirdParty(_)
            ));
            // Should not have a NoSource error since the package exists
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

    // ==================== Phantom Paths Tests ====================

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
}
