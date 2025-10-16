/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::path::Path;
use std::path::PathBuf;

use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

use crate::config::config::ConfigFile;
use crate::module::typeshed::typeshed;
use crate::state::loader::FindError;

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
    fn module_path(self) -> Result<ModulePath, FindError> {
        match self {
            FindResult::SingleFilePyiModule(path)
            | FindResult::SingleFilePyModule(path)
            | FindResult::RegularPackage(path, _) => Ok(ModulePath::filesystem(path)),
            FindResult::NamespacePackage(roots) => {
                // TODO(grievejia): Preserving all info in the list instead of dropping all but the first one.
                Ok(ModulePath::namespace(roots.first().clone()))
            }
            FindResult::CompiledModule(_) => Err(FindError::Ignored),
        }
    }
}

/// In the given root, attempt to find a match for the given [`Name`].
///
/// If `style_filter` is provided, only results matching that style will be returned.
/// The function will check candidates in priority order and return the first match that satisfies the filter.
fn find_one_part_in_root(
    name: &Name,
    root: &Path,
    style_filter: Option<ModuleStyle>,
) -> Option<FindResult> {
    let candidate_dir = root.join(name.as_str());
    // First check if `name` corresponds to a regular package.
    for candidate_init_suffix in ["__init__.pyi", "__init__.py"] {
        let init_path = candidate_dir.join(candidate_init_suffix);
        if init_path.exists() {
            // Note: do not filter by style filter here since __init__.pyi could potentially have .py files covered under it
            // todo(connernilsen): do we filter here?
            return Some(FindResult::RegularPackage(init_path, candidate_dir));
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
        }
    }
    // Finally check if `name` corresponds to a namespace package.
    if candidate_dir.is_dir() {
        let result = FindResult::NamespacePackage(Vec1::new(candidate_dir));
        // Namespace packages don't have a style in the same sense, so we return them regardless of filter
        return Some(result);
    }
    None
}

/// Finds the first package (regular, single file, or namespace) in all search roots.
/// Returns None if no module is found. If `name` is `__pycache__`, we always
/// return `None`.
///
/// If `style_filter` is provided, only results matching that style will be returned.
/// The function will continue searching until it finds a result that matches the style.
fn find_one_part<'a>(
    name: &Name,
    mut roots: impl Iterator<Item = &'a PathBuf>,
    style_filter: Option<ModuleStyle>,
) -> Option<(FindResult, Vec<PathBuf>)> {
    // skip looking in `__pycache__`, since those modules are not accessible
    if name == &Name::new_static("__pycache__") {
        return None;
    }
    let mut namespace_roots = Vec::new();
    while let Some(root) = roots.next() {
        match find_one_part_in_root(name, root, style_filter) {
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
fn continue_find_module(
    start_result: FindResult,
    components_rest: &[Name],
    style_filter: Option<ModuleStyle>,
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
                current_result = find_one_part(part, [next_root].iter(), style_filter).map(|x| x.0);
            }
            Some(FindResult::NamespacePackage(next_roots)) => {
                current_result = find_one_part(part, next_roots.iter(), style_filter).map(|x| x.0);
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
fn find_module_components<'a, I>(
    first: &Name,
    components_rest: &[Name],
    include: I,
    style_filter: Option<ModuleStyle>,
) -> Option<FindResult>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let (first_component_result, fallback_search) =
        find_one_part(first, include.clone(), style_filter)?;

    let current_result =
        continue_find_module(first_component_result, components_rest, style_filter)?;

    match current_result {
        FindResult::SingleFilePyiModule(_) | FindResult::RegularPackage(..) => Some(current_result),
        _ => Some(
            fallback_search
                .into_iter()
                .filter_map(|s| Some(find_one_part(first, [s].iter(), style_filter)?.0))
                .filter_map(|first| {
                    continue_find_module(first.clone(), components_rest, style_filter)
                })
                .fold(current_result, FindResult::best_result),
        ),
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
fn find_module<'a, I>(
    module: ModuleName,
    include: I,
    namespaces_found: &mut Vec<PathBuf>,
    ignore_missing_source: bool,
    style_filter: Option<ModuleStyle>,
) -> Result<Option<ModulePath>, FindError>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    match module.components().as_slice() {
        [] => Ok(None),
        [first, rest @ ..] => {
            // First try finding the module in `-stubs`.
            let stub_first = Name::new(format!("{first}-stubs"));
            let stub_result =
                find_module_components(&stub_first, rest, include.clone(), style_filter);
            if ignore_missing_source && let Some(stub_result) = stub_result {
                if let FindResult::NamespacePackage(namespaces) = stub_result {
                    namespaces_found.append(&mut namespaces.into_vec());
                    return Ok(None);
                } else {
                    return Ok(Some(stub_result.module_path()?));
                }
            }
            //
            // If we couldn't find it in a `-stubs` module or we want to check for missing stubs, look normally.
            let normal_result = find_module_components(first, rest, include, style_filter);

            match (normal_result, stub_result) {
                (None, Some(_)) if !ignore_missing_source => Err(FindError::NoSource(module)),
                (Some(_), Some(stub_result)) => Ok(Some(stub_result.module_path()?)),
                (Some(FindResult::NamespacePackage(namespaces)), _) => {
                    namespaces_found.append(&mut namespaces.into_vec());
                    Ok(None)
                }
                (Some(normal_result), _) => Ok(Some(normal_result.module_path()?)),
                (None, _) => Ok(None),
            }
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
        let mut current_result = find_one_part(first, include, None).map(|x| x.0);
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
                            find_one_part(part, iter::once(&next_root), None).map(|x| x.0);
                    }
                }
                Some(FindResult::NamespacePackage(next_roots)) => {
                    if is_last {
                        results = find_one_part_prefix(part, next_roots.iter());
                        break;
                    } else {
                        current_result = find_one_part(part, next_roots.iter(), None).map(|x| x.0);
                    }
                }
            }
        }
    }
    results.iter().map(|(_, name)| *name).collect::<Vec<_>>()
}

// TODO(connernilsen): refactor/clean this up so that we can have filtering and module priority
// stuff up at this level too.
pub fn find_import_filtered(
    config: &ConfigFile,
    module: ModuleName,
    origin: Option<&ModulePath>,
    style_filter: Option<ModuleStyle>,
) -> Result<ModulePath, FindError> {
    let mut namespaces_found = vec![];
    let origin = origin.map(|p| p.as_path());
    if module != ModuleName::builtins() && config.replace_imports_with_any(origin, module) {
        Err(FindError::Ignored)
    } else if let Some(sourcedb) = config.source_db.as_ref()
        && let Some(path) = sourcedb.lookup(&module, origin, style_filter)
    {
        Ok(path.clone())
    } else if let Some(path) = find_module(
        module,
        config.search_path(),
        &mut namespaces_found,
        true,
        style_filter,
    )? {
        Ok(path)
    } else if let Some(custom_typeshed_path) = &config.typeshed_path
        && let Some(path) = find_module(
            module,
            std::iter::once(&custom_typeshed_path.join("stdlib")),
            &mut namespaces_found,
            true,
            style_filter,
        )?
    {
        Ok(path)
    } else if matches!(style_filter, Some(ModuleStyle::Interface) | None)
        && let Some(path) = typeshed()
            .map_err(|err| FindError::not_found(err, module))?
            .find(module)
    {
        Ok(path)
    } else if !config.disable_search_path_heuristics
        && let Some(path) = find_module(
            module,
            config.fallback_search_path.iter(),
            &mut namespaces_found,
            true,
            style_filter,
        )?
    {
        Ok(path)
    } else if let Some(path) = find_module(
        module,
        config.site_package_path(),
        &mut namespaces_found,
        config.ignore_missing_source,
        style_filter,
    )? {
        Ok(path)
    } else if let Some(namespace) = namespaces_found.into_iter().next() {
        Ok(ModulePath::namespace(namespace))
    } else if config.ignore_missing_imports(origin, module) {
        Err(FindError::Ignored)
    } else {
        Err(FindError::import_lookup_path(
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
) -> Result<ModulePath, FindError> {
    find_import_filtered(config, module, origin, None)
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

    results
}

#[cfg(test)]
mod tests {
    use pyrefly_config::environment::environment::PythonEnvironment;
    use pyrefly_config::environment::interpreters::Interpreters;
    use pyrefly_util::test_path::TestPath;

    use super::*;

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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar.py")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/baz.pyi")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap(),
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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/baz/__init__.pyi")))
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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar.pyi")))
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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
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
            assert!(
                find_module(
                    ModuleName::from_str(name),
                    search_roots.iter(),
                    &mut namespaces,
                    true,
                    None
                )
                .unwrap()
                .is_none()
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
                true,
                None
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("first/c/d/e.py")))
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
                true,
                None,
            )
            .unwrap(),
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
                        TestPath::dir("spp_priority", vec![]),
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
                        vec![TestPath::file("__init__.py")],
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
            find_import_filtered(&config, ModuleName::from_str("a.c"), None, None).unwrap(),
            // We will find `a.c` because `a` is a namespace package whose search roots
            // include both `search_root0/a/` and `search_root1/a/`.
            ModulePath::filesystem(root.join("search_root1/a/c.py"))
        );
        assert_eq!(
            find_import_filtered(&config, ModuleName::from_str("spp_priority"), None, None)
                .unwrap(),
            // We will find `spp_priority` in `site_package_path`, even though it's
            // in a later module find component, because we continue searching for
            // a better option when we find a namespace package
            ModulePath::filesystem(root.join("site_package_path/spp_priority/__init__.py"))
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
            find_one_part(&Name::new("baz"), roots.iter(), None),
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
            ),
            Some(FindResult::SingleFilePyiModule(root.join("foo/baz.py")))
        );
        assert_eq!(
            find_module_components(&Name::new("baz"), &[], roots.iter(), None).unwrap(),
            FindResult::SingleFilePyiModule(root.join("bar/baz.pyi")),
        );

        // py preferred over pyc
        assert_eq!(
            find_one_part(&Name::new("compiled"), roots.iter(), None),
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
            )
            .unwrap(),
            FindResult::CompiledModule(root.join("foo/compiled/a.pyc"))
        );
        assert_eq!(
            find_module_components(
                &Name::new("compiled"),
                &[Name::new("a")],
                roots.iter(),
                None
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
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi")),
        );
        assert!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .is_none()
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
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py"))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))
        );
        assert!(matches!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            ),
            Ok(None)
        ));
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
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py")),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))
        );
        assert!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .is_none()
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
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py")),
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi")),
        );
        assert!(
            find_module(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .unwrap()
            .is_none()
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
        assert!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .is_err()
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
        );
        assert!(
            find_module(
                ModuleName::from_str("baz.qux"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                false,
                None,
            )
            .is_err()
        );
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
        assert!(
            find_module(
                ModuleName::from_str("namespace"),
                [root.to_path_buf()].iter(),
                &mut namespaces,
                true,
                None,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(namespaces, vec![root.join("namespace")]);
        assert_eq!(
            find_module(
                ModuleName::from_str("namespace.a"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("namespace/a/__init__.py"))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("namespace.b"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("namespace/b/__init__.py"))
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
            true,
            None,
        );
        assert!(matches!(find_compiled_result, Err(FindError::Ignored)));
        assert_eq!(
            find_module(
                ModuleName::from_str("compiled_module.nested"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                None,
            )
            .unwrap(),
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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo.py")))
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
                true,
                None,
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("subdir/nested_import.py")))
        );
        let find_compiled_result = find_module(
            ModuleName::from_str("subdir.another_compiled_module"),
            [root.to_path_buf()].iter(),
            &mut vec![],
            true,
            None,
        );
        assert!(matches!(find_compiled_result, Err(FindError::Ignored)));
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
        )
        .unwrap()
        .0;
        assert_eq!(
            result,
            FindResult::CompiledModule(root.join("cython_module.pyx"))
        );
        let result = find_one_part(&Name::new("windows_dll"), [root.to_path_buf()].iter(), None)
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
        )
        .unwrap();
        assert!(matches!(find_result.module_path(), Err(FindError::Ignored)));
        let module_path = find_module_components(
            &first,
            &[Name::new("another_nested_module")],
            [root.to_path_buf()].iter(),
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
        let components_rest = vec![Name::new("test_module")];
        assert!(continue_find_module(start_result, &components_rest, None).is_none());
    }

    #[test]
    fn test_continue_find_module_with_pyc_no_source_ignored() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("module.pyc")]);
        let start_result = find_one_part(&Name::new("module"), [root.to_path_buf()].iter(), None)
            .unwrap()
            .0;
        assert!(matches!(
            continue_find_module(start_result, &[], None).unwrap(),
            FindResult::CompiledModule(_)
        ));
    }

    #[test]
    fn test_find_module_filter_basic() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::file("bar.py"), TestPath::file("bar.pyi")],
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Executable),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("bar.py")))
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Interface),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("bar.pyi")))
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

        assert!(matches!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Executable),
            ),
            Err(FindError::Ignored)
        ));
        assert_eq!(
            find_module(
                ModuleName::from_str("bar"),
                [root.to_path_buf()].iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Interface),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("bar.pyi")))
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
                true,
                Some(ModuleStyle::Executable),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("baz").join("bar.py")))
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
                true,
                Some(ModuleStyle::Executable),
            )
            .unwrap(),
            Some(ModulePath::filesystem(
                root.join("search_root2/standalone.py")
            ))
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("standalone"),
                search_roots.iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Interface),
            )
            .unwrap(),
            Some(ModulePath::filesystem(
                root.join("search_root1/standalone.pyi")
            ))
        );

        assert_eq!(
            find_module(
                ModuleName::from_str("standalone2"),
                search_roots.iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Interface),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            find_module(
                ModuleName::from_str("standalone2"),
                search_roots.iter(),
                &mut vec![],
                true,
                Some(ModuleStyle::Executable),
            )
            .unwrap(),
            Some(ModulePath::filesystem(
                root.join("search_root2/standalone2.py")
            ))
        );
    }
}
