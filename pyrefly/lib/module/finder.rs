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
use std::sync::Mutex;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

use crate::state::loader::FindError;

static PY_TYPED_CACHE: LazyLock<Mutex<SmallMap<PathBuf, PyTyped>>> =
    LazyLock::new(|| Mutex::new(SmallMap::new()));

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Default, Clone, Dupe)]
enum PyTyped {
    #[default]
    Missing,
    Complete,
    Partial,
    Hidden,
}

#[derive(Debug, PartialEq)]
enum FindResult {
    /// Found a single-file module. The path must not point to an __init__ file.
    SingleFileModule(PathBuf),
    /// Found a regular package. First path must point to an __init__ file.
    /// Second path indicates where to continue search next. It should always point to the parent of the __init__ file.
    RegularPackage(PathBuf, PathBuf),
    /// Found a namespace package.
    /// The path component indicates where to continue search next. It may contain more than one directories as the namespace package
    /// may span across multiple search roots.
    NamespacePackage(Vec1<PathBuf>),
    /// Found a compiled Python file (.pyc). Represents a module compiled to bytecode, lacking source and type info.
    /// Treated as `typing.Any` to handle imports without type errors.
    CompiledModule(PathBuf),
}

impl FindResult {
    fn py_typed(&self) -> PyTyped {
        /// Finds a `py.typed` file for the given path, if it exists, and
        /// returns a boolean representing if it is partial or not.
        ///
        /// If we get an error on reading the `py.typed`, treat it as partial,
        /// since that's the most permissive behavior.
        fn py_typed_cached(candidate_path: &Path) -> PyTyped {
            fn get_py_typed(candidate_path: &Path) -> PyTyped {
                let py_typed = candidate_path.join("py.typed");
                if py_typed.exists() {
                    if std::fs::read_to_string(py_typed)
                        .ok()
                        // if we fail to read it (ok() returns None), then treat as partial
                        .is_none_or(|contents| contents.trim() == "partial")
                    {
                        return PyTyped::Partial;
                    } else {
                        return PyTyped::Complete;
                    }
                }
                PyTyped::Missing
            }
            PY_TYPED_CACHE
                .lock()
                .unwrap()
                .entry(candidate_path.to_path_buf())
                .or_insert_with(|| get_py_typed(candidate_path))
                .dupe()
        }
        match self {
            Self::SingleFileModule(candidate_path) | Self::RegularPackage(_, candidate_path) => {
                py_typed_cached(candidate_path)
            }
            Self::NamespacePackage(paths) => paths
                .iter()
                .map(|path| py_typed_cached(path))
                .max()
                .unwrap_or_default(),
            Self::CompiledModule(_) => PyTyped::Hidden,
        }
    }
}

/// Finds the first package (regular, single file, or namespace) in search roots. Returns None if no module is found.
/// name: module name
/// roots: search roots
fn find_one_part<'a>(name: &Name, roots: impl Iterator<Item = &'a PathBuf>) -> Option<FindResult> {
    let mut namespace_roots = Vec::new();
    for root in roots {
        let candidate_dir = root.join(name.as_str());
        // First check if `name` corresponds to a regular package.
        for candidate_init_suffix in ["__init__.pyi", "__init__.py"] {
            let init_path = candidate_dir.join(candidate_init_suffix);
            if init_path.exists() {
                return Some(FindResult::RegularPackage(init_path, candidate_dir));
            }
        }
        // Second check if `name` corresponds to a single-file module.
        for candidate_file_suffix in ["pyi", "py"] {
            let candidate_path = root.join(format!("{name}.{candidate_file_suffix}"));
            if candidate_path.exists() {
                return Some(FindResult::SingleFileModule(candidate_path));
            }
        }
        // Check if `name` corresponds to a compiled module.
        let candidate_pyc_path = root.join(format!("{name}.pyc"));
        if candidate_pyc_path.exists() {
            return Some(FindResult::CompiledModule(candidate_pyc_path));
        }
        // Finally check if `name` corresponds to a namespace package.
        if candidate_dir.is_dir() {
            namespace_roots.push(candidate_dir);
        }
    }
    match Vec1::try_from_vec(namespace_roots) {
        Err(_) => None,
        Ok(namespace_roots) => Some(FindResult::NamespacePackage(namespace_roots)),
    }
}

/// Finds all packages (regular, single file, or namespace) in search roots where the name starts with the given prefix.
/// prefix: module name prefix
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

                            if path.is_dir()
                                && !results.iter().any(|r| match r {
                                    (FindResult::RegularPackage(_, p), _) => p == &path,
                                    _ => false,
                                })
                            {
                                namespace_roots
                                    .entry(ModuleName::from_str(name))
                                    .or_default()
                                    .push(path.clone());
                            }
                        } else if let Some((stem, ext)) = name.rsplit_once('.')
                            && ["pyi", "py"].contains(&ext)
                            && !["__init__", "__main__"].contains(&stem)
                            && path.is_file()
                        {
                            results.push((
                                FindResult::SingleFileModule(path.clone()),
                                ModuleName::from_str(stem),
                            ));
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
fn continue_find_module(
    start_result: FindResult,
    components_rest: &[Name],
) -> Result<Option<ModulePath>, FindError> {
    let mut current_result = Some(start_result);
    for part in components_rest.iter() {
        match current_result {
            None => {
                // Nothing has been found in the previous round. No point keep looking.
                break;
            }
            Some(FindResult::SingleFileModule(_)) | Some(FindResult::CompiledModule(_)) => {
                // We've already reached leaf nodes. Cannot keep searching
                current_result = None;
                break;
            }
            Some(FindResult::RegularPackage(_, next_root)) => {
                current_result = find_one_part(part, [next_root].iter());
            }
            Some(FindResult::NamespacePackage(next_roots)) => {
                current_result = find_one_part(part, next_roots.iter());
            }
        }
    }
    Ok(current_result.map(|x| match x {
        FindResult::SingleFileModule(path) |
        FindResult::RegularPackage(path, _) |
        // TODO(melvinhe): Address CompiledModule to return an ignored Error, for Typing.Any treatment
        FindResult::CompiledModule(path) => {
            ModulePath::filesystem(path)
        }
        FindResult::NamespacePackage(roots) => {
            // TODO(grievejia): Preserving all info in the list instead of dropping all but the first one.
            ModulePath::namespace(roots.first().clone())
        }
    }))
}

pub fn find_module_in_search_path<'a, I>(
    module: ModuleName,
    include: I,
) -> Result<Option<ModulePath>, FindError>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    match module.components().as_slice() {
        [] => Ok(None),
        [first, rest @ ..] => {
            let start_result = find_one_part(first, include.clone());
            let result = start_result
                .map(|start_result| continue_find_module(start_result, rest))
                .transpose()?
                .flatten();
            if result.is_some() {
                return Ok(result);
            }
            // If we can't find the module, try to find a stub by adding the -stubs suffix
            let stub_first = Name::new(format!("{first}-stubs"));
            let start_stub_result = match find_one_part(&stub_first, include) {
                Some(result) => result,
                None => return Ok(None),
            };
            continue_find_module(start_stub_result, rest)
        }
    }
}

pub fn find_module_in_site_package_path<'a, I>(
    module: ModuleName,
    include: I,
    use_untyped_imports: bool,
    ignore_missing_source: bool,
) -> Result<Option<ModulePath>, FindError>
where
    I: Iterator<Item = &'a PathBuf> + Clone,
{
    let components = module.components();
    let first = &components[0];
    let rest = &components[1..];
    let stub_first = Name::new(format!("{first}-stubs"));

    let stub_module_imports = include
        .clone()
        .filter_map(|root| find_one_part(&stub_first, iter::once(root)));

    let mut any_has_partial_py_typed = false;
    let mut checked_one_stub = false;
    let mut found_stubs = None;
    for stub_module_import in stub_module_imports {
        let stub_module_py_typed = stub_module_import.py_typed();
        any_has_partial_py_typed |= stub_module_py_typed == PyTyped::Partial;
        checked_one_stub = true;
        if let Some(stub_result) = continue_find_module(stub_module_import, rest)? {
            found_stubs = Some(stub_result);
            break;
        }
    }

    if found_stubs.is_some() {
        if ignore_missing_source {
            return Ok(found_stubs);
        }
    } else if !use_untyped_imports && checked_one_stub && !any_has_partial_py_typed {
        // return none and stop the search if no stubs declared partial, but we searched at least one module
        return Ok(None);
    }

    let mut fallback_modules = include
        .clone()
        .filter_map(|root| find_one_part(first, iter::once(root)))
        .peekable();

    // check if there's an existing library backing the stubs we have
    if found_stubs.is_some() && fallback_modules.peek().is_some() {
        return Ok(found_stubs);
    } else if found_stubs.is_some() {
        return Err(FindError::no_source(module));
    }

    let mut any_has_none_py_typed = false;
    for module in fallback_modules {
        if !use_untyped_imports
            && !any_has_partial_py_typed
            && module.py_typed() == PyTyped::Missing
        {
            any_has_none_py_typed = true;
        } else if let Some(module_result) = continue_find_module(module, rest)? {
            return Ok(Some(module_result));
        }
    }

    if any_has_none_py_typed {
        return Err(FindError::NoPyTyped);
    }

    Ok(None)
}

pub fn find_module_prefixes<'a>(
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
        let mut current_result = find_one_part(first, include);
        for (i, part) in rest.iter().enumerate() {
            let is_last = i == rest.len() - 1;
            match current_result {
                None => {
                    break;
                }
                Some(FindResult::SingleFileModule(_) | FindResult::CompiledModule(_)) => {
                    break;
                }
                Some(FindResult::RegularPackage(_, next_root)) => {
                    if is_last {
                        results = find_one_part_prefix(part, iter::once(&next_root));
                        break;
                    } else {
                        current_result = find_one_part(part, iter::once(&next_root));
                    }
                }
                Some(FindResult::NamespacePackage(next_roots)) => {
                    if is_last {
                        results = find_one_part_prefix(part, next_roots.iter());
                        break;
                    } else {
                        current_result = find_one_part(part, next_roots.iter());
                    }
                }
            }
        }
    }
    results.iter().map(|(_, name)| *name).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
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
            find_module_in_search_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar.py")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/baz.pyi")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
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
            find_module_in_search_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
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
            find_module_in_search_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
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
            find_module_in_search_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
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
                TestPath::dir("a", vec![]),
                TestPath::dir("b", vec![TestPath::dir("c", vec![])]),
                TestPath::dir("c", vec![TestPath::dir("d", vec![TestPath::file("e.py")])]),
            ],
        );
        let search_roots = [root.to_path_buf()];
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("a"), search_roots.iter()).unwrap(),
            Some(ModulePath::namespace(root.join("a")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("b"), search_roots.iter()).unwrap(),
            Some(ModulePath::namespace(root.join("b")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("c.d"), search_roots.iter()).unwrap(),
            Some(ModulePath::namespace(root.join("c/d")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("c.d.e"), search_roots.iter()).unwrap(),
            Some(ModulePath::filesystem(root.join("c/d/e.py")))
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
            find_module_in_search_path(
                ModuleName::from_str("a.c"),
                [root.join("search_root0"), root.join("search_root1")].iter(),
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
                    vec![TestPath::dir("a", vec![TestPath::file("b.py")])],
                ),
                TestPath::dir(
                    "search_root1",
                    vec![TestPath::dir("a", vec![TestPath::file("c.py")])],
                ),
            ],
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("a.c"),
                [root.join("search_root0"), root.join("search_root1")].iter(),
            )
            .unwrap(),
            // We will find `a.c` because `a` is a namespace package whose search roots
            // include both `search_root0/a/` and `search_root1/a/`.
            Some(ModulePath::filesystem(root.join("search_root1/a/c.py")))
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
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                true,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi")),
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                false,
                false,
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
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                true,
                false
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py"))
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                true,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .is_err()
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
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py")),
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi"))
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                false,
                false,
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
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/bar/__init__.py")),
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo/baz/__init__.pyi")),
        );
        assert!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
                false,
                false,
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
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
                false,
                true,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("foo-stubs/bar/__init__.py")),
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("baz.qux"),
                [root.to_path_buf()].iter(),
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("baz-stubs/qux/__init__.py")),
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
    fn test_find_compiled_module() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("compiled_module.pyc")]);
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("compiled_module"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("compiled_module.pyc")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("compiled_module.nested"),
                [root.to_path_buf()].iter(),
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
            find_module_in_search_path(ModuleName::from_str("foo"), [root.to_path_buf()].iter(),)
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
            find_module_in_search_path(
                ModuleName::from_str("subdir.nested_import"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(root.join("subdir/nested_import.py")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("subdir.another_compiled_module"),
                [root.to_path_buf()].iter(),
            )
            .unwrap(),
            Some(ModulePath::filesystem(
                root.join("subdir/another_compiled_module.pyc")
            ))
        );
    }

    #[test]
    fn test_pyc_file_treated_as_hidden() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "subdir",
                vec![TestPath::file("compiled_module.pyc")],
            )],
        );
        let find_result =
            find_one_part(&Name::new("compiled_module"), [root.join("subdir")].iter()).unwrap();
        assert_eq!(find_result.py_typed(), PyTyped::Hidden);
    }

    #[test]
    fn test_non_pyc_file_not_hidden() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "subdir",
                vec![TestPath::file("non_pyc_module.py")],
            )],
        );
        let find_result =
            find_one_part(&Name::new("non_pyc_module"), [root.join("subdir")].iter()).unwrap();
        assert_ne!(find_result.py_typed(), PyTyped::Hidden);
    }

    #[test]
    fn test_find_one_part_with_pyc() {
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
        let result = find_one_part(&Name::new("nested_module"), [root.join("subdir")].iter());
        assert_eq!(
            result,
            Some(FindResult::CompiledModule(
                root.join("subdir/nested_module.pyc")
            ))
        );
        let result = find_one_part(
            &Name::new("another_nested_module"),
            [root.join("subdir")].iter(),
        );
        assert_eq!(
            result,
            Some(FindResult::SingleFileModule(
                root.join("subdir/another_nested_module.py")
            ))
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
        let start_result =
            find_one_part(&Name::new("subdir"), [root.to_path_buf()].iter()).unwrap();
        let module_path =
            continue_find_module(start_result, &[Name::new("nested_module")]).unwrap();
        assert_eq!(
            module_path,
            Some(ModulePath::filesystem(
                root.join("subdir/nested_module.pyc")
            ))
        );
        let start_result =
            find_one_part(&Name::new("subdir"), [root.to_path_buf()].iter()).unwrap();
        let module_path =
            continue_find_module(start_result, &[Name::new("another_nested_module")]).unwrap();
        assert_eq!(
            module_path,
            Some(ModulePath::filesystem(
                root.join("subdir/another_nested_module.py")
            ))
        );
    }

    #[test]
    fn test_continue_find_module_signature() {
        let start_result =
            FindResult::RegularPackage(PathBuf::from("path/to/init.py"), PathBuf::from("path/to"));
        let components_rest = vec![Name::new("test_module")];
        let result: Result<Option<ModulePath>, FindError> =
            continue_find_module(start_result, &components_rest);
        let unwrapped_result = result.unwrap();
        assert_eq!(unwrapped_result, None);
    }

    #[test]
    fn test_find_module_in_search_path_signature() {
        let module = ModuleName::from_str("test_module");
        let include: Vec<PathBuf> = vec![];
        let result: Result<Option<ModulePath>, FindError> =
            find_module_in_search_path(module, include.iter());
        let unwrapped_result = result.unwrap();
        assert_eq!(unwrapped_result, None);
    }
}
