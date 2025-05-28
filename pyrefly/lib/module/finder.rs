/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::Mutex;

use dupe::Dupe;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::state::loader::FindError;

static PY_TYPED_CACHE: LazyLock<Mutex<SmallMap<PathBuf, PyTyped>>> =
    LazyLock::new(|| Mutex::new(SmallMap::new()));

#[derive(PartialEq, Eq, PartialOrd, Ord, Default, Clone, Dupe)]
enum PyTyped {
    #[default]
    Missing,
    Complete,
    Partial,
}

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
        }
    }
}

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

fn continue_find_module(start_result: FindResult, components_rest: &[Name]) -> Option<ModulePath> {
    let mut current_result = Some(start_result);
    for part in components_rest.iter() {
        match current_result {
            None => {
                // Nothing has been found in the previous round. No point keep looking.
                break;
            }
            Some(FindResult::SingleFileModule(_)) => {
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
    current_result.map(|x| match x {
        FindResult::SingleFileModule(path) | FindResult::RegularPackage(path, _) => {
            ModulePath::filesystem(path)
        }
        FindResult::NamespacePackage(roots) => {
            // TODO(grievejia): Preserving all info in the list instead of dropping all but the first one.
            ModulePath::namespace(roots.first().clone())
        }
    })
}

pub fn find_module_in_search_path<'a>(
    module: ModuleName,
    include: impl Iterator<Item = &'a PathBuf>,
) -> Option<ModulePath> {
    let parts = module.components();
    if parts.is_empty() {
        return None;
    }
    let start_result = find_one_part(&parts[0], include);
    start_result.and_then(|start_result| continue_find_module(start_result, &parts[1..]))
}

pub fn find_module_in_site_package_path(
    module: ModuleName,
    include: &[PathBuf],
    use_untyped_imports: bool,
    ignore_missing_source: bool,
) -> Result<Option<ModulePath>, FindError> {
    let first = module.first_component();
    let mut stub_first = first.as_str().to_owned();
    stub_first.push_str("-stubs");
    let stub_first = Name::new(stub_first);

    let stub_module = ModuleName::from_parts(
        [stub_first.clone()]
            .iter()
            .chain(module.components().iter().skip(1)),
    );
    let stub_rest = &stub_module.components()[1..];

    let stub_module_imports = include
        .iter()
        .filter_map(|root| find_one_part(&stub_first, [root.to_owned()].iter()));

    let mut any_has_partial_py_typed = false;
    let mut checked_one_stub = false;
    let mut found_stubs = None;
    for stub_module_import in stub_module_imports {
        let stub_module_py_typed = stub_module_import.py_typed();
        any_has_partial_py_typed |= stub_module_py_typed == PyTyped::Partial;
        checked_one_stub = true;
        if let Some(stub_result) = continue_find_module(stub_module_import, stub_rest) {
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
        .iter()
        .filter_map(|root| find_one_part(&first, [root.to_owned()].iter()))
        .peekable();

    // check if there's an existing library backing the stubs we have
    if found_stubs.is_some() && fallback_modules.peek().is_some() {
        return Ok(found_stubs);
    } else if found_stubs.is_some() {
        return Err(FindError::no_source(module));
    }

    let module_rest = &module.components()[1..];
    let mut any_has_none_py_typed = false;
    for module in fallback_modules {
        if !use_untyped_imports
            && !any_has_partial_py_typed
            && module.py_typed() == PyTyped::Missing
        {
            any_has_none_py_typed = true;
        } else if let Some(module_result) = continue_find_module(module, module_rest) {
            return Ok(Some(module_result));
        }
    }

    if any_has_none_py_typed {
        return Err(FindError::NoPyTyped);
    }

    Ok(None)
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
            ),
            Some(ModulePath::filesystem(root.join("foo/bar.py")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
            ),
            Some(ModulePath::filesystem(root.join("foo/baz.pyi")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.qux"),
                [root.to_path_buf()].iter(),
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
            find_module_in_search_path(
                ModuleName::from_str("foo.bar"),
                [root.to_path_buf()].iter(),
            ),
            Some(ModulePath::filesystem(root.join("foo/bar/__init__.py")))
        );
        assert_eq!(
            find_module_in_search_path(
                ModuleName::from_str("foo.baz"),
                [root.to_path_buf()].iter(),
            ),
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
            ),
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
            ),
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
            find_module_in_search_path(ModuleName::from_str("a"), search_roots.iter()),
            Some(ModulePath::namespace(root.join("a")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("b"), search_roots.iter()),
            Some(ModulePath::namespace(root.join("b")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("c.d"), search_roots.iter()),
            Some(ModulePath::namespace(root.join("c/d")))
        );
        assert_eq!(
            find_module_in_search_path(ModuleName::from_str("c.d.e"), search_roots.iter()),
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
            ),
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
                false,
                false,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.baz"),
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
                false,
                false,
            )
            .is_err()
        );
        assert_eq!(
            find_module_in_site_package_path(
                ModuleName::from_str("foo.bar"),
                &[root.to_path_buf()],
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
                &[root.to_path_buf()],
                false,
                false,
            )
            .unwrap()
            .unwrap(),
            ModulePath::filesystem(root.join("baz-stubs/qux/__init__.py")),
        );
    }
}
