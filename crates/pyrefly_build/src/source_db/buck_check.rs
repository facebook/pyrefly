/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::fs_anyhow;
use regex::Regex;
use starlark_map::small_map::SmallMap;
use tracing::debug;
use vec1::Vec1;

use crate::handle::Handle;
use crate::source_db::LiveSourceDatabase;
use crate::source_db::ModuleEnumerator;
use crate::source_db::SourceDatabase;

#[derive(Debug, PartialEq, Eq, Clone)]
struct ManifestItem {
    module_name: ModuleName,
    module_path: ModulePath,
}

fn strip_stubs_suffix(path: &Path) -> PathBuf {
    path.components()
        .map(|component| {
            if let Some(component_str) = component.as_os_str().to_str()
                && let Some(stripped) = component_str.strip_suffix("-stubs")
            {
                Path::new(stripped)
            } else {
                Path::new(component.as_os_str())
            }
        })
        .collect()
}

fn path_is_from_stubs_package(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|component| component.ends_with("-stubs"))
    })
}

fn read_manifest_file_data(data: &[u8]) -> anyhow::Result<Vec<ManifestItem>> {
    let raw_items: Vec<Vec<String>> = serde_json::from_slice(data)?;
    let mut results = Vec::new();
    for raw_item in raw_items {
        let module_relative_path = Path::new(raw_item[0].as_str());
        match ModuleName::from_relative_path(&strip_stubs_suffix(module_relative_path)) {
            Ok(module_name) => {
                // We deliberately stick with relative paths, as sometimes we are run on RE,
                // so the absolute path on RE will not match the users absolute path.
                let path = PathBuf::from(raw_item[1].clone());
                if path
                    .iter()
                    .any(|x| x == "pyre_buck_typeshed" || x == "__flattened__")
                {
                    // Filter out Pyre typeshed files from the manifest. These don't
                    // match the versions Pyrefly expects and cause spurious errors.
                    // This catches both the default target (pyre_buck_typeshed:flattened)
                    // and user overrides (e.g. tools/pyre/stubs/typeshed/typeshed:flattened).
                    // Once Pyre is retired, we can remove this filtering.
                    continue;
                }
                results.push(ManifestItem {
                    module_name,
                    module_path: ModulePath::filesystem(path),
                });
            }
            Err(error) => {
                // This often happens for buckified 3rd-party targets
                debug!("Cannot convert path to module name: {error:#}");
            }
        }
    }
    Ok(results)
}

fn read_manifest_file(path: &Path) -> anyhow::Result<Vec<ManifestItem>> {
    let data = fs_anyhow::read(path)?;
    read_manifest_file_data(&data)
        .with_context(|| format!("failed to parse manifest JSON `{}`", path.display()))
}

fn read_manifest_files(manifest_paths: &[PathBuf]) -> anyhow::Result<Vec<ManifestItem>> {
    let mut result = Vec::new();
    for manifest_path in manifest_paths {
        let manifest_items = read_manifest_file(manifest_path.as_path())?;
        result.extend(manifest_items);
    }
    Ok(result)
}

fn same_module_path_compare(left: &ModulePath, right: &ModulePath) -> Ordering {
    match (
        left.as_path().extension().and_then(OsStr::to_str),
        right.as_path().extension().and_then(OsStr::to_str),
    ) {
        (Some("pyi"), Some("py")) => Ordering::Less,
        (Some("py"), Some("pyi")) => Ordering::Greater,
        _ => match (
            path_is_from_stubs_package(left.as_path()),
            path_is_from_stubs_package(right.as_path()),
        ) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        },
    }
}

fn create_manifest_item_index(
    items: impl Iterator<Item = ManifestItem>,
) -> SmallMap<ModuleName, Vec1<ModulePath>> {
    let mut accumulated: SmallMap<ModuleName, Vec<ModulePath>> =
        SmallMap::with_capacity(items.size_hint().0);
    for item in items {
        accumulated
            .entry(item.module_name)
            .or_default()
            .push(item.module_path);
    }
    accumulated
        .into_iter()
        .map(|(name, mut paths)| {
            paths.sort_by(same_module_path_compare);
            (name, Vec1::try_from_vec(paths).unwrap())
        })
        .collect()
}

#[derive(Debug)]
pub struct BuckCheckSourceDatabase {
    lookup: SmallMap<ModuleName, ModulePath>,
    modules_to_check: Vec<Handle>,
    path_to_handle: SmallMap<ModulePath, Handle>,
}

impl SourceDatabase for BuckCheckSourceDatabase {
    fn may_contain_module(&self, module: ModuleName) -> bool {
        self.lookup.contains_key(&module)
    }

    fn lookup(
        &self,
        module: ModuleName,
        _: Option<&Path>,
        _: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        self.lookup.get(&module).map(|path| path.dupe())
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        self.path_to_handle
            .get(module_path)
            .map(|handle| handle.dupe())
    }

    fn as_live_source_database(&self) -> Option<&dyn LiveSourceDatabase> {
        None
    }
}

impl ModuleEnumerator for BuckCheckSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.modules_to_check.clone()
    }
}

impl BuckCheckSourceDatabase {
    pub fn from_manifest_files(
        source_manifests: &[PathBuf],
        dependency_manifests: &[PathBuf],
        typeshed_manifests: &[PathBuf],
        sys_info: SysInfo,
        check_dependencies: bool,
        skip_dependency_modules: Vec<Regex>,
    ) -> anyhow::Result<Self> {
        let sources = read_manifest_files(source_manifests)?;
        let dependencies = read_manifest_files(dependency_manifests)?;
        let typeshed = read_manifest_files(typeshed_manifests)?;
        Ok(Self::from_manifest_items(
            sources,
            dependencies,
            typeshed,
            sys_info,
            check_dependencies,
            skip_dependency_modules,
        ))
    }

    fn from_manifest_items(
        source_items: Vec<ManifestItem>,
        dependency_items: Vec<ManifestItem>,
        typeshed_items: Vec<ManifestItem>,
        sys_info: SysInfo,
        check_dependencies: bool,
        skip_dependency_modules: Vec<Regex>,
    ) -> Self {
        let mut implicit_init = SmallMap::new();
        for x in source_items
            .iter()
            .chain(dependency_items.iter())
            .chain(typeshed_items.iter())
        {
            let mut name = x.module_name;
            let mut path = x.module_path.as_path().to_owned();
            while let Some(parent) = name.parent() {
                path.pop();
                implicit_init.insert(parent, ModulePath::namespace(path.clone()));
                name = parent;
            }
        }

        let sources = create_manifest_item_index(source_items.into_iter());
        let dependencies =
            create_manifest_item_index(dependency_items.into_iter().chain(typeshed_items));

        let mut lookup = dependencies
            .iter()
            .map(|(name, paths)| (name.dupe(), paths.first().dupe()))
            .collect::<SmallMap<_, _>>();
        lookup.extend(
            sources
                .iter()
                .map(|(name, paths)| (name.dupe(), paths.first().dupe())),
        );
        for (name, path) in implicit_init {
            lookup.entry(name).or_insert(path);
        }

        let handle = |name: &ModuleName, path: &ModulePath| {
            Handle::new(name.dupe(), path.dupe(), sys_info.dupe())
        };
        let sources_to_check = sources
            .iter()
            .flat_map(|(name, paths)| paths.iter().map(|path| handle(name, path)));
        let modules_to_check = if check_dependencies {
            let dependencies_to_check = dependencies
                .iter()
                .filter(|(name, _)| {
                    !skip_dependency_modules
                        .iter()
                        .any(|re| re.is_match(name.as_str()))
                })
                .flat_map(|(name, paths)| paths.iter().map(|path| handle(name, path)));
            sources_to_check.chain(dependencies_to_check).collect()
        } else {
            sources_to_check.collect()
        };
        let path_to_handle = dependencies
            .iter()
            .chain(sources.iter())
            .flat_map(|(name, paths)| paths.iter().map(|path| (path.dupe(), handle(name, path))))
            .collect();

        Self {
            lookup,
            modules_to_check,
            path_to_handle,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    impl BuckCheckSourceDatabase {
        fn lookup_for_test(&self, module: ModuleName) -> Option<ModulePath> {
            self.lookup(module, None, None)
        }

        fn paths_to_check_for_test(&self) -> Vec<ModulePath> {
            self.modules_to_check()
                .into_iter()
                .map(|handle| handle.path().dupe())
                .collect()
        }
    }

    #[test]
    fn test_read_manifest() {
        assert_eq!(
            read_manifest_file_data(r#"[["foo/bar.py", "root/foo/bar.py", "derp"]]"#.as_bytes())
                .unwrap(),
            vec![ManifestItem {
                module_name: ModuleName::from_str("foo.bar"),
                module_path: ModulePath::filesystem(PathBuf::from_str("root/foo/bar.py").unwrap())
            }]
        );
        assert_eq!(
            read_manifest_file_data(
                r#"[["foo-stubs/bar/__init__.pyi", "root/foo-stubs/bar/__init__.pyi", "derp"]]"#
                    .as_bytes()
            )
            .unwrap(),
            vec![ManifestItem {
                module_name: ModuleName::from_str("foo.bar"),
                module_path: ModulePath::filesystem(
                    PathBuf::from_str("root/foo-stubs/bar/__init__.pyi").unwrap()
                )
            }]
        );
        assert_eq!(
            read_manifest_file_data(
                r#"[["foo/bar.derp", "root/foo/bar.derp", "derp"]]"#.as_bytes()
            )
            .unwrap(),
            vec![]
        )
    }

    #[test]
    fn test_load_simple() {
        let foo_path = ModulePath::filesystem(PathBuf::from_str("/root/foo.py").unwrap());
        let bar_path = ModulePath::filesystem(PathBuf::from_str("/root/bar.py").unwrap());
        let baz_path = ModulePath::filesystem(PathBuf::from_str("/root/baz.py").unwrap());
        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![ManifestItem {
                module_name: ModuleName::from_str("foo"),
                module_path: foo_path.dupe(),
            }],
            vec![ManifestItem {
                module_name: ModuleName::from_str("bar"),
                module_path: bar_path.dupe(),
            }],
            vec![ManifestItem {
                module_name: ModuleName::from_str("baz"),
                module_path: baz_path.dupe(),
            }],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            Some(foo_path.dupe())
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            Some(bar_path.dupe())
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("baz")),
            Some(baz_path)
        );
        assert_eq!(source_db.lookup_for_test(ModuleName::from_str("qux")), None);
        assert_eq!(source_db.paths_to_check_for_test(), vec![foo_path]);
        assert!(source_db.may_contain_module(ModuleName::from_str("foo")));
        assert!(source_db.may_contain_module(ModuleName::from_str("bar")));
        assert!(source_db.may_contain_module(ModuleName::from_str("baz")));
        assert!(!source_db.may_contain_module(ModuleName::from_str("qux")));
    }

    #[test]
    fn test_load_prefers_stub_package_for_same_module() {
        let runtime_path =
            ModulePath::filesystem(PathBuf::from_str("/root/foo/__init__.pyi").unwrap());
        let stub_path =
            ModulePath::filesystem(PathBuf::from_str("/root/foo-stubs/__init__.pyi").unwrap());
        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: runtime_path,
                },
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: stub_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            Some(stub_path)
        );
    }

    #[test]
    fn test_load_source_over_dependencies() {
        let src_foo_path = ModulePath::filesystem(PathBuf::from_str("/src/foo.py").unwrap());
        let dep_foo_path = ModulePath::filesystem(PathBuf::from_str("/dep/foo.py").unwrap());

        let src_bar_path = ModulePath::filesystem(PathBuf::from_str("/src/bar.py").unwrap());
        let dep_bar_path = ModulePath::filesystem(PathBuf::from_str("/dep/bar.pyi").unwrap());

        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: src_foo_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    module_path: src_bar_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: dep_foo_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    module_path: dep_bar_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            Some(src_foo_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            Some(src_bar_path)
        );
    }

    #[test]
    fn test_load_pyi_over_py() {
        let foo_py_path = ModulePath::filesystem(PathBuf::from_str("/root/foo.py").unwrap());
        let foo_pyi_path = ModulePath::filesystem(PathBuf::from_str("/root/foo.pyi").unwrap());
        let bar_py_path = ModulePath::filesystem(PathBuf::from_str("/root/bar.py").unwrap());
        let bar_pyi_path = ModulePath::filesystem(PathBuf::from_str("/root/bar.pyi").unwrap());

        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: foo_py_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    module_path: foo_pyi_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    module_path: bar_py_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    module_path: bar_pyi_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            Some(foo_pyi_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            Some(bar_pyi_path)
        );
    }

    #[test]
    fn test_load_dependency_typeshed_conflict() {
        let dep_a_path = ModulePath::filesystem(PathBuf::from_str("/dep/a.py").unwrap());
        let dep_b_path = ModulePath::filesystem(PathBuf::from_str("/dep/b.pyi").unwrap());
        let dep_c_path = ModulePath::filesystem(PathBuf::from_str("/dep/c.py").unwrap());
        let dep_d_path = ModulePath::filesystem(PathBuf::from_str("/dep/d.pyi").unwrap());

        let typeshed_a_path = ModulePath::filesystem(PathBuf::from_str("/typeshed/a.py").unwrap());
        let typeshed_b_path = ModulePath::filesystem(PathBuf::from_str("/typeshed/b.py").unwrap());
        let typeshed_c_path = ModulePath::filesystem(PathBuf::from_str("/typeshed/c.pyi").unwrap());
        let typeshed_d_path = ModulePath::filesystem(PathBuf::from_str("/typeshed/d.pyi").unwrap());

        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("a"),
                    module_path: dep_a_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("b"),
                    module_path: dep_b_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("c"),
                    module_path: dep_c_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("d"),
                    module_path: dep_d_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("a"),
                    module_path: typeshed_a_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("b"),
                    module_path: typeshed_b_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("c"),
                    module_path: typeshed_c_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("d"),
                    module_path: typeshed_d_path.dupe(),
                },
            ],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("a")),
            Some(dep_a_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("b")),
            Some(dep_b_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("c")),
            Some(typeshed_c_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("d")),
            Some(dep_d_path)
        );
    }

    #[test]
    fn test_check_dependencies_with_skip_patterns() {
        let foo_path = ModulePath::filesystem(PathBuf::from_str("/src/foo.py").unwrap());
        let bar_path = ModulePath::filesystem(PathBuf::from_str("/dep/bar.py").unwrap());
        let baz_path = ModulePath::filesystem(PathBuf::from_str("/dep/baz.py").unwrap());

        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![ManifestItem {
                module_name: ModuleName::from_str("foo"),
                module_path: foo_path.dupe(),
            }],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    module_path: bar_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("baz"),
                    module_path: baz_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
            true,
            vec![Regex::new("^bar$").unwrap()],
        );

        assert_eq!(
            source_db.paths_to_check_for_test(),
            vec![foo_path, baz_path]
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            Some(bar_path)
        );
    }

    #[test]
    fn test_load_init() {
        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo.bar"),
                    module_path: ModulePath::filesystem(
                        PathBuf::from_str("/root/foo/bar.py").unwrap(),
                    ),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("foo.baz"),
                    module_path: ModulePath::filesystem(
                        PathBuf::from_str("/root/foo/baz.py").unwrap(),
                    ),
                },
            ],
            vec![],
            vec![],
            SysInfo::default(),
            false,
            Vec::new(),
        );
        let res = source_db.lookup(ModuleName::from_str("foo"), None, None);
        assert_eq!(res.unwrap().as_path().to_str().unwrap(), "/root/foo");
    }
}
