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
use pyrefly_python::module_path::ModulePathBuf;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::fs_anyhow;
use pyrefly_util::watch_pattern::WatchPattern;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use vec1::Vec1;

use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, PartialEq, Eq, Clone)]
struct ManifestItem {
    module_name: ModuleName,
    absolute_path: ModulePath,
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

fn read_manifest_file_data(data: &[u8]) -> anyhow::Result<Vec<ManifestItem>> {
    let raw_items: Vec<Vec<String>> = serde_json::from_slice(data)?;
    let mut results = Vec::new();
    for raw_item in raw_items {
        let module_relative_path = Path::new(raw_item[0].as_str());
        match ModuleName::from_relative_path(&strip_stubs_suffix(module_relative_path)) {
            Ok(module_name) => {
                // absolutize should be fine here to get absolute path, since Pyrefly
                // will be run from Buck root.
                let absolute_path = PathBuf::from(raw_item[1].clone()).absolutize();
                if absolute_path.iter().any(|x| x == "pyre_buck_typeshed") {
                    // We sometimes get Pyre typeshed files in the manifest, which don't match the versions we expect.
                    // Once Pyre is retired, we can remove this filtering.
                    continue;
                }
                let absolute_path = ModulePath::filesystem(absolute_path);
                results.push(ManifestItem {
                    module_name,
                    absolute_path,
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
    // .pyi file always comes before .py file
    match (
        left.as_path().extension().and_then(OsStr::to_str),
        right.as_path().extension().and_then(OsStr::to_str),
    ) {
        (Some("pyi"), Some("py")) => Ordering::Less,
        (Some("py"), Some("pyi")) => Ordering::Greater,
        _ => Ordering::Equal, // Preserve original ordering by default
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
            .push(item.absolute_path);
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
    sources: SmallMap<ModuleName, Vec1<ModulePath>>,
    dependencies: SmallMap<ModuleName, Vec1<ModulePath>>,
    /// In Buck, any module that is the parent of a source/dependency is implicitly an empty `__init__.py` file.
    /// See <https://github.com/facebook/buck2/blob/03ed62f85e7cc487fd505ad097ef9f260fae2522/prelude/python/tools/wheel.py#L196C1-L198C1>.
    implicit_init: SmallMap<ModuleName, ModulePath>,
    sys_info: SysInfo,
}

impl SourceDatabase for BuckCheckSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.sources
            .iter()
            .flat_map(|(name, paths)| {
                paths
                    .iter()
                    .map(|path| Handle::new(name.dupe(), path.dupe(), self.sys_info.dupe()))
            })
            .collect()
    }

    fn lookup(
        &self,
        module: ModuleName,
        _: Option<&Path>,
        _: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        match self
            .sources
            .get(&module)
            .or_else(|| self.dependencies.get(&module))
        {
            Some(p) => Some(p.first().dupe()),
            None if let Some(x) = self.implicit_init.get(&module) => Some(x.dupe()),
            None => None,
        }
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        let find = |i: &SmallMap<ModuleName, Vec1<ModulePath>>| {
            i.iter()
                .find(|s| s.1.iter().any(|p| p == module_path))
                .map(|s| s.0.dupe())
        };
        let name = find(&self.sources).or_else(|| find(&self.dependencies))?;
        Some(Handle::new(name, module_path.dupe(), self.sys_info.dupe()))
    }

    fn query_source_db(&self, _: SmallSet<ModulePathBuf>, _: bool) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn get_paths_to_watch(&self) -> SmallSet<WatchPattern<'_>> {
        SmallSet::new()
    }

    fn get_target(&self, _: Option<&Path>) -> Option<Target> {
        None
    }

    fn get_generated_files(&self) -> SmallSet<ModulePathBuf> {
        SmallSet::new()
    }
}

impl BuckCheckSourceDatabase {
    pub fn from_manifest_files(
        source_manifests: &[PathBuf],
        dependency_manifests: &[PathBuf],
        typeshed_manifests: &[PathBuf],
        sys_info: SysInfo,
    ) -> anyhow::Result<Self> {
        let sources = read_manifest_files(source_manifests)?;
        let dependencies = read_manifest_files(dependency_manifests)?;
        let typeshed = read_manifest_files(typeshed_manifests)?;
        Ok(Self::from_manifest_items(
            sources,
            dependencies,
            typeshed,
            sys_info,
        ))
    }

    fn from_manifest_items(
        source_items: Vec<ManifestItem>,
        dependency_items: Vec<ManifestItem>,
        typeshed_items: Vec<ManifestItem>,
        sys_info: SysInfo,
    ) -> Self {
        let mut implicit_init = SmallMap::new();
        for x in source_items
            .iter()
            .chain(dependency_items.iter())
            .chain(typeshed_items.iter())
        {
            let mut name = x.module_name;
            let mut path = x.absolute_path.as_path().to_owned();
            while let Some(parent) = name.parent() {
                path.pop();
                implicit_init.insert(parent, ModulePath::namespace(path.clone()));
                name = parent;
            }
        }

        Self {
            sources: create_manifest_item_index(source_items.into_iter()),
            dependencies: create_manifest_item_index(
                dependency_items.into_iter().chain(typeshed_items),
            ),
            implicit_init,
            sys_info,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    /// Return type from `BuckCheckSourceDatabase::lookup`
    #[derive(Debug, PartialEq, Eq)]
    enum LookupResult {
        /// Source file of this module is owned by the current target.
        /// Type errors in the file should be reported to user.
        OwningSource(ModulePath),
        /// Source file of this module is owned by the dependency of the current target.
        /// The file should be analyzed but no type errors should be reported.
        ExternalSource(ModulePath),
        /// Did not find any source file associated with the given module name.
        NoSource,
    }

    impl BuckCheckSourceDatabase {
        fn lookup_for_test(&self, module: ModuleName) -> LookupResult {
            match self.sources.get(&module) {
                Some(paths) => LookupResult::OwningSource(paths.first().dupe()),
                None => match self.dependencies.get(&module) {
                    Some(paths) => LookupResult::ExternalSource(paths.first().dupe()),
                    None => LookupResult::NoSource,
                },
            }
        }
    }

    #[test]
    fn test_read_manifest() {
        assert_eq!(
            read_manifest_file_data(r#"[["foo/bar.py", "root/foo/bar.py", "derp"]]"#.as_bytes())
                .unwrap(),
            vec![ManifestItem {
                module_name: ModuleName::from_str("foo.bar"),
                absolute_path: ModulePath::filesystem(
                    PathBuf::from_str("root/foo/bar.py").unwrap().absolutize()
                )
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
                absolute_path: ModulePath::filesystem(
                    PathBuf::from_str("root/foo-stubs/bar/__init__.pyi")
                        .unwrap()
                        .absolutize()
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
                absolute_path: foo_path.dupe(),
            }],
            vec![ManifestItem {
                module_name: ModuleName::from_str("bar"),
                absolute_path: bar_path.dupe(),
            }],
            vec![ManifestItem {
                module_name: ModuleName::from_str("baz"),
                absolute_path: baz_path.dupe(),
            }],
            SysInfo::default(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            LookupResult::OwningSource(foo_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            LookupResult::ExternalSource(bar_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("baz")),
            LookupResult::ExternalSource(baz_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("qux")),
            LookupResult::NoSource
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
                    absolute_path: src_foo_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    absolute_path: src_bar_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    absolute_path: dep_foo_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    absolute_path: dep_bar_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            LookupResult::OwningSource(src_foo_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            LookupResult::OwningSource(src_bar_path)
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
                    absolute_path: foo_py_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("foo"),
                    absolute_path: foo_pyi_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    absolute_path: bar_py_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("bar"),
                    absolute_path: bar_pyi_path.dupe(),
                },
            ],
            vec![],
            SysInfo::default(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("foo")),
            LookupResult::OwningSource(foo_pyi_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("bar")),
            LookupResult::ExternalSource(bar_pyi_path)
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
                    absolute_path: dep_a_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("b"),
                    absolute_path: dep_b_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("c"),
                    absolute_path: dep_c_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("d"),
                    absolute_path: dep_d_path.dupe(),
                },
            ],
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("a"),
                    absolute_path: typeshed_a_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("b"),
                    absolute_path: typeshed_b_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("c"),
                    absolute_path: typeshed_c_path.dupe(),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("d"),
                    absolute_path: typeshed_d_path.dupe(),
                },
            ],
            SysInfo::default(),
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("a")),
            LookupResult::ExternalSource(dep_a_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("b")),
            LookupResult::ExternalSource(dep_b_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("c")),
            LookupResult::ExternalSource(typeshed_c_path)
        );
        assert_eq!(
            source_db.lookup_for_test(ModuleName::from_str("d")),
            LookupResult::ExternalSource(dep_d_path)
        );
    }

    #[test]
    fn test_load_init() {
        let source_db = BuckCheckSourceDatabase::from_manifest_items(
            vec![
                ManifestItem {
                    module_name: ModuleName::from_str("foo.bar"),
                    absolute_path: ModulePath::filesystem(
                        PathBuf::from_str("/root/foo/bar.py").unwrap(),
                    ),
                },
                ManifestItem {
                    module_name: ModuleName::from_str("foo.baz"),
                    absolute_path: ModulePath::filesystem(
                        PathBuf::from_str("/root/foo/baz.py").unwrap(),
                    ),
                },
            ],
            vec![],
            vec![],
            SysInfo::default(),
        );
        let res = source_db.lookup(ModuleName::from_str("foo"), None, None);
        assert_eq!(res.unwrap().as_path().to_str().unwrap(), "/root/foo");
    }
}
