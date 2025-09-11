/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::buck::query::PythonLibraryManifest;
use crate::buck::query::TargetManifestDatabase;
use crate::buck::query::query_source_db;
use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, PartialEq, Eq, Hash)]
enum Include {
    #[allow(unused)]
    Target(Target),
    Path(PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
pub struct BuckSourceDatabase {
    db: SmallMap<Target, PythonLibraryManifest>,
    path_lookup: SmallMap<PathBuf, Target>,
    includes: SmallSet<Include>,
}

impl BuckSourceDatabase {
    pub fn new(cwd: PathBuf, files: &SmallSet<PathBuf>) -> anyhow::Result<Self> {
        let raw_db = query_source_db(files.iter(), &cwd)?;

        Ok(Self::from_target_manifest_db(raw_db, files))
    }

    fn from_target_manifest_db(raw_db: TargetManifestDatabase, files: &SmallSet<PathBuf>) -> Self {
        let db = raw_db.produce_map();
        let mut path_lookup: SmallMap<PathBuf, Target> = SmallMap::new();
        for (target, manifest) in db.iter() {
            for source in manifest.srcs.values().flatten() {
                if let Some(old_target) = path_lookup.get_mut(&**source) {
                    let new_target = (&*old_target).min(target);
                    *old_target = new_target.dupe();
                } else {
                    path_lookup.insert(source.to_path_buf(), target.dupe());
                }
            }
        }
        let includes = files
            .into_iter()
            .map(|f| Include::Path(f.to_path_buf()))
            .collect();

        BuckSourceDatabase {
            db,
            path_lookup,
            includes,
        }
    }

    fn handles_for_include(&self, include: &Include) -> Vec<Handle> {
        match include {
            Include::Target(target) => {
                let manifest = self.db.get(target).unwrap();
                manifest
                    .srcs
                    .iter()
                    .flat_map(|(name, paths)| {
                        paths.iter().map(|p| {
                            Handle::new(
                                name.dupe(),
                                ModulePath::filesystem(p.to_path_buf()),
                                manifest.sys_info.dupe(),
                            )
                        })
                    })
                    .collect()
            }
            Include::Path(path) => {
                let target = self.path_lookup.get(path).unwrap();
                let manifest = self.db.get(target).unwrap();
                let module_name = manifest
                    .srcs
                    .iter()
                    .find(|(_, paths)| paths.contains(path))
                    .unwrap()
                    .0;
                vec![Handle::new(
                    module_name.dupe(),
                    ModulePath::filesystem(path.to_path_buf()),
                    manifest.sys_info.dupe(),
                )]
            }
        }
    }
}

impl SourceDatabase for BuckSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.includes
            .iter()
            .flat_map(|i| self.handles_for_include(i))
            .collect()
    }

    fn lookup(&self, _module: &ModuleName, _origin: Option<&Path>) -> Option<ModulePath> {
        // TODO(connernilsen): implement lookup
        None
    }

    fn handle_from_module_path(&self, module_path: ModulePath) -> Handle {
        let Some(target) = self.path_lookup.get(module_path.as_path()) else {
            return Handle::new(ModuleName::unknown(), module_path, SysInfo::default());
        };

        let manifest = self.db.get(target).unwrap();
        let module_name = manifest
            .srcs
            .iter()
            .find(|(_, paths)| paths.iter().any(|p| p == module_path.as_path()))
            .unwrap()
            .0;
        Handle::new(
            module_name.dupe(),
            module_path.dupe(),
            manifest.sys_info.dupe(),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use starlark_map::smallmap;
    use starlark_map::smallset;

    use super::*;

    fn get_db() -> (BuckSourceDatabase, PathBuf) {
        let raw_db = TargetManifestDatabase::get_test_database();
        let root = raw_db.root.to_path_buf();
        let files = smallset! {
            root.join("sub_root/pyre/client/log/log.py"),
            root.join("sub_root/pyre/client/log/log.pyi"),
        };

        (
            BuckSourceDatabase::from_target_manifest_db(raw_db, &files),
            root,
        )
    }

    #[test]
    fn test_path_lookup() {
        let (db, root) = get_db();
        let path_lookup = db.path_lookup;
        let expected = smallmap! {
            root.join("third-party/pypi/colorama/0.4.6/colorama/__init__.py") =>
                    Target::from_string("build_root//third-party/pypi/colorama/0.4.6:py".to_owned()),
                root.join("third-party/pypi/colorama/0.4.6/colorama/__init__.pyi") =>
                    Target::from_string("build_root//third-party/pypi/colorama/0.4.6:py".to_owned()),
                root.join("third-party/pypi/click/8.1.7/src/click/__init__.pyi") =>
                    Target::from_string("build_root//third-party/pypi/click/8.1.7:py".to_owned()),
                root.join("third-party/pypi/click/8.1.7/src/click/__init__.py") =>
                    Target::from_string("build_root//third-party/pypi/click/8.1.7:py".to_owned()),
                root.join("sub_root/pyre/client/log/__init__.py") =>
                    Target::from_string("sub_root//pyre/client/log:log".to_owned()),
                root.join("sub_root/pyre/client/log/log.py") =>
                    Target::from_string("sub_root//pyre/client/log:log".to_owned()),
                root.join("sub_root/pyre/client/log/log.pyi") =>
                    Target::from_string("sub_root//pyre/client/log:log".to_owned()),
        };

        assert_eq!(expected, path_lookup);
    }

    #[test]
    fn test_handles_for_include() {
        let (db, root) = get_db();

        let expected = smallmap! {
            Include::Path(root.join("sub_root/pyre/client/log/log.pyi")) => vec![
                Handle::new(
                    ModuleName::from_str("pyre.client.log.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/log.pyi")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
            ],
            Include::Path(root.join("third-party/pypi/click/8.1.7/src/click/__init__.py")) => vec![
                Handle::new(
                    ModuleName::from_str("click"),
                    ModulePath::filesystem(root.join("third-party/pypi/click/8.1.7/src/click/__init__.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
            ],
            Include::Target(Target::from_string("sub_root//pyre/client/log:log".to_owned())) => vec![
                Handle::new(
                    ModuleName::from_str("pyre.client.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/__init__.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
                Handle::new(
                    ModuleName::from_str("pyre.client.log.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/log.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
                Handle::new(
                    ModuleName::from_str("pyre.client.log.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/log.pyi")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
            ],
            Include::Target(Target::from_string("sub_root//pyre/client/log:log2".to_owned())) => vec![
                Handle::new(
                    ModuleName::from_str("log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/__init__.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
                Handle::new(
                    ModuleName::from_str("log.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/log.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
                Handle::new(
                    ModuleName::from_str("log.log"),
                    ModulePath::filesystem(root.join("sub_root/pyre/client/log/log.pyi")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
            ],
            Include::Target(Target::from_string("build_root//third-party/pypi/click/8.1.7:py".to_owned())) => vec![
                Handle::new(
                    ModuleName::from_str("click"),
                    ModulePath::filesystem(root.join("third-party/pypi/click/8.1.7/src/click/__init__.pyi")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
                Handle::new(
                    ModuleName::from_str("click"),
                    ModulePath::filesystem(root.join("third-party/pypi/click/8.1.7/src/click/__init__.py")),
                    SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                ),
            ],
        };

        for (include, expected) in expected {
            let result = SmallSet::from_iter(db.handles_for_include(&include));
            assert_eq!(result, SmallSet::from_iter(expected));
        }
    }
}
