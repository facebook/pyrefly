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
use crate::buck::query::query_source_db;
use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, PartialEq, Eq, Hash)]
enum Include {
    #[expect(unused)]
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
        Ok(BuckSourceDatabase {
            db,
            path_lookup,
            includes,
        })
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

    fn handle_from_module_path(&self, _module_path: ModulePath) -> Handle {
        // TODO(connernilsen): implement handles_from_module_path
        Handle::new(
            ModuleName::unknown(),
            ModulePath::memory(PathBuf::new()),
            SysInfo::default(),
        )
    }
}
