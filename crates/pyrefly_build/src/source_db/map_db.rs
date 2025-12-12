/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::ops::DerefMut;
use std::path::Path;

use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathBuf;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::watch_pattern::WatchPattern;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

/// A simple [`SourceDatabase`] that can be used for easy setup and testing.
#[derive(Debug, PartialEq, Eq)]
pub struct MapDatabase(SmallMap<ModuleName, Vec1<ModulePath>>, SysInfo);

impl MapDatabase {
    pub fn new(sys_info: SysInfo) -> Self {
        Self(SmallMap::new(), sys_info)
    }

    pub fn insert(&mut self, name: ModuleName, path: ModulePath) {
        match self.0.get_mut(&name) {
            Some(list) => list.push(path),
            None => {
                self.0.insert(name, Vec1::new(path));
            }
        }
    }
}

impl Deref for MapDatabase {
    type Target = SmallMap<ModuleName, Vec1<ModulePath>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MapDatabase {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl SourceDatabase for MapDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.0
            .iter()
            .flat_map(|(name, paths)| paths.iter().map(move |p| (name, p)))
            .map(|(name, path)| Handle::new(name.dupe(), path.dupe(), self.1.dupe()))
            .collect()
    }

    fn lookup(
        &self,
        module: ModuleName,
        _: Option<&Path>,
        style: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        let paths = self.0.get(&module)?;
        let style = style.unwrap_or(ModuleStyle::Interface);
        if let Some(result) = paths.iter().find(|p| p.style() == style) {
            return Some(result.dupe());
        }
        Some(paths.last().dupe())
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        let (name, _) = self
            .0
            .iter()
            .find(|(_, paths)| paths.iter().any(|p| p == module_path))?;
        Some(Handle::new(name.dupe(), module_path.dupe(), self.1.dupe()))
    }

    fn query_source_db(&self, _: SmallSet<ModulePathBuf>, _: bool) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn get_paths_to_watch(&self) -> SmallSet<WatchPattern<'_>> {
        self.0
            .values()
            .flatten()
            .map(|p| WatchPattern::file(p.as_path().to_path_buf()))
            .collect()
    }

    fn get_target(&self, _: Option<&Path>) -> Option<Target> {
        None
    }

    fn get_generated_files(&self) -> SmallSet<ModulePathBuf> {
        SmallSet::new()
    }
}
