/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use serde::Deserialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub(crate) enum TargetManifest {
    Library {
        deps: SmallSet<Target>,
        srcs: SmallMap<ModuleName, Vec1<PathBuf>>,
        python_version: PythonVersion,
        python_platform: PythonPlatform,
    },
    Alias {
        alias: Target,
    },
}

impl TargetManifest {
    #[expect(unused)]
    pub fn new_library(
        srcs: SmallMap<ModuleName, Vec1<PathBuf>>,
        deps: SmallSet<Target>,
        sys_info: SysInfo,
    ) -> Self {
        Self::Library {
            srcs,
            deps,
            python_version: sys_info.version().dupe(),
            python_platform: sys_info.platform().clone(),
        }
    }

    #[expect(unused)]
    pub fn new_alias(alias: Target) -> Self {
        Self::Alias { alias }
    }

    pub(crate) fn iter_srcs<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (&'a ModuleName, &'a Vec1<PathBuf>)> + 'a> {
        match &self {
            Self::Library { srcs, .. } => Box::new(srcs.iter()),
            Self::Alias { .. } => Box::new(std::iter::empty::<(&ModuleName, &Vec1<PathBuf>)>()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
pub(crate) struct TargetManifestDatabase {
    db: SmallMap<Target, TargetManifest>,
    root: PathBuf,
}

impl Deref for TargetManifestDatabase {
    type Target = SmallMap<Target, TargetManifest>;

    fn deref(&self) -> &Self::Target {
        &self.db
    }
}

impl TargetManifestDatabase {
    #[expect(unused)]
    pub fn new(db: SmallMap<Target, TargetManifest>, root: PathBuf) -> Self {
        Self { db, root }
    }

    #[allow(unused)]
    pub(crate) fn iter_srcs<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a Target, &'a ModuleName, Vec1<PathBuf>)> + 'a {
        let root = &self.root;
        self.db.iter().flat_map(move |(target, manifest)| {
            manifest
                .iter_srcs()
                .map(move |(module, paths)| (target, module, paths.mapped_ref(|p| root.join(p))))
        })
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
pub struct BuckSourceDatabase {
    cwd: PathBuf,
    db: Option<TargetManifestDatabase>,
}

impl BuckSourceDatabase {
    pub fn new(config_path: PathBuf) -> Self {
        BuckSourceDatabase {
            cwd: config_path,
            db: None,
        }
    }
}

impl SourceDatabase for BuckSourceDatabase {
    fn modules_to_check(&self) -> Vec<crate::handle::Handle> {
        // TODO(connernilsen): implement modules_to_check
        vec![]
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
