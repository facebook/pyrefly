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

impl Include {
    #[expect(unused)]
    fn to_targets(&self, db: &SmallMap<Target, PythonLibraryManifest>) -> Vec<Target> {
        match &self {
            Self::Target(target) => vec![target.dupe()],
            Self::Path(path) => db
                .iter()
                .filter(|(_, manifest)| manifest.srcs.values().flatten().any(|p| p == path))
                .map(|(t, _)| t.dupe())
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct BuckSourceDatabase {
    db: SmallMap<Target, PythonLibraryManifest>,
    includes: SmallSet<Include>,
}

impl BuckSourceDatabase {
    pub fn new(cwd: PathBuf, files: &SmallSet<PathBuf>) -> anyhow::Result<Self> {
        let raw_db = query_source_db(files.iter(), &cwd)?;
        let db = raw_db.produce_map();
        let includes = files
            .into_iter()
            .map(|f| Include::Path(f.to_path_buf()))
            .collect();
        Ok(BuckSourceDatabase { db, includes })
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
