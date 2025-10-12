/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::VecDeque;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;

use crate::handle::Handle;
use crate::query::Include;
use crate::query::PythonLibraryManifest;
use crate::query::SourceDbQuerier;
use crate::query::TargetManifestDatabase;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, PartialEq, Eq)]
struct Inner {
    /// The mapping from targets to their manifests, including sources, dependencies,
    /// and metadata. This is the source of truth.
    db: SmallMap<Target, PythonLibraryManifest>,
    /// An index for doing fast lookups of a path to its owning target.
    /// Invariants:
    /// - if a path exists in `path_lookup`, its target must exist in `db`.
    /// - if a path exists in `path_lookup`, its target's `srcs` must have a
    ///   module name with `path` as a module path.
    path_lookup: SmallMap<PathBuf, Target>,
}

impl Inner {
    fn new() -> Self {
        Inner {
            db: SmallMap::new(),
            path_lookup: SmallMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct QuerySourceDatabase {
    inner: RwLock<Inner>,
    /// The set of items the sourcedb has been queried for. Not all of the targets
    /// or files listed here will necessarily appear in the sourcedb, for example,
    /// if the given target does not exist, or if the file is not tracked by Buck.
    includes: Mutex<SmallSet<Include>>,
    /// The directory that will be passed into the sourcedb query shell-out. Should
    /// be the same as the directory containing the config this sourcedb is a part of.
    cwd: PathBuf,
    querier: Arc<dyn SourceDbQuerier>,
}

impl QuerySourceDatabase {
    pub fn new(cwd: PathBuf, querier: Arc<dyn SourceDbQuerier>) -> Self {
        QuerySourceDatabase {
            cwd,
            inner: RwLock::new(Inner::new()),
            includes: Mutex::new(SmallSet::new()),
            querier,
        }
    }

    fn update_with_target_manifest(&self, raw_db: TargetManifestDatabase) -> bool {
        let new_db = raw_db.produce_map();
        let read = self.inner.read();
        if new_db == read.db {
            debug!("No source DB changes from Buck query");
            return false;
        }
        drop(read);
        let mut write = self.inner.write();
        write.path_lookup = SmallMap::new();
        for (target, manifest) in new_db.iter() {
            for source in manifest.srcs.values().flatten() {
                if let Some(old_target) = write.path_lookup.get_mut(&**source) {
                    let new_target = (&*old_target).min(target);
                    *old_target = new_target.dupe();
                } else {
                    write.path_lookup.insert(source.clone(), target.dupe());
                }
            }
        }
        write.db = new_db;
        debug!("Finished updating source DB with Buck response");
        true
    }
}

impl SourceDatabase for QuerySourceDatabase {
    fn modules_to_check(&self) -> Vec<crate::handle::Handle> {
        // TODO(connernilsen): implement modules_to_check
        vec![]
    }

    fn lookup(
        &self,
        module: &ModuleName,
        origin: Option<&Path>,
        style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        let origin = origin?;
        let read = self.inner.read();
        let start_target = read.path_lookup.get(origin)?;
        let mut queue = VecDeque::new();
        let mut visited = SmallSet::new();
        queue.push_front(start_target);

        while let Some(current_target) = queue.pop_front() {
            if !visited.insert(current_target) {
                continue;
            }
            let Some(manifest) = read.db.get(current_target) else {
                continue;
            };

            if let Some(paths) = manifest.srcs.get(module) {
                // Since the sourcedb contains the full set of reachable files, if we find a
                // result, we know a module path matching the style filter would exist in `paths`.
                // Therefore, if it's not there, we can immediately fall back to whatever's
                // available instead of re-performing the search like we normally do in module
                // finding.
                let style = style_filter.unwrap_or(ModuleStyle::Interface);
                if let Some(result) = paths.iter().find(|p| ModuleStyle::of_path(p) == style) {
                    return Some(ModulePath::filesystem(result.to_path_buf()));
                }
                return Some(ModulePath::filesystem(paths.first().to_path_buf()));
            }

            manifest.deps.iter().for_each(|t| queue.push_back(t));
        }

        None
    }

    fn handle_from_module_path(&self, module_path: ModulePath) -> Option<Handle> {
        let read = self.inner.read();
        let target = read.path_lookup.get(&module_path.as_path().to_path_buf())?;

        // if we get a target from `path_lookup`, it must exist in manifest
        let manifest = read.db.get(target).unwrap();
        let module_name = manifest
            .srcs
            .iter()
            .find(|(_, paths)| paths.iter().any(|p| *p == module_path.as_path()))
            // similarly, if we get a target for a path in `path_lookup`, it must exist
            // in that target's srcs
            .unwrap()
            .0;
        Some(Handle::new(
            module_name.dupe(),
            module_path.dupe(),
            manifest.sys_info.dupe(),
        ))
    }

    fn requery_source_db(&self, files: SmallSet<PathBuf>) -> anyhow::Result<bool> {
        let new_includes = files.into_iter().map(Include::path).collect();
        let mut includes = self.includes.lock();
        if *includes == new_includes {
            debug!("Not querying Buck source DB, since no inputs have changed");
            return Ok(false);
        }
        *includes = new_includes;
        info!("Querying Buck for source DB");
        let raw_db = self.querier.query_source_db(&includes, &self.cwd)?;
        info!("Finished querying Buck for source DB");
        Ok(self.update_with_target_manifest(raw_db))
    }

    fn get_critical_files(&self) -> SmallSet<PathBuf> {
        let read = self.inner.read();
        read.db
            .values()
            .map(|m| m.buildfile_path.to_path_buf())
            .chain(
                read.db
                    .values()
                    .flat_map(|m| m.srcs.values().flatten().map(|p| p.to_path_buf())),
            )
            .collect()
    }

    fn get_target(&self, origin: Option<&Path>) -> Option<Target> {
        let origin = origin?;
        let read = self.inner.read();
        read.path_lookup.get(origin).copied()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use pyrefly_python::sys_info::SysInfo;
    use starlark_map::small_set::SmallSet;
    use starlark_map::smallmap;
    use starlark_map::smallset;

    use super::*;
    use crate::query::TargetManifest;

    #[derive(Debug)]
    struct DummyQuerier {}

    impl SourceDbQuerier for DummyQuerier {
        fn query_source_db(
            &self,
            _: &SmallSet<Include>,
            _: &Path,
        ) -> anyhow::Result<TargetManifestDatabase> {
            Ok(TargetManifestDatabase::get_test_database())
        }

        fn construct_command(&self) -> std::process::Command {
            panic!("We shouldn't be calling this...");
        }
    }

    impl QuerySourceDatabase {
        fn from_target_manifest_db(
            raw_db: TargetManifestDatabase,
            files: &SmallSet<PathBuf>,
        ) -> Self {
            let new = Self {
                inner: RwLock::new(Inner::new()),
                includes: Mutex::new(
                    files
                        .iter()
                        .map(|p| Include::path(p.to_path_buf()))
                        .collect(),
                ),
                cwd: PathBuf::new(),
                querier: Arc::new(DummyQuerier {}),
            };
            new.update_with_target_manifest(raw_db);
            new
        }
    }

    fn get_db() -> (QuerySourceDatabase, PathBuf) {
        let raw_db = TargetManifestDatabase::get_test_database();
        let root = raw_db.root.to_path_buf();
        let files = smallset! {
            root.join("pyre/client/log/log.py"),
            root.join("pyre/client/log/log.pyi"),
        };

        (
            QuerySourceDatabase::from_target_manifest_db(raw_db, &files),
            root,
        )
    }

    #[test]
    fn test_path_lookup_index_creation() {
        let (db, root) = get_db();
        let path_lookup = db.inner.read().path_lookup.clone();
        let expected = smallmap! {
            root.join("colorama/__init__.py") =>
                Target::from_string("//colorama:py".to_owned()),
            root.join("colorama/__init__.pyi") =>
                Target::from_string("//colorama:py".to_owned()),
            root.join("click/__init__.pyi") =>
                Target::from_string("//click:py".to_owned()),
            root.join("click/__init__.py") =>
                Target::from_string("//click:py".to_owned()),
            root.join("pyre/client/log/__init__.py") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            root.join("pyre/client/log/log.py") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            root.join("pyre/client/log/log.pyi") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
        };

        assert_eq!(expected, path_lookup);
    }

    #[test]
    fn test_sourcedb_lookup() {
        let (db, root) = get_db();

        let assert_lookup = |name, path, style_filter, expected: Option<&str>| {
            assert_eq!(
                db.lookup(
                    &ModuleName::from_str(name),
                    Some(&root.join(path)),
                    style_filter
                ),
                expected.map(|p| ModulePath::filesystem(root.join(p))),
                "got result for {name} {path} {style_filter:?}, expected {expected:?}"
            );
        };

        assert_eq!(
            db.lookup(&ModuleName::from_str("log.log"), None, None),
            None
        );
        assert_lookup("does_not_exist", "pyre/client/log/__init__.py", None, None);
        // can't be found, since the path's target only has `pyre.client.log.log` as reachable
        assert_lookup("log.log", "pyre/client/log/__init__.py", None, None);
        assert_lookup(
            "pyre.client.log.log",
            "pyre/client/log/__init__.py",
            None,
            Some("pyre/client/log/log.pyi"),
        );
        assert_lookup(
            "pyre.client.log.log",
            "pyre/client/log/__init__.py",
            Some(ModuleStyle::Interface),
            Some("pyre/client/log/log.pyi"),
        );
        assert_lookup(
            "pyre.client.log.log",
            "pyre/client/log/__init__.py",
            Some(ModuleStyle::Executable),
            Some("pyre/client/log/log.py"),
        );
        assert_lookup(
            "pyre.client.log",
            "pyre/client/log/__init__.py",
            None,
            Some("pyre/client/log/__init__.py"),
        );
        assert_lookup(
            "pyre.client.log",
            "pyre/client/log/__init__.py",
            Some(ModuleStyle::Interface),
            Some("pyre/client/log/__init__.py"),
        );
        assert_lookup(
            "pyre.client.log",
            "pyre/client/log/__init__.py",
            Some(ModuleStyle::Executable),
            Some("pyre/client/log/__init__.py"),
        );
        assert_lookup(
            "click",
            "pyre/client/log/__init__.py",
            None,
            Some("click/__init__.pyi"),
        );
        assert_lookup(
            "colorama",
            "pyre/client/log/__init__.py",
            None,
            Some("colorama/__init__.pyi"),
        );
        assert_lookup(
            "colorama",
            "click/__init__.pyi",
            None,
            Some("colorama/__init__.pyi"),
        );
        assert_lookup("log.log", "click/__init__.pyi", None, None);
        assert_lookup("log.log", "colorama/__init__.pyi", None, None);
        assert_lookup("pyre.client.log.log", "click/__init__.py", None, None);
    }

    #[test]
    fn test_handle_from_module_path() {
        let (db, root) = get_db();

        let sys_info = SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux());
        let assert_handle = |path, name| {
            let name = ModuleName::from_str(name);
            let module_path = ModulePath::filesystem(root.join(path));
            assert_eq!(
                db.handle_from_module_path(module_path.dupe()),
                Some(Handle::new(name, module_path, sys_info.dupe())),
                "got result for {path}, expected {name}",
            );
        };

        assert_eq!(
            db.handle_from_module_path(ModulePath::filesystem(root.join("does_not_exist"))),
            None,
        );
        assert_handle("pyre/client/log/__init__.py", "pyre.client.log");
        assert_handle("pyre/client/log/log.py", "pyre.client.log.log");
        assert_handle("pyre/client/log/log.pyi", "pyre.client.log.log");
        assert_handle("click/__init__.pyi", "click");
        assert_handle("click/__init__.py", "click");
        assert_handle("click/__init__.py", "click");
        assert_handle("colorama/__init__.py", "colorama");
        assert_handle("colorama/__init__.pyi", "colorama");
    }

    #[test]
    fn test_update_with_target_manifest() {
        let (db, root) = get_db();
        let manifest = TargetManifestDatabase::get_test_database();

        assert!(!db.update_with_target_manifest(manifest));

        let manifest = TargetManifestDatabase::new(
            smallmap! {
                    Target::from_string("//colorama:py".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "colorama",
                            &[
                            "colorama/__init__.py",
                            "colorama/__init__.pyi",
                            ]
                        ),
                        ],
                        &[],
                        "colorama/BUCK",
                    ),
                    Target::from_string("//pyre/client/log:log".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "pyre.client.log",
                            &[
                            "pyre/client/log/__init__.py"
                            ]
                        ),
                        (
                            "pyre.client.log.log",
                            &[
                            "pyre/client/log/log.py",
                            "pyre/client/log/log.pyi",
                            ]
                        ),
                        ],
                        &[],
                        "pyre/client/log/BUCK",
                    ),
            },
            root.clone(),
        );
        let manifest_db = manifest.clone().produce_map();
        assert!(db.update_with_target_manifest(manifest));
        let inner = db.inner.read();
        assert_eq!(inner.db, manifest_db);
        let expected_path_lookup = smallmap! {
            root.join("colorama/__init__.py") =>
                Target::from_string("//colorama:py".to_owned()),
            root.join("colorama/__init__.pyi") =>
                Target::from_string("//colorama:py".to_owned()),
            root.join("pyre/client/log/__init__.py") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            root.join("pyre/client/log/log.py") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            root.join("pyre/client/log/log.pyi") =>
                Target::from_string("//pyre/client/log:log".to_owned()),
        };
        assert_eq!(inner.path_lookup, expected_path_lookup);
    }
}
