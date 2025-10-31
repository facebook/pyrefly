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
use pyrefly_python::module_path::ModulePathBuf;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;
use vec1::Vec1;

use crate::handle::Handle;
use crate::query::Include;
use crate::query::PythonLibraryManifest;
use crate::query::SourceDbQuerier;
use crate::query::TargetManifestDatabase;
use crate::source_db::ModulePathCache;
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
    path_lookup: SmallMap<ModulePathBuf, Target>,
    /// An index for doing fast lookups of a package to possible owning targets.
    /// We keep this, since it's possible an init file is defined in one target, but not
    /// a dependency or dependent target in the same directory. We also need to
    /// perform a search through all synthesized packages, since the import we're looking
    /// for could be in any relevant target there as well. This will happen in directories
    /// defining multiple Python targets, but that don't have any init files.
    ///
    /// The key in this map will always point to an `__init__` file if *any*
    /// target pointing to the "regular package" contains a real `__init__` file on
    /// disk. Otherwise, the key will point to a synthesized package's directory.
    package_lookup: SmallMap<ModulePathBuf, SmallSet<Target>>,
}

impl Inner {
    fn new() -> Self {
        Inner {
            db: SmallMap::new(),
            path_lookup: SmallMap::new(),
            package_lookup: SmallMap::new(),
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
    cached_modules: ModulePathCache,
}

impl QuerySourceDatabase {
    pub fn new(cwd: PathBuf, querier: Arc<dyn SourceDbQuerier>) -> Self {
        QuerySourceDatabase {
            cwd,
            inner: RwLock::new(Inner::new()),
            includes: Mutex::new(SmallSet::new()),
            querier,
            cached_modules: ModulePathCache::new(),
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
        let mut path_lookup: SmallMap<ModulePathBuf, Target> = SmallMap::new();
        let mut package_lookup: SmallMap<ModulePathBuf, SmallSet<Target>> = SmallMap::new();
        for (target, manifest) in new_db.iter() {
            for source in manifest.srcs.values().flatten() {
                if let Some(old_target) = path_lookup.get_mut(source) {
                    let new_target = (&*old_target).min(target);
                    *old_target = new_target.dupe();
                } else {
                    path_lookup.insert(source.dupe(), target.dupe());
                }
            }
            for paths in manifest.packages.values() {
                for path in paths {
                    package_lookup
                        .entry(path.dupe())
                        .or_default()
                        .insert(target.dupe());
                }
            }
        }
        let mut write = self.inner.write();
        write.db = new_db;
        write.path_lookup = path_lookup;
        write.package_lookup = package_lookup;
        debug!("Finished updating source DB with Buck response");
        true
    }

    /// Attempts to search in the given [`PythonLibraryManifest`] for the import,
    /// checking `srcs` and `package_lookup`.
    /// Returns the module path, if one can be found as [`Ok`], otherwise,
    /// if a synthesized dunder init is found, returns that value as [`Err`].
    fn find_in_manifest<'a>(
        &'a self,
        module: ModuleName,
        manifest: &'a PythonLibraryManifest,
        style_filter: Option<ModuleStyle>,
    ) -> Option<Result<ModulePath, ModulePath>> {
        let try_get_filtered = |paths: &Vec1<ModulePathBuf>| {
            // Since the sourcedb contains the full set of reachable files, if we find a
            // result, we know a module path matching the style filter would exist in `paths`.
            // Therefore, if it's not there, we can immediately fall back to whatever's
            // available instead of re-performing the search like we normally do in module
            // finding.
            let style = style_filter.unwrap_or(ModuleStyle::Interface);
            if let Some(result) = paths.iter().find(|p| ModuleStyle::of_path(p) == style) {
                return self.cached_modules.get(result);
            }
            self.cached_modules.get(paths.first())
        };
        if let Some(paths) = manifest.srcs.get(&module) {
            return Some(Ok(try_get_filtered(paths)));
        }

        // Take the first dunder init package we see, since it will have an `__init__.py` file
        // output on disk during actual build system building.
        manifest
            .packages
            .get(&module)
            .map(|p| Err(try_get_filtered(p)))
    }

    /// Perform a lookup, starting from the given target, and searching through all of
    /// its dependencies. The first found result (file, regular package,
    /// or implicit package) is returned.
    fn lookup_from_target<'a>(
        &'a self,
        read: &'a Inner,
        module: ModuleName,
        target: Target,
        style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        let mut queue = VecDeque::new();
        let mut visited = SmallSet::new();
        queue.push_front(target);

        while let Some(current_target) = queue.pop_front() {
            if !visited.insert(current_target) {
                continue;
            }
            let Some(manifest) = read.db.get(&current_target) else {
                continue;
            };

            match self.find_in_manifest(module, manifest, style_filter) {
                // Return the value immediately. It's safe to return
                // a implicit package here instead of continuing to search
                // because during build system building, an __init__.py file will
                // be output.
                Some(Ok(result) | Err(result)) => return Some(result),
                _ => (),
            }

            manifest.deps.iter().for_each(|t| queue.push_back(t.dupe()));
        }

        None
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
        let origin = ModulePathBuf::from_path(origin?);
        let read = self.inner.read();
        if let Some(start_target) = read.path_lookup.get(&origin)
            && let Some(result) =
                self.lookup_from_target(&read, *module, *start_target, style_filter)
        {
            return Some(result);
        } else if let Some(package_targets) = read.package_lookup.get(&origin) {
            let mut package_matches = SmallSet::new();
            // Gather all of the implicit package targets and deterministically return
            // one of them. There will either be:
            // 1. no results, in which case we return `None`
            // 2. a result that's a src file included in a package's manifest, in which
            //    case we can return immediately. The build system will complain if
            //    multiple reachable targets have files that write to the same output
            //    location, so we don't need to check for/handle that.
            // 3. a result that's a package file, which might refer to mulitple directories.
            //    In this case, we collect all possible results, and deterministically
            //    return one of them.
            for target in package_targets {
                let Some(manifest) = read.db.get(target) else {
                    continue;
                };
                // We only do a search with depth 1, since an import originating from an
                // implicit package will be defined in that package's target.
                // If we've reached this (top level) 'else if' block and found a match in
                // `init_lookup`, then our origin must be some kind of package.
                // There are two cases here:
                // 1. The package is pointing to an `__init__.py` file, in which case,
                //    we would have checked the target and its dependencies in the
                //    previous (top level) `if` block. That would find anything from
                //    another target the actual `__init__.py` file is attempting to
                //    import, in which case the user is required to specify the dependency
                //    on the target owning the init file by virtue of using a build system.
                // 2. The package is pointing to an `__init__.py` file or synthesized package
                //    directory. In this case, the dependencies don't really matter, because
                //    what we're looking for is a relative import within *this* target
                //    or another target that has a package at the origin's location.
                match self.find_in_manifest(*module, manifest, style_filter) {
                    Some(Ok(result)) => return Some(result),
                    Some(Err(implicit_package)) => {
                        package_matches.insert(implicit_package);
                    }
                    _ => (),
                }
            }
            return package_matches.into_iter().min();
        }

        None
    }

    fn handle_from_module_path(&self, module_path: ModulePath) -> Option<Handle> {
        let read = self.inner.read();
        let target = read.path_lookup.get(&module_path.module_path_buf())?;

        // if we get a target from `path_lookup`, it must exist in manifest
        let manifest = read.db.get(target).unwrap();
        let module_name = manifest
            .srcs
            .iter()
            .find(|(_, paths)| paths.iter().any(|p| *p == module_path.module_path_buf()))
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

    fn requery_source_db(&self, files: SmallSet<ModulePathBuf>) -> anyhow::Result<bool> {
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
        let origin = ModulePathBuf::from_path(origin?);
        let read = self.inner.read();
        read.path_lookup.get(&origin).copied()
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
                        .map(|p| Include::path(ModulePathBuf::new(p.to_path_buf())))
                        .collect(),
                ),
                cwd: PathBuf::new(),
                querier: Arc::new(DummyQuerier {}),
                cached_modules: ModulePathCache::new(),
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
            ModulePathBuf::new(root.join("colorama/__init__.py")) =>
                Target::from_string("//colorama:py".to_owned()),
            ModulePathBuf::new(root.join("colorama/__init__.pyi")) =>
                Target::from_string("//colorama:py".to_owned()),
            ModulePathBuf::new(root.join("click/__init__.pyi")) =>
                Target::from_string("//click:py".to_owned()),
            ModulePathBuf::new(root.join("click/__init__.py")) =>
                Target::from_string("//click:py".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/__init__.py")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/log.py")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/log.pyi")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/format.py")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("implicit_package/test/main.py")) =>
                Target::from_string("//implicit_package/test:main".to_owned()),
            ModulePathBuf::new(root.join("implicit_package/test/lib/utils.py")) =>
                Target::from_string("//implicit_package/test:lib".to_owned()),
            ModulePathBuf::new(root.join("implicit_package/package_boundary_violation.py")) =>
                Target::from_string("//implicit_package/test:lib".to_owned()),
            ModulePathBuf::new(root.join("implicit_package/test/deeply/nested/package/file.py")) =>
                Target::from_string("//implicit_package/test:lib".to_owned()),
        };

        assert_eq!(expected, path_lookup);
    }

    #[test]
    fn test_implicit_package_lookup_index_creation() {
        let (db, root) = get_db();
        let path_lookup = db.inner.read().package_lookup.clone();
        let expected = smallmap! {
            ModulePathBuf::new(root.join("click/__init__.py")) => smallset! {
                Target::from_string("//click:py".to_owned()),
            },
            ModulePathBuf::new(root.join("click/__init__.pyi")) => smallset! {
                Target::from_string("//click:py".to_owned()),
            },
            ModulePathBuf::new(root.join("colorama/__init__.py")) => smallset! {
                Target::from_string("//colorama:py".to_owned()),
            },
            ModulePathBuf::new(root.join("colorama/__init__.pyi")) => smallset! {
                Target::from_string("//colorama:py".to_owned()),
            },
            ModulePathBuf::new(root.join("pyre/client/log/__init__.py")) => smallset! {
                Target::from_string("//pyre/client/log:log".to_owned()),
                Target::from_string("//pyre/client/log:log2".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test")) => smallset! {
                Target::from_string("//implicit_package/test:main".to_owned()),
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test/lib")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test/deeply/nested/package")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test/deeply/nested")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test/deeply")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
        };
        assert_eq!(path_lookup, expected);
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
        // can be found, since we do an `init_lookup` and can find a result.
        assert_lookup(
            "log.log",
            "pyre/client/log/__init__.py",
            None,
            Some("pyre/client/log/log.pyi"),
        );
        assert_lookup(
            "pyre.client.log.log",
            "pyre/client/log/__init__.py",
            None,
            Some("pyre/client/log/log.pyi"),
        );
        // can't be found, since we're not doing an `init_lookup`, so this should only
        // be looked up through `pyre.client.log.format`.
        assert_lookup("log.format", "pyre/client/log/log.pyi", None, None);
        assert_lookup(
            "pyre.client.log.format",
            "pyre/client/log/log.pyi",
            None,
            Some("pyre/client/log/format.py"),
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
        // test implicit_package lookups
        assert_lookup(
            "implicit_package.lib.utils",
            "implicit_package/test/main.py",
            None,
            Some("implicit_package/test/lib/utils.py"),
        );
        assert_lookup(
            "implicit_package.lib",
            "implicit_package/test/main.py",
            None,
            Some("implicit_package/test/lib"),
        );
        assert_lookup(
            "implicit_package.lib",
            "implicit_package/test/lib/utils.py",
            None,
            Some("implicit_package/test/lib"),
        );
        assert_lookup(
            "implicit_package",
            "implicit_package/test/main.py",
            None,
            Some("implicit_package/test"),
        );
        assert_lookup(
            "implicit_package",
            "implicit_package/test/lib/utils.py",
            None,
            Some("implicit_package/test"),
        );
        assert_lookup(
            "implicit_package.main",
            "implicit_package/test",
            None,
            Some("implicit_package/test/main.py"),
        );
        assert_lookup(
            "implicit_package.lib",
            "implicit_package/test",
            None,
            Some("implicit_package/test/lib"),
        );
        assert_lookup(
            "implicit_package",
            "implicit_package/test/lib/utils.py",
            None,
            Some("implicit_package/test"),
        );
        assert_lookup(
            "implicit_package",
            "implicit_package/test/lib",
            None,
            Some("implicit_package/test"),
        );
        assert_lookup(
            "implicit_package.main",
            "implicit_package/test/lib/utils.py",
            None,
            None,
        );
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
                        &[],
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
                        &[],
                    ),
                    Target::from_string("//implicit_package/test:lib".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "implicit_package.lib.utils", &[
                                "implicit_package/test/lib/utils.py"
                            ]
                        )
                        ],
                        &[],
                        "implicit_package/test/BUCK",
                        &[
                        ("implicit_package", &["implicit_package/test"]),
                        ("implicit_package/lib", &["implicit_package/test/lib"]),
                        ],
                    ),
            },
            root.clone(),
        );
        let manifest_db = manifest.clone().produce_map();
        assert!(db.update_with_target_manifest(manifest));
        let inner = db.inner.read();
        assert_eq!(inner.db, manifest_db);
        let expected_path_lookup = smallmap! {
            ModulePathBuf::new(root.join("colorama/__init__.py")) =>
                Target::from_string("//colorama:py".to_owned()),
            ModulePathBuf::new(root.join("colorama/__init__.pyi")) =>
                Target::from_string("//colorama:py".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/__init__.py")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/log.py")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("pyre/client/log/log.pyi")) =>
                Target::from_string("//pyre/client/log:log".to_owned()),
            ModulePathBuf::new(root.join("implicit_package/test/lib/utils.py")) =>
                Target::from_string("//implicit_package/test:lib".to_owned()),
        };
        assert_eq!(inner.path_lookup, expected_path_lookup);
        let expected_package_lookup = smallmap! {
            ModulePathBuf::new(root.join("implicit_package/test")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("implicit_package/test/lib")) => smallset! {
                Target::from_string("//implicit_package/test:lib".to_owned()),
            },
            ModulePathBuf::new(root.join("colorama/__init__.py")) => smallset! {
                Target::from_string("//colorama:py".to_owned()),
            },
            ModulePathBuf::new(root.join("colorama/__init__.pyi")) => smallset! {
                Target::from_string("//colorama:py".to_owned()),
            },
            ModulePathBuf::new(root.join("pyre/client/log/__init__.py")) => smallset! {
                Target::from_string("//pyre/client/log:log".to_owned()),
            },
        };
        assert_eq!(inner.package_lookup, expected_package_lookup);
    }
}
