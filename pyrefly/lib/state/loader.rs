/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::locked_map::LockedMap;
use vec1::Vec1;
use vec1::vec1;

use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::FallbackSearchPath;
use crate::config::config::ImportLookupPathPart;
use crate::error::context::ErrorContext;
use crate::module::finder::DirEntryCache;
use crate::module::finder::find_import;
use crate::module::finder::find_import_filtered;
use crate::module::finder::suggest_stdlib_import;
use crate::state::state::TransactionTimingCounters;

#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub enum FindError {
    /// This module could not be found, and we should emit an error
    MissingImport(ModuleName, Arc<Vec1<String>>),
    /// This import could not be found, but the user configured it to be ignored
    Ignored,
    /// We found stubs, but no source files were found. This means it's likely stubs
    /// are installed for a project, but the library is not actually importable
    MissingSource(ModuleName),
    /// We have the source files, but do not have the stubs. In this case we should send
    /// a message to the user which will allow them to install the stubs for the package.
    /// The string will hold the name of the pip package that we will tell the user to install.
    UntypedImport(ModuleName, Arc<String>),
    /// This is the condition where we are using stubs but we do not have the source files
    MissingSourceForStubs(ModuleName),
    /// The module resolved successfully, but only via the directory-upward fallback
    /// search path (not through any configured absolute root). This is a non-fatal
    /// caveat on a successful resolution: the module is still usable, but the import
    /// is fragile (breaks when the importing file moves, can shadow installed packages).
    ImplicitRelativeImport(ModuleName),
}

impl FindError {
    pub fn missing_import(err: anyhow::Error, module: ModuleName) -> Self {
        Self::MissingImport(module, Arc::new(vec1![format!("{err:#}")]))
    }

    pub fn import_lookup_path(
        path: Vec<ImportLookupPathPart>,
        module: ModuleName,
        config_source: &ConfigSource,
    ) -> FindError {
        let config_suffix = match config_source {
            ConfigSource::File(p) => format!(" (from config in `{}`)", p.display()),
            ConfigSource::PythonToolMarker(p) | ConfigSource::Marker(p) => {
                format!(
                    " (from default config for project root marked by `{}`)",
                    p.display()
                )
            }
            ConfigSource::FailedParse(p) => {
                format!(
                    " (from default config for `{}` which failed to parse)",
                    p.display()
                )
            }
            _ => "".to_owned(),
        };
        let nonempty_paths = path
            .iter()
            .filter_map(|path| {
                if path.is_empty() {
                    None
                } else {
                    Some(format!("{path}"))
                }
            })
            .collect::<Vec<_>>();
        let mut explanation = vec1![if nonempty_paths.is_empty() {
            format!("No search path or site package path{config_suffix}")
        } else {
            format!("Looked in these locations{config_suffix}:")
        }];
        explanation.extend(nonempty_paths);
        FindError::MissingImport(module, Arc::new(explanation))
    }

    pub fn display(&self) -> (Option<Box<dyn Fn() -> ErrorContext + '_>>, Vec1<String>) {
        match self {
            Self::MissingImport(module, err) => {
                let mut lines = (**err).clone();
                // Compute suggestion lazily at display time, using global cache
                if let Some(suggested) = suggest_stdlib_import(*module) {
                    lines.insert(0, format!("Did you mean `{suggested}`?"));
                }
                (
                    Some(Box::new(|| ErrorContext::ImportNotFound(*module))),
                    lines,
                )
            }
            Self::Ignored => (None, vec1!["Ignored import".to_owned()]),
            Self::MissingSource(module) => (
                None,
                vec1![format!(
                    "Found stubs for `{module}`, but no source. This means it's likely not \
                    installed/unimportable."
                )],
            ),
            Self::MissingSourceForStubs(module) => (
                None,
                vec1![format!(
                    "Stubs for `{module}` are bundled with Pyrefly but the source files for the package are not found."
                )],
            ),
            Self::UntypedImport(source_package, stubs_package) => (
                Some(Box::new(|| ErrorContext::ImportNotTyped(*source_package))),
                vec1![format!("Hint: install the `{stubs_package}` package")],
            ),
            Self::ImplicitRelativeImport(module) => (
                None,
                vec1![format!(
                    "Module `{module}` was imported using an implicit relative import. \
                    Prefer an explicit relative import (`from . import {module}`) or add the \
                    module's root to the configured search path."
                )],
            ),
        }
    }

    pub fn kind(&self) -> Option<ErrorKind> {
        match self {
            Self::MissingImport(..) => Some(ErrorKind::MissingImport),
            Self::MissingSource(..) => Some(ErrorKind::MissingSource),
            Self::MissingSourceForStubs(..) => Some(ErrorKind::MissingSourceForStubs),
            Self::UntypedImport(..) => Some(ErrorKind::UntypedImport),
            Self::ImplicitRelativeImport(..) => Some(ErrorKind::ImplicitRelativeImport),
            Self::Ignored => None,
        }
    }
}

#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub struct Finding<T> {
    pub finding: T,
    pub error: Option<FindError>,
}

/// Result of an attempt to find a module
#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub enum FindingOrError<T> {
    /// Information about a found module. May have a non-fatal error attached.
    Finding(Finding<T>),
    /// A fatal error that prevented us from finding a module.
    Error(FindError),
}

impl<T> FindingOrError<T> {
    pub fn new_finding(finding: T) -> Self {
        Self::Finding(Finding {
            finding,
            error: None,
        })
    }

    pub fn finding(self) -> Option<T> {
        match self {
            Self::Finding(finding) => Some(finding.finding),
            Self::Error(_) => None,
        }
    }

    pub fn error(self) -> Option<FindError> {
        match self {
            Self::Finding(Finding { error, finding: _ }) => error,
            Self::Error(error) => Some(error),
        }
    }

    pub fn map<T2>(self, f: impl FnOnce(T) -> T2) -> FindingOrError<T2> {
        match self {
            Self::Finding(Finding { finding, error }) => FindingOrError::Finding(Finding {
                finding: f(finding),
                error,
            }),
            Self::Error(e) => FindingOrError::Error(e),
        }
    }

    pub fn with_error(self, error: FindError) -> Self {
        match self {
            Self::Finding(x) if x.error.is_none() => Self::Finding(Finding {
                finding: x.finding,
                error: Some(error),
            }),
            x => x,
        }
    }
}

#[derive(Debug)]
pub struct LoaderFindCache {
    config: ArcId<ConfigFile>,
    /// When true, all import resolution steps are origin-independent: no
    /// source_db, no sub_configs, and fallback_search_path is not
    /// DirectoryRelative. This lets us cache every module by ModuleName
    /// alone instead of (ModuleName, Option<ModulePath>).
    is_origin_independent: bool,
    cache: LockedMap<
        (ModuleName, Option<ModulePath>),
        (FindingOrError<ModulePath>, Arc<Vec<PathBuf>>),
    >,
    // If a python executable module (excludes .pyi) exists and differs from the imported python module, store it here
    executable_cache: LockedMap<(ModuleName, Option<ModulePath>), Option<ModulePath>>,
    dir_cache: DirEntryCache,
}

impl LoaderFindCache {
    pub fn new(config: ArcId<ConfigFile>) -> Self {
        // When no config feature uses origin, all import resolutions produce
        // the same result regardless of which file is importing. We can then
        // cache by ModuleName alone, reducing millions of cache entries
        // (112K files × thousands of modules) to just thousands.
        let is_origin_independent = config.source_db.is_none()
            && config.sub_configs.is_empty()
            && !matches!(
                config.fallback_search_path,
                FallbackSearchPath::DirectoryRelative(_)
            );
        Self {
            config,
            is_origin_independent,
            cache: Default::default(),
            executable_cache: Default::default(),
            dir_cache: DirEntryCache::new(),
        }
    }

    pub fn find_import_prefer_executable(
        &self,
        module: ModuleName,
        origin: Option<&ModulePath>,
        timing: Option<&TransactionTimingCounters>,
    ) -> FindingOrError<ModulePath> {
        let key = (module.dupe(), origin.cloned());
        match self.executable_cache.get(&key) {
            Some(Some(module)) => FindingOrError::new_finding(module.dupe()),
            Some(None) => self.find_import(module, origin, timing),
            None => {
                match find_import_filtered(
                    &self.config,
                    module,
                    origin,
                    Some(ModuleStyle::Executable),
                    &self.dir_cache,
                    timing,
                ) {
                    FindingOrError::Finding(import) => {
                        self.executable_cache
                            .insert(key, Some(import.finding.dupe()));
                        FindingOrError::Finding(import)
                    }
                    FindingOrError::Error(_) => {
                        self.executable_cache.insert(key, None);
                        self.find_import(module, origin, timing)
                    }
                }
            }
        }
    }

    pub fn find_import(
        &self,
        module: ModuleName,
        origin: Option<&ModulePath>,
        timing: Option<&TransactionTimingCounters>,
    ) -> FindingOrError<ModulePath> {
        // When all resolution steps are origin-independent, use None as the
        // cache key. This reduces entries from O(files × modules) to O(modules).
        let effective_origin = if self.is_origin_independent {
            None
        } else {
            origin.cloned()
        };

        // Fast path: if origin is Some, check (module, None) first for
        // previously-promoted bundled results that resolve identically
        // regardless of origin.
        if effective_origin.is_some()
            && let Some(cached) = self.cache.get(&(module.dupe(), None))
        {
            return cached.0.dupe();
        }

        let result = self
            .cache
            .ensure(&(module.dupe(), effective_origin.clone()), || {
                let phantom_paths = Vec::new();
                let result =
                    find_import(&self.config, module, origin, None, &self.dir_cache, timing);
                (result, Arc::new(phantom_paths))
            })
            .0
            .0
            .dupe();

        // Promote bundled modules to (module, None) so future lookups from
        // other origins hit the cache without redundant resolution.
        if effective_origin.is_some()
            && let FindingOrError::Finding(ref import) = result
            && import.finding.is_bundled()
        {
            self.cache
                .insert((module, None), (result.dupe(), Arc::new(Vec::new())));
        }

        result
    }

    pub fn find_import_for_tensor_shapes(
        &self,
        origin: Option<&ModulePath>,
        timing: Option<&TransactionTimingCounters>,
    ) -> FindingOrError<ModulePath> {
        let module = ModuleName::from_str("shape_extensions");
        if self.can_cache_missing_shape_extensions_independent_of_origin(module) {
            self.find_import(module, None, timing)
        } else {
            self.find_import(module, origin, timing)
        }
    }

    fn can_cache_missing_shape_extensions_independent_of_origin(&self, module: ModuleName) -> bool {
        self.config
            .source_db
            .as_ref()
            .is_some_and(|source_db| !source_db.may_contain_module(module))
            && self.config.sub_configs.is_empty()
            && !matches!(
                self.config.fallback_search_path,
                FallbackSearchPath::DirectoryRelative(_)
            )
    }

    #[allow(unused)] // will be used soon
    pub fn find_import_with_phantom_paths(
        &self,
        module: ModuleName,
        origin: Option<&ModulePath>,
        timing: Option<&TransactionTimingCounters>,
    ) -> (FindingOrError<ModulePath>, Arc<Vec<PathBuf>>) {
        let cached = self
            .cache
            .ensure(&(module.dupe(), origin.cloned()), || {
                let phantom_paths = Vec::new();
                let result =
                    find_import(&self.config, module, origin, None, &self.dir_cache, timing);
                (result, Arc::new(phantom_paths))
            })
            .0;
        (cached.0.dupe(), cached.1.dupe())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_build::source_db::map_db::MapDatabase;

    use crate::config::config::DirectoryRelativeFallbackSearchPathCache;

    use super::*;

    #[test]
    fn test_tensor_shapes_missing_marker_uses_origin_independent_cache_entry() {
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        let sys_info = config.get_sys_info();
        let mut sourcedb = MapDatabase::new(sys_info);
        sourcedb.insert(
            ModuleName::from_str("a"),
            ModulePath::memory(PathBuf::from("a.py")),
        );
        sourcedb.insert(
            ModuleName::from_str("b"),
            ModulePath::memory(PathBuf::from("b.py")),
        );
        config.source_db = Some(ArcId::new(Box::new(sourcedb)));
        config.configure();

        let loader = LoaderFindCache::new(ArcId::new(config));
        let origin_a = ModulePath::memory(PathBuf::from("a.py"));
        let origin_b = ModulePath::memory(PathBuf::from("b.py"));

        assert!(
            loader
                .find_import_for_tensor_shapes(Some(&origin_a), None)
                .finding()
                .is_none()
        );
        assert!(
            loader
                .find_import_for_tensor_shapes(Some(&origin_b), None)
                .finding()
                .is_none()
        );

        let shape_extensions = ModuleName::from_str("shape_extensions");
        let keys = loader
            .cache
            .keys()
            .filter(|(module, _)| *module == shape_extensions)
            .map(|(_, origin)| origin.dupe())
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            vec![None],
            "missing shape_extensions should be cached once under the origin-independent key"
        );
    }

    /// Precedence: when a successful resolution already carries a more specific
    /// caveat (`UntypedImport`), attaching `ImplicitRelativeImport` afterwards
    /// is a no-op. Both describe the same resolution, and `UntypedImport`
    /// ("install stubs") is strictly more actionable. This is the contract the
    /// fallback-tier attach in `find_import_internal` relies on: it calls
    /// `with_error(ImplicitRelativeImport)` unconditionally, and this test pins
    /// that an existing error is preserved rather than clobbered.
    #[test]
    fn test_with_error_implicit_relative_does_not_clobber_existing_caveat() {
        let module = ModuleName::from_str("sibling");
        let path = ModulePath::filesystem(PathBuf::from("src/sibling.py"));
        // Simulate a fallback-tier hit whose resolution already lacks stubs.
        let with_untyped = FindingOrError::Finding(Finding {
            finding: path,
            error: Some(FindError::UntypedImport(
                module,
                "types-sibling".to_owned().into(),
            )),
        });
        let after_implicit = with_untyped.with_error(FindError::ImplicitRelativeImport(module));
        match after_implicit {
            FindingOrError::Finding(Finding {
                error: Some(FindError::UntypedImport(..)),
                ..
            }) => {}
            other => panic!("expected UntypedImport to be preserved, got: {other:?}"),
        }
        // And the reverse ordering invariant: a clean implicit-relative finding
        // does get the caveat attached (the no-existing-error case).
        let clean = FindingOrError::<ModulePath>::new_finding(ModulePath::filesystem(
            PathBuf::from("src/sibling.py"),
        ));
        match clean.with_error(FindError::ImplicitRelativeImport(module)) {
            FindingOrError::Finding(Finding {
                error: Some(FindError::ImplicitRelativeImport(m)),
                ..
            }) => assert_eq!(m, module),
            other => panic!("expected ImplicitRelativeImport to attach, got: {other:?}"),
        }
    }

    /// Cache-safety: an origin-dependent implicit-relative resolution must be
    /// cached under `(module, Some(origin))` and NEVER promoted to the
    /// origin-independent `(module, None)` key. The `(module, None)` fast-path
    /// (`find_import`, above) returns to ANY origin, so a promotion there would
    /// leak the implicit-relative caveat to origins where the module actually
    /// resolves through a configured absolute root.
    ///
    /// Today the promotion gate is `import.finding.is_bundled()` (typeshed-only,
    /// `module_path.rs`), so a filesystem fallback hit is never promoted. This
    /// test pins that invariant against a future widening of `is_bundled` (or a
    /// new promotion path) that would reintroduce the leak.
    #[test]
    fn test_implicit_relative_resolution_not_promoted_to_origin_independent_key() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        // `src/main.py` imports `sibling`, which lives next to it and is on no
        // configured search path — so it resolves only via the directory walk
        // (fallback tier) and carries the implicit-relative caveat.
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.py"), "import sibling").unwrap();
        std::fs::write(root.join("src/sibling.py"), "x: int = 1").unwrap();

        let mut config = ConfigFile {
            search_path_from_file: vec![],
            fallback_search_path: FallbackSearchPath::DirectoryRelative(
                DirectoryRelativeFallbackSearchPathCache::new(None),
            ),
            ..ConfigFile::default()
        };
        config.python_environment.set_empty_to_default();
        config.configure();

        let loader = LoaderFindCache::new(ArcId::new(config));
        let module = ModuleName::from_str("sibling");
        let origin = ModulePath::filesystem(root.join("src/main.py"));

        // Resolve from `origin`. The fallback tier finds `src/sibling.py` and
        // attaches the implicit-relative caveat.
        let result = loader.find_import(module, Some(&origin), None);
        assert!(
            matches!(
                result,
                FindingOrError::Finding(Finding {
                    error: Some(FindError::ImplicitRelativeImport(..)),
                    ..
                })
            ),
            "expected the fallback resolution to carry the caveat, got: {result:?}"
        );

        // The resolution must be cached under the origin-keyed entry, NOT the
        // origin-independent `(module, None)` entry. The `(module, None)` key
        // is the fast-path that any future origin would hit, so a promotion
        // here would leak the caveat across origins.
        let keys: Vec<Option<ModulePath>> = loader
            .cache
            .keys()
            .filter(|(m, _)| *m == module)
            .map(|(_, o)| o.dupe())
            .collect();
        assert_eq!(
            keys,
            vec![Some(origin.clone())],
            "implicit-relative resolution must cache under (module, Some(origin)) only, \
             never (module, None); got keys: {keys:?}"
        );
    }
}
