/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::arc_id::WeakArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;

use crate::commands::config_finder::standard_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::environment::environment::PythonEnvironment;
use crate::config::finder::ConfigFinder;

/// Information about the Python environment p
#[derive(Debug, Clone)]
pub struct PythonInfo {
    /// The path to the interpreter used to query this `PythonInfo`'s [`PythonEnvironment`].
    interpreter: PathBuf,
    /// The [`PythonEnvironment`] values all [`ConfigFile`]s in a given workspace should
    /// use if no explicit [`ConfigFile::python_interpreter`] is provided, or any
    /// `PythonEnvironment` values in that `ConfigFile` are unfiled. If the `interpreter
    /// provided fails to execute or is invalid, this `PythonEnvironment` might instead
    /// be a system interpreter or [`PythonEnvironment::pyrefly_default()`].
    env: PythonEnvironment,
}

impl PythonInfo {
    pub fn new(interpreter: PathBuf) -> Self {
        let (env, query_error) = PythonEnvironment::get_interpreter_env(&interpreter);
        if let Some(error) = query_error {
            error!("{error}");
        }
        Self { interpreter, env }
    }
}

/// LSP workspace settings: this is all that is necessary to run an LSP at a given root.
#[derive(Debug, Clone)]
pub struct Workspace {
    #[expect(dead_code)]
    pub root: PathBuf,
    pub python_info: Option<PythonInfo>,
    pub search_path: Option<Vec<PathBuf>>,
    pub disable_language_services: bool,
    pub disable_type_errors: bool,
}

impl Workspace {
    pub fn new(workspace_root: &Path, python_info: Option<PythonInfo>) -> Self {
        Self {
            root: workspace_root.to_path_buf(),
            python_info,
            search_path: None,
            disable_language_services: false,
            disable_type_errors: false,
        }
    }

    pub fn new_with_default_env(workspace_root: &Path) -> Self {
        Self::new(workspace_root, None)
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            root: PathBuf::from("/"),
            python_info: None,
            search_path: None,
            disable_language_services: Default::default(),
            disable_type_errors: false,
        }
    }
}

/// A cache of loaded configs from the LSP's [`standard_config_finder`]. These
/// values are [`WeakArcId`]s, so they will be dropped when no other references
/// point to them. We use this for determining the list of files we need to watch
/// when setting up the watcher.
pub struct WeakConfigCache(Mutex<HashSet<WeakArcId<ConfigFile>>>);

impl WeakConfigCache {
    pub fn new() -> Self {
        Self(Mutex::new(HashSet::new()))
    }

    pub fn insert(&self, config: WeakArcId<ConfigFile>) {
        self.0.lock().insert(config);
    }

    /// Purge any [`WeakArcId`]s that are [`WeakArcId::vacant`], and return
    /// the remaining configs, converted to [`ArcId`]s.
    pub fn clean_and_get_configs(&self) -> SmallSet<ArcId<ConfigFile>> {
        let mut configs = self.0.lock();
        let purged_config_count = configs.extract_if(|c| c.vacant()).count();
        if purged_config_count != 0 {
            eprintln!("Cleared {purged_config_count} dropped configs from config cache");
        }
        SmallSet::from_iter(configs.iter().filter_map(|c| c.upgrade()))
    }

    /// Given an [`ArcId<ConfigFile>`], get glob patterns that should be watched by a file watcher.
    /// We return a tuple of root (non-pattern part of the path) and a pattern.
    pub fn get_loaded_config_paths_to_watch(config: ArcId<ConfigFile>) -> Vec<(PathBuf, String)> {
        let mut result = Vec::new();
        let config_root = if let ConfigSource::File(config_path) = &config.source
            && let Some(root) = config_path.parent()
        {
            Some(root)
        } else {
            config.source.root()
        };
        if let Some(config_root) = config_root {
            ConfigFile::CONFIG_FILE_NAMES.iter().for_each(|config| {
                result.push((config_root.to_path_buf(), format!("**/{config}")));
            });
        }
        config
            .search_path()
            .chain(config.site_package_path())
            .cartesian_product(PYTHON_EXTENSIONS)
            .for_each(|(s, suffix)| {
                result.push((s.to_owned(), format!("**/*.{suffix}")));
            });
        result
    }

    /// Get glob patterns to watch all Python files under [`ConfigFile::search_path`] and
    /// [`ConfigFile::site_package_path`], clearing out any removed [`ConfigFile`]s as we go.
    pub fn get_patterns_for_cached_configs(&self) -> Vec<(PathBuf, String)> {
        self.clean_and_get_configs()
            .into_iter()
            .flat_map(Self::get_loaded_config_paths_to_watch)
            .collect::<Vec<_>>()
    }
}

pub struct Workspaces {
    /// If a workspace is not found, this one is used. It contains every possible file on the system but is lowest priority.
    pub default: RwLock<Workspace>,
    pub workspaces: RwLock<SmallMap<PathBuf, Workspace>>,
    pub loaded_configs: Arc<WeakConfigCache>,
}

impl Workspaces {
    pub fn new(default: Workspace) -> Self {
        Self {
            default: RwLock::new(default),
            workspaces: RwLock::new(SmallMap::new()),
            loaded_configs: Arc::new(WeakConfigCache::new()),
        }
    }

    pub fn get_with<F, R>(&self, uri: PathBuf, f: F) -> R
    where
        F: FnOnce(&Workspace) -> R,
    {
        let workspaces = self.workspaces.read();
        let default_workspace = self.default.read();
        let workspace = workspaces
            .iter()
            .filter(|(key, _)| uri.starts_with(key))
            .max_by(|(key1, _), (key2, _)| key2.ancestors().count().cmp(&key1.ancestors().count()))
            .map(|(_, workspace)| workspace);
        f(workspace.unwrap_or(&default_workspace))
    }

    pub fn config_finder(
        workspaces: &Arc<Workspaces>,
        loaded_configs: Arc<WeakConfigCache>,
    ) -> ConfigFinder {
        let workspaces = workspaces.dupe();
        standard_config_finder(Arc::new(move |dir, mut config| {
            if let Some(dir) = dir
                && config.interpreters.is_empty()
            {
                workspaces.get_with(dir.to_owned(), |w| {
                    if let Some(search_path) = w.search_path.clone() {
                        config.search_path_from_args = search_path;
                    }
                    if let Some(PythonInfo {
                        interpreter,
                        mut env,
                    }) = w.python_info.clone()
                    {
                        let site_package_path = config.python_environment.site_package_path.take();
                        env.site_package_path = site_package_path;
                        config.interpreters.set_python_interpreter(interpreter);
                        config.python_environment = env;
                    }
                })
            };

            // we print the errors here instead of returning them since
            // it gives the most immediate feedback for config loading errors
            for error in config.configure() {
                error!("Error configuring `ConfigFile`: {}", error.get_message());
            }
            let config = ArcId::new(config);

            loaded_configs.insert(config.downgrade());

            (config, Vec::new())
        }))
    }
}
