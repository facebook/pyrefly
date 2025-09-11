/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use lsp_types::Url;
use lsp_types::WorkspaceFoldersChangeEvent;
use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::arc_id::WeakArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use serde::Deserialize;
use serde_json::Value;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;

use crate::commands::config_finder::standard_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::environment::environment::PythonEnvironment;
use crate::config::finder::ConfigFinder;
use crate::state::lsp::DisplayTypeErrors;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::InlayHintConfig;

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
#[derive(Debug, Clone, Default)]
pub struct Workspace {
    python_info: Option<PythonInfo>,
    search_path: Option<Vec<PathBuf>>,
    pub disable_language_services: bool,
    pub display_type_errors: Option<DisplayTypeErrors>,
    pub lsp_analysis_config: Option<LspAnalysisConfig>,
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
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
            .cartesian_product(PYTHON_EXTENSIONS.iter().chain(COMPILED_FILE_SUFFIXES))
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

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyreflyClientConfig {
    display_type_errors: Option<DisplayTypeErrors>,
    disable_language_services: Option<bool>,
    extra_paths: Option<Vec<PathBuf>>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DiagnosticMode {
    #[serde(rename = "workspace")]
    Workspace,
    #[serde(rename = "openFilesOnly")]
    OpenFilesOnly,
}

/// https://code.visualstudio.com/docs/python/settings-reference#_pylance-language-server
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspAnalysisConfig {
    #[expect(unused)]
    pub diagnostic_mode: Option<DiagnosticMode>,
    pub import_format: Option<ImportFormat>,
    pub inlay_hints: Option<InlayHintConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LspConfig {
    analysis: Option<LspAnalysisConfig>,
    python_path: Option<String>,
    pyrefly: Option<PyreflyClientConfig>,
}

pub struct Workspaces {
    /// If a workspace is not found, this one is used. It contains every possible file on the system but is lowest priority.
    default: RwLock<Workspace>,
    pub workspaces: RwLock<SmallMap<PathBuf, Workspace>>,
    pub loaded_configs: Arc<WeakConfigCache>,
}

impl Workspaces {
    pub fn new(default: Workspace, folders: &[PathBuf]) -> Self {
        Self {
            default: RwLock::new(default),
            workspaces: RwLock::new(
                folders
                    .iter()
                    .map(|x| (x.clone(), Workspace::new()))
                    .collect(),
            ),
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

    pub fn config_finder(workspaces: &Arc<Workspaces>) -> ConfigFinder {
        let workspaces = workspaces.dupe();
        standard_config_finder(Arc::new(move |dir, mut config| {
            if let Some(dir) = dir {
                workspaces.get_with(dir.to_owned(), |w| {
                    if let Some(search_path) = w.search_path.clone() {
                        config.search_path_from_args = search_path;
                    }
                    if let Some(PythonInfo {
                        interpreter,
                        mut env,
                    }) = w.python_info.clone()
                    {
                        let site_package_path: Option<Vec<PathBuf>> =
                            config.python_environment.site_package_path.take();
                        env.site_package_path = site_package_path;
                        config.interpreters.set_lsp_python_interpreter(interpreter);
                        config.python_environment = env;
                        // skip interpreter query because we already have the interpreter from the workspace
                        config.interpreters.skip_interpreter_query = true;
                    }
                })
            };

            // we print the errors here instead of returning them since
            // it gives the most immediate feedback for config loading errors
            let config_errors = config.configure();
            if !config_errors.is_empty() {
                for error in config.configure() {
                    error.print();
                }
            }
            let config = ArcId::new(config);

            workspaces.loaded_configs.insert(config.downgrade());

            (config, Vec::new())
        }))
    }

    pub fn roots(&self) -> Vec<PathBuf> {
        self.workspaces.read().keys().cloned().collect::<Vec<_>>()
    }

    pub fn changed(&self, event: WorkspaceFoldersChangeEvent) {
        let mut workspaces = self.workspaces.write();
        for x in event.removed {
            workspaces.shift_remove(&x.uri.to_file_path().unwrap());
        }
        for x in event.added {
            workspaces.insert(x.uri.to_file_path().unwrap(), Workspace::new());
        }
    }

    /// Applies the LSP client configuration to the `scope_uri` (workspace) given.
    ///
    /// The `modified` flag is changed to `true` when the configuration gets applied to the
    /// `scope_uri` matching a valid workspace
    pub fn apply_client_configuration(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        config: Value,
    ) {
        let config = match serde_json::from_value::<LspConfig>(config.clone()) {
            Err(e) => {
                eprintln!(
                    "Could not decode `LspConfig` from {config:?}, skipping client configuration request: {e}."
                );
                return;
            }
            Ok(x) => x,
        };

        if let Some(python_path) = config.python_path {
            self.update_pythonpath(modified, scope_uri, &python_path);
        }

        if let Some(pyrefly) = config.pyrefly {
            if let Some(extra_paths) = pyrefly.extra_paths {
                self.update_search_paths(modified, scope_uri, extra_paths);
            }
            if let Some(disable_language_services) = pyrefly.disable_language_services {
                self.update_disable_language_services(scope_uri, disable_language_services);
            }
            self.update_display_type_errors(modified, scope_uri, pyrefly.display_type_errors);
        }
        if let Some(analysis) = config.analysis {
            self.update_ide_settings(modified, scope_uri, analysis);
        }
    }

    /// Update disableLanguageServices setting for scope_uri, None if default workspace
    fn update_disable_language_services(
        &self,
        scope_uri: &Option<Url>,
        disable_language_services: bool,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    workspace.disable_language_services = disable_language_services;
                }
            }
            None => self.default.write().disable_language_services = disable_language_services,
        }
    }

    /// Update typeCheckingMode setting for scope_uri, None if default workspace
    fn update_display_type_errors(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        display_type_errors: Option<DisplayTypeErrors>,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    *modified = true;
                    workspace.display_type_errors = display_type_errors;
                }
            }
            None => {
                *modified = true;
                self.default.write().display_type_errors = display_type_errors
            }
        }
    }

    fn update_ide_settings(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        lsp_analysis_config: LspAnalysisConfig,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    *modified = true;
                    workspace.lsp_analysis_config = Some(lsp_analysis_config);
                }
            }
            None => {
                *modified = true;
                self.default.write().lsp_analysis_config = Some(lsp_analysis_config);
            }
        }
    }

    /// Updates pythonpath with specified python path
    /// scope_uri = None for default workspace
    fn update_pythonpath(&self, modified: &mut bool, scope_uri: &Option<Url>, python_path: &str) {
        let mut workspaces = self.workspaces.write();
        let interpreter = PathBuf::from(python_path);
        let python_info = Some(PythonInfo::new(interpreter));
        match scope_uri {
            Some(scope_uri) => {
                let workspace_path = scope_uri.to_file_path().unwrap();
                if let Some(workspace) = workspaces.get_mut(&workspace_path) {
                    *modified = true;
                    workspace.python_info = python_info;
                }
            }
            None => {
                *modified = true;
                self.default.write().python_info = python_info;
            }
        }
    }

    // Updates search paths for scope uri.
    fn update_search_paths(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        search_paths: Vec<PathBuf>,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                let workspace_path = scope_uri.to_file_path().unwrap();
                if let Some(workspace) = workspaces.get_mut(&workspace_path) {
                    *modified = true;
                    workspace.search_path = Some(search_paths);
                }
            }
            None => {
                *modified = true;
                self.default.write().search_path = Some(search_paths);
            }
        }
    }
}
