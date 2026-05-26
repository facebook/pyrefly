/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use lsp_types::Url;
use lsp_types::WorkspaceFoldersChangeEvent;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_config::config::FallbackSearchPath;
use pyrefly_config::resolve_unconfigured::UnconfiguredOverride;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::arc_id::WeakArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use serde::Deserialize;
use serde_json::Value;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::commands::config_finder::ConfigConfigurer;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::config_finder::apply_unconfigured_resolver_if_applicable;
use crate::commands::config_finder::standard_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::environment::environment::PythonEnvironment;
use crate::config::finder::ConfigFinder;
use crate::state::lsp::DisplayTypeErrors;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::InlayHintConfig;
use crate::state::lsp::TypeCheckingMode;

/// Information about the Python environment provided by this workspace.
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
    pub disabled_language_services: Option<DisabledLanguageServices>,
    pub runnable_code_lens: bool,
    pub display_type_errors: Option<DisplayTypeErrors>,
    pub type_checking_mode: Option<TypeCheckingMode>,
    /// Workspace-scoped IDE-only kill switch. When `true`, all type-error
    /// diagnostics are suppressed for files in this workspace, regardless
    /// of whether they're covered by a `pyrefly.toml`. Mirrors the
    /// in-config `disable_type_errors_in_ide` setting at the workspace
    /// level. The legacy `displayTypeErrors = "force-off"` value is
    /// mapped onto this in `apply_client_configuration`.
    pub disable_type_errors: bool,
    pub lsp_analysis_config: Option<LspAnalysisConfig>,
    pub stream_diagnostics: Option<bool>,
    pub diagnostic_mode: Option<DiagnosticMode>,
    pub workspace_config: Option<PathBuf>,
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
    }
}

struct WorkspaceConfigConfigurer(Arc<Workspaces>);

impl ConfigConfigurer for WorkspaceConfigConfigurer {
    fn configure(
        &self,
        root: Option<&std::path::Path>,
        mut config: ConfigFile,
        mut errors: Vec<pyrefly_config::finder::ConfigError>,
    ) -> (ArcId<ConfigFile>, Vec<pyrefly_config::finder::ConfigError>) {
        // The unconfigured resolver runs against the workspace's chosen
        // `typeCheckingMode` (or `Auto` if unset). Run it *before* the
        // workspace overrides below so workspace-level
        // search-paths/python-info still apply on top of the resolved
        // base.
        //
        // Pass through `get_with` even when `root` is `None` (rootless
        // in-memory paths, the parent-less fallback config in
        // `standard_config_finder`): an empty `PathBuf` matches no
        // workspace prefix, so `get_with` falls back to the default
        // workspace — and the default's `typeCheckingMode` should still
        // apply.
        let workspace_override =
            self.0
                .get_with(root.map(Path::to_owned).unwrap_or_default(), |(_, w)| {
                    w.type_checking_mode
                        .map(Into::into)
                        .unwrap_or(UnconfiguredOverride::Auto)
                });
        apply_unconfigured_resolver_if_applicable(&mut config, root, workspace_override);

        if let Some(dir) = root {
            self.0.get_with(dir.to_owned(), |(workspace_root, w)| {
                if let Some(workspace_config_path) = &w.workspace_config {
                    let (new_config, new_errors) = ConfigFile::from_file(workspace_config_path);
                    if matches!(new_config.source, ConfigSource::File(_)) {
                        // Config was parsed successfully (possibly with non-fatal
                        // warnings like extra keys). Use it.
                        config = new_config;
                        errors = new_errors;
                    } else {
                        // Config file couldn't be read or parsed. Fall back to
                        // auto-discovered config but still report the errors.
                        warn!(
                            "Failed to load workspace config at `{}`, falling back to auto-discovered config",
                            workspace_config_path.display()
                        );
                        errors = new_errors;
                    }
                }
                if let Some(search_path) = w.search_path.clone() {
                    config.search_path_from_args = search_path;
                }
                // If we already have a static fallback search path (meaning no config was found
                // and we're already using heuristics), insert workspace root as first
                // fallback_search_path so our handles (which are created from first fallback)
                // are created correctly for workspaces
                if let FallbackSearchPath::Explicit(fallback_search_path) =
                    &config.fallback_search_path
                    && let Some(workspace_root) = workspace_root
                {
                    let mut new_fallback_search_path = (**fallback_search_path).clone();
                    new_fallback_search_path.insert(0, workspace_root.to_path_buf());
                    config.fallback_search_path =
                        FallbackSearchPath::Explicit(Arc::new(new_fallback_search_path));
                }
                if let Some(PythonInfo {
                    interpreter,
                    mut env,
                }) = w.python_info.clone()
                    && config.interpreters.is_empty()
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
        for error in errors.drain(..).chain(config.configure()) {
            error.print();
        }
        let config = ArcId::new(config);

        if let Some(source_db) = &config.source_db {
            self.0
                .source_db_config_map
                .lock()
                .entry(source_db.downgrade())
                .or_default()
                .insert(config.downgrade());
        }

        self.0.loaded_configs.insert(config.downgrade());

        (config, errors)
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
            info!("Cleared {purged_config_count} dropped configs from config cache");
        }
        SmallSet::from_iter(configs.iter().filter_map(|c| c.upgrade()))
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyreflyClientConfig {
    display_type_errors: Option<DisplayTypeErrors>,
    /// Replaces `display_type_errors`. When both are set, this wins. See
    /// `apply_client_configuration` for the legacy mapping. Only governs
    /// files not covered by a `pyrefly.toml` or `[tool.pyrefly]` section,
    /// since those always take precedence.
    type_checking_mode: Option<TypeCheckingMode>,
    /// Workspace-scoped kill switch for type errors in the IDE. When
    /// `true`, all diagnostics are suppressed regardless of preset or
    /// pyrefly.toml. Replaces the global "off" half of the legacy
    /// `displayTypeErrors = "force-off"` semantics — splitting the old
    /// setting in two so users can choose between "no diagnostics
    /// anywhere" (this) and "no diagnostics outside a Pyrefly config"
    /// (`typeCheckingMode = "off"`). Defaults to `false` when absent.
    #[serde(default)]
    disable_type_errors: bool,
    disable_language_services: Option<bool>,
    extra_paths: Option<Vec<PathBuf>>,
    runnable_code_lens: Option<bool>,
    diagnostic_mode: Option<DiagnosticMode>,
    #[serde(default, deserialize_with = "deserialize_analysis")]
    analysis: Option<LspAnalysisConfig>,
    #[serde(default)]
    disabled_language_services: Option<DisabledLanguageServices>,
    stream_diagnostics: Option<bool>,
    config_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum DiagnosticMode {
    /// Compute and publish diagnostics for all files in the workspace, not just open files.
    #[serde(rename = "workspace")]
    Workspace,
    #[default]
    #[serde(rename = "openFilesOnly")]
    OpenFilesOnly,
}

/// Configuration for which language services should be disabled
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisabledLanguageServices {
    #[serde(default)]
    pub definition: bool,
    #[serde(default)]
    pub declaration: bool,
    #[serde(default)]
    pub type_definition: bool,
    #[serde(default)]
    pub code_action: bool,
    #[serde(default)]
    pub completion: bool,
    #[serde(default)]
    pub document_highlight: bool,
    #[serde(default)]
    pub references: bool,
    #[serde(default)]
    pub rename: bool,
    #[serde(default)]
    pub signature_help: bool,
    #[serde(default)]
    pub hover: bool,
    #[serde(default)]
    pub inlay_hint: bool,
    #[serde(default)]
    pub document_symbol: bool,
    #[serde(default)]
    pub code_lens: bool,
    #[serde(default)]
    pub semantic_tokens: bool,
    #[serde(default)]
    pub implementation: bool,
}

impl DisabledLanguageServices {
    /// Check if a language service is disabled based on the LSP request METHOD string
    /// Uses the METHOD constants from lsp_types::request::* types
    pub fn is_disabled(&self, method: &str) -> bool {
        match method {
            "textDocument/definition" => self.definition,
            "textDocument/declaration" => self.declaration,
            "textDocument/typeDefinition" => self.type_definition,
            "textDocument/codeAction" => self.code_action,
            "textDocument/completion" => self.completion,
            "textDocument/documentHighlight" => self.document_highlight,
            "textDocument/references" => self.references,
            "textDocument/rename" => self.rename,
            "textDocument/signatureHelp" => self.signature_help,
            "textDocument/hover" => self.hover,
            "textDocument/inlayHint" => self.inlay_hint,
            "textDocument/documentSymbol" => self.document_symbol,
            "textDocument/codeLens" => self.code_lens,
            "textDocument/semanticTokens/full" | "textDocument/semanticTokens/range" => {
                self.semantic_tokens
            }
            "textDocument/implementation" => self.implementation,
            _ => false, // Unknown methods are not disabled
        }
    }
}

/// https://code.visualstudio.com/docs/python/settings-reference#_pylance-language-server
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspAnalysisConfig {
    pub diagnostic_mode: Option<DiagnosticMode>,
    pub import_format: Option<ImportFormat>,
    pub complete_function_parens: Option<bool>,
    pub inlay_hints: Option<InlayHintConfig>,
    // TODO: this is not a pylance setting. it should be in pyrefly settings
    #[serde(default)]
    pub show_hover_go_to_links: Option<bool>,
}

fn deserialize_analysis<'de, D>(deserializer: D) -> Result<Option<LspAnalysisConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<LspAnalysisConfig>::deserialize(deserializer) {
        Ok(value) => Ok(value),
        Err(e) => {
            info!("Could not decode analysis config: {e}");
            Ok(None)
        }
    }
}

/// If both `type_checking_mode` and the legacy `display_type_errors`
/// setting are present, the new setting wins. Otherwise the legacy
/// value maps onto this setting as follows:
///
/// - `force-on` → `Default` (force-on always meant "show errors", which
///   in the new model corresponds to the Default preset on files
///   without a Pyrefly configuration).
/// - `default` and `error-missing-imports` → `Auto`. The legacy values
///   are no-ops on this axis, but we must still return `Some` so that a
///   user moving from `force-on` back to `default` actually clears the
///   override on the workspace — returning `None` would leave the old
///   `Default` sticking around.
///
/// `force-off` is intentionally NOT mapped here — its global "kill
/// switch" semantics are handled by [`resolve_disable_type_errors`]
/// instead, and on this axis it's a no-op.
///
/// Returning `None` means neither field is set; the caller writes
/// `None` back to the workspace, which clears any prior override.
fn resolve_type_checking_mode(
    new: Option<TypeCheckingMode>,
    legacy: Option<DisplayTypeErrors>,
) -> Option<TypeCheckingMode> {
    if let Some(value) = new {
        return Some(value);
    }
    match legacy? {
        DisplayTypeErrors::ForceOn => Some(TypeCheckingMode::Default),
        DisplayTypeErrors::Default | DisplayTypeErrors::ErrorMissingImports => {
            Some(TypeCheckingMode::Auto)
        }
        DisplayTypeErrors::ForceOff => None,
    }
}

/// Resolves the workspace's `disable_type_errors` kill switch from the
/// new `disableTypeErrors` field plus the legacy `displayTypeErrors`
/// setting. Output is a plain bool because the workspace state is a
/// plain bool — there is no "leave alone" case; every
/// `apply_client_configuration` call writes a definitive value.
///
/// - `new == true` → `true`. The new setting is unambiguous.
/// - `new == false`:
///   - legacy `force-off` → `true`. Historical mapping: `force-off`
///     suppressed every diagnostic regardless of where the file was,
///     and that's the new home of those semantics.
///   - everything else (`force-on`, `default`, `error-missing-imports`,
///     or absent) → `false`.
///
/// Legacy `force-on` historically pierced an in-config
/// `disable-type-errors-in-ide = true` to force errors visible. That
/// override is gone — the project's committed config now wins. Users
/// who relied on it should remove `disable-type-errors-in-ide` from
/// their `pyrefly.toml`.
fn resolve_disable_type_errors(new: bool, legacy: Option<DisplayTypeErrors>) -> bool {
    new || matches!(legacy, Some(DisplayTypeErrors::ForceOff))
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LspConfig {
    /// Settings we share with the Pylance extension for backwards compatibility
    /// See LspAnalysisConfig's docstring for more details
    #[serde(default, deserialize_with = "deserialize_analysis")]
    analysis: Option<LspAnalysisConfig>,
    python_path: Option<String>,
    /// Settings we've added that are specific to Pyrefly
    pyrefly: Option<PyreflyClientConfig>,
}

pub struct Workspaces {
    /// If a workspace is not found, this one is used. It contains every possible file on the system but is lowest priority.
    default: RwLock<Workspace>,
    pub workspaces: RwLock<SmallMap<PathBuf, Workspace>>,
    pub loaded_configs: Arc<WeakConfigCache>,
    source_db_config_map: Mutex<
        HashMap<WeakArcId<Box<dyn SourceDatabase + 'static>>, HashSet<WeakArcId<ConfigFile>>>,
    >,
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
            source_db_config_map: Mutex::new(HashMap::new()),
        }
    }

    /// Get best workspace for a given file. Return pathbuf of workspace root along
    /// with the workspace. If PathBuf is None, we could not find
    /// a workspace and are returning the default workspace.
    pub fn get_with<F, R>(&self, uri: PathBuf, f: F) -> R
    where
        F: FnOnce((Option<&PathBuf>, &Workspace)) -> R,
    {
        let workspaces = self.workspaces.read();
        let default_workspace = self.default.read();
        let workspace = workspaces
            .iter()
            .filter(|(key, _)| uri.starts_with(key))
            // select the LONGEST match (most specific workspace folder)
            .max_by(|(key1, _), (key2, _)| key1.ancestors().count().cmp(&key2.ancestors().count()))
            .map(|(path, workspace)| (Some(path), workspace));
        f(workspace.unwrap_or((None, &default_workspace)))
    }

    pub fn config_finder(
        workspaces: Arc<Workspaces>,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> ConfigFinder {
        let configure: Arc<dyn ConfigConfigurer> = Arc::new(WorkspaceConfigConfigurer(workspaces));
        standard_config_finder(configure, wrapper)
    }

    pub fn roots(&self) -> Vec<PathBuf> {
        self.workspaces.read().keys().cloned().collect::<Vec<_>>()
    }

    pub fn changed(&self, event: WorkspaceFoldersChangeEvent) {
        let mut workspaces = self.workspaces.write();
        for x in event.removed {
            if let Ok(path) = x.uri.to_file_path() {
                workspaces.shift_remove(&path);
            }
        }
        for x in event.added {
            if let Ok(path) = x.uri.to_file_path() {
                workspaces.insert(path, Workspace::new());
            }
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
                info!(
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
            if let Some(disabled_language_services) = pyrefly.disabled_language_services {
                self.update_disabled_language_services(scope_uri, disabled_language_services);
            }
            if let Some(runnable_code_lens) = pyrefly.runnable_code_lens {
                self.update_runnable_code_lens(scope_uri, runnable_code_lens);
            }
            if let Some(stream_diagnostics) = pyrefly.stream_diagnostics {
                self.update_stream_diagnostics(scope_uri, stream_diagnostics);
            }
            if let Some(diagnostic_mode) = pyrefly.diagnostic_mode {
                self.update_diagnostic_mode(scope_uri, diagnostic_mode);
            }
            // Always write a definitive value for each of these three
            // settings — including `None` when absent — so that removing a
            // setting from VS Code clears the previously-stored workspace
            // value. The `update_*` helpers compare against the current
            // value and only flip `*modified = true` on an actual change,
            // so a partial `did_change_configuration` touching some other
            // key won't pay for a needless recheck.
            self.update_display_type_errors(modified, scope_uri, pyrefly.display_type_errors);
            self.update_type_checking_mode(
                modified,
                scope_uri,
                resolve_type_checking_mode(pyrefly.type_checking_mode, pyrefly.display_type_errors),
            );
            self.update_disable_type_errors(
                modified,
                scope_uri,
                resolve_disable_type_errors(
                    pyrefly.disable_type_errors,
                    pyrefly.display_type_errors,
                ),
            );
            // Handle analysis config nested under pyrefly (e.g., pyrefly.analysis)
            if let Some(analysis) = pyrefly.analysis {
                self.update_ide_settings(modified, scope_uri, analysis);
            }
            if let Some(config_path) = pyrefly.config_path {
                self.update_workspace_config(modified, scope_uri, config_path);
            }
        }
        // Always handle analysis at top level (no longer conditional on analysis_handled)
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
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
                    workspace.disable_language_services = disable_language_services;
                }
            }
            None => self.default.write().disable_language_services = disable_language_services,
        }
    }

    /// Update disabledLanguageServices setting for scope_uri, None if default workspace
    fn update_disabled_language_services(
        &self,
        scope_uri: &Option<Url>,
        disabled_language_services: DisabledLanguageServices,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
                    workspace.disabled_language_services = Some(disabled_language_services);
                }
            }
            None => {
                self.default.write().disabled_language_services = Some(disabled_language_services);
            }
        }
    }

    fn update_runnable_code_lens(&self, scope_uri: &Option<Url>, runnable_code_lens: bool) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
                    workspace.runnable_code_lens = runnable_code_lens;
                }
            }
            None => self.default.write().runnable_code_lens = runnable_code_lens,
        }
    }

    /// Update streamDiagnostics setting for scope_uri, None if default workspace
    fn update_stream_diagnostics(&self, scope_uri: &Option<Url>, stream_diagnostics: bool) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
                    workspace.stream_diagnostics = Some(stream_diagnostics);
                }
            }
            None => self.default.write().stream_diagnostics = Some(stream_diagnostics),
        }
    }

    /// Update diagnosticMode setting for scope_uri, None if default workspace
    fn update_diagnostic_mode(&self, scope_uri: &Option<Url>, diagnostic_mode: DiagnosticMode) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
                    workspace.diagnostic_mode = Some(diagnostic_mode);
                }
            }
            None => self.default.write().diagnostic_mode = Some(diagnostic_mode),
        }
    }

    /// Update displayTypeErrors setting for scope_uri, None if default workspace
    fn update_display_type_errors(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        display_type_errors: Option<DisplayTypeErrors>,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                    && workspace.display_type_errors != display_type_errors
                {
                    *modified = true;
                    workspace.display_type_errors = display_type_errors;
                }
            }
            None => {
                let mut default = self.default.write();
                if default.display_type_errors != display_type_errors {
                    *modified = true;
                    default.display_type_errors = display_type_errors;
                }
            }
        }
    }

    fn update_type_checking_mode(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        mode: Option<TypeCheckingMode>,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                    && workspace.type_checking_mode != mode
                {
                    *modified = true;
                    workspace.type_checking_mode = mode;
                }
            }
            None => {
                let mut default = self.default.write();
                if default.type_checking_mode != mode {
                    *modified = true;
                    default.type_checking_mode = mode;
                }
            }
        }
    }

    fn update_disable_type_errors(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        disable: bool,
    ) {
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                    && workspace.disable_type_errors != disable
                {
                    *modified = true;
                    workspace.disable_type_errors = disable;
                }
            }
            None => {
                let mut default = self.default.write();
                if default.disable_type_errors != disable {
                    *modified = true;
                    default.disable_type_errors = disable;
                }
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
                if let Ok(path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&path)
                {
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
                if let Ok(workspace_path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&workspace_path)
                {
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
                if let Ok(workspace_path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&workspace_path)
                {
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

    /// Update workspace config path for scope_uri, None if default workspace.
    /// An empty path clears the workspace config (reverts to auto-discovery).
    fn update_workspace_config(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        config_path: PathBuf,
    ) {
        let workspace_config = if config_path.as_os_str().is_empty() {
            None
        } else {
            Some(config_path)
        };
        let mut workspaces = self.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Ok(workspace_path) = scope_uri.to_file_path()
                    && let Some(workspace) = workspaces.get_mut(&workspace_path)
                {
                    *modified = true;
                    workspace.workspace_config = workspace_config;
                }
            }
            None => {
                *modified = true;
                self.default.write().workspace_config = workspace_config;
            }
        }
    }

    pub fn get_configs_for_source_db(
        &self,
        source_db: ArcId<Box<dyn SourceDatabase + 'static>>,
    ) -> SmallSet<ArcId<ConfigFile>> {
        let mut map = self.source_db_config_map.lock();
        let mut result = SmallSet::new();
        let weak_source_db = source_db.downgrade();
        let Some(sourcedb_configs) = map.get_mut(&weak_source_db) else {
            return result;
        };

        sourcedb_configs.retain(|config| {
            if let Some(c) = config.upgrade() {
                result.insert(c);
                true
            } else {
                false
            }
        });
        if sourcedb_configs.is_empty() {
            map.remove(&weak_source_db);
        }

        result
    }

    pub fn sourcedb_available(&self) -> bool {
        !self.source_db_config_map.lock().is_empty()
    }

    /// Check if diagnostics should be streamed for a file at the given path.
    /// Defaults to true if not explicitly configured.
    pub fn should_stream_diagnostics(&self, path: &Path) -> bool {
        self.get_with(path.to_path_buf(), |(_, workspace)| {
            workspace.stream_diagnostics.unwrap_or(true)
        })
    }

    /// Get the diagnostic mode for a file at the given path.
    /// Checks `pyrefly.diagnosticMode` first, then falls back to
    /// `analysis.diagnosticMode`, and defaults to `OpenFilesOnly`.
    ///
    /// Workspace diagnostic mode is only honored for explicit workspace folders,
    /// never for the catch-all default workspace. If the file resolves to the
    /// default workspace, `OpenFilesOnly` is returned regardless of the default
    /// workspace's settings, preventing the server from scanning the entire filesystem.
    pub fn diagnostic_mode(&self, path: &Path) -> DiagnosticMode {
        self.get_with(path.to_path_buf(), |(workspace_root, workspace)| {
            // Only honor Workspace mode for explicit workspace folders, not
            // the catch-all default (workspace_root == None).
            if workspace_root.is_none() {
                return DiagnosticMode::OpenFilesOnly;
            }
            workspace
                .diagnostic_mode
                .or_else(|| {
                    workspace
                        .lsp_analysis_config
                        .and_then(|c| c.diagnostic_mode)
                })
                .unwrap_or_default()
        })
    }

    /// Returns the workspace roots that have `DiagnosticMode::Workspace` enabled.
    pub fn workspace_diagnostic_roots(&self) -> Vec<PathBuf> {
        self.workspaces
            .read()
            .iter()
            .filter(|(_, workspace)| {
                let mode = workspace
                    .diagnostic_mode
                    .or_else(|| {
                        workspace
                            .lsp_analysis_config
                            .and_then(|c| c.diagnostic_mode)
                    })
                    .unwrap_or_default();
                mode == DiagnosticMode::Workspace
            })
            .map(|(path, _)| path.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_get_with_selects_longest_match() {
        let workspace_root = PathBuf::from("/projects");
        let workspace_nested = PathBuf::from("/projects/my_project");

        let folders = vec![workspace_root.clone(), workspace_nested.clone()];

        let workspaces = Workspaces::new(Workspace::new(), &folders);
        {
            let mut workspaces_map = workspaces.workspaces.write();
            // Set root workspace to have a search_path
            workspaces_map.get_mut(&workspace_root).unwrap().search_path =
                Some(vec![PathBuf::from("/projects/search_root")]);
            // Set nested workspace to have a different search_path
            workspaces_map
                .get_mut(&workspace_nested)
                .unwrap()
                .search_path = Some(vec![PathBuf::from("/projects/my_project/search_nested")]);
        }

        let (workspace_path, search_path) = workspaces.get_with(
            PathBuf::from("/projects/my_project/nested_file.py"),
            |(path, w)| (path.cloned(), w.search_path.clone()),
        );
        assert_eq!(
            workspace_path,
            Some(workspace_nested.clone()),
            "Nested file should return nested workspace path"
        );
        assert_eq!(
            search_path,
            Some(vec![PathBuf::from("/projects/my_project/search_nested")]),
            "Nested file should match nested workspace (longest match), not root"
        );

        let (workspace_path, search_path) = workspaces
            .get_with(PathBuf::from("/projects/file.py"), |(path, w)| {
                (path.cloned(), w.search_path.clone())
            });
        assert_eq!(
            workspace_path,
            Some(workspace_root.clone()),
            "Root file should return root workspace path"
        );
        assert_eq!(
            search_path,
            Some(vec![PathBuf::from("/projects/search_root")]),
            "Root file should match root workspace"
        );

        let (workspace_path, search_path) = workspaces
            .get_with(PathBuf::from("/other/path/file.py"), |(path, w)| {
                (path.cloned(), w.search_path.clone())
            });
        assert_eq!(
            workspace_path, None,
            "File outside workspaces should return None for workspace path"
        );
        assert_eq!(
            search_path, None,
            "File outside workspaces should use default workspace"
        );
    }

    #[test]
    fn test_get_with_filters_prefixes() {
        let workspace_a = PathBuf::from("/workspace");
        let workspace_b = PathBuf::from("/workspace_other");

        let folders = vec![workspace_a.clone(), workspace_b.clone()];
        let workspaces = Workspaces::new(Workspace::new(), &folders);

        {
            let mut workspaces_map = workspaces.workspaces.write();
            workspaces_map.get_mut(&workspace_a).unwrap().search_path =
                Some(vec![PathBuf::from("/workspace/search_a")]);
            workspaces_map.get_mut(&workspace_b).unwrap().search_path =
                Some(vec![PathBuf::from("/workspace_other/search_b")]);
        }

        let file_a = PathBuf::from("/workspace/file.py");
        let (workspace_path, search_path) =
            workspaces.get_with(file_a, |(path, w)| (path.cloned(), w.search_path.clone()));
        assert_eq!(
            workspace_path,
            Some(workspace_a.clone()),
            "File in /workspace should return /workspace workspace path"
        );
        assert_eq!(
            search_path,
            Some(vec![PathBuf::from("/workspace/search_a")]),
            "File in /workspace should match /workspace workspace"
        );

        let file_b = PathBuf::from("/workspace_other/file.py");
        let (workspace_path, search_path) =
            workspaces.get_with(file_b, |(path, w)| (path.cloned(), w.search_path.clone()));
        assert_eq!(
            workspace_path,
            Some(workspace_b.clone()),
            "File in /workspace_other should return /workspace_other workspace path"
        );
        assert_eq!(
            search_path,
            Some(vec![PathBuf::from("/workspace_other/search_b")]),
            "File in /workspace_other should match /workspace_other workspace"
        );
    }

    #[test]
    fn test_broken_analysis_config_still_creates_lsp_config() {
        let broken_config = json!({
            "pythonPath": "/usr/bin/python3",
            "analysis": {
                "invalidField": true,
                "diagnosticMode": "invalidMode",
                "importFormat": "invalidFormat"
            },
            "pyrefly": {
                "disableLanguageServices": false,
                "extraPaths": ["/some/path"]
            }
        });

        let lsp_config: Result<LspConfig, _> = serde_json::from_value(broken_config);

        assert!(lsp_config.is_ok());
        let config = lsp_config.unwrap();
        assert!(config.analysis.is_none());
        assert_eq!(config.python_path, Some("/usr/bin/python3".to_owned()));
        assert!(config.pyrefly.is_some());
        let pyrefly = config.pyrefly.unwrap();
        assert_eq!(pyrefly.disable_language_services, Some(false));
        assert_eq!(pyrefly.extra_paths, Some(vec![PathBuf::from("/some/path")]));
    }

    #[test]
    fn test_valid_analysis_config_creates_lsp_config_with_analysis() {
        let valid_config = json!({
            "pythonPath": "/usr/bin/python3",
            "analysis": {
                "diagnosticMode": "openFilesOnly",
                "importFormat": "absolute"
            },
            "pyrefly": {
                "disableLanguageServices": false
            }
        });

        let lsp_config: Result<LspConfig, _> = serde_json::from_value(valid_config);
        assert!(lsp_config.is_ok());
        let config = lsp_config.unwrap();
        assert!(config.analysis.is_some());
        let analysis = config.analysis.unwrap();
        assert!(matches!(
            analysis.diagnostic_mode,
            Some(DiagnosticMode::OpenFilesOnly)
        ));
        assert!(matches!(
            analysis.import_format,
            Some(ImportFormat::Absolute)
        ));
        assert_eq!(config.python_path, Some("/usr/bin/python3".to_owned()));
        assert!(config.pyrefly.is_some());
    }

    /// Legacy `displayTypeErrors` maps onto the two new axes:
    /// `force-on` sets the typeCheckingMode to Default and is a no-op
    /// on the kill switch; `force-off` sets the kill switch and is a
    /// no-op on the typeCheckingMode axis. `default` and
    /// `error-missing-imports` reset the typeCheckingMode axis to
    /// `Some(Auto)` so the caller can clear a prior `force-on`
    /// override; on the kill-switch axis they're plain `false` (no
    /// reset is needed since the kill switch is a plain bool that
    /// defaults to off).
    ///
    /// Note: legacy `force-on` historically also pierced an in-config
    /// `disable-type-errors-in-ide = true`. That override is dropped
    /// here — `disable_type_errors` is a clean boolean, so there's no
    /// way to express "force show even when the project disables."
    /// Users who relied on the override should remove the in-config
    /// disable from their config.
    #[test]
    fn test_legacy_force_on_maps_to_default_preset() {
        assert_eq!(
            resolve_type_checking_mode(None, Some(DisplayTypeErrors::ForceOn)),
            Some(TypeCheckingMode::Default)
        );
        assert!(
            !resolve_disable_type_errors(false, Some(DisplayTypeErrors::ForceOn)),
            "force-on does not touch the kill switch; the in-config disable wins now"
        );
    }

    #[test]
    fn test_legacy_force_off_maps_to_kill_switch() {
        assert!(resolve_disable_type_errors(
            false,
            Some(DisplayTypeErrors::ForceOff)
        ));
        assert_eq!(
            resolve_type_checking_mode(None, Some(DisplayTypeErrors::ForceOff)),
            None,
            "force-off doesn't change the typeCheckingMode axis; kill switch covers it"
        );
    }

    /// On the `typeCheckingMode` axis, `default` / `error-missing-imports`
    /// must reset back to `Auto` so a user moving from `force-on` to
    /// `default` actually clears the prior override (returning `None`
    /// would leave the stale `Default` in place). On the
    /// `disableTypeErrors` axis the resolver doesn't need a reset case:
    /// the new field is a plain bool that defaults to `false`, so a
    /// payload without `disableTypeErrors` already starts from `false`,
    /// and `default` / `error-missing-imports` simply produce `false`.
    #[test]
    fn test_legacy_default_and_error_missing_imports_reset_type_checking_mode() {
        for variant in [
            DisplayTypeErrors::Default,
            DisplayTypeErrors::ErrorMissingImports,
        ] {
            assert_eq!(
                resolve_type_checking_mode(None, Some(variant)),
                Some(TypeCheckingMode::Auto),
                "explicit `default` / `error-missing-imports` clears any prior force-on"
            );
            assert!(
                !resolve_disable_type_errors(false, Some(variant)),
                "`default` / `error-missing-imports` produces no kill switch"
            );
        }
    }

    /// When both settings are present, the new one wins on its own
    /// axis. On the `typeCheckingMode` axis, an explicit `Strict`
    /// overrides legacy `force-off`'s lack of a no-op semantic. On the
    /// `disableTypeErrors` axis, `new == true` short-circuits the
    /// legacy mapping (regardless of legacy value).
    #[test]
    fn test_new_setting_wins_when_both_set() {
        assert_eq!(
            resolve_type_checking_mode(
                Some(TypeCheckingMode::Strict),
                Some(DisplayTypeErrors::ForceOff),
            ),
            Some(TypeCheckingMode::Strict)
        );
        assert!(
            resolve_disable_type_errors(true, Some(DisplayTypeErrors::ForceOn)),
            "new `disableTypeErrors = true` short-circuits — legacy value is irrelevant"
        );
    }

    /// Both unset → `None` on the typeCheckingMode axis (caller writes
    /// `None` to clear any prior workspace override) and `false` on the
    /// kill-switch axis (the kill switch defaults to off).
    #[test]
    fn test_neither_setting_returns_default() {
        assert_eq!(resolve_type_checking_mode(None, None), None);
        assert!(!resolve_disable_type_errors(false, None));
    }

    /// `apply_client_configuration` decides whether to flip the
    /// `modified` bit, which downstream triggers a config-cache
    /// invalidate and a full recheck. The semantics for the three
    /// type-error settings are subtle: removing a setting from VS Code
    /// settings should clear it AND flag `modified`, but a payload that
    /// merely repeats the current value (e.g. when the user toggles an
    /// unrelated setting) must NOT flag `modified`. These tests pin
    /// each interesting transition.
    mod apply_client_configuration_modified_bit {
        use super::*;

        /// Empty payload + empty workspace state → no change anywhere.
        #[test]
        fn empty_payload_does_not_flag_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            let mut modified = false;
            workspaces.apply_client_configuration(&mut modified, &None, json!({}));
            assert!(!modified);
        }

        /// Touching only an unrelated setting must not flag `modified`,
        /// even though `disableTypeErrors` defaults to `false` and the
        /// helper is called unconditionally (regression guard for the
        /// spurious-recheck bug).
        #[test]
        fn unrelated_change_does_not_flag_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            let mut modified = false;
            workspaces.apply_client_configuration(
                &mut modified,
                &None,
                json!({ "pyrefly": { "disableLanguageServices": true } }),
            );
            assert!(!modified);
            assert!(!workspaces.default.read().disable_type_errors);
        }

        /// Setting `disableTypeErrors = true` flips both the field and
        /// `modified`.
        #[test]
        fn disable_type_errors_change_flags_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            let mut modified = false;
            workspaces.apply_client_configuration(
                &mut modified,
                &None,
                json!({ "pyrefly": { "disableTypeErrors": true } }),
            );
            assert!(modified);
            assert!(workspaces.default.read().disable_type_errors);
        }

        /// A user removing `displayTypeErrors` from settings must clear
        /// the prior workspace value AND flag `modified` — otherwise a
        /// previously-set `force-on` keeps forcing diagnostics forever.
        #[test]
        fn remove_display_type_errors_clears_and_flags_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            workspaces.default.write().display_type_errors = Some(DisplayTypeErrors::ForceOn);
            let mut modified = false;
            workspaces.apply_client_configuration(&mut modified, &None, json!({ "pyrefly": {} }));
            assert!(modified);
            assert_eq!(workspaces.default.read().display_type_errors, None);
        }

        /// Same for the new setting: removing `typeCheckingMode` from
        /// settings clears the workspace and flags `modified`.
        #[test]
        fn remove_type_checking_mode_clears_and_flags_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            workspaces.default.write().type_checking_mode = Some(TypeCheckingMode::Strict);
            let mut modified = false;
            workspaces.apply_client_configuration(&mut modified, &None, json!({ "pyrefly": {} }));
            assert!(modified);
            assert_eq!(workspaces.default.read().type_checking_mode, None);
        }

        /// Re-asserting the same `typeCheckingMode` value must not flag
        /// `modified` (otherwise a partial payload re-stating the
        /// current value triggers a full recheck).
        #[test]
        fn same_type_checking_mode_does_not_flag_modified() {
            let workspaces = Workspaces::new(Workspace::new(), &[]);
            workspaces.default.write().type_checking_mode = Some(TypeCheckingMode::Strict);
            let mut modified = false;
            workspaces.apply_client_configuration(
                &mut modified,
                &None,
                json!({ "pyrefly": { "typeCheckingMode": "strict" } }),
            );
            assert!(!modified);
        }
    }
}
