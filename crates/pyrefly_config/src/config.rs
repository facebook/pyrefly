/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use derivative::Derivative;
use dupe::Dupe as _;
use itertools::Itertools;
use pyrefly_build::BuildSystem;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_build::source_db::Target;
use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ignore::Tool;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathBuf;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use pyrefly_util::lock::RwLock;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::telemetry::TelemetrySourceDbRebuildStats;
use pyrefly_util::watch_pattern::WatchPattern;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::error;

use crate::base::ConfigBase;
use crate::base::UntypedDefBehavior;
use crate::environment::environment::PythonEnvironment;
use crate::environment::interpreters::Interpreters;
use crate::error::ErrorConfig;
use crate::error::ErrorDisplayConfig;
use crate::finder::ConfigError;
use crate::module_wildcard::Match;
use crate::pyproject::PyProject;

pub static GENERATED_FILE_CONFIG_OVERRIDE: LazyLock<
    RwLock<SmallMap<ModulePathBuf, ArcId<ConfigFile>>>,
> = LazyLock::new(|| RwLock::new(SmallMap::new()));

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
pub struct SubConfig {
    pub matches: Glob,
    #[serde(flatten)]
    pub settings: ConfigBase,
}

impl SubConfig {
    fn rewrite_with_path_to_config(&mut self, config_root: &Path) {
        self.matches = self.matches.clone().from_root(config_root);
    }
}

/// Where did this config come from?
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub enum ConfigSource {
    /// This config was read from a file
    File(PathBuf),
    /// This config was synthesized with path-specific defaults, based on the location of a
    /// "marker" file that contains no pyrefly configuration but marks a project root (e.g., a
    /// `pyproject.toml` file with no `[tool.pyrefly]` section)
    Marker(PathBuf),
    #[default]
    Synthetic,
}

impl ConfigSource {
    pub fn root(&self) -> Option<&Path> {
        match &self {
            Self::File(path) | Self::Marker(path) => path.parent(),
            Self::Synthetic => None,
        }
    }
}

/// Where the importable Python code in a project lives. There are two common Python project layouts, src and flat.
/// See: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/#src-layout-vs-flat-layout
#[derive(Default)]
pub enum ProjectLayout {
    /// Python packages live directly in the project root
    #[default]
    Flat,
    /// Python packages live in a src/ subdirectory
    Src,
    /// The parent directory of the project root is the import root
    /// (this is how pandas is set up for some reason)
    Parent,
}

impl ProjectLayout {
    pub fn new(project_root: &Path) -> Self {
        let error = |path: PathBuf, error| {
            debug!(
                "Error checking for existence of path {}: {}",
                path.display(),
                error
            );
            Self::default()
        };
        let src_subdir = project_root.join("src");
        match src_subdir.try_exists() {
            Ok(true) => return Self::Src,
            Ok(false) => (),
            Err(e) => return error(src_subdir, e),
        }
        for suffix in ["py", "pyi"] {
            let init_file = project_root.join(format!("__init__.{suffix}"));
            match init_file.try_exists() {
                Ok(true) => return Self::Parent,
                Ok(false) => (),
                Err(e) => return error(init_file, e),
            }
        }
        Self::Flat
    }

    fn get_import_root(&self, project_root: &Path) -> PathBuf {
        match self {
            Self::Flat => project_root.to_path_buf(),
            Self::Src => project_root.join("src"),
            Self::Parent => project_root.parent().unwrap_or(project_root).to_path_buf(),
        }
    }
}

/// A cache for managing and producing a fallback search path from
/// some directory up to and including a root (`up_to`, which is usually a
/// config directory or filesystem root if none is provided).
/// The fallback search path consists of a given directory and its ancestors
/// up to `up_to` or `/`.
#[derive(Clone, PartialEq, Eq)]
pub struct DirectoryRelativeFallbackSearchPathCache {
    /// The cache of previously found answers.
    cache: ArcId<RwLock<SmallMap<PathBuf, Arc<Vec<PathBuf>>>>>,
    /// When running [`Self::get_ancestors`], produce paths up to and including
    /// this path. If it is `None`, produce paths up to `/`.
    up_to: Option<PathBuf>,
}

impl DirectoryRelativeFallbackSearchPathCache {
    pub fn new(up_to: Option<PathBuf>) -> Self {
        Self {
            cache: ArcId::new(RwLock::new(SmallMap::new())),
            up_to,
        }
    }

    pub fn clear(&self) {
        self.cache.write().clear()
    }

    /// Produce a vec of path ancestors from the provided path up to and including
    /// `up_to`. If any values were previously filled in, we return the cached value.
    /// Generally, this should be the directory containing a Python file, not the
    /// file itself.
    pub fn get_ancestors(&self, path: &Path) -> Arc<Vec<PathBuf>> {
        let read = self.cache.read();
        if let Some(result) = read.get(path) {
            return result.dupe();
        }
        drop(read);
        let ancestors = Arc::new(
            path.ancestors()
                .take_while(|p| self.up_to.as_ref().is_none_or(|c| p.starts_with(c)))
                .map(|p| p.to_owned())
                .collect::<Vec<_>>(),
        );
        let mut write = self.cache.write();
        if let Some(result) = write.get(path) {
            // someone beat us to it, so return their result
            return result.dupe();
        }
        write.insert(path.to_path_buf(), ancestors.dupe());
        ancestors
    }
}

impl fmt::Debug for DirectoryRelativeFallbackSearchPathCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Search path with all ancestors of paths up to config at {}",
            self.up_to.as_ref().map_or(Path::new("/"), |p| p).display()
        )
    }
}

impl fmt::Display for DirectoryRelativeFallbackSearchPathCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}

/// A struct for getting, storing, and evaluating fallback search paths. A fallback
/// search path is a search path consisting of ancestor paths from a start path
/// (usually some Python file) up to and including an end directory, which is usually
/// the filesystem root (`/`), but can also be the config.
#[derive(Default, Clone, PartialEq, Eq)]
pub enum FallbackSearchPath {
    /// A constructed fallback search path that will never change. We use this in
    /// configs where we have no idea what the project root is, and just try to
    /// import anything. This will usually be a path consisting
    /// of the starting path to the filesystem root, but extra paths may be added
    /// based on heuristics, or if we can determine an import root but aren't sure
    /// enough about it to try placing it in a higher precedence than typeshed.
    Explicit(Arc<Vec<PathBuf>>),
    /// A fallback search path where construct it based on the path we're getting an
    /// import for, but is different for every directory under a config (or filesystem
    /// root). We use this to do best-effort importing when there's an on-disk config,
    /// especially if every file should be able to attempt an import, as long as
    /// the import is relative to one of its parent directories. (One example of this
    /// is attempting to perform a loose file import in a build system. We don't know
    /// where a loose file's import root will be relative to, but we kinda just want
    /// to try everything, since for the IDE experience, we want to just find
    /// anything that matches.
    ///
    /// Example: given a project
    /// |- pyrefly.toml
    /// |- project_root/
    ///    |- a/
    ///    |  |- b/c.py
    ///    |  |- d/e.py
    ///    |- f.py
    ///
    /// If the cache's `up_to` is set to `project_root`, then:
    /// - for project_root/a/b/c.py, we would call for_directory(project_root/a/b)
    ///   and get [project_root/a/b, project_root/a, project_root]
    /// - for project_root/a/d/e.py, we would call for_directory(project_root/a/d)
    ///   and get [project_root/a/d, project_root/a, project_root]
    /// - for project_root/f.py, we would call for_directory(project_root)
    ///   and get [project_root]
    ///
    /// If the cache's `up_to` is empty, then the resulting list of paths above would
    /// continue all the way up to `/`.
    DirectoryRelative(DirectoryRelativeFallbackSearchPathCache),
    /// There is no fallback search path. These aren't the droids you're looking for.
    #[default]
    Empty,
}

impl fmt::Debug for FallbackSearchPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_for_directory(None))
    }
}

impl fmt::Display for FallbackSearchPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}

impl FallbackSearchPath {
    /// Attempt to get a fallback search path for the given directory, if any.
    /// When we have a Static variant, we return the stored path without doing anything.
    /// When we have a Dynamic variant, it only has meaning in the context of
    /// the provided path, so we can only (possibly) return a non-empty vec if the provided path
    /// is `Some`.
    pub fn for_directory(&self, directory: Option<&Path>) -> Arc<Vec<PathBuf>> {
        match (self, directory) {
            (Self::Explicit(paths), _) => paths.dupe(),
            (Self::DirectoryRelative(s), Some(path)) => s.get_ancestors(path),
            (Self::DirectoryRelative(_), None) | (Self::Empty, _) => Arc::new(vec![]),
        }
    }

    pub fn repr_for_directory(&self, directory: Option<&Path>) -> String {
        match (self, directory) {
            (Self::Explicit(paths), _) => format!("{:?}", &**paths),
            (Self::DirectoryRelative(c), Some(start)) => format!("{:?}", &**c.get_ancestors(start)),
            (Self::DirectoryRelative(c), None) => format!(
                "<paths from parent directory of all files up to {:?}>",
                c.up_to
                    .as_ref()
                    .map(|p| p.to_string_lossy())
                    .unwrap_or(Cow::Borrowed("/"))
            ),
            (Self::Empty, _) => "None".to_owned(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Explicit(paths) => paths.is_empty(),
            Self::DirectoryRelative(_) => false,
            Self::Empty => true,
        }
    }
}

pub enum ImportLookupPathPart<'a> {
    SearchPathFromArgs(&'a [PathBuf]),
    SearchPathFromFile(&'a [PathBuf]),
    ImportRoot(Option<&'a PathBuf>),
    FallbackSearchPath(&'a FallbackSearchPath, Option<&'a Path>),
    SitePackagePath(&'a [PathBuf]),
    InterpreterSitePackagePath(&'a [PathBuf]),
    BuildSystem(Option<Target>),
}

impl Display for ImportLookupPathPart<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SearchPathFromArgs(paths) => {
                write!(f, "Search path override (from command line): {paths:?}")
            }
            Self::SearchPathFromFile(paths) => {
                write!(f, "Search path (from config file): {paths:?}")
            }
            Self::ImportRoot(Some(root)) => {
                write!(f, "Import root (inferred from project layout): {root:?}")
            }
            Self::ImportRoot(None) => write!(f, "Import root (inferred from project layout): None"),
            Self::FallbackSearchPath(fallback, start) => {
                let guessed_from = if let FallbackSearchPath::Explicit(_) = &fallback {
                    " (guessed from importing file with heuristics)"
                } else if let FallbackSearchPath::DirectoryRelative(_) = &fallback
                    && start.is_some()
                {
                    " (expanded directory relative paths for file)"
                } else {
                    ""
                };

                write!(
                    f,
                    "Fallback search path{guessed_from}: {}",
                    fallback.repr_for_directory(*start),
                )
            }
            Self::SitePackagePath(paths) => {
                write!(f, "Site package path from user: {paths:?}")
            }
            Self::InterpreterSitePackagePath(paths) => {
                write!(f, "Site package path queried from interpreter: {paths:?}")
            }
            Self::BuildSystem(target) => {
                write!(f, "Build system source database")?;
                if let Some(target) = target {
                    write!(f, ": target sources and dependencies for {target}")?;
                }
                Ok(())
            }
        }
    }
}

impl ImportLookupPathPart<'_> {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::SearchPathFromArgs(paths)
            | Self::SearchPathFromFile(paths)
            | Self::SitePackagePath(paths)
            | Self::InterpreterSitePackagePath(paths) => paths.is_empty(),
            Self::ImportRoot(root) => root.is_none(),
            Self::FallbackSearchPath(inner, _) => inner.is_empty(),
            Self::BuildSystem(_) => false,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Derivative)]
#[serde(rename_all = "kebab-case")]
#[derivative(PartialEq, Eq)]
pub struct ConfigFile {
    #[serde(skip)]
    pub source: ConfigSource,

    /// Files that should be counted as sources (e.g. user-space code).
    /// NOTE: unlike other args, this is never replaced with CLI arg overrides
    /// in this config, but may be overridden by CLI args where used.
    #[serde(
         default = "ConfigFile::default_project_includes",
         skip_serializing_if = "Globs::is_empty",
         // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
         // alias while we migrate existing fields from snake case to kebab case.
         alias = "project_includes",
     )]
    pub project_includes: Globs,

    /// Files that should be excluded as sources (e.g. user-space code). These take
    /// precedence over `project_includes`.
    /// NOTE: unlike other configs, this is never replaced with CLI arg overrides
    /// in this config, but may be overridden by CLI args where used.
    #[serde(
             default,
             skip_serializing_if = "Globs::is_empty",
             // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
             // alias while we migrate existing fields from snake case to kebab case.
             alias = "project_excludes",
         )]
    pub project_excludes: Globs,

    /// Should we filter out the required excludes or filter things in your site package path?
    #[serde(default, skip_serializing_if = "crate::util::skip_default_false")]
    pub disable_project_excludes_heuristics: bool,

    #[serde(skip)]
    pub search_path_from_args: Vec<PathBuf>,

    /// The list of directories where imports are
    /// imported from, including type checked files.
    /// Does not include command-line overrides or the import root!
    /// Use ConfigFile::search_path() to get the full search path.
    #[serde(
             default,
             skip_serializing_if = "Vec::is_empty",
             rename = "search-path",
             // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
             // alias while we migrate existing fields from snake case to kebab case.
             alias = "search_path"
         )]
    pub search_path_from_file: Vec<PathBuf>,

    /// The automatically inferred subdirectory that importable Python packages live in.
    #[serde(skip)]
    pub import_root: Option<PathBuf>,

    /// Not exposed to the user. When we aren't able to determine the root of a
    /// project, we guess some fallback search paths that are checked after
    /// typeshed (so we don't clobber the stdlib) and before site_package_path.
    #[serde(default, skip)]
    pub fallback_search_path: FallbackSearchPath,

    /// Disable Pyrefly default heuristics, specifically those around
    /// constructing a modified search path. Setting this flag will instruct
    /// Pyrefly to use the exact `search_path` you give it through your config
    /// file and CLI args.
    #[serde(default, skip_serializing_if = "crate::util::skip_default_false")]
    pub disable_search_path_heuristics: bool,

    /// Override the bundled typeshed with a custom path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typeshed_path: Option<PathBuf>,

    /// Pyrefly's configurations around interpreter querying/finding.
    #[serde(flatten)]
    pub interpreters: Interpreters,

    /// Values representing the environment of the Python interpreter
    /// (which platform, Python version, ...). When we parse, these values
    /// are default to false so we know to query the `python_interpreter_path` before falling
    /// back to Pyrefly's defaults.
    #[serde(flatten)]
    pub python_environment: PythonEnvironment,

    /// The `ConfigBase` values for the whole project.
    #[serde(default, flatten)]
    pub root: ConfigBase,

    /// Sub-configs that can override specific `ConfigBase` settings
    /// based on path matching.
    #[serde(
                 default,
                 rename = "sub-config",
                 skip_serializing_if = "Vec::is_empty",
                 // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
                 // alias while we migrate existing fields from snake case to kebab case.
                 alias = "sub_config"
             )]
    pub sub_configs: Vec<SubConfig>,

    /// Whether to respect ignore files (.gitignore, .ignore, .git/exclude).
    #[serde(
        default = "ConfigFile::default_true",
        skip_serializing_if = "crate::util::skip_default_true"
    )]
    pub use_ignore_files: bool,

    /// Should this config use a build system? If so, which one?
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_system: Option<BuildSystem>,

    /// Database understanding the mapping between source files and import paths,
    /// especially within the context of a build system. This is used for getting handles
    /// for a path and doing module finding.
    #[serde(skip, default)]
    #[derivative(PartialEq = "ignore")]
    pub source_db: Option<ArcId<Box<dyn SourceDatabase>>>,

    /// Should we let Pyrefly try to index the project's files? Disabling this
    /// may speed up LSP operations on large projects.
    #[serde(default, skip_serializing_if = "crate::util::skip_default_false")]
    pub skip_lsp_config_indexing: bool,
}

impl Default for ConfigFile {
    /// An empty `ConfigFile`
    fn default() -> Self {
        ConfigFile {
            source: ConfigSource::Synthetic,
            project_includes: Default::default(),
            project_excludes: Default::default(),
            interpreters: Interpreters {
                python_interpreter_path: None,
                fallback_python_interpreter_name: None,
                conda_environment: None,
                skip_interpreter_query: false,
            },
            search_path_from_args: Vec::new(),
            search_path_from_file: Vec::new(),
            disable_search_path_heuristics: false,
            disable_project_excludes_heuristics: false,
            import_root: None,
            fallback_search_path: Default::default(),
            python_environment: Default::default(),
            root: Default::default(),
            sub_configs: Default::default(),
            build_system: Default::default(),
            source_db: Default::default(),
            use_ignore_files: true,
            typeshed_path: None,
            skip_lsp_config_indexing: false,
        }
    }
}

impl ConfigFile {
    /// Gets a ConfigFile for a project directory. `fallback` indicates whether this is a guessed
    /// project root that we're falling back to after failing to otherwise find an import.
    pub fn init_at_root(root: &Path, layout: &ProjectLayout, fallback: bool) -> Self {
        let mut result = Self {
            project_includes: Self::default_project_includes(),
            root: ConfigBase::default_for_ide_without_config(),
            ..Default::default()
        };
        let import_root = layout.get_import_root(root);
        if fallback {
            // De-prioritize guessed import roots, so they don't shadow typeshed. In particular,
            // we don't want the typing-extensions package to shadow the corresponding stub.
            result.fallback_search_path = FallbackSearchPath::Explicit(Arc::new(vec![import_root]));
        } else {
            result.import_root = Some(import_root);
        }
        // ignore failures rewriting path to config, since we're trying to construct
        // an ephemeral config for the user, and it's not fatal (but things might be
        // a little weird)
        result.rewrite_with_path_to_config(root);
        result
    }

    /// Get the project excludes, properly excluding site packages and required excludes.
    fn get_full_project_excludes(&self, mut excludes: Globs) -> Globs {
        excludes.append(Self::required_project_excludes().globs());
        excludes.append(
            &self
                .site_package_path()
                .filter(|p| !self.search_path().any(|r| r.starts_with(p)))
                .filter_map(|p| Glob::new(p.to_string_lossy().to_string()).ok())
                .collect::<Vec<_>>(),
        );
        excludes
    }

    /// Gets a [`FilteredGlobs`] from the optional `custom_excludes` or this
    /// [`ConfigFile`]s `project_excludes`, adding all `site_package_path` entries
    /// as extra exclude items.
    pub fn get_filtered_globs(&self, custom_excludes: Option<Globs>) -> FilteredGlobs {
        let project_excludes = match custom_excludes {
            None => self.project_excludes.clone(),
            Some(custom_excludes) if !self.disable_project_excludes_heuristics => {
                self.get_full_project_excludes(custom_excludes)
            }
            Some(custom_excludes) => custom_excludes,
        };
        let root = if self.use_ignore_files {
            self.import_root.as_deref()
        } else {
            None
        };
        FilteredGlobs::new(self.project_includes.clone(), project_excludes, root)
    }
}

impl ConfigFile {
    pub const PYREFLY_FILE_NAME: &str = "pyrefly.toml";
    pub const PYPROJECT_FILE_NAME: &str = "pyproject.toml";
    pub const CONFIG_FILE_NAMES: &[&str] = &[Self::PYREFLY_FILE_NAME, Self::PYPROJECT_FILE_NAME];
    /// Files that don't contain pyrefly-specific config information but indicate that we're at the
    /// root of a Python project, which should be added to the search path.
    pub const ADDITIONAL_ROOT_FILE_NAMES: &[&str] = &["mypy.ini", "pyrightconfig.json"];

    /// Writes the configuration to a file in the specified directory.
    pub fn write_to_toml_in_directory(&self, directory: &Path) -> Result<()> {
        let config_str =
            toml::to_string_pretty(&self).context("Failed to serialize config to TOML")?;

        fs_anyhow::write(&directory.join("pyrefly.toml"), config_str)
            .with_context(|| format!("Failed to write config to {}", directory.display()))?;

        Ok(())
    }

    pub fn default_project_includes() -> Globs {
        Globs::new(vec!["**/*.py*".to_owned(), "**/*.ipynb".to_owned()])
            .unwrap_or_else(|_| Globs::empty())
    }

    /// Project excludes that should always be set, even if a user or config specifies
    /// something else. These should not be absolutized, since we always want to block these
    /// files and directories, no matter where on disk they occur (outside of the project too).
    pub fn required_project_excludes() -> Globs {
        Globs::new(vec![
            // Align with https://code.visualstudio.com/docs/python/settings-reference#_pylance-language-server
            "**/node_modules".to_owned(),
            "**/__pycache__".to_owned(),
            // match any `venv` directory
            "**/venv/**".to_owned(),
            // Dot directories aside from `.` and `..` (will include .venv and .env)
            "**/.[!/.]*/**".to_owned(),
        ])
        .unwrap_or_else(|_| Globs::empty())
    }

    pub fn default_true() -> bool {
        true
    }

    pub fn from_real_config_file(&self) -> bool {
        matches!(self.source, ConfigSource::File(_))
    }

    pub fn python_version(&self) -> PythonVersion {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment.python_version.unwrap()
    }

    pub fn python_platform(&self) -> &PythonPlatform {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment.python_platform.as_ref().unwrap()
    }

    pub fn search_path(&self) -> impl Iterator<Item = &PathBuf> + Clone {
        self.search_path_from_args
            .iter()
            .chain(self.search_path_from_file.iter())
            .chain(if self.disable_search_path_heuristics {
                None.iter()
            } else {
                self.import_root.iter()
            })
    }

    pub fn site_package_path(&self) -> impl Iterator<Item = &PathBuf> + Clone {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment
            .site_package_path
            .as_ref()
            .unwrap()
            .iter()
            .chain(self.python_environment.interpreter_site_package_path.iter())
    }

    /// Gets the full, ordered path used for import lookup. Used for pretty-printing.
    pub fn structured_import_lookup_path<'a>(
        &'a self,
        origin: Option<&'a Path>,
    ) -> Vec<ImportLookupPathPart<'a>> {
        let mut result = vec![];
        if let Some(source_db) = &self.source_db {
            result.push(ImportLookupPathPart::BuildSystem(
                source_db.get_target(origin),
            ));
        }
        result.push(ImportLookupPathPart::SearchPathFromArgs(
            &self.search_path_from_args,
        ));
        result.push(ImportLookupPathPart::SearchPathFromFile(
            &self.search_path_from_file,
        ));
        if !self.disable_search_path_heuristics {
            result.push(ImportLookupPathPart::ImportRoot(self.import_root.as_ref()));
            result.push(ImportLookupPathPart::FallbackSearchPath(
                &self.fallback_search_path,
                origin.and_then(|p| p.parent()),
            ));
        }
        result.push(ImportLookupPathPart::SitePackagePath(
            self.python_environment.site_package_path.as_ref().unwrap(),
        ));
        result.push(ImportLookupPathPart::InterpreterSitePackagePath(
            &self.python_environment.interpreter_site_package_path,
        ));
        result
    }

    pub fn get_sys_info(&self) -> SysInfo {
        SysInfo::new(self.python_version(), self.python_platform().clone())
    }

    pub fn errors(&self, path: &Path) -> &ErrorDisplayConfig {
        self.get_from_sub_configs(ConfigBase::get_errors, path)
            .unwrap_or_else(||
                 // we can use unwrap here, because the value in the root config must
                 // be set in `ConfigFile::configure()`.
                 self.root.errors.as_ref().unwrap())
    }

    pub fn replace_imports_with_any(&self, path: Option<&Path>, module: ModuleName) -> bool {
        let wildcards = path
            .and_then(|path| {
                self.get_from_sub_configs(ConfigBase::get_replace_imports_with_any, path)
            })
            .unwrap_or_else(||
             // we can use unwrap here, because the value in the root config must
             // be set in `ConfigFile::configure()`.
             self.root.replace_imports_with_any.as_deref().unwrap());
        // Need to filter out any files that would be a not case.
        let found_match = wildcards.iter().find_map(|w| {
            if w.matches(module) == Match::Negative {
                Some(false)
            } else if w.matches(module) == Match::Positive {
                Some(true)
            } else {
                None
            }
        });
        found_match == Some(true)
    }

    pub fn ignore_missing_imports(&self, path: Option<&Path>, module: ModuleName) -> bool {
        let wildcards = path
            .and_then(|path| {
                self.get_from_sub_configs(ConfigBase::get_ignore_missing_imports, path)
            })
            .unwrap_or_else(||
             // we can use unwrap here, because the value in the root config must
             // be set in `ConfigFile::configure()`.
             self.root.ignore_missing_imports.as_deref().unwrap());
        let found_match = wildcards.iter().find_map(|w| {
            if w.matches(module) == Match::Negative {
                Some(false)
            } else if w.matches(module) == Match::Positive {
                Some(true)
            } else {
                None
            }
        });
        found_match == Some(true)
    }

    pub fn untyped_def_behavior(&self, path: &Path) -> UntypedDefBehavior {
        self.get_from_sub_configs(ConfigBase::get_untyped_def_behavior, path)
            .unwrap_or_else(||
                 // we can use unwrap here, because the value in the root config must
                 // be set in `ConfigFile::configure()`.
                 self.root.untyped_def_behavior.unwrap())
    }

    pub fn disable_type_errors_in_ide(&self, path: &Path) -> bool {
        self.get_from_sub_configs(ConfigBase::get_disable_type_errors_in_ide, path)
            .unwrap_or_else(|| self.root.disable_type_errors_in_ide.unwrap_or_default())
    }

    fn ignore_errors_in_generated_code(&self, path: &Path) -> bool {
        self.get_from_sub_configs(ConfigBase::get_ignore_errors_in_generated_code, path)
            .unwrap_or_else(||
                 // we can use unwrap here, because the value in the root config must
                 // be set in `ConfigFile::configure()`.
                 self.root.ignore_errors_in_generated_code.unwrap())
    }

    pub fn infer_with_first_use(&self, path: &Path) -> bool {
        self.get_from_sub_configs(ConfigBase::get_infer_with_first_use, path)
            .unwrap_or_else(||
                 // we can use unwrap here, because the value in the root config must
                 // be set in `ConfigFile::configure()`.
                 self.root.infer_with_first_use.unwrap())
    }

    pub fn enabled_ignores(&self, path: &Path) -> &SmallSet<Tool> {
        self.get_from_sub_configs(ConfigBase::get_enabled_ignores, path)
            .unwrap_or_else(||
                 // we can use unwrap here, because the value in the root config must
                 // be set in `ConfigFile::configure()`.
                 self.root.enabled_ignores.as_ref().unwrap())
    }

    pub fn get_error_config(&self, path: &Path) -> ErrorConfig<'_> {
        ErrorConfig::new(
            self.errors(path),
            self.ignore_errors_in_generated_code(path),
            self.enabled_ignores(path).clone(),
        )
    }

    /// Filter to sub configs whose matches succeed for the given `path`,
    /// then return the first non-None value the getter returns, or None
    /// if a non-empty value can't be found.
    fn get_from_sub_configs<'a, T>(
        &'a self,
        getter: impl Fn(&'a ConfigBase) -> Option<T>,
        path: &Path,
    ) -> Option<T> {
        self.sub_configs.iter().find_map(|c| {
            if c.matches.matches(path) {
                return getter(&c.settings);
            }
            None
        })
    }

    pub fn handle_from_module_path(&self, module_path: ModulePath) -> Handle {
        self.handle_from_module_path_with_fallback(module_path, std::iter::empty())
    }

    pub fn handle_from_module_path_with_fallback<'a>(
        &'a self,
        module_path: ModulePath,
        fallback_search_paths: impl Iterator<Item = &'a PathBuf>,
    ) -> Handle {
        match &self
            .source_db
            .as_ref()
            .and_then(|db| db.handle_from_module_path(&module_path))
        {
            Some(handle) => handle.dupe(),
            None => {
                let name = ModuleName::from_path(
                    module_path.as_path(),
                    self.search_path().chain(fallback_search_paths),
                )
                .unwrap_or_else(ModuleName::unknown);
                Handle::new(name, module_path, self.get_sys_info())
            }
        }
    }

    /// Get glob patterns that should be watched by a file watcher.
    /// We return a tuple of root (non-pattern part of the path) and a pattern.
    /// If pattern is None, then the root should contain the whole path to watch.
    pub fn get_paths_to_watch(&self) -> Vec<WatchPattern<'_>> {
        let mut result = Vec::new();
        if let Some(source_db) = &self.source_db {
            result.extend(source_db.get_paths_to_watch())
        }
        let config_root = self.source.root();
        if let Some(config_root) = config_root {
            Self::CONFIG_FILE_NAMES.iter().for_each(|config| {
                result.push(WatchPattern::root(config_root, format!("**/{config}")));
            });
        }
        self.search_path()
            .chain(self.site_package_path())
            .cartesian_product(PYTHON_EXTENSIONS.iter().chain(COMPILED_FILE_SUFFIXES))
            .for_each(|(s, suffix)| {
                result.push(WatchPattern::root(s, format!("**/*.{suffix}")));
            });
        result
    }

    /// Requery the source database, if one is available, for any changes that may
    /// occur with the current open set of files.
    ///
    /// When `force` is true, ignore any heuristics that would exit early if the open
    /// set of files has not changed. Should be used when a build system file
    /// or configuration file might have changed, or if we suspect the build system
    /// may produce changes in generated files.
    pub fn query_source_db(
        configs_to_files: &SmallMap<ArcId<ConfigFile>, SmallSet<ModulePath>>,
        force: bool,
    ) -> (
        SmallSet<ArcId<Box<dyn SourceDatabase + 'static>>>,
        TelemetrySourceDbRebuildStats,
    ) {
        let mut stats: TelemetrySourceDbRebuildStats = Default::default();
        let mut reloaded_source_dbs = SmallSet::new();
        let mut sourcedb_configs: SmallMap<_, Vec<_>> = SmallMap::new();
        for (config, files) in configs_to_files {
            let Some(source_db) = &config.source_db else {
                continue;
            };
            sourcedb_configs
                .entry(source_db)
                .or_default()
                .push((config, files));
            // Files can be uniquely tied to a config, so we will be counting each file at most
            // once here.
            stats.files += files.len();
        }

        stats.count = sourcedb_configs.len();
        for (source_db, configs_and_files) in sourcedb_configs {
            let all_files = configs_and_files
                .iter()
                .flat_map(|x| x.1.iter())
                .map(|p| p.module_path_buf())
                .collect::<SmallSet<_>>();
            let changed = match source_db.query_source_db(all_files, force) {
                Err(error) => {
                    error!("Error reloading source database for config: {error:?}");
                    stats.had_error = true;
                    continue;
                }
                Ok(r) => r,
            };
            let generated_files = source_db.get_generated_files();
            if !generated_files.is_empty() {
                let mut write = GENERATED_FILE_CONFIG_OVERRIDE.write();
                // we don't need any specific config here, any config for this sourcedb will work
                let first_config = configs_and_files.first().unwrap().0;
                for file in generated_files {
                    write.insert(file, first_config.dupe());
                }
            }
            if changed {
                stats.changed = true;
                debug!(
                    "Performed grouped source db query for configs at {:?}",
                    configs_and_files
                        .iter()
                        .filter_map(|x| x.0.source.root())
                        .collect::<Vec<_>>(),
                );
                reloaded_source_dbs.insert(source_db.dupe());
            }
        }
        (reloaded_source_dbs, stats)
    }

    /// Configures values that must be updated *after* overwriting with CLI flag values,
    /// which should probably be everything except for `PathBuf` or `Globs` types.
    pub fn configure(&mut self) -> Vec<ConfigError> {
        let mut configure_errors = Vec::new();

        if self.interpreters.skip_interpreter_query {
            self.python_environment.set_empty_to_default();
        } else {
            if self.interpreters.python_interpreter_path.is_some()
                && self.interpreters.fallback_python_interpreter_name.is_some()
            {
                configure_errors.push(anyhow::anyhow!(
                        "`python-interpreter-path` and `fallback-python-interpreter-name` both set, but only one can be used."
                ));
            }
            match self.interpreters.find_interpreter(self.source.root()) {
                Ok(interpreter) => {
                    let (env, error) = PythonEnvironment::get_interpreter_env(&interpreter);
                    self.python_environment.override_empty(env);
                    self.interpreters.python_interpreter_path = Some(interpreter);
                    if let Some(error) = error {
                        configure_errors.push(error);
                    }
                }
                Err(error) => {
                    self.python_environment.set_empty_to_default();
                    configure_errors.push(error.context("While finding Python interpreter"));
                }
            }
        }

        if !self.disable_project_excludes_heuristics {
            let project_excludes = mem::take(&mut self.project_excludes);
            // do this after overwriting CLI values so that we can preserve the required
            // project excludes and add the site package path.
            self.project_excludes = self.get_full_project_excludes(project_excludes);
        }

        if self.root.errors.is_none() {
            self.root.errors = Some(Default::default());
        }

        if self.root.replace_imports_with_any.is_none() {
            self.root.replace_imports_with_any = Some(Default::default());
        }

        if self.root.ignore_missing_imports.is_none() {
            self.root.ignore_missing_imports = Some(Default::default());
        }

        if self.root.untyped_def_behavior.is_none() {
            self.root.untyped_def_behavior = Some(Default::default());
        }

        if self.root.ignore_errors_in_generated_code.is_none() {
            self.root.ignore_errors_in_generated_code = Some(Default::default());
        }

        if self.root.infer_with_first_use.is_none() {
            self.root.infer_with_first_use = Some(true);
        }

        let tools_from_permissive_ignores = match self.root.permissive_ignores {
            Some(true) => Some(Tool::all()),
            Some(false) => Some(Tool::default_enabled()),
            None => None,
        };

        let enabled_ignores = match (
            tools_from_permissive_ignores,
            self.root.enabled_ignores.clone(),
        ) {
            (None, None) => Tool::default_enabled(),
            (None, Some(tools)) | (Some(tools), None) => tools,
            (Some(_), Some(tools)) => {
                configure_errors.push(anyhow!("Cannot use both `permissive-ignores` and `enabled-ignores`: `permissive-ignores` will be ignored."));
                tools
            }
        };
        self.root.enabled_ignores = Some(enabled_ignores);

        let mut configure_source_db = |build_system: &mut BuildSystem| {
            let root = match &self.source {
                ConfigSource::File(path) => {
                    let mut root = path.to_path_buf();
                    root.pop();
                    root
                }
                _ => {
                    return Some(anyhow::anyhow!(
                        "Invalid config state: `build-system` is set on project without config."
                    ));
                }
            };

            match build_system.get_source_db(root.to_path_buf())? {
                Ok(source_db) => {
                    self.source_db = Some(source_db);
                    self.fallback_search_path = FallbackSearchPath::DirectoryRelative(
                        DirectoryRelativeFallbackSearchPathCache::new(Some(root)),
                    );
                    None
                }
                Err(error) => Some(error),
            }
        };

        // TODO(connernilsen): remove once PyTorch performs an upgrade
        if cfg!(fbcode_build) {
            let root = match &self.source {
                ConfigSource::File(path) => {
                    let mut root = path.to_path_buf();
                    root.pop();
                    Some(root)
                }
                _ => None,
            };
            if let Some(root) = root
                && root.ends_with("fbsource/fbcode/caffe2")
            {
                self.build_system = Some(BuildSystem::new(
                    Some(".pyrelsp".to_owned()),
                    Some(vec![
                        "--oncall=pyre".to_owned(),
                        "--client-metadata=id=pyrefly".to_owned(),
                    ]),
                    true,
                    vec![
                        "python/typeshed_experimental".into(),
                        "python/typeshed_internal".into(),
                        "python/pyre_temporary_stubs".into(),
                    ],
                ));
            }
        }

        if let Some(build_system) = &mut self.build_system
            && let Some(error) = configure_source_db(build_system)
        {
            configure_errors.push(error)
        }

        fn validate<'a>(
            paths: &'a [PathBuf],
            field: &'a str,
        ) -> impl Iterator<Item = anyhow::Error> + 'a {
            paths.iter().filter_map(move |p| {
                validate_path(p)
                    .err()
                    .map(|err| err.context(format!("Invalid {field}")))
            })
        }
        if let Some(site_package_path) = &self.python_environment.site_package_path {
            configure_errors.extend(validate(site_package_path.as_ref(), "site-package-path"));
        }
        configure_errors.extend(validate(&self.search_path_from_file, "search-path"));

        if self.interpreters.python_interpreter_path.is_some()
            && self.interpreters.conda_environment.is_some()
        {
            configure_errors.push(anyhow::anyhow!(
                     "Cannot use both `python-interpreter-path` and `conda-environment`. Finding environment info using `python-interpreter-path`.",
             ));
        }

        if let ConfigSource::File(path) = &self.source {
            configure_errors
                .into_map(|e| ConfigError::warn(e.context(format!("{}", path.display()))))
        } else {
            configure_errors.into_map(ConfigError::warn)
        }
    }

    /// Rewrites any config values that must be updated *before* applying CLI flag values, namely
    /// rewriting any `PathBuf`s and `Globs` to be relative to `config_root`.
    /// We do this as a step separate from `configure()` because CLI args may override some of these
    /// values, but CLI args will always be relative to CWD, whereas config values should be relative
    /// to the config root.
    pub fn rewrite_with_path_to_config(&mut self, config_root: &Path) {
        self.project_includes = self.project_includes.clone().from_root(config_root);
        self.project_excludes = self.project_excludes.clone().from_root(config_root);
        self.search_path_from_file
            .iter_mut()
            .for_each(|search_root| {
                *search_root = search_root.absolutize_from(config_root);
            });
        if let Some(import_root) = &self.import_root {
            self.import_root = Some(import_root.absolutize_from(config_root));
        }
        if let Some(typeshed_path) = &self.typeshed_path {
            self.typeshed_path = Some(typeshed_path.absolutize_from(config_root));
        }
        self.python_environment
            .site_package_path
            .iter_mut()
            .for_each(|v| {
                v.iter_mut().for_each(|site_package_path| {
                    *site_package_path = site_package_path.absolutize_from(config_root);
                });
            });
        self.interpreters.python_interpreter_path = self
            .interpreters
            .python_interpreter_path
            .take()
            .map(|s| s.map(|i| i.absolutize_from(config_root)));
        self.sub_configs
            .iter_mut()
            .for_each(|c| c.rewrite_with_path_to_config(config_root));
    }

    pub fn from_file(config_path: &Path) -> (ConfigFile, Vec<ConfigError>) {
        fn read_path(config_path: &Path) -> anyhow::Result<Option<ConfigFile>> {
            let config_str = fs_anyhow::read_to_string(config_path)?;
            if config_path.file_name() == Some(OsStr::new(ConfigFile::PYPROJECT_FILE_NAME)) {
                Ok(ConfigFile::parse_pyproject_toml(&config_str)?)
            } else if config_path.file_name().is_some_and(|fi| {
                fi.to_str()
                    .is_some_and(|fi| ConfigFile::ADDITIONAL_ROOT_FILE_NAMES.contains(&fi))
            }) {
                // We'll create a file with default options but treat config_root as the project root.
                Ok(None)
            } else {
                Ok(Some(ConfigFile::parse_config(&config_str)?))
            }
        }
        fn f(config_path: &Path) -> (ConfigFile, Vec<ConfigError>) {
            let mut errors = Vec::new();
            let (maybe_config, config_source) = match read_path(config_path) {
                Ok(Some(config)) => (Some(config), ConfigSource::File(config_path.to_path_buf())),
                Ok(None) => (None, ConfigSource::Marker(config_path.to_path_buf())),
                Err(e) => {
                    errors.push(ConfigError::error(e));
                    (None, ConfigSource::File(config_path.to_path_buf()))
                }
            };
            let mut config = match config_path.parent() {
                Some(config_root) => {
                    let layout = ProjectLayout::new(config_root);
                    if let Some(mut config) = maybe_config {
                        config.rewrite_with_path_to_config(config_root);
                        config.import_root = Some(layout.get_import_root(config_root));
                        config
                    } else {
                        ConfigFile::init_at_root(config_root, &layout, false)
                    }
                }
                None => {
                    errors.push(ConfigError::error(anyhow!(
                        "Could not find parent of path `{}`",
                        config_path.display()
                    )));
                    maybe_config.unwrap_or_else(ConfigFile::default)
                }
            };
            config.source = config_source;

            if !config.root.extras.0.is_empty() {
                let extra_keys = config.root.extras.0.keys().join(", ");
                errors.push(ConfigError::warn(anyhow!(
                    "Extra keys found in config: {extra_keys}"
                )));
            }
            for sub_config in &config.sub_configs {
                if !sub_config.settings.extras.0.is_empty() {
                    let extra_keys = sub_config.settings.extras.0.keys().join(", ");
                    errors.push(ConfigError::warn(anyhow!(
                        "Extra keys found in sub config matching {}: {extra_keys}",
                        sub_config.matches
                    )));
                }
            }
            (config, errors)
        }
        let config_path = config_path.absolutize();
        let (config, errors) = f(&config_path);
        let errors = errors.into_map(|err| err.context(format!("{}", config_path.display())));
        (config, errors)
    }

    fn parse_config(config_str: &str) -> anyhow::Result<ConfigFile> {
        toml::from_str::<ConfigFile>(config_str).map_err(|err| anyhow::Error::msg(err.to_string()))
    }

    fn parse_pyproject_toml(config_str: &str) -> anyhow::Result<Option<ConfigFile>> {
        Ok(toml::from_str::<PyProject>(config_str)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?
            .pyrefly())
    }
}

impl Display for ConfigFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{source: {:?}, project_includes: {}, project_excludes: {}, search_path: [{}], python_interpreter_path: {:?}, python_environment: {}, replace_imports_with_any: [{}], ignore_missing_imports: [{}]}}",
            self.source,
            self.project_includes,
            self.project_excludes,
            self.search_path().map(|p| p.display()).join(", "),
            self.interpreters.python_interpreter_path,
            self.python_environment,
            self.root
                .replace_imports_with_any
                .as_ref()
                .map(|r| { r.iter().map(|p| p.as_str()).join(", ") })
                .unwrap_or_default(),
            self.root
                .ignore_missing_imports
                .as_ref()
                .map(|r| { r.iter().map(|p| p.as_str()).join(", ") })
                .unwrap_or_default(),
        )
    }
}

/// Returns an error if the path is definitely invalid.
pub fn validate_path(path: &Path) -> anyhow::Result<()> {
    match path.try_exists() {
        Ok(true) => Ok(()),
        Err(err) => {
            debug!(
                "Error checking for existence of path {}: {}",
                path.display(),
                err
            );
            Ok(())
        }
        Ok(false) => Err(anyhow!("`{}` does not exist", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path;

    use pretty_assertions::assert_eq;
    use pyrefly_util::test_path::TestPath;
    use tempfile::TempDir;
    use toml::Table;
    use toml::Value;

    use super::*;
    use crate::base::ExtraConfigs;
    use crate::error_kind::ErrorKind;
    use crate::error_kind::Severity;
    use crate::module_wildcard::ModuleWildcard;
    use crate::util::ConfigOrigin;

    #[test]
    fn deserialize_pyrefly_config() {
        let config_str = r#"
             project-includes = ["tests", "./implementation"]
             project-excludes = ["tests/untyped/**"]
             untyped-def-behavior = "check-and-infer-return-type"
             search-path = ["../.."]
             python-platform = "darwin"
             python-version = "1.2.3"
             site-package-path = ["venv/lib/python1.2.3/site-packages"]
             python-interpreter = "venv/my/python"
             replace-imports-with-any = ["fibonacci"]
             ignore-missing-imports = ["sprout"]
             ignore-errors-in-generated-code = true
             ignore-missing-source = true
             use-ignore-files = true

             [errors]
             assert-type = true
             bad-return = false

             [[sub-config]]
             matches = "sub/project/**"

             untyped-def-behavior = "check-and-infer-return-any"
             replace-imports-with-any = []
             ignore-missing-imports = []
             ignore-errors-in-generated-code = false
             infer-with-first-use = false
             [sub-config.errors]
             assert-type = false
             invalid-yield = false
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                source: ConfigSource::Synthetic,
                project_includes: Globs::new(vec![
                    "tests".to_owned(),
                    "./implementation".to_owned()
                ])
                .unwrap(),
                project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]).unwrap(),
                search_path_from_args: Vec::new(),
                search_path_from_file: vec![PathBuf::from("../..")],
                disable_search_path_heuristics: false,
                disable_project_excludes_heuristics: false,
                import_root: None,
                build_system: Default::default(),
                use_ignore_files: true,
                fallback_search_path: Default::default(),
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::mac()),
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    site_package_path: Some(vec![PathBuf::from(
                        "venv/lib/python1.2.3/site-packages"
                    )]),
                    interpreter_stdlib_path: vec![],
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                },
                interpreters: Interpreters {
                    python_interpreter_path: Some(ConfigOrigin::config(PathBuf::from(
                        "venv/my/python"
                    ))),
                    fallback_python_interpreter_name: None,
                    conda_environment: None,
                    skip_interpreter_query: false,
                },
                root: ConfigBase {
                    extras: Default::default(),
                    errors: Some(ErrorDisplayConfig::new(HashMap::from_iter([
                        (ErrorKind::BadReturn, Severity::Ignore),
                        (ErrorKind::AssertType, Severity::Error),
                    ]))),
                    disable_type_errors_in_ide: None,
                    ignore_errors_in_generated_code: Some(true),
                    infer_with_first_use: None,
                    replace_imports_with_any: Some(vec![ModuleWildcard::new("fibonacci").unwrap()]),
                    ignore_missing_imports: Some(vec![ModuleWildcard::new("sprout").unwrap()]),
                    untyped_def_behavior: Some(UntypedDefBehavior::CheckAndInferReturnType),
                    permissive_ignores: None,
                    enabled_ignores: None,
                },
                source_db: Default::default(),
                sub_configs: vec![SubConfig {
                    matches: Glob::new("sub/project/**".to_owned()).unwrap(),
                    settings: ConfigBase {
                        extras: Default::default(),
                        errors: Some(ErrorDisplayConfig::new(HashMap::from_iter([
                            (ErrorKind::InvalidYield, Severity::Ignore),
                            (ErrorKind::AssertType, Severity::Ignore),
                        ]))),
                        disable_type_errors_in_ide: None,
                        ignore_errors_in_generated_code: Some(false),
                        infer_with_first_use: Some(false),
                        replace_imports_with_any: Some(Vec::new()),
                        ignore_missing_imports: Some(Vec::new()),
                        untyped_def_behavior: Some(UntypedDefBehavior::CheckAndInferReturnAny),
                        permissive_ignores: None,
                        enabled_ignores: None,
                    }
                }],
                typeshed_path: None,
                skip_lsp_config_indexing: false,
            }
        );
    }

    #[test]
    fn deserialize_pyrefly_config_snake_case() {
        let config_str = r#"
             project_includes = ["tests", "./implementation"]
             project_excludes = ["tests/untyped/**"]
             untyped_def_behavior = "check-and-infer-return-type"
             search_path = ["../.."]
             python_platform = "darwin"
             python_version = "1.2.3"
             site_package_path = ["venv/lib/python1.2.3/site-packages"]
             python-interpreter-path = "venv/my/python"
             replace_imports_with_any = ["fibonacci"]
             ignore_errors_in_generated_code = true

             [errors]
             assert-type = "error"
             bad-return = "ignore"

             [[sub_config]]
             matches = "sub/project/**"

             untyped_def_behavior = "check-and-infer-return-any"
             replace_imports_with_any = []
             ignore_errors_in_generated_code = false
             [sub_config.errors]
             assert-type = "warn"
             invalid-yield = "ignore"
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(config.root.extras.0, ExtraConfigs::default().0);
        assert!(
            config
                .sub_configs
                .iter()
                .all(|c| c.settings.extras.0.is_empty())
        );
    }

    #[test]
    fn deserialize_pyrefly_config_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: ConfigFile::default_project_includes(),
                ..Default::default()
            }
        );
    }

    #[test]
    fn deserialize_pyrefly_config_with_unknown() {
        let config_str = r#"
             laszewo = "good kids"
             python_platform = "windows"

             [[sub_config]]
             matches = "abcd"

                 atliens = 1
                 "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config.root.extras.0,
            Table::from_iter([("laszewo".to_owned(), Value::String("good kids".to_owned())),])
        );
        assert_eq!(
            config.sub_configs[0].settings.extras.0,
            Table::from_iter([("atliens".to_owned(), Value::Integer(1))])
        );
    }

    #[test]
    fn deserialize_pyproject_toml() {
        let config_str = r#"
             [tool.pyrefly]
             project_includes = ["./tests", "./implementation"]
                 python_platform = "darwin"
                 python_version = "1.2.3"
                 "#;
        let config = ConfigFile::parse_pyproject_toml(config_str)
            .unwrap()
            .unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "./tests".to_owned(),
                    "./implementation".to_owned()
                ])
                .unwrap(),
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::mac()),
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                ..Default::default()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown() {
        let config_str = r#"
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
            [tool.pyrefly]
            python_version = "1.2.3"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str)
            .unwrap()
            .unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: ConfigFile::default_project_includes(),
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    python_platform: None,
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                ..Default::default()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_without_pyrefly() {
        let config_str = "
             top_level = 1
             [table1]
             table1_value = 2
                 [tool.pysa]
                 pysa_value = 2
                     ";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown_in_pyrefly() {
        let config_str = r#"
             top_level = 1
             [table1]
             table1_value = 2
                 [tool.pysa]
                 pysa_value = 2
                     [tool.pyrefly]
                     python_version = "1.2.3"
                         inzo = "overthinker"
                         "#;
        let config = ConfigFile::parse_pyproject_toml(config_str)
            .unwrap()
            .unwrap();
        assert_eq!(
            config.root.extras.0,
            Table::from_iter([("inzo".to_owned(), Value::String("overthinker".to_owned()))])
        );
    }

    #[test]
    fn test_rewrite_with_path_to_config() {
        fn with_sep(s: &str) -> String {
            s.replace("/", path::MAIN_SEPARATOR_STR)
        }
        let typeshed = "path/to/typeshed";
        let mut python_environment = PythonEnvironment {
            site_package_path: Some(vec![PathBuf::from("venv/lib/python1.2.3/site-packages")]),
            ..PythonEnvironment::default()
        };
        let interpreter = "venv/bin/python3".to_owned();
        let mut config = ConfigFile {
            source: ConfigSource::Synthetic,
            project_includes: Globs::new(vec!["path1/**".to_owned(), "path2/path3".to_owned()])
                .unwrap(),
            project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]).unwrap(),
            search_path_from_args: Vec::new(),
            search_path_from_file: vec![PathBuf::from("../..")],
            disable_search_path_heuristics: false,
            disable_project_excludes_heuristics: false,
            import_root: None,
            use_ignore_files: true,
            fallback_search_path: Default::default(),
            python_environment: python_environment.clone(),
            interpreters: Interpreters {
                python_interpreter_path: Some(ConfigOrigin::config(PathBuf::from(
                    interpreter.clone(),
                ))),
                fallback_python_interpreter_name: None,
                conda_environment: None,
                skip_interpreter_query: false,
            },
            root: Default::default(),
            source_db: Default::default(),
            build_system: Default::default(),
            sub_configs: vec![SubConfig {
                matches: Glob::new("sub/project/**".to_owned()).unwrap(),
                settings: Default::default(),
            }],
            typeshed_path: Some(PathBuf::from(typeshed)),
            skip_lsp_config_indexing: false,
        };

        let current_dir = std::env::current_dir().unwrap();
        let path_str = with_sep("path/to/my/config");
        let test_path = current_dir.join(&path_str);

        let project_includes_vec = vec![
            test_path.join("path1/**").to_string_lossy().into_owned(),
            test_path.join("path2/path3").to_string_lossy().into_owned(),
        ];
        let project_excludes_vec = vec![
            test_path
                .join("tests/untyped/**")
                .to_string_lossy()
                .into_owned(),
        ];
        let search_path = vec![test_path.parent().unwrap().parent().unwrap().to_path_buf()];
        let expected_typeshed = test_path.join(typeshed);
        python_environment.site_package_path =
            Some(vec![test_path.join("venv/lib/python1.2.3/site-packages")]);

        let sub_config_matches = Glob::new(
            test_path
                .join("sub/project/**")
                .to_string_lossy()
                .into_owned(),
        )
        .unwrap();

        config.rewrite_with_path_to_config(&test_path);

        let expected_config = ConfigFile {
            source: ConfigSource::Synthetic,
            project_includes: Globs::new(project_includes_vec).unwrap(),
            project_excludes: Globs::new(project_excludes_vec).unwrap(),
            interpreters: Interpreters {
                python_interpreter_path: Some(ConfigOrigin::config(test_path.join(interpreter))),
                fallback_python_interpreter_name: None,
                conda_environment: None,
                skip_interpreter_query: false,
            },
            search_path_from_args: Vec::new(),
            search_path_from_file: search_path,
            disable_search_path_heuristics: false,
            disable_project_excludes_heuristics: false,
            use_ignore_files: true,
            import_root: None,
            fallback_search_path: Default::default(),
            python_environment,
            root: Default::default(),
            build_system: Default::default(),
            source_db: Default::default(),
            sub_configs: vec![SubConfig {
                matches: sub_config_matches,
                settings: Default::default(),
            }],
            typeshed_path: Some(expected_typeshed),
            skip_lsp_config_indexing: false,
        };
        assert_eq!(config, expected_config);
    }

    #[test]
    fn test_deserializing_unknown_error_errors() {
        let config_str = "
             [errors]
             subtronics = true
                 zeds_dead = false
                 GRiZ = true
                 ";
        let err = ConfigFile::parse_config(config_str).unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }

    #[test]
    fn test_deserializing_sub_config_missing_matches() {
        let config_str = r#"
             [[sub_config]]
             search_path = ["../../.."]
                 "#;
        let err = ConfigFile::parse_config(config_str).unwrap_err();
        assert!(err.to_string().contains("missing field `matches`"));
    }

    #[test]
    fn test_expect_all_fields_set_in_root_config() {
        let root = TempDir::new().unwrap();
        let mut config = ConfigFile::init_at_root(root.path(), &ProjectLayout::default(), false);
        config.configure();

        let table: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();

        let ignore_keys: Vec<String> = vec![
            // top level configs, where null values (if possible), should be allowed
            "project-includes",
            "project-excludes",
            "python-interpreter-path",
            "fallback-python-interpreter-name",
            // values we won't be getting
            "extras",
            // values that must be Some (if flattened, their contents will be checked)
            "python_environment",
        ]
        .into_iter()
        .map(|k| k.to_owned())
        .collect();

        table.keys().for_each(|k| {
            if ignore_keys.contains(k) {
                return;
            }

            assert!(
                table.get(k).is_some_and(|v| !v.is_null()),
                "Value for {k} is None after ConfigFile::configure()"
            );
        });
    }

    #[test]
    fn test_get_from_sub_configs() {
        let config = ConfigFile {
            root: ConfigBase {
                errors: Some(Default::default()),
                replace_imports_with_any: Some(vec![ModuleWildcard::new("root").unwrap()]),
                ignore_missing_imports: None,
                untyped_def_behavior: Some(UntypedDefBehavior::CheckAndInferReturnType),
                disable_type_errors_in_ide: Some(true),
                ignore_errors_in_generated_code: Some(false),
                infer_with_first_use: Some(true),
                extras: Default::default(),
                permissive_ignores: Some(false),
                enabled_ignores: None,
            },
            sub_configs: vec![
                SubConfig {
                    matches: Glob::new("**/highest/**".to_owned()).unwrap(),
                    settings: ConfigBase {
                        replace_imports_with_any: Some(vec![
                            ModuleWildcard::new("highest").unwrap(),
                        ]),
                        ignore_errors_in_generated_code: None,
                        ..Default::default()
                    },
                },
                SubConfig {
                    matches: Glob::new("**/priority*".to_owned()).unwrap(),
                    settings: ConfigBase {
                        replace_imports_with_any: Some(vec![
                            ModuleWildcard::new("second").unwrap(),
                        ]),
                        ignore_errors_in_generated_code: Some(true),
                        ..Default::default()
                    },
                },
            ],
            ..Default::default()
        };

        // test precedence (two configs match, one higher priority)
        assert!(config.replace_imports_with_any(
            Some(Path::new("this/is/highest/priority")),
            ModuleName::from_str("highest")
        ));

        // test find fallback match
        assert!(config.replace_imports_with_any(
            Some(Path::new("this/is/second/priority")),
            ModuleName::from_str("second")
        ));

        // test empty value falls back to next
        assert!(config.ignore_errors_in_generated_code(Path::new("this/is/highest/priority")));

        // test no pattern match
        assert!(config.replace_imports_with_any(
            Some(Path::new("this/does/not/match/any")),
            ModuleName::from_str("root")
        ));

        // test replace_imports_with_any special case None path
        assert!(config.replace_imports_with_any(None, ModuleName::from_str("root")));
    }

    #[test]
    fn test_default_search_path() {
        let tempdir = TempDir::new().unwrap();
        let config = ConfigFile::init_at_root(tempdir.path(), &ProjectLayout::default(), false);
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![tempdir.path().to_path_buf()]
        );
    }

    #[test]
    fn test_pyproject_toml_search_path() {
        let root = TempDir::new().unwrap();
        let path = root.path().join(ConfigFile::PYPROJECT_FILE_NAME);
        fs::write(&path, "[tool.pyrefly]").unwrap();
        let config = ConfigFile::from_file(&path).0;
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![root.path().to_path_buf()]
        );
    }

    fn create_empty_file_and_parse_config(root: &TempDir, name: &str) -> ConfigFile {
        let path = root.path().join(name);
        fs::write(&path, "").unwrap();
        ConfigFile::from_file(&path).0
    }

    #[test]
    fn test_pyproject_toml_no_pyrefly_search_path() {
        let root = TempDir::new().unwrap();
        let config = create_empty_file_and_parse_config(&root, ConfigFile::PYPROJECT_FILE_NAME);
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![root.path().to_path_buf()]
        );
    }

    #[test]
    fn test_mypy_config_search_path() {
        let root = TempDir::new().unwrap();
        let config = create_empty_file_and_parse_config(&root, "mypy.ini");
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![root.path().to_path_buf()]
        );
    }

    #[test]
    fn test_pyright_config_search_path() {
        let root = TempDir::new().unwrap();
        let config = create_empty_file_and_parse_config(&root, "pyrightconfig.json");
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![root.path().to_path_buf()]
        );
    }

    #[test]
    fn test_src_layout_default_config() {
        // root/
        // - pyproject.toml (empty)
        // - src/
        // - my_amazing_scripts/
        //   - foo.py
        let root = TempDir::new().unwrap();
        let src_dir = root.path().join("src");
        let scripts_dir = root.path().join("my_amazing_scripts");
        let python_file = scripts_dir.join("foo.py");
        fs::create_dir(&src_dir).unwrap();
        fs::create_dir(&scripts_dir).unwrap();
        fs::write(&python_file, "").unwrap();
        let config = create_empty_file_and_parse_config(&root, ConfigFile::PYPROJECT_FILE_NAME);
        // We should still find Python files (commonly scripts and tests) outside src/.
        assert_eq!(config.project_includes.files().unwrap(), vec![python_file]);
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![src_dir]
        );
    }

    #[test]
    fn test_src_layout_with_config() {
        // root/
        // - pyrefly.toml
        // - src/
        let root = TempDir::new().unwrap();
        let src_dir = root.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();
        let pyrefly_path = root.path().join(ConfigFile::PYREFLY_FILE_NAME);
        fs::write(&pyrefly_path, "project_includes = [\"**/*\"]").unwrap();
        let config = ConfigFile::from_file(&pyrefly_path).0;
        // File contents should still be relative to the location of the config file, not src/.
        assert_eq!(
            config.project_includes,
            Globs::new(vec![root.path().join("**/*").to_string_lossy().to_string()]).unwrap(),
        );
        assert_eq!(
            config.search_path().cloned().collect::<Vec<_>>(),
            vec![src_dir]
        );
    }

    #[test]
    fn test_get_filtered_globs() {
        let mut config = ConfigFile::default();
        let site_package_path = vec![
            "venv/site_packages".to_owned(),
            "system/site_packages".to_owned(),
            "my_search_path".to_owned(),
        ];
        config.interpreters.skip_interpreter_query = true;
        config.python_environment.site_package_path = Some(
            site_package_path
                .iter()
                .map(PathBuf::from)
                .collect::<Vec<_>>(),
        );
        config.search_path_from_file = vec![PathBuf::from("my_search_path")];
        config.project_excludes = ConfigFile::required_project_excludes();

        config.configure();

        let mut expected_site_package_path = site_package_path;
        // get rid of "my_search_path" in site package path, since it's going to be removed
        // when we add site package path to project excludes
        expected_site_package_path.pop();

        assert_eq!(
            config.get_filtered_globs(None),
            FilteredGlobs::new(
                config.project_includes.clone(),
                Globs::new(
                    vec![
                        "**/node_modules".to_owned(),
                        "**/__pycache__".to_owned(),
                        "**/venv/**".to_owned(),
                        "**/.[!/.]*/**".to_owned(),
                    ]
                    .into_iter()
                    .chain(vec![
                        "**/node_modules".to_owned(),
                        "**/__pycache__".to_owned(),
                        "**/venv/**".to_owned(),
                        "**/.[!/.]*/**".to_owned(),
                    ])
                    .chain(expected_site_package_path.clone())
                    .collect::<Vec<_>>()
                )
                .unwrap(),
                None,
            )
        );
        assert_eq!(
            config.get_filtered_globs(Some(
                Globs::new(vec!["custom_excludes".to_owned()]).unwrap()
            )),
            FilteredGlobs::new(
                config.project_includes.clone(),
                Globs::new(
                    vec!["custom_excludes".to_owned()]
                        .into_iter()
                        .chain(vec![
                            "**/node_modules".to_owned(),
                            "**/__pycache__".to_owned(),
                            "**/venv/**".to_owned(),
                            "**/.[!/.]*/**".to_owned(),
                        ])
                        .chain(expected_site_package_path)
                        .collect::<Vec<_>>()
                )
                .unwrap(),
                None,
            )
        );
    }

    #[test]
    fn test_python_interpreter_conda_environment() {
        let mut config = ConfigFile {
            interpreters: Interpreters {
                python_interpreter_path: Some(ConfigOrigin::config(PathBuf::new())),
                fallback_python_interpreter_name: None,
                conda_environment: Some(ConfigOrigin::config("".to_owned())),
                skip_interpreter_query: false,
            },
            ..Default::default()
        };

        let validation_errors = config.configure();

        assert!(
             validation_errors.iter().any(|e| {
                 e.get_message() == "Cannot use both `python-interpreter-path` and `conda-environment`. Finding environment info using `python-interpreter-path`."
             })
         );
    }

    #[test]
    fn test_interpreter_not_queried_with_skip_interpreter_query() {
        let mut config = ConfigFile {
            interpreters: Interpreters {
                skip_interpreter_query: true,
                ..Default::default()
            },
            ..Default::default()
        };

        config.configure();
        assert!(config.interpreters.python_interpreter_path.is_none());
        assert!(config.interpreters.conda_environment.is_none());
    }

    #[test]
    fn test_serializing_config_origins() {
        let mut config = ConfigFile {
            interpreters: Interpreters {
                python_interpreter_path: Some(ConfigOrigin::config(PathBuf::from("abcd"))),
                fallback_python_interpreter_name: None,
                conda_environment: None,
                skip_interpreter_query: false,
            },
            project_includes: ConfigFile::default_project_includes(),
            ..Default::default()
        };
        let reparsed = ConfigFile::parse_config(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(reparsed, config);

        config.interpreters.python_interpreter_path =
            Some(ConfigOrigin::auto(PathBuf::from("abcd")));
        let reparsed = ConfigFile::parse_config(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(reparsed.interpreters.python_interpreter_path, None);

        config.interpreters.python_interpreter_path =
            Some(ConfigOrigin::cli(PathBuf::from("abcd")));
        let reparsed = ConfigFile::parse_config(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(reparsed.interpreters.python_interpreter_path, None);
    }

    #[test]
    fn test_negation_replace_imports_with_any() {
        let config = ConfigFile {
            root: ConfigBase {
                errors: Some(Default::default()),
                replace_imports_with_any: Some(vec![
                    ModuleWildcard::new("!example.path.specific.*").unwrap(),
                    ModuleWildcard::new("example.path.*").unwrap(),
                ]),
                ignore_missing_imports: None,
                untyped_def_behavior: Some(UntypedDefBehavior::CheckAndInferReturnType),
                disable_type_errors_in_ide: Some(true),
                ignore_errors_in_generated_code: Some(false),
                infer_with_first_use: Some(true),
                extras: Default::default(),
                permissive_ignores: Some(false),
                enabled_ignores: None,
            },
            sub_configs: vec![],
            ..Default::default()
        };

        assert!(!config.replace_imports_with_any(
            Some(Path::new("example/path")),
            ModuleName::from_str("example.path.specific.a")
        ));
        assert!(config.replace_imports_with_any(
            Some(Path::new("example/path")),
            ModuleName::from_str("example.path.b")
        ));
    }

    #[test]
    fn test_negation_replace_imports_with_any_reorder() {
        let config = ConfigFile {
            root: ConfigBase {
                errors: Some(Default::default()),
                replace_imports_with_any: Some(vec![
                    ModuleWildcard::new("example.path.*").unwrap(),
                    ModuleWildcard::new("!example.path.specific.*").unwrap(),
                ]),
                ignore_missing_imports: None,
                untyped_def_behavior: Some(UntypedDefBehavior::CheckAndInferReturnType),
                disable_type_errors_in_ide: Some(true),
                ignore_errors_in_generated_code: Some(false),
                infer_with_first_use: Some(true),
                extras: Default::default(),
                permissive_ignores: Some(false),
                enabled_ignores: None,
            },
            sub_configs: vec![],
            ..Default::default()
        };
        // Based on the order this one will always be true.
        assert!(config.replace_imports_with_any(
            Some(Path::new("example/path")),
            ModuleName::from_str("example.path.specific.a")
        ));
        assert!(config.replace_imports_with_any(
            Some(Path::new("example/path")),
            ModuleName::from_str("example.path.b")
        ));
    }

    #[test]
    fn test_dynamic_fallback_search_path() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "foo",
                vec![
                    TestPath::dir("bar", vec![]),
                    TestPath::dir("baz", vec![TestPath::dir("quux", vec![])]),
                ],
            )],
        );

        let bounded = DirectoryRelativeFallbackSearchPathCache::new(Some(root.to_path_buf()));
        let unbounded = DirectoryRelativeFallbackSearchPathCache::new(None);

        let compare_paths = |start: PathBuf, expected_bounded: Vec<PathBuf>| {
            let bounded_result = bounded.get_ancestors(&start);
            let unbounded_result = unbounded.get_ancestors(&start);
            let expected_unbounded = expected_bounded
                .iter()
                .map(|p| &**p)
                .chain(root.ancestors().skip(1))
                .map(PathBuf::from)
                .collect::<Vec<PathBuf>>();
            assert_eq!(
                *bounded_result, expected_bounded,
                "Got different results for bounded {start:?}",
            );
            assert_eq!(
                *unbounded_result, expected_unbounded,
                "Got different results for unbounded {start:?}",
            );
        };

        compare_paths(
            root.join("foo/baz/quux"),
            vec![
                root.join("foo/baz/quux"),
                root.join("foo/baz"),
                root.join("foo"),
                root.to_path_buf(),
            ],
        );
        compare_paths(
            root.join("foo/baz"),
            vec![root.join("foo/baz"), root.join("foo"), root.to_path_buf()],
        );
        compare_paths(root.join("foo"), vec![root.join("foo"), root.to_path_buf()]);
        compare_paths(root.join("bar"), vec![root.join("bar"), root.to_path_buf()]);
        compare_paths(root.to_path_buf(), vec![root.to_path_buf()]);
        // test this one again to make sure caching works
        compare_paths(
            root.join("foo/baz/quux"),
            vec![
                root.join("foo/baz/quux"),
                root.join("foo/baz"),
                root.join("foo"),
                root.to_path_buf(),
            ],
        );
    }

    #[test]
    fn test_disable_excludes_heuristics() {
        let mut disabled_config = ConfigFile {
            disable_project_excludes_heuristics: true,
            interpreters: Interpreters {
                skip_interpreter_query: true,
                ..Default::default()
            },
            python_environment: PythonEnvironment {
                site_package_path: Some(vec![PathBuf::from("spp")]),
                ..Default::default()
            },
            project_excludes: Globs::new(vec!["my_project_excludes".to_owned()]).unwrap(),
            ..Default::default()
        };
        let mut enabled_config = disabled_config.clone();
        enabled_config.disable_project_excludes_heuristics = false;

        disabled_config.configure();
        enabled_config.configure();

        assert_eq!(
            &disabled_config.project_excludes,
            &Globs::new(vec!["my_project_excludes".to_owned()]).unwrap(),
        );
        let mut full_project_excludes = Globs::new(vec!["my_project_excludes".to_owned()]).unwrap();

        full_project_excludes.append(ConfigFile::required_project_excludes().globs());
        full_project_excludes.append(&[Glob::new("spp".to_owned()).unwrap()]);
        assert_eq!(&enabled_config.project_excludes, &full_project_excludes);
    }
}
