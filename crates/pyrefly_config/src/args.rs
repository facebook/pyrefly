/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use pyrefly_python::ignore::Tool;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::display;

use crate::base::UntypedDefBehavior;
use crate::config::ConfigFile;
use crate::config::validate_path;
use crate::error::ErrorDisplayConfig;
use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;
use crate::finder::ConfigError;
use crate::module_wildcard::ModuleWildcard;
use crate::util::ConfigOrigin;

/// Parser function to convert paths to absolute paths
fn absolute_path_parser(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    Ok(path.absolutize())
}

/// config overrides
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone, Default)]
pub struct ConfigOverrideArgs {
    /// Configures Pyrefly to replace `project-excludes` fully rather than
    /// append whatever is in your configuration or passed by CLI to Pyrefly's
    /// defaults.
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1,
    )]
    disable_project_excludes_heuristics: Option<bool>,

    /// The list of directories where imports are imported from, including
    /// type checked files.
    #[arg(long, value_parser = absolute_path_parser)]
    search_path: Option<Vec<PathBuf>>,

    /// Disable Pyrefly default heuristics, specifically those around
    /// constructing a modified search path. Setting this flag will instruct
    /// Pyrefly to use the exact `search_path` you give it through your config
    /// file and CLI args.
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1,
    )]
    disable_search_path_heuristics: Option<bool>,

    /// The Python version any `sys.version` checks should evaluate against.
    #[arg(long)]
    python_version: Option<PythonVersion>,

    /// The platform any `sys.platform` checks should evaluate against.
    #[arg(long)]
    python_platform: Option<PythonPlatform>,

    /// Directories containing third-party package imports, searched
    /// after first checking `search_path` and `typeshed`.
    #[arg(long)]
    site_package_path: Option<Vec<PathBuf>>,

    /// Use a specific Conda environment to query Python environment information,
    /// even if it isn't activated.
    #[arg(long, group = "env_source")]
    conda_environment: Option<String>,

    /// The path to a Python executable that will be queried for `python-version`
    /// `python-platform`, or `site-package-path` if any of the values are missing.
    #[arg(long, value_name = "EXE_PATH", group = "env_source")]
    python_interpreter_path: Option<PathBuf>,

    /// The Python executable name available on your PATH that will be queried for your
    /// `python-version`, `python-platform`, or `site-package-path` if any of the values
    /// are missing. We execute `which <COMMAND>` to fill in your `python-interpreter-path`,
    /// which is useful if you don't know where the Python executable will be on a given
    /// machine, but want to use one other than the default.
    /// When this and `python-interpreter-path` are unset, we query for `python3` and `python`.
    #[arg(long, value_name = "COMMAND", group = "env_source")]
    fallback_python_interpreter_name: Option<String>,

    /// Skip doing any automatic querying for `python-interpreter-path`,
    /// `fallback-python-interpreter-name`, or `conda-environment`
    #[arg(long, group = "env_source")]
    skip_interpreter_query: bool,

    /// Override the bundled typeshed with a custom path.
    #[arg(long)]
    typeshed_path: Option<PathBuf>,

    /// Always replace specified imports with typing.Any, suppressing related import errors even if the module is found.
    #[arg(long)]
    replace_imports_with_any: Option<Vec<String>>,
    /// If the specified imported module can't be found, replace it with typing.Any, suppressing
    /// related import errors.
    #[arg(long)]
    ignore_missing_imports: Option<Vec<String>>,
    /// Whether to ignore type errors in generated code.
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    ignore_errors_in_generated_code: Option<bool>,
    /// If this is true, infer type variables not determined by a call or constructor based on their first usage.
    /// For example, the type of an empty container would be determined by the first thing you put into it.
    /// If this is false, any unsolved type variables at the end of a call or constructor will be replaced with `Any`.
    /// Defaults to true.
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    infer_with_first_use: Option<bool>,
    /// Whether to respect ignore files (.gitignore, .ignore, .git/exclude).
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    use_ignore_files: Option<bool>,
    /// Controls how Pyrefly analyzes function definitions that lack type annotations on parameters and return values.
    #[arg(long)]
    untyped_def_behavior: Option<UntypedDefBehavior>,
    /// Whether Pyrefly will respect ignore statements for other tools, e.g. `# pyright: ignore`.
    /// Equivalent to passing the names of all tools to `--enabled-ignores`.
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    permissive_ignores: Option<bool>,
    /// Respect ignore directives from only these tools. Can be passed multiple times or as a comma-separated list.
    /// Defaults to type,pyrefly. Passing the names of all tools is equivalent to `--permissive-ignores`.
    #[arg(long, value_delimiter = ',')]
    enabled_ignores: Option<Vec<Tool>>,
    /// Force this rule to emit an error. Can be passed multiple times or as a comma-separated list.
    #[arg(long, hide_possible_values = true, value_delimiter = ',')]
    error: Vec<ErrorKind>,
    /// Force this rule to emit a warning. Can be passed multiple times or as a comma-separated list.
    #[arg(long, hide_possible_values = true, value_delimiter = ',')]
    warn: Vec<ErrorKind>,
    /// Do not emit diagnostics for this rule. Can be passed multiple times or as a comma-separated list.
    #[arg(long, hide_possible_values = true, value_delimiter = ',')]
    ignore: Vec<ErrorKind>,
    /// Force this rule to emit an info-level diagnostic. Can be passed multiple times or as a comma-separated list.
    #[arg(long, hide_possible_values = true, value_delimiter = ',')]
    info: Vec<ErrorKind>,
}

impl ConfigOverrideArgs {
    pub fn validate(&self) -> anyhow::Result<()> {
        fn validate_arg(arg_name: &str, paths: Option<&[PathBuf]>) -> anyhow::Result<()> {
            if let Some(paths) = paths {
                for path in paths {
                    validate_path(path).with_context(|| format!("Invalid {arg_name}"))?;
                }
            }
            Ok(())
        }
        validate_arg("--site-package-path", self.site_package_path.as_deref())?;
        validate_arg("--search-path", self.search_path.as_deref())?;
        let ignored_errors = &self.ignore.iter().collect::<HashSet<_>>();
        let warn_errors = &self.warn.iter().collect::<HashSet<_>>();
        let error_errors = self.error.iter().collect::<HashSet<_>>();
        let info_errors = self.info.iter().collect::<HashSet<_>>();
        let error_ignore_conflicts: Vec<_> = error_errors.intersection(ignored_errors).collect();
        if !error_ignore_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --ignore and --error: [{}]",
                display::commas_iter(|| error_ignore_conflicts.iter().map(|&&s| s))
            ));
        }
        let error_warn_conflicts: Vec<_> = error_errors.intersection(warn_errors).collect();
        if !error_warn_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --warn and --error: [{}]",
                display::commas_iter(|| error_warn_conflicts.iter().map(|&&s| s))
            ));
        }
        let ignore_warn_conflicts: Vec<_> = ignored_errors.intersection(warn_errors).collect();
        if !ignore_warn_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --warn and --ignore: [{}]",
                display::commas_iter(|| ignore_warn_conflicts.iter().map(|&&s| s))
            ));
        }
        let error_info_conflicts: Vec<_> = error_errors.intersection(&info_errors).collect();
        if !error_info_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --info and --error: [{}]",
                display::commas_iter(|| error_info_conflicts.iter().map(|&&s| s))
            ));
        }
        let warn_info_conflicts: Vec<_> = warn_errors.intersection(&info_errors).collect();
        if !warn_info_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --info and --warn: [{}]",
                display::commas_iter(|| warn_info_conflicts.iter().map(|&&s| s))
            ));
        }
        let ignore_info_conflicts: Vec<_> = ignored_errors.intersection(&info_errors).collect();
        if !ignore_info_conflicts.is_empty() {
            return Err(anyhow::anyhow!(
                "Error types are specified for both --info and --ignore: [{}]",
                display::commas_iter(|| ignore_info_conflicts.iter().map(|&&s| s))
            ));
        }
        if self.permissive_ignores.is_some() && self.enabled_ignores.is_some() {
            return Err(anyhow::anyhow!(
                "Cannot use both `--permissive-ignores` and `--enabled-ignores`"
            ));
        }
        Ok(())
    }

    pub fn override_config(&self, mut config: ConfigFile) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
        if let Some(x) = &self.python_platform {
            config.python_environment.python_platform = Some(x.clone());
        }
        if let Some(x) = &self.python_version {
            config.python_environment.python_version = Some(*x);
        }
        if let Some(x) = &self.search_path {
            config.search_path_from_args = x.clone();
        }
        if let Some(x) = &self.disable_search_path_heuristics {
            config.disable_search_path_heuristics = *x;
        }
        if let Some(x) = &self.disable_project_excludes_heuristics {
            config.disable_project_excludes_heuristics = *x;
        }
        if let Some(x) = &self.site_package_path {
            config.python_environment.site_package_path = Some(x.clone());
        }

        if self.skip_interpreter_query || config.interpreters.skip_interpreter_query {
            config.interpreters.skip_interpreter_query = true;
            config.interpreters.python_interpreter_path = None;
            config.interpreters.fallback_python_interpreter_name = None;
            config.interpreters.conda_environment = None;
        }
        if let Some(x) = &self.python_interpreter_path {
            config.interpreters.python_interpreter_path = Some(ConfigOrigin::cli(x.clone()));
            config.interpreters.fallback_python_interpreter_name = None;
            config.interpreters.conda_environment = None;
        }
        if let Some(x) = &self.fallback_python_interpreter_name {
            config.interpreters.fallback_python_interpreter_name =
                Some(ConfigOrigin::cli(x.clone()));
            config.interpreters.python_interpreter_path = None;
            config.interpreters.conda_environment = None;
        }
        if let Some(conda_environment) = &self.conda_environment {
            config.interpreters.conda_environment =
                Some(ConfigOrigin::cli(conda_environment.clone()));
            config.interpreters.python_interpreter_path = None;
            config.interpreters.fallback_python_interpreter_name = None;
        }
        if let Some(x) = &self.typeshed_path {
            config.typeshed_path = Some(x.clone());
        }
        if let Some(x) = &self.use_ignore_files {
            config.use_ignore_files = *x;
        }
        if let Some(x) = &self.untyped_def_behavior {
            config.root.untyped_def_behavior = Some(*x);
        }
        match (self.permissive_ignores, &self.enabled_ignores) {
            // Special case: if the underlying config sets enabled-ignores and --permissive-ignores
            // is passed on the command-line, we overwrite enabled-ignores.
            (Some(x), None)
                if config.root.permissive_ignores.is_none()
                    && config.root.enabled_ignores.is_some() =>
            {
                config.root.enabled_ignores = Some(if x {
                    Tool::all()
                } else {
                    Tool::default_enabled()
                });
            }
            // Special case: if the underlying config sets permissive-ignores and --enabled-ignores
            // is passed on the command-line, we disable permissive-ignores and use the specified tools.
            (None, Some(x))
                if config.root.permissive_ignores.is_some()
                    && config.root.enabled_ignores.is_none() =>
            {
                config.root.permissive_ignores = None;
                config.root.enabled_ignores = Some(x.iter().cloned().collect());
            }
            _ => {
                if let Some(x) = self.permissive_ignores {
                    config.root.permissive_ignores = Some(x);
                }
                if let Some(x) = &self.enabled_ignores {
                    config.root.enabled_ignores = Some(x.iter().cloned().collect());
                }
            }
        }
        if let Some(wildcards) = &self.replace_imports_with_any {
            config.root.replace_imports_with_any = Some(
                wildcards
                    .iter()
                    .filter_map(|x| ModuleWildcard::new(x).ok())
                    .collect(),
            );
        }
        if let Some(wildcards) = &self.ignore_missing_imports {
            config.root.ignore_missing_imports = Some(
                wildcards
                    .iter()
                    .filter_map(|x| ModuleWildcard::new(x).ok())
                    .collect(),
            );
        }
        if let Some(x) = &self.ignore_errors_in_generated_code {
            config.root.ignore_errors_in_generated_code = Some(*x);
        }
        if let Some(x) = &self.infer_with_first_use {
            config.root.infer_with_first_use = Some(*x);
        }
        let apply_error_settings = |error_config: &mut ErrorDisplayConfig| {
            for error_kind in &self.error {
                error_config.set_error_severity(*error_kind, Severity::Error);
            }
            for error_kind in &self.warn {
                error_config.set_error_severity(*error_kind, Severity::Warn);
            }
            for error_kind in &self.ignore {
                error_config.set_error_severity(*error_kind, Severity::Ignore);
            }
            for error_kind in &self.info {
                error_config.set_error_severity(*error_kind, Severity::Info);
            }
        };
        let root_errors = config.root.errors.get_or_insert_default();
        apply_error_settings(root_errors);
        for sub_config in config.sub_configs.iter_mut() {
            let sub_config_errors = sub_config.settings.errors.get_or_insert_default();
            apply_error_settings(sub_config_errors);
        }
        let errors = config.configure();
        (ArcId::new(config), errors)
    }

    pub fn disable_project_excludes_heuristics(&self) -> Option<bool> {
        self.disable_project_excludes_heuristics
    }
}
