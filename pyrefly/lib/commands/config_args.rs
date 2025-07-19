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
use path_absolutize::Absolutize;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::display;

use crate::config::base::UntypedDefBehavior;
use crate::config::config::ConfigFile;
use crate::config::config::validate_path;
use crate::config::error::ErrorDisplayConfig;
use crate::config::finder::ConfigError;
use crate::config::util::ConfigOrigin;
use crate::error::kind::ErrorKind;
use crate::error::kind::Severity;
use crate::module::wildcard::ModuleWildcard;

/// config overrides
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone, Default)]
pub struct ConfigOverrideArgs {
    /// The list of directories where imports are imported from, including
    /// type checked files.
    #[arg(long, env = clap_env("SEARCH_PATH"))]
    search_path: Option<Vec<PathBuf>>,

    /// The Python version any `sys.version` checks should evaluate against.
    #[arg(long, env = clap_env("PYTHON_VERSION"))]
    python_version: Option<PythonVersion>,

    /// The platform any `sys.platform` checks should evaluate against.
    #[arg(long, env = clap_env("PLATFORM"))]
    python_platform: Option<PythonPlatform>,

    /// Directories containing third-party package imports, searched
    /// after first checking `search_path` and `typeshed`.
    #[arg(long, env = clap_env("SITE_PACKAGE_PATH"))]
    site_package_path: Option<Vec<PathBuf>>,

    /// Use a specific Conda environment to query Python environment information,
    /// even if it isn't activated.
    #[arg(long, env = clap_env("CONDA_ENVIRONMENT"), group = "env_source")]
    conda_environment: Option<String>,

    /// The Python executable that will be queried for `python_version`
    /// `python_platform`, or `site_package_path` if any of the values are missing.
    #[arg(long, env = clap_env("PYTHON_INTERPRETER"), value_name = "EXE_PATH", group = "env_source")]
    python_interpreter: Option<PathBuf>,

    /// Skip doing any automatic querying for `python-interpreter` or `conda-environment`
    #[arg(long, env = clap_env("SKIP_INTERPRETER_QUERY"), group = "env_source")]
    skip_interpreter_query: bool,

    /// Override the bundled typeshed with a custom path.
    #[arg(long, env = clap_env("TYPESHED_PATH"))]
    typeshed_path: Option<PathBuf>,

    /// Whether to search imports in `site-package-path` that do not have a `py.typed` file unconditionally.
    #[arg(long, env = clap_env("USE_UNTYPED_IMPORTS"))]
    use_untyped_imports: Option<bool>,
    /// Replace specified imports with typing.Any, suppressing related import errors even if the module is found.
    #[arg(long, env = clap_env("REPLACE_IMPORTS_WITH_ANY"))]
    replace_imports_with_any: Option<Vec<String>>,
    /// Ignore missing source packages when only type stubs are available, allowing imports to proceed without source validation.
    #[arg(long, env = clap_env("IGNORE_MISSING_SOURCE"))]
    ignore_missing_source: Option<bool>,
    /// Whether to ignore type errors in generated code.
    #[arg(long, env = clap_env("IGNORE_ERRORS_IN_GENERATED_CODE"))]
    ignore_errors_in_generated_code: Option<bool>,
    /// Controls how Pyrefly analyzes function definitions that lack type annotations on parameters and return values.
    #[arg(long, env = clap_env("UNTYPED_DEF_BEHAVIOR"))]
    untyped_def_behavior: Option<UntypedDefBehavior>,
    /// Whether Pyrefly will respect ignore statements for other tools, e.g. `# mypy: ignore`.
    #[arg(long, env = clap_env("PERMISSIVE_IGNORES"))]
    permissive_ignores: Option<bool>,
    /// Force this rule to emit an error. Can be used multiple times.
    #[arg(long, env = clap_env("ERROR"), hide_possible_values = true)]
    error: Vec<ErrorKind>,
    /// Force this rule to emit a warning. Can be used multiple times.
    #[arg(long, env = clap_env("WARN"), hide_possible_values = true)]
    warn: Vec<ErrorKind>,
    /// Do not emit diagnostics for this rule. Can be used multiple times.
    #[arg(long, env = clap_env("IGNORE"), hide_possible_values = true)]
    ignore: Vec<ErrorKind>,
}

impl ConfigOverrideArgs {
    pub fn absolute_search_path(&mut self) {
        if let Some(paths) = self.search_path.as_mut() {
            for x in paths.iter_mut() {
                if let Ok(v) = x.absolutize() {
                    *x = v.into_owned();
                }
            }
        }
    }

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
        if let Some(x) = &self.site_package_path {
            config.python_environment.site_package_path = Some(x.clone());
        }

        if self.skip_interpreter_query || config.interpreters.skip_interpreter_query {
            config.interpreters.skip_interpreter_query = true;
            config.interpreters.python_interpreter = None;
            config.interpreters.conda_environment = None;
        }
        if let Some(conda_environment) = &self.conda_environment {
            config.interpreters.conda_environment =
                Some(ConfigOrigin::cli(conda_environment.clone()));
            config.interpreters.python_interpreter = None;
        }
        if let Some(x) = &self.typeshed_path {
            config.typeshed_path = Some(x.clone());
        }
        if let Some(x) = &self.python_interpreter {
            config.interpreters.python_interpreter = Some(ConfigOrigin::cli(x.clone()));
            config.interpreters.conda_environment = None;
        }
        if let Some(x) = &self.use_untyped_imports {
            config.use_untyped_imports = *x;
        }
        if let Some(x) = &self.ignore_missing_source {
            config.ignore_missing_source = *x;
        }
        if let Some(x) = &self.untyped_def_behavior {
            config.root.untyped_def_behavior = Some(*x);
        }
        if let Some(x) = self.permissive_ignores {
            config.root.permissive_ignores = Some(x);
        }
        if let Some(wildcards) = &self.replace_imports_with_any {
            config.root.replace_imports_with_any = Some(
                wildcards
                    .iter()
                    .filter_map(|x| ModuleWildcard::new(x).ok())
                    .collect(),
            );
        }
        if let Some(x) = &self.ignore_errors_in_generated_code {
            config.root.ignore_errors_in_generated_code = Some(*x);
        }
        let apply_error_settings = |error_config: &mut ErrorDisplayConfig| {
            for error_kind in &self.error {
                error_config.with_error_setting(*error_kind, Severity::Error);
            }
            for error_kind in &self.warn {
                error_config.with_error_setting(*error_kind, Severity::Warn);
            }
            for error_kind in &self.ignore {
                error_config.with_error_setting(*error_kind, Severity::Ignore);
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
}
