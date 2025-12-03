/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Globs;
use pyrefly_util::includes::Includes;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::commands::config_finder::default_config_finder_with_overrides;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::ProjectLayout;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::config::finder::debug_log;

/// Arguments regarding which files to pick.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct FilesArgs {
    /// Files to check (glob supported).
    /// If no file is specified, switch to project-checking mode where the files to
    /// check are determined from the closest configuration file.
    /// When supplied, `project_excludes` in any config files loaded for these files to check
    /// are ignored, and we use the default excludes unless overridden with the `--project-excludes` flag.
    files: Vec<String>,
    /// Files to exclude when type checking.
    #[arg(long)]
    project_excludes: Option<Vec<String>>,

    /// Explicitly set the Pyrefly configuration to use when type checking or starting a language server.
    /// In "single-file checking mode," this config is applied to all files being checked, ignoring
    /// the config's `project_includes` and `project_excludes` and ignoring any config-finding approach
    /// that would otherwise be used.
    /// When not set, Pyrefly will perform an upward-filesystem-walk approach to find the nearest
    /// pyrefly.toml or pyproject.toml with `tool.pyrefly` section'. If no config is found, Pyrefly exits with error.
    /// If both a pyrefly.toml and valid pyproject.toml are found, pyrefly.toml takes precedence.
    #[arg(long, short, value_name = "FILE", env = clap_env("CONFIG"))]
    config: Option<PathBuf>,
}

fn absolutize(globs: Globs) -> Globs {
    globs.from_root(&PathBuf::new().absolutize())
}

fn get_explicit_config(
    path: &Path,
    args: ConfigOverrideArgs,
) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
    let (file_config, parse_errors) = ConfigFile::from_file(path);
    let (config, validation_errors) = args.override_config(file_config);
    (
        config,
        parse_errors.into_iter().chain(validation_errors).collect(),
    )
}

fn add_config_errors(config_finder: &ConfigFinder, errors: Vec<ConfigError>) -> anyhow::Result<()> {
    if errors.iter().any(|e| e.severity() == Severity::Error) {
        for e in errors {
            e.print();
        }
        Err(anyhow::anyhow!("Fatal configuration error"))
    } else {
        config_finder.add_errors(errors);
        Ok(())
    }
}

/// Gets a project config for the current directory, overriding with the given
/// [`ConfigOverrideArgs`].
///
/// This does not do any glob processing like
/// [`get_globs_and_config_for_project`], which can use the given `args` and found
/// config to determine the right globs to check. It also does not block the use of
/// a config with a build system in project type checking mode, but should be done
/// by [`FilesArgs::resolve`].
pub fn get_project_config_for_current_dir(
    args: ConfigOverrideArgs,
) -> anyhow::Result<(ArcId<ConfigFile>, Vec<ConfigError>)> {
    let current_dir = std::env::current_dir().context("cannot identify current dir")?;
    let config_finder = default_config_finder_with_overrides(args.clone(), false);
    let config = config_finder.directory(&current_dir).unwrap_or_else(|| {
        let (config, errors) = args.override_config(ConfigFile::init_at_root(
            &current_dir,
            &ProjectLayout::new(&current_dir),
            false,
        ));
        // Since this is a config we generated, these are likely internal errors.
        debug_log(errors);
        config
    });
    Ok((config, config_finder.errors()))
}

/// Get inputs for a full-project check. We will look for a config file and type-check the project it defines.
fn get_globs_and_config_for_project(
    config: Option<PathBuf>,
    project_excludes: Option<Globs>,
    args: ConfigOverrideArgs,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder)> {
    let (config, mut errors) = match config {
        Some(explicit) => get_explicit_config(&explicit, args),
        None => get_project_config_for_current_dir(args)?,
    };
    match &config.source {
        ConfigSource::File(path) => {
            info!("Checking project configured at `{}`", path.display());
        }
        ConfigSource::Marker(path) => {
            info!(
                "Found `{}` marking project root, checking root directory with default configuration",
                path.display(),
            );
        }
        ConfigSource::Synthetic => {
            info!("Checking current directory with default configuration");
        }
    }
    let current_dir = std::env::current_dir().ok();
    if let Some(project_dir) = config.source.root().or(current_dir.as_deref())
        && let Some(home_dir) = std::env::home_dir()
        && home_dir.starts_with(project_dir)
        && config.project_includes == ConfigFile::default_project_includes().from_root(project_dir)
    {
        // Trying to type-check your entire home directory doesn't usually end well.
        warn!(
            "Pyrefly is checking everything under `{}`. This may take a while...",
            project_dir.display()
        );
    }

    if config.build_system.is_some() {
        return Err(anyhow::anyhow!(
            "Cannot run build system in project mode, you must provide files to check"
        ));
    }

    let config_finder = ConfigFinder::new_constant(config.dupe());

    debug!("Config is: {}", config);

    let mut filtered_globs = config.get_filtered_globs(project_excludes);
    filtered_globs
        .errors()
        .into_iter()
        .map(ConfigError::warn)
        .for_each(|e| errors.push(e));

    add_config_errors(&config_finder, errors)?;

    Ok((Box::new(filtered_globs), config_finder))
}

/// Get inputs for a per-file check. If an explicit config is passed in, we use it; otherwise, we
/// find configs via upward search from each file.
fn get_globs_and_config_for_files(
    config: Option<PathBuf>,
    files_to_check: Globs,
    project_excludes: Option<Globs>,
    args: ConfigOverrideArgs,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder)> {
    let files_to_check = absolutize(files_to_check);
    let args_disable_excludes_heuristics = args.disable_project_excludes_heuristics();
    let get_project_excludes = move |config: Option<&ConfigFile>| {
        let mut project_excludes = project_excludes.unwrap_or_default();
        if !args_disable_excludes_heuristics
            .or_else(|| Some(config?.disable_project_excludes_heuristics))
            .unwrap_or(false)
        {
            project_excludes.append(ConfigFile::required_project_excludes().globs());
        }
        project_excludes
    };
    let (config_finder, errors, project_excludes) = match config {
        Some(explicit) => {
            let (config, errors) = get_explicit_config(&explicit, args);
            let project_excludes = get_project_excludes(Some(&config));
            let config_finder = ConfigFinder::new_constant(config);
            (config_finder, errors, project_excludes)
        }
        None => {
            let config_finder = default_config_finder_with_overrides(args, false);
            // If there is only one input and one root, we treat config parse errors as fatal,
            // so that `pyrefly check .` exits immediately on an unparsable config, matching the
            // behavior of `pyrefly check` (see get_globs_and_config_for_project).
            let solo_root = if files_to_check.len() == 1 {
                files_to_check.roots().first().cloned()
            } else {
                None
            };
            let project_excludes = get_project_excludes(None);
            if let Some(root) = solo_root {
                // We don't care about the contents of the config, only if we generated any errors while parsing it.
                config_finder.directory(&root);
                let errors = config_finder.errors();
                (config_finder, errors, project_excludes)
            } else {
                (config_finder, Vec::new(), project_excludes)
            }
        }
    };
    add_config_errors(&config_finder, errors)?;
    Ok((
        Box::new(FilteredGlobs::new(files_to_check, project_excludes, None)),
        config_finder,
    ))
}

impl FilesArgs {
    pub fn resolve(
        self,
        config_override: ConfigOverrideArgs,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder)> {
        let project_excludes = if let Some(project_excludes) = self.project_excludes {
            Some(absolutize(Globs::new(project_excludes)?))
        } else {
            None
        };
        if self.files.is_empty() {
            get_globs_and_config_for_project(self.config, project_excludes, config_override)
        } else {
            get_globs_and_config_for_files(
                self.config,
                Globs::new(self.files)?,
                project_excludes,
                config_override,
            )
        }
    }

    pub fn get(
        files: Vec<String>,
        config: Option<PathBuf>,
        args: ConfigOverrideArgs,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder)> {
        FilesArgs {
            files,
            config,
            project_excludes: None,
        }
        .resolve(args)
    }
}
