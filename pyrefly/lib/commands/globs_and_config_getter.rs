/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use dupe::Dupe;
use path_absolutize::Absolutize;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Globs;
use tracing::debug;
use tracing::info;

use crate::commands::config_args::ConfigOverrideArgs;
use crate::commands::config_finder::standard_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::ProjectLayout;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::config::finder::debug_log;
use crate::error::kind::Severity;

fn config_finder(args: ConfigOverrideArgs) -> ConfigFinder {
    standard_config_finder(Arc::new(move |_, x| args.override_config(x)))
}

fn absolutize(globs: Globs) -> anyhow::Result<Globs> {
    Ok(globs.from_root(PathBuf::new().absolutize()?.as_ref()))
}

fn get_explicit_config(
    path: &Path,
    args: &ConfigOverrideArgs,
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

/// Get inputs for a full-project check. We will look for a config file and type-check the project it defines.
fn get_globs_and_config_for_project(
    config: Option<PathBuf>,
    project_excludes: Option<Globs>,
    args: &ConfigOverrideArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    let (config, errors) = match config {
        Some(explicit) => get_explicit_config(&explicit, args),
        None => {
            let current_dir = std::env::current_dir().context("cannot identify current dir")?;
            let config_finder = config_finder(args.clone());
            let config = config_finder.directory(&current_dir).unwrap_or_else(|| {
                let (config, errors) = args.override_config(ConfigFile::init_at_root(
                    &current_dir,
                    &ProjectLayout::new(&current_dir),
                ));
                // Since this is a config we generated, these are likely internal errors.
                debug_log(errors);
                config
            });
            (config, config_finder.errors())
        }
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

    // We want our config_finder to never actually
    let config_finder = ConfigFinder::new_constant(config.dupe());
    add_config_errors(&config_finder, errors)?;

    debug!("Config is: {}", config);

    Ok((config.get_filtered_globs(project_excludes), config_finder))
}

/// Get inputs for a per-file check. If an explicit config is passed in, we use it; otherwise, we
/// find configs via upward search from each file.
fn get_globs_and_config_for_files(
    config: Option<PathBuf>,
    files_to_check: Globs,
    project_excludes: Option<Globs>,
    args: &ConfigOverrideArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    let project_excludes = project_excludes.unwrap_or_else(ConfigFile::default_project_excludes);
    let files_to_check = absolutize(files_to_check)?;
    let (config_finder, errors) = match config {
        Some(explicit) => {
            let (config, errors) = get_explicit_config(&explicit, args);
            let config_finder = ConfigFinder::new_constant(config);
            (config_finder, errors)
        }
        None => {
            let config_finder = config_finder(args.clone());
            // If there is only one input and one root, we treat config parse errors as fatal,
            // so that `pyrefly check .` exits immediately on an unparsable config, matching the
            // behavior of `pyrefly check` (see get_globs_and_config_for_project).
            let solo_root = if files_to_check.len() == 1 {
                files_to_check.roots().first().cloned()
            } else {
                None
            };
            if let Some(root) = solo_root {
                // We don't care about the contents of the config, only if we generated any errors while parsing it.
                config_finder.directory(&root);
                let errors = config_finder.errors();
                (config_finder, errors)
            } else {
                (config_finder, Vec::new())
            }
        }
    };
    add_config_errors(&config_finder, errors)?;
    Ok((
        FilteredGlobs::new(files_to_check, project_excludes),
        config_finder,
    ))
}

pub fn get(
    files: Vec<String>,
    project_excludes: Option<Vec<String>>,
    config: Option<PathBuf>,
    args: &mut ConfigOverrideArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    args.absolute_search_path();
    args.validate()?;
    let project_excludes = if let Some(project_excludes) = project_excludes {
        Some(absolutize(Globs::new(project_excludes))?)
    } else {
        None
    };
    if files.is_empty() {
        get_globs_and_config_for_project(config, project_excludes, args)
    } else {
        get_globs_and_config_for_files(config, Globs::new(files), project_excludes, args)
    }
}
