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
use pyrefly_config::resolve_unconfigured::UnconfiguredOverride;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Globs;
use pyrefly_util::globs::HiddenDirFilter;
use pyrefly_util::includes::Includes;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::config_finder::apply_unconfigured_resolver_if_applicable;
use crate::commands::config_finder::default_config_finder_with_overrides;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::ProjectLayout;
use crate::config::config::SynthesizedPresetReason;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::config::finder::debug_log;

/// Whether (and how) to emit the "no `pyrefly.toml` found" upsell after
/// a CLI check. Computed once during file resolution so the hot path
/// can short-circuit instead of iterating every checked module.
#[derive(Debug, Clone, Copy)]
pub enum UpsellDecision {
    /// Never upsell. Used when an explicit `--config` was provided, or
    /// when a real on-disk config covers the project.
    Skip,
    /// Upsell with this reason. Used in project mode (`pyrefly check`
    /// no file args) where every checked file shares the cwd config —
    /// a single field access suffices, no per-handle walk required.
    Show(SynthesizedPresetReason),
    /// File-args mode without `--config`: files may resolve to distinct
    /// configs via per-file upward search. Caller decides at run time
    /// by walking handles with a short-circuiting same-config check.
    /// Iteration here is bounded by the user's explicit args (not a
    /// project-wide expansion), so the cost stays user-proportional.
    Determine,
}

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

    /// Explicitly set the Pyrefly configuration to use when type checking.
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
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<(ArcId<ConfigFile>, Vec<ConfigError>)> {
    let current_dir = std::env::current_dir().context("cannot identify current dir")?;
    let config_finder = default_config_finder_with_overrides(args.clone(), false, wrapper);
    let config = config_finder.directory(&current_dir).unwrap_or_else(|| {
        // No marker found upward. Run the synthesized config through the
        // same unconfigured resolver wiring that the file-mode path uses
        // (via `DefaultConfigConfigurerWithOverrides`); without this,
        // `pyrefly check` (project mode) and `pyrefly check <file>` (file
        // mode) produce divergent configs in unconfigured repos — the
        // file-mode upsell would never fire for project-mode checks, and
        // a nearby mypy.ini / pyrightconfig.json would be migrated for
        // file mode but ignored here.
        let mut synthesized =
            ConfigFile::init_at_root(&current_dir, &ProjectLayout::new(&current_dir), false);
        apply_unconfigured_resolver_if_applicable(
            &mut synthesized,
            Some(&current_dir),
            UnconfiguredOverride::Auto,
        );
        let (config, errors) = args.override_config(synthesized);
        // Since this is a config we generated, these are likely internal errors.
        debug_log(errors);
        config
    });
    Ok((config, config_finder.errors()))
}

/// Get inputs for a full-project check. We will look for a config file and type-check the project it defines.
///
/// Also returns the `UpsellDecision`: in project mode every checked
/// file shares the single returned config, so the upsell decision is
/// known up front — `Show(reason)` if the config carries a synthesized
/// preset reason, `Skip` otherwise. The caller never needs to walk
/// handles to decide.
fn get_globs_and_config_for_project(
    config: Option<PathBuf>,
    project_excludes: Option<Globs>,
    args: ConfigOverrideArgs,
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
    let (config, mut errors) = match config {
        Some(explicit) => get_explicit_config(&explicit, args),
        None => get_project_config_for_current_dir(args, wrapper)?,
    };
    match &config.source {
        ConfigSource::File(path) => {
            info!("Checking project configured at `{}`", path.display());
        }
        ConfigSource::FailedParse(path) => {
            warn!(
                "Config at `{}` failed to parse, checking with default configuration",
                path.display()
            );
        }
        ConfigSource::PythonToolMarker(path) | ConfigSource::Marker(path) => {
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

    let upsell = match config.synthesized_preset_reason {
        Some(reason) => UpsellDecision::Show(reason),
        None => UpsellDecision::Skip,
    };
    Ok((Box::new(filtered_globs), config_finder, upsell))
}

/// Get inputs for a per-file check. If an explicit config is passed in, we use it; otherwise, we
/// find configs via upward search from each file.
///
/// Also returns the `UpsellDecision`: explicit `--config` always means
/// `Skip` (the loaded config is real and never carries a synthesized
/// reason). Per-file mode defers to `Determine` because each file may
/// hit a different config; the run-time iteration over user-arg
/// handles short-circuits on the first mismatch.
fn get_globs_and_config_for_files(
    config: Option<PathBuf>,
    files_to_check: Globs,
    project_excludes: Option<Globs>,
    args: ConfigOverrideArgs,
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
    let files_to_check = absolutize(files_to_check);
    let args_disable_excludes_heuristics = args.disable_project_excludes_heuristics();
    let get_project_excludes_and_heuristics = move |config: Option<&ConfigFile>| {
        let mut project_excludes = project_excludes.unwrap_or_default();
        let disable = args_disable_excludes_heuristics
            .or_else(|| Some(config?.disable_project_excludes_heuristics))
            .unwrap_or(false);
        if !disable {
            project_excludes.append(ConfigFile::required_project_excludes().globs());
        }
        (project_excludes, !disable)
    };
    let (config_finder, errors, project_excludes, use_heuristics, upsell) = match config {
        Some(explicit) => {
            let (config, errors) = get_explicit_config(&explicit, args);
            let (project_excludes, use_heuristics) =
                get_project_excludes_and_heuristics(Some(&config));
            let config_finder = ConfigFinder::new_constant(config);
            (
                config_finder,
                errors,
                project_excludes,
                use_heuristics,
                UpsellDecision::Skip,
            )
        }
        None => {
            let config_finder = default_config_finder_with_overrides(args, false, wrapper);
            // If there is only one input and one root, we treat config parse errors as fatal,
            // so that `pyrefly check .` exits immediately on an unparsable config, matching the
            // behavior of `pyrefly check` (see get_globs_and_config_for_project).
            let solo_root = if files_to_check.len() == 1 {
                files_to_check.roots().first().cloned()
            } else {
                None
            };
            let (project_excludes, use_heuristics) = get_project_excludes_and_heuristics(None);
            let (config_finder, errors) = if let Some(root) = solo_root {
                // We don't care about the contents of the config, only if we generated any errors while parsing it.
                config_finder.directory(&root);
                let errors = config_finder.errors();
                (config_finder, errors)
            } else {
                (config_finder, Vec::new())
            };
            (
                config_finder,
                errors,
                project_excludes,
                use_heuristics,
                UpsellDecision::Determine,
            )
        }
    };
    add_config_errors(&config_finder, errors)?;
    let hidden_dir_filter = if use_heuristics {
        let roots = files_to_check.roots();
        if roots.is_empty() {
            HiddenDirFilter::All
        } else {
            HiddenDirFilter::RelativeTo(roots)
        }
    } else {
        HiddenDirFilter::Disabled
    };
    let globs = FilteredGlobs::new(files_to_check, project_excludes, None, hidden_dir_filter);
    Ok((Box::new(globs), config_finder, upsell))
}

impl FilesArgs {
    pub fn resolve(
        self,
        config_override: ConfigOverrideArgs,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
        let project_excludes = if let Some(project_excludes) = self.project_excludes {
            Some(absolutize(Globs::new(project_excludes)?))
        } else {
            None
        };
        if self.files.is_empty() {
            get_globs_and_config_for_project(
                self.config,
                project_excludes,
                config_override,
                wrapper,
            )
        } else {
            get_globs_and_config_for_files(
                self.config,
                Globs::new(self.files)?,
                project_excludes,
                config_override,
                wrapper,
            )
        }
    }

    pub fn get(
        files: Vec<String>,
        config: Option<PathBuf>,
        args: ConfigOverrideArgs,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
        FilesArgs {
            files,
            config,
            project_excludes: None,
        }
        .resolve(args, wrapper)
    }
}
