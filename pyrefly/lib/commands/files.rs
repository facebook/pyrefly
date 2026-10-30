/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::environment::script;
use pyrefly_config::resolve_unconfigured::UnconfiguredOverride;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Globs;
use pyrefly_util::globs::HiddenDirFilter;
use pyrefly_util::includes::Includes;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::config_finder::apply_unconfigured_resolver_if_applicable;
use crate::commands::config_finder::default_config_finder_with_overrides;
use crate::commands::config_finder::project_dirs_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigScope;
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
    /// Files to exclude when type checking, in addition to the `project-excludes` of the config
    /// in project mode, or to `--project-excludes` and the default excludes when files are given.
    #[arg(long)]
    extend_project_excludes: Vec<String>,
    /// Skip the PEP 723 scripts that Pyrefly finds in the project or in the given directories.
    /// Files given explicitly are still checked.
    #[arg(long)]
    exclude_scripts: bool,
    /// Check the project of this directory, limited to the files under it. The directory uses the
    /// config that `pyrefly check` finds from it, with the `project-includes` and
    /// `project-excludes` of that config. Repeat the flag to check several directories in one run.
    /// A file belongs to the deepest given directory that contains it.
    #[arg(long, value_name = "DIR", conflicts_with = "files")]
    project_dir: Vec<PathBuf>,

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
    let config = get_project_config_for_dir(&config_finder, &current_dir, &args);
    Ok((config, config_finder.errors()))
}

/// The config that `config_finder` finds for `dir`, or a config synthesized at `dir` if there is
/// none.
fn get_project_config_for_dir(
    config_finder: &ConfigFinder,
    dir: &Path,
    args: &ConfigOverrideArgs,
) -> ArcId<ConfigFile> {
    config_finder.directory(dir).unwrap_or_else(|| {
        // No marker found upward. Run the synthesized config through the
        // same unconfigured resolver wiring that the file-mode path uses
        // (via `DefaultConfigConfigurerWithOverrides`); without this,
        // `pyrefly check` (project mode) and `pyrefly check <file>` (file
        // mode) produce divergent configs in unconfigured repos — the
        // file-mode upsell would never fire for project-mode checks, and
        // a nearby mypy.ini / pyrightconfig.json would be migrated for
        // file mode but ignored here.
        let mut synthesized = ConfigFile::init_at_root(dir, &ProjectLayout::new(dir), false);
        apply_unconfigured_resolver_if_applicable(
            &mut synthesized,
            Some(dir),
            UnconfiguredOverride::Auto,
        );
        let (config, errors) = args.override_config_at(synthesized, Some(dir));
        // Since this is a config we generated, these are likely internal errors.
        debug_log(errors);
        config
    })
}

pub fn get_config_finder_for_snippet(
    config: Option<PathBuf>,
    args: ConfigOverrideArgs,
) -> anyhow::Result<ConfigFinder> {
    let (config, errors) = match config {
        Some(explicit) => get_explicit_config(&explicit, args),
        None => {
            let current_dir = std::env::current_dir().context("cannot identify current dir")?;
            let finder = default_config_finder_with_overrides(args.clone(), false, None);
            match finder.directory(&current_dir) {
                Some(config) => (config, finder.errors()),
                None => args.override_config_at(ConfigFile::default(), Some(&current_dir)),
            }
        }
    };
    let config_finder = ConfigFinder::new_constant(config);
    add_config_errors(&config_finder, errors)?;
    Ok(config_finder)
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
    extend_project_excludes: &Globs,
    args: ConfigOverrideArgs,
    wrapper: Option<ConfigConfigurerWrapper>,
    scope: ConfigScope,
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
                "Config at `{}` failed to parse, checking with auto configuration",
                path.display()
            );
        }
        ConfigSource::PythonToolMarker(path) | ConfigSource::Marker(path) => {
            info!(
                "Found `{}` marking project root, checking root directory with auto configuration",
                path.display(),
            );
        }
        ConfigSource::Synthetic(_) => {
            info!("Checking current directory with auto configuration");
        }
    }
    let current_dir = std::env::current_dir().ok();
    if let Some(project_dir) = config.source.root_from_file().or(current_dir.as_deref())
        && let Some(home_dir) = std::env::home_dir()
        && home_dir.starts_with(project_dir)
        && *config.includes(scope) == ConfigFile::default_project_includes().from_root(project_dir)
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

    let mut filtered_globs = config.get_filtered_globs(project_excludes, scope);
    filtered_globs.add_excludes(extend_project_excludes.globs());
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

/// Get inputs for `--project-dir`. Each directory gets the files and the config that
/// `pyrefly check` gives it, limited to the files under the directory.
fn get_globs_and_config_for_project_dirs(
    config: Option<PathBuf>,
    project_dirs: Vec<PathBuf>,
    project_excludes: Option<Globs>,
    extend_project_excludes: &Globs,
    args: ConfigOverrideArgs,
    wrapper: Option<ConfigConfigurerWrapper>,
    scope: ConfigScope,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
    let mut errors = Vec::new();
    let explicit_config = config.map(|path| {
        let (config, explicit_errors) = get_explicit_config(&path, args.clone());
        errors.extend(explicit_errors);
        config
    });
    let finder = default_config_finder_with_overrides(args.clone(), false, wrapper.clone());
    let mut dirs = Vec::new();
    for dir in project_dirs {
        let dir = dir.absolutize();
        if !dir.is_dir() {
            return Err(anyhow::anyhow!(
                "`--project-dir` path `{}` is not a directory",
                dir.display()
            ));
        }
        let config = explicit_config
            .dupe()
            .unwrap_or_else(|| get_project_config_for_dir(&finder, &dir, &args));
        if config.build_system.is_some() {
            return Err(anyhow::anyhow!(
                "Cannot run build system in project mode, you must provide files to check"
            ));
        }
        debug!(
            "Project directory `{}` has config: {}",
            dir.display(),
            config
        );
        dirs.push((dir, config));
    }
    errors.extend(finder.errors());
    // A path sorts after its ancestors, so the reverse order puts the deepest directory first.
    dirs.sort_by(|(a, _), (b, _)| b.cmp(a));
    dirs.dedup_by(|(a, _), (b, _)| a == b);

    let mut includes: Vec<(ArcId<ConfigFile>, FilteredGlobs)> = Vec::new();
    for (_, config) in &dirs {
        if !includes.iter().any(|(seen, _)| seen == config) {
            let mut globs = config.get_filtered_globs(project_excludes.clone(), scope);
            globs.add_excludes(extend_project_excludes.globs());
            errors.extend(globs.errors().into_iter().map(ConfigError::warn));
            includes.push((config.dupe(), globs));
        }
    }
    let upsell = match dirs
        .iter()
        .find_map(|(_, config)| config.synthesized_preset_reason)
    {
        Some(reason) => UpsellDecision::Show(reason),
        None => UpsellDecision::Skip,
    };
    let config_finder = project_dirs_config_finder(dirs.clone(), args, wrapper);
    add_config_errors(&config_finder, errors)?;
    Ok((
        Box::new(ProjectDirs { dirs, includes }),
        config_finder,
        upsell,
    ))
}

/// The files of `--project-dir`. A file belongs to the deepest given directory that contains it,
/// and is checked if the config of that directory includes it.
struct ProjectDirs {
    /// Each given directory and its config, deepest first.
    dirs: Vec<(PathBuf, ArcId<ConfigFile>)>,
    /// The includes of each distinct config in `dirs`.
    includes: Vec<(ArcId<ConfigFile>, FilteredGlobs)>,
}

impl ProjectDirs {
    /// The config of the directory that `path` belongs to.
    fn config(&self, path: &Path) -> Option<&ArcId<ConfigFile>> {
        self.dirs
            .iter()
            .find(|(dir, _)| path.starts_with(dir))
            .map(|(_, config)| config)
    }

    /// The includes of the config of the directory that `path` belongs to.
    fn includes(&self, path: &Path) -> Option<&FilteredGlobs> {
        let config = self.config(path)?;
        self.includes
            .iter()
            .find(|(seen, _)| seen == config)
            .map(|(_, includes)| includes)
    }
}

impl Includes for ProjectDirs {
    fn roots(&self) -> Vec<PathBuf> {
        self.dirs.iter().map(|(dir, _)| dir.clone()).collect()
    }

    fn files_iter(&self) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf> + '_>> {
        // Each config is walked once. A file is kept only by the config of its directory, so no
        // file is returned twice.
        let mut files = Vec::new();
        for (config, includes) in &self.includes {
            files.extend(
                includes
                    .files_iter()?
                    .filter(|path| self.config(path) == Some(config)),
            );
        }
        files.sort();
        Ok(Box::new(files.into_iter()))
    }

    fn covers(&self, path: &Path) -> bool {
        self.includes(path)
            .is_some_and(|includes| includes.covers(path))
    }

    fn covers_ignoring_excludes(&self, path: &Path) -> bool {
        self.includes(path)
            .is_some_and(|includes| includes.covers_ignoring_excludes(path))
    }

    fn errors(&mut self) -> Vec<anyhow::Error> {
        self.includes
            .iter_mut()
            .flat_map(|(_, includes)| includes.errors())
            .collect()
    }
}

/// Get inputs for a per-file check. If an explicit config is passed in, we use it; otherwise, we
/// find configs via upward search from each file. `files_to_check` must be absolute.
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
    extend_project_excludes: &Globs,
    args: ConfigOverrideArgs,
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
    let args_disable_excludes_heuristics = args.disable_project_excludes_heuristics();
    let get_project_excludes_and_heuristics = move |config: Option<&ConfigFile>| {
        let mut project_excludes = project_excludes.unwrap_or_default();
        project_excludes.append(extend_project_excludes.globs());
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

/// Skips the PEP 723 scripts among the files of `inner`, except the `explicit_files` that the user
/// names, which `inner` yields as they are. A script runs in its own environment, so a tool such
/// as `uv check` checks scripts separately from the project.
struct ExcludeScripts {
    inner: Box<dyn Includes>,
    explicit_files: SmallSet<PathBuf>,
}

impl ExcludeScripts {
    fn keeps(&self, path: &Path) -> bool {
        self.explicit_files.contains(path)
            || path.extension().is_none_or(|extension| extension != "py")
            || fs::read_to_string(path).map_or(true, |source| script::metadata(&source).is_none())
    }
}

impl Includes for ExcludeScripts {
    fn roots(&self) -> Vec<PathBuf> {
        self.inner.roots()
    }

    fn files_iter(&self) -> anyhow::Result<Box<dyn Iterator<Item = PathBuf> + '_>> {
        Ok(Box::new(
            self.inner.files_iter()?.filter(|path| self.keeps(path)),
        ))
    }

    fn covers(&self, path: &Path) -> bool {
        self.inner.covers(path) && self.keeps(path)
    }

    fn covers_ignoring_excludes(&self, path: &Path) -> bool {
        self.inner.covers_ignoring_excludes(path)
    }

    fn errors(&mut self) -> Vec<anyhow::Error> {
        self.inner.errors()
    }
}

impl FilesArgs {
    /// The file arguments as given on the command line. Empty means project-checking mode.
    pub fn files(&self) -> &[String] {
        &self.files
    }

    pub fn resolve(
        self,
        config_override: ConfigOverrideArgs,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
        self.resolve_scoped(config_override, wrapper, ConfigScope::Default)
    }

    /// [`FilesArgs::resolve`], reading the project-mode globs from `scope`.
    pub fn resolve_scoped(
        self,
        config_override: ConfigOverrideArgs,
        wrapper: Option<ConfigConfigurerWrapper>,
        scope: ConfigScope,
    ) -> anyhow::Result<(Box<dyn Includes>, ConfigFinder, UpsellDecision)> {
        let project_excludes = if let Some(project_excludes) = self.project_excludes {
            Some(absolutize(Globs::new(project_excludes)?))
        } else {
            None
        };
        let extend_project_excludes = absolutize(Globs::new(self.extend_project_excludes)?);
        let files_to_check = absolutize(Globs::new(self.files)?);
        let explicit_files = files_to_check
            .explicit_files()
            .map(Path::to_path_buf)
            .collect();
        let (includes, config_finder, upsell) = if !self.project_dir.is_empty() {
            get_globs_and_config_for_project_dirs(
                self.config,
                self.project_dir,
                project_excludes,
                &extend_project_excludes,
                config_override,
                wrapper,
                scope,
            )?
        } else if files_to_check.is_empty() {
            get_globs_and_config_for_project(
                self.config,
                project_excludes,
                &extend_project_excludes,
                config_override,
                wrapper,
                scope,
            )?
        } else {
            // File mode bypasses the config's globs, so `scope` is intentionally unused.
            get_globs_and_config_for_files(
                self.config,
                files_to_check,
                project_excludes,
                &extend_project_excludes,
                config_override,
                wrapper,
            )?
        };
        let includes = if self.exclude_scripts {
            Box::new(ExcludeScripts {
                inner: includes,
                explicit_files,
            })
        } else {
            includes
        };
        Ok((includes, config_finder, upsell))
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
            extend_project_excludes: Vec::new(),
            exclude_scripts: false,
            project_dir: Vec::new(),
        }
        .resolve(args, wrapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclude_scripts() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let script = root.join("script.py");
        let module = root.join("module.py");
        let stub = root.join("stub.pyi");
        let metadata = "# /// script\n# dependencies = [\"six\"]\n# ///\n";
        fs::write(&script, metadata).unwrap();
        fs::write(&module, "x = 1\n").unwrap();
        fs::write(&stub, metadata).unwrap();
        let files = |explicit_files: Vec<PathBuf>| {
            let includes = ExcludeScripts {
                inner: Box::new(FilteredGlobs::new(
                    Globs::new(vec![root.to_string_lossy().into_owned()]).unwrap(),
                    Globs::empty(),
                    None,
                    HiddenDirFilter::Disabled,
                )),
                explicit_files: explicit_files.into_iter().collect(),
            };
            let mut files = includes.files_iter().unwrap().collect::<Vec<_>>();
            files.sort();
            files
        };
        assert_eq!(
            files(Vec::new()),
            vec![module.clone(), stub.clone()],
            "a script found in a directory is skipped, and a stub is never a script",
        );
        assert_eq!(
            files(vec![script.clone()]),
            vec![module, script, stub],
            "an explicitly given script is checked",
        );
    }
}
