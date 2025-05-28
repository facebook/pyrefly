/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::backtrace::Backtrace;
use std::env::args_os;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use clap::Subcommand;
use dupe::Dupe;
use library::ConfigFile;
use library::ConfigSource;
use library::ModulePath;
use library::ProjectLayout;
use library::finder::ConfigFinder;
use library::finder::debug_log;
use library::run::BuckCheckArgs;
use library::run::CheckArgs;
use library::run::CommandExitStatus;
use library::run::CommonGlobalArgs;
use library::run::InitArgs;
use library::run::LspArgs;
use library::standard_config_finder;
use path_absolutize::Absolutize;
use pyrefly::library::library::library::library;
use pyrefly::library::library::library::library::finder::ConfigError;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::args::get_args_expanded;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::globs::Globs;
use pyrefly_util::watcher::Watcher;
use starlark_map::small_map::SmallMap;
use tracing::debug;
use tracing::info;

// fbcode likes to set its own allocator in fbcode.default_allocator
// So when we set our own allocator, buck build buck2 or buck2 build buck2 often breaks.
// Making jemalloc the default only when we do a cargo build.
#[global_allocator]
#[cfg(all(any(target_os = "linux", target_os = "macos"), not(fbcode_build)))]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Debug, Parser)]
#[command(name = "pyrefly")]
#[command(about = "Next generation of Pyre type checker", long_about = None)]
#[command(version)]
struct Args {
    /// Set this to true to run profiling of fast jobs.
    /// Will run the command repeatedly.
    #[clap(long = "profiling", global = true, hide = true, env = clap_env("PROFILING"))]
    profiling: bool,

    #[clap(flatten)]
    common: CommonGlobalArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Parser)]
struct FullCheckArgs {
    /// Files to check (glob supported).
    /// If no file is specified, switch to project-checking mode where the files to
    /// check are determined from the closest configuration file.
    /// When supplied, `project_excludes` in any config files loaded for these files to check
    /// are ignored, and we use the default excludes unless overridden with the `--project-excludes` flag.
    files: Vec<String>,
    /// Files to exclude when type checking.
    #[clap(long, env = clap_env("PROJECT_EXCLUDES"))]
    project_excludes: Option<Vec<String>>,
    /// Watch for file changes and re-check them.
    #[clap(long, env = clap_env("WATCH"), conflicts_with = "check_all")]
    watch: bool,

    /// Explicitly set the Pyre configuration to use when type checking or starting a language server.
    /// In "single-file checking mode," this config is applied to all files being checked, ignoring
    /// the config's `project_includes` and `project_excludes` and ignoring any config-finding approach
    /// that would otherwise be used.
    /// When not set, Pyre will perform an upward-filesystem-walk approach to find the nearest
    /// pyrefly.toml or pyproject.toml with `tool.pyre` section'. If no config is found, Pyre exits with error.
    /// If both a pyrefly.toml and valid pyproject.toml are found, pyrefly.toml takes precedence.
    #[clap(long, short, env = clap_env("CONFIG"))]
    config: Option<PathBuf>,

    #[clap(flatten)]
    args: CheckArgs,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    /// Full type checking on a file or a project
    Check(FullCheckArgs),

    /// Dump info about pyrefly's configuration. Use by replacing `check` with `dump-config` in your pyrefly invocation.
    DumpConfig(FullCheckArgs),

    /// Entry point for Buck integration
    BuckCheck(BuckCheckArgs),

    /// Initialize a new pyrefly config in the given directory,
    /// or migrate an existing mypy or pyright config to pyrefly.
    Init(InitArgs),

    /// Start an LSP server
    Lsp(LspArgs),
}

fn exit_on_panic() {
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("Thread panicked, shutting down: {}", info);
        eprintln!("Backtrace:\n{}", Backtrace::force_capture());
        std::process::exit(1);
    }));
}

async fn run_check(
    args: library::run::CheckArgs,
    watch: bool,
    files_to_check: FilteredGlobs,
    config_finder: ConfigFinder,
    allow_forget: bool,
) -> anyhow::Result<CommandExitStatus> {
    if watch {
        let watcher = Watcher::notify(&files_to_check.roots())?;
        args.run_watch(watcher, files_to_check, config_finder)
            .await?;
        Ok(CommandExitStatus::Success)
    } else {
        args.run_once(files_to_check, config_finder, allow_forget)
    }
}

fn config_finder(args: library::run::CheckArgs) -> ConfigFinder {
    standard_config_finder(Arc::new(move |_, x| args.override_config(x)))
}

fn absolutize(globs: Globs) -> anyhow::Result<Globs> {
    Ok(globs.from_root(PathBuf::new().absolutize()?.as_ref()))
}

fn get_explicit_config(
    path: &Path,
    args: &library::run::CheckArgs,
) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
    let (file_config, parse_errors) = ConfigFile::from_file(path);
    let (config, validation_errors) = args.override_config(file_config);
    (
        config,
        parse_errors.into_iter().chain(validation_errors).collect(),
    )
}

/// Get inputs for a full-project check. We will look for a config file and type-check the project it defines.
fn get_globs_and_config_for_project(
    config: Option<PathBuf>,
    project_excludes: Option<Globs>,
    args: &library::run::CheckArgs,
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
    config_finder.add_errors(errors);

    debug!("Config is: {}", config);

    Ok((config.get_filtered_globs(project_excludes), config_finder))
}

/// Get inputs for a per-file check. If an explicit config is passed in, we use it; otherwise, we
/// find configs via upward search from each file.
fn get_globs_and_config_for_files(
    config: Option<PathBuf>,
    files_to_check: Globs,
    project_excludes: Option<Globs>,
    args: &library::run::CheckArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    let project_excludes = project_excludes.unwrap_or_else(ConfigFile::default_project_excludes);
    let files_to_check = absolutize(files_to_check)?;
    let config_finder = match config {
        Some(explicit) => {
            let (config, errors) = get_explicit_config(&explicit, args);
            let config_finder = ConfigFinder::new_constant(config);
            config_finder.add_errors(errors);
            config_finder
        }
        None => config_finder(args.clone()),
    };
    Ok((
        FilteredGlobs::new(files_to_check, project_excludes),
        config_finder,
    ))
}

fn get_globs_and_config(
    files: Vec<String>,
    project_excludes: Option<Vec<String>>,
    config: Option<PathBuf>,
    args: &mut library::run::CheckArgs,
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

async fn run_command(command: Command, allow_forget: bool) -> anyhow::Result<CommandExitStatus> {
    match command {
        Command::Check(FullCheckArgs {
            files,
            project_excludes,
            watch,
            config,
            mut args,
        }) => {
            let (files_to_check, config_finder) =
                get_globs_and_config(files, project_excludes, config, &mut args)?;
            run_check(args, watch, files_to_check, config_finder, allow_forget).await
        }
        Command::BuckCheck(args) => args.run(),
        Command::Lsp(args) => args.run(),
        Command::Init(args) => args.run(),
        // We intentionally make DumpConfig take the same arguments as Check so that dumping the
        // config is as easy as changing the command name.
        Command::DumpConfig(FullCheckArgs {
            files,
            project_excludes,
            config,
            mut args,
            ..
        }) => {
            let mut configs_to_files: SmallMap<ArcId<ConfigFile>, Vec<ModulePath>> =
                SmallMap::new();
            let (files_to_check, config_finder) =
                get_globs_and_config(files, project_excludes, config, &mut args)?;
            let mut handles = args
                .get_handles(files_to_check, &config_finder)?
                .into_iter()
                .map(|(handle, _)| handle)
                .collect::<Vec<_>>();
            handles.sort_by(|a, b| a.path().cmp(b.path()));
            for handle in handles {
                let path = handle.path();
                let config = config_finder.python_file(handle.module(), path);
                configs_to_files
                    .entry(config)
                    .or_default()
                    .push(path.clone());
            }
            for error in config_finder.errors() {
                error.print();
            }
            for (config, files) in configs_to_files.into_iter() {
                match &config.source {
                    ConfigSource::Synthetic => {
                        println!("Default configuration");
                    }
                    ConfigSource::Marker(path) => {
                        println!(
                            "Default configuration for project root marked by `{}`",
                            path.display()
                        );
                    }
                    ConfigSource::File(path) => {
                        println!("Configuration at `{}`", path.display());
                    }
                }
                println!("  Covered files:");
                for (i, fi) in files.iter().enumerate() {
                    if i < 10 {
                        println!("    {fi}");
                    } else {
                        println!("    ...and {} more", files.len() - 10);
                        break;
                    }
                }
                for path_part in config.structured_import_lookup_path() {
                    if !path_part.is_empty() {
                        println!("  {path_part}");
                    }
                }
            }
            Ok(CommandExitStatus::Success)
        }
    }
}

/// Run based on the command line arguments.
async fn run() -> anyhow::Result<ExitCode> {
    let args = Args::parse_from(get_args_expanded(args_os())?);
    args.common.init();
    if args.profiling {
        loop {
            let _ = run_command(args.command.clone(), false).await;
        }
    } else {
        Ok(run_command(args.command, true).await?.to_exit_code())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    exit_on_panic();
    let res = run().await;
    match res {
        Ok(code) => code,
        Err(e) => {
            // If you return a Result from main, and RUST_BACKTRACE=1 is set, then
            // it will print a backtrace - which is not what we want.
            eprintln!("{:#}", e);
            ExitCode::FAILURE
        }
    }
}
