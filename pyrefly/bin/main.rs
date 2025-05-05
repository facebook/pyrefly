/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::backtrace::Backtrace;
use std::env::args_os;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::anyhow;
use clap::Parser;
use clap::Subcommand;
use dupe::Dupe;
use library::ArcId;
use library::ConfigFile;
use library::Watcher;
use library::clap_env;
use library::finder::ConfigFinder;
use library::get_args_expanded;
use library::globs::FilteredGlobs;
use library::globs::Globs;
use library::run::BuckCheckArgs;
use library::run::CheckArgs;
use library::run::CommandExitStatus;
use library::run::CommonGlobalArgs;
use library::run::LspArgs;
use library::standard_config_finder;
use path_absolutize::Absolutize;
use pyrefly::library::library::library::library;
use pyrefly::library::library::library::library::ConfigSource;
use pyrefly::library::library::library::library::ModulePath;
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
    /// It is an error to pass this flag in "single-file checking mode".
    /// When not set, Pyre will perform an upward-filesystem-walk approach to find the nearest
    /// pyrefly.toml or 'pyproject.toml with `tool.pyre` section'. If no config is found, Pyre exits with error.
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

fn get_globs_and_config_for_project(
    config: Option<PathBuf>,
    project_excludes: Option<Vec<String>>,
    args: &library::run::CheckArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    let config = match config {
        Some(explicit) => {
            // We deliberately don't use the cached object, since we want errors in an explicit config to be fatal
            ArcId::new(args.override_config(ConfigFile::from_file(&explicit, true)?))
        }
        None => {
            let current_dir = std::env::current_dir().context("cannot identify current dir")?;
            let config_finder = config_finder(args.clone());
            config_finder
                .directory(&current_dir)
                .unwrap_or_else(|| ArcId::new(args.override_config(ConfigFile::default())))
        }
    };
    match &config.source {
        ConfigSource::File(path) => {
            info!("Checking project configured at {path:?}");
        }
        ConfigSource::Synthetic => {
            info!("Checking current directory with default configuration");
        }
    }

    // We want our config_finder to never actually
    let config_finder = ConfigFinder::new_constant(config.dupe());

    debug!("Config is: {}", config);
    let project_excludes =
        project_excludes.map_or_else(|| config.project_excludes.clone(), Globs::new);

    Ok((
        FilteredGlobs::new(config.project_includes.clone(), project_excludes),
        config_finder,
    ))
}

fn get_globs_and_config_for_files(
    files_to_check: Globs,
    project_excludes: Option<Vec<String>>,
    args: &library::run::CheckArgs,
) -> anyhow::Result<(FilteredGlobs, ConfigFinder)> {
    let project_excludes =
        project_excludes.map_or_else(ConfigFile::default_project_excludes, Globs::new);
    let files_to_check = files_to_check.from_root(PathBuf::new().absolutize()?.as_ref());
    let config_finder = config_finder(args.clone());
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
    if !files.is_empty() && config.is_some() {
        return Err(anyhow!(
            "Can either supply `FILES...` OR `--config/-c`, not both."
        ));
    }
    args.absolute_search_path();
    args.validate()?;
    if files.is_empty() {
        get_globs_and_config_for_project(config, project_excludes, args)
    } else {
        get_globs_and_config_for_files(Globs::new(files), project_excludes, args)
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
        Command::Lsp(args) => args.run(Vec::new()),
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
            for (config, files) in configs_to_files.into_iter() {
                match &config.source {
                    ConfigSource::Synthetic => {
                        println!("Default configuration");
                    }
                    ConfigSource::File(path) => {
                        println!("Configuration at {path:?}");
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
                println!("  Search path: {:?}", config.search_path);
                println!("  Site package path: {:?}", config.site_package_path());
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
