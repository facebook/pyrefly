/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env::args_os;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use clap::Subcommand;
use library::finder::ConfigFinder;
use library::globs_and_config_getter;
use library::run::AutotypeArgs;
use library::run::BuckCheckArgs;
use library::run::CheckArgs;
use library::run::CommandExitStatus;
use library::run::CommonGlobalArgs;
use library::run::InitArgs;
use library::run::LspArgs;
use library::run::dump_config;
use pyrefly::library::library::library::library;
use pyrefly_util::args::clap_env;
use pyrefly_util::args::get_args_expanded;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::panic::exit_on_panic;
use pyrefly_util::watcher::Watcher;

// fbcode likes to set its own allocator in fbcode.default_allocator
// So when we set our own allocator, buck build buck2 or buck2 build buck2 often breaks.
// Making jemalloc the default only when we do a cargo build.
#[global_allocator]
#[cfg(all(any(target_os = "linux", target_os = "macos"), not(fbcode_build)))]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "windows")]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Main CLI entrypoint for Pyrefly.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser)]
#[command(name = "pyrefly")]
#[command(about = "A fast Python type checker", long_about = None)]
#[command(version)]
struct Args {
    /// Common global arguments shared across commands.
    #[command(flatten)]
    common: CommonGlobalArgs,

    /// Subcommand execution args.
    #[command(subcommand)]
    command: Command,
}

#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
struct FullCheckArgs {
    /// Files to check (glob supported).
    /// If no file is specified, switch to project-checking mode where the files to
    /// check are determined from the closest configuration file.
    /// When supplied, `project_excludes` in any config files loaded for these files to check
    /// are ignored, and we use the default excludes unless overridden with the `--project-excludes` flag.
    files: Vec<String>,
    /// Files to exclude when type checking.
    #[arg(long, env = clap_env("PROJECT_EXCLUDES"))]
    project_excludes: Option<Vec<String>>,
    /// Watch for file changes and re-check them.
    #[arg(long, env = clap_env("WATCH"), conflicts_with = "check_all")]
    watch: bool,

    /// Explicitly set the Pyrefly configuration to use when type checking or starting a language server.
    /// In "single-file checking mode," this config is applied to all files being checked, ignoring
    /// the config's `project_includes` and `project_excludes` and ignoring any config-finding approach
    /// that would otherwise be used.
    /// When not set, Pyrefly will perform an upward-filesystem-walk approach to find the nearest
    /// pyrefly.toml or pyproject.toml with `tool.pyrefly` section'. If no config is found, Pyrefly exits with error.
    /// If both a pyrefly.toml and valid pyproject.toml are found, pyrefly.toml takes precedence.
    #[arg(long, short, env = clap_env("CONFIG"), value_name = "FILE")]
    config: Option<PathBuf>,

    /// Type checking arguments and configuration
    #[command(flatten)]
    args: CheckArgs,
}

/// Subcommands to run Pyrefly with.
#[deny(clippy::missing_docs_in_private_items)]
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

    /// Automatically add type annotations to a file or directory.
    Autotype(FullCheckArgs),
}

async fn run_autotype(
    args: library::run::AutotypeArgs,
    files_to_check: FilteredGlobs,
    config_finder: ConfigFinder,
) -> anyhow::Result<CommandExitStatus> {
    args.run(files_to_check, config_finder)
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
        match args.run_once(files_to_check, config_finder, allow_forget) {
            Ok((status, _)) => Ok(status),
            Err(e) => Err(e),
        }
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
            let (files_to_check, config_finder) = globs_and_config_getter::get(
                files,
                project_excludes,
                config,
                &mut args.config_override,
            )?;
            run_check(args, watch, files_to_check, config_finder, allow_forget).await
        }
        Command::BuckCheck(args) => args.run(),
        Command::Lsp(args) => args.run(),
        Command::Init(args) => args.run(),
        Command::Autotype(FullCheckArgs {
            files,
            project_excludes,
            config,
            watch: _,
            mut args,
        }) => {
            let (files_to_check, config_finder) = globs_and_config_getter::get(
                files,
                project_excludes,
                config,
                &mut args.config_override,
            )?;
            run_autotype(AutotypeArgs::new(), files_to_check, config_finder).await
        }
        // We intentionally make DumpConfig take the same arguments as Check so that dumping the
        // config is as easy as changing the command name.
        Command::DumpConfig(FullCheckArgs {
            files,
            project_excludes,
            config,
            mut args,
            ..
        }) => {
            let (files_to_check, config_finder) = globs_and_config_getter::get(
                files,
                project_excludes,
                config,
                &mut args.config_override,
            )?;
            dump_config(files_to_check, config_finder, args)
        }
    }
}

/// Run based on the command line arguments.
async fn run() -> anyhow::Result<ExitCode> {
    let args = Args::parse_from(get_args_expanded(args_os())?);
    args.common.init(false);
    Ok(run_command(args.command, true).await?.to_exit_code())
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
