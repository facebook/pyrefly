/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env::args_os;
use std::io;
use std::io::Write;
use std::process::ExitCode;

use clap::CommandFactory;
use clap::Parser;
use clap::Subcommand;
use clap::crate_version;
use clap_complete::Shell;
use clap_complete::generate;
use library::Command as StandardCommand;
use library::util::CommandExitStatus;
use library::util::CommonGlobalArgs;
use pyrefly::commands::lsp::filter_unrecognized_lsp_args;
use pyrefly::library::library::library::library;
use pyrefly_util::args::get_args_expanded;
use pyrefly_util::panic::exit_on_panic;
use pyrefly_util::telemetry::NoTelemetry;

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

/// Subcommands of the open-source `pyrefly` binary.
///
/// `Completion` lives here rather than in the library's shared `Command` so
/// that each binary generates completions for its own argument tree under its
/// own name. A binary that embeds the shared commands in a larger CLI (such as
/// the Meta-internal wrapper) would otherwise emit a script describing a
/// command tree it does not have.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a shell completion script on stdout.
    Completion {
        /// Shell to generate completions for.
        #[arg(long, value_enum)]
        shell: Shell,
    },

    /// The commands shared with every other Pyrefly frontend.
    #[command(flatten)]
    Standard(StandardCommand),
}

/// Run based on the command line arguments.
async fn run() -> anyhow::Result<ExitCode> {
    let expanded_args = get_args_expanded(args_os())?;
    let filtered_args = filter_unrecognized_lsp_args(expanded_args);
    let args = Args::parse_from(filtered_args);
    args.common.init(false);
    let thread_count = args.common.thread_count();
    let status = match args.command {
        Command::Completion { shell } => {
            // Buffer generation so stdout errors are returned instead of panicking.
            let mut script = Vec::new();
            generate(shell, &mut Args::command(), "pyrefly", &mut script);
            io::stdout().lock().write_all(&script)?;
            CommandExitStatus::Success
        }
        Command::Standard(command) => {
            let (status, _) = command
                .run(crate_version!(), &NoTelemetry, None, thread_count)
                .await?;
            status
        }
    };
    Ok(status.to_exit_code())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    // Enable stack overflow backtraces for debugging.
    // This is unsafe and only intended for debug builds.
    #[cfg(not(windows))]
    #[cfg(feature = "debug-stack-overflow")]
    unsafe {
        backtrace_on_stack_overflow::enable();
    }
    exit_on_panic();
    let res = run().await;
    match res {
        Ok(code) => code,
        Err(e) => {
            // If you return a Result from main, and RUST_BACKTRACE=1 is set, then
            // it will print a backtrace - which is not what we want.
            eprintln!("{e:#}");
            ExitCode::FAILURE
        }
    }
}
