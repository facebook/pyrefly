/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env::args_os;
use std::process::ExitCode;

use clap::Parser;
use clap::crate_version;
use library::Command;
use library::util::CommonGlobalArgs;
use pyrefly::commands::lsp::filter_unrecognized_lsp_args;
use pyrefly::library::library::library::library;
use pyrefly_util::args::get_args_expanded;
use pyrefly_util::panic::exit_on_panic;
use pyrefly_util::telemetry::NoTelemetry;

// fbcode sets its own allocator, so only select a custom allocator for Cargo builds.
// Cargo's musl binaries use mimalloc because jemalloc is not reliable on older Linux hosts.
#[global_allocator]
#[cfg(all(
    any(target_os = "macos", all(target_os = "linux", target_env = "gnu")),
    not(fbcode_build)
))]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[global_allocator]
#[cfg(any(target_os = "windows", target_env = "musl"))]
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

/// Run based on the command line arguments.
async fn run() -> anyhow::Result<ExitCode> {
    let expanded_args = get_args_expanded(args_os())?;
    let filtered_args = filter_unrecognized_lsp_args(expanded_args);
    let args = Args::parse_from(filtered_args);
    args.common.init(false);
    let thread_count = args.common.thread_count();
    let (status, _) = args
        .command
        .run(crate_version!(), &NoTelemetry, None, thread_count)
        .await?;
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
