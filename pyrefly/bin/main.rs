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

use clap::Arg;
use clap::Command as ClapCommand;
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
    Standard(Box<StandardCommand>),
}

/// Create a completion-only tree because `clap_complete`'s AOT generators do
/// not honor hidden commands or arguments. The projection copies the command
/// properties those generators read; the generator-output test guards that list.
fn without_hidden(cmd: &ClapCommand) -> ClapCommand {
    let args: Vec<Arg> = cmd
        .get_arguments()
        .filter(|arg| !arg.is_hide_set())
        .cloned()
        .collect();
    let subcommands: Vec<ClapCommand> = cmd
        .get_subcommands()
        .filter(|subcommand| !subcommand.is_hide_set())
        .map(without_hidden)
        .collect();
    let aliases: Vec<String> = cmd.get_visible_aliases().map(str::to_owned).collect();

    let mut rebuilt = ClapCommand::new(cmd.get_name().to_owned())
        .display_order(cmd.get_display_order())
        .visible_aliases(aliases)
        .args(args)
        .subcommands(subcommands);
    if let Some(about) = cmd.get_about() {
        rebuilt = rebuilt.about(about.clone());
    }
    if let Some(long_about) = cmd.get_long_about() {
        rebuilt = rebuilt.long_about(long_about.clone());
    }
    if let Some(version) = cmd.get_version() {
        rebuilt = rebuilt.version(version.to_owned());
    }
    rebuilt
}

/// The command tree that completion scripts are generated from.
///
/// Filtering applies to generation only. The tree used for parsing still
/// contains every hidden command and flag, so they keep working when typed out
/// in full.
fn completion_command() -> ClapCommand {
    without_hidden(&Args::command())
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
            generate(shell, &mut completion_command(), "pyrefly", &mut script);
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

#[cfg(test)]
mod tests {
    use clap::ValueEnum;

    use super::*;

    fn assert_nothing_hidden(cmd: &ClapCommand) {
        for arg in cmd.get_arguments() {
            assert!(
                !arg.is_hide_set(),
                "hidden argument `{}` of `{}` leaked into the completion tree",
                arg.get_id(),
                cmd.get_name()
            );
        }
        for subcommand in cmd.get_subcommands() {
            assert!(
                !subcommand.is_hide_set(),
                "hidden subcommand `{}` leaked into the completion tree",
                subcommand.get_name()
            );
            assert_nothing_hidden(subcommand);
        }
    }

    fn show_hidden(cmd: ClapCommand) -> ClapCommand {
        cmd.mut_args(|arg| arg.hide(false))
            .mut_subcommands(|subcommand| show_hidden(subcommand.hide(false)))
    }

    fn generate_script(shell: Shell, mut cmd: ClapCommand) -> Vec<u8> {
        let mut script = Vec::new();
        generate(shell, &mut cmd, "pyrefly", &mut script);
        script
    }

    #[test]
    fn completion_tree_drops_hidden_items() {
        let tree = completion_command();
        assert_nothing_hidden(&tree);
        assert!(
            !tree.get_subcommands().any(|s| s.get_name() == "report"),
            "the deprecated `report` alias is hidden, so it must not be completed"
        );
        let check = tree
            .find_subcommand("check")
            .expect("`check` is visible and must survive filtering");
        assert!(
            !check
                .get_arguments()
                .any(|arg| arg.get_long() == Some("report-cinderx")),
            "`--report-cinderx` is hidden, so it must not be completed"
        );
    }

    #[test]
    fn completion_tree_builds_after_filtering() {
        completion_command().debug_assert();
    }

    #[test]
    fn completion_projection_preserves_generator_output() {
        let full = show_hidden(Args::command());
        for &shell in Shell::value_variants() {
            let expected = generate_script(shell, full.clone());
            let actual = generate_script(shell, without_hidden(&full));
            assert_eq!(actual, expected, "{shell} output changed during projection");
        }
    }

    #[test]
    fn hidden_items_are_still_parseable() {
        // Filtering applies to completion generation only; a hidden command or
        // flag typed out in full must keep working.
        let args = Args::try_parse_from(["pyrefly", "report"])
            .expect("the deprecated `report` alias should still parse");
        assert!(matches!(
            args.command,
            Command::Standard(ref command)
                if matches!(**command, StandardCommand::Report(_))
        ));
        Args::try_parse_from(["pyrefly", "check", "--no-progress-bar"])
            .expect("the hidden `--no-progress-bar` flag should still parse");
    }
}
