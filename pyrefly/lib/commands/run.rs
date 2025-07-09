/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::process::ExitCode;

use clap::ColorChoice;
use clap::Parser;
use pyrefly_util::args::clap_env;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::init_thread_pool;
use pyrefly_util::trace::init_tracing;

pub use crate::commands::autotype::Args as AutotypeArgs;
pub use crate::commands::buck_check::Args as BuckCheckArgs;
pub use crate::commands::check::Args as CheckArgs;
pub use crate::commands::init::Args as InitArgs;
pub use crate::commands::lsp::Args as LspArgs;

/// Arguments shared between all commands.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct CommonGlobalArgs {
    /// Number of threads to use for parallelization.
    /// Setting the value to 1 implies sequential execution without any parallelism.
    /// Setting the value to 0 means to pick the number of threads automatically using default heuristics.
    #[arg(long, short = 'j', default_value = "0", global = true, env = clap_env("THREADS"))]
    pub threads: ThreadCount,

    /// Control whether colored output is used.
    #[arg(long, default_value = "auto", global = true, env = clap_env("COLOR"))]
    color: ColorChoice,

    /// Enable verbose logging.
    #[arg(long = "verbose", short = 'v', global = true, env = clap_env("VERBOSE"))]
    pub verbose: bool,
}

fn init_color(color: ColorChoice) {
    match color {
        ColorChoice::Never => {
            anstream::ColorChoice::write_global(anstream::ColorChoice::Never);
        }
        ColorChoice::Always => {
            anstream::ColorChoice::write_global(anstream::ColorChoice::Always);
        }
        ColorChoice::Auto => {
            // Do nothing: the default is auto-determine
        }
    }
}

impl CommonGlobalArgs {
    pub fn init(&self) {
        init_tracing(self.verbose, false);
        init_thread_pool(self.threads);
        init_color(self.color);
    }
}

/// Exit status of a command, if the run is completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandExitStatus {
    /// The command completed without an issue.
    Success,
    /// The command completed, but problems (e.g. type errors) were found.
    UserError,
    /// An error occurred in the environment or the underlying infrastructure,
    /// which prevents the command from completing.
    InfraError,
}

impl CommandExitStatus {
    pub fn to_exit_code(self) -> ExitCode {
        match self {
            CommandExitStatus::Success => ExitCode::SUCCESS,
            CommandExitStatus::UserError => ExitCode::FAILURE,
            // Exit code 2 is reserved for Meta-internal usages
            CommandExitStatus::InfraError => ExitCode::from(3),
        }
    }
}
