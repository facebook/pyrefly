/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod check;
mod collect;
pub(crate) mod report;
mod types;

use clap::Subcommand;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::coverage::check::CheckArgs;
use crate::commands::coverage::report::ReportArgs;
use crate::commands::util::CommandExitStatus;

/// Subcommands of `pyrefly coverage`.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Subcommand)]
pub enum CoverageCommand {
    /// Generate a machine-readable type-coverage report from pyrefly type checking results.
    Report(ReportArgs),
    /// Check that type coverage from pyrefly type checking results meets a minimum threshold.
    Check(CheckArgs),
}

impl CoverageCommand {
    pub fn run(
        self,
        config_configurer_wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        match self {
            CoverageCommand::Report(args) => args.run(config_configurer_wrapper, thread_count),
            CoverageCommand::Check(args) => args.run(config_configurer_wrapper, thread_count),
        }
    }
}
