/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Subcommand;
use pyrefly_util::telemetry::Telemetry;

use crate::commands::buck_check::BuckCheckArgs;
use crate::commands::check::FullCheckArgs;
use crate::commands::check::SnippetCheckArgs;
use crate::commands::dump_config::DumpConfigArgs;
use crate::commands::infer::InferArgs;
use crate::commands::init::InitArgs;
use crate::commands::lsp::LspArgs;
use crate::commands::report::ReportArgs;
use crate::commands::tsp::TspArgs;
use crate::commands::util::CommandExitStatus;

/// Subcommands to run Pyrefly with.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Full type checking on a file or a project
    Check(FullCheckArgs),

    /// Check a Python code snippet
    Snippet(SnippetCheckArgs),

    /// Dump info about pyrefly's configuration. Use by replacing `check` with `dump-config` in your pyrefly invocation.
    DumpConfig(DumpConfigArgs),

    /// Entry point for Buck integration
    BuckCheck(BuckCheckArgs),

    /// Initialize a new pyrefly config in the given directory,
    /// or migrate an existing mypy or pyright config to pyrefly.
    Init(InitArgs),

    /// Start an LSP server
    Lsp(LspArgs),

    /// Start a TSP server
    Tsp(TspArgs),
    /// Automatically add type annotations to a file or directory.
    Infer(InferArgs),
    /// Generate reports from pyrefly type checking results.
    Report(ReportArgs),
}

impl Command {
    pub async fn run(
        self,
        version_string: &str,
        telemetry: &impl Telemetry,
    ) -> anyhow::Result<CommandExitStatus> {
        match self {
            Command::Check(args) => args.run().await,
            Command::Snippet(args) => args.run().await,
            Command::BuckCheck(args) => args.run(),
            Command::Lsp(args) => args.run(version_string, telemetry),
            Command::Tsp(args) => args.run(telemetry),
            Command::Init(args) => args.run(),
            Command::Infer(args) => args.run(),
            Command::DumpConfig(args) => args.run(),
            Command::Report(args) => args.run(),
        }
    }
}
