/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Subcommand;

use crate::commands::autotype::AutotypeArgs;
use crate::commands::buck_check::BuckCheckArgs;
use crate::commands::check::FullCheckArgs;
use crate::commands::check::SnippetCheckArgs;
use crate::commands::dump_config::DumpConfigArgs;
use crate::commands::init::InitArgs;
use crate::commands::lsp::LspArgs;
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

    /// Automatically add type annotations to a file or directory.
    Autotype(AutotypeArgs),
}

impl Command {
    pub async fn run(self, allow_forget: bool) -> anyhow::Result<CommandExitStatus> {
        match self {
            Command::Check(args) => args.run(allow_forget).await,
            Command::Snippet(args) => args.run(allow_forget).await,
            Command::BuckCheck(args) => args.run(),
            Command::Lsp(args) => args.run(),
            Command::Init(args) => args.run(),
            Command::Autotype(args) => args.run(),
            Command::DumpConfig(args) => args.run(),
        }
    }
}
