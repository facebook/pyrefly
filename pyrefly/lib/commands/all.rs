/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use clap::Subcommand;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::buck_check::BuckCheckArgs;
use crate::commands::check::CheckResult;
use crate::commands::check::FullCheckArgs;
use crate::commands::check::SnippetCheckArgs;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::dump_config::DumpConfigArgs;
use crate::commands::infer::InferArgs;
use crate::commands::init::InitArgs;
use crate::commands::lsp::LspArgs;
use crate::commands::report::ReportArgs;
use crate::commands::stubgen::StubgenArgs;
use crate::commands::suppress::SuppressArgs;
use crate::commands::tsp::TspArgs;
use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::external_provider::NoExternalProvider;

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
    /// Suppress type errors by adding ignore comments, or remove unused ignores.
    Suppress(SuppressArgs),
    /// Generate .pyi stub files from Python source files.
    Stubgen(StubgenArgs),
}

impl Command {
    pub async fn run(
        self,
        version: &str,
        telemetry: &impl Telemetry,
        config_configurer_wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<(CommandExitStatus, Option<CheckResult>)> {
        match self {
            Command::Check(args) => args.run(config_configurer_wrapper, thread_count).await,
            Command::Snippet(args) => args.run(config_configurer_wrapper, thread_count).await,
            Command::BuckCheck(args) => Ok((args.run(thread_count)?, None)),
            Command::Lsp(args) => Ok((
                args.run(
                    version,
                    None,
                    None,
                    telemetry,
                    Arc::new(NoExternalProvider),
                    config_configurer_wrapper,
                    thread_count,
                )?,
                None,
            )),
            Command::Tsp(args) => Ok((
                args.run(telemetry, config_configurer_wrapper, thread_count)?,
                None,
            )),
            Command::Init(args) => Ok((
                args.run(config_configurer_wrapper.clone(), thread_count)?,
                None,
            )),
            Command::Infer(args) => Ok((args.run(config_configurer_wrapper, thread_count)?, None)),
            Command::DumpConfig(args) => Ok((args.run(config_configurer_wrapper)?, None)),
            Command::Report(args) => Ok((args.run(config_configurer_wrapper, thread_count)?, None)),
            Command::Suppress(args) => {
                Ok((args.run(config_configurer_wrapper, thread_count)?, None))
            }
            Command::Stubgen(args) => {
                Ok((args.run(config_configurer_wrapper, thread_count)?, None))
            }
        }
    }
}
