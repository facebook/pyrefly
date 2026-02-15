/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Write;

use clap::Parser;
use clap::ValueEnum;
use lsp_types::ServerInfo;
use pyrefly_util::telemetry::Telemetry;

use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::server::Connection;
use crate::lsp::non_wasm::server::InitializeInfo;
use crate::lsp::non_wasm::server::capabilities;
use crate::lsp::non_wasm::server::initialize_finish;
use crate::lsp::non_wasm::server::initialize_start;
use crate::lsp::non_wasm::server::lsp_loop;

/// Pyrefly's indexing strategy for open projects when performing go-to-definition
/// requests.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum IndexingMode {
    /// Do not index anything. Features that depend on indexing (e.g. find-refs) will be disabled.
    None,
    /// Start indexing when opening a file that belongs to a config in the background.
    /// Indexing will happen in another thread, so that normal IDE services are not blocked.
    #[default]
    LazyNonBlockingBackground,
    /// Start indexing when opening a file that belongs to a config in the background.
    /// Indexing will happen in the main thread, so that IDE services will be blocked.
    /// However, this is useful for deterministic testing.
    LazyBlocking,
}

/// Arguments for LSP server
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct LspArgs {
    /// Find the struct that contains this field and add the indexing mode used by the language server
    #[arg(long, value_enum, default_value_t)]
    pub(crate) indexing_mode: IndexingMode,

    /// Sets the maximum number of user files for Pyrefly to index in the workspace.
    /// Note that indexing files is a performance-intensive task.
    #[arg(long, default_value_t = if cfg!(fbcode_build) {0} else {2000})]
    pub(crate) workspace_indexing_limit: usize,

    /// Block for build system operations, only using fallback heuristics after checking
    /// an up-to-date source DB. Only useful for benchmarking.
    #[arg(long)]
    pub(crate) build_system_blocking: bool,
}

/// Run LSP server with optional path remapping.
/// When a path remapper is provided, go-to-definition will use the remapped
/// paths for URIs, allowing navigation to source files instead of installed
/// package files.
pub fn run_lsp(
    connection: Connection,
    args: LspArgs,
    server_info: Option<ServerInfo>,
    path_remapper: Option<PathRemapper>,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    if let Some(initialize_info) =
        initialize_connection(&connection, args.indexing_mode, server_info)?
    {
        lsp_loop(
            connection,
            initialize_info,
            args.indexing_mode,
            args.workspace_indexing_limit,
            args.build_system_blocking,
            path_remapper,
            telemetry,
        )?;
    }
    Ok(())
}

fn initialize_connection(
    connection: &Connection,
    indexing_mode: IndexingMode,
    server_info: Option<ServerInfo>,
) -> anyhow::Result<Option<InitializeInfo>> {
    let Some((id, initialize_info)) = initialize_start(connection)? else {
        return Ok(None);
    };
    let capabilities = capabilities(indexing_mode, &initialize_info.params);
    if !initialize_finish(connection, id, capabilities, server_info)? {
        return Ok(None);
    }
    Ok(Some(initialize_info))
}

impl LspArgs {
    /// Run LSP with optional path remapping.
    /// When a path remapper is provided, go-to-definition will navigate to
    /// remapped source files instead of installed package files.
    pub fn run(
        self,
        version: &str,
        path_remapper: Option<PathRemapper>,
        telemetry: &impl Telemetry,
    ) -> anyhow::Result<CommandExitStatus> {
        // Note that we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        let server_info = ServerInfo {
            name: "pyrefly-lsp".to_owned(),
            version: Some(version.to_owned()),
        };

        run_lsp(
            connection,
            self,
            Some(server_info),
            path_remapper,
            telemetry,
        )?;
        io_threads.join()?;
        // We have shut down gracefully.
        // Use writeln! instead of eprintln! to avoid panicking if stderr is closed.
        // This can happen, for example, when stderr is connected to an LSP client which
        // closes the connection before Pyrefly language server exits.
        let _ = writeln!(std::io::stderr(), "shutting down server");
        Ok(CommandExitStatus::Success)
    }
}
