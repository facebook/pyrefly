/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Write;
use std::sync::Arc;

use clap::Parser;
use pyrefly_util::telemetry::Telemetry;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::lsp::IndexingMode;
use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::external_references::NoExternalReferences;
use crate::lsp::non_wasm::queue::LspQueue;
use crate::lsp::non_wasm::server::Connection;
use crate::lsp::non_wasm::server::InitializeInfo;
use crate::lsp::non_wasm::server::MessageReader;
use crate::lsp::non_wasm::server::initialize_finish;
use crate::lsp::non_wasm::server::initialize_start;
use crate::tsp::server::tsp_capabilities;
use crate::tsp::server::tsp_loop;

/// Arguments for TSP server
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct TspArgs {
    /// Find the struct that contains this field and add the indexing mode used by the language server
    #[arg(long, value_enum, default_value_t)]
    pub(crate) indexing_mode: IndexingMode,
    /// Sets the maximum number of user files for Pyrefly to index in the workspace.
    /// Note that indexing files is a performance-intensive task.
    #[arg(long, default_value_t = if cfg!(fbcode_build) {0} else {2000})]
    pub(crate) workspace_indexing_limit: usize,
}

pub fn run_tsp(
    connection: Connection,
    mut reader: MessageReader,
    args: TspArgs,
    telemetry: &impl Telemetry,
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<()> {
    if let Some(initialize_info) =
        initialize_tsp_connection(&connection, &mut reader, args.indexing_mode)?
    {
        // Create an LSP server instance for the TSP server to use.
        let lsp_queue = LspQueue::new();
        let surface = telemetry.surface();
        let lsp_server = crate::lsp::non_wasm::server::Server::new(
            connection,
            lsp_queue,
            initialize_info.params.clone(),
            initialize_info.supports_diagnostic_markdown,
            args.indexing_mode,
            args.workspace_indexing_limit,
            false,
            surface,
            None, // No path remapping for TSP
            Arc::new(NoExternalReferences),
            wrapper,
        );

        // Reuse the existing lsp_loop but with TSP initialization
        tsp_loop(lsp_server, reader, initialize_info, telemetry)?;
    }
    Ok(())
}

fn initialize_tsp_connection(
    connection: &Connection,
    reader: &mut MessageReader,
    indexing_mode: IndexingMode,
) -> anyhow::Result<Option<InitializeInfo>> {
    let Some((id, initialize_info)) = initialize_start(&connection.sender, reader)? else {
        return Ok(None);
    };
    let capabilities = tsp_capabilities(indexing_mode, &initialize_info.params);
    // Note: TSP doesn't include serverInfo, unlike LSP
    if !initialize_finish(&connection.sender, reader, id, capabilities, None)? {
        return Ok(None);
    }
    Ok(Some(initialize_info))
}

impl TspArgs {
    pub fn run(
        self,
        telemetry: &impl Telemetry,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        // Note that we must have our logging only write out to stderr.
        eprintln!("starting TSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, reader, io_threads) = Connection::stdio();

        run_tsp(connection, reader, self, telemetry, wrapper)?;
        io_threads.join()?;
        // We have shut down gracefully.
        // Use writeln! instead of eprintln! to avoid panicking if stderr is closed.
        // This can happen, for example, when stderr is connected to an LSP client which
        // closes the connection before Pyrefly language server exits.
        let _ = writeln!(std::io::stderr(), "shutting down TSP server");
        Ok(CommandExitStatus::Success)
    }
}
