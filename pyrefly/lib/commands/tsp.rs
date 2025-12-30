/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Parser;
use lsp_server::Connection;
use lsp_server::ProtocolError;
use lsp_types::InitializeParams;
use pyrefly_util::telemetry::Telemetry;

use crate::commands::lsp::IndexingMode;
use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::queue::LspQueue;
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
    args: TspArgs,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    let initialization_params = match initialize_tsp_connection(&connection, &args) {
        Ok(it) => it,
        Err(e) => {
            return Err(e.into());
        }
    };

    // Create an LSP server instance for the TSP server to use.
    let lsp_queue = LspQueue::new();
    let lsp_server = crate::lsp::non_wasm::server::Server::new(
        connection,
        lsp_queue,
        initialization_params.clone(),
        args.indexing_mode,
        args.workspace_indexing_limit,
        false,
    );

    // Reuse the existing lsp_loop but with TSP initialization
    tsp_loop(lsp_server, initialization_params, telemetry)?;
    Ok(())
}

fn initialize_tsp_connection(
    connection: &Connection,
    args: &TspArgs,
) -> Result<InitializeParams, ProtocolError> {
    let (request_id, initialization_params) = connection.initialize_start()?;
    let initialization_params: InitializeParams =
        serde_json::from_value(initialization_params).unwrap();

    // Use TSP-specific capabilities (same as LSP but without serverInfo)
    let server_capabilities =
        serde_json::to_value(tsp_capabilities(args.indexing_mode, &initialization_params)).unwrap();
    let initialize_data = serde_json::json!({
        "capabilities": server_capabilities,
        // Note: TSP doesn't include serverInfo, unlike LSP
    });

    connection.initialize_finish(request_id, initialize_data)?;
    Ok(initialization_params)
}

impl TspArgs {
    pub fn run(self, telemetry: &impl Telemetry) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting TSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        run_tsp(connection, self, telemetry)?;
        io_threads.join()?;
        // We have shut down gracefully.
        eprintln!("shutting down TSP server");
        Ok(CommandExitStatus::Success)
    }
}
