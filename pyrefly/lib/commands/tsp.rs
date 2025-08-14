/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use clap::Parser;
use lsp_server::Connection;
use lsp_server::ProtocolError;
use lsp_types::InitializeParams;

use crate::commands::lsp::IndexingMode;
use crate::commands::util::CommandExitStatus;
use crate::lsp::server::lsp_loop;
use crate::tsp::server::tsp_capabilities;

/// Arguments for TSP server
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct TspArgs {
    /// Find the struct that contains this field and add the indexing mode used by the language server
    #[arg(long, value_enum, default_value_t)]
    pub(crate) indexing_mode: IndexingMode,
}

pub fn run_tsp(connection: Arc<Connection>, args: TspArgs) -> anyhow::Result<()> {
    let initialization_params = match initialize_tsp_connection(&connection, &args) {
        Ok(it) => it,
        Err(e) => {
            return Err(e.into());
        }
    };
    // Reuse the existing lsp_loop but with TSP initialization
    lsp_loop(connection, initialization_params, args.indexing_mode)?;
    Ok(())
}

fn initialize_tsp_connection(
    connection: &Connection,
    args: &TspArgs,
) -> Result<InitializeParams, ProtocolError> {
    let (request_id, initialization_params) = connection.initialize_start()?;
    let initialization_params: InitializeParams =
        serde_json::from_value(initialization_params).unwrap();
    
    // Use TSP-specific capabilities
    let server_capabilities = serde_json::to_value(tsp_capabilities(args.indexing_mode, &initialization_params)).unwrap();
    
    let initialize_data = serde_json::json!({
        "capabilities": server_capabilities,
    });

    connection.initialize_finish(request_id, initialize_data)?;
    Ok(initialization_params)
}

impl TspArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting TSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        run_tsp(Arc::new(connection), self)?;
        io_threads.join()?;
        // We have shut down gracefully.
        eprintln!("shutting down TSP server");
        Ok(CommandExitStatus::Success)
    }
}
