/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Parser;
use clap::ValueEnum;
use lsp_server::Connection;
use lsp_server::ProtocolError;
use lsp_types::InitializeParams;
use pyrefly_util::telemetry::Telemetry;

use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::server::capabilities;
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

pub fn run_lsp(
    connection: Connection,
    args: LspArgs,
    version_string: &str,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    let initialization_params = match initialize_connection(&connection, &args, version_string) {
        Ok(it) => it,
        Err(e) => {
            // Use this in later versions of LSP server
            // if e.channel_is_disconnected() {
            // io_threads.join()?;
            // }
            return Err(e.into());
        }
    };
    lsp_loop(
        connection,
        initialization_params,
        args.indexing_mode,
        args.workspace_indexing_limit,
        args.build_system_blocking,
        telemetry,
    )?;
    Ok(())
}

fn initialize_connection(
    connection: &Connection,
    args: &LspArgs,
    version_string: &str,
) -> Result<InitializeParams, ProtocolError> {
    let (request_id, initialization_params) = connection.initialize_start()?;
    let initialization_params: InitializeParams =
        serde_json::from_value(initialization_params).unwrap();
    let server_capabilities =
        serde_json::to_value(capabilities(args.indexing_mode, &initialization_params)).unwrap();
    let initialize_data = serde_json::json!({
        "capabilities": server_capabilities,
        "serverInfo": {
            "name": "pyrefly-lsp",
            "version": version_string,
        }
    });

    connection.initialize_finish(request_id, initialize_data)?;
    Ok(initialization_params)
}

impl LspArgs {
    pub fn run(
        self,
        version_string: &str,
        telemetry: &impl Telemetry,
    ) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        run_lsp(connection, self, version_string, telemetry)?;
        io_threads.join()?;
        // We have shut down gracefully.
        eprintln!("shutting down server");
        Ok(CommandExitStatus::Success)
    }
}
