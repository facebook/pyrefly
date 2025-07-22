/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::sync::Arc;

use clap::Parser;
use clap::ValueEnum;
use crossbeam_channel::Select;
use dupe::Dupe;
use lsp_server::Connection;

use crate::commands::util::CommandExitStatus;
use crate::lsp::server::ProcessEvent;
use crate::lsp::server::Server;
use crate::lsp::server::dispatch_lsp_events;
use crate::lsp::server::initialize_connection;
use crate::lsp::transaction_manager::IDETransactionManager;

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
}

pub fn run_lsp(
    connection: Arc<Connection>,
    wait_on_connection: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
    args: LspArgs,
) -> anyhow::Result<CommandExitStatus> {
    let initialization_params = match initialize_connection(&connection, &args) {
        Ok(it) => it,
        Err(e) => {
            // Use this in later versions of LSP server
            // if e.channel_is_disconnected() {
            // io_threads.join()?;
            // }
            return Err(e.into());
        }
    };
    eprintln!("Reading messages");
    let connection_for_dispatcher = connection.dupe();
    let (queued_events_sender, queued_events_receiver) = crossbeam_channel::unbounded();
    let (priority_events_sender, priority_events_receiver) = crossbeam_channel::unbounded();
    let priority_events_sender = Arc::new(priority_events_sender);
    let mut event_receiver_selector = Select::new_biased();
    // Biased selector will pick the receiver with lower index over higher ones,
    // so we register priority_events_receiver first.
    let priority_receiver_index = event_receiver_selector.recv(&priority_events_receiver);
    let queued_events_receiver_index = event_receiver_selector.recv(&queued_events_receiver);
    let server = Server::new(
        connection,
        priority_events_sender.dupe(),
        initialization_params,
        args.indexing_mode,
    );
    std::thread::spawn(move || {
        dispatch_lsp_events(
            &connection_for_dispatcher,
            priority_events_sender,
            queued_events_sender,
        );
    });
    let mut ide_transaction_manager = IDETransactionManager::default();
    let mut canceled_requests = HashSet::new();
    loop {
        let selected = event_receiver_selector.select();
        let received = match selected.index() {
            i if i == priority_receiver_index => selected.recv(&priority_events_receiver),
            i if i == queued_events_receiver_index => selected.recv(&queued_events_receiver),
            _ => unreachable!(),
        };
        if let Ok(event) = received {
            match server.process_event(
                &mut ide_transaction_manager,
                &mut canceled_requests,
                event,
            )? {
                ProcessEvent::Continue => {}
                ProcessEvent::Exit => break,
            }
        } else {
            break;
        }
    }
    eprintln!("waiting for connection to close");
    drop(server); // close connection
    wait_on_connection()?;

    // Shut down gracefully.
    eprintln!("shutting down server");
    Ok(CommandExitStatus::Success)
}

impl LspArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        run_lsp(
            Arc::new(connection),
            move || io_threads.join().map_err(anyhow::Error::from),
            self,
        )
    }
}
