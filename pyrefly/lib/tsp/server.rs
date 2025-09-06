/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::sync::Arc;

use dupe::Dupe;
use lsp_server::Connection;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_types::InitializeParams;
use lsp_types::ServerCapabilities;
use tsp_types::TSPRequests;

use crate::commands::lsp::IndexingMode;
use crate::lsp::lsp::new_response;
use crate::lsp::queue::LspEvent;
use crate::lsp::queue::LspQueue;
use crate::lsp::server::ProcessEvent;
use crate::lsp::server::TspInterface;
use crate::lsp::server::capabilities;
use crate::lsp::server::dispatch_lsp_events;
use crate::lsp::transaction_manager::TransactionManager;

/// TSP server that delegates to LSP server infrastructure while handling only TSP requests
pub struct TspServer {
    pub inner: Box<dyn TspInterface>,
}

impl TspServer {
    pub fn new(lsp_server: Box<dyn TspInterface>) -> Self {
        Self { inner: lsp_server }
    }

    pub fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent> {
        // For TSP requests, handle them specially
        if let LspEvent::LspRequest(ref request) = event {
            if self.handle_tsp_request(ide_transaction_manager, request)? {
                return Ok(ProcessEvent::Continue);
            }
            // If it's not a TSP request, let the LSP server reject it since TSP server shouldn't handle LSP requests
            self.inner.send_response(lsp_server::Response::new_err(
                request.id.clone(),
                lsp_server::ErrorCode::MethodNotFound as i32,
                format!("TSP server does not support LSP method: {}", request.method),
            ));
            return Ok(ProcessEvent::Continue);
        }

        // For all other events (notifications, responses, etc.), delegate to inner server
        self.inner.process_event(
            ide_transaction_manager,
            canceled_requests,
            subsequent_mutation,
            event,
        )
    }

    fn handle_tsp_request<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        request: &Request,
    ) -> anyhow::Result<bool> {
        // Convert the request into a TSPRequests enum
        let wrapper = serde_json::json!({
            "method": request.method,
            "id": request.id,
            "params": request.params
        });

        let Ok(msg) = serde_json::from_value::<TSPRequests>(wrapper) else {
            // Not a TSP request
            return Ok(false);
        };

        match msg {
            TSPRequests::GetSupportedProtocolVersionRequest { .. } => {
                let transaction =
                    ide_transaction_manager.non_committable_transaction(self.inner.state());
                self.inner.send_response(new_response(
                    request.id.clone(),
                    Ok(self.get_supported_protocol_version(&transaction)),
                ));
                ide_transaction_manager.save(transaction);
                Ok(true)
            }
            _ => {
                // Other TSP requests not yet implemented
                Ok(false)
            }
        }
    }
}

pub fn tsp_loop(
    lsp_server: Box<dyn TspInterface>,
    connection: Arc<Connection>,
    _initialization_params: InitializeParams,
) -> anyhow::Result<()> {
    eprintln!("Reading TSP messages");
    let connection_for_dispatcher = connection.dupe();
    let lsp_queue = LspQueue::new();

    let server = TspServer::new(lsp_server);

    let lsp_queue2 = lsp_queue.dupe();
    std::thread::spawn(move || {
        dispatch_lsp_events(&connection_for_dispatcher, lsp_queue2);
    });

    let mut ide_transaction_manager = TransactionManager::default();
    let mut canceled_requests = HashSet::new();

    while let Ok((subsequent_mutation, event)) = lsp_queue.recv() {
        match server.process_event(
            &mut ide_transaction_manager,
            &mut canceled_requests,
            subsequent_mutation,
            event,
        )? {
            ProcessEvent::Continue => {}
            ProcessEvent::Exit => break,
        }
    }

    Ok(())
}

/// Generate TSP-specific server capabilities using the same capabilities as LSP
pub fn tsp_capabilities(
    indexing_mode: IndexingMode,
    initialization_params: &InitializeParams,
) -> ServerCapabilities {
    // Use the same capabilities as LSP - TSP server supports the same features
    // but will only respond to TSP protocol requests
    capabilities(indexing_mode, initialization_params)
}
