/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use lsp_server::Request;
use lsp_server::RequestId;
use lsp_types::InitializeParams;
use lsp_types::ServerCapabilities;
use tracing::info;
use tsp_types::TSPRequests;

use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::lsp::new_response;
use crate::lsp::non_wasm::queue::LspEvent;
use crate::lsp::non_wasm::server::ProcessEvent;
use crate::lsp::non_wasm::server::TspInterface;
use crate::lsp::non_wasm::server::capabilities;
use crate::lsp::non_wasm::server::dispatch_lsp_events;
use crate::lsp::non_wasm::transaction_manager::TransactionManager;

/// TSP server that delegates to LSP server infrastructure while handling only TSP requests
pub struct TspServer {
    pub inner: Box<dyn TspInterface>,
    /// Current snapshot version, updated on RecheckFinished events
    pub(crate) current_snapshot: Arc<Mutex<i32>>,
}

impl TspServer {
    pub fn new(lsp_server: Box<dyn TspInterface>) -> Self {
        Self {
            inner: lsp_server,
            current_snapshot: Arc::new(Mutex::new(0)), // Start at 0, increments on RecheckFinished
        }
    }

    pub fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent> {
        // Remember if this event should increment the snapshot after processing
        let should_increment_snapshot = match &event {
            LspEvent::RecheckFinished => true,
            // Increment on DidChange since it affects type checker state via synchronous validation
            LspEvent::DidChangeTextDocument(_) => true,
            // Don't increment on DidChangeWatchedFiles directly since it triggers RecheckFinished
            // LspEvent::DidChangeWatchedFiles(_) => true,
            // Don't increment on DidOpen since it triggers RecheckFinished events that will increment
            // LspEvent::DidOpenTextDocument(_) => true,
            _ => false,
        };

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
        let result = self.inner.process_event(
            ide_transaction_manager,
            canceled_requests,
            subsequent_mutation,
            event,
        )?;

        // Increment snapshot after the inner server has processed the event
        if should_increment_snapshot && let Ok(mut current) = self.current_snapshot.lock() {
            *current += 1;
        }

        Ok(result)
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
            TSPRequests::GetSnapshotRequest { .. } => {
                // Get snapshot doesn't need a transaction since it just returns the cached value
                self.inner
                    .send_response(new_response(request.id.clone(), Ok(self.get_snapshot())));
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
    _initialization_params: InitializeParams,
) -> anyhow::Result<()> {
    eprintln!("Reading TSP messages");
    let server = TspServer::new(lsp_server);

    std::thread::scope(|scope| {
        // Start the recheck queue thread to process async tasks
        scope.spawn(|| {
            server
                .inner
                .recheck_queue()
                .run_until_stopped(|task| server.inner.run_task(task));
        });

        scope.spawn(|| {
            dispatch_lsp_events(server.inner.connection(), server.inner.lsp_queue());
        });

        let mut ide_transaction_manager = TransactionManager::default();
        let mut canceled_requests = HashSet::new();

        while let Ok((subsequent_mutation, event, queue_time)) = server.inner.lsp_queue().recv() {
            let queue_duration = queue_time.elapsed().as_secs_f32();
            let process_start = Instant::now();
            let event_description = event.describe();
            match server.process_event(
                &mut ide_transaction_manager,
                &mut canceled_requests,
                subsequent_mutation,
                event,
            )? {
                ProcessEvent::Continue => {
                    let process_duration = process_start.elapsed().as_secs_f32();
                    info!(
                        "Type server processed event `{}` in {:.2}s ({:.2}s waiting)",
                        event_description, process_duration, queue_duration
                    );
                }
                ProcessEvent::Exit => break,
            }
        }

        server.inner.recheck_queue().stop();
        Ok(())
    })
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
