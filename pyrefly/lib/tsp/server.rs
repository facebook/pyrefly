/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use lsp_server::ErrorCode;
use lsp_server::RequestId;
use lsp_server::ResponseError;
use lsp_types::InitializeParams;
use pyrefly_util::telemetry::QueueName;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryEventKind;
use tracing::info;
use tracing::warn;
use tsp_types::ConnectionRequestParams;
use tsp_types::ConnectionRequestResult;
use tsp_types::ConnectionTransportKind;
use tsp_types::GetTypeParams;
use tsp_types::TSPNotificationMethods;
use tsp_types::TSPRequests;

use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Notification;
use crate::lsp::non_wasm::protocol::Request;
use crate::lsp::non_wasm::protocol::Response;
use crate::lsp::non_wasm::queue::LspEvent;
use crate::lsp::non_wasm::server::Connection;
use crate::lsp::non_wasm::server::InitializeInfo;
use crate::lsp::non_wasm::server::MessageReader;
use crate::lsp::non_wasm::server::ProcessEvent;
use crate::lsp::non_wasm::server::ServerCapabilitiesWithTypeHierarchy;
use crate::lsp::non_wasm::server::TspInterface;
use crate::lsp::non_wasm::server::capabilities;
use crate::lsp::non_wasm::transaction_manager::TransactionManager;
use crate::tsp::type_conversion::convert_type_with_resolver;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectionRole {
    Main,
    ReadOnlyExtra,
}

struct ExtraConnectionHandle {
    close_tx: crossbeam_channel::Sender<()>,
}

/// TSP server that delegates to LSP server infrastructure while handling only TSP requests
pub struct TspServer<T: TspInterface> {
    pub inner: Arc<T>,
    /// Current snapshot version, updated on RecheckFinished events
    pub(crate) current_snapshot: Arc<Mutex<i32>>,
    response_sender: crossbeam_channel::Sender<Message>,
    extra_connections: Arc<Mutex<HashMap<String, ExtraConnectionHandle>>>,
    role: ConnectionRole,
}

impl<T: TspInterface + 'static> TspServer<T> {
    pub fn new(lsp_server: T) -> Self {
        let inner = Arc::new(lsp_server);
        let response_sender = inner.sender().clone();
        Self::with_connection(
            inner,
            Arc::new(Mutex::new(0)),
            response_sender,
            Arc::new(Mutex::new(HashMap::new())),
            ConnectionRole::Main,
        )
    }

    fn with_connection(
        inner: Arc<T>,
        current_snapshot: Arc<Mutex<i32>>,
        response_sender: crossbeam_channel::Sender<Message>,
        extra_connections: Arc<Mutex<HashMap<String, ExtraConnectionHandle>>>,
        role: ConnectionRole,
    ) -> Self {
        Self {
            inner,
            current_snapshot,
            response_sender,
            extra_connections,
            role,
        }
    }

    /// Convert a pyrefly `Type` to a TSP protocol `Type`, resolving function
    /// declaration ranges via the binding table.
    pub(crate) fn convert_type(&self, ty: &pyrefly_types::types::Type) -> tsp_types::Type {
        let resolver =
            |func_id: &pyrefly_types::callable::FuncId| self.inner.resolve_func_def_range(func_id);
        convert_type_with_resolver(ty, &resolver)
    }

    pub fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        telemetry: &'a impl Telemetry,
        telemetry_event: &mut TelemetryEvent,
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent> {
        // Remember if this event should increment the snapshot after processing
        let should_increment_snapshot = match &event {
            LspEvent::RecheckFinished => true,
            // Increment on DidChange since it affects type checker state via synchronous validation
            LspEvent::DidChangeTextDocument(_) => true,
            // Don't increment on DidChangeWatchedFiles directly since it triggers RecheckFinished
            // LspEvent::DidChangeWatchedFiles => true,
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
            self.send_response(Response::new_err(
                request.id.clone(),
                ErrorCode::MethodNotFound as i32,
                format!("TSP server does not support LSP method: {}", request.method),
            ));
            return Ok(ProcessEvent::Continue);
        }

        // For all other events (notifications, responses, etc.), delegate to inner server
        let result = self.inner.process_event(
            ide_transaction_manager,
            canceled_requests,
            telemetry,
            telemetry_event,
            subsequent_mutation,
            event,
        )?;

        // Increment snapshot after the inner server has processed the event
        if should_increment_snapshot && let Ok(mut current) = self.current_snapshot.lock() {
            let old_snapshot = *current;
            *current += 1;
            let new_snapshot = *current;
            drop(current); // Release the lock before sending the notification
            self.send_snapshot_changed_notification(old_snapshot, new_snapshot);
        }

        Ok(result)
    }

    /// Send a `typeServer/snapshotChanged` notification to the client.
    ///
    /// Called whenever the snapshot counter increments, so the client knows
    /// any previously-returned types are stale.
    fn send_snapshot_changed_notification(&self, old_snapshot: i32, new_snapshot: i32) {
        let method = serde_json::to_value(TSPNotificationMethods::TypeServerSnapshotChanged)
            .expect("TSPNotificationMethods serialization is infallible");
        let method_str = method
            .as_str()
            .expect("TSPNotificationMethods serializes to a string")
            .to_owned();

        if self.role != ConnectionRole::Main {
            return;
        }

        if let Err(e) = self
            .response_sender
            .send(Message::Notification(Notification {
                method: method_str,
                params: serde_json::json!({ "old": old_snapshot, "new": new_snapshot }),
                activity_key: None,
            }))
        {
            warn!("Failed to send snapshotChanged notification: {e}");
        }
    }

    pub(crate) fn send_response(&self, response: Response) {
        if let Err(error) = self.response_sender.send(Message::Response(response)) {
            warn!("Failed to send TSP response: {error}");
        }
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

        if self.role == ConnectionRole::ReadOnlyExtra
            && matches!(msg, TSPRequests::ConnectionRequest { .. })
        {
            self.send_err(
                request.id.clone(),
                ResponseError {
                    code: ErrorCode::InvalidRequest as i32,
                    message: format!(
                        "TSP method {} is only allowed on the main connection",
                        request.method
                    ),
                    data: None,
                },
            );
            return Ok(true);
        }

        match msg {
            TSPRequests::ConnectionRequest { params, .. } => {
                self.handle_connection_request(request.id.clone(), params);
                Ok(true)
            }
            TSPRequests::GetSupportedProtocolVersionRequest { .. } => {
                self.send_ok(request.id.clone(), self.get_supported_protocol_version());
                Ok(true)
            }
            TSPRequests::GetSnapshotRequest { .. } => {
                // Get snapshot doesn't need a transaction since it just returns the cached value
                self.send_ok(request.id.clone(), self.get_snapshot());
                Ok(true)
            }
            TSPRequests::ResolveImportRequest { params, .. } => {
                self.handle_resolve_import(request.id.clone(), params, ide_transaction_manager);
                Ok(true)
            }
            TSPRequests::GetPythonSearchPathsRequest { params, .. } => {
                self.handle_get_python_search_paths(request.id.clone(), params);
                Ok(true)
            }
            TSPRequests::GetDeclaredTypeRequest { params, .. } => {
                self.dispatch_get_type_request(request.id.clone(), params, |s, p| {
                    s.handle_get_declared_type(p)
                });
                Ok(true)
            }
            TSPRequests::GetComputedTypeRequest { params, .. } => {
                self.dispatch_get_type_request(request.id.clone(), params, |s, p| {
                    s.handle_get_computed_type(p)
                });
                Ok(true)
            }
            TSPRequests::GetExpectedTypeRequest { params, .. } => {
                self.dispatch_get_type_request(request.id.clone(), params, |s, p| {
                    s.handle_get_expected_type(p)
                });
                Ok(true)
            }
        }
    }

    /// Deserialize `serde_json::Value` params into [`GetTypeParams`], call the
    /// handler, and send the response. Shared by getDeclaredType,
    /// getComputedType, and getExpectedType.
    fn dispatch_get_type_request(
        &self,
        id: RequestId,
        raw_params: serde_json::Value,
        handler: impl FnOnce(
            &Self,
            GetTypeParams,
        ) -> Result<Option<tsp_types::Type>, lsp_server::ResponseError>,
    ) {
        let params: GetTypeParams = match serde_json::from_value::<GetTypeParams>(raw_params) {
            Ok(p) => p,
            Err(e) => {
                self.send_err(
                    id,
                    crate::tsp::validation::invalid_params_error(&e.to_string()),
                );
                return;
            }
        };
        match handler(self, params) {
            Ok(result) => {
                self.send_ok(id, result);
            }
            Err(err) => {
                self.send_err(id, err);
            }
        }
    }

    fn handle_connection_request(&self, id: RequestId, params: ConnectionRequestParams) {
        if self.role != ConnectionRole::Main {
            self.send_err(
                id,
                ResponseError {
                    code: ErrorCode::InvalidRequest as i32,
                    message: "Connection management is only supported on the main TSP connection"
                        .to_owned(),
                    data: None,
                },
            );
            return;
        }

        let result = match params.type_.as_str() {
            "open" => self.open_extra_connection(params),
            "close" => Ok(self.close_extra_connection(params)),
            other => Err(crate::tsp::validation::invalid_params_error(&format!(
                "Unsupported connection request type: {other}"
            ))),
        };

        match result {
            Ok(connection_result) => self.send_ok(id, connection_result),
            Err(error) => self.send_err(id, error),
        }
    }

    fn open_extra_connection(
        &self,
        params: ConnectionRequestParams,
    ) -> Result<ConnectionRequestResult, ResponseError> {
        let pipe_name = self.get_pipe_name(&params)?;

        let mut extra_connections = self.extra_connections.lock().map_err(|_| {
            crate::tsp::validation::internal_error("extra connection state was poisoned")
        })?;

        if extra_connections.contains_key(&pipe_name) {
            return Ok(ConnectionRequestResult {
                success: true,
                message: Some(format!("Extra connection already open: {pipe_name}")),
            });
        }

        let (connection, mut reader, _io_thread) =
            Connection::ipc(&pipe_name).map_err(|error| {
                crate::tsp::validation::internal_error(&format!(
                    "Failed to connect to IPC endpoint {pipe_name}: {error}"
                ))
            })?;

        let extra_server = Self::with_connection(
            self.inner.clone(),
            self.current_snapshot.clone(),
            connection.sender.clone(),
            self.extra_connections.clone(),
            ConnectionRole::ReadOnlyExtra,
        );
        let (message_tx, message_rx) = crossbeam_channel::unbounded();
        let (close_tx, close_rx) = crossbeam_channel::bounded::<()>(1);
        let pipe_name_for_thread = pipe_name.clone();

        extra_connections.insert(
            pipe_name.clone(),
            ExtraConnectionHandle {
                close_tx: close_tx.clone(),
            },
        );
        drop(extra_connections);

        std::thread::spawn(move || {
            std::thread::spawn(move || {
                while let Some(message) = reader.recv() {
                    if message_tx.send(message).is_err() {
                        break;
                    }
                }
            });

            loop {
                crossbeam_channel::select! {
                    recv(close_rx) -> _ => break,
                    recv(message_rx) -> message => {
                        let Ok(message) = message else {
                            break;
                        };

                        match message {
                            Message::Request(request) => {
                                let mut ide_transaction_manager = TransactionManager::default();
                                if let Err(error) = extra_server.handle_extra_request(&mut ide_transaction_manager, request) {
                                    warn!("Extra TSP connection exited with error: {error}");
                                    break;
                                }
                            }
                            Message::Notification(_) | Message::Response(_) => {
                                // Extra connections are read-only query channels.
                            }
                        }
                    }
                }
            }

            if let Ok(mut handles) = extra_server.extra_connections.lock() {
                handles.remove(&pipe_name_for_thread);
            }
        });

        Ok(ConnectionRequestResult {
            success: true,
            message: Some(format!("Opened extra IPC connection: {pipe_name}")),
        })
    }

    fn close_extra_connection(&self, params: ConnectionRequestParams) -> ConnectionRequestResult {
        let Ok(pipe_name) = self.get_pipe_name(&params) else {
            return ConnectionRequestResult {
                success: false,
                message: Some("Missing IPC pipe name in connection args".to_owned()),
            };
        };

        let handle = self
            .extra_connections
            .lock()
            .ok()
            .and_then(|mut handles| handles.remove(&pipe_name));

        if let Some(handle) = handle {
            let _ = handle.close_tx.send(());
            ConnectionRequestResult {
                success: true,
                message: Some(format!("Closing extra IPC connection: {pipe_name}")),
            }
        } else {
            ConnectionRequestResult {
                success: true,
                message: Some(format!("Extra IPC connection already closed: {pipe_name}")),
            }
        }
    }

    fn get_pipe_name(&self, params: &ConnectionRequestParams) -> Result<String, ResponseError> {
        if params.kind != ConnectionTransportKind::Ipc {
            return Err(crate::tsp::validation::invalid_params_error(
                "Only IPC extra connections are supported",
            ));
        }

        params
            .args
            .as_ref()
            .and_then(|args| args.first())
            .filter(|pipe_name| !pipe_name.is_empty())
            .cloned()
            .ok_or_else(|| {
                crate::tsp::validation::invalid_params_error(
                    "Connection request args must include the IPC pipe name",
                )
            })
    }

    fn handle_extra_request<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        request: Request,
    ) -> anyhow::Result<()> {
        if !self.handle_tsp_request(ide_transaction_manager, &request)? {
            self.send_response(Response::new_err(
                request.id,
                ErrorCode::MethodNotFound as i32,
                format!(
                    "Extra TSP connection does not support method: {}",
                    request.method
                ),
            ));
        }

        Ok(())
    }
}

pub fn tsp_loop(
    lsp_server: impl TspInterface + 'static,
    mut reader: MessageReader,
    _initialization: InitializeInfo,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    let server = TspServer::new(lsp_server);

    std::thread::scope(|scope| {
        // Start the recheck queue thread to process async tasks
        scope.spawn(|| server.inner.run_recheck_queue(telemetry));

        scope.spawn(|| {
            server.inner.dispatch_lsp_events(&mut reader);
        });

        let mut ide_transaction_manager = TransactionManager::default();
        let mut canceled_requests = HashSet::new();
        let mut next_task_id = 0_usize;

        while let Ok((subsequent_mutation, event, enqueued_at)) = server.inner.lsp_queue().recv() {
            let task_id = next_task_id;
            next_task_id += 1;
            let (mut event_telemetry, queue_duration) = TelemetryEvent::new_dequeued(
                TelemetryEventKind::LspEvent(event.describe()),
                enqueued_at,
                server.inner.telemetry_state(),
                QueueName::LspQueue,
                task_id,
            );
            let event_description = event.describe();

            let result = server.process_event(
                &mut ide_transaction_manager,
                &mut canceled_requests,
                telemetry,
                &mut event_telemetry,
                subsequent_mutation,
                event,
            );
            let process_duration =
                event_telemetry.finish_and_record(telemetry, result.as_ref().err());
            match result? {
                ProcessEvent::Continue => {
                    info!(
                        "Type server processed event `{}` in {:.2}s ({:.2}s waiting)",
                        event_description,
                        process_duration.as_secs_f32(),
                        queue_duration.as_secs_f32()
                    );
                }
                ProcessEvent::Exit => break,
            }
        }

        server.inner.stop_recheck_queue();
        Ok(())
    })
}

/// Generate TSP-specific server capabilities using the same capabilities as LSP
pub fn tsp_capabilities(
    indexing_mode: IndexingMode,
    initialization_params: &InitializeParams,
) -> ServerCapabilitiesWithTypeHierarchy {
    let mut result = capabilities(indexing_mode, initialization_params);
    result.base.experimental = Some(serde_json::json!({
        "typeServerMultiConnection": {
            "supportedTransports": ["ipc"]
        }
    }));
    result
}
