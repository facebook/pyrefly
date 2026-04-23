/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Deref;
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
use serde::Serialize;
use tracing::info;
use tracing::warn;
use tsp_types::ConnectionRequestParams;
use tsp_types::ConnectionRequestResult;
use tsp_types::ConnectionTransportKind;
use tsp_types::GetTypeParams;
use tsp_types::TSPNotificationMethods;
use tsp_types::TSPRequests;

use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::lsp::new_response;
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
use crate::tsp::validation::internal_error;
use crate::tsp::validation::invalid_params_error;
use crate::tsp::validation::snapshot_outdated_error;

struct ExtraConnectionHandle {
    close_tx: crossbeam_channel::Sender<()>,
    response_sender: crossbeam_channel::Sender<Message>,
}

pub struct TspServer<T: TspInterface> {
    pub(crate) inner: Arc<T>,
    /// Current snapshot version, updated on RecheckFinished events.
    pub(crate) current_snapshot: Arc<Mutex<i32>>,
    extra_connections: Mutex<HashMap<String, ExtraConnectionHandle>>,
}

// Runs the TSP server.
impl<T: TspInterface> TspServer<T> {
    fn new(lsp_server: T) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(lsp_server),
            current_snapshot: Arc::new(Mutex::new(0)),
            extra_connections: Mutex::new(HashMap::new()),
        })
    }

    /// Send a `snapshotChanged` notification to the main connection and every extra connection.
    fn broadcast_snapshot_changed(
        &self,
        main_sender: &crossbeam_channel::Sender<Message>,
        old_snapshot: i32,
        new_snapshot: i32,
    ) {
        let notification = snapshot_changed_notification(old_snapshot, new_snapshot);
        if let Err(e) = main_sender.send(Message::Notification(notification.clone())) {
            warn!("Failed to send snapshotChanged notification: {e}");
        }
        let handles = self
            .extra_connections
            .lock()
            .expect("extra_connections mutex poisoned");
        for (name, handle) in handles.iter() {
            if let Err(e) = handle
                .response_sender
                .send(Message::Notification(notification.clone()))
            {
                warn!("Failed to send snapshotChanged to extra connection {name}: {e}");
            }
        }
    }
}

/// A single JSON-RPC connection to the TSP server.
///
/// Each connection has its own response channel but shares the underlying
/// `TspServer` core with all other connections.
pub struct TspConnection<T: TspInterface> {
    pub(crate) server: Arc<TspServer<T>>,
    response_sender: crossbeam_channel::Sender<Message>,
}

impl<T: TspInterface> TspConnection<T> {
    fn new(server: Arc<TspServer<T>>, response_sender: crossbeam_channel::Sender<Message>) -> Self {
        Self {
            server,
            response_sender,
        }
    }

    /// Convenience accessor for the inner LSP server.
    pub(crate) fn inner(&self) -> &T {
        &self.server.inner
    }

    /// Convert a pyrefly `Type` to a TSP protocol `Type`, resolving function
    /// declaration ranges via the binding table.
    pub(crate) fn convert_type(&self, ty: &pyrefly_types::types::Type) -> tsp_types::Type {
        let resolver = |func_id: &pyrefly_types::callable::FuncId| {
            self.inner().resolve_func_def_range(func_id)
        };
        convert_type_with_resolver(ty, &resolver)
    }

    pub(crate) fn send_response(&self, response: Response) {
        if let Err(error) = self.response_sender.send(Message::Response(response)) {
            warn!("Failed to send TSP response: {error}");
        }
    }

    /// Send a successful JSON-RPC response for `id` with `result`.
    pub(crate) fn send_ok<R: Serialize>(&self, id: RequestId, result: R) {
        self.send_response(new_response(id, Ok(result)));
    }

    /// Send a JSON-RPC error response for `id`.
    pub(crate) fn send_err(&self, id: RequestId, error: ResponseError) {
        self.send_response(Response {
            id,
            result: None,
            error: Some(error),
        });
    }

    /// Validate that the client-supplied snapshot matches the server's current
    /// snapshot. Returns `Ok(())` on match or `Err(ResponseError)` on mismatch.
    pub(crate) fn validate_snapshot(&self, client_snapshot: i32) -> Result<(), ResponseError> {
        let current = self.get_snapshot();
        if client_snapshot != current {
            Err(snapshot_outdated_error(client_snapshot, current))
        } else {
            Ok(())
        }
    }

    fn dispatch_tsp_request<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        request: &Request,
        msg: TSPRequests,
    ) -> anyhow::Result<bool> {
        match msg {
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
            TSPRequests::ConnectionRequest { .. } => {
                // Multi-connection management is handled at the transport layer,
                // not inside the TSP request loop.
                unreachable!("ConnectionRequest should be handled before reaching the TSP server")
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
                self.send_err(id, invalid_params_error(&e.to_string()));
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
}

/// The main (stdio) connection. Only this type can manage extra connections
/// and trigger `snapshotChanged` notifications.
pub struct TspMainConnection<T: TspInterface>(TspConnection<T>);

impl<T: TspInterface> TspMainConnection<T> {
    fn new(server: Arc<TspServer<T>>, response_sender: crossbeam_channel::Sender<Message>) -> Self {
        Self(TspConnection::new(server, response_sender))
    }
}

impl<T: TspInterface> Deref for TspMainConnection<T> {
    type Target = TspConnection<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: TspInterface> TspMainConnection<T> {
    /// Process a single event on the main connection.
    fn process_event<'a>(
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
            match parse_tsp_request(request) {
                Some(TSPRequests::ConnectionRequest { params, .. }) => {
                    self.handle_connection_request(request.id.clone(), params);
                }
                Some(msg) => {
                    self.dispatch_tsp_request(ide_transaction_manager, request, msg)?;
                }
                None => {
                    self.send_response(Response::new_err(
                        request.id.clone(),
                        ErrorCode::MethodNotFound as i32,
                        format!("TSP server does not support LSP method: {}", request.method),
                    ));
                }
            }
            return Ok(ProcessEvent::Continue);
        }

        let result = self.inner().process_event(
            ide_transaction_manager,
            canceled_requests,
            telemetry,
            telemetry_event,
            subsequent_mutation,
            event,
        )?;

        // Increment snapshot after the inner server has processed the event
        if should_increment_snapshot {
            let mut current = self
                .server
                .current_snapshot
                .lock()
                .expect("current_snapshot mutex poisoned");
            let old_snapshot = *current;
            *current += 1;
            let new_snapshot = *current;
            drop(current);
            self.server.broadcast_snapshot_changed(
                &self.0.response_sender,
                old_snapshot,
                new_snapshot,
            );
        }

        Ok(result)
    }

    fn handle_connection_request(&self, id: RequestId, params: ConnectionRequestParams) {
        let result = match params.type_.as_str() {
            "open" => self.open_extra_connection(params),
            "close" => Ok(self.close_extra_connection(params)),
            other => Err(invalid_params_error(&format!(
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
        let name = pipe_name(&params)?;

        let mut extra_connections = self
            .server
            .extra_connections
            .lock()
            .map_err(|_| internal_error("extra connection state was poisoned"))?;

        if extra_connections.contains_key(name) {
            return Ok(ConnectionRequestResult {
                success: true,
                message: Some(format!("Extra connection already open: {name}")),
            });
        }

        // IoThread owns the writer JoinHandle. Dropping it detaches the thread
        // (no Drop impl), but the writer stays alive as long as the channel
        // sender (`extra_sender`) is alive — stored in ExtraConnectionHandle.
        let (ipc_connection, reader, _io_thread) = Connection::ipc(name).map_err(|error| {
            internal_error(&format!(
                "Failed to connect to IPC endpoint {name}: {error}"
            ))
        })?;

        let extra_sender = ipc_connection.sender.clone();
        let extra_conn = TspExtraConnection::new(self.server.clone(), extra_sender.clone());
        let (close_tx, close_rx) = crossbeam_channel::bounded::<()>(1);
        let name_owned = name.to_owned();

        extra_connections.insert(
            name_owned.clone(),
            ExtraConnectionHandle {
                close_tx,
                response_sender: extra_sender,
            },
        );
        drop(extra_connections);

        extra_conn.run(reader, close_rx, name_owned);

        Ok(ConnectionRequestResult {
            success: true,
            message: Some(format!("Opened extra IPC connection: {name}")),
        })
    }

    /// Close is idempotent: closing an already-closed connection succeeds.
    fn close_extra_connection(&self, params: ConnectionRequestParams) -> ConnectionRequestResult {
        let Ok(name) = pipe_name(&params) else {
            return ConnectionRequestResult {
                success: false,
                message: Some("Missing IPC pipe name in connection args".to_owned()),
            };
        };

        let handle = self
            .server
            .extra_connections
            .lock()
            .expect("extra_connections mutex poisoned")
            .remove(name);

        if let Some(handle) = handle {
            let _ = handle.close_tx.send(());
            ConnectionRequestResult {
                success: true,
                message: Some(format!("Closing extra IPC connection: {name}")),
            }
        } else {
            ConnectionRequestResult {
                success: true,
                message: Some(format!("Extra IPC connection already closed: {name}")),
            }
        }
    }
}

/// An extra (IPC) connection. Can handle TSP query requests but cannot
/// manage connections or process LSP lifecycle events.
struct TspExtraConnection<T: TspInterface>(TspConnection<T>);

impl<T: TspInterface> TspExtraConnection<T> {
    fn new(server: Arc<TspServer<T>>, response_sender: crossbeam_channel::Sender<Message>) -> Self {
        Self(TspConnection::new(server, response_sender))
    }
}

impl<T: TspInterface> TspExtraConnection<T> {
    /// Run the request loop for this extra connection until closed or
    /// the IPC pipe disconnects. Consumes `self` because the connection
    /// is moved into the spawned thread.
    fn run(
        self,
        mut reader: MessageReader,
        close_rx: crossbeam_channel::Receiver<()>,
        pipe_name: String,
    ) {
        let (message_tx, message_rx) = crossbeam_channel::unbounded();

        std::thread::spawn(move || {
            std::thread::spawn(move || {
                while let Some(message) = reader.recv() {
                    if message_tx.send(message).is_err() {
                        break;
                    }
                }
            });

            let mut selector = crossbeam_channel::Select::new();
            let close_index = selector.recv(&close_rx);
            let message_index = selector.recv(&message_rx);
            loop {
                let selected = selector.select();
                match selected.index() {
                    i if i == close_index => break,
                    i if i == message_index => {
                        let Ok(message) = selected.recv(&message_rx) else {
                            break;
                        };

                        match message {
                            Message::Request(request) => {
                                let mut tm = TransactionManager::default();
                                match parse_tsp_request(&request) {
                                    Some(TSPRequests::ConnectionRequest { .. }) => {
                                        self.send_err(
                                            request.id,
                                            ResponseError {
                                                code: ErrorCode::InvalidRequest as i32,
                                                message: format!(
                                                    "TSP method {} is only allowed on the main connection",
                                                    request.method
                                                ),
                                                data: None,
                                            },
                                        );
                                    }
                                    Some(msg) => {
                                        if let Err(error) =
                                            self.dispatch_tsp_request(&mut tm, &request, msg)
                                        {
                                            warn!("Extra TSP connection error: {error}");
                                            break;
                                        }
                                    }
                                    None => {
                                        self.send_response(Response::new_err(
                                            request.id,
                                            ErrorCode::MethodNotFound as i32,
                                            format!(
                                                "Extra TSP connection does not support method: {}",
                                                request.method
                                            ),
                                        ));
                                    }
                                }
                            }
                            Message::Notification(_) | Message::Response(_) => {}
                        }
                    }
                    _ => unreachable!(),
                }
            }

            self.server
                .extra_connections
                .lock()
                .expect("extra_connections mutex poisoned")
                .remove(&pipe_name);
        });
    }
}

impl<T: TspInterface> Deref for TspExtraConnection<T> {
    type Target = TspConnection<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Build a `typeServer/snapshotChanged` notification.
fn snapshot_changed_notification(old_snapshot: i32, new_snapshot: i32) -> Notification {
    let method = serde_json::to_value(TSPNotificationMethods::TypeServerSnapshotChanged)
        .expect("TSPNotificationMethods serialization is infallible");
    let method_str = method
        .as_str()
        .expect("TSPNotificationMethods serializes to a string")
        .to_owned();
    Notification {
        method: method_str,
        params: serde_json::json!({ "old": old_snapshot, "new": new_snapshot }),
        activity_key: None,
    }
}

/// Try to parse a request as a `TSPRequests` enum variant.
fn parse_tsp_request(request: &Request) -> Option<TSPRequests> {
    let wrapper = serde_json::json!({
        "method": request.method,
        "id": request.id,
        "params": request.params
    });
    serde_json::from_value::<TSPRequests>(wrapper).ok()
}

/// Extract the IPC pipe name from connection request `args`, or return an error.
fn pipe_name(params: &ConnectionRequestParams) -> Result<&str, ResponseError> {
    if params.kind != ConnectionTransportKind::Ipc {
        return Err(invalid_params_error(
            "Only IPC extra connections are supported",
        ));
    }

    params
        .args
        .as_ref()
        .and_then(|args| args.first())
        .map(|s| s.as_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            invalid_params_error("Connection request args must include the IPC pipe name")
        })
}

pub fn tsp_loop(
    lsp_server: impl TspInterface,
    mut reader: MessageReader,
    _initialization: InitializeInfo,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    let server = TspServer::new(lsp_server);
    let main_conn = TspMainConnection::new(server.clone(), server.inner.sender().clone());

    std::thread::scope(|scope| {
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

            let result = main_conn.process_event(
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

/// Generate TSP-specific server capabilities.
pub fn tsp_capabilities(
    indexing_mode: IndexingMode,
    initialization_params: &InitializeParams,
) -> ServerCapabilitiesWithTypeHierarchy {
    let mut result = capabilities(indexing_mode, initialization_params);
    result.set_experimental(serde_json::json!({
        "typeServerMultiConnection": {
            "supportedTransports": ["ipc"]
        }
    }));
    result
}
