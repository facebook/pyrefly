/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZero;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use crossbeam_channel::Sender;
use dupe::Dupe;
use itertools::__std_iter::once;
use lsp_server::Connection;
use lsp_server::ErrorCode;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::ProtocolError;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::CodeAction;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOptions;
use lsp_types::CodeActionOrCommand;
use lsp_types::CodeActionParams;
use lsp_types::CodeActionProviderCapability;
use lsp_types::CodeActionResponse;
use lsp_types::CompletionList;
use lsp_types::CompletionOptions;
use lsp_types::CompletionParams;
use lsp_types::CompletionResponse;
use lsp_types::ConfigurationItem;
use lsp_types::ConfigurationParams;
use lsp_types::Diagnostic;
use lsp_types::DidChangeConfigurationParams;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidChangeWatchedFilesClientCapabilities;
use lsp_types::DidChangeWatchedFilesParams;
use lsp_types::DidChangeWatchedFilesRegistrationOptions;
use lsp_types::DidChangeWorkspaceFoldersParams;
use lsp_types::DidCloseTextDocumentParams;
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::DidSaveTextDocumentParams;
use lsp_types::DocumentDiagnosticParams;
use lsp_types::DocumentDiagnosticReport;
use lsp_types::DocumentHighlight;
use lsp_types::DocumentHighlightParams;
use lsp_types::DocumentSymbol;
use lsp_types::DocumentSymbolParams;
use lsp_types::DocumentSymbolResponse;
use lsp_types::FileSystemWatcher;
use lsp_types::FullDocumentDiagnosticReport;
use lsp_types::GlobPattern;
use lsp_types::GotoDefinitionParams;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::HoverParams;
use lsp_types::HoverProviderCapability;
use lsp_types::InitializeParams;
use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;
use lsp_types::InlayHintParams;
use lsp_types::Location;
use lsp_types::NumberOrString;
use lsp_types::OneOf;
use lsp_types::Position;
use lsp_types::PositionEncodingKind;
use lsp_types::PrepareRenameResponse;
use lsp_types::PublishDiagnosticsParams;
use lsp_types::Range;
use lsp_types::ReferenceParams;
use lsp_types::Registration;
use lsp_types::RegistrationParams;
use lsp_types::RelatedFullDocumentDiagnosticReport;
use lsp_types::RelativePattern;
use lsp_types::RenameOptions;
use lsp_types::RenameParams;
use lsp_types::SemanticTokens;
use lsp_types::SemanticTokensFullOptions;
use lsp_types::SemanticTokensOptions;
use lsp_types::SemanticTokensParams;
use lsp_types::SemanticTokensRangeParams;
use lsp_types::SemanticTokensRangeResult;
use lsp_types::SemanticTokensResult;
use lsp_types::SemanticTokensServerCapabilities;
use lsp_types::ServerCapabilities;
use lsp_types::SignatureHelp;
use lsp_types::SignatureHelpOptions;
use lsp_types::SignatureHelpParams;
use lsp_types::SymbolInformation;
use lsp_types::TextDocumentContentChangeEvent;
use lsp_types::TextDocumentPositionParams;
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::TextEdit;
use lsp_types::Unregistration;
use lsp_types::UnregistrationParams;
use lsp_types::Url;
use lsp_types::VersionedTextDocumentIdentifier;
use lsp_types::WatchKind;
use lsp_types::WorkspaceClientCapabilities;
use lsp_types::WorkspaceEdit;
use lsp_types::WorkspaceFoldersServerCapabilities;
use lsp_types::WorkspaceServerCapabilities;
use lsp_types::WorkspaceSymbolResponse;
use lsp_types::notification::Cancel;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidChangeWatchedFiles;
use lsp_types::notification::DidChangeWorkspaceFolders;
use lsp_types::notification::DidCloseTextDocument;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::notification::DidSaveTextDocument;
use lsp_types::notification::Exit;
use lsp_types::notification::Notification as _;
use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::Completion;
use lsp_types::request::DocumentDiagnosticRequest;
use lsp_types::request::DocumentHighlightRequest;
use lsp_types::request::DocumentSymbolRequest;
use lsp_types::request::GotoDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::PrepareRenameRequest;
use lsp_types::request::References;
use lsp_types::request::RegisterCapability;
use lsp_types::request::Rename;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::request::SemanticTokensRangeRequest;
use lsp_types::request::SignatureHelpRequest;
use lsp_types::request::UnregisterCapability;
use lsp_types::request::WorkspaceConfiguration;
use lsp_types::request::WorkspaceSymbolRequest;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::CancellationHandle;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::ThreadPool;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::SourceLocation;
use ruff_text_size::Ranged;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use starlark_map::small_map::SmallMap;

use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::LspArgs;
use crate::config::config::ConfigFile;
use crate::config::error_kind::Severity;
use crate::error::error::Error;
use crate::lsp::features::hover::get_hover;
use crate::lsp::module_helpers::handle_from_module_path;
use crate::lsp::module_helpers::make_open_handle;
use crate::lsp::module_helpers::module_info_to_uri;
use crate::lsp::module_helpers::to_real_path;
use crate::lsp::transaction_manager::IDETransactionManager;
use crate::lsp::workspace::PythonInfo;
use crate::lsp::workspace::Workspace;
use crate::lsp::workspace::Workspaces;
use crate::module::from_path::module_from_path;
use crate::state::handle::Handle;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::require::Require;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::state::State;
use crate::state::state::Transaction;

pub enum ServerEvent {
    // Part 1: Events that the server should try to handle first.
    /// Notify the server that recheck finishes, so server can revalidate all in-memory content
    /// based on the latest `State`.
    RecheckFinished,
    /// Inform the server that a request is cancelled.
    /// Server should know about this ASAP to avoid wasting time on cancelled requests.
    CancelRequest(RequestId),
    // Part 2: Events that can be queued in FIFO order and handled at a later time.
    DidOpenTextDocument(DidOpenTextDocumentParams),
    DidChangeTextDocument(DidChangeTextDocumentParams),
    DidCloseTextDocument(DidCloseTextDocumentParams),
    DidSaveTextDocument(DidSaveTextDocumentParams),
    DidChangeWatchedFiles(DidChangeWatchedFilesParams),
    DidChangeWorkspaceFolders(DidChangeWorkspaceFoldersParams),
    DidChangeConfiguration(DidChangeConfigurationParams),
    LspResponse(Response),
    LspRequest(Request),
    Exit,
}

#[derive(Clone, Dupe)]
struct ServerConnection(Arc<Connection>);

impl ServerConnection {
    fn send(&self, msg: Message) {
        if self.0.sender.send(msg).is_err() {
            // On error, we know the channel is closed.
            // https://docs.rs/crossbeam/latest/crossbeam/channel/struct.Sender.html#method.send
            eprintln!("Connection closed.");
        };
    }

    fn publish_diagnostics_for_uri(&self, uri: Url, diags: Vec<Diagnostic>, version: Option<i32>) {
        self.send(Message::Notification(
            new_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
                uri, diags, version,
            )),
        ));
    }

    fn publish_diagnostics(&self, diags: SmallMap<PathBuf, Vec<Diagnostic>>) {
        for (path, diags) in diags {
            let path = std::fs::canonicalize(&path).unwrap_or(path);
            match Url::from_file_path(&path) {
                Ok(uri) => self.publish_diagnostics_for_uri(uri, diags, None),
                Err(_) => eprint!("Unable to convert path to uri: {path:?}"),
            }
        }
    }
}

pub struct Server {
    connection: ServerConnection,
    /// A thread pool of size one for heavy read operations on the State
    async_state_read_threads: ThreadPool,
    priority_events_sender: Arc<Sender<ServerEvent>>,
    initialize_params: InitializeParams,
    indexing_mode: IndexingMode,
    state: Arc<State>,
    open_files: Arc<RwLock<HashMap<PathBuf, Arc<String>>>>,
    /// A set of configs where we have already indexed all the files within the config.
    indexed_configs: Mutex<HashSet<ArcId<ConfigFile>>>,
    cancellation_handles: Arc<Mutex<HashMap<RequestId, CancellationHandle>>>,
    workspaces: Arc<Workspaces>,
    outgoing_request_id: Arc<AtomicI32>,
    outgoing_requests: Mutex<HashMap<RequestId, Request>>,
    filewatcher_registered: Arc<AtomicBool>,
    version_info: Mutex<HashMap<PathBuf, i32>>,
}

/// At the time when we are ready to handle a new LSP event, it will help if we know the a list of
/// buffered requests and notifications ready to be processed, because we can potentially make smart
/// decisions (e.g. not process cancelled requests).
///
/// This function listens to the LSP events in the order they arrive, and dispatch them into event
/// channels with various priority:
/// - priority_events includes those that should be handled as soon as possible (e.g. know that a
///   request is cancelled)
/// - queued_events includes most of the other events.
pub fn dispatch_lsp_events(
    connection: &Connection,
    priority_events_sender: Arc<Sender<ServerEvent>>,
    queued_events_sender: Sender<ServerEvent>,
) {
    for msg in &connection.receiver {
        match msg {
            Message::Request(x) => {
                match connection.handle_shutdown(&x) {
                    Ok(is_shutdown) => {
                        if is_shutdown {
                            return;
                        }
                    }
                    Err(_) => {
                        return;
                    }
                }
                if queued_events_sender
                    .send(ServerEvent::LspRequest(x))
                    .is_err()
                {
                    return;
                }
            }
            Message::Response(x) => {
                if queued_events_sender
                    .send(ServerEvent::LspResponse(x))
                    .is_err()
                {
                    return;
                }
            }
            Message::Notification(x) => {
                let send_result = if let Some(params) = as_notification::<DidOpenTextDocument>(&x) {
                    queued_events_sender.send(ServerEvent::DidOpenTextDocument(params))
                } else if let Some(params) = as_notification::<DidChangeTextDocument>(&x) {
                    queued_events_sender.send(ServerEvent::DidChangeTextDocument(params))
                } else if let Some(params) = as_notification::<DidCloseTextDocument>(&x) {
                    queued_events_sender.send(ServerEvent::DidCloseTextDocument(params))
                } else if let Some(params) = as_notification::<DidSaveTextDocument>(&x) {
                    queued_events_sender.send(ServerEvent::DidSaveTextDocument(params))
                } else if let Some(params) = as_notification::<DidChangeWatchedFiles>(&x) {
                    queued_events_sender.send(ServerEvent::DidChangeWatchedFiles(params))
                } else if let Some(params) = as_notification::<DidChangeWorkspaceFolders>(&x) {
                    queued_events_sender.send(ServerEvent::DidChangeWorkspaceFolders(params))
                } else if let Some(params) = as_notification::<DidChangeConfiguration>(&x) {
                    queued_events_sender.send(ServerEvent::DidChangeConfiguration(params))
                } else if let Some(params) = as_notification::<Cancel>(&x) {
                    let id = match params.id {
                        NumberOrString::Number(i) => RequestId::from(i),
                        NumberOrString::String(s) => RequestId::from(s),
                    };
                    priority_events_sender.send(ServerEvent::CancelRequest(id))
                } else if as_notification::<Exit>(&x).is_some() {
                    queued_events_sender.send(ServerEvent::Exit)
                } else {
                    eprintln!("Unhandled notification: {x:?}");
                    Ok(())
                };
                if send_result.is_err() {
                    return;
                }
            }
        }
    }
}

pub fn initialize_connection(
    connection: &Connection,
    args: &LspArgs,
) -> Result<InitializeParams, ProtocolError> {
    let (request_id, initialization_params) = connection.initialize_start()?;
    let initialization_params: InitializeParams =
        serde_json::from_value(initialization_params).unwrap();
    let augments_syntax_tokens = initialization_params
        .capabilities
        .text_document
        .as_ref()
        .and_then(|c| c.semantic_tokens.as_ref())
        .and_then(|c| c.augments_syntax_tokens)
        .unwrap_or(false);
    // Run the server and wait for the two threads to end (typically by trigger LSP Exit event).
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        position_encoding: Some(PositionEncodingKind::UTF16),
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::INCREMENTAL,
        )),
        definition_provider: Some(OneOf::Left(true)),
        code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
            code_action_kinds: Some(vec![CodeActionKind::QUICKFIX]),
            ..Default::default()
        })),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_owned()]),
            ..Default::default()
        }),
        document_highlight_provider: Some(OneOf::Left(true)),
        // Find references won't work properly if we don't know all the files.
        references_provider: match args.indexing_mode {
            IndexingMode::None => None,
            IndexingMode::LazyNonBlockingBackground | IndexingMode::LazyBlocking => {
                Some(OneOf::Left(true))
            }
        },
        rename_provider: match args.indexing_mode {
            IndexingMode::None => None,
            IndexingMode::LazyNonBlockingBackground | IndexingMode::LazyBlocking => {
                Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: Default::default(),
                }))
            }
        },
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_owned(), ",".to_owned()]),
            ..Default::default()
        }),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        semantic_tokens_provider: if augments_syntax_tokens {
            // We currently only return partial tokens (e.g. no tokens for keywords right now).
            // If the client doesn't support `augments_syntax_tokens` to fallback baseline
            // syntax highlighting for tokens we don't provide, it will be a regression
            // (e.g. users might lose keyword highlighting).
            // Therefore, we should not produce semantic tokens if the client doesn't support `augments_syntax_tokens`.
            Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
                SemanticTokensOptions {
                    legend: SemanticTokensLegends::lsp_semantic_token_legends(),
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                    range: Some(true),
                    ..Default::default()
                },
            ))
        } else {
            None
        },
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(true)),
            }),
            file_operations: None,
        }),
        ..Default::default()
    })
    .unwrap();
    let initialize_data = serde_json::json!({
        "capabilities": server_capabilities,
    });

    connection.initialize_finish(request_id, initialize_data)?;
    Ok(initialization_params)
}

pub enum ProcessEvent {
    Continue,
    Exit,
}

const PYTHON_SECTION: &str = "python";

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyreflyClientConfig {
    disable_type_errors: Option<bool>,
    disable_language_services: Option<bool>,
    extra_paths: Option<Vec<PathBuf>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LspConfig {
    python_path: Option<String>,
    pyrefly: Option<PyreflyClientConfig>,
}

impl Server {
    const FILEWATCHER_ID: &str = "FILEWATCHER";

    /// Process the event and return next step.
    pub fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        event: ServerEvent,
    ) -> anyhow::Result<ProcessEvent> {
        match event {
            ServerEvent::Exit => {
                return Ok(ProcessEvent::Exit);
            }
            ServerEvent::RecheckFinished => {
                self.validate_in_memory(ide_transaction_manager)?;
            }
            ServerEvent::CancelRequest(id) => {
                eprintln!("We should cancel request {id:?}");
                if let Some(cancellation_handle) = self.cancellation_handles.lock().remove(&id) {
                    cancellation_handle.cancel();
                }
                canceled_requests.insert(id);
            }
            ServerEvent::DidOpenTextDocument(params) => {
                self.did_open(ide_transaction_manager, params)?;
            }
            ServerEvent::DidChangeTextDocument(params) => {
                self.did_change(ide_transaction_manager, params)?;
            }
            ServerEvent::DidCloseTextDocument(params) => {
                self.did_close(params)?;
            }
            ServerEvent::DidSaveTextDocument(params) => {
                self.did_save(params)?;
            }
            ServerEvent::DidChangeWatchedFiles(params) => {
                self.did_change_watched_files(params)?;
            }
            ServerEvent::DidChangeWorkspaceFolders(params) => {
                self.workspace_folders_changed(params);
            }
            ServerEvent::DidChangeConfiguration(params) => {
                self.did_change_configuration(ide_transaction_manager, params)?;
            }
            ServerEvent::LspResponse(x) => {
                if let Some(request) = self.outgoing_requests.lock().remove(&x.id) {
                    self.handle_response(ide_transaction_manager, &request, &x)?;
                } else {
                    eprintln!("Response for unknown request: {x:?}");
                }
            }
            ServerEvent::LspRequest(x) => {
                if canceled_requests.remove(&x.id) {
                    let message = format!("Request {} is canceled", x.id);
                    eprintln!("{message}");
                    self.send_response(Response::new_err(
                        x.id,
                        ErrorCode::RequestCanceled as i32,
                        message,
                    ));
                    return Ok(ProcessEvent::Continue);
                }
                eprintln!("Handling non-canceled request {} ({})", x.method, x.id);
                if let Some(params) = as_request::<GotoDefinition>(&x) {
                    let default_response = GotoDefinitionResponse::Array(Vec::new());
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self
                            .goto_definition(&transaction, params)
                            .unwrap_or(default_response)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<CodeActionRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.code_action(&transaction, params).unwrap_or_default()),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<Completion>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(x.id, self.completion(&transaction, params)));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<DocumentHighlightRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.document_highlight(&transaction, params)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<References>(&x) {
                    self.references(x.id, ide_transaction_manager, params);
                } else if let Some(params) = as_request::<PrepareRenameRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.prepare_rename(&transaction, params)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<Rename>(&x) {
                    self.rename(x.id, ide_transaction_manager, params);
                } else if let Some(params) = as_request::<SignatureHelpRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.signature_help(&transaction, params)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<HoverRequest>(&x) {
                    let default_response = Hover {
                        contents: HoverContents::Array(Vec::new()),
                        range: None,
                    };
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.hover(&transaction, params).unwrap_or(default_response)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<InlayHintRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.inlay_hints(&transaction, params).unwrap_or_default()),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<SemanticTokensFullRequest>(&x) {
                    let default_response = SemanticTokensResult::Tokens(SemanticTokens {
                        result_id: None,
                        data: Vec::new(),
                    });
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self
                            .semantic_tokens_full(&transaction, params)
                            .unwrap_or(default_response)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<SemanticTokensRangeRequest>(&x) {
                    let default_response = SemanticTokensRangeResult::Tokens(SemanticTokens {
                        result_id: None,
                        data: Vec::new(),
                    });
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self
                            .semantic_tokens_ranged(&transaction, params)
                            .unwrap_or(default_response)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<DocumentSymbolRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(DocumentSymbolResponse::Nested(
                            self.hierarchical_document_symbols(&transaction, params)
                                .unwrap_or_default(),
                        )),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<WorkspaceSymbolRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(WorkspaceSymbolResponse::Flat(
                            self.workspace_symbols(&transaction, &params.query),
                        )),
                    ));
                    ide_transaction_manager.save(transaction);
                } else if let Some(params) = as_request::<DocumentDiagnosticRequest>(&x) {
                    let transaction =
                        ide_transaction_manager.non_commitable_transaction(&self.state);
                    self.send_response(new_response(
                        x.id,
                        Ok(self.document_diagnostics(&transaction, params)),
                    ));
                    ide_transaction_manager.save(transaction);
                } else {
                    eprintln!("Unhandled request: {x:?}");
                }
            }
        }
        Ok(ProcessEvent::Continue)
    }

    pub fn new(
        connection: Arc<Connection>,
        priority_events_sender: Arc<Sender<ServerEvent>>,
        initialize_params: InitializeParams,
        indexing_mode: IndexingMode,
    ) -> Self {
        let folders = if let Some(capability) = &initialize_params.capabilities.workspace
            && let Some(true) = capability.workspace_folders
            && let Some(folders) = &initialize_params.workspace_folders
        {
            folders
                .iter()
                .map(|x| x.uri.to_file_path().unwrap())
                .collect()
        } else {
            Vec::new()
        };

        let workspaces = Arc::new(Workspaces::new(Workspace::default()));

        let config_finder =
            Workspaces::config_finder(&workspaces, workspaces.loaded_configs.clone());
        let s = Self {
            connection: ServerConnection(connection),
            async_state_read_threads: ThreadPool::with_thread_count(ThreadCount::NumThreads(
                NonZero::new(1).unwrap(),
            )),
            priority_events_sender,
            initialize_params,
            indexing_mode,
            state: Arc::new(State::new(config_finder)),
            open_files: Arc::new(RwLock::new(HashMap::new())),
            indexed_configs: Mutex::new(HashSet::new()),
            cancellation_handles: Arc::new(Mutex::new(HashMap::new())),
            workspaces,
            outgoing_request_id: Arc::new(AtomicI32::new(1)),
            outgoing_requests: Mutex::new(HashMap::new()),
            filewatcher_registered: Arc::new(AtomicBool::new(false)),
            version_info: Mutex::new(HashMap::new()),
        };
        s.configure(&folders, &[]);

        s
    }

    fn send_response(&self, x: Response) {
        self.connection.send(Message::Response(x))
    }

    fn send_request<T>(&self, params: T::Params)
    where
        T: lsp_types::request::Request,
    {
        let id = RequestId::from(self.outgoing_request_id.fetch_add(1, Ordering::SeqCst));
        let request = Request {
            id: id.clone(),
            method: T::METHOD.to_owned(),
            params: serde_json::to_value(params).unwrap(),
        };
        self.connection.send(Message::Request(request.clone()));
        self.outgoing_requests.lock().insert(id, request);
    }

    fn validate_in_memory_for_transaction(
        state: &State,
        open_files: &RwLock<HashMap<PathBuf, Arc<String>>>,
        transaction: &mut Transaction<'_>,
    ) -> Vec<(Handle, Require)> {
        let handles = open_files
            .read()
            .keys()
            .map(|x| (make_open_handle(state, x), Require::Everything))
            .collect::<Vec<_>>();
        transaction.set_memory(
            open_files
                .read()
                .iter()
                .map(|x| (x.0.clone(), Some(x.1.dupe())))
                .collect::<Vec<_>>(),
        );
        transaction.run(&handles);
        handles
    }

    fn get_diag_if_shown(
        &self,
        e: &Error,
        open_files: &HashMap<PathBuf, Arc<String>>,
    ) -> Option<(PathBuf, Diagnostic)> {
        if let Some(path) = to_real_path(e.path()) {
            // When no file covers this, we'll get the default configured config which includes "everything"
            // and excludes `.<file>`s.
            let config = self
                .state
                .config_finder()
                .python_file(ModuleName::unknown(), e.path());
            if open_files.contains_key(&path)
                && !config.project_excludes.covers(&path)
                && !self
                    .workspaces
                    .get_with(path.to_path_buf(), |w| w.disable_type_errors)
            {
                return Some((
                    path.to_path_buf(),
                    Diagnostic {
                        range: e.lined_buffer().to_lsp_range(e.range()),
                        severity: Some(match e.severity() {
                            Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
                            Severity::Warn => lsp_types::DiagnosticSeverity::WARNING,
                            Severity::Info => lsp_types::DiagnosticSeverity::INFORMATION,
                            // Ignored errors shouldn't be here
                            Severity::Ignore => lsp_types::DiagnosticSeverity::INFORMATION,
                        }),
                        source: Some("Pyrefly".to_owned()),
                        message: e.msg().to_owned(),
                        code: Some(lsp_types::NumberOrString::String(
                            e.error_kind().to_name().to_owned(),
                        )),
                        ..Default::default()
                    },
                ));
            }
        }
        None
    }

    fn validate_in_memory<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
    ) -> anyhow::Result<()> {
        let mut possibly_committable_transaction =
            ide_transaction_manager.get_possibly_committable_transaction(&self.state);
        let transaction = match &mut possibly_committable_transaction {
            Ok(transaction) => transaction.as_mut(),
            Err(transaction) => transaction,
        };
        let handles =
            Self::validate_in_memory_for_transaction(&self.state, &self.open_files, transaction);

        let publish = |transaction: &Transaction| {
            let mut diags: SmallMap<PathBuf, Vec<Diagnostic>> = SmallMap::new();
            let open_files = self.open_files.read();
            for x in open_files.keys() {
                diags.insert(x.as_path().to_owned(), Vec::new());
            }
            for e in transaction
                .get_errors(handles.iter().map(|(handle, _)| handle))
                .collect_errors()
                .shown
            {
                if let Some((path, diag)) = self.get_diag_if_shown(&e, &open_files) {
                    diags.entry(path.to_owned()).or_default().push(diag);
                }
            }
            self.connection.publish_diagnostics(diags);
        };

        match possibly_committable_transaction {
            Ok(transaction) => {
                self.state.commit_transaction(transaction);
                // In the case where we can commit transactions, `State` already has latest updates.
                // Therefore, we can compute errors from transactions freshly created from `State``.
                let transaction = self.state.transaction();
                publish(&transaction);
            }
            Err(transaction) => {
                // In the case where transaction cannot be committed because there is an ongoing
                // recheck, we still want to update the diagnostics. In this case, we compute them
                // from the transactions that won't be committed. It will still contain all the
                // up-to-date in-memory content, but can have stale main `State` content.
                publish(&transaction);
                ide_transaction_manager.save(transaction);
            }
        }

        Ok(())
    }

    fn populate_project_files_if_necessary(
        &self,
        config_to_populate_files: Option<ArcId<ConfigFile>>,
    ) {
        if let Some(config) = config_to_populate_files {
            match self.indexing_mode {
                IndexingMode::None => {}
                IndexingMode::LazyNonBlockingBackground => {
                    if self.indexed_configs.lock().insert(config.dupe()) {
                        let state = self.state.dupe();
                        let priority_events_sender = self.priority_events_sender.dupe();
                        std::thread::spawn(move || {
                            Self::populate_all_project_files_in_config(
                                config,
                                state,
                                priority_events_sender,
                            );
                        });
                    }
                }
                IndexingMode::LazyBlocking => {
                    if self.indexed_configs.lock().insert(config.dupe()) {
                        Self::populate_all_project_files_in_config(
                            config,
                            self.state.dupe(),
                            self.priority_events_sender.dupe(),
                        );
                    }
                }
            }
        }
    }

    /// Perform an invalidation of elements on `State` and commit them.
    /// Runs asynchronously. Returns immediately and may wait a while for a committable transaction.
    fn invalidate(&self, f: impl FnOnce(&mut Transaction) + Send + 'static) {
        let state = self.state.dupe();
        let priority_events_sender = self.priority_events_sender.dupe();
        let cancellation_handles = self.cancellation_handles.dupe();
        std::thread::spawn(move || {
            let mut transaction = state.new_committable_transaction(Require::Indexing, None);
            f(transaction.as_mut());
            // Commit will be blocked until there are no ongoing reads.
            // If we have some long running read jobs that can be cancelled, we should cancel them
            // to unblock committing transactions.
            for (_, cancellation_handle) in cancellation_handles.lock().drain() {
                cancellation_handle.cancel();
            }
            // we have to run, not just commit to process updates
            state.run_with_committing_transaction(transaction, &[]);
            // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
            // the main event loop of the server. As a result, the server can do a revalidation of
            // all the in-memory files based on the fresh main State as soon as possible.
            let _ = priority_events_sender.send(ServerEvent::RecheckFinished);
        });
    }

    fn validate_with_disk_invalidation(&self, invalidate_disk: Vec<PathBuf>) -> anyhow::Result<()> {
        if !invalidate_disk.is_empty() {
            self.invalidate(move |t| t.invalidate_disk(&invalidate_disk));
        }
        Ok(())
    }

    /// Certain IDE features (e.g. find-references) require us to know the dependency graph of the
    /// entire project to work. This blocking function should be called when we know that a project
    /// file is opened and if we intend to provide features like find-references, and should be
    /// called when config changes (currently this is a TODO).
    fn populate_all_project_files_in_config(
        config: ArcId<ConfigFile>,
        state: Arc<State>,
        priority_events_sender: Arc<Sender<ServerEvent>>,
    ) {
        let unknown = ModuleName::unknown();

        eprintln!("Populating all files in the config ({:?}).", config.root);
        let mut transaction = state.new_committable_transaction(Require::Indexing, None);

        let project_path_blobs = config.get_filtered_globs(None);
        let paths = project_path_blobs.files().unwrap_or_default();
        let mut handles = Vec::new();
        for path in paths {
            let module_path = ModulePath::filesystem(path.clone());
            let path_config = state.config_finder().python_file(unknown, &module_path);
            if config != path_config {
                continue;
            }
            let module_name = module_from_path(&path, path_config.search_path())
                .unwrap_or_else(ModuleName::unknown);
            handles.push((
                Handle::new(module_name, module_path, path_config.get_sys_info()),
                Require::Indexing,
            ));
        }

        eprintln!("Prepare to check {} files.", handles.len());
        transaction.as_mut().run(&handles);
        state.commit_transaction(transaction);
        // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
        // the main event loop of the server. As a result, the server can do a revalidation of
        // all the in-memory files based on the fresh main State as soon as possible.
        let _ = priority_events_sender.send(ServerEvent::RecheckFinished);
        eprintln!("Populated all files in the project path.");
    }

    fn did_save(&self, params: DidSaveTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_file_path().unwrap();
        self.validate_with_disk_invalidation(vec![uri])
    }

    fn did_open<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: DidOpenTextDocumentParams,
    ) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_file_path().unwrap();
        let config_to_populate_files = if self.indexing_mode != IndexingMode::None
            && let Some(directory) = uri.as_path().parent()
        {
            self.state.config_finder().directory(directory)
        } else {
            None
        };
        self.version_info
            .lock()
            .insert(uri.clone(), params.text_document.version);
        self.open_files
            .write()
            .insert(uri, Arc::new(params.text_document.text));
        self.validate_in_memory(ide_transaction_manager)?;
        self.populate_project_files_if_necessary(config_to_populate_files);
        // rewatch files in case we loaded or dropped any configs
        self.setup_file_watcher_if_necessary(
            self.workspaces
                .workspaces
                .read()
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .as_slice(),
        );
        Ok(())
    }

    fn did_change<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: DidChangeTextDocumentParams,
    ) -> anyhow::Result<()> {
        /// Convert lsp_types::Position to usize index for a given text.
        fn position_to_usize(
            position: lsp_types::Position,
            index: &LineIndex,
            source_text: &str,
        ) -> usize {
            let source_location = SourceLocation {
                line: OneIndexed::from_zero_indexed(position.line as usize),
                character_offset: OneIndexed::from_zero_indexed(position.character as usize),
            };
            let text_size = index.offset(
                source_location,
                source_text,
                ruff_source_file::PositionEncoding::Utf16,
            );
            text_size.to_usize()
        }

        let VersionedTextDocumentIdentifier { uri, version } = params.text_document;
        let file_path = uri.to_file_path().unwrap();

        let mut version_info = self.version_info.lock();
        let old_version = version_info.get(&file_path).unwrap_or(&0);
        if version < *old_version {
            return Err(anyhow::anyhow!(
                "Unexpected version in didChange notification: {version:?} is less than {old_version:?}"
            ));
        }
        version_info.insert(file_path.clone(), version);
        let mut new_text = String::from(self.open_files.read().get(&file_path).unwrap().as_ref());
        for change in params.content_changes {
            let TextDocumentContentChangeEvent { range, text, .. } = change;
            // If no range is given, we can full text replace.
            let Some(range) = range else {
                new_text = text;
                continue;
            };
            let index = LineIndex::from_source_text(&new_text);
            let start = position_to_usize(range.start, &index, &new_text);
            let end = position_to_usize(range.end, &index, &new_text);
            new_text.replace_range(start..end, &text);
        }
        self.open_files
            .write()
            .insert(file_path.clone(), Arc::new(new_text));
        self.validate_in_memory(ide_transaction_manager)
    }

    pub fn categorized_events(events: Vec<lsp_types::FileEvent>) -> CategorizedEvents {
        let mut created = Vec::new();
        let mut modified = Vec::new();
        let mut removed = Vec::new();
        let mut unknown = Vec::new();

        for event in events {
            match event.typ {
                lsp_types::FileChangeType::CREATED => {
                    created.push(event.uri.to_file_path().unwrap());
                }
                lsp_types::FileChangeType::CHANGED => {
                    modified.push(event.uri.to_file_path().unwrap());
                }
                lsp_types::FileChangeType::DELETED => {
                    removed.push(event.uri.to_file_path().unwrap());
                }
                _ => {
                    unknown.push(event.uri.to_file_path().unwrap());
                }
            }
        }

        CategorizedEvents {
            created,
            modified,
            removed,
            unknown,
        }
    }

    fn did_change_watched_files(&self, params: DidChangeWatchedFilesParams) -> anyhow::Result<()> {
        if !params.changes.is_empty() {
            self.invalidate(move |t| {
                t.invalidate_events(&Self::categorized_events(params.changes))
            });
        }
        // rewatch files in case we loaded or dropped any configs
        self.setup_file_watcher_if_necessary(
            self.workspaces
                .workspaces
                .read()
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .as_slice(),
        );
        Ok(())
    }

    fn did_close(&self, params: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_file_path().unwrap();
        self.version_info.lock().remove(&uri);
        self.open_files.write().remove(&uri);
        self.connection
            .publish_diagnostics_for_uri(params.text_document.uri, Vec::new(), None);
        Ok(())
    }

    fn workspace_folders_changed(&self, params: DidChangeWorkspaceFoldersParams) {
        let removed = params
            .event
            .removed
            .iter()
            .map(|x| x.uri.to_file_path().unwrap())
            .collect::<Vec<_>>();
        let added = params
            .event
            .added
            .iter()
            .map(|x| x.uri.to_file_path().unwrap())
            .collect::<Vec<_>>();

        self.configure(&added, &removed);
    }

    fn did_change_configuration<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: DidChangeConfigurationParams,
    ) -> anyhow::Result<()> {
        if let Some(workspace) = &self.initialize_params.capabilities.workspace
            && workspace.configuration == Some(true)
        {
            self.request_settings_for_all_workspaces();
            return Ok(());
        }

        let mut modified = false;
        if let Some(python) = params.settings.get(PYTHON_SECTION) {
            let config: LspConfig = serde_json::from_value(python.clone())?;
            self.apply_client_configuration(&mut modified, &None, config);
        }

        if modified {
            self.validate_in_memory(ide_transaction_manager)?;
        }
        Ok(())
    }

    /// Configure the server with a new set of workspace folders
    fn configure(&self, workspace_paths_added: &[PathBuf], workspace_paths_removed: &[PathBuf]) {
        let mut all_workspaces = Vec::new();
        {
            let mut workspaces = self.workspaces.workspaces.write();
            for x in workspace_paths_added {
                workspaces.insert(x.clone(), Workspace::new_with_default_env(x));
            }
            for x in workspace_paths_removed {
                workspaces.shift_remove(x);
            }
            workspaces
                .keys()
                .for_each(|uri| all_workspaces.push(uri.clone()));
        }
        self.setup_file_watcher_if_necessary(all_workspaces.as_slice());
        self.request_settings_for_all_workspaces();
    }

    /// Create a handle. Return None if the workspace has language services disabled (and thus you shouldn't do anything).
    fn make_handle_if_enabled(&self, uri: &Url) -> Option<Handle> {
        let path = uri.to_file_path().unwrap();
        self.workspaces.get_with(path.clone(), |workspace| {
            if workspace.disable_language_services {
                eprintln!("Skipping request - language services disabled");
                None
            } else {
                let module_path = if self.open_files.read().contains_key(&path) {
                    ModulePath::memory(path)
                } else {
                    ModulePath::filesystem(path)
                };
                Some(handle_from_module_path(&self.state, module_path))
            }
        })
    }

    fn to_lsp_location(&self, location: &TextRangeWithModule) -> Option<Location> {
        let TextRangeWithModule {
            module: definition_module_info,
            range,
        } = location;
        let uri = module_info_to_uri(definition_module_info)?;
        Some(Location {
            uri,
            range: definition_module_info.lined_buffer().to_lsp_range(*range),
        })
    }

    fn goto_definition(
        &self,
        transaction: &Transaction<'_>,
        params: GotoDefinitionParams,
    ) -> Option<GotoDefinitionResponse> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let range = info
            .lined_buffer()
            .from_lsp_position(params.text_document_position_params.position);
        let targets = transaction.goto_definition(&handle, range);
        let mut lsp_targets = targets
            .into_iter()
            .filter_map(|t| self.to_lsp_location(&t))
            .collect::<Vec<_>>();
        if lsp_targets.is_empty() {
            None
        } else if lsp_targets.len() == 1 {
            Some(GotoDefinitionResponse::Scalar(lsp_targets.pop().unwrap()))
        } else {
            Some(GotoDefinitionResponse::Array(lsp_targets))
        }
    }

    fn completion(
        &self,
        transaction: &Transaction<'_>,
        params: CompletionParams,
    ) -> anyhow::Result<CompletionResponse> {
        let uri = &params.text_document_position.text_document.uri;
        let handle = match self.make_handle_if_enabled(uri) {
            None => {
                return Ok(CompletionResponse::List(CompletionList {
                    is_incomplete: false,
                    items: Vec::new(),
                }));
            }
            Some(x) => x,
        };
        let items = transaction
            .get_module_info(&handle)
            .map(|info| {
                transaction.completion(
                    &handle,
                    info.lined_buffer()
                        .from_lsp_position(params.text_document_position.position),
                )
            })
            .unwrap_or_default();
        Ok(CompletionResponse::List(CompletionList {
            is_incomplete: false,
            items,
        }))
    }

    fn code_action(
        &self,
        transaction: &Transaction<'_>,
        params: CodeActionParams,
    ) -> Option<CodeActionResponse> {
        let uri = &params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let module_info = transaction.get_module_info(&handle)?;
        let range = module_info.lined_buffer().from_lsp_range(params.range);
        let code_actions = transaction
            .local_quickfix_code_actions(&handle, range)?
            .into_map(|(title, info, range, insert_text)| {
                CodeActionOrCommand::CodeAction(CodeAction {
                    title,
                    kind: Some(CodeActionKind::QUICKFIX),
                    edit: Some(WorkspaceEdit {
                        changes: Some(HashMap::from([(
                            uri.clone(),
                            vec![TextEdit {
                                range: info.lined_buffer().to_lsp_range(range),
                                new_text: insert_text,
                            }],
                        )])),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            });
        Some(code_actions)
    }

    fn document_highlight(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentHighlightParams,
    ) -> Option<Vec<DocumentHighlight>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let position = info
            .lined_buffer()
            .from_lsp_position(params.text_document_position_params.position);
        Some(
            transaction
                .find_local_references(&handle, position)
                .into_map(|range| DocumentHighlight {
                    range: info.lined_buffer().to_lsp_range(range),
                    kind: None,
                }),
        )
    }

    /// Compute references of a symbol at a given position. This is a non-blocking function, the
    /// it will send a response to the LSP client once the results are found and transformed by
    /// `map_result`.
    fn async_find_references_helper<'a, V: serde::Serialize>(
        &'a self,
        request_id: RequestId,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        uri: &Url,
        position: Position,
        map_result: impl FnOnce(Vec<(Url, Vec<Range>)>) -> V + Send + 'static,
    ) {
        let Some(handle) = self.make_handle_if_enabled(uri) else {
            return self.send_response(new_response::<Option<V>>(request_id, Ok(None)));
        };
        let transaction = ide_transaction_manager.non_commitable_transaction(&self.state);
        let Some(info) = transaction.get_module_info(&handle) else {
            ide_transaction_manager.save(transaction);
            return self.send_response(new_response::<Option<V>>(request_id, Ok(None)));
        };
        let position = info.lined_buffer().from_lsp_position(position);
        let Some(FindDefinitionItemWithDocstring {
            metadata,
            definition_range,
            module,
            docstring_range: _,
        }) = transaction
            .find_definition(&handle, position, false)
            // TODO: handle more than 1 definition
            .into_iter()
            .next()
        else {
            ide_transaction_manager.save(transaction);
            return self.send_response(new_response::<Option<V>>(request_id, Ok(None)));
        };
        ide_transaction_manager.save(transaction);
        let state = self.state.dupe();
        let open_files = self.open_files.dupe();
        let cancellation_handles = self.cancellation_handles.dupe();

        let connection = self.connection.dupe();
        self.async_state_read_threads.async_spawn(move || {
            let mut transaction = state.cancellable_transaction();
            cancellation_handles
                .lock()
                .insert(request_id.clone(), transaction.get_cancellation_handle());
            Self::validate_in_memory_for_transaction(&state, &open_files, transaction.as_mut());
            match transaction.find_global_references_from_definition(
                handle.sys_info(),
                metadata,
                TextRangeWithModule::new(module, definition_range),
            ) {
                Ok(global_references) => {
                    let mut locations = Vec::new();
                    for (info, ranges) in global_references {
                        if let Some(uri) = module_info_to_uri(&info) {
                            locations.push((
                                uri,
                                ranges.into_map(|range| info.lined_buffer().to_lsp_range(range)),
                            ));
                        };
                    }
                    connection.send(Message::Response(new_response(
                        request_id,
                        Ok(Some(map_result(locations))),
                    )));
                }
                Err(Cancelled) => {
                    let message = format!("Find reference request {request_id} is canceled");
                    eprintln!("{message}");
                    connection.send(Message::Response(Response::new_err(
                        request_id,
                        ErrorCode::RequestCanceled as i32,
                        message,
                    )))
                }
            }
        });
    }

    fn references<'a>(
        &'a self,
        request_id: RequestId,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: ReferenceParams,
    ) {
        self.async_find_references_helper(
            request_id,
            ide_transaction_manager,
            &params.text_document_position.text_document.uri,
            params.text_document_position.position,
            move |results| {
                let mut locations = Vec::new();
                for (uri, ranges) in results {
                    for range in ranges {
                        locations.push(Location {
                            uri: uri.clone(),
                            range,
                        })
                    }
                }
                locations
            },
        );
    }

    fn rename<'a>(
        &'a self,
        request_id: RequestId,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: RenameParams,
    ) {
        self.async_find_references_helper(
            request_id,
            ide_transaction_manager,
            &params.text_document_position.text_document.uri,
            params.text_document_position.position,
            move |results| {
                let mut changes = HashMap::new();
                for (uri, ranges) in results {
                    changes.insert(
                        uri,
                        ranges.into_map(|range| TextEdit {
                            range,
                            new_text: params.new_name.clone(),
                        }),
                    );
                }
                WorkspaceEdit {
                    changes: Some(changes),
                    ..Default::default()
                }
            },
        );
    }

    fn prepare_rename(
        &self,
        transaction: &Transaction<'_>,
        params: TextDocumentPositionParams,
    ) -> Option<PrepareRenameResponse> {
        let uri = &params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let position = info.lined_buffer().from_lsp_position(params.position);
        transaction
            .prepare_rename(&handle, position)
            .map(|range| PrepareRenameResponse::Range(info.lined_buffer().to_lsp_range(range)))
    }

    fn signature_help(
        &self,
        transaction: &Transaction<'_>,
        params: SignatureHelpParams,
    ) -> Option<SignatureHelp> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let position = info
            .lined_buffer()
            .from_lsp_position(params.text_document_position_params.position);
        transaction.get_signature_help_at(&handle, position)
    }

    fn hover(&self, transaction: &Transaction<'_>, params: HoverParams) -> Option<Hover> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let position = info
            .lined_buffer()
            .from_lsp_position(params.text_document_position_params.position);

        get_hover(transaction, &handle, position)
    }

    fn inlay_hints(
        &self,
        transaction: &Transaction<'_>,
        params: InlayHintParams,
    ) -> Option<Vec<InlayHint>> {
        let uri = &params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let t = transaction.inlay_hints(&handle)?;
        Some(t.into_map(|x| {
            let position = info.lined_buffer().to_lsp_position(x.0);
            InlayHint {
                position,
                label: InlayHintLabel::String(x.1.clone()),
                kind: None,
                text_edits: Some(vec![TextEdit {
                    range: Range::new(position, position),
                    new_text: x.1,
                }]),
                tooltip: None,
                padding_left: None,
                padding_right: None,
                data: None,
            }
        }))
    }

    fn semantic_tokens_full(
        &self,
        transaction: &Transaction<'_>,
        params: SemanticTokensParams,
    ) -> Option<SemanticTokensResult> {
        let uri = &params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: transaction
                .semantic_tokens(&handle, None)
                .unwrap_or_default(),
        }))
    }

    fn semantic_tokens_ranged(
        &self,
        transaction: &Transaction<'_>,
        params: SemanticTokensRangeParams,
    ) -> Option<SemanticTokensRangeResult> {
        let uri = &params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let module_info = transaction.get_module_info(&handle)?;
        let range = module_info.lined_buffer().from_lsp_range(params.range);
        Some(SemanticTokensRangeResult::Tokens(SemanticTokens {
            result_id: None,
            data: transaction
                .semantic_tokens(&handle, Some(range))
                .unwrap_or_default(),
        }))
    }

    fn hierarchical_document_symbols(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentSymbolParams,
    ) -> Option<Vec<DocumentSymbol>> {
        let uri = &params.text_document.uri;
        if self
            .workspaces
            .get_with(uri.to_file_path().unwrap(), |workspace| {
                workspace.disable_language_services
            })
            || !self
                .initialize_params
                .capabilities
                .text_document
                .as_ref()?
                .document_symbol
                .as_ref()?
                .hierarchical_document_symbol_support?
        {
            return None;
        }
        let handle = self.make_handle_if_enabled(uri)?;
        transaction.symbols(&handle)
    }

    #[allow(deprecated)] // The `deprecated` field
    fn workspace_symbols(
        &self,
        transaction: &Transaction<'_>,
        query: &str,
    ) -> Vec<SymbolInformation> {
        transaction
            .workspace_symbols(query)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(name, kind, location)| {
                self.to_lsp_location(&location)
                    .map(|location| SymbolInformation {
                        name,
                        kind,
                        location,
                        tags: None,
                        deprecated: None,
                        container_name: None,
                    })
            })
            .collect()
    }

    fn document_diagnostics(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentDiagnosticParams,
    ) -> DocumentDiagnosticReport {
        let handle = make_open_handle(
            &self.state,
            &params.text_document.uri.to_file_path().unwrap(),
        );
        let mut items = Vec::new();
        let open_files = &self.open_files.read();
        for e in transaction.get_errors(once(&handle)).collect_errors().shown {
            if let Some((_, diag)) = self.get_diag_if_shown(&e, open_files) {
                items.push(diag);
            }
        }
        DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
            full_document_diagnostic_report: FullDocumentDiagnosticReport {
                items,
                result_id: None,
            },
            related_documents: None,
        })
    }

    fn get_pattern_to_watch(
        root: &Path,
        pattern: String,
        relative_pattern_support: bool,
    ) -> GlobPattern {
        if relative_pattern_support && let Ok(url) = Url::from_directory_path(root) {
            GlobPattern::Relative(RelativePattern {
                base_uri: OneOf::Right(url),
                pattern,
            })
        } else {
            GlobPattern::String(root.join(pattern).to_string_lossy().into_owned())
        }
    }

    fn setup_file_watcher_if_necessary(&self, roots: &[PathBuf]) {
        match self.initialize_params.capabilities.workspace {
            Some(WorkspaceClientCapabilities {
                did_change_watched_files:
                    Some(DidChangeWatchedFilesClientCapabilities {
                        dynamic_registration: Some(true),
                        relative_pattern_support,
                        ..
                    }),
                ..
            }) => {
                let relative_pattern_support = relative_pattern_support.is_some_and(|b| b);
                if self.filewatcher_registered.load(Ordering::Relaxed) {
                    self.send_request::<UnregisterCapability>(UnregistrationParams {
                        unregisterations: Vec::from([Unregistration {
                            id: Self::FILEWATCHER_ID.to_owned(),
                            method: DidChangeWatchedFiles::METHOD.to_owned(),
                        }]),
                    });
                }
                // TODO(connernilsen): we need to dedup filewatcher patterns
                // preferably by figuring out if they're under another wildcard pattern with the same suffix
                let mut glob_patterns = Vec::new();
                for root in roots {
                    PYTHON_EXTENSIONS.iter().for_each(|suffix| {
                        glob_patterns.push(Self::get_pattern_to_watch(
                            root,
                            format!("**/*.{suffix}"),
                            relative_pattern_support,
                        ));
                    });
                    ConfigFile::CONFIG_FILE_NAMES.iter().for_each(|config| {
                        glob_patterns.push(Self::get_pattern_to_watch(
                            root,
                            format!("**/{config}"),
                            relative_pattern_support,
                        ))
                    });
                }
                for (root, pattern) in self
                    .workspaces
                    .loaded_configs
                    .get_patterns_for_cached_configs()
                {
                    glob_patterns.push(Self::get_pattern_to_watch(
                        &root,
                        pattern,
                        relative_pattern_support,
                    ));
                }
                let watchers = glob_patterns
                    .into_iter()
                    .map(|glob_pattern| FileSystemWatcher {
                        glob_pattern,
                        kind: Some(WatchKind::Create | WatchKind::Change | WatchKind::Delete),
                    })
                    .collect::<Vec<_>>();
                self.send_request::<RegisterCapability>(RegistrationParams {
                    registrations: Vec::from([Registration {
                        id: Self::FILEWATCHER_ID.to_owned(),
                        method: DidChangeWatchedFiles::METHOD.to_owned(),
                        register_options: Some(
                            serde_json::to_value(DidChangeWatchedFilesRegistrationOptions {
                                watchers,
                            })
                            .unwrap(),
                        ),
                    }]),
                });
                self.filewatcher_registered.store(true, Ordering::Relaxed);
            }
            _ => (),
        }
    }

    fn request_settings_for_all_workspaces(&self) {
        if let Some(workspace) = &self.initialize_params.capabilities.workspace
            && workspace.configuration == Some(true)
        {
            self.send_request::<WorkspaceConfiguration>(ConfigurationParams {
                items: self
                    .workspaces
                    .workspaces
                    .read()
                    .keys()
                    .map(|uri| Some(Url::from_file_path(uri).unwrap()))
                    // add default workspace
                    .chain(once(None))
                    .map(|url| ConfigurationItem {
                        scope_uri: url,
                        section: Some(PYTHON_SECTION.to_owned()),
                    })
                    .collect::<Vec<_>>(),
            });
        }
    }

    fn handle_response<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        request: &Request,
        response: &Response,
    ) -> anyhow::Result<()> {
        if let Some((request, response)) =
            as_request_response_pair::<WorkspaceConfiguration>(request, response)
        {
            let mut modified = false;
            for (i, id) in request.items.iter().enumerate() {
                let config: LspConfig = if let Some(value) = response.get(i) {
                    serde_json::from_value(value.clone()).unwrap_or_default()
                } else {
                    continue;
                };
                self.apply_client_configuration(&mut modified, &id.scope_uri, config);
            }
            if modified {
                return self.validate_in_memory(ide_transaction_manager);
            }
        }
        Ok(())
    }

    /// Applies the LSP client configuration to the `scope_uri` (workspace) given.
    ///
    /// The `modified` flag is changed to `true` when the configuration gets applied to the
    /// `scope_uri` matching a valid workspace
    fn apply_client_configuration(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        config: LspConfig,
    ) {
        if let Some(python_path) = config.python_path {
            self.update_pythonpath(modified, scope_uri, &python_path);
        }

        if let Some(pyrefly) = config.pyrefly {
            if let Some(extra_paths) = pyrefly.extra_paths {
                self.update_search_paths(modified, scope_uri, extra_paths);
            }
            if let Some(disable_language_services) = pyrefly.disable_language_services {
                self.update_disable_language_services(scope_uri, disable_language_services);
            }
            if let Some(disable_type_errors) = pyrefly.disable_type_errors {
                self.update_disable_type_errors(modified, scope_uri, disable_type_errors);
            }
        }
    }

    /// Update disableLanguageServices setting for scope_uri, None if default workspace
    fn update_disable_language_services(
        &self,
        scope_uri: &Option<Url>,
        disable_language_services: bool,
    ) {
        let mut workspaces = self.workspaces.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    workspace.disable_language_services = disable_language_services;
                }
            }
            None => {
                self.workspaces.default.write().disable_language_services =
                    disable_language_services
            }
        }
    }

    /// Update typeCheckingMode setting for scope_uri, None if default workspace
    fn update_disable_type_errors(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        disable_type_errors: bool,
    ) {
        let mut workspaces = self.workspaces.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    *modified = true;
                    workspace.disable_type_errors = disable_type_errors;
                }
            }
            None => {
                *modified = true;
                self.workspaces.default.write().disable_type_errors = disable_type_errors
            }
        }
    }

    fn invalidate_config(&self) {
        self.invalidate(|t| t.invalidate_config());
    }

    /// Updates pythonpath with specified python path
    /// scope_uri = None for default workspace
    fn update_pythonpath(&self, modified: &mut bool, scope_uri: &Option<Url>, python_path: &str) {
        let mut workspaces = self.workspaces.workspaces.write();
        let interpreter = PathBuf::from(python_path);
        let python_info = Some(PythonInfo::new(interpreter));
        match scope_uri {
            Some(scope_uri) => {
                let workspace_path = scope_uri.to_file_path().unwrap();
                if let Some(workspace) = workspaces.get_mut(&workspace_path) {
                    *modified = true;
                    workspace.python_info = python_info;
                }
            }
            None => {
                *modified = true;
                self.workspaces.default.write().python_info = python_info;
            }
        }
        self.invalidate_config();
    }

    // Updates search paths for scope uri.
    fn update_search_paths(
        &self,
        modified: &mut bool,
        scope_uri: &Option<Url>,
        search_paths: Vec<PathBuf>,
    ) {
        let mut workspaces = self.workspaces.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                let workspace_path = scope_uri.to_file_path().unwrap();
                if let Some(workspace) = workspaces.get_mut(&workspace_path) {
                    *modified = true;
                    workspace.search_path = Some(search_paths);
                }
            }
            None => {
                *modified = true;
                self.workspaces.default.write().search_path = Some(search_paths);
            }
        }
        self.invalidate_config();
    }
}

fn as_notification<T>(x: &Notification) -> Option<T::Params>
where
    T: lsp_types::notification::Notification,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        let params = serde_json::from_value(x.params.clone()).unwrap_or_else(|err| {
            panic!(
                "Invalid notification\nMethod: {}\n error: {}",
                x.method, err
            )
        });
        Some(params)
    } else {
        None
    }
}

fn as_request<T>(x: &Request) -> Option<T::Params>
where
    T: lsp_types::request::Request,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        let params = serde_json::from_value(x.params.clone()).unwrap_or_else(|err| {
            panic!(
                "Invalid request\n  method: {}\n  error: {}\n  request: {:?}\n",
                x.method, err, x
            )
        });
        Some(params)
    } else {
        None
    }
}

fn as_request_response_pair<T>(
    request: &Request,
    response: &Response,
) -> Option<(T::Params, T::Result)>
where
    T: lsp_types::request::Request,
    T::Params: DeserializeOwned,
{
    if response.id != request.id {
        return None;
    }
    let params = as_request::<T>(request)?;
    let result = serde_json::from_value(response.result.clone()?).unwrap_or_else(|err| {
        panic!(
            "Invalid response\n  method: {}\n response:{:?}\n, response error:{:?}\n, error: {}\n",
            request.method, response.result, response.error, err
        )
    });
    Some((params, result))
}

/// Create a new `Notification` object with the correct name from the given params.
fn new_notification<T>(params: T::Params) -> Notification
where
    T: lsp_types::notification::Notification,
{
    Notification {
        method: T::METHOD.to_owned(),
        params: serde_json::to_value(&params).unwrap(),
    }
}

fn new_response<T>(id: RequestId, params: anyhow::Result<T>) -> Response
where
    T: serde::Serialize,
{
    match params {
        Ok(params) => Response {
            id,
            result: Some(serde_json::to_value(params).unwrap()),
            error: None,
        },
        Err(e) => Response {
            id,
            result: None,
            error: Some(ResponseError {
                code: 0,
                message: format!("{e:#?}"),
                data: None,
            }),
        },
    }
}
