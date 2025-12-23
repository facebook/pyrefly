/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::once;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use std::time::Instant;

use dupe::Dupe;
use dupe::OptionDupedExt;
use itertools::Itertools;
use lsp_server::Connection;
use lsp_server::ErrorCode;
use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::CallHierarchyServerCapability;
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
use lsp_types::DeclarationCapability;
use lsp_types::Diagnostic;
use lsp_types::DiagnosticSeverity;
use lsp_types::DiagnosticTag;
use lsp_types::DidChangeConfigurationParams;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidChangeWatchedFilesClientCapabilities;
use lsp_types::DidChangeWatchedFilesParams;
use lsp_types::DidChangeWatchedFilesRegistrationOptions;
use lsp_types::DidChangeWorkspaceFoldersParams;
use lsp_types::DocumentDiagnosticParams;
use lsp_types::DocumentDiagnosticReport;
use lsp_types::DocumentHighlight;
use lsp_types::DocumentHighlightParams;
use lsp_types::DocumentSymbol;
use lsp_types::DocumentSymbolParams;
use lsp_types::DocumentSymbolResponse;
use lsp_types::FileSystemWatcher;
use lsp_types::FoldingRange;
use lsp_types::FoldingRangeParams;
use lsp_types::FoldingRangeProviderCapability;
use lsp_types::FullDocumentDiagnosticReport;
use lsp_types::GlobPattern;
use lsp_types::GotoDefinitionParams;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::HoverParams;
use lsp_types::HoverProviderCapability;
use lsp_types::ImplementationProviderCapability;
use lsp_types::InitializeParams;
use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;
use lsp_types::InlayHintLabelPart;
use lsp_types::InlayHintParams;
use lsp_types::Location;
use lsp_types::NotebookCellSelector;
use lsp_types::NotebookDocumentSyncOptions;
use lsp_types::NotebookSelector;
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
use lsp_types::RenameFilesParams;
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
use lsp_types::TextDocumentIdentifier;
use lsp_types::TextDocumentPositionParams;
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::TextEdit;
use lsp_types::TypeDefinitionProviderCapability;
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
use lsp_types::request::CallHierarchyIncomingCalls;
use lsp_types::request::CallHierarchyOutgoingCalls;
use lsp_types::request::CallHierarchyPrepare;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::Completion;
use lsp_types::request::DocumentDiagnosticRequest;
use lsp_types::request::DocumentHighlightRequest;
use lsp_types::request::DocumentSymbolRequest;
use lsp_types::request::FoldingRangeRequest;
use lsp_types::request::GotoDeclaration;
use lsp_types::request::GotoDefinition;
use lsp_types::request::GotoImplementation;
use lsp_types::request::GotoImplementationParams;
use lsp_types::request::GotoImplementationResponse;
use lsp_types::request::GotoTypeDefinition;
use lsp_types::request::GotoTypeDefinitionParams;
use lsp_types::request::GotoTypeDefinitionResponse;
use lsp_types::request::HoverRequest;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::PrepareRenameRequest;
use lsp_types::request::References;
use lsp_types::request::RegisterCapability;
use lsp_types::request::Rename;
use lsp_types::request::Request as _;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::request::SemanticTokensRangeRequest;
use lsp_types::request::SemanticTokensRefresh;
use lsp_types::request::SignatureHelpRequest;
use lsp_types::request::UnregisterCapability;
use lsp_types::request::WillRenameFiles;
use lsp_types::request::WorkspaceConfiguration;
use lsp_types::request::WorkspaceSymbolRequest;
use pyrefly_build::SourceDatabase;
use pyrefly_build::handle::Handle;
use pyrefly_config::config::ConfigSource;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::includes::Includes as _;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::CancellationHandle;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryEventKind;
use pyrefly_util::telemetry::TelemetryFileStats;
use pyrefly_util::telemetry::TelemetryServerState;
use pyrefly_util::watch_pattern::WatchPattern;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;
use tracing::info;
use uuid::Uuid;

use crate::ModuleInfo;
use crate::commands::lsp::IndexingMode;
use crate::config::config::ConfigFile;
use crate::error::error::Error;
use crate::lsp::module_helpers::to_real_path;
use crate::lsp::non_wasm::build_system::should_requery_build_system;
use crate::lsp::non_wasm::lsp::apply_change_events;
use crate::lsp::non_wasm::lsp::as_notification;
use crate::lsp::non_wasm::lsp::as_request;
use crate::lsp::non_wasm::lsp::as_request_response_pair;
use crate::lsp::non_wasm::lsp::new_notification;
use crate::lsp::non_wasm::lsp::new_response;
use crate::lsp::non_wasm::module_helpers::handle_from_module_path;
use crate::lsp::non_wasm::module_helpers::make_open_handle;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::lsp::non_wasm::queue::HeavyTaskQueue;
use crate::lsp::non_wasm::queue::LspEvent;
use crate::lsp::non_wasm::queue::LspQueue;
use crate::lsp::non_wasm::stdlib::is_python_stdlib_file;
use crate::lsp::non_wasm::stdlib::should_show_error_for_display_mode;
use crate::lsp::non_wasm::stdlib::should_show_stdlib_error;
use crate::lsp::non_wasm::transaction_manager::TransactionManager;
use crate::lsp::non_wasm::unsaved_file_tracker::UnsavedFileTracker;
use crate::lsp::non_wasm::will_rename_files::will_rename_files;
use crate::lsp::non_wasm::workspace::LspAnalysisConfig;
use crate::lsp::non_wasm::workspace::Workspace;
use crate::lsp::non_wasm::workspace::Workspaces;
use crate::lsp::wasm::hover::get_hover;
use crate::lsp::wasm::notebook::DidChangeNotebookDocument;
use crate::lsp::wasm::notebook::DidChangeNotebookDocumentParams;
use crate::lsp::wasm::notebook::DidCloseNotebookDocument;
use crate::lsp::wasm::notebook::DidOpenNotebookDocument;
use crate::lsp::wasm::notebook::DidSaveNotebookDocument;
use crate::lsp::wasm::provide_type::ProvideType;
use crate::lsp::wasm::provide_type::ProvideTypeResponse;
use crate::lsp::wasm::provide_type::provide_type;
use crate::state::load::LspFile;
use crate::state::lsp::DisplayTypeErrors;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::notebook::LspNotebook;
use crate::state::require::Require;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::semantic_tokens::disabled_ranges_for_module;
use crate::state::state::CancellableTransaction;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TypeErrorDisplayStatus {
    DisabledInIdeConfig,
    EnabledInIdeConfig,
    DisabledInConfigFile,
    EnabledInConfigFile,
    DisabledDueToMissingConfigFile,
}

impl TypeErrorDisplayStatus {
    fn is_enabled(self) -> bool {
        match self {
            TypeErrorDisplayStatus::DisabledInIdeConfig
            | TypeErrorDisplayStatus::DisabledInConfigFile
            | TypeErrorDisplayStatus::DisabledDueToMissingConfigFile => false,
            TypeErrorDisplayStatus::EnabledInIdeConfig
            | TypeErrorDisplayStatus::EnabledInConfigFile => true,
        }
    }
}

/// Interface exposed for TSP to interact with the LSP server
pub trait TspInterface: Send + Sync {
    /// Send a response back to the LSP client
    fn send_response(&self, response: Response);

    fn connection(&self) -> &Connection;

    fn lsp_queue(&self) -> &LspQueue;

    /// Get access to the recheck queue for async task processing
    fn run_recheck_queue(&self, telemetry: &impl Telemetry);

    fn stop_recheck_queue(&self);

    /// Process an LSP event and return the next step
    fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        telemetry: &mut TelemetryEvent,
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent>;

    fn telemetry_state(&self) -> TelemetryServerState;
}

struct ServerConnection(Connection);

impl ServerConnection {
    fn send(&self, msg: Message) {
        if self.0.sender.send(msg).is_err() {
            // On error, we know the channel is closed.
            // https://docs.rs/crossbeam/latest/crossbeam/channel/struct.Sender.html#method.send
            info!("Connection closed.");
        };
    }

    fn publish_diagnostics_for_uri(&self, uri: Url, diags: Vec<Diagnostic>, version: Option<i32>) {
        self.send(Message::Notification(
            new_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
                uri, diags, version,
            )),
        ));
    }

    fn publish_diagnostics(
        &self,
        diags: SmallMap<PathBuf, Vec<Diagnostic>>,
        notebook_cell_urls: SmallMap<PathBuf, Url>,
        version_info: HashMap<PathBuf, i32>,
    ) {
        for (path, diags) in diags {
            if let Some(url) = notebook_cell_urls.get(&path) {
                self.publish_diagnostics_for_uri(url.clone(), diags, None)
            } else {
                let path = path.absolutize();
                let version = version_info.get(&path).copied();
                match Url::from_file_path(&path) {
                    Ok(uri) => self.publish_diagnostics_for_uri(uri, diags, version),
                    Err(_) => eprint!("Unable to convert path to uri: {path:?}"),
                }
            }
        }
    }
}

pub struct Server {
    connection: ServerConnection,
    lsp_queue: LspQueue,
    recheck_queue: HeavyTaskQueue,
    find_reference_queue: HeavyTaskQueue,
    sourcedb_queue: HeavyTaskQueue,
    /// Any configs whose find cache should be invalidated.
    invalidated_source_dbs: Mutex<SmallSet<ArcId<Box<dyn SourceDatabase + 'static>>>>,
    initialize_params: InitializeParams,
    indexing_mode: IndexingMode,
    workspace_indexing_limit: usize,
    build_system_blocking: bool,
    state: State,
    /// This is a mapping from open notebook cells to the paths of the notebooks they belong to,
    /// which can be used to look up the notebook contents in `open_files`.
    ///
    /// Notebook cell URIs are entirely arbitrary, and any URI received from the language client
    /// should be mapped through here in case they correspond to a cell.
    open_notebook_cells: RwLock<HashMap<Url, PathBuf>>,
    open_files: RwLock<HashMap<PathBuf, Arc<LspFile>>>,
    /// Tracks URIs (including virtual/untitled ones) to synthetic on-disk paths so we can
    /// treat them like regular files throughout the server.
    unsaved_file_tracker: UnsavedFileTracker,
    /// A set of configs where we have already indexed all the files within the config.
    indexed_configs: Mutex<HashSet<ArcId<ConfigFile>>>,
    /// A set of workspaces where we have already performed best-effort indexing.
    /// The user might open vscode at the root of the filesystem, so workspace indexing is
    /// performed with best effort up to certain limit of user files. When the workspace changes,
    /// we rely on file watchers to catch up.
    indexed_workspaces: Mutex<HashSet<PathBuf>>,
    cancellation_handles: Mutex<HashMap<RequestId, CancellationHandle>>,
    workspaces: Arc<Workspaces>,
    outgoing_request_id: AtomicI32,
    outgoing_requests: Mutex<HashMap<RequestId, Request>>,
    filewatcher_registered: AtomicBool,
    version_info: Mutex<HashMap<PathBuf, i32>>,
    id: Uuid,
}

/// At the time when we are ready to handle a new LSP event, it will help if we know the list of
/// buffered requests and notifications ready to be processed, because we can potentially make smart
/// decisions (e.g. not process cancelled requests).
///
/// This function listens to the LSP events in the order they arrive, and dispatch them into event
/// channels with various priority:
/// - priority_events includes those that should be handled as soon as possible (e.g. know that a
///   request is cancelled)
/// - queued_events includes most of the other events.
pub fn dispatch_lsp_events(connection: &Connection, lsp_queue: &LspQueue) {
    for msg in &connection.receiver {
        match msg {
            Message::Request(x) => {
                // Catch panics from lsp_server's handle_shutdown which may unwrap on SendError
                // when the client disconnects abruptly
                let shutdown_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        connection.handle_shutdown(&x)
                    }));

                match shutdown_result {
                    Ok(Ok(is_shutdown)) => {
                        if is_shutdown {
                            // break to ensure we send exit event
                            break;
                        }
                    }
                    Ok(Err(e)) => {
                        error!("Error handling shutdown: {:?}", e);
                        // still exit in the case of error
                        break;
                    }
                    Err(_) => {
                        info!("Shutdown panicked (likely client disconnected abruptly), exiting");
                        break;
                    }
                }
                if lsp_queue.send(LspEvent::LspRequest(x)).is_err() {
                    return;
                }
            }
            Message::Response(x) => {
                if lsp_queue.send(LspEvent::LspResponse(x)).is_err() {
                    return;
                }
            }
            Message::Notification(x) => {
                let send_result = if let Some(Ok(params)) =
                    as_notification::<DidOpenTextDocument>(&x)
                {
                    lsp_queue.send(LspEvent::DidOpenTextDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidChangeTextDocument>(&x) {
                    lsp_queue.send(LspEvent::DidChangeTextDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidCloseTextDocument>(&x) {
                    lsp_queue.send(LspEvent::DidCloseTextDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidSaveTextDocument>(&x) {
                    lsp_queue.send(LspEvent::DidSaveTextDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidOpenNotebookDocument>(&x) {
                    lsp_queue.send(LspEvent::DidOpenNotebookDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidChangeNotebookDocument>(&x) {
                    lsp_queue.send(LspEvent::DidChangeNotebookDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidCloseNotebookDocument>(&x) {
                    lsp_queue.send(LspEvent::DidCloseNotebookDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidSaveNotebookDocument>(&x) {
                    lsp_queue.send(LspEvent::DidSaveNotebookDocument(params))
                } else if let Some(Ok(params)) = as_notification::<DidChangeWatchedFiles>(&x) {
                    lsp_queue.send(LspEvent::DidChangeWatchedFiles(params))
                } else if let Some(Ok(params)) = as_notification::<DidChangeWorkspaceFolders>(&x) {
                    lsp_queue.send(LspEvent::DidChangeWorkspaceFolders(params))
                } else if let Some(Ok(params)) = as_notification::<DidChangeConfiguration>(&x) {
                    lsp_queue.send(LspEvent::DidChangeConfiguration(params))
                } else if let Some(Ok(params)) = as_notification::<Cancel>(&x) {
                    let id = match params.id {
                        NumberOrString::Number(i) => RequestId::from(i),
                        NumberOrString::String(s) => RequestId::from(s),
                    };
                    lsp_queue.send(LspEvent::CancelRequest(id))
                } else if as_notification::<Exit>(&x).is_some() {
                    // Send LspEvent::Exit and stop listening
                    break;
                } else {
                    info!("Unhandled notification: {x:?}");
                    Ok(())
                };
                if send_result.is_err() {
                    return;
                }
            }
        }
    }
    // when the connection closes, make sure we send an exit to the other thread
    let _ = lsp_queue.send(LspEvent::Exit);
}

pub fn capabilities(
    indexing_mode: IndexingMode,
    initialization_params: &InitializeParams,
) -> ServerCapabilities {
    let augments_syntax_tokens = initialization_params
        .capabilities
        .text_document
        .as_ref()
        .and_then(|c| c.semantic_tokens.as_ref())
        .and_then(|c| c.augments_syntax_tokens)
        .unwrap_or(false);
    ServerCapabilities {
        position_encoding: Some(PositionEncodingKind::UTF16),
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::INCREMENTAL,
        )),
        definition_provider: Some(OneOf::Left(true)),
        declaration_provider: Some(DeclarationCapability::Simple(true)),
        type_definition_provider: Some(TypeDefinitionProviderCapability::Simple(true)),
        implementation_provider: Some(ImplementationProviderCapability::Simple(true)),
        code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
            code_action_kinds: Some(vec![
                CodeActionKind::QUICKFIX,
                CodeActionKind::REFACTOR_EXTRACT,
            ]),
            ..Default::default()
        })),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_owned(), "'".to_owned(), "\"".to_owned()]),
            ..Default::default()
        }),
        document_highlight_provider: Some(OneOf::Left(true)),
        // Find references won't work properly if we don't know all the files.
        references_provider: match indexing_mode {
            IndexingMode::None => None,
            IndexingMode::LazyNonBlockingBackground | IndexingMode::LazyBlocking => {
                Some(OneOf::Left(true))
            }
        },
        rename_provider: match indexing_mode {
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
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        // Call hierarchy needs indexing to find cross-file callers/callees
        call_hierarchy_provider: match indexing_mode {
            IndexingMode::None => None,
            IndexingMode::LazyNonBlockingBackground | IndexingMode::LazyBlocking => {
                Some(CallHierarchyServerCapability::Simple(true))
            }
        },
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
            file_operations: Some(lsp_types::WorkspaceFileOperationsServerCapabilities {
                will_rename: Some(lsp_types::FileOperationRegistrationOptions {
                    filters: vec![lsp_types::FileOperationFilter {
                        pattern: lsp_types::FileOperationPattern {
                            glob: "**/*.{py,pyi}".to_owned(),

                            matches: Some(lsp_types::FileOperationPatternKind::File),
                            options: None,
                        },
                        scheme: Some("file".to_owned()),
                    }],
                }),
                ..Default::default()
            }),
        }),
        notebook_document_sync: Some(OneOf::Left(NotebookDocumentSyncOptions {
            notebook_selector: vec![NotebookSelector::ByCells {
                notebook: None,
                cells: vec![NotebookCellSelector {
                    language: "python".into(),
                }],
            }],
            save: None,
        })),
        ..Default::default()
    }
}

pub enum ProcessEvent {
    Continue,
    Exit,
}

const PYTHON_SECTION: &str = "python";

pub fn lsp_loop(
    connection: Connection,
    initialization_params: InitializeParams,
    indexing_mode: IndexingMode,
    workspace_indexing_limit: usize,
    build_system_blocking: bool,
    telemetry: &impl Telemetry,
) -> anyhow::Result<()> {
    info!("Reading messages");
    let lsp_queue = LspQueue::new();
    let server = Server::new(
        connection,
        lsp_queue,
        initialization_params,
        indexing_mode,
        workspace_indexing_limit,
        build_system_blocking,
    );
    std::thread::scope(|scope| {
        scope.spawn(|| {
            dispatch_lsp_events(&server.connection.0, &server.lsp_queue);
        });
        scope.spawn(|| {
            server.recheck_queue.run_until_stopped(&server, telemetry);
        });
        scope.spawn(|| {
            server
                .find_reference_queue
                .run_until_stopped(&server, telemetry);
        });
        scope.spawn(|| {
            server.sourcedb_queue.run_until_stopped(&server, telemetry);
        });
        let mut ide_transaction_manager = TransactionManager::default();
        let mut canceled_requests = HashSet::new();
        while let Ok((subsequent_mutation, event, enqueue_time)) = server.lsp_queue.recv() {
            let mut event_telemetry = TelemetryEvent::new_dequeued(
                TelemetryEventKind::LspEvent(event.describe()),
                enqueue_time,
                server.telemetry_state(),
            );
            let queue_duration = event_telemetry.queue;
            let event_description = event.describe();
            let result = server.process_event(
                &mut ide_transaction_manager,
                &mut canceled_requests,
                &mut event_telemetry,
                subsequent_mutation,
                event,
            );
            let process_duration =
                event_telemetry.finish_and_record(telemetry, result.as_ref().err());
            match result {
                Ok(ProcessEvent::Continue) => {
                    info!(
                        "Language server processed event `{}` in {:.2}s ({:.2}s waiting)",
                        event_description,
                        process_duration.as_secs_f32(),
                        queue_duration.as_secs_f32()
                    );
                }
                Ok(ProcessEvent::Exit) => break,
                Err(e) => {
                    // Log the error and continue processing the next event
                    error!("Error processing event `{}`: {:?}", event_description, e);
                }
            }
        }
        info!("waiting for connection to close");
        server.recheck_queue.stop();
        server.find_reference_queue.stop();
        server.sourcedb_queue.stop();
    });
    drop(server); // close connection
    Ok(())
}

impl Server {
    const FILEWATCHER_ID: &str = "FILEWATCHER";

    fn path_for_uri(&self, uri: &Url) -> Option<PathBuf> {
        if let Ok(path) = uri.to_file_path() {
            return Some(path);
        }
        if let Some(path) = self.unsaved_file_tracker.path_for_uri(uri) {
            return Some(path);
        }
        info!("Could not convert uri to filepath: {}", uri);
        None
    }

    fn extract_request_params_or_send_err_response<T>(
        &self,
        params: Result<T::Params, serde_json::Error>,
        id: &RequestId,
    ) -> Option<T::Params>
    where
        T: lsp_types::request::Request,
        T::Params: DeserializeOwned,
    {
        match params {
            Ok(params) => Some(params),
            Err(err) => {
                self.send_response(Response::new_err(
                    id.clone(),
                    ErrorCode::InvalidParams as i32,
                    err.to_string(),
                ));
                None
            }
        }
    }

    /// Process the event and return next step.
    fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        telemetry: &mut TelemetryEvent,
        // After this event there is another mutation
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent> {
        match event {
            LspEvent::Exit => {
                return Ok(ProcessEvent::Exit);
            }
            LspEvent::RecheckFinished => {
                // We did a commit and want to get back to a stable state.
                self.validate_in_memory_and_commit_if_possible(ide_transaction_manager, telemetry);
            }
            LspEvent::CancelRequest(id) => {
                info!("We should cancel request {id:?}");
                if let Some(cancellation_handle) = self.cancellation_handles.lock().remove(&id) {
                    cancellation_handle.cancel();
                }
                canceled_requests.insert(id);
            }
            LspEvent::InvalidateConfigFind => {
                let mut lock = self.invalidated_source_dbs.lock();
                let invalidated_source_dbs = std::mem::take(&mut *lock);
                drop(lock);
                if !invalidated_source_dbs.is_empty() {
                    // a sourcedb rebuild completed before this, so it's okay
                    // to re-setup the file watcher right now
                    self.setup_file_watcher_if_necessary();
                    let invalidated_configs = invalidated_source_dbs
                        .into_iter()
                        .flat_map(|db| self.workspaces.get_configs_for_source_db(db))
                        .collect();
                    self.invalidate_find_for_configs(invalidated_configs);
                }
            }
            LspEvent::DidOpenTextDocument(params) => {
                let lsp_types::DidOpenTextDocumentParams { text_document } = params;
                let lsp_types::TextDocumentItem {
                    uri, version, text, ..
                } = text_document;
                self.set_file_stats(uri.clone(), telemetry);
                let contents = Arc::new(LspFile::from_source(text));
                self.did_open(
                    ide_transaction_manager,
                    telemetry,
                    subsequent_mutation,
                    uri,
                    version,
                    contents,
                )?;
            }
            LspEvent::DidChangeTextDocument(params) => {
                self.set_file_stats(params.text_document.uri.clone(), telemetry);
                self.text_document_did_change(
                    ide_transaction_manager,
                    subsequent_mutation,
                    params,
                    telemetry,
                )?;
            }
            LspEvent::DidCloseTextDocument(params) => {
                self.set_file_stats(params.text_document.uri.clone(), telemetry);
                self.did_close(params.text_document.uri, telemetry);
            }
            LspEvent::DidSaveTextDocument(params) => {
                self.set_file_stats(params.text_document.uri.clone(), telemetry);
                self.did_save(params.text_document.uri);
            }
            LspEvent::DidOpenNotebookDocument(params) => {
                let url = params.notebook_document.uri.clone();
                self.set_file_stats(url.clone(), telemetry);
                let version = params.notebook_document.version;
                let notebook_document = params.notebook_document.clone();
                let cell_contents: HashMap<Url, String> = params
                    .cell_text_documents
                    .iter()
                    .map(|doc| (doc.uri.clone(), doc.text.clone()))
                    .collect();
                let ruff_notebook = params.notebook_document.to_ruff_notebook(&cell_contents)?;
                let lsp_notebook = LspNotebook::new(ruff_notebook, notebook_document);
                let notebook_path = url.to_file_path().map_err(|_| {
                    anyhow::anyhow!(
                        "Could not convert uri to filepath: {}, expected a notebook",
                        url
                    )
                })?;
                for cell_url in lsp_notebook.cell_urls() {
                    self.open_notebook_cells
                        .write()
                        .insert(cell_url.clone(), notebook_path.clone());
                }
                self.did_open(
                    ide_transaction_manager,
                    telemetry,
                    subsequent_mutation,
                    url,
                    version,
                    Arc::new(LspFile::Notebook(Arc::new(lsp_notebook))),
                )?;
            }
            LspEvent::DidChangeNotebookDocument(params) => {
                self.set_file_stats(params.notebook_document.uri.clone(), telemetry);
                self.notebook_document_did_change(
                    ide_transaction_manager,
                    subsequent_mutation,
                    params,
                    telemetry,
                )?;
            }
            LspEvent::DidCloseNotebookDocument(params) => {
                self.set_file_stats(params.notebook_document.uri.clone(), telemetry);
                self.did_close(params.notebook_document.uri, telemetry);
            }
            LspEvent::DidSaveNotebookDocument(params) => {
                self.set_file_stats(params.notebook_document.uri.clone(), telemetry);
                self.did_save(params.notebook_document.uri);
            }
            LspEvent::DidChangeWatchedFiles(params) => {
                self.did_change_watched_files(params, telemetry);
            }
            LspEvent::DidChangeWorkspaceFolders(params) => {
                self.workspace_folders_changed(params);
            }
            LspEvent::DidChangeConfiguration(params) => {
                self.did_change_configuration(params);
            }
            LspEvent::LspResponse(x) => {
                if let Some(request) = self.outgoing_requests.lock().remove(&x.id) {
                    if let Some((request, response)) =
                        as_request_response_pair::<WorkspaceConfiguration>(&request, &x)
                    {
                        self.workspace_configuration_response(&request, &response);
                    }
                } else {
                    info!("Response for unknown request: {x:?}");
                }
            }
            LspEvent::LspRequest(x) => {
                // These are messages where VS Code will use results from previous document versions,
                // we really don't want to implicitly cancel those.
                const ONLY_ONCE: &[&str] = &[
                    Completion::METHOD,
                    SignatureHelpRequest::METHOD,
                    GotoDefinition::METHOD,
                ];

                let in_cancelled_requests = canceled_requests.remove(&x.id);
                if in_cancelled_requests
                    || (subsequent_mutation && !ONLY_ONCE.contains(&x.method.as_str()))
                {
                    let message = format!(
                        "Request {} ({}) is canceled due to {}",
                        x.method,
                        x.id,
                        if in_cancelled_requests {
                            "explicit cancellation"
                        } else {
                            "subsequent mutation"
                        }
                    );
                    info!("{message}");
                    self.send_response(Response::new_err(
                        x.id,
                        ErrorCode::RequestCanceled as i32,
                        message,
                    ));
                    return Ok(ProcessEvent::Continue);
                }

                let mut transaction =
                    ide_transaction_manager.non_committable_transaction(&self.state);

                // As an over-approximation, validate open files. This request might be based on a transaction where we
                // skipped this step due to a subsequent mutation. We might also have a stale saved state, which we needed
                // to throw away because the underlying state has since changed.
                //
                // Validating in-memory files is relatively cheap, since we only actually recheck open files which have
                // changed file contents, so it's simpler to just always do it.
                self.validate_in_memory_for_transaction(&mut transaction, telemetry);

                info!("Handling non-canceled request {} ({})", x.method, &x.id);
                if let Some(params) = as_request::<GotoDefinition>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<GotoDefinition>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        let default_response = GotoDefinitionResponse::Array(Vec::new());
                        self.send_response(new_response(
                            x.id,
                            Ok(self
                                .goto_definition(&transaction, params)
                                .unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<GotoDeclaration>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<GotoDeclaration>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        let default_response = GotoDefinitionResponse::Array(Vec::new());
                        self.send_response(new_response(
                            x.id,
                            Ok(self
                                .goto_declaration(&transaction, params)
                                .unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<GotoTypeDefinition>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<GotoTypeDefinition>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        let default_response = GotoTypeDefinitionResponse::Array(Vec::new());
                        self.send_response(new_response(
                            x.id,
                            Ok(self
                                .goto_type_definition(&transaction, params)
                                .unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<GotoImplementation>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<GotoImplementation>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        self.async_go_to_implementations(x.id, &transaction, params);
                    }
                } else if let Some(params) = as_request::<CodeActionRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<CodeActionRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.code_action(&transaction, params).unwrap_or_default()),
                        ));
                    }
                } else if let Some(params) = as_request::<Completion>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<Completion>(params, &x.id)
                    {
                        self.set_file_stats(
                            params.text_document_position.text_document.uri.clone(),
                            telemetry,
                        );
                        self.send_response(new_response(
                            x.id,
                            self.completion(&transaction, params),
                        ));
                    }
                } else if let Some(params) = as_request::<DocumentHighlightRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<DocumentHighlightRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        self.send_response(new_response(
                            x.id,
                            Ok(self.document_highlight(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<References>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<References>(params, &x.id)
                        && !self
                            .open_notebook_cells
                            .read()
                            .contains_key(&params.text_document_position.text_document.uri)
                    {
                        self.set_file_stats(
                            params.text_document_position.text_document.uri.clone(),
                            telemetry,
                        );
                        self.references(x.id, &transaction, params);
                    } else {
                        // TODO(yangdanny) handle notebooks
                        let locations: Vec<Location> = Vec::new();
                        self.connection
                            .send(Message::Response(new_response(x.id, Ok(Some(locations)))));
                    }
                } else if let Some(params) = as_request::<PrepareRenameRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<PrepareRenameRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.prepare_rename(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<Rename>(&x) {
                    if let Some(params) =
                        self.extract_request_params_or_send_err_response::<Rename>(params, &x.id)
                        && !self
                            .open_notebook_cells
                            .read()
                            .contains_key(&params.text_document_position.text_document.uri)
                    {
                        self.set_file_stats(
                            params.text_document_position.text_document.uri.clone(),
                            telemetry,
                        );
                        // TODO(yangdanny) handle notebooks
                        // First check if rename is allowed via prepare_rename. If a rename is not allowed we
                        // send back an error. Otherwise we continue with the rename operation.
                        if let Some(_range) =
                            self.prepare_rename(&transaction, params.text_document_position.clone())
                        {
                            self.rename(x.id, &transaction, params);
                        } else {
                            self.send_response(Response {
                                id: x.id,
                                result: None,
                                error: Some(ResponseError {
                                    code: ErrorCode::InvalidRequest as i32,
                                    message: "Third-party symbols cannot be renamed".to_owned(),
                                    data: None,
                                }),
                            });
                        }
                    }
                } else if let Some(params) = as_request::<SignatureHelpRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<SignatureHelpRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        self.send_response(new_response(
                            x.id,
                            Ok(self.signature_help(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<HoverRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<HoverRequest>(params, &x.id)
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        let default_response = Hover {
                            contents: HoverContents::Array(Vec::new()),
                            range: None,
                        };
                        self.send_response(new_response(
                            x.id,
                            Ok(self.hover(&transaction, params).unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<InlayHintRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<InlayHintRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.inlay_hints(&transaction, params).unwrap_or_default()),
                        ));
                    }
                } else if let Some(params) = as_request::<SemanticTokensFullRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<SemanticTokensFullRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        let default_response = SemanticTokensResult::Tokens(SemanticTokens {
                            result_id: None,
                            data: Vec::new(),
                        });
                        self.send_response(new_response(
                            x.id,
                            Ok(self
                                .semantic_tokens_full(&transaction, params)
                                .unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<SemanticTokensRangeRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<SemanticTokensRangeRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        let default_response = SemanticTokensRangeResult::Tokens(SemanticTokens {
                            result_id: None,
                            data: Vec::new(),
                        });
                        self.send_response(new_response(
                            x.id,
                            Ok(self
                                .semantic_tokens_ranged(&transaction, params)
                                .unwrap_or(default_response)),
                        ));
                    }
                } else if let Some(params) = as_request::<DocumentSymbolRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<DocumentSymbolRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(DocumentSymbolResponse::Nested(
                                self.hierarchical_document_symbols(&transaction, params)
                                    .unwrap_or_default(),
                            )),
                        ));
                    }
                } else if let Some(params) = as_request::<WorkspaceSymbolRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<WorkspaceSymbolRequest>(
                            params, &x.id,
                        )
                    {
                        self.send_response(new_response(
                            x.id,
                            Ok(WorkspaceSymbolResponse::Flat(
                                self.workspace_symbols(&transaction, &params.query),
                            )),
                        ));
                    }
                } else if let Some(params) = as_request::<DocumentDiagnosticRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<DocumentDiagnosticRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.document_diagnostics(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<ProvideType>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<ProvideType>(params, &x.id)
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.provide_type(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<WillRenameFiles>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<WillRenameFiles>(
                            params, &x.id,
                        )
                    {
                        let supports_document_changes = self
                            .initialize_params
                            .capabilities
                            .workspace
                            .as_ref()
                            .and_then(|w| w.workspace_edit.as_ref())
                            .and_then(|we| we.document_changes)
                            .unwrap_or(false);
                        self.send_response(new_response(
                            x.id,
                            Ok(self.will_rename_files(
                                &transaction,
                                params,
                                supports_document_changes,
                            )),
                        ));
                    }
                } else if let Some(params) = as_request::<FoldingRangeRequest>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<FoldingRangeRequest>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.text_document.uri.clone(), telemetry);
                        let result = self
                            .folding_ranges(&transaction, params)
                            .unwrap_or_default();
                        self.send_response(new_response(x.id, Ok(result)));
                    }
                } else if let Some(params) = as_request::<CallHierarchyPrepare>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<CallHierarchyPrepare>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(
                            params
                                .text_document_position_params
                                .text_document
                                .uri
                                .clone(),
                            telemetry,
                        );
                        self.send_response(new_response(
                            x.id,
                            Ok(self.prepare_call_hierarchy(&transaction, params)),
                        ));
                    }
                } else if let Some(params) = as_request::<CallHierarchyIncomingCalls>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<CallHierarchyIncomingCalls>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.item.uri.clone(), telemetry);
                        self.async_call_hierarchy_incoming_calls(x.id, &transaction, params);
                    }
                } else if let Some(params) = as_request::<CallHierarchyOutgoingCalls>(&x) {
                    if let Some(params) = self
                        .extract_request_params_or_send_err_response::<CallHierarchyOutgoingCalls>(
                            params, &x.id,
                        )
                    {
                        self.set_file_stats(params.item.uri.clone(), telemetry);
                        self.async_call_hierarchy_outgoing_calls(x.id, &transaction, params);
                    }
                } else if &x.method == "pyrefly/textDocument/docstringRanges" {
                    let text_document: TextDocumentIdentifier = serde_json::from_value(x.params)?;
                    self.set_file_stats(text_document.uri.clone(), telemetry);
                    let ranges = self
                        .docstring_ranges(&transaction, &text_document)
                        .unwrap_or_default();
                    self.send_response(new_response(x.id, Ok(ranges)));
                } else if &x.method == "pyrefly/textDocument/typeErrorDisplayStatus" {
                    let text_document: TextDocumentIdentifier = serde_json::from_value(x.params)?;
                    self.set_file_stats(text_document.uri.clone(), telemetry);
                    if !self
                        .open_notebook_cells
                        .read()
                        .contains_key(&text_document.uri)
                        && let Some(path) = self.path_for_uri(&text_document.uri)
                    {
                        self.send_response(new_response(
                            x.id,
                            Ok(self.type_error_display_status(path.as_path())),
                        ));
                    } else {
                        // TODO(yangdanny): handle notebooks
                        self.send_response(new_response(
                            x.id,
                            Ok(TypeErrorDisplayStatus::DisabledDueToMissingConfigFile),
                        ));
                    }
                } else {
                    self.send_response(Response::new_err(
                        x.id.clone(),
                        ErrorCode::MethodNotFound as i32,
                        format!("Unknown request: {}", x.method),
                    ));
                    info!("Unhandled request: {x:?}");
                }
                ide_transaction_manager.save(transaction, telemetry);
            }
        }
        Ok(ProcessEvent::Continue)
    }

    pub fn new(
        connection: Connection,
        lsp_queue: LspQueue,
        initialize_params: InitializeParams,
        indexing_mode: IndexingMode,
        workspace_indexing_limit: usize,
        build_system_blocking: bool,
    ) -> Self {
        let folders = if let Some(capability) = &initialize_params.capabilities.workspace
            && let Some(true) = capability.workspace_folders
            && let Some(folders) = &initialize_params.workspace_folders
        {
            folders
                .iter()
                .filter_map(|x| x.uri.to_file_path().ok())
                .collect()
        } else {
            Vec::new()
        };

        let workspaces = Arc::new(Workspaces::new(Workspace::default(), &folders));

        let config_finder = Workspaces::config_finder(workspaces.dupe());
        let s = Self {
            connection: ServerConnection(connection),
            lsp_queue,
            recheck_queue: HeavyTaskQueue::new("recheck_queue"),
            find_reference_queue: HeavyTaskQueue::new("find_reference_queue"),
            sourcedb_queue: HeavyTaskQueue::new("sourcedb_queue"),
            invalidated_source_dbs: Mutex::new(SmallSet::new()),
            initialize_params,
            indexing_mode,
            workspace_indexing_limit,
            build_system_blocking,
            state: State::new(config_finder),
            open_notebook_cells: RwLock::new(HashMap::new()),
            open_files: RwLock::new(HashMap::new()),
            unsaved_file_tracker: UnsavedFileTracker::new(),
            indexed_configs: Mutex::new(HashSet::new()),
            indexed_workspaces: Mutex::new(HashSet::new()),
            cancellation_handles: Mutex::new(HashMap::new()),
            workspaces,
            outgoing_request_id: AtomicI32::new(1),
            outgoing_requests: Mutex::new(HashMap::new()),
            filewatcher_registered: AtomicBool::new(false),
            version_info: Mutex::new(HashMap::new()),
            id: Uuid::new_v4(),
        };

        if let Some(init_options) = &s.initialize_params.initialization_options {
            let mut modified = false;
            s.workspaces
                .apply_client_configuration(&mut modified, &None, init_options.clone());
            if let Some(workspace_folders) = &s.initialize_params.workspace_folders {
                for folder in workspace_folders {
                    s.workspaces.apply_client_configuration(
                        &mut modified,
                        &Some(folder.uri.clone()),
                        init_options.clone(),
                    );
                }
            }
        }

        s.setup_file_watcher_if_necessary();
        s.request_settings_for_all_workspaces();
        s
    }

    pub fn telemetry_state(&self) -> TelemetryServerState {
        TelemetryServerState {
            has_sourcedb: self.workspaces.sourcedb_available(),
            id: self.id,
        }
    }

    pub fn set_file_stats(&self, uri: Url, telemetry: &mut TelemetryEvent) {
        let config_root = if let Ok(path) = uri.to_file_path() {
            let config = self
                .state
                .config_finder()
                .python_file(ModuleName::unknown(), &ModulePath::filesystem(path));
            config
                .source
                .root()
                .and_then(|p| Url::from_file_path(p).ok())
        } else {
            None
        };

        telemetry.set_file_stats(TelemetryFileStats { uri, config_root });
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

    /// Run the transaction with the in-memory content of open files. Returns the handles of open files when the transaction is done.
    fn validate_in_memory_for_transaction(
        &self,
        transaction: &mut Transaction<'_>,
        telemetry: &mut TelemetryEvent,
    ) -> Vec<Handle> {
        let validate_start = Instant::now();
        let handles = self
            .open_files
            .read()
            .keys()
            .map(|x| make_open_handle(&self.state, x))
            .collect::<Vec<_>>();
        transaction.set_memory(
            self.open_files
                .read()
                .iter()
                .map(|x| (x.0.clone(), Some(Arc::new(x.1.to_file_contents()))))
                .collect::<Vec<_>>(),
        );
        transaction.run(&handles, Require::Everything);
        telemetry.set_validate_duration(validate_start.elapsed());
        handles
    }

    fn get_diag_if_shown(
        &self,
        e: &Error,
        open_files: &HashMap<PathBuf, Arc<LspFile>>,
        cell_uri: Option<&Url>, // If the file is a notebook, only show diagnostics for the matching cell
    ) -> Option<(PathBuf, Diagnostic)> {
        if let Some(path) = to_real_path(e.path()) {
            // When no file covers this, we'll get the default configured config which includes "everything"
            // and excludes `.<file>`s.
            let config = self
                .state
                .config_finder()
                .python_file(ModuleName::unknown(), e.path());

            let type_error_status = self.type_error_display_status(e.path().as_path());

            let should_show_stdlib_error =
                should_show_stdlib_error(&config, type_error_status, &path);

            if is_python_stdlib_file(&path) && !should_show_stdlib_error {
                return None;
            }

            // Check if we should filter based on error kind for ErrorMissingImports mode
            let display_type_errors_mode = self
                .workspaces
                .get_with(path.to_path_buf(), |(_, w)| w.display_type_errors);

            if !should_show_error_for_display_mode(e, display_type_errors_mode) {
                return None;
            }

            if let Some(lsp_file) = open_files.get(&path)
                && config.project_includes.covers(&path)
                && !config.project_excludes.covers(&path)
                && type_error_status.is_enabled()
            {
                return match &**lsp_file {
                    LspFile::Notebook(notebook) => {
                        let error_cell = e.get_notebook_cell()?;
                        let error_cell_uri = notebook.get_cell_url(error_cell)?;
                        if let Some(filter_cell) = cell_uri
                            && error_cell_uri != filter_cell
                        {
                            None
                        } else {
                            Some((PathBuf::from(error_cell_uri.to_string()), e.to_diagnostic()))
                        }
                    }
                    LspFile::Source(_) => Some((path.to_path_buf(), e.to_diagnostic())),
                };
            }
        }
        None
    }

    fn provide_type(
        &self,
        transaction: &Transaction<'_>,
        params: crate::lsp::wasm::provide_type::ProvideTypeParams,
    ) -> Option<ProvideTypeResponse> {
        let uri = &params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self.make_handle_if_enabled(uri, None)?;
        provide_type(transaction, &handle, params.positions)
    }

    fn type_error_display_status(&self, path: &Path) -> TypeErrorDisplayStatus {
        let handle = make_open_handle(&self.state, path);
        let config = self
            .state
            .config_finder()
            .python_file(handle.module(), handle.path());
        match self
            .workspaces
            .get_with(path.to_path_buf(), |(_, w)| w.display_type_errors)
        {
            Some(DisplayTypeErrors::ForceOn) => TypeErrorDisplayStatus::EnabledInIdeConfig,
            Some(DisplayTypeErrors::ErrorMissingImports) => {
                TypeErrorDisplayStatus::EnabledInIdeConfig
            }
            Some(DisplayTypeErrors::ForceOff) => TypeErrorDisplayStatus::DisabledInIdeConfig,
            Some(DisplayTypeErrors::Default) | None => match &config.source {
                // In this case, we don't have a config file.
                ConfigSource::Synthetic => TypeErrorDisplayStatus::DisabledDueToMissingConfigFile,
                // In this case, we have a config file like mypy.ini, but we don't parse it.
                // We only use it as a sensible project root, and create a default config anyways.
                // Therefore, we should treat it as if we don't have any config.
                ConfigSource::Marker(_) => TypeErrorDisplayStatus::DisabledDueToMissingConfigFile,
                // We actually have a pyrefly.toml, so we can decide based on the config.
                ConfigSource::File(_) => {
                    if config.disable_type_errors_in_ide(path) {
                        TypeErrorDisplayStatus::DisabledInConfigFile
                    } else {
                        TypeErrorDisplayStatus::EnabledInConfigFile
                    }
                }
            },
        }
    }

    fn validate_in_memory_and_commit_if_possible<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        telemetry: &mut TelemetryEvent,
    ) {
        let possibly_committable_transaction =
            ide_transaction_manager.get_possibly_committable_transaction(&self.state);
        self.validate_in_memory_for_possibly_committable_transaction(
            ide_transaction_manager,
            possibly_committable_transaction,
            telemetry,
        );
    }

    fn validate_in_memory_without_committing<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        telemetry: &mut TelemetryEvent,
    ) {
        let noncommittable_transaction =
            ide_transaction_manager.non_committable_transaction(&self.state);
        self.validate_in_memory_for_possibly_committable_transaction(
            ide_transaction_manager,
            Err(noncommittable_transaction),
            telemetry,
        );
    }

    fn supports_completion_item_details(&self) -> bool {
        self.initialize_params
            .capabilities
            .text_document
            .as_ref()
            .and_then(|t| t.completion.as_ref())
            .and_then(|c| c.completion_item.as_ref())
            .and_then(|ci| ci.label_details_support)
            .unwrap_or(false)
    }

    /// Helper to append all additional diagnostics (unreachable, unused parameters/imports/variables)
    fn append_ide_specific_diagnostics(
        transaction: &Transaction<'_>,
        handle: &Handle,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        Self::append_unreachable_diagnostics(transaction, handle, diagnostics);
        Self::append_unused_parameter_diagnostics(transaction, handle, diagnostics);
        Self::append_unused_import_diagnostics(transaction, handle, diagnostics);
        Self::append_unused_variable_diagnostics(transaction, handle, diagnostics);
    }

    /// Validate open files and send errors to the LSP. In the case of an ongoing recheck
    /// (i.e., another transaction is already being committed or the state is locked for writing),
    /// we still update diagnostics using a non-committable transaction, which may have slightly stale
    /// data compared to the main state
    fn validate_in_memory_for_possibly_committable_transaction<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        mut possibly_committable_transaction: Result<CommittingTransaction<'a>, Transaction<'a>>,
        telemetry: &mut TelemetryEvent,
    ) {
        let transaction = match &mut possibly_committable_transaction {
            Ok(transaction) => transaction.as_mut(),
            Err(transaction) => transaction,
        };
        let handles = self.validate_in_memory_for_transaction(transaction, telemetry);

        let publish = |transaction: &Transaction| {
            let mut diags: SmallMap<PathBuf, Vec<Diagnostic>> = SmallMap::new();
            let open_files = self.open_files.read();
            let open_notebook_cells = self.open_notebook_cells.read();
            let mut notebook_cell_urls = SmallMap::new();
            for x in open_notebook_cells.keys() {
                diags.insert(PathBuf::from(x.to_string()), Vec::new());
                notebook_cell_urls.insert(PathBuf::from(x.to_string()), x.clone());
            }
            for (x, file) in open_files.iter() {
                if !file.is_notebook() {
                    diags.insert(x.as_path().to_owned(), Vec::new());
                }
            }
            for e in transaction.get_errors(&handles).collect_errors().shown {
                if let Some((path, diag)) = self.get_diag_if_shown(&e, &open_files, None) {
                    diags.entry(path.to_owned()).or_default().push(diag);
                }
            }
            for (path, diagnostics) in diags.iter_mut() {
                if notebook_cell_urls.contains_key(path) {
                    continue;
                }
                let handle = make_open_handle(&self.state, path);
                Self::append_ide_specific_diagnostics(transaction, &handle, diagnostics);
            }
            self.connection.publish_diagnostics(
                diags,
                notebook_cell_urls,
                self.version_info.lock().clone(),
            );
            if self
                .initialize_params
                .capabilities
                .workspace
                .as_ref()
                .and_then(|w| w.semantic_tokens.as_ref())
                .and_then(|st| st.refresh_support)
                .unwrap_or(false)
            {
                self.send_request::<SemanticTokensRefresh>(());
            }
        };

        match possibly_committable_transaction {
            Ok(transaction) => {
                self.state.commit_transaction(transaction, Some(telemetry));
                // In the case where we can commit transactions, `State` already has latest updates.
                // Therefore, we can compute errors from transactions freshly created from `State``.
                let transaction = self.state.transaction();
                publish(&transaction);
                info!("Validated open files and committed transaction.");
            }
            Err(transaction) => {
                // In the case where transaction cannot be committed because there is an ongoing
                // recheck, we still want to update the diagnostics. In this case, we compute them
                // from the transactions that won't be committed. It will still contain all the
                // up-to-date in-memory content, but can have stale main `State` content.
                // Note: if this changes, update this function's docstring.
                publish(&transaction);
                ide_transaction_manager.save(transaction, telemetry);
                info!("Validated open files and saved non-committable transaction.");
            }
        }
    }

    fn invalidate_find_for_configs(&self, invalidated_configs: SmallSet<ArcId<ConfigFile>>) {
        self.invalidate(|t| t.invalidate_find_for_configs(invalidated_configs));
    }

    fn populate_project_files_if_necessary(
        &self,
        config_to_populate_files: Option<ArcId<ConfigFile>>,
        telemetry: &mut TelemetryEvent,
    ) {
        if let Some(config) = config_to_populate_files {
            if config.skip_lsp_config_indexing {
                return;
            }
            match self.indexing_mode {
                IndexingMode::None => {}
                IndexingMode::LazyNonBlockingBackground => {
                    if self.indexed_configs.lock().insert(config.dupe()) {
                        self.recheck_queue.queue_task(
                            TelemetryEventKind::PopulateProjectFiles,
                            Box::new(move |server, telemetry| {
                                server.populate_all_project_files_in_config(config, telemetry);
                            }),
                        );
                    }
                }
                IndexingMode::LazyBlocking => {
                    if self.indexed_configs.lock().insert(config.dupe()) {
                        self.populate_all_project_files_in_config(config, telemetry);
                    }
                }
            }
        }
    }

    fn populate_workspace_files_if_necessary(&self, telemetry: &mut TelemetryEvent) {
        let mut indexed_workspaces = self.indexed_workspaces.lock();
        let roots_to_populate_files = self
            .workspaces
            .roots()
            .into_iter()
            .filter(|root| !indexed_workspaces.contains(root))
            .collect_vec();
        let workspace_indexing_limit = self.workspace_indexing_limit;
        if roots_to_populate_files.is_empty() || workspace_indexing_limit == 0 {
            return;
        }
        match self.indexing_mode {
            IndexingMode::None => {}
            IndexingMode::LazyNonBlockingBackground => {
                indexed_workspaces.extend(roots_to_populate_files.iter().cloned());
                drop(indexed_workspaces);
                self.recheck_queue.queue_task(
                    TelemetryEventKind::PopulateWorkspaceFiles,
                    Box::new(move |server, telemetry| {
                        server.populate_all_workspaces_files(roots_to_populate_files, telemetry);
                    }),
                );
            }
            IndexingMode::LazyBlocking => {
                indexed_workspaces.extend(roots_to_populate_files.iter().cloned());
                drop(indexed_workspaces);
                self.populate_all_workspaces_files(roots_to_populate_files, telemetry);
            }
        }
    }

    /// Perform an invalidation of elements on `State` and commit them.
    /// Runs asynchronously. Returns immediately and may wait a while for a committable transaction.
    fn invalidate(&self, f: impl FnOnce(&mut Transaction) + Send + Sync + 'static) {
        self.recheck_queue.queue_task(
            TelemetryEventKind::Invalidate,
            Box::new(move |server, telemetry| {
                let mut transaction = server
                    .state
                    .new_committable_transaction(Require::indexing(), None);

                let invalidate_start = Instant::now();
                f(transaction.as_mut());
                telemetry.set_invalidate_duration(invalidate_start.elapsed());

                server.validate_in_memory_for_transaction(transaction.as_mut(), telemetry);

                // Commit will be blocked until there are no ongoing reads.
                // If we have some long running read jobs that can be cancelled, we should cancel them
                // to unblock committing transactions.
                for (_, cancellation_handle) in server.cancellation_handles.lock().drain() {
                    cancellation_handle.cancel();
                }
                // we have to run, not just commit to process updates
                server.state.run_with_committing_transaction(
                    transaction,
                    &[],
                    Require::Everything,
                    Some(telemetry),
                );
                // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
                // the main event loop of the server. As a result, the server can do a revalidation of
                // all the in-memory files based on the fresh main State as soon as possible.
                info!("Invalidated state, prepare to recheck open files.");
                let _ = server.lsp_queue.send(LspEvent::RecheckFinished);
            }),
        );
    }

    /// Certain IDE features (e.g. find-references) require us to know the dependency graph of the
    /// entire project to work. This blocking function should be called when we know that a project
    /// file is opened and if we intend to provide features like find-references, and should be
    /// called when config changes (currently this is a TODO).
    fn populate_all_project_files_in_config(
        &self,
        config: ArcId<ConfigFile>,
        telemetry: &mut TelemetryEvent,
    ) {
        let unknown = ModuleName::unknown();

        info!("Populating all files in the config ({:?}).", config.source);
        let mut transaction = self
            .state
            .new_committable_transaction(Require::indexing(), None);

        let project_path_blobs = config.get_filtered_globs(None);
        let paths = project_path_blobs.files().unwrap_or_default();
        let mut handles = Vec::new();
        for path in paths {
            let module_path = ModulePath::filesystem(path.clone());
            let path_config = self
                .state
                .config_finder()
                .python_file(unknown, &module_path);
            if config != path_config {
                continue;
            }
            handles.push(handle_from_module_path(&self.state, module_path));
        }

        info!("Prepare to check {} files.", handles.len());
        transaction.as_mut().run(&handles, Require::indexing());
        self.state.commit_transaction(transaction, Some(telemetry));
        // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
        // the main event loop of the server. As a result, the server can do a revalidation of
        // all the in-memory files based on the fresh main State as soon as possible.
        info!("Populated all files in the project path, prepare to recheck open files.");
        let _ = self.lsp_queue.send(LspEvent::RecheckFinished);
    }

    fn populate_all_workspaces_files(
        &self,
        workspace_roots: Vec<PathBuf>,
        telemetry: &mut TelemetryEvent,
    ) {
        for workspace_root in workspace_roots {
            info!(
                "Populating up to {} files in the workspace ({workspace_root:?}).",
                self.workspace_indexing_limit
            );
            let mut transaction = self
                .state
                .new_committable_transaction(Require::indexing(), None);

            let includes =
                ConfigFile::default_project_includes().from_root(workspace_root.as_path());
            let globs = FilteredGlobs::new(includes, ConfigFile::required_project_excludes(), None);
            let paths = globs
                .files_with_limit(self.workspace_indexing_limit)
                .unwrap_or_default();
            let mut handles = Vec::new();
            for path in paths {
                handles.push(handle_from_module_path(
                    &self.state,
                    ModulePath::filesystem(path.clone()),
                ));
            }

            info!("Prepare to check {} files.", handles.len());
            transaction.as_mut().run(&handles, Require::indexing());
            self.state.commit_transaction(transaction, Some(telemetry));
            // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
            // the main event loop of the server. As a result, the server can do a revalidation of
            // all the in-memory files based on the fresh main State as soon as possible.
            info!("Populated all files in the workspace, prepare to recheck open files.");
            let _ = self.lsp_queue.send(LspEvent::RecheckFinished);
        }
    }

    /// Attempts to requery any open sourced_dbs for open files, and if there are changes,
    /// invalidate find and perform a recheck.
    fn queue_source_db_rebuild_and_recheck(&self, telemetry: &mut TelemetryEvent, force: bool) {
        let run = move |server: &Server, telemetry: &mut TelemetryEvent| {
            let mut configs_to_paths: SmallMap<ArcId<ConfigFile>, SmallSet<ModulePath>> =
                SmallMap::new();
            let config_finder = server.state.config_finder();
            let handles = server
                .open_files
                .read()
                .keys()
                .map(|x| make_open_handle(&server.state, x))
                .collect::<Vec<_>>();
            for handle in handles {
                let config = config_finder.python_file(handle.module(), handle.path());
                configs_to_paths
                    .entry(config)
                    .or_default()
                    .insert(handle.path().dupe());
            }
            let (new_invalidated_source_dbs, rebuild_stats) =
                ConfigFile::query_source_db(&configs_to_paths, force);
            telemetry.set_sourcedb_rebuild_stats(rebuild_stats);
            if !new_invalidated_source_dbs.is_empty() {
                let mut lock = server.invalidated_source_dbs.lock();
                for db in new_invalidated_source_dbs {
                    lock.insert(db);
                }
                let _ = server.lsp_queue.send(LspEvent::InvalidateConfigFind);
            }
        };

        if self.build_system_blocking {
            run(self, telemetry);
        } else {
            self.sourcedb_queue
                .queue_task(TelemetryEventKind::SourceDbRebuild, Box::new(run));
        }
    }

    fn did_save(&self, url: Url) {
        if let Some(path) = self.path_for_uri(&url) {
            self.invalidate(move |t| t.invalidate_disk(&[path]))
        }
    }

    fn did_open<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        telemetry: &mut TelemetryEvent,
        subsequent_mutation: bool,
        url: Url,
        version: i32,
        contents: Arc<LspFile>,
    ) -> anyhow::Result<()> {
        let path = url
            .to_file_path()
            .or_else(|_| {
                if url.scheme() == "untitled" {
                    Ok(self
                        .unsaved_file_tracker
                        .ensure_path_for_open(&url, "python"))
                } else {
                    Err(())
                }
            })
            .map_err(|_| {
                anyhow::anyhow!("Could not convert uri to filepath for didOpen: {}", url)
            })?;
        let config_to_populate_files = if self.indexing_mode != IndexingMode::None
            && let Some(directory) = path.as_path().parent()
        {
            self.state.config_finder().directory(directory)
        } else {
            None
        };
        self.version_info.lock().insert(path.clone(), version);
        self.open_files.write().insert(path.clone(), contents);
        self.queue_source_db_rebuild_and_recheck(telemetry, false);
        if !subsequent_mutation {
            // In order to improve perceived startup perf, when a file is opened, we run a
            // non-committing transaction that indexes the file with default require level Exports.
            // This is very fast but doesn't follow transitive dependencies, so completions are
            // incomplete. This makes most IDE features available immediately while
            // populate_{project,workspace}_files below runs a transaction at default require level
            // Indexing in the background, generating a more complete index which becomes available
            // a few seconds later.
            //
            // Note that this trick works only when a pyrefly config file is present. In the absence
            // of a config file, all features become available when background indexing completes.
            info!(
                "File {} opened, prepare to validate open files.",
                path.display()
            );
            self.validate_in_memory_without_committing(ide_transaction_manager, telemetry);
        }
        self.populate_project_files_if_necessary(config_to_populate_files, telemetry);
        self.populate_workspace_files_if_necessary(telemetry);
        // rewatch files in case we loaded or dropped any configs
        self.setup_file_watcher_if_necessary();
        Ok(())
    }

    fn text_document_did_change<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        subsequent_mutation: bool,
        params: DidChangeTextDocumentParams,
        telemetry: &mut TelemetryEvent,
    ) -> anyhow::Result<()> {
        let VersionedTextDocumentIdentifier { uri, version } = params.text_document;
        let Some(file_path) = self.path_for_uri(&uri) else {
            return Err(anyhow::anyhow!(
                "Received textDocument/didChange for unknown uri: {uri}"
            ));
        };

        let mut version_info = self.version_info.lock();
        let old_version = version_info.get(&file_path).unwrap_or(&0);
        if version < *old_version {
            return Err(anyhow::anyhow!(
                "new_version < old_version in `textDocument/didChange` notification: new_version={version:?} old_version={old_version:?} text_document.uri={uri:?}"
            ));
        }
        version_info.insert(file_path.clone(), version);
        // drop this to avoid deadlock
        drop(version_info);
        let mut lock = self.open_files.write();
        let Some(original) = lock.get_mut(&file_path) else {
            return Err(anyhow::anyhow!(
                "File not found in open_files: {}",
                file_path.display()
            ));
        };
        *original = Arc::new(LspFile::from_source(apply_change_events(
            original.get_string(),
            params.content_changes,
        )));
        drop(lock);
        if !subsequent_mutation {
            info!(
                "File {} changed, prepare to validate open files.",
                file_path.display()
            );
            self.validate_in_memory_and_commit_if_possible(ide_transaction_manager, telemetry);
        }
        Ok(())
    }

    fn notebook_document_did_change<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        subsequent_mutation: bool,
        params: DidChangeNotebookDocumentParams,
        telemetry: &mut TelemetryEvent,
    ) -> anyhow::Result<()> {
        let uri = params.notebook_document.uri.clone();
        let version = params.notebook_document.version;
        let Some(file_path) = self.path_for_uri(&uri) else {
            return Err(anyhow::anyhow!(
                "Received notebookDocument/didChange for unknown uri: {uri}"
            ));
        };

        let mut version_info = self.version_info.lock();
        let old_version = version_info.get(&file_path).unwrap_or(&0);
        if version < *old_version {
            return Err(anyhow::anyhow!(
                "new_version < old_version in `notebookDocument/didChange` notification: new_version={version:?} old_version={old_version:?} notebook_document.uri={uri:?}"
            ));
        }

        let mut lock = self.open_files.write();
        let Some(original) = lock.get_mut(&file_path) else {
            return Err(anyhow::anyhow!(
                "File not found in open_files: {}",
                file_path.display()
            ));
        };

        let original_notebook = match original.as_ref() {
            LspFile::Notebook(notebook) => notebook.clone(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Expected notebook file for {}, but got text file",
                    uri
                ));
            }
        };

        let mut notebook_document = original_notebook.notebook_document().clone();
        let mut cell_content_map: HashMap<Url, String> = HashMap::new();
        // Changed metadata
        if let Some(metadata) = &params.change.metadata {
            notebook_document.metadata = Some(metadata.clone());
        }
        version_info.insert(file_path.clone(), version);
        drop(version_info);
        notebook_document.version = version;
        // Changes to cells
        if let Some(change) = &params.change.cells {
            // Track existing cell contents
            for cell in &notebook_document.cells {
                let cell_contents = original_notebook
                    .get_cell_contents(&cell.document)
                    .unwrap_or_default();
                cell_content_map.insert(cell.document.clone(), cell_contents);
            }
            // Structural changes
            if let Some(structure) = &change.structure {
                let start = structure.array.start as usize;
                let delete_count = structure.array.delete_count as usize;
                // Delete cells
                // Do not remove the cells from `open_notebook_cells`, since
                // incoming requests could still reference them.
                if delete_count > 0 {
                    notebook_document.cells.drain(start..start + delete_count);
                }
                // Insert new cells
                if let Some(new_cells) = &structure.array.cells {
                    for (i, cell) in new_cells.iter().enumerate() {
                        notebook_document.cells.insert(start + i, cell.clone());
                    }
                }
                // Set contents for new cells
                if let Some(opened_cells) = &structure.did_open {
                    for opened_cell in opened_cells {
                        cell_content_map.insert(opened_cell.uri.clone(), opened_cell.text.clone());
                        self.open_notebook_cells
                            .write()
                            .insert(opened_cell.uri.clone(), file_path.clone());
                    }
                }
            }
            // Cell metadata changes
            if let Some(cell_data) = &change.data {
                for updated_cell in cell_data {
                    if let Some(cell) = notebook_document
                        .cells
                        .iter_mut()
                        .find(|c| c.document == updated_cell.document)
                    {
                        cell.kind = updated_cell.kind;
                        cell.metadata = updated_cell.metadata.clone();
                        cell.execution_summary = updated_cell.execution_summary.clone();
                    }
                }
            }
            // Cell content changes
            if let Some(text_content_changes) = &change.text_content {
                for text_change in text_content_changes {
                    let cell_uri = text_change.document.uri.clone();
                    let original_text = cell_content_map
                        .get(&cell_uri)
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    let content_changes: Vec<TextDocumentContentChangeEvent> = text_change
                        .changes
                        .iter()
                        .filter_map(|v| serde_json::from_value(v.clone()).ok())
                        .collect();
                    let new_text = apply_change_events(original_text, content_changes);
                    cell_content_map.insert(cell_uri, new_text);
                }
            }
        }
        // Convert new notebook contents into a Ruff Notebook
        let ruff_notebook = notebook_document
            .clone()
            .to_ruff_notebook(&cell_content_map)?;

        let new_notebook = Arc::new(LspNotebook::new(ruff_notebook, notebook_document));
        *original = Arc::new(LspFile::Notebook(new_notebook));
        drop(lock);

        if !subsequent_mutation {
            info!(
                "Notebook {} changed, prepare to validate open files.",
                file_path.display()
            );
            self.validate_in_memory_and_commit_if_possible(ide_transaction_manager, telemetry);
        }
        Ok(())
    }

    /// Determines whether file watchers should be re-registered based on event types.
    /// Returns true if config files changed or files were created/removed/unknown.
    fn should_rewatch(events: &CategorizedEvents) -> bool {
        let config_changed = events.iter().any(|x| {
            x.file_name()
                .and_then(|x| x.to_str())
                .is_some_and(|x| ConfigFile::CONFIG_FILE_NAMES.contains(&x))
        });

        // Re-register watchers if files were created/removed (pip install, new files, etc.)
        // or if unknown events occurred. This ensures we discover new files while avoiding
        // unnecessary re-registration on simple file modifications.
        let files_added_or_removed =
            !events.created.is_empty() || !events.removed.is_empty() || !events.unknown.is_empty();

        config_changed || files_added_or_removed
    }

    fn did_change_watched_files(
        &self,
        params: DidChangeWatchedFilesParams,
        telemetry: &mut TelemetryEvent,
    ) {
        let events = CategorizedEvents::new_lsp(params.changes);
        if events.is_empty() {
            return;
        }
        let should_requery_build_system = should_requery_build_system(&events);

        // Rewatch files if necessary (config changed, files added/removed, etc.)
        if Self::should_rewatch(&events) {
            info!("[Pyrefly] Re-registering file watchers");
            self.setup_file_watcher_if_necessary();
        }

        self.invalidate(move |t| t.invalidate_events(&events));

        // If a non-Python, non-config file was changed, then try rebuilding build systems.
        // If no build system file was changed, then we should just not do anything. If
        // a build system file was changed, then the change should take effect soon.
        if should_requery_build_system {
            self.queue_source_db_rebuild_and_recheck(telemetry, true);
        }
    }

    fn did_close(&self, url: Url, telemetry: &mut TelemetryEvent) {
        let Some(path) = self.path_for_uri(&url) else {
            return;
        };
        let version = self
            .version_info
            .lock()
            .remove(&path)
            .map(|version| version + 1);
        if let Some(LspFile::Notebook(notebook)) = self.open_files.write().remove(&path).as_deref()
        {
            for cell in notebook.cell_urls() {
                self.connection
                    .publish_diagnostics_for_uri(cell.clone(), Vec::new(), version);
                self.open_notebook_cells.write().remove(cell);
            }
        } else {
            self.connection
                .publish_diagnostics_for_uri(url.clone(), Vec::new(), version);
        }
        self.unsaved_file_tracker.forget_uri_path(&url);
        self.queue_source_db_rebuild_and_recheck(telemetry, false);
        self.recheck_queue.queue_task(
            TelemetryEventKind::InvalidateOnClose,
            Box::new(move |server, telemetry| {
                // Clear out the memory associated with this file.
                // Not a race condition because we immediately call validate_in_memory to put back the open files as they are now.
                // Having the extra file hanging around doesn't harm anything, but does use extra memory.
                let mut transaction = server
                    .state
                    .new_committable_transaction(Require::indexing(), None);
                transaction.as_mut().set_memory(vec![(path, None)]);
                let _ = server.validate_in_memory_for_transaction(transaction.as_mut(), telemetry);
                server
                    .state
                    .commit_transaction(transaction, Some(telemetry));
            }),
        );
    }

    fn workspace_folders_changed(&self, params: DidChangeWorkspaceFoldersParams) {
        self.workspaces.changed(params.event);
        self.setup_file_watcher_if_necessary();
        self.request_settings_for_all_workspaces();
    }

    fn did_change_configuration<'a>(&'a self, params: DidChangeConfigurationParams) {
        if let Some(workspace) = &self.initialize_params.capabilities.workspace
            && workspace.configuration == Some(true)
        {
            self.request_settings_for_all_workspaces();
            return;
        }

        let mut modified = false;
        if let Some(python) = params.settings.get(PYTHON_SECTION) {
            self.workspaces
                .apply_client_configuration(&mut modified, &None, python.clone());
        }

        if modified {
            self.invalidate_config_and_validate_in_memory();
        }
    }

    fn workspace_configuration_response<'a>(
        &'a self,
        request: &ConfigurationParams,
        response: &[Value],
    ) {
        let mut modified = false;
        for (i, id) in request.items.iter().enumerate() {
            if let Some(value) = response.get(i) {
                self.workspaces.apply_client_configuration(
                    &mut modified,
                    &id.scope_uri,
                    value.clone(),
                );
                info!(
                    "Client configuration applied to workspace: {:?}",
                    id.scope_uri
                );
            }
        }
        if modified {
            self.invalidate_config_and_validate_in_memory();
        }
    }

    /// Create a handle with analysis config that decides language service behavior.
    /// Return None if the workspace has language services disabled (and thus you shouldn't do anything).
    ///
    /// `method` should be the LSP request METHOD string from lsp_types::request::* types
    /// (e.g., GotoDefinition::METHOD, HoverRequest::METHOD, etc.)
    fn make_handle_with_lsp_analysis_config_if_enabled(
        &self,
        uri: &Url,
        method: Option<&str>,
    ) -> Option<(Handle, Option<LspAnalysisConfig>)> {
        let path = if let Some(notebook_path) = self.open_notebook_cells.read().get(uri) {
            notebook_path.clone()
        } else {
            self.path_for_uri(uri)?
        };
        self.workspaces.get_with(path.clone(), |(_, workspace)| {
            // Check if all language services are disabled
            if workspace.disable_language_services {
                info!("Skipping request - language services disabled");
                return None;
            }

            // Check if the specific service is disabled
            if let Some(disabled_services) = workspace.disabled_language_services
                && let Some(method) = method
                && disabled_services.is_disabled(method)
            {
                info!("Skipping request - {} service disabled", method);
                return None;
            }

            let module_path = if self.open_files.read().contains_key(&path) {
                ModulePath::memory(path)
            } else {
                ModulePath::filesystem(path)
            };
            Some((
                handle_from_module_path(&self.state, module_path),
                workspace.lsp_analysis_config,
            ))
        })
    }

    /// make handle if enabled
    /// if method (the lsp method str exactly) is provided, we will check workspace settings
    /// for whether to enable it
    fn make_handle_if_enabled(&self, uri: &Url, method: Option<&str>) -> Option<Handle> {
        self.make_handle_with_lsp_analysis_config_if_enabled(uri, method)
            .map(|(handle, _)| handle)
    }

    fn goto_definition(
        &self,
        transaction: &Transaction<'_>,
        params: GotoDefinitionParams,
    ) -> Option<GotoDefinitionResponse> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri, Some(GotoDefinition::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let range =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        let targets = transaction.goto_definition(&handle, range);
        let mut lsp_targets = targets
            .iter()
            .filter_map(|x| self.to_lsp_location(x))
            .collect::<Vec<_>>();
        if lsp_targets.is_empty() {
            None
        } else if lsp_targets.len() == 1 {
            Some(GotoDefinitionResponse::Scalar(lsp_targets.pop().unwrap()))
        } else {
            Some(GotoDefinitionResponse::Array(lsp_targets))
        }
    }

    fn goto_declaration(
        &self,
        transaction: &Transaction<'_>,
        params: GotoDefinitionParams,
    ) -> Option<GotoDefinitionResponse> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri, Some(GotoDeclaration::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let range =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        let targets = transaction.goto_declaration(&handle, range);
        let mut lsp_targets = targets
            .iter()
            .filter_map(|x| self.to_lsp_location(x))
            .collect::<Vec<_>>();
        if lsp_targets.is_empty() {
            None
        } else if lsp_targets.len() == 1 {
            Some(GotoDefinitionResponse::Scalar(lsp_targets.pop().unwrap()))
        } else {
            Some(GotoDefinitionResponse::Array(lsp_targets))
        }
    }

    fn goto_type_definition(
        &self,
        transaction: &Transaction<'_>,
        params: GotoTypeDefinitionParams,
    ) -> Option<GotoTypeDefinitionResponse> {
        let uri = &params.text_document_position_params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self.make_handle_if_enabled(uri, Some(GotoTypeDefinition::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let range =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        let targets = transaction.goto_type_definition(&handle, range);
        let mut lsp_targets = targets
            .iter()
            .filter_map(|x| self.to_lsp_location(x))
            .collect::<Vec<_>>();
        if lsp_targets.is_empty() {
            None
        } else if lsp_targets.len() == 1 {
            Some(GotoTypeDefinitionResponse::Scalar(
                lsp_targets.pop().unwrap(),
            ))
        } else {
            Some(GotoTypeDefinitionResponse::Array(lsp_targets))
        }
    }

    fn async_go_to_implementations<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        params: GotoImplementationParams,
    ) {
        let uri = &params.text_document_position_params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return self.send_response(new_response::<Option<GotoImplementationResponse>>(
                request_id,
                Ok(None),
            ));
        }
        let Some(handle) = self.make_handle_if_enabled(uri, Some(GotoImplementation::METHOD))
        else {
            return self.send_response(new_response::<Option<GotoImplementationResponse>>(
                request_id,
                Ok(None),
            ));
        };
        self.async_find_from_definition_helper(
            request_id,
            transaction,
            handle,
            uri,
            params.text_document_position_params.position,
            FindPreference {
                import_behavior: ImportBehavior::StopAtRenamedImports,
                ..Default::default()
            },
            move |transaction, handle, definition| {
                let FindDefinitionItemWithDocstring {
                    metadata: _,
                    definition_range,
                    module,
                    docstring_range: _,
                    ..
                } = definition;
                // find_global_implementations_from_definition returns Vec<TextRangeWithModule>
                // but we need to return Vec<(ModuleInfo, Vec<TextRange>)> to match the helper's
                // expected format. Group implementations by module while preserving order.
                let implementations = transaction.find_global_implementations_from_definition(
                    handle.sys_info(),
                    TextRangeWithModule::new(module, definition_range),
                )?;

                // Group consecutive implementations by module, preserving the sorted order
                let mut grouped: Vec<(ModuleInfo, Vec<TextRange>)> = Vec::new();
                for impl_with_module in implementations {
                    if let Some((last_module, ranges)) = grouped.last_mut()
                        && last_module.path() == impl_with_module.module.path()
                    {
                        ranges.push(impl_with_module.range);
                        continue;
                    }
                    grouped.push((impl_with_module.module, vec![impl_with_module.range]));
                }
                Ok(grouped)
            },
            move |results: Vec<(ModuleInfo, Vec<TextRange>)>| {
                let mut lsp_targets = Vec::new();
                for (info, ranges) in results {
                    if let Some(uri) = module_info_to_uri(&info) {
                        for range in ranges {
                            lsp_targets.push(Location {
                                uri: uri.clone(),
                                range: info.to_lsp_range(range),
                            });
                        }
                    }
                }
                if lsp_targets.is_empty() {
                    None
                } else if lsp_targets.len() == 1 {
                    Some(GotoImplementationResponse::Scalar(
                        lsp_targets.pop().unwrap(),
                    ))
                } else {
                    Some(GotoImplementationResponse::Array(lsp_targets))
                }
            },
        );
    }

    fn completion(
        &self,
        transaction: &Transaction<'_>,
        params: CompletionParams,
    ) -> anyhow::Result<CompletionResponse> {
        let uri = &params.text_document_position.text_document.uri;
        let (handle, import_format) = match self
            .make_handle_with_lsp_analysis_config_if_enabled(uri, Some(Completion::METHOD))
        {
            None => {
                return Ok(CompletionResponse::List(CompletionList {
                    is_incomplete: false,
                    items: Vec::new(),
                }));
            }
            Some((x, config)) => (x, config.and_then(|c| c.import_format).unwrap_or_default()),
        };
        let (items, is_incomplete) = transaction
            .get_module_info(&handle)
            .map(|info| {
                transaction.completion_with_incomplete(
                    &handle,
                    self.from_lsp_position(uri, &info, params.text_document_position.position),
                    import_format,
                    self.supports_completion_item_details(),
                )
            })
            .unwrap_or_default();
        Ok(CompletionResponse::List(CompletionList {
            is_incomplete,
            items,
        }))
    }

    fn code_action(
        &self,
        transaction: &Transaction<'_>,
        params: CodeActionParams,
    ) -> Option<CodeActionResponse> {
        let uri = &params.text_document.uri;
        let (handle, lsp_config) = self.make_handle_with_lsp_analysis_config_if_enabled(
            uri,
            Some(CodeActionRequest::METHOD),
        )?;
        let import_format = lsp_config.and_then(|c| c.import_format).unwrap_or_default();
        let module_info = transaction.get_module_info(&handle)?;
        let range = self.from_lsp_range(uri, &module_info, params.range);
        let mut actions = Vec::new();
        if let Some(quickfixes) =
            transaction.local_quickfix_code_actions_sorted(&handle, range, import_format)
        {
            actions.extend(quickfixes.into_iter().filter_map(
                |(title, info, range, insert_text)| {
                    let lsp_location = self.to_lsp_location(&TextRangeWithModule {
                        module: info,
                        range,
                    })?;
                    Some(CodeActionOrCommand::CodeAction(CodeAction {
                        title,
                        kind: Some(CodeActionKind::QUICKFIX),
                        edit: Some(WorkspaceEdit {
                            changes: Some(HashMap::from([(
                                lsp_location.uri,
                                vec![TextEdit {
                                    range: lsp_location.range,
                                    new_text: insert_text,
                                }],
                            )])),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }))
                },
            ));
        }
        let mut push_refactor_actions = |refactors: Vec<LocalRefactorCodeAction>| {
            for action in refactors {
                let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();
                for (module, edit_range, new_text) in action.edits {
                    let Some(lsp_location) = self.to_lsp_location(&TextRangeWithModule {
                        module,
                        range: edit_range,
                    }) else {
                        continue;
                    };
                    changes.entry(lsp_location.uri).or_default().push(TextEdit {
                        range: lsp_location.range,
                        new_text,
                    });
                }
                if changes.is_empty() {
                    continue;
                }
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: action.title,
                    kind: Some(action.kind),
                    edit: Some(WorkspaceEdit {
                        changes: Some(changes),
                        ..Default::default()
                    }),
                    ..Default::default()
                }));
            }
        };
        if let Some(refactors) = transaction.extract_variable_code_actions(&handle, range) {
            push_refactor_actions(refactors);
        }
        if let Some(refactors) = transaction.extract_function_code_actions(&handle, range) {
            push_refactor_actions(refactors);
        }
        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    fn document_highlight(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentHighlightParams,
    ) -> Option<Vec<DocumentHighlight>> {
        let uri = &params.text_document_position_params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self.make_handle_if_enabled(uri, Some(DocumentHighlightRequest::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let position =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        Some(
            transaction
                .find_local_references(&handle, position)
                .into_map(|range| DocumentHighlight {
                    range: info.to_lsp_range(range),
                    kind: None,
                }),
        )
    }

    /// Compute references or implementations of a symbol at a given position. This is a non-blocking
    /// function that will send a response to the LSP client once the results are found and
    /// transformed by `transform_result`.
    ///
    /// The `find_fn` closure is called with the cancellable transaction, handle, and definition
    /// information, and should return a generic result type `T`.
    ///
    /// The `transform_result` closure transforms the result of type `T` into the final response
    /// type `V` that will be sent to the LSP client.
    fn async_find_from_definition_helper<'a, T: Send + 'static, V: serde::Serialize>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        handle: Handle,
        uri: &Url,
        position: Position,
        find_preference: FindPreference,
        find_fn: impl FnOnce(
            &mut CancellableTransaction,
            &Handle,
            FindDefinitionItemWithDocstring,
        ) -> Result<T, Cancelled>
        + Send
        + Sync
        + 'static,
        transform_result: impl FnOnce(T) -> V + Send + Sync + 'static,
    ) {
        let Some(info) = transaction.get_module_info(&handle) else {
            return self.send_response(new_response::<Option<V>>(request_id, Ok(None)));
        };
        let position = self.from_lsp_position(uri, &info, position);
        let Some(definition) = transaction
            .find_definition(&handle, position, find_preference)
            // TODO: handle more than 1 definition
            .into_iter()
            .next()
        else {
            return self.send_response(new_response::<Option<V>>(request_id, Ok(None)));
        };
        self.find_reference_queue.queue_task(
            TelemetryEventKind::FindFromDefinition,
            Box::new(move |server, telemetry| {
                let mut transaction = server.state.cancellable_transaction();
                server
                    .cancellation_handles
                    .lock()
                    .insert(request_id.clone(), transaction.get_cancellation_handle());
                server.validate_in_memory_for_transaction(transaction.as_mut(), telemetry);
                match find_fn(&mut transaction, &handle, definition) {
                    Ok(results) => {
                        server.cancellation_handles.lock().remove(&request_id);
                        server.connection.send(Message::Response(new_response(
                            request_id,
                            Ok(Some(transform_result(results))),
                        )));
                    }
                    Err(Cancelled) => {
                        let message = format!("Request {request_id} is canceled");
                        info!("{message}");
                        server.connection.send(Message::Response(Response::new_err(
                            request_id,
                            ErrorCode::RequestCanceled as i32,
                            message,
                        )));
                    }
                }
            }),
        );
    }

    /// Compute references of a symbol at a given position using the standard find_global_references_from_definition
    /// strategy. This is a convenience wrapper around async_find_from_definition_helper that handles
    /// the common case of finding references.
    fn async_find_references_helper<'a, V: serde::Serialize>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        handle: Handle,
        uri: &Url,
        position: Position,
        map_result: impl FnOnce(Vec<(Url, Vec<Range>)>) -> V + Send + Sync + 'static,
    ) {
        self.async_find_from_definition_helper(
            request_id,
            transaction,
            handle,
            uri,
            position,
            FindPreference {
                import_behavior: ImportBehavior::StopAtRenamedImports,
                ..Default::default()
            },
            |transaction, handle, definition| {
                let FindDefinitionItemWithDocstring {
                    metadata,
                    definition_range,
                    module,
                    docstring_range: _,
                    ..
                } = definition;
                transaction.find_global_references_from_definition(
                    handle.sys_info(),
                    metadata,
                    TextRangeWithModule::new(module, definition_range),
                )
            },
            move |results: Vec<(ModuleInfo, Vec<TextRange>)>| {
                // Transform ModuleInfo -> Url and TextRange -> Range
                let mut locations = Vec::new();
                for (info, ranges) in results {
                    if let Some(uri) = module_info_to_uri(&info) {
                        locations.push((uri, ranges.into_map(|range| info.to_lsp_range(range))));
                    };
                }
                map_result(locations)
            },
        )
    }

    fn references<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        params: ReferenceParams,
    ) {
        let uri = &params.text_document_position.text_document.uri;
        let Some(handle) = self.make_handle_if_enabled(uri, Some(References::METHOD)) else {
            return self.send_response(new_response::<Option<Vec<Location>>>(request_id, Ok(None)));
        };
        self.async_find_references_helper(
            request_id,
            transaction,
            handle,
            uri,
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
        transaction: &Transaction<'a>,
        params: RenameParams,
    ) {
        let uri = &params.text_document_position.text_document.uri;
        let Some(handle) = self.make_handle_if_enabled(uri, Some(Rename::METHOD)) else {
            return self.send_response(new_response::<Option<WorkspaceEdit>>(request_id, Ok(None)));
        };
        self.async_find_references_helper(
            request_id,
            transaction,
            handle,
            uri,
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
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self.make_handle_if_enabled(uri, Some(Rename::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let position = self.from_lsp_position(uri, &info, params.position);
        transaction
            .prepare_rename(&handle, position)
            .map(|range| PrepareRenameResponse::Range(info.to_lsp_range(range)))
    }

    fn signature_help(
        &self,
        transaction: &Transaction<'_>,
        params: SignatureHelpParams,
    ) -> Option<SignatureHelp> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri, Some(SignatureHelpRequest::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let position =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        transaction.get_signature_help_at(&handle, position)
    }

    fn hover(&self, transaction: &Transaction<'_>, params: HoverParams) -> Option<Hover> {
        let uri = &params.text_document_position_params.text_document.uri;
        let (handle, lsp_config) =
            self.make_handle_with_lsp_analysis_config_if_enabled(uri, Some(HoverRequest::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let position =
            self.from_lsp_position(uri, &info, params.text_document_position_params.position);
        let show_go_to_links = lsp_config
            .and_then(|c| c.show_hover_go_to_links)
            .unwrap_or(true);
        get_hover(transaction, &handle, position, show_go_to_links)
    }

    fn inlay_hints(
        &self,
        transaction: &Transaction<'_>,
        params: InlayHintParams,
    ) -> Option<Vec<InlayHint>> {
        let uri = &params.text_document.uri;
        let maybe_cell_idx = self.maybe_get_cell_index(uri);
        let range = &params.range;
        let (handle, lsp_analysis_config) = self
            .make_handle_with_lsp_analysis_config_if_enabled(uri, Some(InlayHintRequest::METHOD))?;
        let info = transaction.get_module_info(&handle)?;
        let t = transaction.inlay_hints(
            &handle,
            lsp_analysis_config
                .and_then(|c| c.inlay_hints)
                .unwrap_or_default(),
        )?;
        let res = t
            .into_iter()
            .filter_map(|(text_size, label_parts)| {
                // If the url is a notebook cell, filter out inlay hints for other cells
                if info.to_cell_for_lsp(text_size) != maybe_cell_idx {
                    return None;
                }
                let position = info.to_lsp_position(text_size);
                // The range is half-open, so the end position is exclusive according to the spec.
                if position >= range.start && position < range.end {
                    let label = InlayHintLabel::LabelParts(
                        label_parts
                            .iter()
                            .map(|(text, location_opt)| {
                                let location = location_opt
                                    .as_ref()
                                    .and_then(|loc| self.to_lsp_location(loc));

                                InlayHintLabelPart {
                                    value: text.clone(),
                                    tooltip: None,
                                    location,
                                    command: None,
                                }
                            })
                            .collect(),
                    );

                    Some(InlayHint {
                        position,
                        label,
                        kind: None,
                        text_edits: Some(vec![TextEdit {
                            range: Range::new(position, position),
                            new_text: label_parts.iter().map(|(text, _)| text.as_str()).collect(),
                        }]),
                        tooltip: None,
                        padding_left: None,
                        padding_right: None,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect();
        Some(res)
    }

    fn semantic_tokens_full(
        &self,
        transaction: &Transaction<'_>,
        params: SemanticTokensParams,
    ) -> Option<SemanticTokensResult> {
        let uri = &params.text_document.uri;
        let maybe_cell_idx = self.maybe_get_cell_index(uri);
        let handle = self.make_handle_if_enabled(uri, Some(SemanticTokensFullRequest::METHOD))?;
        Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: transaction
                .semantic_tokens(&handle, None, maybe_cell_idx)
                .unwrap_or_default(),
        }))
    }

    fn semantic_tokens_ranged(
        &self,
        transaction: &Transaction<'_>,
        params: SemanticTokensRangeParams,
    ) -> Option<SemanticTokensRangeResult> {
        let uri = &params.text_document.uri;
        let maybe_cell_idx = self.maybe_get_cell_index(uri);
        let handle = self.make_handle_if_enabled(uri, Some(SemanticTokensRangeRequest::METHOD))?;
        let module_info = transaction.get_module_info(&handle)?;
        let range = self.from_lsp_range(uri, &module_info, params.range);
        Some(SemanticTokensRangeResult::Tokens(SemanticTokens {
            result_id: None,
            data: transaction
                .semantic_tokens(&handle, Some(range), maybe_cell_idx)
                .unwrap_or_default(),
        }))
    }

    fn hierarchical_document_symbols(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentSymbolParams,
    ) -> Option<Vec<DocumentSymbol>> {
        let uri = &params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let path = self.path_for_uri(uri)?;
        if self
            .workspaces
            .get_with(path, |(_, workspace)| workspace.disable_language_services)
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
        let handle = self.make_handle_if_enabled(uri, Some(DocumentSymbolRequest::METHOD))?;
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

    fn append_unreachable_diagnostics(
        transaction: &Transaction<'_>,
        handle: &Handle,
        items: &mut Vec<Diagnostic>,
    ) {
        if let (Some(ast), Some(module_info)) = (
            transaction.get_ast(handle),
            transaction.get_module_info(handle),
        ) {
            let disabled_ranges = disabled_ranges_for_module(ast.as_ref(), handle.sys_info());
            let mut seen = HashSet::new();
            for range in disabled_ranges {
                if range.is_empty() || !seen.insert(range) {
                    continue;
                }
                let lsp_range = module_info.to_lsp_range(range);
                items.push(Diagnostic {
                    range: lsp_range,
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("Pyrefly".to_owned()),
                    message: "This code is unreachable for the current configuration".to_owned(),
                    code: Some(NumberOrString::String("unreachable-code".to_owned())),
                    code_description: None,
                    related_information: None,
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    data: None,
                });
            }
        }
    }

    fn append_unused_parameter_diagnostics(
        transaction: &Transaction<'_>,
        handle: &Handle,
        items: &mut Vec<Diagnostic>,
    ) {
        if let Some(bindings) = transaction.get_bindings(handle) {
            let module_info = bindings.module();
            for unused in bindings.unused_parameters() {
                let lsp_range = module_info.to_lsp_range(unused.range);
                items.push(Diagnostic {
                    range: lsp_range,
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("Pyrefly".to_owned()),
                    message: format!("Parameter `{}` is unused", unused.name.as_str()),
                    code: Some(NumberOrString::String("unused-parameter".to_owned())),
                    code_description: None,
                    related_information: None,
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    data: None,
                });
            }
        }
    }

    fn append_unused_import_diagnostics(
        transaction: &Transaction<'_>,
        handle: &Handle,
        items: &mut Vec<Diagnostic>,
    ) {
        if let Some(bindings) = transaction.get_bindings(handle) {
            let module_info = bindings.module();
            for unused in bindings.unused_imports() {
                let lsp_range = module_info.to_lsp_range(unused.range);
                items.push(Diagnostic {
                    range: lsp_range,
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("Pyrefly".to_owned()),
                    message: format!("Import `{}` is unused", unused.name.as_str()),
                    code: Some(NumberOrString::String("unused-import".to_owned())),
                    code_description: None,
                    related_information: None,
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    data: None,
                });
            }
        }
    }

    fn append_unused_variable_diagnostics(
        transaction: &Transaction<'_>,
        handle: &Handle,
        items: &mut Vec<Diagnostic>,
    ) {
        if let Some(bindings) = transaction.get_bindings(handle) {
            let module_info = bindings.module();
            for unused in bindings.unused_variables() {
                let lsp_range = module_info.to_lsp_range(unused.range);
                items.push(Diagnostic {
                    range: lsp_range,
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("Pyrefly".to_owned()),
                    message: format!("Variable `{}` is unused", unused.name.as_str()),
                    code: Some(NumberOrString::String("unused-variable".to_owned())),
                    code_description: None,
                    related_information: None,
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    data: None,
                });
            }
        }
    }

    fn docstring_ranges(
        &self,
        transaction: &Transaction<'_>,
        text_document: &TextDocumentIdentifier,
    ) -> Option<Vec<Range>> {
        if self
            .open_notebook_cells
            .read()
            .contains_key(&text_document.uri)
        {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self.make_handle_if_enabled(&text_document.uri, None)?;
        let module = transaction.get_module_info(&handle)?;
        let docstring_ranges = transaction.docstring_ranges(&handle)?;
        Some(
            docstring_ranges
                .into_iter()
                .map(|range| module.to_lsp_range(range))
                .collect(),
        )
    }

    fn folding_ranges(
        &self,
        transaction: &Transaction<'_>,
        params: FoldingRangeParams,
    ) -> Option<Vec<FoldingRange>> {
        if self
            .open_notebook_cells
            .read()
            .contains_key(&params.text_document.uri)
        {
            // TODO(yangdanny) handle notebooks
            return None;
        }
        let handle = self
            .make_handle_if_enabled(&params.text_document.uri, Some(FoldingRangeRequest::METHOD))?;
        let module = transaction.get_module_info(&handle)?;
        let ranges = transaction.folding_ranges(&handle)?;

        Some(
            ranges
                .into_iter()
                .filter_map(|(range, kind)| {
                    let lsp_range = module.to_lsp_range(range);
                    if lsp_range.start.line >= lsp_range.end.line {
                        return None;
                    }
                    let (end_line, end_character) = if lsp_range.end.character == 0
                        && lsp_range.end.line > lsp_range.start.line
                    {
                        (lsp_range.end.line - 1, None)
                    } else {
                        (lsp_range.end.line, Some(lsp_range.end.character))
                    };
                    if end_line <= lsp_range.start.line {
                        return None;
                    }
                    Some(FoldingRange {
                        start_line: lsp_range.start.line,
                        start_character: Some(lsp_range.start.character),
                        end_line,
                        end_character,
                        kind,
                        collapsed_text: None,
                    })
                })
                .collect(),
        )
    }

    fn document_diagnostics(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentDiagnosticParams,
    ) -> DocumentDiagnosticReport {
        let uri = &params.text_document.uri;
        let mut cell_uri = None;
        let path = if let Some(notebook_path) = self.open_notebook_cells.read().get(uri) {
            cell_uri = Some(uri);
            notebook_path.as_path().to_owned()
        } else {
            let Some(path) = self.path_for_uri(uri) else {
                return DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
                    full_document_diagnostic_report: FullDocumentDiagnosticReport {
                        items: Vec::new(),
                        result_id: None,
                    },
                    related_documents: None,
                });
            };
            path
        };
        let handle = make_open_handle(&self.state, &path);
        let mut items = Vec::new();
        let open_files = &self.open_files.read();
        for e in transaction.get_errors(once(&handle)).collect_errors().shown {
            if let Some((_, diag)) = self.get_diag_if_shown(&e, open_files, cell_uri) {
                items.push(diag);
            }
        }
        Self::append_ide_specific_diagnostics(transaction, &handle, &mut items);
        DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
            full_document_diagnostic_report: FullDocumentDiagnosticReport {
                items,
                result_id: None,
            },
            related_documents: None,
        })
    }

    /// Converts a [`WatchPattern`] into a [`GlobPattern`] that can be used and watched
    /// by VSCode, provided its `relative_pattern_support`.
    fn get_pattern_to_watch(
        pattern: WatchPattern<'_>,
        relative_pattern_support: bool,
    ) -> GlobPattern {
        match pattern {
            WatchPattern::File(root) => GlobPattern::String(root.to_string_lossy().into_owned()),
            WatchPattern::Root(root, pattern)
                if relative_pattern_support && let Ok(url) = Url::from_directory_path(root) =>
            {
                GlobPattern::Relative(RelativePattern {
                    base_uri: OneOf::Right(url),
                    pattern,
                })
            }
            WatchPattern::OwnedRoot(root, pattern)
                if relative_pattern_support && let Ok(url) = Url::from_directory_path(&root) =>
            {
                GlobPattern::Relative(RelativePattern {
                    base_uri: OneOf::Right(url),
                    pattern,
                })
            }
            WatchPattern::Root(root, pattern) => {
                GlobPattern::String(root.join(pattern).to_string_lossy().into_owned())
            }
            WatchPattern::OwnedRoot(root, pattern) => {
                GlobPattern::String(root.join(pattern).to_string_lossy().into_owned())
            }
        }
    }

    fn setup_file_watcher_if_necessary(&self) {
        let roots = self.workspaces.roots();
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
                for root in &roots {
                    PYTHON_EXTENSIONS.iter().for_each(|suffix| {
                        glob_patterns.push(Self::get_pattern_to_watch(
                            WatchPattern::root(root, format!("**/*.{suffix}")),
                            relative_pattern_support,
                        ));
                    });
                    ConfigFile::CONFIG_FILE_NAMES.iter().for_each(|config| {
                        glob_patterns.push(Self::get_pattern_to_watch(
                            WatchPattern::root(root, format!("**/{config}")),
                            relative_pattern_support,
                        ))
                    });
                }
                for config in self.workspaces.loaded_configs.clean_and_get_configs() {
                    config.get_paths_to_watch().into_iter().for_each(|pattern| {
                        glob_patterns.push(Self::get_pattern_to_watch(
                            pattern,
                            relative_pattern_support,
                        ));
                    });
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

    fn should_request_workspace_settings(&self) -> bool {
        self.initialize_params
            .capabilities
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.configuration)
            == Some(true)
    }

    fn request_settings_for_all_workspaces(&self) {
        if self.should_request_workspace_settings() {
            let roots = self.workspaces.roots();
            self.send_request::<WorkspaceConfiguration>(ConfigurationParams {
                items: roots
                    .iter()
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

    /// Asynchronously invalidate configuration and then validate in-memory files
    /// This ensures validate_in_memory() only runs after config invalidation completes
    fn invalidate_config_and_validate_in_memory(&self) {
        self.recheck_queue.queue_task(
            TelemetryEventKind::InvalidateConfig,
            Box::new(move |server, telemetry| {
                let mut transaction = server
                    .state
                    .new_committable_transaction(Require::indexing(), None);

                let invalidate_start = Instant::now();
                transaction.as_mut().invalidate_config();
                telemetry.set_invalidate_duration(invalidate_start.elapsed());

                server.validate_in_memory_for_transaction(transaction.as_mut(), telemetry);

                // Commit will be blocked until there are no ongoing reads.
                // If we have some long running read jobs that can be cancelled, we should cancel them
                // to unblock committing transactions.
                for (_, cancellation_handle) in server.cancellation_handles.lock().drain() {
                    cancellation_handle.cancel();
                }
                // we have to run, not just commit to process updates
                server.state.run_with_committing_transaction(
                    transaction,
                    &[],
                    Require::Everything,
                    Some(telemetry),
                );
                // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
                // the main event loop of the server. As a result, the server can do a revalidation of
                // all the in-memory files based on the fresh main State as soon as possible.
                // Only send RecheckFinished if there are actually open files to revalidate.
                if !server.open_files.read().is_empty() {
                    info!("Invalidated config, prepare to recheck open files.");
                    let _ = server.lsp_queue.send(LspEvent::RecheckFinished);
                } else {
                    info!("Invalidated config, but no open files to recheck.");
                }
            }),
        );
    }

    fn will_rename_files(
        &self,
        transaction: &Transaction<'_>,
        params: RenameFilesParams,
        supports_document_changes: bool,
    ) -> Option<WorkspaceEdit> {
        will_rename_files(
            &self.state,
            transaction,
            &self.open_files,
            params,
            supports_document_changes,
        )
    }

    pub fn to_lsp_location(&self, location: &TextRangeWithModule) -> Option<Location> {
        let TextRangeWithModule {
            module: definition_module_info,
            range,
        } = location;
        let mut uri = module_info_to_uri(definition_module_info)?;
        if let Some(cell_idx) = definition_module_info.to_cell_for_lsp(range.start()) {
            // We only have this information for open notebooks, without being provided the URI from the client
            // we don't know what URI refers to which cell.
            let path = to_real_path(definition_module_info.path())?;
            if let LspFile::Notebook(notebook) = &**self.open_files.read().get(&path)?
                && let Some(cell_url) = notebook.get_cell_url(cell_idx)
            {
                uri = cell_url.clone();
            }
        }
        Some(Location {
            uri,
            range: definition_module_info.to_lsp_range(*range),
        })
    }

    /// If the uri is an open notebook cell, return the index of the cell within the notebook
    /// otherwise, return None.
    fn maybe_get_cell_index(&self, cell_uri: &Url) -> Option<usize> {
        self.open_notebook_cells
            .read()
            .get(cell_uri)
            .and_then(|path| self.open_files.read().get(path).duped())
            .and_then(|file| match &*file {
                LspFile::Notebook(notebook) => notebook.get_cell_index(cell_uri),
                _ => None,
            })
    }

    pub fn from_lsp_position(
        &self,
        uri: &Url,
        module: &ModuleInfo,
        position: Position,
    ) -> TextSize {
        let notebook_cell = self.maybe_get_cell_index(uri);
        module.from_lsp_position(position, notebook_cell)
    }

    pub fn from_lsp_range(&self, uri: &Url, module: &ModuleInfo, position: Range) -> TextRange {
        let notebook_cell = self.maybe_get_cell_index(uri);
        module.from_lsp_range(position, notebook_cell)
    }

    /// Asynchronously finds incoming calls (callers) of a function.
    ///
    /// This queues work on the find_reference_queue to avoid blocking the LSP server
    /// while searching for callers across potentially many files.
    fn async_call_hierarchy_incoming_calls<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        params: lsp_types::CallHierarchyIncomingCallsParams,
    ) {
        let uri = params.item.uri.clone();

        let Some(handle) =
            self.make_handle_if_enabled(&uri, Some(CallHierarchyIncomingCalls::METHOD))
        else {
            return self.send_response(new_response::<
                Option<Vec<lsp_types::CallHierarchyIncomingCall>>,
            >(request_id, Ok(None)));
        };

        // The CallHierarchyItem we receive is already at the definition position
        // (thanks to prepare_call_hierarchy doing the go-to-definition step).
        self.async_find_from_definition_helper(
            request_id,
            transaction,
            handle,
            &uri,
            params.item.selection_range.start,
            FindPreference::default(),
            |transaction, handle, definition| {
                let target_def =
                    TextRangeWithModule::new(definition.module.dupe(), definition.definition_range);

                transaction.find_global_incoming_calls_from_function_definition(
                    handle.sys_info(),
                    definition.metadata.clone(),
                    &target_def,
                )
            },
            |callers| {
                let mut incoming_calls = Vec::new();
                for (caller_module, call_sites) in callers {
                    for (call_range, caller_name, caller_def_range) in call_sites {
                        let Some(caller_uri) = module_info_to_uri(&caller_module) else {
                            continue;
                        };

                        let from = lsp_types::CallHierarchyItem {
                            name: caller_name
                                .split('.')
                                .next_back()
                                .unwrap_or(&caller_name)
                                .to_owned(),
                            kind: lsp_types::SymbolKind::FUNCTION,
                            tags: None,
                            detail: Some(caller_name),
                            uri: caller_uri,
                            range: caller_module.to_lsp_range(caller_def_range),
                            selection_range: caller_module.to_lsp_range(caller_def_range),
                            data: None,
                        };

                        incoming_calls.push(lsp_types::CallHierarchyIncomingCall {
                            from,
                            from_ranges: vec![caller_module.to_lsp_range(call_range)],
                        });
                    }
                }
                incoming_calls
            },
        );
    }

    /// Asynchronously finds outgoing calls (callees) of a function.
    ///
    /// This queues work on the find_reference_queue to avoid blocking the LSP server
    /// while searching for callees across potentially many files.
    fn async_call_hierarchy_outgoing_calls<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        params: lsp_types::CallHierarchyOutgoingCallsParams,
    ) {
        let uri = params.item.uri.clone();

        let Some(handle) =
            self.make_handle_if_enabled(&uri, Some(CallHierarchyOutgoingCalls::METHOD))
        else {
            return self.send_response(new_response::<
                Option<Vec<lsp_types::CallHierarchyOutgoingCall>>,
            >(request_id, Ok(None)));
        };

        // Clone uri for use in the transform closure
        let uri_for_transform = uri.clone();

        // The CallHierarchyItem we receive is already at the definition position
        // (thanks to prepare_call_hierarchy doing the go-to-definition step).
        self.async_find_from_definition_helper(
            request_id,
            transaction,
            handle.dupe(),
            &uri,
            params.item.selection_range.start,
            FindPreference::default(),
            move |transaction, handle, definition| {
                // find_global_outgoing_calls_from_function_definition expects a position
                let position = definition.definition_range.start();

                let callees = transaction
                    .find_global_outgoing_calls_from_function_definition(handle, position)?;

                // Return both the callees and the module we need for LSP range conversion
                Ok((callees, definition.module))
            },
            move |(callees, source_module)| {
                let mut outgoing_calls = Vec::new();
                for (target_module, calls) in callees {
                    let target_uri = lsp_types::Url::from_file_path(target_module.path().as_path())
                        .unwrap_or_else(|()| uri_for_transform.clone());

                    for (call_range, target_def_range) in calls {
                        let target_name_short = target_module.code_at(target_def_range);
                        let target_name = format!("{}.{}", target_module.name(), target_name_short);

                        let to = lsp_types::CallHierarchyItem {
                            name: target_name_short.to_owned(),
                            kind: lsp_types::SymbolKind::FUNCTION,
                            tags: None,
                            detail: Some(target_name),
                            uri: target_uri.clone(),
                            range: target_module.to_lsp_range(target_def_range),
                            selection_range: target_module.to_lsp_range(target_def_range),
                            data: None,
                        };

                        outgoing_calls.push(lsp_types::CallHierarchyOutgoingCall {
                            to,
                            from_ranges: vec![source_module.to_lsp_range(call_range)],
                        });
                    }
                }
                outgoing_calls
            },
        );
    }

    /// Prepares the call hierarchy by validating that the symbol at the cursor is a function/method.
    /// This can be called from anywhere within the function definition, or also on a call site of the function.
    ///
    /// This is the entry point for LSP Call Hierarchy. It checks if the symbol at the given
    /// position is a callable (function or method) and returns a CallHierarchyItem if valid,
    /// or None if the symbol is not callable.
    fn prepare_call_hierarchy(
        &self,
        transaction: &Transaction<'_>,
        params: lsp_types::CallHierarchyPrepareParams,
    ) -> Option<Vec<lsp_types::CallHierarchyItem>> {
        use lsp_types::CallHierarchyItem;
        use lsp_types::SymbolKind;
        use ruff_text_size::Ranged;

        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri, None)?;
        let module_info = transaction.get_module_info(&handle)?;
        let position = self.from_lsp_position(
            uri,
            &module_info,
            params.text_document_position_params.position,
        );

        let definitions = transaction.find_definition(&handle, position, FindPreference::default());

        for def in definitions {
            // Get the URI for the definition's module
            let Some(def_uri) = module_info_to_uri(&def.module) else {
                continue;
            };

            // Get the handle for the definition's module (could be different from the current file)
            let Some(def_handle) = self.make_handle_if_enabled(&def_uri, None) else {
                continue;
            };

            let Some(ast) = transaction.get_ast(&def_handle) else {
                continue;
            };

            // Look for function at the definition position, not the original cursor position
            if let Some(func_def) =
                crate::state::state::CancellableTransaction::find_function_at_position_in_ast(
                    &ast,
                    def.definition_range.start(),
                )
            {
                let name = func_def.name.id.to_string();
                let detail = Some(format!("{}.{}", def.module.name(), name));
                let kind = SymbolKind::FUNCTION;

                let range = def.module.to_lsp_range(func_def.range());
                let selection_range = def.module.to_lsp_range(func_def.name.range());

                return Some(vec![CallHierarchyItem {
                    name,
                    kind,
                    tags: None,
                    detail,
                    uri: def_uri,
                    range,
                    selection_range,
                    data: None,
                }]);
            }
        }
        None
    }
}

impl TspInterface for Server {
    fn send_response(&self, response: Response) {
        self.send_response(response)
    }

    fn connection(&self) -> &Connection {
        &self.connection.0
    }

    fn lsp_queue(&self) -> &LspQueue {
        &self.lsp_queue
    }

    fn run_recheck_queue(&self, telemetry: &impl Telemetry) {
        self.recheck_queue.run_until_stopped(self, telemetry);
    }

    fn stop_recheck_queue(&self) {
        self.recheck_queue.stop();
    }

    fn process_event<'a>(
        &'a self,
        ide_transaction_manager: &mut TransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        telemetry: &mut TelemetryEvent,
        subsequent_mutation: bool,
        event: LspEvent,
    ) -> anyhow::Result<ProcessEvent> {
        self.process_event(
            ide_transaction_manager,
            canceled_requests,
            telemetry,
            subsequent_mutation,
            event,
        )
    }

    fn telemetry_state(&self) -> TelemetryServerState {
        self.telemetry_state()
    }
}
