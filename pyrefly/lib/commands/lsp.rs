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

use base64::Engine;
use base64::engine::general_purpose;
use clap::Parser;
use clap::ValueEnum;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use dupe::Dupe;
use itertools::__std_iter::once;
use itertools::Itertools;
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
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::NumberOrString;
use lsp_types::OneOf;
use lsp_types::PublishDiagnosticsParams;
use lsp_types::Range;
use lsp_types::ReferenceParams;
use lsp_types::Registration;
use lsp_types::RegistrationParams;
use lsp_types::RelatedFullDocumentDiagnosticReport;
use lsp_types::RelativePattern;
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
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::TextEdit;
use lsp_types::Unregistration;
use lsp_types::UnregistrationParams;
use lsp_types::Url;
use lsp_types::WatchKind;
use lsp_types::WorkspaceClientCapabilities;
use lsp_types::WorkspaceEdit;
use lsp_types::WorkspaceFoldersServerCapabilities;
use lsp_types::WorkspaceServerCapabilities;
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
use lsp_types::request::References;
use lsp_types::request::RegisterCapability;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::request::SemanticTokensRangeRequest;
use lsp_types::request::SignatureHelpRequest;
use lsp_types::request::UnregisterCapability;
use lsp_types::request::WorkspaceConfiguration;
use path_absolutize::Absolutize;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::arc_id::WeakArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::CancellationHandle;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::ThreadPool;
use ruff_text_size::TextRange;
use serde::de::DeserializeOwned;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::commands::config_finder::standard_config_finder;
use crate::commands::run::CommandExitStatus;
use crate::commands::util::module_from_path;
use crate::common::files::PYTHON_FILE_SUFFIXES_TO_WATCH;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::environment::environment::PythonEnvironment;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::kind::Severity;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionData;
use crate::types::lsp::position_to_text_size;
use crate::types::lsp::source_range_to_range;
use crate::types::lsp::text_size_to_position;

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

#[derive(Debug, Parser, Clone)]
pub struct Args {
    #[arg(long, value_enum, default_value_t, env = clap_env("INDEXING_MODE"))]
    pub(crate) indexing_mode: IndexingMode,
}

/// `IDETransactionManager` aims to always produce a transaction that contains the up-to-date
/// in-memory contents.
#[derive(Default)]
struct IDETransactionManager<'a> {
    /// Invariant:
    /// If it's None, then the main `State` already contains up-to-date checked content
    /// of all in-memory files.
    /// Otherwise, it will contains up-to-date checked content of all in-memory files.
    saved_state: Option<TransactionData<'a>>,
}

impl<'a> IDETransactionManager<'a> {
    #[expect(clippy::result_large_err)] // Both results are basically the same size
    /// Produce a possibly committable transaction in order to recheck in-memory files.
    fn get_possibly_committable_transaction(
        &mut self,
        state: &'a State,
    ) -> Result<CommittingTransaction<'a>, Transaction<'a>> {
        // If there is no ongoing recheck due to on-disk changes, we should prefer to commit
        // the in-memory changes into the main state.
        if let Some(transaction) = state.try_new_committable_transaction(Require::Indexing, None) {
            // If we can commit in-memory changes, then there is no point of holding the
            // non-commitable transaction with a possibly outdated view of the `ReadableState`
            // so we can destroy the saved state.
            self.saved_state = None;
            Ok(transaction)
        } else {
            // If there is an ongoing recheck, trying to get a committable transaction will block
            // until the recheck is finished. This is bad for perceived perf. Therefore, we will
            // temporarily use a non-commitable transaction to hold the information that's necessary
            // to power IDE services.
            Err(self.non_commitable_transaction(state))
        }
    }

    /// Produce a `Transaction` to power readonly IDE services.
    /// This transaction will never be able to be committed.
    /// After using it, the state should be saved by calling the `save` method.
    ///
    /// The `Transaction` will always contain the handles of all open files with the latest content.
    /// It might be created fresh from state, or reused from previously saved state.
    fn non_commitable_transaction(&mut self, state: &'a State) -> Transaction<'a> {
        if let Some(saved_state) = self.saved_state.take() {
            saved_state.into_transaction()
        } else {
            state.transaction()
        }
    }

    /// This function should be called once we finished using transaction for an LSP request.
    fn save(&mut self, transaction: Transaction<'a>) {
        self.saved_state = Some(transaction.into_data())
    }
}

enum ServerEvent {
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
    DidChangeConfiguration,
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

struct Server {
    connection: ServerConnection,
    /// A thread pool of size one for heavy read operations on the State
    async_state_read_threads: ThreadPool,
    priority_events_sender: Arc<Sender<ServerEvent>>,
    initialize_params: InitializeParams,
    indexing_mode: IndexingMode,
    state: Arc<State>,
    open_files: Arc<RwLock<HashMap<PathBuf, Arc<String>>>>,
    cancellation_handles: Arc<Mutex<HashMap<RequestId, CancellationHandle>>>,
    workspaces: Arc<Workspaces>,
    outgoing_request_id: Arc<AtomicI32>,
    outgoing_requests: Mutex<HashMap<RequestId, Request>>,
    filewatcher_registered: Arc<AtomicBool>,
}

/// Information about the Python environment p
#[derive(Debug, Clone)]
struct PythonInfo {
    /// The path to the interpreter used to query this `PythonInfo`'s [`PythonEnvironment`].
    interpreter: PathBuf,
    /// The [`PythonEnvironment`] values all [`ConfigFile`]s in a given workspace should
    /// use if no explicit [`ConfigFile::python_interpreter`] is provided, or any
    /// `PythonEnvironment` values in that `ConfigFile` are unfiled. If the `interpreter
    /// provided fails to execute or is invalid, this `PythonEnvironment` might instead
    /// be a system interpreter or [`PythonEnvironment::pyrefly_default()`].
    env: PythonEnvironment,
}

impl PythonInfo {
    fn new(interpreter: PathBuf) -> Self {
        let env = PythonEnvironment::get_interpreter_env(&interpreter);
        Self { interpreter, env }
    }
}

/// LSP workspace settings: this is all that is necessary to run an LSP at a given root.
#[derive(Debug, Clone)]
struct Workspace {
    #[expect(dead_code)]
    root: PathBuf,
    python_info: Option<PythonInfo>,
    disable_language_services: bool,
    disable_type_errors: bool,
}

impl Workspace {
    fn new(workspace_root: &Path, python_info: Option<PythonInfo>) -> Self {
        Self {
            root: workspace_root.to_path_buf(),
            python_info,
            disable_language_services: false,
            disable_type_errors: false,
        }
    }

    fn new_with_default_env(workspace_root: &Path) -> Self {
        Self::new(workspace_root, None)
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            root: PathBuf::from("/"),
            python_info: None,
            disable_language_services: Default::default(),
            disable_type_errors: false,
        }
    }
}

/// A cache of loaded configs from the LSP's [`standard_config_finder`]. These
/// values are [`WeakArcId`]s, so they will be dropped when no other references
/// point to them. We use this for determining the list of files we need to watch
/// when setting up the watcher.
struct WeakConfigCache(Mutex<HashSet<WeakArcId<ConfigFile>>>);

impl WeakConfigCache {
    fn new() -> Self {
        Self(Mutex::new(HashSet::new()))
    }

    fn insert(&self, config: WeakArcId<ConfigFile>) {
        self.0.lock().insert(config);
    }

    /// Purge any [`WeakArcId`]s that are [`WeakArcId::vacant`], and return
    /// the remaining configs, converted to [`ArcId`]s.
    fn clean_and_get_configs(&self) -> SmallSet<ArcId<ConfigFile>> {
        let mut configs = self.0.lock();
        let purged_config_count = configs.extract_if(|c| c.vacant()).count();
        if purged_config_count != 0 {
            eprintln!("Cleared {purged_config_count} dropped configs from config cache");
        }
        SmallSet::from_iter(configs.iter().filter_map(|c| c.upgrade()))
    }

    /// Given an [`ArcId<ConfigFile>`], get glob patterns that should be watched by a file watcher.
    /// We return a tuple of root (non-pattern part of the path) and a pattern.
    fn get_loaded_config_paths_to_watch(config: ArcId<ConfigFile>) -> Vec<(PathBuf, String)> {
        let mut result = Vec::new();
        let config_root = if let ConfigSource::File(config_path) = &config.source
            && let Some(root) = config_path.parent()
        {
            Some(root)
        } else {
            config.source.root()
        };
        if let Some(config_root) = config_root {
            ConfigFile::CONFIG_FILE_NAMES.iter().for_each(|config| {
                result.push((config_root.to_path_buf(), format!("**/{config}")));
            });
        }
        config
            .search_path()
            .chain(config.site_package_path())
            .cartesian_product(PYTHON_FILE_SUFFIXES_TO_WATCH)
            .for_each(|(s, suffix)| {
                result.push((s.to_owned(), format!("**/*.{suffix}")));
            });
        result
    }

    /// Get glob patterns to watch all Python files under [`ConfigFile::search_path`] and
    /// [`ConfigFile::site_package_path`], clearing out any removed [`ConfigFile`]s as we go.
    fn get_patterns_for_cached_configs(&self) -> Vec<(PathBuf, String)> {
        self.clean_and_get_configs()
            .into_iter()
            .flat_map(Self::get_loaded_config_paths_to_watch)
            .collect::<Vec<_>>()
    }
}

struct Workspaces {
    /// If a workspace is not found, this one is used. It contains every possible file on the system but is lowest priority.
    default: RwLock<Workspace>,
    workspaces: RwLock<SmallMap<PathBuf, Workspace>>,
    loaded_configs: Arc<WeakConfigCache>,
}

impl Workspaces {
    fn new(default: Workspace) -> Self {
        Self {
            default: RwLock::new(default),
            workspaces: RwLock::new(SmallMap::new()),
            loaded_configs: Arc::new(WeakConfigCache::new()),
        }
    }

    fn get_with<F, R>(&self, uri: PathBuf, f: F) -> R
    where
        F: FnOnce(&Workspace) -> R,
    {
        let workspaces = self.workspaces.read();
        let default_workspace = self.default.read();
        let workspace = workspaces
            .iter()
            .filter(|(key, _)| uri.starts_with(key))
            .max_by(|(key1, _), (key2, _)| key2.ancestors().count().cmp(&key1.ancestors().count()))
            .map(|(_, workspace)| workspace);
        f(workspace.unwrap_or(&default_workspace))
    }

    fn config_finder(
        workspaces: &Arc<Workspaces>,
        loaded_configs: Arc<WeakConfigCache>,
    ) -> ConfigFinder {
        let workspaces = workspaces.dupe();
        standard_config_finder(Arc::new(move |dir, mut config| {
            if let Some(dir) = dir
                && config.python_interpreter.is_none()
            {
                workspaces.get_with(dir.to_owned(), |w| {
                    let Some(PythonInfo { interpreter, env }) = w.python_info.clone() else {
                        return;
                    };
                    let site_package_path = config.python_environment.site_package_path.take();
                    config.python_interpreter = Some(interpreter);
                    config.python_environment = env;
                    if let Some(new) = site_package_path {
                        let mut workspace = config
                            .python_environment
                            .site_package_path
                            .take()
                            .unwrap_or_default();
                        workspace.extend(new);
                        config.python_environment.site_package_path = Some(workspace);
                    }
                })
            };
            config.configure();
            let config = ArcId::new(config);

            loaded_configs.insert(config.downgrade());

            (config, Vec::new())
        }))
    }
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
fn dispatch_lsp_events(
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
                } else if let Some(params) = as_notification::<Cancel>(&x) {
                    let id = match params.id {
                        NumberOrString::Number(i) => RequestId::from(i),
                        NumberOrString::String(s) => RequestId::from(s),
                    };
                    priority_events_sender.send(ServerEvent::CancelRequest(id))
                } else if as_notification::<DidChangeConfiguration>(&x).is_some() {
                    queued_events_sender.send(ServerEvent::DidChangeConfiguration)
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

fn initialize_connection(
    connection: &Connection,
    args: &Args,
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
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
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
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_owned(), ",".to_owned()]),
            ..Default::default()
        }),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
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

pub fn run_lsp(
    connection: Arc<Connection>,
    wait_on_connection: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
    args: Args,
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

impl Args {
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

/// Convert to a path we can show to the user. The contents may not match the disk, but it has
/// to be basically right.
fn to_real_path(path: &ModulePath) -> Option<&Path> {
    match path.details() {
        ModulePathDetails::FileSystem(path)
        | ModulePathDetails::Memory(path)
        | ModulePathDetails::Namespace(path) => Some(path),
        ModulePathDetails::BundledTypeshed(_) => None,
    }
}

fn module_info_to_uri(module_info: &ModuleInfo) -> Option<Url> {
    let path = to_real_path(module_info.path())?;
    let abs_path = path.absolutize();
    let abs_path = abs_path.as_deref().unwrap_or(path);
    Some(Url::from_file_path(abs_path).unwrap())
}

/// VSCode exposes a textDocumentContentProvider API where language clients can define schemes
/// for rendering read only content. We have defined a scheme `contentsAsUri` which encodes the
/// entire file contents into the uri of the URL. Since a definition response only contains a URL,
/// this is an easy way to send file contents to the language client to be displayed.
fn module_info_to_uri_with_document_content_provider(module_info: &ModuleInfo) -> Option<Url> {
    match module_info.path().details() {
        ModulePathDetails::FileSystem(path)
        | ModulePathDetails::Memory(path)
        | ModulePathDetails::Namespace(path) => {
            let abs_path = path.absolutize();
            let abs_path = abs_path.as_deref().unwrap_or(path);
            Some(Url::from_file_path(abs_path).unwrap())
        }
        ModulePathDetails::BundledTypeshed(path) => {
            Url::parse(&format!(
                // This is the URI displayed in the opened file - note the extra `/` to make it absolute,
                // otherwise VSCode will not display it
                "contentsAsUri:///{}?{}",
                path.to_string_lossy().into_owned(),
                general_purpose::STANDARD.encode(module_info.contents().as_str())
            ))
            .ok()
        }
    }
}

enum ProcessEvent {
    Continue,
    Exit,
}

impl Server {
    const FILEWATCHER_ID: &str = "FILEWATCHER";

    /// Process the event and return next step.
    fn process_event<'a>(
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
                eprintln!("We should cancel request {:?}", id);
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
            ServerEvent::DidChangeConfiguration => {
                self.change_workspace();
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

    fn new(
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
            cancellation_handles: Arc::new(Mutex::new(HashMap::new())),
            workspaces,
            outgoing_request_id: Arc::new(AtomicI32::new(1)),
            outgoing_requests: Mutex::new(HashMap::new()),
            filewatcher_registered: Arc::new(AtomicBool::new(false)),
        };
        s.configure(&folders, &Vec::new());

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
            .map(|x| (Self::make_open_handle(state, x), Require::Everything))
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
            if open_files.contains_key(path)
                && !config.project_excludes.covers(path)
                && !self
                    .workspaces
                    .get_with(path.to_path_buf(), |w| w.disable_type_errors)
            {
                return Some((
                    path.to_path_buf(),
                    Diagnostic {
                        range: source_range_to_range(e.source_range()),
                        severity: Some(match e.error_kind().severity() {
                            Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
                            Severity::Warn => lsp_types::DiagnosticSeverity::WARNING,
                            Severity::Info => lsp_types::DiagnosticSeverity::INFORMATION,
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
                IndexingMode::LazyBlocking => {
                    Self::populate_all_project_files_in_config(
                        config,
                        self.state.dupe(),
                        self.priority_events_sender.dupe(),
                    );
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
        // We asked for Sync full, so can just grab all the text from params
        let change = params.content_changes.into_iter().next().unwrap();
        let uri = params.text_document.uri.to_file_path().unwrap();
        self.open_files.write().insert(uri, Arc::new(change.text));
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

    /// Configure the server with a new set of workspace folders
    fn configure(
        &self,
        workspace_paths_added: &Vec<PathBuf>,
        workspace_paths_removed: &Vec<PathBuf>,
    ) {
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

    fn handle_from_module_path(state: &State, path: ModulePath) -> Handle {
        let unknown = ModuleName::unknown();
        let config = state.config_finder().python_file(unknown, &path);
        let module_name = to_real_path(&path)
            .and_then(|path| module_from_path(path, config.search_path()))
            .unwrap_or(unknown);
        Handle::new(module_name, path, config.get_sys_info())
    }

    fn make_open_handle(state: &State, path: &Path) -> Handle {
        let path = ModulePath::memory(path.to_owned());
        Self::handle_from_module_path(state, path)
    }

    /// Create a handle. Return None if the workspace has language services disabled (and thus you shouldn't do anything).
    fn make_handle_if_enabled(&self, uri: &Url) -> Option<Handle> {
        let path = uri.to_file_path().unwrap();
        self.workspaces.get_with(path.clone(), |workspace| {
            if workspace.disable_language_services {
                None
            } else {
                let module_path = if self.open_files.read().contains_key(&path) {
                    ModulePath::memory(path)
                } else {
                    ModulePath::filesystem(path)
                };
                Some(Self::handle_from_module_path(&self.state, module_path))
            }
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
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let TextRangeWithModuleInfo {
            module_info: definition_module_info,
            range,
        } = transaction.goto_definition(&handle, range)?;
        let uri = match &self.initialize_params.initialization_options {
            Some(serde_json::Value::Object(map))
                if (map.get("supportContentsAsUri") == Some(&serde_json::Value::Bool(true))) =>
            {
                module_info_to_uri_with_document_content_provider(&definition_module_info)?
            }
            Some(_) | None => module_info_to_uri(&definition_module_info)?,
        };
        Some(GotoDefinitionResponse::Scalar(Location {
            uri,
            range: source_range_to_range(&definition_module_info.source_range(range)),
        }))
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
                    position_to_text_size(&info, params.text_document_position.position),
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
        let range = TextRange::new(
            position_to_text_size(&module_info, params.range.start),
            position_to_text_size(&module_info, params.range.end),
        );
        let code_actions = transaction
            .local_quickfix_code_actions(&handle, range)?
            .into_map(|(title, range, insert_text)| {
                CodeActionOrCommand::CodeAction(CodeAction {
                    title,
                    kind: Some(CodeActionKind::QUICKFIX),
                    edit: Some(WorkspaceEdit {
                        changes: Some(HashMap::from([(
                            uri.clone(),
                            vec![TextEdit {
                                range: source_range_to_range(&range),
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
        let position = position_to_text_size(&info, params.text_document_position_params.position);
        Some(
            transaction
                .find_local_references(&handle, position)
                .into_map(|range| DocumentHighlight {
                    range: source_range_to_range(&info.source_range(range)),
                    kind: None,
                }),
        )
    }

    fn references<'a>(
        &'a self,
        request_id: RequestId,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        params: ReferenceParams,
    ) {
        let uri = &params.text_document_position.text_document.uri;
        let Some(handle) = self.make_handle_if_enabled(uri) else {
            return self.send_response(new_response::<Option<Vec<Location>>>(request_id, Ok(None)));
        };
        let transaction = ide_transaction_manager.non_commitable_transaction(&self.state);
        let Some(info) = transaction.get_module_info(&handle) else {
            ide_transaction_manager.save(transaction);
            return self.send_response(new_response::<Option<Vec<Location>>>(request_id, Ok(None)));
        };
        let position = position_to_text_size(&info, params.text_document_position.position);
        let Some((definition_kind, definition, _docstring)) =
            transaction.find_definition(&handle, position)
        else {
            ide_transaction_manager.save(transaction);
            return self.send_response(new_response::<Option<Vec<Location>>>(request_id, Ok(None)));
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
                definition_kind,
                definition,
            ) {
                Ok(global_references) => {
                    let mut locations = Vec::new();
                    for (info, ranges) in global_references {
                        if let Some(uri) = module_info_to_uri(&info) {
                            for range in ranges {
                                locations.push(Location {
                                    uri: uri.clone(),
                                    range: source_range_to_range(&info.source_range(range)),
                                });
                            }
                        };
                    }
                    connection.send(Message::Response(new_response(
                        request_id,
                        Ok(Some(locations)),
                    )))
                }
                Err(Cancelled) => {
                    let message = format!("Find reference request {} is canceled", request_id);
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

    fn signature_help(
        &self,
        transaction: &Transaction<'_>,
        params: SignatureHelpParams,
    ) -> Option<SignatureHelp> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let position = position_to_text_size(&info, params.text_document_position_params.position);
        transaction.get_signature_help_at(&handle, position)
    }

    fn hover(&self, transaction: &Transaction<'_>, params: HoverParams) -> Option<Hover> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let t = transaction.get_type_at(&handle, range)?;
        let mut kind_formatted: String = "".to_owned();
        let mut docstring_formatted: String = "".to_owned();
        if let Some((definition_metadata, text_range_with_module_info, docstring)) =
            transaction.find_definition(&handle, range)
        {
            if let Some(symbol_kind) = definition_metadata.symbol_kind() {
                kind_formatted = format!(
                    "{} {}: ",
                    &symbol_kind.display_for_hover(),
                    text_range_with_module_info
                        .module_info
                        .code_at(text_range_with_module_info.range)
                );
            }
            if let Some(docstring) = docstring {
                docstring_formatted = format!("\n---\n{}", docstring.as_string().trim());
            }
        }
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n{}{}\n```{}",
                    kind_formatted, t, docstring_formatted
                ),
            }),
            range: None,
        })
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
            let position = text_size_to_position(&info, x.0);
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
        let range = TextRange::new(
            position_to_text_size(&module_info, params.range.start),
            position_to_text_size(&module_info, params.range.end),
        );
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

    fn document_diagnostics(
        &self,
        transaction: &Transaction<'_>,
        params: DocumentDiagnosticParams,
    ) -> DocumentDiagnosticReport {
        let handle = Self::make_open_handle(
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

    fn change_workspace(&self) {
        self.request_settings_for_all_workspaces();
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
                    PYTHON_FILE_SUFFIXES_TO_WATCH.iter().for_each(|suffix| {
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
                        section: Some("python".to_owned()),
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
                match response.get(i) {
                    Some(serde_json::Value::Object(map)) => {
                        if let Some(serde_json::Value::String(python_path)) = map.get("pythonPath")
                        {
                            self.update_pythonpath(&mut modified, &id.scope_uri, python_path);
                        }
                        if let Some(serde_json::Value::Object(pyrefly_settings)) =
                            map.get("pyrefly")
                        {
                            if let Some(serde_json::Value::Bool(disable_language_services)) =
                                pyrefly_settings.get("disableLanguageServices")
                            {
                                self.update_disable_language_services(
                                    &id.scope_uri,
                                    *disable_language_services,
                                );
                            }
                            if let Some(serde_json::Value::Bool(disable_language_services)) =
                                pyrefly_settings.get("disableTypeErrors")
                            {
                                self.update_disable_type_errors(
                                    &id.scope_uri,
                                    *disable_language_services,
                                );
                            }
                        }
                    }
                    _ => {
                        // Non-map value or no configuration returned for request
                    }
                }
            }
            if modified {
                return self.validate_in_memory(ide_transaction_manager);
            }
        }
        Ok(())
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
    fn update_disable_type_errors(&self, scope_uri: &Option<Url>, disable_type_errors: bool) {
        let mut workspaces = self.workspaces.workspaces.write();
        match scope_uri {
            Some(scope_uri) => {
                if let Some(workspace) = workspaces.get_mut(&scope_uri.to_file_path().unwrap()) {
                    workspace.disable_type_errors = disable_type_errors;
                }
            }
            None => self.workspaces.default.write().disable_type_errors = disable_type_errors,
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
                message: format!("{:#?}", e),
                data: None,
            }),
        },
    }
}
