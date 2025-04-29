/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::iter;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use clap::Parser;
use dupe::Dupe;
use lsp_server::Connection;
use lsp_server::ErrorCode;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::CompletionList;
use lsp_types::CompletionOptions;
use lsp_types::CompletionParams;
use lsp_types::CompletionResponse;
use lsp_types::ConfigurationItem;
use lsp_types::ConfigurationParams;
use lsp_types::Diagnostic;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidCloseTextDocumentParams;
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::DidSaveTextDocumentParams;
use lsp_types::DocumentHighlight;
use lsp_types::DocumentHighlightParams;
use lsp_types::DocumentSymbol;
use lsp_types::DocumentSymbolParams;
use lsp_types::DocumentSymbolResponse;
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
use lsp_types::ServerCapabilities;
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::notification::Cancel;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidCloseTextDocument;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::notification::DidSaveTextDocument;
use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::Completion;
use lsp_types::request::DocumentHighlightRequest;
use lsp_types::request::DocumentSymbolRequest;
use lsp_types::request::GotoDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::WorkspaceConfiguration;
use path_absolutize::Absolutize;
use serde::de::DeserializeOwned;
use starlark_map::small_map::SmallMap;

use crate::commands::config_finder::standard_config_finder;
use crate::commands::run::CommandExitStatus;
use crate::commands::util::module_from_path;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::environment::PythonEnvironment;
use crate::config::finder::ConfigFinder;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionData;
use crate::sys_info::SysInfo;
use crate::types::lsp::position_to_text_size;
use crate::types::lsp::source_range_to_range;
use crate::types::lsp::text_size_to_position;
use crate::util::arc_id::ArcId;
use crate::util::args::clap_env;
use crate::util::globs::Globs;
use crate::util::lock::Mutex;
use crate::util::lock::RwLock;
use crate::util::prelude::VecExt;

#[derive(Debug, Parser, Clone)]
pub struct Args {
    #[clap(long = "search-path", env = clap_env("SEARCH_PATH"))]
    pub(crate) search_path: Vec<PathBuf>,
    #[clap(long = "site-package-path", env = clap_env("SITE_PACKAGE_PATH"))]
    pub(crate) site_package_path: Vec<PathBuf>,
    /// Temporary way to obtain the list of all the files belong to a project.
    /// This is necessary to know what files should be considered during find references.
    /// The information should eventually come from configs. TODO(@connernilsen)
    #[clap(long = "experimental_project-path", env = clap_env("EXPERIMENTAL_PROJECT_PATH"))]
    pub(crate) experimental_project_path: Vec<PathBuf>,
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
    /// Produce a possibly committable transaction in order to recheck in-memory files.
    fn get_possibly_committable_transaction(
        &mut self,
        state: &'a State,
    ) -> Result<CommittingTransaction<'a>, Transaction<'a>> {
        // If there is no ongoing recheck due to on-disk changes, we should prefer to commit
        // the in-memory changes into the main state.
        if let Some(transaction) = state.try_new_committable_transaction(Require::Exports, None) {
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

/// Events that must be handled by the server as soon as possible.
/// The server will clear the queue of such event after processing each LSP message.
enum ImmediatelyHandledEvent {
    /// Notify the server that recheck finishes, so server can revalidate all in-memory content
    /// based on the latest `State`.
    RecheckFinished,
}

struct Server {
    send: Arc<dyn Fn(Message) + Send + Sync + 'static>,
    immediately_handled_events: Arc<Mutex<Vec<ImmediatelyHandledEvent>>>,
    initialize_params: InitializeParams,
    state: Arc<State>,
    open_files: RwLock<HashMap<PathBuf, Arc<String>>>,
    workspaces: Arc<Workspaces>,
    search_path: Vec<PathBuf>,
    site_package_path: Vec<PathBuf>,
    outgoing_request_id: Arc<AtomicI32>,
    outgoing_requests: Mutex<HashMap<RequestId, Request>>,
}

/// Temporary "configuration": this is all that is necessary to run an LSP at a given root.
/// TODO(connernilsel): replace with real config logic
#[derive(Debug)]
struct Workspace {
    sys_info: SysInfo,
    search_path: Vec<PathBuf>,
    /// The config implied by these settings
    config_file: ArcId<ConfigFile>,
    disable_language_services: bool,
}

impl Workspace {
    fn new(search_path: Vec<PathBuf>, site_package_path: Vec<PathBuf>, sys_info: SysInfo) -> Self {
        let mut config_file = ConfigFile::default();
        config_file.python_environment.python_version = Some(sys_info.version());
        config_file.python_environment.python_platform = Some(sys_info.platform().clone());
        config_file.python_environment.site_package_path = Some(site_package_path);
        config_file.search_path = search_path.clone();
        config_file.configure();
        let config_file = ArcId::new(config_file);

        Self {
            sys_info,
            search_path: search_path.clone(),
            config_file,
            disable_language_services: false,
        }
    }
}

struct Workspaces {
    default: Workspace,
    workspaces: RwLock<SmallMap<PathBuf, Workspace>>,
}

impl Workspaces {
    fn new(default: Workspace) -> Self {
        Self {
            default,
            workspaces: RwLock::new(SmallMap::new()),
        }
    }

    /// TODO(connernilsen): replace with real config logic
    fn get_with<F, R>(&self, uri: PathBuf, f: F) -> R
    where
        F: FnOnce(&Workspace) -> R,
    {
        let workspaces = self.workspaces.read();
        let workspace = workspaces
            .iter()
            .filter(|(key, _)| uri.starts_with(key))
            .max_by(|(key1, _), (key2, _)| key2.ancestors().count().cmp(&key1.ancestors().count()))
            .map_or(&self.default, |(_, workspace)| workspace);
        f(workspace)
    }

    fn config_finder(workspaces: &Arc<Workspaces>) -> ConfigFinder {
        let workspaces = workspaces.dupe();
        standard_config_finder(Arc::new(move |dir, mut config| {
            if let Some(dir) = dir {
                workspaces.get_with(dir.to_owned(), |w| {
                    config.python_environment = w.config_file.python_environment.clone();
                    config
                        .search_path
                        .extend_from_slice(&w.config_file.search_path);
                })
            };
            config.configure();
            config
        }))
    }
}

pub fn run_lsp(
    connection: Arc<Connection>,
    wait_on_connection: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
    args: Args,
) -> anyhow::Result<CommandExitStatus> {
    // Run the server and wait for the two threads to end (typically by trigger LSP Exit event).
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_owned()]),
            ..Default::default()
        }),
        document_highlight_provider: Some(OneOf::Left(true)),
        // Find references won't work properly if we don't know all the files.
        references_provider: if args.experimental_project_path.is_empty() {
            None
        } else {
            Some(OneOf::Left(true))
        },
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        ..Default::default()
    })
    .unwrap();
    let initialization_params = match connection.initialize(server_capabilities) {
        Ok(it) => serde_json::from_value(it).unwrap(),
        Err(e) => {
            // Use this in later versions of LSP server
            // if e.channel_is_disconnected() {
            // io_threads.join()?;
            // }
            return Err(e.into());
        }
    };
    let search_path = args.search_path;
    let site_package_path = args.site_package_path;
    let connection_for_send = connection.dupe();
    let send = move |msg| {
        if connection_for_send.sender.send(msg).is_err() {
            // On error, we know the channel is closed.
            // https://docs.rs/crossbeam/latest/crossbeam/channel/struct.Sender.html#method.send
            eprintln!("Connection closed.");
        };
    };
    let server = Server::new(
        Arc::new(send),
        initialization_params,
        search_path,
        site_package_path,
    );
    server.populate_all_project_files(&args.experimental_project_path);
    eprintln!("Reading messages");
    let mut ide_transaction_manager = IDETransactionManager::default();
    let mut canceled_requests = HashSet::new();
    for msg in &connection.receiver {
        if matches!(&msg, Message::Request(req) if connection.handle_shutdown(req)?) {
            break;
        }
        server.process_lsp_message(&mut ide_transaction_manager, &mut canceled_requests, msg)?;
        let immediately_handled_events = mem::take(&mut *server.immediately_handled_events.lock());
        for msg in immediately_handled_events {
            server.process_immediately_handled_event(&mut ide_transaction_manager, msg)?;
        }
    }
    wait_on_connection()?;

    // Shut down gracefully.
    eprintln!("shutting down server");
    Ok(CommandExitStatus::Success)
}

impl Args {
    pub fn run(mut self, extra_search_paths: Vec<PathBuf>) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        self.search_path.extend(extra_search_paths);
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

fn publish_diagnostics_for_uri(
    send: &Arc<dyn Fn(Message) + Send + Sync + 'static>,
    uri: Url,
    diags: Vec<Diagnostic>,
    version: Option<i32>,
) {
    send(Message::Notification(
        new_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(uri, diags, version)),
    ));
}

fn publish_diagnostics(
    send: Arc<dyn Fn(Message) + Send + Sync + 'static>,
    diags: SmallMap<PathBuf, Vec<Diagnostic>>,
) {
    for (path, diags) in diags {
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        match Url::from_file_path(&path) {
            Ok(uri) => publish_diagnostics_for_uri(&send, uri, diags, None),
            Err(_) => eprint!("Unable to convert path to uri: {path:?}"),
        }
    }
}

impl Server {
    fn process_immediately_handled_event<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        msg: ImmediatelyHandledEvent,
    ) -> anyhow::Result<()> {
        match msg {
            ImmediatelyHandledEvent::RecheckFinished => {
                self.validate_in_memory(ide_transaction_manager)?;
            }
        }
        Ok(())
    }

    fn process_lsp_message<'a>(
        &'a self,
        ide_transaction_manager: &mut IDETransactionManager<'a>,
        canceled_requests: &mut HashSet<RequestId>,
        msg: Message,
    ) -> anyhow::Result<()> {
        match msg {
            Message::Request(x) => {
                if canceled_requests.remove(&x.id) {
                    let message = format!("Request {} is canceled", x.id);
                    eprintln!("{message}");
                    self.send_response(Response::new_err(
                        x.id,
                        ErrorCode::RequestCanceled as i32,
                        message,
                    ));
                    return Ok(());
                }
                eprintln!("Handling non-canceled request ({})", x.id);
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
                } else {
                    eprintln!("Unhandled request: {x:?}");
                }
                Ok(())
            }
            Message::Response(x) => {
                if let Some(request) = self.outgoing_requests.lock().remove(&x.id) {
                    self.handle_response(ide_transaction_manager, &request, &x)
                } else {
                    eprintln!("Response for unknown request: {x:?}");
                    Ok(())
                }
            }
            Message::Notification(x) => {
                if let Some(params) = as_notification::<DidOpenTextDocument>(&x) {
                    self.did_open(ide_transaction_manager, params)
                } else if let Some(params) = as_notification::<DidChangeTextDocument>(&x) {
                    self.did_change(ide_transaction_manager, params)
                } else if let Some(params) = as_notification::<DidCloseTextDocument>(&x) {
                    self.did_close(params)
                } else if let Some(params) = as_notification::<DidSaveTextDocument>(&x) {
                    self.did_save(params)
                } else if let Some(params) = as_notification::<Cancel>(&x) {
                    let id = match params.id {
                        NumberOrString::Number(i) => RequestId::from(i),
                        NumberOrString::String(s) => RequestId::from(s),
                    };
                    canceled_requests.insert(id);
                    Ok(())
                } else if as_notification::<DidChangeConfiguration>(&x).is_some() {
                    self.change_workspace();
                    Ok(())
                } else {
                    eprintln!("Unhandled notification: {x:?}");
                    Ok(())
                }
            }
        }
    }

    fn new(
        send: Arc<dyn Fn(Message) + Send + Sync + 'static>,
        initialize_params: InitializeParams,
        search_path: Vec<PathBuf>,
        site_package_path: Vec<PathBuf>,
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

        let workspaces = Arc::new(Workspaces::new(Workspace::new(
            search_path.clone(),
            site_package_path.clone(),
            SysInfo::default(),
        )));

        let config_finder = Workspaces::config_finder(&workspaces);
        let s = Self {
            send,
            immediately_handled_events: Default::default(),
            initialize_params,
            state: Arc::new(State::new(config_finder)),
            open_files: RwLock::new(HashMap::new()),
            workspaces,
            search_path,
            site_package_path,
            outgoing_request_id: Arc::new(AtomicI32::new(1)),
            outgoing_requests: Mutex::new(HashMap::new()),
        };
        s.configure(folders);

        s
    }

    fn send_response(&self, x: Response) {
        (self.send)(Message::Response(x))
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
        (self.send)(Message::Request(request.clone()));
        self.outgoing_requests.lock().insert(id, request);
    }

    fn publish_diagnostics_for_uri(&self, uri: Url, diags: Vec<Diagnostic>, version: Option<i32>) {
        publish_diagnostics_for_uri(&self.send, uri, diags, version);
    }

    fn publish_diagnostics(&self, diags: SmallMap<PathBuf, Vec<Diagnostic>>) {
        publish_diagnostics(self.send.dupe(), diags);
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
        let handles = self
            .open_files
            .read()
            .keys()
            .map(|x| (self.make_open_handle(x), Require::Everything))
            .collect::<Vec<_>>();
        transaction.set_memory(
            self.open_files
                .read()
                .iter()
                .map(|x| (x.0.clone(), Some(x.1.dupe())))
                .collect::<Vec<_>>(),
        );
        transaction.run(&handles);

        let publish = |transaction: &Transaction| {
            let mut diags: SmallMap<PathBuf, Vec<Diagnostic>> = SmallMap::new();
            let open_files = self.open_files.read();
            for x in open_files.keys() {
                diags.insert(x.as_path().to_owned(), Vec::new());
            }
            // TODO(connernilsen): replace with real error config from config file
            for e in transaction
                .get_errors(handles.iter().map(|(handle, _)| handle))
                .collect_errors()
                .shown
            {
                if let Some(path) = to_real_path(e.path()) {
                    if open_files.contains_key(path) {
                        diags.entry(path.to_owned()).or_default().push(Diagnostic {
                            range: source_range_to_range(e.source_range()),
                            severity: Some(lsp_types::DiagnosticSeverity::ERROR),
                            source: Some("Pyrefly".to_owned()),
                            message: e.msg().to_owned(),
                            code: Some(lsp_types::NumberOrString::String(
                                e.error_kind().to_name().to_owned(),
                            )),
                            ..Default::default()
                        });
                    }
                }
            }
            self.publish_diagnostics(diags);
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

    /// Perform an invalidation of elements on `State` and commit them.
    /// Runs asynchronously. Returns immediately and may wait a while for a committable transaction.
    fn invalidate(&self, f: impl FnOnce(&mut Transaction) + Send + 'static) {
        let state = self.state.dupe();
        let immediately_handled_events = self.immediately_handled_events.dupe();
        std::thread::spawn(move || {
            let mut transaction = state.new_committable_transaction(Require::Exports, None);
            f(transaction.as_mut());
            state.commit_transaction(transaction);
            // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
            // the main event loop of the server. As a result, the server can do a revalidation of
            // all the in-memory files based on the fresh main State as soon as possible.
            immediately_handled_events
                .lock()
                .push(ImmediatelyHandledEvent::RecheckFinished);
        });
    }

    fn validate_with_disk_invalidation(&self, invalidate_disk: Vec<PathBuf>) -> anyhow::Result<()> {
        if !invalidate_disk.is_empty() {
            self.invalidate(move |t| t.invalidate_disk(&invalidate_disk));
        }
        Ok(())
    }

    /// Certain IDE features (e.g. find-references) require us to know the dependency graph of the
    /// entire project to work. This blocking function should be called during server initialization
    /// time if we intend to provide features like find-references, and should be called when config
    /// changes (currently this is a TODO).
    fn populate_all_project_files(&self, search_paths: &[PathBuf]) {
        if search_paths.is_empty() {
            return;
        }
        let immediately_handled_events = self.immediately_handled_events.dupe();
        eprintln!(
            "Populating all files in the project path ({:?}).",
            search_paths
        );
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        let mut handles = Vec::new();
        for search_path in search_paths {
            let paths = Globs::new_with_root(search_path, vec![".".to_owned()])
                .files()
                .unwrap_or_default();
            for path in paths {
                let module_name =
                    module_from_path(&path, search_paths).unwrap_or_else(ModuleName::unknown);
                handles.push((
                    Handle::new(
                        module_name,
                        ModulePath::filesystem(path),
                        SysInfo::default(),
                    ),
                    Require::Exports,
                ));
            }
        }
        eprintln!("Prepare to check {} files.", handles.len());
        transaction.as_mut().run(&handles);
        self.state.commit_transaction(transaction);
        // After we finished a recheck asynchronously, we immediately send `RecheckFinished` to
        // the main event loop of the server. As a result, the server can do a revalidation of
        // all the in-memory files based on the fresh main State as soon as possible.
        immediately_handled_events
            .lock()
            .push(ImmediatelyHandledEvent::RecheckFinished);
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
        self.open_files
            .write()
            .insert(uri, Arc::new(params.text_document.text));
        self.validate_in_memory(ide_transaction_manager)
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

    fn did_close(&self, params: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_file_path().unwrap();
        self.open_files.write().remove(&uri);
        self.publish_diagnostics_for_uri(params.text_document.uri, Vec::new(), None);
        Ok(())
    }

    /// Configure the server with a new set of workspace folders
    fn configure(&self, workspace_paths: Vec<PathBuf>) {
        let mut new_workspaces = SmallMap::new();
        for x in &workspace_paths {
            new_workspaces.insert(
                x.clone(),
                Workspace::new(
                    iter::once(x.clone())
                        .chain(self.search_path.clone())
                        .collect(),
                    self.site_package_path.clone(),
                    SysInfo::default(),
                ),
            );
            // todo(kylei): request settings for <DEFAULT> config (files not in any workspace folders)
            self.request_settings_for_workspace(&Url::from_file_path(x).unwrap());
        }
        *self.workspaces.workspaces.write() = new_workspaces;
    }

    fn module_name(&self, workspace: &Workspace, path: &ModulePath) -> ModuleName {
        let unknown = ModuleName::unknown();
        let config = self.state.config_finder().python_file(unknown, path);
        let mut search_path = Vec::new();
        if config.source != ConfigSource::Synthetic {
            search_path = config.search_path.clone();
        }
        search_path.extend_from_slice(&workspace.search_path);
        to_real_path(path)
            .and_then(|path| module_from_path(path, &search_path))
            .unwrap_or(unknown)
    }

    fn make_open_handle(&self, path: &Path) -> Handle {
        self.workspaces.get_with(path.to_owned(), |workspace| {
            let path = ModulePath::memory(path.to_owned());
            let name = self.module_name(workspace, &path);
            Handle::new(name, path, workspace.sys_info.dupe())
        })
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
                let name = self.module_name(workspace, &module_path);
                Some(Handle::new(name, module_path, workspace.sys_info.dupe()))
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
        Some(GotoDefinitionResponse::Scalar(Location {
            uri: module_info_to_uri(&definition_module_info)?,
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

    fn hover(&self, transaction: &Transaction<'_>, params: HoverParams) -> Option<Hover> {
        let uri = &params.text_document_position_params.text_document.uri;
        let handle = self.make_handle_if_enabled(uri)?;
        let info = transaction.get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let t = transaction.get_type_at(&handle, range)?;
        let docstring = transaction.docstring(&handle, range);
        let value = match docstring {
            None => format!("```python\n{}\n```", t),
            Some(docstring) => format!(
                "```python\n{}\n```\n---\n{}",
                t,
                docstring.as_string().trim()
            ),
        };
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value,
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

    fn change_workspace(&self) {
        self.workspaces
            .workspaces
            .read()
            .keys()
            .for_each(|scope_uri| {
                self.request_settings_for_workspace(&Url::from_file_path(scope_uri).unwrap())
            });
    }

    fn request_settings_for_workspace(&self, scope_uri: &Url) {
        if let Some(workspace) = &self.initialize_params.capabilities.workspace
            && workspace.configuration == Some(true)
        {
            self.send_request::<WorkspaceConfiguration>(ConfigurationParams {
                items: Vec::from([ConfigurationItem {
                    scope_uri: Some(scope_uri.clone()),
                    section: Some("python".to_owned()),
                }]),
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
                if let Some(scope_uri) = &id.scope_uri {
                    match response.get(i) {
                        Some(serde_json::Value::Object(map)) => {
                            if let Some(serde_json::Value::String(python_path)) =
                                map.get("pythonPath")
                            {
                                self.update_pythonpath(&mut modified, scope_uri, python_path);
                            }
                            if let Some(serde_json::Value::Object(pyrefly_settings)) =
                                map.get("pyrefly")
                                && let Some(serde_json::Value::Bool(disable_language_services)) =
                                    pyrefly_settings.get("disableLanguageServices")
                            {
                                self.update_disable_language_services(
                                    scope_uri,
                                    *disable_language_services,
                                );
                            }
                        }
                        _ => {
                            // Non-map value or no configuration returned for request
                        }
                    };
                }
            }
            if modified {
                return self.validate_in_memory(ide_transaction_manager);
            }
        }
        Ok(())
    }

    fn update_disable_language_services(&self, scope_uri: &Url, disable_language_services: bool) {
        let mut workspaces = self.workspaces.workspaces.write();
        let workspace_path = scope_uri.to_file_path().unwrap();
        if let Some(workspace) = workspaces.get_mut(&workspace_path) {
            workspace.disable_language_services = disable_language_services;
        }
    }

    fn invalidate_config(&self) {
        self.invalidate(|t| t.invalidate_config());
    }

    fn update_pythonpath(&self, modified: &mut bool, scope_uri: &Url, python_path: &str) {
        let mut workspaces = self.workspaces.workspaces.write();
        let workspace_path = scope_uri.to_file_path().unwrap();
        // Currently uses the default interpreter if the pythonPath is invalid
        let env = PythonEnvironment::get_interpreter_env(Path::new(python_path));
        // TODO(kylei): warn if interpreter could not be found
        if let Some(site_package_path) = &env.site_package_path
            && let Some(workspace) = workspaces.get(&workspace_path)
        {
            *modified = true;
            let search_path = workspace.search_path.clone();
            let new_workspace = Workspace::new(
                search_path,
                site_package_path.clone(),
                // this is okay, since `get_interpreter_env()` must return an environment with all values as `Some()`
                SysInfo::new(env.python_version.unwrap(), env.python_platform.unwrap()),
            );
            workspaces.insert(workspace_path, new_workspace);
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
