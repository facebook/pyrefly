/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::iter::once;
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
use lsp_types::request::GotoDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::WorkspaceConfiguration;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextSize;
use serde::de::DeserializeOwned;
use starlark_map::small_map::Iter;
use starlark_map::small_map::SmallMap;

use crate::PythonEnvironment;
use crate::clap_env;
use crate::commands::util::module_from_path;
use crate::config::error::ErrorConfigs;
use crate::metadata::RuntimeMetadata;
use crate::module::bundled::typeshed;
use crate::module::finder::find_module;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::SourceRange;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::run::CommandExitStatus;
use crate::state::handle::Handle;
use crate::state::loader::FindError;
use crate::state::loader::Loader;
use crate::state::loader::LoaderId;
use crate::state::require::Require;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionSavedState;
use crate::util::lock::Mutex;
use crate::util::lock::RwLock;
use crate::util::prelude::VecExt;

#[derive(Debug, Parser, Clone)]
pub struct Args {
    #[clap(long = "search-path", env = clap_env("SEARCH_PATH"))]
    pub(crate) search_path: Vec<PathBuf>,
    #[clap(long = "site-package-path", env = clap_env("SITE_PACKAGE_PATH"))]
    pub(crate) site_package_path: Vec<PathBuf>,
}

/// `IDETransactionManager` aims to always produce a transaction that contains the up-to-date
/// in-memory contents.
#[derive(Default)]
struct IDETransactionManager<'a> {
    /// Invariant:
    /// If it's None, then the main `State` already contains up-to-date checked content
    /// of all in-memory files.
    /// Otherwise, it will contains up-to-date checked content of all in-memory files.
    saved_state: Option<TransactionSavedState<'a>>,
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
            // so we can destory the saved state.
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
    /// After using it, the state should be saved by caling the `save` method.
    ///
    /// The `Transaction` will always contain the handles of all open files with the latest content.
    /// It might be created fresh from state, or reused from previously saved state.
    fn non_commitable_transaction(&mut self, state: &'a State) -> Transaction<'a> {
        let mut saved_state = None;
        std::mem::swap(&mut self.saved_state, &mut saved_state);
        if let Some(saved_state) = saved_state {
            saved_state.into_running()
        } else {
            state.transaction()
        }
    }

    /// This function should be called once we finished using transaction for an LSP request.
    fn save(&mut self, transaction: Transaction<'a>) {
        self.saved_state = Some(transaction.into_saved_state())
    }
}

struct Server {
    send: Arc<dyn Fn(Message) + Send + Sync + 'static>,
    initialize_params: InitializeParams,
    state: Arc<State>,
    configs: Arc<RwLock<SmallMap<PathBuf, Config>>>,
    default_config: Arc<Config>,
    search_path: Vec<PathBuf>,
    site_package_path: Vec<PathBuf>,
    outgoing_request_id: Arc<AtomicI32>,
    outgoing_requests: Mutex<SmallMap<RequestId, Request>>,
}

/// Temporary "configuration": this is all that is necessary to run an LSP at a given root.
/// TODO(connernilsel): replace with real config logic
#[derive(Debug)]
struct Config {
    open_files: Arc<Mutex<SmallMap<PathBuf, (i32, Arc<String>)>>>,
    runtime_metadata: RuntimeMetadata,
    search_path: Vec<PathBuf>,
    loader: LoaderId,
}

impl Config {
    fn new(
        search_path: Vec<PathBuf>,
        site_package_path: Vec<PathBuf>,
        runtime_metadata: RuntimeMetadata,
        open_files: SmallMap<PathBuf, (i32, Arc<String>)>,
    ) -> Self {
        let open_files = Arc::new(Mutex::new(open_files));
        Self {
            open_files: open_files.clone(),
            runtime_metadata,
            search_path: search_path.clone(),
            loader: LoaderId::new(LspLoader {
                open_files,
                search_path,
                site_package_path,
            }),
        }
    }

    fn open_file_handles(&self) -> Vec<(Handle, Require)> {
        self.open_files
            .lock()
            .keys()
            .map(|x| {
                (
                    Handle::new(
                        module_from_path(x, &self.search_path),
                        ModulePath::memory(x.clone()),
                        self.runtime_metadata.dupe(),
                        self.loader.dupe(),
                    ),
                    Require::Everything,
                )
            })
            .collect::<Vec<_>>()
    }

    fn compute_diagnostics(
        &self,
        transaction: &Transaction,
        handles: Vec<(Handle, Require)>,
    ) -> SmallMap<PathBuf, Vec<Diagnostic>> {
        let mut diags: SmallMap<PathBuf, Vec<Diagnostic>> = SmallMap::new();
        let open_files = self.open_files.lock();
        for x in open_files.keys() {
            diags.insert(x.as_path().to_owned(), Vec::new());
        }
        // TODO(connernilsen): replace with real error config from config file
        for e in transaction
            .get_loads(handles.iter().map(|(handle, _)| handle))
            .collect_errors(&ErrorConfigs::default())
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
        diags
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
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
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
    let send = move |msg| connection_for_send.sender.send(msg).unwrap();
    let server = Server::new(
        Arc::new(send),
        initialization_params,
        search_path,
        site_package_path,
    );
    eprintln!("Reading messages");
    let mut ide_transaction_manager = IDETransactionManager::default();
    let mut canceled_requests = HashSet::new();
    for msg in &connection.receiver {
        if matches!(&msg, Message::Request(req) if connection.handle_shutdown(req)?) {
            break;
        }
        server.process_lsp_message(&mut ide_transaction_manager, &mut canceled_requests, msg)?;
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

#[derive(Debug, Clone)]
struct LspLoader {
    open_files: Arc<Mutex<SmallMap<PathBuf, (i32, Arc<String>)>>>,
    search_path: Vec<PathBuf>,
    site_package_path: Vec<PathBuf>,
}

impl Loader for LspLoader {
    fn find_import(&self, module: ModuleName) -> Result<ModulePath, FindError> {
        if let Some(path) = find_module(module, &self.search_path) {
            Ok(path)
        } else if let Some(path) = typeshed().map_err(FindError::not_found)?.find(module) {
            Ok(path)
        } else if let Some(path) = find_module(module, &self.site_package_path) {
            Ok(path)
        } else {
            Err(FindError::search_path(
                &self.search_path,
                &self.site_package_path,
            ))
        }
    }

    fn load_from_memory(&self, path: &Path) -> Option<Arc<String>> {
        Some(self.open_files.lock().get(path)?.1.dupe())
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
                } else {
                    eprintln!("Unhandled request: {x:?}");
                }
                Ok(())
            }
            Message::Response(x) => {
                if let Some(request) = self.outgoing_requests.lock().shift_remove(&x.id) {
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
                    self.change_configuration();
                    Ok(())
                } else {
                    eprintln!("Unhandled notification: {x:?}");
                    Ok(())
                }
            }
        }
    }

    /// Finds a config for a file path: longest config which is a prefix of the file wins
    fn get_config<'a>(
        &self,
        configs: Iter<'a, PathBuf, Config>,
        default: &'a Config,
        uri: &Path,
    ) -> &'a Config {
        configs
            .filter(|(key, _)| uri.starts_with(key))
            .max_by(|(key1, _), (key2, _)| key2.ancestors().count().cmp(&key1.ancestors().count()))
            .map_or(default, |(_, config)| config)
    }

    /// TODO(connernilsen): replace with real config logic
    fn get_config_with<F, R>(&self, uri: PathBuf, f: F) -> R
    where
        F: FnOnce(&Config) -> R,
    {
        f(self.get_config(self.configs.read().iter(), &self.default_config, &uri))
    }

    fn new(
        send: Arc<dyn Fn(Message) + Send + Sync + 'static>,
        initialize_params: InitializeParams,
        search_path: Vec<PathBuf>,
        site_package_path: Vec<PathBuf>,
    ) -> Self {
        let folders = if let Some(capability) = initialize_params.clone().capabilities.workspace
            && let Some(true) = capability.workspace_folders
            && let Some(folders) = initialize_params.clone().workspace_folders
        {
            folders
                .iter()
                .map(|x| x.uri.to_file_path().unwrap())
                .collect()
        } else {
            Vec::new()
        };

        let mut s = Self {
            send,
            initialize_params,
            state: Arc::new(State::new()),
            configs: Arc::new(RwLock::new(SmallMap::new())),
            default_config: Arc::new(Config::new(
                search_path.clone(),
                site_package_path.clone(),
                RuntimeMetadata::default(),
                SmallMap::new(),
            )),
            search_path,
            site_package_path,
            outgoing_request_id: Arc::new(AtomicI32::new(1)),
            outgoing_requests: Mutex::new(SmallMap::new()),
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
        let mut all_handles = Vec::new();
        let mut config_with_handles = Vec::new();
        let configs = self.configs.read();
        configs
            .values()
            .chain(once(self.default_config.as_ref()))
            .for_each(|config| {
                let handles = config.open_file_handles();
                transaction.invalidate_memory(
                    config.loader.dupe(),
                    &config.open_files.lock().keys().cloned().collect::<Vec<_>>(),
                );
                all_handles.append(&mut handles.clone());
                config_with_handles.push((config, handles));
            });
        transaction.run(&all_handles);
        match possibly_committable_transaction {
            Ok(transaction) => {
                self.state.commit_transaction(transaction.into_changes());
                // In the case where we can commit transactions, `State` already has latest updates.
                // Therefore, we can compute errors from transactions freshly created from `State``.
                let transaction = self.state.transaction();
                for (config, handles) in config_with_handles {
                    self.publish_diagnostics(config.compute_diagnostics(&transaction, handles));
                }
            }
            Err(transaction) => {
                ide_transaction_manager.save(transaction);
            }
        }
        Ok(())
    }

    fn validate_with_disk_invalidation(&self, invalidate_disk: Vec<PathBuf>) -> anyhow::Result<()> {
        let state = self.state.dupe();
        let send = self.send.dupe();
        let default_config = self.default_config.dupe();
        let configs = self.configs.dupe();
        std::thread::spawn(move || {
            let mut transaction = state.new_committable_transaction(Require::Exports, None);
            transaction.as_mut().invalidate_disk(&invalidate_disk);
            state.run_with_committing_transaction(transaction, &[]);
            let transaction = state.transaction();
            for config in configs.read().values().chain(once(default_config.as_ref())) {
                publish_diagnostics(
                    send.dupe(),
                    config.compute_diagnostics(&transaction, config.open_file_handles()),
                );
            }
        });
        Ok(())
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
        self.get_config_with(uri.clone(), |config| {
            config.open_files.lock().insert(
                uri,
                (
                    params.text_document.version,
                    Arc::new(params.text_document.text),
                ),
            );
        });
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
        self.get_config_with(uri.clone(), |config| {
            config
                .open_files
                .lock()
                .insert(uri, (params.text_document.version, Arc::new(change.text)));
        });
        self.validate_in_memory(ide_transaction_manager)
    }

    fn did_close(&self, params: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_file_path().unwrap();
        self.get_config_with(uri.clone(), |config| {
            config
                .open_files
                .lock()
                .shift_remove(&params.text_document.uri.to_file_path().unwrap());
        });
        self.publish_diagnostics_for_uri(params.text_document.uri, Vec::new(), None);
        Ok(())
    }

    /// Configure the server with a new set of workspace folders
    fn configure(&mut self, config_paths: Vec<PathBuf>) {
        let mut new_configs = SmallMap::new();
        config_paths.iter().for_each(|x| {
            new_configs.insert(
                x.clone(),
                Config::new(
                    [x.clone()]
                        .into_iter()
                        .chain(self.search_path.clone())
                        .collect(),
                    self.site_package_path.clone(),
                    RuntimeMetadata::default(),
                    SmallMap::new(),
                ),
            );
            // todo(kylei): request settings for <DEFAULT> config (files not in any workspace folders)
            self.request_settings_for_config(&Url::from_file_path(x).unwrap());
        });
        self.configs = Arc::new(RwLock::new(new_configs));
    }

    fn make_handle(&self, uri: &Url) -> Handle {
        let path = uri.to_file_path().unwrap();
        self.get_config_with(path.clone(), |config| {
            let module = module_from_path(&path, &config.search_path);
            let module_path = if config.open_files.lock().contains_key(&path) {
                ModulePath::memory(path)
            } else {
                ModulePath::filesystem(path)
            };
            Handle::new(
                module,
                module_path,
                config.runtime_metadata.dupe(),
                config.loader.dupe(),
            )
        })
    }

    fn goto_definition(
        &self,
        transaction: &Transaction<'_>,
        params: GotoDefinitionParams,
    ) -> Option<GotoDefinitionResponse> {
        let handle = self.make_handle(&params.text_document_position_params.text_document.uri);
        let info = transaction.get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let TextRangeWithModuleInfo {
            module_info: definition_module_info,
            range,
        } = transaction.goto_definition(&handle, range)?;
        let path = to_real_path(definition_module_info.path())?;
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
        Some(GotoDefinitionResponse::Scalar(Location {
            uri: Url::from_file_path(path).unwrap(),
            range: source_range_to_range(&definition_module_info.source_range(range)),
        }))
    }

    fn completion(
        &self,
        transaction: &Transaction<'_>,
        params: CompletionParams,
    ) -> anyhow::Result<CompletionResponse> {
        let handle = self.make_handle(&params.text_document_position.text_document.uri);
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

    fn hover(&self, transaction: &Transaction<'_>, params: HoverParams) -> Option<Hover> {
        let handle = self.make_handle(&params.text_document_position_params.text_document.uri);
        let info = transaction.get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let t = transaction.hover(&handle, range)?;
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    r#"```python
{}
```"#,
                    t
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
        let handle = self.make_handle(&params.text_document.uri);
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

    fn change_configuration(&self) {
        self.configs.read().iter().for_each(|(scope_uri, _)| {
            self.request_settings_for_config(&Url::from_file_path(scope_uri).unwrap())
        });
    }

    fn request_settings_for_config(&self, scope_uri: &Url) {
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
                        Some(serde_json::Value::Object(map))
                            if let Some(serde_json::Value::String(python_path)) =
                                map.get("pythonPath") =>
                        {
                            self.update_pythonpath(&mut modified, scope_uri, python_path);
                        }
                        _ => {
                            // Received a response without a pythonPath
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

    fn update_pythonpath(&self, modified: &mut bool, scope_uri: &Url, python_path: &str) {
        let mut configs = self.configs.write();
        let config_path = scope_uri.to_file_path().unwrap();
        // Currently uses the default interpreter if the pythonPath is invalid
        let env = PythonEnvironment::get_interpreter_env(&PathBuf::from(python_path));
        // TODO(kylei): warn if interpreter could not be found
        if let Some(site_package_path) = env.clone().site_package_path
            && let Some(config) = configs.get(&config_path)
        {
            *modified = true;
            let search_path = config.search_path.clone();
            let new_config = Config::new(
                search_path,
                site_package_path,
                // this is okay, since `get_interpreter_env()` must return an environment with all values as `Some()`
                RuntimeMetadata::new(env.python_version.unwrap(), env.python_platform.unwrap()),
                config.open_files.lock().clone(),
            );
            configs.insert(config_path, new_config);
        }
    }
}

fn source_range_to_range(x: &SourceRange) -> lsp_types::Range {
    lsp_types::Range::new(
        source_location_to_position(&x.start),
        source_location_to_position(&x.end),
    )
}

fn source_location_to_position(x: &SourceLocation) -> lsp_types::Position {
    lsp_types::Position {
        line: x.row.to_zero_indexed() as u32,
        character: x.column.to_zero_indexed() as u32,
    }
}

fn text_size_to_position(info: &ModuleInfo, x: TextSize) -> lsp_types::Position {
    source_location_to_position(&info.source_location(x))
}

fn position_to_text_size(info: &ModuleInfo, position: lsp_types::Position) -> TextSize {
    info.to_text_size(position.line, position.character)
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
