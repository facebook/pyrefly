/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This file contains a new implementation of the lsp_interaction test suite. Soon it will replace the old one.
use std::iter::once;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use std::thread::{self};
use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use lsp_server::Connection;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::CompletionList;
use lsp_types::CompletionResponse;
use lsp_types::ConfigurationItem;
use lsp_types::ConfigurationParams;
use lsp_types::HoverContents;
use lsp_types::PublishDiagnosticsParams;
use lsp_types::RegistrationParams;
use lsp_types::UnregistrationParams;
use lsp_types::Url;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::notification::DidChangeNotebookDocument;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidChangeWatchedFiles;
use lsp_types::notification::DidCloseNotebookDocument;
use lsp_types::notification::DidOpenNotebookDocument;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::notification::Exit;
use lsp_types::notification::Initialized;
use lsp_types::notification::Notification as _;
use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::Completion;
use lsp_types::request::DocumentDiagnosticRequest;
use lsp_types::request::GotoDefinition;
use lsp_types::request::GotoImplementation;
use lsp_types::request::GotoTypeDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::Initialize;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::References;
use lsp_types::request::RegisterCapability;
use lsp_types::request::Request as _;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::request::SemanticTokensRangeRequest;
use lsp_types::request::Shutdown;
use lsp_types::request::SignatureHelpRequest;
use lsp_types::request::UnregisterCapability;
use lsp_types::request::WillRenameFiles;
use lsp_types::request::WorkspaceConfiguration;
use pretty_assertions::assert_eq;
use pyrefly_util::fs_anyhow::read_to_string;
use pyrefly_util::lock::FinishHandle;
use pyrefly_util::telemetry::NoTelemetry;
use serde_json::Value;
use serde_json::json;

use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::LspArgs;
use crate::commands::lsp::run_lsp;
use crate::lsp::wasm::provide_type::ProvideType;
use crate::test::util::init_test;

#[derive(Debug)]
pub enum LspMessageError {
    Timeout { description: String },
    Disconnected { description: String },
}

impl std::fmt::Display for LspMessageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LspMessageError::Timeout { description } => {
                write!(f, "Timeout waiting for message: {description}")
            }
            LspMessageError::Disconnected { description } => {
                write!(
                    f,
                    "Channel disconnected while waiting for message: {description}"
                )
            }
        }
    }
}

impl std::error::Error for LspMessageError {}

#[derive(Default)]
pub struct InitializeSettings {
    pub workspace_folders: Option<Vec<(String, Url)>>,
    // initial configuration to send after initialization
    // When Some, configuration will be sent after initialization
    // When None, no configuration will be sent
    // When Some(None), empty configuration will be sent
    pub configuration: Option<Option<serde_json::Value>>,
    pub file_watch: bool,
    // Additional capabilities to merge into the initialize params
    pub capabilities: Option<serde_json::Value>,
    // initialization_options to send in the initialize request
    pub initialization_options: Option<serde_json::Value>,
}

pub struct ClientRequestHandle<'a, R: lsp_types::request::Request> {
    id: RequestId,
    client: &'a TestClient,
    _type: PhantomData<R>,
}

pub struct ServerRequestHandle<'a, R: lsp_types::request::Request> {
    id: RequestId,
    client: &'a TestClient,
    _type: PhantomData<R>,
}

impl<'a, R: lsp_types::request::Request> ClientRequestHandle<'a, R> {
    pub fn expect_response(self, expected: Value) -> Result<(), LspMessageError> {
        self.client.expect_response::<R>(self.id, expected)
    }

    pub fn expect_response_error(self, expected: Value) -> Result<(), LspMessageError> {
        self.client.expect_response_error(self.id, expected)
    }

    pub fn expect_response_with(
        self,
        matcher: impl Fn(R::Result) -> bool,
    ) -> Result<(), LspMessageError> {
        self.client.expect_response_with::<R>(self.id, matcher)
    }
}

impl<'a> ClientRequestHandle<'a, Completion> {
    pub fn expect_completion_response_with(
        self,
        matcher: impl Fn(&CompletionList) -> bool,
    ) -> Result<(), LspMessageError> {
        self.client
            .expect_completion_response_with(self.id, matcher)
    }
}

impl<'a> ClientRequestHandle<'a, HoverRequest> {
    pub fn expect_hover_response_with_markup(
        self,
        matcher: impl Fn(Option<&str>) -> bool,
    ) -> Result<(), LspMessageError> {
        self.client
            .expect_hover_response_with_markup(self.id, matcher)
    }
}

impl<'a> ClientRequestHandle<'a, GotoDefinition> {
    pub fn expect_definition_response_from_root(
        self,
        file: &'static str,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) -> Result<(), LspMessageError> {
        self.client.expect_definition_response_from_root(
            self.id, file, line_start, char_start, line_end, char_end,
        )
    }
}

impl<'a> ClientRequestHandle<'a, GotoTypeDefinition> {
    pub fn expect_definition_response_from_root(
        self,
        file: &'static str,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) -> Result<(), LspMessageError> {
        self.client.expect_definition_response_from_root(
            self.id, file, line_start, char_start, line_end, char_end,
        )
    }
}

impl<'a> ClientRequestHandle<'a, GotoImplementation> {
    pub fn expect_implementation_response_from_root(
        self,
        implementations: Vec<(&'static str, u32, u32, u32, u32)>,
    ) -> Result<(), LspMessageError> {
        self.client
            .expect_implementation_response_from_root(self.id, implementations)
    }
}

impl<'a, R: lsp_types::request::Request> ServerRequestHandle<'a, R> {
    pub fn send_response(self, result: Value) {
        self.client.send_response::<R>(self.id, result)
    }
}

impl<'a> ServerRequestHandle<'a, WorkspaceConfiguration> {
    pub fn send_configuration_response(self, result: Value) {
        self.client.send_configuration_response(self.id, result);
    }
}

pub struct TestClient {
    conn: Option<Connection>,
    root: Option<PathBuf>,
    send_timeout: Duration,
    recv_timeout: Duration,
    /// Request ID for requests sent to the server
    request_idx: AtomicI32,
    /// Handle to wait for the server to exit
    finish_handle: Arc<FinishHandle>,
}

impl TestClient {
    pub fn new(connection: Connection, finish_handle: Arc<FinishHandle>) -> Self {
        Self {
            conn: Some(connection),
            root: None,
            send_timeout: Duration::from_secs(25),
            recv_timeout: Duration::from_secs(50),
            request_idx: AtomicI32::new(0),
            finish_handle,
        }
    }

    fn get_root_or_panic(&self) -> PathBuf {
        self.root
            .clone()
            .expect("Root not set, please call set_root")
    }

    fn next_request_id(&self) -> RequestId {
        let idx = self.request_idx.fetch_add(1, Ordering::SeqCst);
        RequestId::from(idx + 1)
    }

    pub fn drop_connection(&mut self) {
        // Take and drop to close the connection
        drop(std::mem::take(&mut self.conn))
    }

    pub fn expect_stop(&self) {
        if !self.finish_handle.wait_for_finish(Duration::from_secs(10)) {
            panic!("Server did not shutdown in time");
        }
    }

    #[allow(clippy::result_large_err)]
    fn send_timeout(
        &self,
        message: Message,
    ) -> Result<(), crossbeam_channel::SendTimeoutError<Message>> {
        self.conn
            .as_ref()
            .unwrap()
            .sender
            .send_timeout(message, self.send_timeout)
    }

    fn recv_timeout(&self) -> Result<Message, crossbeam_channel::RecvTimeoutError> {
        self.conn
            .as_ref()
            .unwrap()
            .receiver
            .recv_timeout(self.recv_timeout)
    }

    pub fn send_message(&self, message: Message) {
        eprintln!(
            "client--->server {}",
            serde_json::to_string(&message).unwrap()
        );
        if let Err(err) = self.send_timeout(message.clone()) {
            panic!("Failed to send message to language server: {err}");
        }
    }

    pub fn send_request<R: lsp_types::request::Request>(
        &self,
        params: serde_json::Value,
    ) -> ClientRequestHandle<'_, R> {
        // Ensure the passed value can be parsed as the desired request params
        let params = serde_json::from_value::<R::Params>(params).unwrap();
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id: id.clone(),
            method: R::METHOD.to_owned(),
            params: serde_json::to_value(params).unwrap(),
        }));
        ClientRequestHandle {
            id,
            client: self,
            _type: PhantomData,
        }
    }

    pub fn send_response<R: lsp_types::request::Request>(&self, id: RequestId, result: Value) {
        // Ensure the passed value can be parsed as the desired response result
        let result = serde_json::from_value::<R::Result>(result).unwrap();
        self.send_message(Message::Response(Response {
            id,
            result: Some(serde_json::to_value(result).unwrap()),
            error: None,
        }));
    }

    pub fn send_notification<N: lsp_types::notification::Notification>(
        &self,
        params: serde_json::Value,
    ) {
        // Ensure the passed value can be parsed as the desired notification params
        let params = serde_json::from_value::<N::Params>(params).unwrap();
        self.send_message(Message::Notification(Notification {
            method: N::METHOD.to_owned(),
            params: serde_json::to_value(params).unwrap(),
        }));
    }

    pub fn send_initialize(&self, params: Value) -> ClientRequestHandle<'_, Initialize> {
        self.send_request(params)
    }

    pub fn send_initialized(&self) {
        self.send_notification::<Initialized>(json!({}));
    }

    pub fn send_shutdown(&self) -> ClientRequestHandle<'_, Shutdown> {
        self.send_request(json!(null))
    }

    pub fn send_exit(&self) {
        self.send_notification::<Exit>(json!(null));
    }

    pub fn type_definition(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, GotoTypeDefinition> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
            },
            "position": {
                "line": line,
                "character": col,
            },
        }))
    }

    pub fn definition(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, GotoDefinition> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
            },
            "position": {
                "line": line,
                "character": col,
            },
        }))
    }

    pub fn implementation(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, GotoImplementation> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
            },
            "position": {
                "line": line,
                "character": col,
            },
        }))
    }

    pub fn did_open(&self, file: &'static str) {
        let path = self.get_root_or_panic().join(file);
        self.send_notification::<DidOpenTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 1,
                "text": read_to_string(&path).unwrap(),
            },
        }));
    }

    pub fn did_open_uri(&self, uri: &Url, language_id: &str, text: impl Into<String>) {
        self.send_notification::<DidOpenTextDocument>(json!({
            "textDocument": {
                "uri": uri.to_string(),
                "languageId": language_id,
                "version": 1,
                "text": text.into(),
            },
        }));
    }

    pub fn did_change(&self, file: &str, contents: &str) {
        let path = self.get_root_or_panic().join(file);
        self.send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": contents.to_owned()
            }],
        }));
    }

    pub fn did_change_configuration(&self) {
        self.send_notification::<DidChangeConfiguration>(json!({"settings": {}}));
    }

    pub fn completion(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, Completion> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    pub fn diagnostic(
        &self,
        file: &'static str,
    ) -> ClientRequestHandle<'_, DocumentDiagnosticRequest> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
        "textDocument": {
            "uri": Url::from_file_path(&path).unwrap().to_string()
        }}))
    }

    pub fn hover(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, HoverRequest> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    pub fn provide_type(
        &self,
        file: &'static str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, ProvideType> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "positions": [{
                "line": line,
                "character": col
            }]
        }))
    }

    pub fn references(
        &self,
        file: &str,
        line: u32,
        col: u32,
        include_declaration: bool,
    ) -> ClientRequestHandle<'_, References> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": line,
                "character": col
            },
            "context": {
                "includeDeclaration": include_declaration
            },
        }))
    }

    pub fn inlay_hint(
        &self,
        file: &'static str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) -> ClientRequestHandle<'_, InlayHintRequest> {
        let path = self.get_root_or_panic().join(file);
        self.send_request(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "range": {
                "start": {
                    "line": start_line,
                    "character": start_char
                },
                "end": {
                    "line": end_line,
                    "character": end_char
                }
            }
        }))
    }

    pub fn send_configuration_response(&self, id: RequestId, result: serde_json::Value) {
        self.send_response::<WorkspaceConfiguration>(id, result);
    }

    pub fn will_rename_files(
        &self,
        old_file: &'static str,
        new_file: &'static str,
    ) -> ClientRequestHandle<'_, WillRenameFiles> {
        let root = self.get_root_or_panic();
        let old_path = root.join(old_file);
        let new_path = root.join(new_file);
        self.send_request(json!({
            "files": [{
                "oldUri": Url::from_file_path(&old_path).unwrap().to_string(),
                "newUri": Url::from_file_path(&new_path).unwrap().to_string()
            }]
        }))
    }

    /// Send a file creation event notification
    pub fn file_created(&self, file: &str) {
        let path = self.get_root_or_panic().join(file);
        self.send_notification::<DidChangeWatchedFiles>(json!({
            "changes": [{
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "type": 1,  // FileChangeType::CREATED
            }],
        }));
    }

    /// Send a file modification event notification
    pub fn file_modified(&self, file: &str) {
        let path = self.get_root_or_panic().join(file);
        self.send_notification::<DidChangeWatchedFiles>(json!({
            "changes": [{
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "type": 2,  // FileChangeType::CHANGED
            }],
        }));
    }

    /// Send a file deletion event notification
    pub fn file_deleted(&self, file: &str) {
        let path = self.get_root_or_panic().join(file);
        self.send_notification::<DidChangeWatchedFiles>(json!({
            "changes": [{
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "type": 3,  // FileChangeType::DELETED
            }],
        }));
    }

    pub fn get_initialize_params(&self, settings: &InitializeSettings) -> Value {
        let mut params: Value = json!({
            "rootPath": "/",
            "processId": std::process::id(),
            "trace": "verbose",
            "clientInfo": { "name": "debug" },
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": true,
                        "versionSupport": true,
                        "tagSupport": {
                            "valueSet": [1, 2],
                        },
                        "codeDescriptionSupport": true,
                        "dataSupport": true,
                    },
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": true
                    },
                },
            },
        });

        if let Some(folders) = &settings.workspace_folders {
            params["capabilities"]["workspace"]["workspaceFolders"] = json!(true);
            params["workspaceFolders"] = json!(
                folders
                    .iter()
                    .map(|(name, path)| json!({"name": name, "uri": path.to_string()}))
                    .collect::<Vec<_>>()
            );
        }
        if settings.file_watch {
            params["capabilities"]["workspace"]["didChangeWatchedFiles"] =
                json!({"dynamicRegistration": true});
        }
        if settings.configuration.is_some() {
            params["capabilities"]["workspace"]["configuration"] = json!(true);
        }
        if let Some(init_options) = &settings.initialization_options {
            params["initializationOptions"] = init_options.clone();
        }

        // Merge custom capabilities if provided
        if let Some(custom_capabilities) = &settings.capabilities {
            Self::merge_json(&mut params["capabilities"], custom_capabilities);
        }

        params
    }

    pub fn expect_message<T>(
        &self,
        description: &str,
        matcher: impl Fn(Message) -> Option<T>,
    ) -> Result<T, LspMessageError> {
        loop {
            match self.recv_timeout() {
                Ok(msg) => {
                    eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                    if let Some(actual) = matcher(msg) {
                        return Ok(actual);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(LspMessageError::Timeout {
                        description: format!("Timeout waiting for message: {description}"),
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(LspMessageError::Disconnected {
                        description: format!(
                            "Channel disconnected while waiting for message: {description}"
                        ),
                    });
                }
            }
        }
    }

    pub fn expect_request<R: lsp_types::request::Request>(
        &self,
        expected: Value,
    ) -> Result<ServerRequestHandle<'_, R>, LspMessageError> {
        // Validate that expected can be parsed as R::Params
        let expected: R::Params = serde_json::from_value(expected.clone()).unwrap();
        let id = self.expect_message(&format!("Request {}", R::METHOD), |msg| {
            if let Message::Request(x) = msg {
                assert_eq!(x.method, R::METHOD);
                let actual: R::Params = serde_json::from_value(x.params.clone()).unwrap();
                assert_eq!(json!(expected), json!(actual));
                Some(x.id)
            } else {
                None
            }
        })?;
        Ok(ServerRequestHandle {
            id,
            client: self,
            _type: PhantomData,
        })
    }

    pub fn expect_response<R: lsp_types::request::Request>(
        &self,
        id: RequestId,
        expected: Value,
    ) -> Result<(), LspMessageError> {
        // Validate that expected can be parsed as R::Result
        let expected: R::Result = serde_json::from_value(expected.clone()).unwrap();
        let actual: R::Result =
            self.expect_message(&format!("Response {} id={}", R::METHOD, id), |msg| {
                if let Message::Response(x) = msg
                    && x.id == id
                {
                    Some(serde_json::from_value(x.result.unwrap()).unwrap())
                } else {
                    None
                }
            })?;
        assert_eq!(json!(expected), json!(actual));
        Ok(())
    }

    pub fn expect_response_error(
        &self,
        id: RequestId,
        expected: Value,
    ) -> Result<(), LspMessageError> {
        let expected: ResponseError = serde_json::from_value(expected).unwrap();
        let actual = self.expect_message(&format!("Response error id={}", id), |msg| {
            if let Message::Response(x) = msg
                && x.id == id
            {
                Some(x.error.unwrap())
            } else {
                None
            }
        })?;
        assert_eq!(json!(expected), json!(actual));
        Ok(())
    }

    pub fn expect_response_with<R: lsp_types::request::Request>(
        &self,
        id: RequestId,
        matcher: impl Fn(R::Result) -> bool,
    ) -> Result<(), LspMessageError> {
        self.expect_message(
            &format!("Response {} matching condition", R::METHOD),
            |msg| {
                if let Message::Response(x) = msg
                    && x.id == id
                    && matcher(serde_json::from_value::<R::Result>(x.result.unwrap()).unwrap())
                {
                    Some(())
                } else {
                    None
                }
            },
        )?;
        Ok(())
    }

    pub fn expect_completion_response_with(
        &self,
        id: RequestId,
        matcher: impl Fn(&CompletionList) -> bool,
    ) -> Result<(), LspMessageError> {
        self.expect_response_with::<Completion>(id, |result| {
            // Pyrefly always returns a CompletionList
            match result {
                Some(CompletionResponse::List(x)) => matcher(&x),
                _ => panic!("Unexpected completion response: {result:?}"),
            }
        })
    }

    pub fn expect_hover_response_with_markup(
        &self,
        id: RequestId,
        matcher: impl Fn(Option<&str>) -> bool,
    ) -> Result<(), LspMessageError> {
        self.expect_response_with::<HoverRequest>(id, |result| {
            // Pyrefly always returns either an empty array or markup
            let hover = result.expect("Unexpected null completion response");
            match hover.contents {
                HoverContents::Array(xs) if xs.is_empty() => matcher(None),
                HoverContents::Markup(content) => matcher(Some(&content.value)),
                _ => panic!("Unexpected completion response: {hover:?}"),
            }
        })
    }

    /// Wait for a publishDiagnostics notification, then check if it contains the message
    pub fn expect_publish_diagnostics_message_contains(
        &self,
        path: PathBuf,
        message: &str,
    ) -> Result<(), LspMessageError> {
        self.expect_message(
            &format!(
                "publishDiagnostics notification containing message '{message}' for file: {}",
                path.display()
            ),
            |msg| {
                if let Message::Notification(x) = msg
                    && x.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(x.params).unwrap();
                    if params.uri.to_file_path().ok().as_ref() == Some(&path)
                        && params
                            .diagnostics
                            .iter()
                            .any(|d| d.message.contains(message))
                    {
                        Some(())
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        )?;
        Ok(())
    }

    /// Wait for a publishDiagnostics notification, then check if it has the correct path and count
    pub fn expect_publish_diagnostics_error_count(
        &self,
        path: PathBuf,
        count: usize,
    ) -> Result<(), LspMessageError> {
        self.expect_message(
            &format!(
                "publishDiagnostics notification with {count} errors for file: {}",
                path.display()
            ),
            |msg| {
                if let Message::Notification(x) = msg
                    && x.method == PublishDiagnostics::METHOD
                {
                    let params =
                        serde_json::from_value::<PublishDiagnosticsParams>(x.params).unwrap();
                    if params.uri.to_file_path().unwrap() == path
                        && params.diagnostics.len() == count
                    {
                        Some(())
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        )?;
        Ok(())
    }

    pub fn expect_publish_diagnostics_uri(
        &self,
        uri: &Url,
        count: usize,
    ) -> Result<(), LspMessageError> {
        self.expect_message(
            &format!("publishDiagnostics notification {count} errors for uri: {uri}"),
            |msg| {
                if let Message::Notification(x) = msg
                    && x.method == PublishDiagnostics::METHOD
                {
                    let params: PublishDiagnosticsParams =
                        serde_json::from_value(x.params).unwrap();
                    if params.uri == *uri && params.diagnostics.len() == count {
                        Some(())
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        )?;
        Ok(())
    }

    pub fn expect_definition_response_from_root(
        &self,
        id: RequestId,
        file: &'static str,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) -> Result<(), LspMessageError> {
        self.expect_response::<GotoDefinition>(
            id,
            json!(
            {
                "uri": Url::from_file_path(self.get_root_or_panic().join(file)).unwrap().to_string(),
                "range": {
                    "start": {"line": line_start, "character": char_start},
                    "end": {"line": line_end, "character": char_end}
                },
                }),
        )
    }

    pub fn expect_implementation_response_from_root(
        &self,
        id: RequestId,
        implementations: Vec<(&'static str, u32, u32, u32, u32)>,
    ) -> Result<(), LspMessageError> {
        let locations: Vec<_> = implementations
            .into_iter()
            .map(|(file, line_start, char_start, line_end, char_end)| {
                json!({
                    "uri": Url::from_file_path(self.get_root_or_panic().join(file)).unwrap().to_string(),
                    "range": {
                        "start": {"line": line_start, "character": char_start},
                        "end": {"line": line_end, "character": char_end}
                    },
                })
            })
            .collect();

        self.expect_response::<GotoImplementation>(id, json!(locations))
    }

    pub fn expect_any_message(&self) -> Result<(), LspMessageError> {
        match self.recv_timeout() {
            Ok(msg) => {
                eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                Ok(())
            }
            Err(RecvTimeoutError::Timeout) => Err(LspMessageError::Timeout {
                description: "Timeout waiting for response message".to_owned(),
            }),
            Err(RecvTimeoutError::Disconnected) => Err(LspMessageError::Disconnected {
                description: "Channel disconnected".to_owned(),
            }),
        }
    }

    pub fn expect_configuration_request(
        &self,
        scope_uris: Option<Vec<&Url>>,
    ) -> Result<ServerRequestHandle<'_, WorkspaceConfiguration>, LspMessageError> {
        let items = scope_uris
            .unwrap_or_default()
            .into_iter()
            .cloned()
            .map(Some)
            .chain(once(None))
            .map(|scope_uri| ConfigurationItem {
                scope_uri,
                section: Some("python".to_owned()),
            })
            .collect::<Vec<_>>();

        self.expect_request(json!(ConfigurationParams { items }))
    }

    /// Expect a file watcher registration request.
    /// Validates that the request is specifically registering the file watcher (ID: "FILEWATCHER").
    pub fn expect_file_watcher_register(&self) -> Result<(), LspMessageError> {
        let params: RegistrationParams =
            self.expect_message(&format!("Request {}", RegisterCapability::METHOD), |msg| {
                if let Message::Request(x) = msg
                    && x.method == RegisterCapability::METHOD
                {
                    Some(serde_json::from_value(x.params).unwrap())
                } else {
                    None
                }
            })?;
        assert!(params.registrations.iter().any(|x| x.id == "FILEWATCHER"));
        Ok(())
    }

    /// Expect a file watcher unregistration request.
    /// Validates that the request is specifically unregistering the file watcher (ID: "FILEWATCHER").
    pub fn expect_file_watcher_unregister(&self) -> Result<(), LspMessageError> {
        let params: UnregistrationParams = self.expect_message(
            &format!("Request {}", UnregisterCapability::METHOD),
            |msg| {
                if let Message::Request(x) = msg
                    && x.method == UnregisterCapability::METHOD
                {
                    Some(serde_json::from_value(x.params).unwrap())
                } else {
                    None
                }
            },
        )?;
        assert!(
            params
                .unregisterations
                .iter()
                .any(|x| x.id == "FILEWATCHER")
        );
        Ok(())
    }

    #[allow(dead_code)]
    pub fn untyped_import_diagnostic_response(
        package_name: &str,
        line: u32,
        start_character: u32,
        end_character: u32,
        severity: u32,
    ) -> Value {
        json!({
            "items": [
                {
                    "code": "untyped-import",
                    "codeDescription": {
                        "href": "https://pyrefly.org/en/docs/error-kinds/#untyped-import"
                    },
                    "message": format!("Missing type stubs for `{}`\n  Hint: install the `{}-stubs` package", package_name, package_name),
                    "range": {
                        "start": {"line": line, "character": start_character},
                        "end": {"line": line, "character": end_character}
                    },
                    "severity": severity,
                    "source": "Pyrefly"
                }
            ],
            "kind": "full"
        })
    }

    /// Helper function to merge JSON values, with the source taking precedence
    fn merge_json(target: &mut Value, source: &Value) {
        if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
            for (key, value) in source_obj {
                if let Some(target_value) = target_obj.get_mut(key) {
                    // If both are objects, merge recursively
                    if target_value.is_object() && value.is_object() {
                        Self::merge_json(target_value, value);
                    } else {
                        // Otherwise, overwrite with source value
                        *target_value = value.clone();
                    }
                } else {
                    // Key doesn't exist in target, insert it
                    target_obj.insert(key.clone(), value.clone());
                }
            }
        }
    }
}

pub struct LspInteraction {
    pub client: TestClient,
}

impl LspInteraction {
    pub fn new() -> Self {
        Self::new_with_indexing_mode(IndexingMode::None)
    }

    pub fn new_with_indexing_mode(indexing_mode: IndexingMode) -> Self {
        init_test();

        let (conn_client, conn_server) = Connection::memory();

        let finish_handle = Arc::new(FinishHandle::new());
        let finish_server = finish_handle.clone();

        // Spawn the server thread notify when finished
        thread::spawn(move || {
            let args = LspArgs {
                indexing_mode,
                workspace_indexing_limit: 50,
                build_system_blocking: false,
            };
            let _ = run_lsp(conn_server, args, "pyrefly-lsp-test-version", &NoTelemetry);
            finish_server.notify_finished();
        });

        let client = TestClient::new(conn_client, finish_handle);

        Self { client }
    }

    pub fn initialize(&self, settings: InitializeSettings) -> Result<(), LspMessageError> {
        self.client
            .send_initialize(self.client.get_initialize_params(&settings));
        self.client.expect_any_message()?;
        self.client.send_initialized();
        if let Some(settings) = settings.configuration {
            self.client.expect_any_message()?;
            self.client.send_response::<WorkspaceConfiguration>(
                RequestId::from(1),
                settings.unwrap_or(json!([])),
            );
        }
        Ok(())
    }

    pub fn shutdown(&self) -> Result<(), LspMessageError> {
        self.client.send_shutdown().expect_response(json!(null))?;

        self.client.send_exit();
        Ok(())
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.client.root = Some(root.clone());
        self.client.root = Some(root);
    }

    /// Opens a notebook document with the given cell contents.
    /// Each string in `cell_contents` becomes a separate code cell in the notebook.
    pub fn open_notebook(&self, file_name: &str, cell_contents: Vec<&str>) {
        let root = self.client.get_root_or_panic();
        let notebook_path = root.join(file_name);
        let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();

        let mut cells = Vec::new();
        let mut cell_text_documents = Vec::new();

        for (i, text) in cell_contents.iter().enumerate() {
            let cell_uri = self.cell_uri(file_name, &format!("cell{}", i + 1));
            cells.push(json!({
                "kind": 2,
                "document": cell_uri,
            }));
            cell_text_documents.push(json!({
                "uri": cell_uri,
                "languageId": "python",
                "version": 1,
                "text": *text
            }));
        }

        self.client
            .send_notification::<DidOpenNotebookDocument>(json!({
                "notebookDocument": {
                    "uri": notebook_uri,
                    "notebookType": "jupyter-notebook",
                    "version": 1,
                    "metadata": {
                        "language_info": {
                            "name": "python"
                        }
                    },
                    "cells": cells
                },
                "cellTextDocuments": cell_text_documents
            }));
    }

    pub fn close_notebook(&self, file_name: &str) {
        let root = self.client.get_root_or_panic();
        let notebook_path = root.join(file_name);
        let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();
        self.client
            .send_notification::<DidCloseNotebookDocument>(json!({
                "notebookDocument": { "uri": notebook_uri },
                "cellTextDocuments": [],
            }));
    }

    /// Updates a notebook document with the specified changes.
    /// This sends a notebookDocument/didChange notification with the change event.
    pub fn change_notebook(&self, file_name: &str, version: i32, change_event: serde_json::Value) {
        let root = self.client.get_root_or_panic();
        let notebook_path = root.join(file_name);
        let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();

        self.client
            .send_notification::<DidChangeNotebookDocument>(json!({
                "notebookDocument": {
                    "version": version,
                    "uri": notebook_uri,
                },
                "change": change_event
            }));
    }

    pub fn diagnostic_for_cell(
        &self,
        file: &str,
        cell: &str,
    ) -> ClientRequestHandle<'_, DocumentDiagnosticRequest> {
        self.client.send_request(json!({
        "textDocument": {
            "uri": self.cell_uri(file, cell)
        }}))
    }

    /// Returns the URI for a notebook cell
    pub fn cell_uri(&self, file_name: &str, cell_name: &str) -> Url {
        let root = self.client.get_root_or_panic();
        // Parse this as a file to preserve the C: prefix for windows
        let file_uri = Url::from_file_path(root.join(file_name)).unwrap();
        // Replace the scheme & add the cell name as a fragment
        // This is a bit awkward because the url library does not allow changing the scheme for file:// URLs
        Url::parse(&format!(
            "vscode-notebook-cell://{}#{}",
            file_uri.path(),
            cell_name
        ))
        .unwrap()
    }

    /// Sends a hover request for a notebook cell at the specified position
    pub fn hover_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, HoverRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    /// Sends a signature help request for a notebook cell at the specified position
    pub fn signature_help_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, SignatureHelpRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    /// Sends a definition request for a notebook cell at the specified position
    pub fn definition_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, GotoDefinition> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    /// Sends a references request for a notebook cell at the specified position
    pub fn references_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
        include_declaration: bool,
    ) -> ClientRequestHandle<'_, References> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri,
            },
            "position": {
                "line": line,
                "character": col
            },
            "context": {
                "includeDeclaration": include_declaration,
            },
        }))
    }

    pub fn completion_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
    ) -> ClientRequestHandle<'_, Completion> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri,
            },
            "position": {
                "line": line,
                "character": col
            }
        }))
    }

    /// Sends an inlay hint request for a notebook cell in the specified range
    pub fn inlay_hint_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) -> ClientRequestHandle<'_, InlayHintRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "range": {
                "start": {
                    "line": start_line,
                    "character": start_char
                },
                "end": {
                    "line": end_line,
                    "character": end_char
                }
            }
        }))
    }

    /// Sends a full semantic tokens request for a notebook cell
    pub fn semantic_tokens_cell(
        &self,
        file_name: &str,
        cell_name: &str,
    ) -> ClientRequestHandle<'_, SemanticTokensFullRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            }
        }))
    }

    /// Sends a ranged semantic tokens request for a notebook cell
    pub fn semantic_tokens_ranged_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) -> ClientRequestHandle<'_, SemanticTokensRangeRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "range": {
                "start": {
                    "line": start_line,
                    "character": start_char
                },
                "end": {
                    "line": end_line,
                    "character": end_char
                }
            }
        }))
    }

    /// Sends a code action request for a notebook cell at the specified range
    pub fn code_action_cell(
        &self,
        file_name: &str,
        cell_name: &str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) -> ClientRequestHandle<'_, CodeActionRequest> {
        let cell_uri = self.cell_uri(file_name, cell_name);
        self.client.send_request(json!({
            "textDocument": {
                "uri": cell_uri
            },
            "range": {
                "start": {
                    "line": start_line,
                    "character": start_char
                },
                "end": {
                    "line": end_line,
                    "character": end_char
                }
            },
            "context": {
                "diagnostics": []
            }
        }))
    }
}
