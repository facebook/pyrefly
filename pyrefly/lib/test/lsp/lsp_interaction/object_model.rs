/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This file contains a new implementation of the lsp_interaction test suite. Soon it will replace the old one.
use std::iter::once;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use std::thread::{self};
use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::bounded;
use itertools::Itertools;
use lsp_server::Connection;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::PublishDiagnosticsParams;
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
use lsp_types::request::Completion;
use lsp_types::request::DocumentDiagnosticRequest;
use lsp_types::request::GotoDefinition;
use lsp_types::request::GotoImplementation;
use lsp_types::request::GotoTypeDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::Initialize;
use lsp_types::request::InlayHintRequest;
use lsp_types::request::References;
use lsp_types::request::Request as _;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::request::SemanticTokensRangeRequest;
use lsp_types::request::Shutdown;
use lsp_types::request::SignatureHelpRequest;
use lsp_types::request::WillRenameFiles;
use lsp_types::request::WorkspaceConfiguration;
use pretty_assertions::assert_eq;
use pyrefly_util::fs_anyhow::read_to_string;
use pyrefly_util::lock::Condvar;
use pyrefly_util::lock::Mutex;
use serde_json::Value;
use serde_json::json;

use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::LspArgs;
use crate::commands::lsp::run_lsp;
use crate::lsp::wasm::provide_type::ProvideType;
use crate::test::util::init_test;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Skip this message and continue waiting
    Skip,
    /// Message succeeds validation
    Pass,
    /// Message fails validation
    Fail,
}
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
}

pub struct FinishHandle {
    finished: Mutex<bool>,
    cvar: Condvar,
}

impl FinishHandle {
    pub fn new() -> Self {
        Self {
            finished: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    pub fn notify_finished(&self) {
        let mut finished = self.finished.lock();
        *finished = true;
        self.cvar.notify_one();
    }

    pub fn wait_for_finish(&self, timeout: Duration) -> bool {
        let finished = self.finished.lock();
        *self.cvar.wait_timeout(finished, timeout).0
    }
}

pub struct TestServer {
    sender: Option<crossbeam_channel::Sender<Message>>,
    timeout: Duration,
    finish_handle: Arc<FinishHandle>,
    root: Option<PathBuf>,
    /// Request ID for requests sent to the server
    request_idx: Arc<AtomicI32>,
}

impl TestServer {
    pub fn new(
        sender: crossbeam_channel::Sender<Message>,
        finish_handle: Arc<FinishHandle>,
        request_idx: Arc<AtomicI32>,
    ) -> Self {
        Self {
            sender: Some(sender),
            timeout: Duration::from_secs(25),
            finish_handle,
            root: None,
            request_idx,
        }
    }

    pub fn drop_connection(&mut self) {
        // Take and drop the sender to close the connection
        drop(std::mem::take(&mut self.sender))
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
        self.sender
            .as_ref()
            .unwrap()
            .send_timeout(message, self.timeout)
    }

    /// Send a message to this server
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
        id: RequestId,
        params: serde_json::Value,
    ) {
        // Ensure the passed value can be parsed as the desired request params
        let params = serde_json::from_value::<R::Params>(params).unwrap();
        self.send_message(Message::Request(Request {
            id,
            method: R::METHOD.to_owned(),
            params: serde_json::to_value(params).unwrap(),
        }));
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

    pub fn send_initialize(&mut self, params: Value) {
        let id = self.next_request_id();
        self.send_request::<Initialize>(id, params);
    }

    pub fn send_initialized(&self) {
        self.send_notification::<Initialized>(json!({}));
    }

    pub fn send_shutdown(&self, id: RequestId) {
        self.send_request::<Shutdown>(id, json!(null));
    }

    pub fn send_exit(&self) {
        self.send_notification::<Exit>(json!(null));
    }

    pub fn type_definition(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<GotoTypeDefinition>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                },
                "position": {
                    "line": line,
                    "character": col,
                },
            }),
        );
    }

    pub fn definition(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<GotoDefinition>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                },
                "position": {
                    "line": line,
                    "character": col,
                },
            }),
        );
    }

    pub fn implementation(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<GotoImplementation>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                },
                "position": {
                    "line": line,
                    "character": col,
                },
            }),
        );
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

    pub fn completion(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<Completion>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string()
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    pub fn diagnostic(&mut self, file: &'static str) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<DocumentDiagnosticRequest>(
            id,
            json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            }}),
        );
    }

    pub fn hover(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<HoverRequest>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string()
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    pub fn provide_type(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<ProvideType>(
            id,
            json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string()
                },
                "positions": [{
                    "line": line,
                    "character": col
                }]
            }),
        );
    }

    pub fn references(&mut self, file: &str, line: u32, col: u32, include_declaration: bool) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<References>(
            id,
            json!({
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
            }),
        );
    }

    pub fn inlay_hint(
        &mut self,
        file: &'static str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_request::<InlayHintRequest>(
            id,
            json!({
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
            }),
        );
    }

    pub fn send_configuration_response(&self, id: i32, result: serde_json::Value) {
        self.send_response::<WorkspaceConfiguration>(RequestId::from(id), result);
    }

    pub fn will_rename_files(&mut self, old_file: &'static str, new_file: &'static str) {
        let root = self.get_root_or_panic();
        let old_path = root.join(old_file);
        let new_path = root.join(new_file);
        let id = self.next_request_id();
        self.send_request::<WillRenameFiles>(
            id,
            json!({
                "files": [{
                    "oldUri": Url::from_file_path(&old_path).unwrap().to_string(),
                    "newUri": Url::from_file_path(&new_path).unwrap().to_string()
                }]
            }),
        );
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
                        "versionSupport": false,
                        "tagSupport": {
                            "valueSet": [1, 2],
                        },
                        "codeDescriptionSupport": true,
                        "dataSupport": true,
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

        // Merge custom capabilities if provided
        if let Some(custom_capabilities) = &settings.capabilities {
            Self::merge_json(&mut params["capabilities"], custom_capabilities);
        }

        params
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

    fn next_request_id(&mut self) -> RequestId {
        let idx = self.request_idx.fetch_add(1, Ordering::SeqCst);
        RequestId::from(idx + 1)
    }

    pub fn current_request_id(&self) -> RequestId {
        let idx = self.request_idx.load(Ordering::Acquire);
        RequestId::from(idx)
    }

    fn get_root_or_panic(&self) -> PathBuf {
        self.root
            .clone()
            .expect("Root not set, please call set_root")
    }
}

pub struct TestClient {
    receiver: crossbeam_channel::Receiver<Message>,
    timeout: Duration,
    root: Option<PathBuf>,
    request_idx: Arc<AtomicI32>,
}

impl TestClient {
    pub fn new(
        receiver: crossbeam_channel::Receiver<Message>,
        request_idx: Arc<AtomicI32>,
    ) -> Self {
        Self {
            receiver,
            timeout: Duration::from_secs(50),
            root: None,
            request_idx,
        }
    }

    fn current_request_id(&self) -> RequestId {
        let idx = self.request_idx.load(Ordering::Acquire);
        RequestId::from(idx)
    }

    pub fn drop_connection(self) {
        drop(self.receiver);
    }

    pub fn expect_message_helper<F>(&self, validator: F, description: &str)
    where
        F: Fn(&Message) -> ValidationResult,
    {
        loop {
            match self.receiver.recv_timeout(self.timeout) {
                Ok(msg) => {
                    eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                    match validator(&msg) {
                        ValidationResult::Skip => continue,
                        ValidationResult::Pass => return,
                        ValidationResult::Fail => {
                            panic!("Message validation failed: {description}. Message: {msg:?}");
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("Timeout waiting for message: {description}");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("Channel disconnected while waiting for message: {description}");
                }
            }
        }
    }

    pub fn expect_message<T>(
        &self,
        description: &str,
        matcher: impl Fn(Message) -> Option<T>,
    ) -> T {
        loop {
            match self.receiver.recv_timeout(self.timeout) {
                Ok(msg) => {
                    eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                    if let Some(actual) = matcher(msg) {
                        return actual;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("Timeout waiting for message: {description}");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("Channel disconnected while waiting for message: {description}");
                }
            }
        }
    }

    pub fn expect_request<R: lsp_types::request::Request>(&self, id: RequestId, expected: Value) {
        // Validate that expected can be parsed as R::Params
        let expected: R::Params = serde_json::from_value(expected.clone()).unwrap();
        let actual: R::Params =
            self.expect_message(&format!("Request {} with id={}", R::METHOD, id), |msg| {
                if let Message::Request(x) = msg
                    && x.id == id
                {
                    assert_eq!(x.method, R::METHOD);
                    Some(serde_json::from_value(x.params.clone()).unwrap())
                } else {
                    None
                }
            });
        assert_eq!(json!(expected), json!(actual));
    }

    pub fn expect_response<R: lsp_types::request::Request>(&self, id: RequestId, expected: Value) {
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
            });
        assert_eq!(json!(expected), json!(actual));
    }

    pub fn expect_response_error(&self, id: RequestId, expected: Value) {
        let expected: ResponseError = serde_json::from_value(expected).unwrap();
        let actual = self.expect_message(&format!("Response error id={}", id), |msg| {
            if let Message::Response(x) = msg
                && x.id == id
            {
                Some(x.error.unwrap())
            } else {
                None
            }
        });
        assert_eq!(json!(expected), json!(actual));
    }

    /// Wait for a publishDiagnostics notification, then check if it has the correct path and count
    pub fn expect_publish_diagnostics_error_count(&self, path: PathBuf, count: usize) {
        let params: PublishDiagnosticsParams = self.expect_message(
            &format!(
                "publishDiagnostics notification with {count} errors for file: {}",
                path.display()
            ),
            |msg| {
                if let Message::Notification(x) = msg
                    && x.method == PublishDiagnostics::METHOD
                {
                    Some(serde_json::from_value(x.params).unwrap())
                } else {
                    None
                }
            },
        );
        assert_eq!(params.uri.to_file_path().unwrap(), path);
        assert_eq!(params.diagnostics.len(), count);
    }

    pub fn expect_publish_diagnostics_exact_uri(&self, expected_uri: &str, count: usize) {
        self.expect_message_helper(
            |msg| match msg {
                Message::Notification(Notification { method, params })
                    if method == "textDocument/publishDiagnostics" =>
                {
                    if params.get("uri").and_then(|v| v.as_str()) == Some(expected_uri) {
                        if let Some(diagnostics) = params.get("diagnostics")
                            && let Some(diagnostics_array) = diagnostics.as_array()
                        {
                            if diagnostics_array.len() == count {
                                return ValidationResult::Pass;
                            } else {
                                return ValidationResult::Skip;
                            }
                        } else {
                            panic!("publishDiagnostics notification malformed: missing or invalid 'diagnostics' field");
                        }
                    }
                    ValidationResult::Skip
                }
                _ => ValidationResult::Skip,
            },
            &format!(
                "publishDiagnostics notification with uri {expected_uri} containing {count} errors"
            ),
        );
    }

    pub fn expect_definition_response_absolute(
        &self,
        file: String,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) {
        self.expect_response::<GotoDefinition>(
            self.current_request_id(),
            json!(
            {
                "uri": Url::from_file_path(file).unwrap().to_string(),
                "range": {
                    "start": {"line": line_start, "character": char_start},
                    "end": {"line": line_end, "character": char_end}
                },
                }),
        )
    }

    pub fn expect_definition_response_from_root(
        &self,
        file: &'static str,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) {
        self.expect_response::<GotoDefinition>(
            self.current_request_id(),
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
        implementations: Vec<(&'static str, u32, u32, u32, u32)>,
    ) {
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

        self.expect_response::<GotoImplementation>(self.current_request_id(), json!(locations))
    }

    pub fn expect_response_with<F>(&self, validator: F, description: &str)
    where
        F: Fn(&Response) -> bool,
    {
        self.expect_message_helper(
            |msg| match msg {
                Message::Notification(_) | Message::Request(_) => ValidationResult::Skip,
                Message::Response(response) => {
                    if validator(response) {
                        ValidationResult::Pass
                    } else {
                        ValidationResult::Fail
                    }
                }
            },
            description,
        );
    }

    pub fn expect_response_with_item(&self, item: Value, description: &str) {
        self.expect_response_with(
            |response| {
                if response.id != RequestId::from(2) {
                    return false;
                }
                if let Some(result) = &response.result
                    && let Some(items) = result.get("items")
                    && let Some(items_array) = items.as_array()
                {
                    return items_array.iter().contains(&item);
                }
                false
            },
            description,
        );
    }

    pub fn expect_any_message(&self) {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(msg) => {
                eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("Timeout waiting for response");
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("Channel disconnected");
            }
        }
    }

    pub fn expect_configuration_request(&self, id: i32, scope_uris: Option<Vec<&Url>>) {
        use lsp_types::ConfigurationItem;
        use lsp_types::ConfigurationParams;
        use lsp_types::request::WorkspaceConfiguration;

        let items = if let Some(uris) = scope_uris {
            uris.into_iter()
                .map(|uri| ConfigurationItem {
                    scope_uri: Some(uri.clone()),
                    section: Some("python".to_owned()),
                })
                .chain(once(ConfigurationItem {
                    scope_uri: None,
                    section: Some("python".to_owned()),
                }))
                .collect::<Vec<_>>()
        } else {
            Vec::from([ConfigurationItem {
                scope_uri: None,
                section: Some("python".to_owned()),
            }])
        };

        let expected_msg = Message::Request(Request {
            id: RequestId::from(id),
            method: WorkspaceConfiguration::METHOD.to_owned(),
            params: json!(ConfigurationParams { items }),
        });
        let expected_str = serde_json::to_string(&expected_msg).unwrap();
        self.expect_message_helper(
            |msg| match msg {
                Message::Notification(_) => ValidationResult::Skip,
                _ => {
                    let actual_str = serde_json::to_string(msg).unwrap();
                    assert_eq!(&expected_str, &actual_str, "Configuration request mismatch");
                    ValidationResult::Pass
                }
            },
            &format!("Expected configuration request: {expected_msg:?}"),
        );
    }

    /// Expect a file watcher registration request.
    /// Validates that the request is specifically registering the file watcher (ID: "FILEWATCHER").
    pub fn expect_file_watcher_register(&self) {
        self.expect_message_helper(
            |msg| match msg {
                Message::Request(req)
                    if req.method == "client/registerCapability"
                        && req
                            .params
                            .get("registrations")
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter().any(|reg| {
                                    reg.get("id").and_then(|id| id.as_str()) == Some("FILEWATCHER")
                                })
                            })
                            .unwrap_or(false) =>
                {
                    ValidationResult::Pass
                }
                Message::Notification(_) => ValidationResult::Skip,
                _ => ValidationResult::Fail,
            },
            "Expected file watcher registerCapability",
        );
    }

    /// Expect a file watcher unregistration request.
    /// Validates that the request is specifically unregistering the file watcher (ID: "FILEWATCHER").
    pub fn expect_file_watcher_unregister(&self) {
        self.expect_message_helper(
            |msg| match msg {
                Message::Request(req)
                    if req.method == "client/unregisterCapability"
                        && req
                            .params
                            .get("unregisterations")
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter().any(|reg| {
                                    reg.get("id").and_then(|id| id.as_str()) == Some("FILEWATCHER")
                                })
                            })
                            .unwrap_or(false) =>
                {
                    ValidationResult::Pass
                }
                Message::Notification(_) => ValidationResult::Skip,
                _ => ValidationResult::Fail,
            },
            "Expected file watcher unregisterCapability",
        );
    }

    fn get_root_or_panic(&self) -> PathBuf {
        self.root
            .clone()
            .expect("Root not set, please call set_root")
    }
}

pub struct LspInteraction {
    pub server: TestServer,
    pub client: TestClient,
}

impl LspInteraction {
    pub fn new() -> Self {
        Self::new_with_indexing_mode(IndexingMode::None)
    }

    pub fn new_with_indexing_mode(indexing_mode: IndexingMode) -> Self {
        init_test();

        let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
        let (language_server_sender, language_server_receiver) = bounded::<Message>(0);

        let args = LspArgs {
            indexing_mode,
            workspace_indexing_limit: 50,
        };
        let connection = Connection {
            sender: language_client_sender,
            receiver: language_server_receiver,
        };

        let connection = Arc::new(connection);
        let args = args.clone();

        let finish_handle = Arc::new(FinishHandle::new());
        let finish_server = finish_handle.clone();

        // Spawn the server thread notify when finished
        thread::spawn(move || {
            let _ = run_lsp(connection, args, "pyrefly-lsp-test-version");
            finish_server.notify_finished();
        });

        let request_idx = Arc::new(AtomicI32::new(0));
        let server = TestServer::new(language_server_sender, finish_handle, request_idx.clone());
        let client = TestClient::new(language_client_receiver, request_idx);

        Self { server, client }
    }

    pub fn initialize(&mut self, settings: InitializeSettings) {
        self.server
            .send_initialize(self.server.get_initialize_params(&settings));
        self.client.expect_any_message();
        self.server.send_initialized();
        if let Some(settings) = settings.configuration {
            self.client.expect_any_message();
            self.server.send_response::<WorkspaceConfiguration>(
                RequestId::from(1),
                settings.unwrap_or(json!([])),
            );
        }
    }

    pub fn shutdown(&self) {
        let shutdown_id = RequestId::from(999);
        self.server.send_shutdown(shutdown_id.clone());

        self.client
            .expect_response::<Shutdown>(shutdown_id, json!(null));

        self.server.send_exit();
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.server.root = Some(root.clone());
        self.client.root = Some(root);
    }

    /// Opens a notebook document with the given cell contents.
    /// Each string in `cell_contents` becomes a separate code cell in the notebook.
    pub fn open_notebook(&mut self, file_name: &str, cell_contents: Vec<&str>) {
        let root = self.server.get_root_or_panic();
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

        self.server
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

    pub fn close_notebook(&mut self, file_name: &str) {
        let root = self.server.get_root_or_panic();
        let notebook_path = root.join(file_name);
        let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();
        self.server
            .send_notification::<DidCloseNotebookDocument>(json!({
                "notebookDocument": { "uri": notebook_uri },
                "cellTextDocuments": [],
            }));
    }

    /// Updates a notebook document with the specified changes.
    /// This sends a notebookDocument/didChange notification with the change event.
    pub fn change_notebook(
        &mut self,
        file_name: &str,
        version: i32,
        change_event: serde_json::Value,
    ) {
        let root = self.server.get_root_or_panic();
        let notebook_path = root.join(file_name);
        let notebook_uri = Url::from_file_path(&notebook_path).unwrap().to_string();

        self.server
            .send_notification::<DidChangeNotebookDocument>(json!({
                "notebookDocument": {
                    "version": version,
                    "uri": notebook_uri,
                },
                "change": change_event
            }));
    }

    pub fn diagnostic_for_cell(&mut self, file: &str, cell: &str) {
        let id = self.server.next_request_id();
        self.server.send_request::<DocumentDiagnosticRequest>(
            id,
            json!({
            "textDocument": {
                "uri": self.cell_uri(file, cell)
            }}),
        );
    }

    /// Returns the URI for a notebook cell
    pub fn cell_uri(&self, file_name: &str, cell_name: &str) -> String {
        let root = self.server.get_root_or_panic();
        // Parse this as a file to preserve the C: prefix for windows
        let cell_uri =
            Url::from_file_path(format!("{}", root.join(file_name).to_string_lossy())).unwrap();
        // Replace the scheme & add the cell name as a fragment
        format!(
            "{}#{}",
            cell_uri
                .to_string()
                .replace("file://", "vscode-notebook-cell://"),
            cell_name
        )
    }

    /// Sends a hover request for a notebook cell at the specified position
    pub fn hover_cell(&mut self, file_name: &str, cell_name: &str, line: u32, col: u32) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<HoverRequest>(
            id,
            json!({
                "textDocument": {
                    "uri": cell_uri
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    /// Sends a signature help request for a notebook cell at the specified position
    pub fn signature_help_cell(&mut self, file_name: &str, cell_name: &str, line: u32, col: u32) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<SignatureHelpRequest>(
            id,
            json!({
                "textDocument": {
                    "uri": cell_uri
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    /// Sends a definition request for a notebook cell at the specified position
    pub fn definition_cell(&mut self, file_name: &str, cell_name: &str, line: u32, col: u32) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<GotoDefinition>(
            id,
            json!({
                "textDocument": {
                    "uri": cell_uri
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    /// Sends a references request for a notebook cell at the specified position
    pub fn references_cell(
        &mut self,
        file_name: &str,
        cell_name: &str,
        line: u32,
        col: u32,
        include_declaration: bool,
    ) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<References>(
            id,
            json!({
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
            }),
        );
    }

    pub fn completion_cell(&mut self, file_name: &str, cell_name: &str, line: u32, col: u32) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<Completion>(
            id,
            json!({
                "textDocument": {
                    "uri": cell_uri,
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        );
    }

    /// Sends an inlay hint request for a notebook cell in the specified range
    pub fn inlay_hint_cell(
        &mut self,
        file_name: &str,
        cell_name: &str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<InlayHintRequest>(
            id,
            json!({
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
            }),
        );
    }

    /// Sends a full semantic tokens request for a notebook cell
    pub fn semantic_tokens_cell(&mut self, file_name: &str, cell_name: &str) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<SemanticTokensFullRequest>(
            id,
            json!({
                "textDocument": {
                    "uri": cell_uri
                }
            }),
        );
    }

    /// Sends a ranged semantic tokens request for a notebook cell
    pub fn semantic_tokens_ranged_cell(
        &mut self,
        file_name: &str,
        cell_name: &str,
        start_line: u32,
        start_char: u32,
        end_line: u32,
        end_char: u32,
    ) {
        let cell_uri = self.cell_uri(file_name, cell_name);
        let id = self.server.next_request_id();
        self.server.send_request::<SemanticTokensRangeRequest>(
            id,
            json!({
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
            }),
        );
    }
}
