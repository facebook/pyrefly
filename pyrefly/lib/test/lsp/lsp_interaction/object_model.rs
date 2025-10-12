/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This file contains a new implementation of the lsp_interaction test suite. Soon it will replace the old one.
use std::io;
use std::iter::once;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::thread::{self};
use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::bounded;
use lsp_server::Connection;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;
use lsp_types::notification::Exit;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use pyrefly_util::fs_anyhow::read_to_string;
use serde_json::Value;

use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::LspArgs;
use crate::commands::lsp::run_lsp;
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
}

pub struct TestServer {
    sender: crossbeam_channel::Sender<Message>,
    timeout: Duration,
    /// Handle to the spawned server thread
    server_thread: Option<JoinHandle<Result<(), io::Error>>>,
    root: Option<PathBuf>,
    /// Request ID for requests sent to the server
    request_idx: Arc<Mutex<i32>>,
}

impl TestServer {
    pub fn new(sender: crossbeam_channel::Sender<Message>, request_idx: Arc<Mutex<i32>>) -> Self {
        Self {
            sender,
            timeout: Duration::from_secs(25),
            server_thread: None,
            root: None,
            request_idx,
        }
    }

    pub fn expect_stop(&self) {
        let start = std::time::Instant::now();
        while let Some(thread) = &self.server_thread
            && !thread.is_finished()
        {
            if start.elapsed() > Duration::from_secs(10) {
                panic!("Server did not shutdown in time");
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    /// Send a message to this server
    pub fn send_message(&self, message: Message) {
        eprintln!(
            "client--->server {}",
            serde_json::to_string(&message).unwrap()
        );
        if let Err(err) = self.sender.send_timeout(message.clone(), self.timeout) {
            panic!("Failed to send message to language server: {err:?}");
        }
    }
    pub fn send_initialize(&mut self, params: Value) {
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "initialize".to_owned(),
            params,
        }))
    }

    pub fn send_initialized(&self) {
        self.send_message(Message::Notification(Notification {
            method: "initialized".to_owned(),
            params: serde_json::json!({}),
        }));
    }

    pub fn send_shutdown(&self, id: RequestId) {
        self.send_message(Message::Request(Request {
            id,
            method: lsp_types::request::Shutdown::METHOD.to_owned(),
            params: serde_json::json!(null),
        }));
    }

    pub fn send_exit(&self) {
        self.send_message(Message::Notification(Notification {
            method: Exit::METHOD.to_owned(),
            params: serde_json::json!(null),
        }));
    }

    pub fn type_definition(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "textDocument/typeDefinition".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                },
                "position": {
                    "line": line,
                    "character": col,
                },
            }),
        }));
    }

    pub fn definition(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "textDocument/definition".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                },
                "position": {
                    "line": line,
                    "character": col,
                },
            }),
        }));
    }

    pub fn did_open(&self, file: &'static str) {
        let path = self.get_root_or_panic().join(file);
        self.send_message(Message::Notification(Notification {
            method: "textDocument/didOpen".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 1,
                    "text": read_to_string(&path).unwrap(),
                },
            }),
        }));
    }

    pub fn did_change_configuration(&self) {
        self.send_message(Message::Notification(Notification {
            method: lsp_types::notification::DidChangeConfiguration::METHOD.to_owned(),
            params: serde_json::json!({"settings": {}}),
        }));
    }

    pub fn diagnostic(&mut self, file: &'static str) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "textDocument/diagnostic".to_owned(),
            params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            }}),
        }));
    }

    pub fn hover(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "textDocument/hover".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string()
                },
                "position": {
                    "line": line,
                    "character": col
                }
            }),
        }));
    }

    pub fn provide_type(&mut self, file: &'static str, line: u32, col: u32) {
        let path = self.get_root_or_panic().join(file);
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "types/provide-type".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string()
                },
                "positions": [{
                    "line": line,
                    "character": col
                }]
            }),
        }));
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
        self.send_message(Message::Request(Request {
            id,
            method: "textDocument/inlayHint".to_owned(),
            params: serde_json::json!({
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
        }));
    }

    pub fn send_configuration_response(&self, id: i32, result: serde_json::Value) {
        self.send_message(Message::Response(Response {
            id: RequestId::from(id),
            result: Some(result),
            error: None,
        }));
    }

    pub fn get_initialize_params(&self, settings: &InitializeSettings) -> Value {
        let mut params: Value = serde_json::json!({
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
            params["capabilities"]["workspace"]["workspaceFolders"] = serde_json::json!(true);
            params["workspaceFolders"] = serde_json::json!(
                folders
                    .iter()
                    .map(|(name, path)| serde_json::json!({"name": name, "uri": path.to_string()}))
                    .collect::<Vec<_>>()
            );
        }
        if settings.file_watch {
            params["capabilities"]["workspace"]["didChangeWatchedFiles"] =
                serde_json::json!({"dynamicRegistration": true});
        }
        if settings.configuration.is_some() {
            params["capabilities"]["workspace"]["configuration"] = serde_json::json!(true);
        }

        params
    }

    fn next_request_id(&mut self) -> RequestId {
        let mut idx = self.request_idx.lock().unwrap();
        *idx += 1;
        RequestId::from(*idx)
    }

    pub fn current_request_id(&self) -> RequestId {
        RequestId::from(*self.request_idx.lock().unwrap())
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
    request_idx: Arc<Mutex<i32>>,
}

impl TestClient {
    pub fn new(
        receiver: crossbeam_channel::Receiver<Message>,
        request_idx: Arc<Mutex<i32>>,
    ) -> Self {
        Self {
            receiver,
            timeout: Duration::from_secs(50),
            root: None,
            request_idx,
        }
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

    pub fn expect_message(&self, expected_message: Message) {
        let expected_str = serde_json::to_string(&expected_message).unwrap();
        self.expect_message_helper(
            |msg| {
                let actual_str = serde_json::to_string(msg).unwrap();
                assert_eq!(&expected_str, &actual_str, "Response mismatch");
                ValidationResult::Pass
            },
            &format!("Expected message: {expected_message:?}"),
        );
    }

    pub fn expect_request(&self, expected_request: Request) {
        let expected_str =
            serde_json::to_string(&Message::Request(expected_request.clone())).unwrap();
        self.expect_message_helper(
            |msg| match msg {
                Message::Notification(_) | Message::Response(_) => ValidationResult::Skip,
                Message::Request(_) => {
                    let actual_str = serde_json::to_string(msg).unwrap();
                    assert_eq!(&expected_str, &actual_str, "Request mismatch");
                    ValidationResult::Pass
                }
            },
            &format!("Expected Request: {expected_request:?}"),
        );
    }

    pub fn expect_response(&self, expected_response: Response) {
        let expected_str =
            serde_json::to_string(&Message::Response(expected_response.clone())).unwrap();
        self.expect_message_helper(
            |msg| match msg {
                Message::Notification(_) | Message::Request(_) => ValidationResult::Skip,
                Message::Response(_) => {
                    let actual_str = serde_json::to_string(msg).unwrap();
                    assert_eq!(&expected_str, &actual_str, "Response mismatch");
                    ValidationResult::Pass
                }
            },
            &format!("Expected response: {expected_response:?}"),
        );
    }

    /// Wait until we get a publishDiagnostics notification with the correct number of errors
    pub fn expect_publish_diagnostics_error_count(&self, path: PathBuf, count: usize) {
        self.expect_message_helper(
            |msg| {
                match msg {
                    Message::Notification(Notification { method, params}) if method == "textDocument/publishDiagnostics" => {
                        // Check if this notification is for the expected file
                        if let Some(uri) = params.get("uri")
                            && let Some(uri_str) = uri.as_str()
                            && let (Ok(expected_url), Ok(actual_url)) = (Url::parse(Url::from_file_path(&path).unwrap().as_ref()), Url::parse(uri_str))
                            && let (Ok(expected_path), Ok(actual_path)) = (expected_url.to_file_path(), actual_url.to_file_path()) {
                                // Canonicalize both paths for comparison to handle symlinks and normalize case
                                // This is very relevant for publish diagnostics, where the LS might send a notification for
                                // a file that does not exactly match the file_open message.
                                // This is very relevant for publish diagnostics, where the LS might send a notification for
                                // a file that does not exactly match the file_open message.
                                // This is very relevant for publish diagnostics, where the LS might send a notification for
                                // a file that does not exactly match the file_open message.
                                let expected_canonical = expected_path.canonicalize().unwrap_or(expected_path);
                                let actual_canonical = actual_path.canonicalize().unwrap_or(actual_path);

                                if expected_canonical == actual_canonical
                                    && let Some(diagnostics) = params.get("diagnostics")
                                    && let Some(diagnostics_array) = diagnostics.as_array() {
                                        let actual_count = diagnostics_array.len();
                                        if actual_count == count {
                                            return ValidationResult::Pass;
                                        } else {
                                            // If the counts do not match, we continue waiting
                                            return ValidationResult::Skip;
                                        }
                                    } else if expected_canonical == actual_canonical {
                                        panic!("publishDiagnostics notification malformed: missing or invalid 'diagnostics' field");
                                    }
                            }
                        ValidationResult::Skip
                    }
                    _ => ValidationResult::Skip
                }
            },
            &format!("publishDiagnostics notification with {count} errors for file: {}", path.display()),
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
        self.expect_response(Response {
            id: RequestId::from(*self.request_idx.lock().unwrap()),
            result: Some(serde_json::json!(
            {
                "uri": Url::from_file_path(file).unwrap().to_string(),
                "range": {
                    "start": {"line": line_start, "character": char_start},
                    "end": {"line": line_end, "character": char_end}
                },
                })),
            error: None,
        })
    }

    pub fn expect_definition_response_from_root(
        &self,
        file: &'static str,
        line_start: u32,
        char_start: u32,
        line_end: u32,
        char_end: u32,
    ) {
        self.expect_response(Response {
            id: RequestId::from(*self.request_idx.lock().unwrap()),
            result: Some(serde_json::json!(
            {
                "uri": Url::from_file_path(self.get_root_or_panic().join(file)).unwrap().to_string(),
                "range": {
                    "start": {"line": line_start, "character": char_start},
                    "end": {"line": line_end, "character": char_end}
                },
                })),
            error: None,
        })
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
            params: serde_json::json!(ConfigurationParams { items }),
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
        init_test();

        let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
        let (language_server_sender, language_server_receiver) = bounded::<Message>(0);

        let args = LspArgs {
            indexing_mode: IndexingMode::None,
            workspace_indexing_limit: 0,
        };
        let connection = Connection {
            sender: language_client_sender,
            receiver: language_server_receiver,
        };

        let connection = Arc::new(connection);
        let args = args.clone();

        let request_idx = Arc::new(Mutex::new(0));

        let mut server = TestServer::new(language_server_sender, request_idx.clone());

        // Spawn the server thread and store its handle
        let thread_handle = thread::spawn(move || {
            run_lsp(connection, args, "pyrefly-lsp-test-version")
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });

        server.server_thread = Some(thread_handle);

        let client = TestClient::new(language_client_receiver, request_idx.clone());

        Self { server, client }
    }

    pub fn initialize(&mut self, settings: InitializeSettings) {
        self.server
            .send_initialize(self.server.get_initialize_params(&settings));
        self.client.expect_any_message();
        self.server.send_initialized();
        if let Some(settings) = settings.configuration {
            self.client.expect_any_message();
            self.server.send_message(Message::Response(Response {
                id: RequestId::from(1),
                result: settings,
                error: None,
            }));
        }
    }

    pub fn shutdown(&self) {
        let shutdown_id = RequestId::from(999);
        self.server.send_shutdown(shutdown_id.clone());

        self.client.expect_response(Response {
            id: shutdown_id,
            result: Some(serde_json::json!(null)),
            error: None,
        });

        self.server.send_exit();
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.server.root = Some(root.clone());
        self.client.root = Some(root);
    }
}
