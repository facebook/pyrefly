/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This file contains a new implementation of the lsp_interaction test suite. Soon it will replace the old one.
use std::io;
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
use pretty_assertions::assert_eq;
use pyrefly_util::fs_anyhow::read_to_string;
use serde_json::Value;

use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::LspArgs;
use crate::commands::lsp::run_lsp;
use crate::test::util::init_test;
#[derive(Default)]
pub struct InitializeSettings {
    pub workspace_folders: Option<Vec<(String, Url)>>,
    pub configuration: bool,
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

    pub fn get_initialize_params(&self, settings: InitializeSettings) -> Value {
        let mut params = serde_json::json!({
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

        if let Some(folders) = settings.workspace_folders {
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
        if settings.configuration {
            params["capabilities"]["workspace"]["configuration"] = serde_json::json!(true);
        }

        params
    }

    fn next_request_id(&mut self) -> RequestId {
        let mut idx = self.request_idx.lock().unwrap();
        *idx += 1;
        RequestId::from(*idx)
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
            timeout: Duration::from_secs(25),
            root: None,
            request_idx,
        }
    }

    pub fn expect_message_helper(
        &self,
        message: Result<Message, RecvTimeoutError>,
        expected_message: Message,
    ) {
        match message {
            Ok(msg) => {
                eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());

                let assert_match = |expected: &str, actual: &str| {
                    assert_eq!(actual, expected, "Response mismatch");
                };

                let expected_str = serde_json::to_string(&expected_message).unwrap();
                let actual_str = serde_json::to_string(&msg).unwrap();

                assert_match(&expected_str, &actual_str);
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("Timeout waiting for response. Expected: {expected_message:?}");
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("Channel disconnected. Expected: {expected_message:?}");
            }
        }
    }

    pub fn expect_message(&self, expected_message: Message) {
        self.expect_message_helper(self.receiver.recv_timeout(self.timeout), expected_message);
    }

    pub fn expect_response(&self, expected_response: Response) {
        loop {
            match self.receiver.recv_timeout(self.timeout) {
                Ok(Message::Notification(notification)) => {
                    eprintln!("received notification, expecting response");
                    eprintln!(
                        "client<---server {}",
                        serde_json::to_string(&notification).unwrap()
                    );
                }
                Ok(Message::Request(request)) => {
                    eprintln!("received request, expecting response");
                    eprintln!(
                        "client<---server {}",
                        serde_json::to_string(&request).unwrap()
                    );
                }
                result => {
                    self.expect_message_helper(result, Message::Response(expected_response));
                    return;
                }
            }
        }
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
        loop {
            match self.receiver.recv_timeout(self.timeout) {
                Ok(Message::Notification(notification)) => {
                    eprintln!("received notification, expecting response");
                    eprintln!(
                        "client<---server {}",
                        serde_json::to_string(&notification).unwrap()
                    );
                }
                Ok(Message::Request(request)) => {
                    eprintln!("received request, expecting response");
                    eprintln!(
                        "client<---server {}",
                        serde_json::to_string(&request).unwrap()
                    );
                }
                Ok(Message::Response(response)) => {
                    eprintln!(
                        "client<---server {}",
                        serde_json::to_string(&response).unwrap()
                    );

                    if validator(&response) {
                        return;
                    } else {
                        panic!("Response validation failed: {description}. Response: {response:?}");
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("Timeout waiting for response. Expected: {description}");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("Channel disconnected. Expected: {description}");
                }
            }
        }
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
            run_lsp(connection, args)
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });

        server.server_thread = Some(thread_handle);

        let client = TestClient::new(language_client_receiver, request_idx.clone());

        Self { server, client }
    }

    pub fn initialize(&mut self, settings: InitializeSettings) {
        self.server
            .send_initialize(self.server.get_initialize_params(settings));
        self.client.expect_any_message();
        self.server.send_initialized();
    }

    pub fn shutdown(&self) {
        let shutdown_id = RequestId::from(999);
        self.server.send_shutdown(shutdown_id.clone());

        self.client.expect_message(Message::Response(Response {
            id: shutdown_id,
            result: Some(serde_json::json!(null)),
            error: None,
        }));

        self.server.send_exit();
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.server.root = Some(root.clone());
        self.client.root = Some(root);
    }
}
