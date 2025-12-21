/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This file contains a new implementation of the tsp_interaction test suite that follows
/// the same pattern as the LSP object model tests.
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
use pyrefly_util::telemetry::NoTelemetry;
use serde_json::Value;

use crate::commands::lsp::IndexingMode;
use crate::commands::tsp::TspArgs;
use crate::commands::tsp::run_tsp;
use crate::test::util::init_test;

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

pub struct TestTspServer {
    sender: crossbeam_channel::Sender<Message>,
    timeout: Duration,
    /// Handle to the spawned server thread
    server_thread: Option<JoinHandle<Result<(), io::Error>>>,
    root: Option<PathBuf>,
    /// Request ID for requests sent to the server
    request_idx: Arc<Mutex<i32>>,
}

impl TestTspServer {
    pub fn new(sender: crossbeam_channel::Sender<Message>, request_idx: Arc<Mutex<i32>>) -> Self {
        Self {
            sender,
            timeout: Duration::from_secs(25),
            server_thread: None,
            root: None,
            request_idx,
        }
    }

    /// Send a message to this server
    pub fn send_message(&self, message: Message) {
        eprintln!(
            "client--->server {}",
            serde_json::to_string(&message).unwrap()
        );
        if let Err(err) = self.sender.send_timeout(message.clone(), self.timeout) {
            panic!("Failed to send message to TSP server: {err:?}");
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

    pub fn get_supported_protocol_version(&mut self) {
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "typeServer/getSupportedProtocolVersion".to_owned(),
            params: serde_json::json!(null),
        }));
    }

    pub fn get_snapshot(&mut self) {
        let id = self.next_request_id();
        self.send_message(Message::Request(Request {
            id,
            method: "typeServer/getSnapshot".to_owned(),
            params: serde_json::json!(null),
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

    pub fn did_change(&self, file: &'static str, content: &str, version: i32) {
        let path = self.get_root_or_panic().join(file);
        self.send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                    "version": version
                },
                "contentChanges": [{
                    "text": content
                }]
            }),
        }));
    }

    pub fn did_change_watched_files(&self, file: &'static str, change_type: &str) {
        let path = self.get_root_or_panic().join(file);
        let file_change_type = match change_type {
            "created" => 1, // FileChangeType::CREATED
            "changed" => 2, // FileChangeType::CHANGED
            "deleted" => 3, // FileChangeType::DELETED
            _ => 2,         // Default to changed
        };
        self.send_message(Message::Notification(Notification {
            method: "workspace/didChangeWatchedFiles".to_owned(),
            params: serde_json::json!({
                "changes": [{
                    "uri": Url::from_file_path(&path).unwrap().to_string(),
                    "type": file_change_type
                }]
            }),
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

    fn get_root_or_panic(&self) -> PathBuf {
        self.root
            .clone()
            .expect("Root not set, please call set_root")
    }
}

pub struct TestTspClient {
    receiver: crossbeam_channel::Receiver<Message>,
    timeout: Duration,
    root: Option<PathBuf>,
}

impl TestTspClient {
    pub fn new(receiver: crossbeam_channel::Receiver<Message>) -> Self {
        Self {
            receiver,
            timeout: Duration::from_secs(25),
            root: None,
        }
    }

    pub fn expect_message_helper<F>(&self, expected_message: Message, should_skip: F)
    where
        F: Fn(&Message) -> bool,
    {
        loop {
            match self.receiver.recv_timeout(self.timeout) {
                Ok(msg) => {
                    eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());

                    if should_skip(&msg) {
                        continue;
                    }

                    let expected_str = serde_json::to_string(&expected_message).unwrap();
                    let actual_str = serde_json::to_string(&msg).unwrap();

                    assert_eq!(&expected_str, &actual_str, "Response mismatch");
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("Timeout waiting for response. Expected: {expected_message:?}");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("Channel disconnected. Expected: {expected_message:?}");
                }
            }
        }
    }

    pub fn expect_response(&self, expected_response: Response) {
        self.expect_message_helper(Message::Response(expected_response), |msg| {
            matches!(msg, Message::Notification(_) | Message::Request(_))
        });
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

    #[expect(dead_code)]
    pub fn receive_any_message(&self) -> Message {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(msg) => {
                eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                msg
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("Timeout waiting for response");
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("Channel disconnected");
            }
        }
    }
}

pub struct TspInteraction {
    pub server: TestTspServer,
    pub client: TestTspClient,
}

impl TspInteraction {
    pub fn new() -> Self {
        init_test();

        let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
        let (language_server_sender, language_server_receiver) = bounded::<Message>(0);

        let args = TspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            workspace_indexing_limit: 0,
        };
        let connection = Connection {
            sender: language_client_sender,
            receiver: language_server_receiver,
        };

        let args = args.clone();

        let request_idx = Arc::new(Mutex::new(0));

        let mut server = TestTspServer::new(language_server_sender, request_idx.clone());

        // Spawn the server thread and store its handle
        let thread_handle = thread::spawn(move || {
            run_tsp(connection, args, &NoTelemetry)
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });

        server.server_thread = Some(thread_handle);

        let client = TestTspClient::new(language_client_receiver);

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
