/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
use core::panic;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::bounded;
use lsp_server::Connection;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::CodeActionKind;
use lsp_types::CodeActionOptions;
use lsp_types::CodeActionProviderCapability;
use lsp_types::CompletionOptions;
use lsp_types::HoverProviderCapability;
use lsp_types::OneOf;
use lsp_types::PositionEncodingKind;
use lsp_types::RenameOptions;
use lsp_types::ServerCapabilities;
use lsp_types::SignatureHelpOptions;
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::Url;
use lsp_types::WorkspaceFoldersServerCapabilities;
use lsp_types::WorkspaceServerCapabilities;
use pretty_assertions::assert_eq;
use pyrefly_util::fs_anyhow;
use tempfile::TempDir;

use crate::commands::lsp::Args;
use crate::commands::lsp::IndexingMode;
use crate::commands::lsp::run_lsp;
use crate::test::util::init_test;

#[derive(Default)]
pub struct TestCase {
    pub(crate) messages_from_language_client: Vec<Message>,
    pub(crate) expected_messages_from_language_server: Vec<Message>,
    pub(crate) indexing_mode: IndexingMode,
    /// workspace folders open in the client
    pub(crate) workspace_folders: Option<Vec<(String, Url)>>,
    /// if client has configuration capability
    pub(crate) configuration: bool,
    /// if client has file watch capability
    pub(crate) file_watch: bool,
    /// if client has contents as uri capability
    pub(crate) contents_as_uri: bool,
}

pub fn run_test_lsp(test_case: TestCase) {
    init_test();
    let timeout = Duration::from_secs(25);
    let args = Args {
        indexing_mode: test_case.indexing_mode,
    };
    // language_client_sender is used to send messages to the language client
    // language_client_receiver sees messages sent to the language client
    let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
    // language_server_sender is used to send messages to the language server
    // language_server_receiver sees messages sent to the language server
    let (language_server_sender, language_server_receiver) = bounded::<Message>(0);

    // was there a response after a client -> server request
    let (server_response_received_sender, server_response_received_receiver) =
        bounded::<RequestId>(0);
    // was there a response after a server -> client request
    let (client_request_received_sender, client_request_received_receiver) =
        bounded::<RequestId>(0);

    let connection = Connection {
        sender: language_client_sender,
        receiver: language_server_receiver,
    };

    thread::scope(|scope| {
        // this thread runs the language server
        scope.spawn(move || {
            run_lsp(Arc::new(connection), || Ok(()), args)
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });
        // this thread sends messages to the language server (from test case)
        scope.spawn(move || {
            for msg in
                get_initialize_messages(&test_case.workspace_folders, test_case.configuration, test_case.file_watch, test_case.contents_as_uri)
                    .into_iter()
                    .chain(test_case.messages_from_language_client)
            {
                let send = || {
                    eprintln!("client--->server {}", serde_json::to_string(&msg).unwrap());
                    if let Err(err) = language_server_sender.send_timeout(msg.clone(), timeout) {
                        panic!("Failed to send message to language server: {:?}", err);
                    }
                };
                match &msg {
                    Message::Request(Request {
                        id,
                        method: _,
                        params: _,
                    }) => {
                        send();
                        if let Ok(response) =
                            server_response_received_receiver.recv_timeout(timeout)
                            && response == *id
                        {
                            // continue
                        } else {
                            panic!("Did not receive response for request {:?}", id);
                        }
                    }
                    Message::Notification(_) => send(),
                    // Language client responses need to ensure the request was sent first
                    Message::Response(Response {
                        id: response_id,
                        result: _,
                        error: _,
                    }) => {
                        let request_id = client_request_received_receiver
                            .recv_timeout(timeout)
                            .unwrap();
                        if request_id == *response_id {
                            send();
                        } else {
                            panic!(
                                "language client received request {}, expecting to send response for {}",
                                request_id, response_id
                            );
                        }
                    }
                }
            }
        });
        // this thread receives messages from the language server and validates responses
        scope.spawn(move || {
            let mut responses =
                get_initialize_responses(test_case.indexing_mode != IndexingMode::None)
                    .into_iter()
                    .chain(test_case.expected_messages_from_language_server)
                    .collect::<Vec<_>>();

            loop {
                if responses.is_empty() {
                    break;
                }
                match language_client_receiver.recv_timeout(timeout) {
                    Ok(msg) => {
                        eprintln!("client<---server {}", serde_json::to_string(&msg).unwrap());
                        let assert = |expected_response: String, response: String| {
                            if let Some(index) = expected_response.find("$$MATCH_EVERYTHING$$") {
                                assert_eq!(
                                    response[..index].to_string(),
                                    expected_response[..index].to_string(),
                                    "Response mismatch"
                                );
                            } else {
                                assert_eq!(response, expected_response, "Response mismatch");
                            }
                        };
                        match &msg {
                            Message::Response(Response {
                                id,
                                result: _,
                                error: _,
                            }) => {
                                assert(
                                    serde_json::to_string(&responses.remove(0)).unwrap(),
                                    serde_json::to_string(&msg).unwrap(),
                                );
                                server_response_received_sender.send(id.clone()).unwrap();
                            }
                            Message::Notification(notification) => {
                                eprintln!("Received notification: {:?}", notification);
                            }
                            Message::Request(Request {
                                id,
                                method: _,
                                params: _,
                            }) => {
                                assert(
                                    serde_json::to_string(&responses.remove(0)).unwrap(),
                                    serde_json::to_string(&msg).unwrap(),
                                );
                                client_request_received_sender.send(id.clone()).unwrap();
                            }
                        };
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        panic!("Timeout waiting for response. Expected ${:?}.", responses,);
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        panic!("Channel disconnected. Expected ${:?}.", responses);
                    }
                }
            }
        });
    });
}

fn get_initialize_params(
    workspace_folders: &Option<Vec<(String, Url)>>,
    configuration: bool,
    file_watch: bool,
    contents_as_uri: bool,
) -> serde_json::Value {
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

    if let Some(folders) = workspace_folders {
        params["capabilities"]["workspace"]["workspaceFolders"] = serde_json::json!(true);
        params["workspaceFolders"] = serde_json::json!(
            folders
                .iter()
                .map(|(name, path)| serde_json::json!({"name": name, "uri": path.to_string()}))
                .collect::<Vec<_>>()
        );
    }
    if file_watch {
        params["capabilities"]["workspace"]["didChangeWatchedFiles"] =
            serde_json::json!({"dynamicRegistration": true});
    }
    if configuration {
        params["capabilities"]["workspace"]["configuration"] = serde_json::json!(true);
    }
    if contents_as_uri {
        params["initializationOptions"] = serde_json::json!({"supportContentsAsUri": true});
    }

    params
}

fn get_initialize_messages(
    workspace_folders: &Option<Vec<(String, Url)>>,
    configuration: bool,
    file_watch: bool,
    contents_as_uri: bool,
) -> std::vec::Vec<lsp_server::Message> {
    vec![
        Message::from(Request {
            id: RequestId::from(1),
            method: "initialize".to_owned(),
            params: get_initialize_params(
                workspace_folders,
                configuration,
                file_watch,
                contents_as_uri,
            ),
        }),
        Message::from(Notification {
            method: "initialized".to_owned(),
            params: serde_json::json!({}),
        }),
    ]
}

fn get_initialize_responses(find_refs: bool) -> Vec<Message> {
    vec![Message::Response(Response {
        id: RequestId::from(1),
        result: Some(serde_json::json!({"capabilities": &ServerCapabilities {
            position_encoding: Some(PositionEncodingKind::UTF16),
            text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::INCREMENTAL)),
            definition_provider: Some(OneOf::Left(true)),
            code_action_provider: Some(CodeActionProviderCapability::Options(
                CodeActionOptions {
                    code_action_kinds: Some(vec![CodeActionKind::QUICKFIX]),
                    ..Default::default()
                },
            )),
            completion_provider: Some(CompletionOptions {
                trigger_characters: Some(vec![".".to_owned()]),
                ..Default::default()
            }),
            document_highlight_provider: Some(OneOf::Left(true)),
            // Find references won't work properly if we don't know all the files.
            references_provider: if find_refs {
                Some(OneOf::Left(true))
            } else {
                None
            },
            rename_provider: if find_refs {
                Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: Default::default(),
                }))
            } else {
                None
            },
            signature_help_provider: Some(SignatureHelpOptions {
                trigger_characters: Some(vec!["(".to_owned(), ",".to_owned()]),
                ..Default::default()
            }),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            inlay_hint_provider: Some(OneOf::Left(true)),
            document_symbol_provider: Some(OneOf::Left(true)),
            workspace: Some(WorkspaceServerCapabilities {
                workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                    supported: Some(true),
                    change_notifications: Some(OneOf::Left(true)),
                }),
                file_operations: None,
            }),
            semantic_tokens_provider: None,
            ..Default::default()
        }})),
        error: None,
    })]
}

pub fn build_did_open_notification(path: PathBuf) -> lsp_server::Notification {
    Notification {
        method: "textDocument/didOpen".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string(),
                "languageId": "python",
                "version": 1,
                "text": std::fs::read_to_string(path).unwrap()
            }
        }),
    }
}

pub fn get_test_files_root() -> TempDir {
    let mut source_files =
        std::env::current_dir().expect("std:env::current_dir() unavailable for test");
    let test_files_path = std::env::var("TEST_FILES_PATH")
        .expect("TEST_FILES_PATH env var not set: cargo or buck should set this automatically");
    source_files.push(test_files_path);

    // We copy all files over to a separate temp directory so we are consistent between Cargo and Buck.
    // In particular, given the current directory, Cargo is likely to find a pyproject.toml, but Buck won't.
    let t = TempDir::new().unwrap();
    copy_dir_recursively(&source_files, t.path());

    t
}

fn copy_dir_recursively(src: &Path, dst: &Path) {
    if !dst.exists() {
        std::fs::create_dir_all(dst).unwrap();
    }

    for entry in fs_anyhow::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let file_type = entry.file_type().unwrap();
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursively(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap();
        }
    }
}
