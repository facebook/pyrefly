/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
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
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::TextDocumentItem;
use lsp_types::Url;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::notification::Exit;
use lsp_types::notification::Notification as _;
use pretty_assertions::assert_eq;
use regex::Regex;

use crate::commands::lsp::IndexingMode;
use crate::commands::tsp::TspArgs;
use crate::commands::tsp::run_tsp;
// Re-export necessary types and functions for TSP tests
pub use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::get_initialize_messages;
use crate::test::lsp::lsp_interaction::util::get_initialize_responses;
pub use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::util::init_test;

pub struct TspTestCase {
    pub messages_from_language_client: Vec<Message>,
    pub expected_messages_from_language_server: Vec<Message>,
}

/// Enhanced TSP test runner that supports variable capture and substitution
///
/// Supports special placeholders:
/// - `$$MATCH_EVERYTHING$$` - matches any value (like existing infrastructure)
/// - `$$CAPTURE_<NAME>$$` - captures a value from server response into variable <NAME>
/// - `$$<NAME>$$` - substitutes the captured value of variable <NAME>
///
/// Example usage:
/// ```
/// // Step 1: Capture type handle from getType response
/// expected_response: {"handle": "$$CAPTURE_TYPE_HANDLE$$", "name": "$$MATCH_EVERYTHING$$"}
/// // Step 2: Use captured handle in getFunctionParts request
/// request: {"type": {"handle": "$$TYPE_HANDLE$$"}, "snapshot": 2}
/// ```
pub fn run_test_tsp_with_capture(test_case: TspTestCase) {
    init_test();
    let timeout = Duration::from_secs(25);
    let args = TspArgs {
        indexing_mode: IndexingMode::LazyBlocking,
    };

    let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
    let (language_server_sender, language_server_receiver) = bounded::<Message>(0);
    let (server_response_received_sender, server_response_received_receiver) =
        bounded::<RequestId>(0);
    let (client_request_received_sender, client_request_received_receiver) =
        bounded::<RequestId>(0);

    // Shared captured variables between threads
    let captured_variables = Arc::new(Mutex::new(HashMap::<String, serde_json::Value>::new()));

    let connection = Connection {
        sender: language_client_sender,
        receiver: language_server_receiver,
    };

    thread::scope(|scope| {
        // Language server thread
        scope.spawn(move || {
            run_tsp(Arc::new(connection), args)
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });

        // Client sender thread with variable substitution
        let captured_variables_sender = captured_variables.clone();
        scope.spawn(move || {
            let exit_message = Message::Notification(Notification {
                method: Exit::METHOD.to_owned(),
                params: serde_json::json!(null),
            });

            for msg in get_initialize_messages(&None, false, false)
                .into_iter()
                .chain(test_case.messages_from_language_client)
                .chain(std::iter::once(exit_message.clone()))
            {
                let stop_language_server = || {
                    language_server_sender.send_timeout(exit_message.clone(), timeout).unwrap();
                };

                // Apply variable substitution to outgoing messages
                let captured_vars = captured_variables_sender.lock().unwrap();
                let substituted_msg = substitute_variables_in_message(&msg, &captured_vars);
                drop(captured_vars); // Release the lock

                let send = || {
                    eprintln!("client--->server {}", serde_json::to_string(&substituted_msg).unwrap());
                    if let Err(err) = language_server_sender.send_timeout(substituted_msg.clone(), timeout) {
                        panic!("Failed to send message to language server: {err:?}");
                    }
                };

                match &substituted_msg {
                    Message::Request(Request { id, .. }) => {
                        send();
                        if let Ok(response) = server_response_received_receiver.recv_timeout(timeout)
                            && response == *id
                        {
                            // Continue to next message
                        } else {
                            stop_language_server();
                            panic!("Did not receive response for request {id:?}");
                        }
                    }
                    Message::Notification(_) => send(),
                    Message::Response(Response { id: response_id, .. }) => {
                        let request_id = client_request_received_receiver.recv_timeout(timeout).unwrap();
                        if request_id == *response_id {
                            send();
                        } else {
                            stop_language_server();
                            panic!(
                                "language client received request {request_id}, expecting to send response for {response_id}"
                            );
                        }
                    }
                }
            }
        });

        // Server receiver thread with variable capture
        let captured_variables_receiver = captured_variables.clone();
        scope.spawn(move || {
            let mut responses = get_initialize_responses(false)
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

                        match &msg {
                            Message::Response(Response { id, .. }) => {
                                let expected = responses.remove(0);

                                // Perform advanced matching with variable capture
                                let mut captured_vars = captured_variables_receiver.lock().unwrap();
                                if let Err(err) =
                                    match_with_capture(&expected, &msg, &mut captured_vars)
                                {
                                    panic!("Response mismatch: {err}");
                                }
                                drop(captured_vars); // Release the lock

                                server_response_received_sender.send(id.clone()).unwrap();
                            }
                            Message::Notification(notification) => {
                                eprintln!("Received notification: {notification:?}");
                            }
                            Message::Request(Request { id, .. }) => {
                                let expected = responses.remove(0);

                                let mut captured_vars = captured_variables_receiver.lock().unwrap();
                                if let Err(err) =
                                    match_with_capture(&expected, &msg, &mut captured_vars)
                                {
                                    panic!("Request mismatch: {err}");
                                }
                                drop(captured_vars); // Release the lock

                                client_request_received_sender.send(id.clone()).unwrap();
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        panic!("Timeout waiting for response. Expected {responses:?}.");
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        panic!("Channel disconnected. Expected {responses:?}.");
                    }
                }
            }
        });
    });
}

/// Simple TSP test runner that follows the same pattern as LSP tests but uses TSP server
pub fn run_test_tsp(test_case: TestCase) {
    init_test();
    let timeout = Duration::from_secs(25);
    let args = TspArgs {
        indexing_mode: test_case.indexing_mode,
    };

    let (language_client_sender, language_client_receiver) = bounded::<Message>(0);
    let (language_server_sender, language_server_receiver) = bounded::<Message>(0);
    let (server_response_received_sender, server_response_received_receiver) =
        bounded::<RequestId>(0);
    let (client_request_received_sender, client_request_received_receiver) =
        bounded::<RequestId>(0);

    let connection = Connection {
        sender: language_client_sender,
        receiver: language_server_receiver,
    };

    thread::scope(|scope| {
        // TSP server thread
        scope.spawn(move || {
            run_tsp(Arc::new(connection), args)
                .map(|_| ())
                .map_err(|e| std::io::Error::other(e.to_string()))
        });

        // Client sender thread
        scope.spawn(move || {
            let exit_message = Message::Notification(Notification {
                method: Exit::METHOD.to_owned(),
                params: serde_json::json!(null),
            });

            for msg in get_initialize_messages(&test_case.workspace_folders, test_case.configuration, test_case.file_watch)
                .into_iter()
                .chain(test_case.messages_from_language_client)
                .chain(std::iter::once(exit_message.clone()))
            {
                let stop_language_server = || {
                    language_server_sender.send_timeout(exit_message.clone(), timeout).unwrap();
                };

                let send = || {
                    eprintln!("client--->server {}", serde_json::to_string(&msg).unwrap());
                    if let Err(err) = language_server_sender.send_timeout(msg.clone(), timeout) {
                        panic!("Failed to send message to language server: {err:?}");
                    }
                };

                match &msg {
                    Message::Request(Request { id, .. }) => {
                        send();
                        if let Ok(response) = server_response_received_receiver.recv_timeout(timeout)
                            && response == *id
                        {
                            // Continue to next message
                        } else {
                            stop_language_server();
                            panic!("Did not receive response for request {id:?}");
                        }
                    }
                    Message::Notification(_) => send(),
                    Message::Response(lsp_server::Response { id: response_id, .. }) => {
                        let request_id = client_request_received_receiver.recv_timeout(timeout).unwrap();
                        if request_id == *response_id {
                            send();
                        } else {
                            stop_language_server();
                            panic!(
                                "language client received request {request_id}, expecting to send response for {response_id}"
                            );
                        }
                    }
                }
            }
        });

        // Server receiver thread
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

                        match &msg {
                            Message::Response(Response { id, .. }) => {
                                let expected = responses.remove(0);
                                let assert = |expected_response: String, response: String| {
                                    if let Some(index) =
                                        expected_response.find("$$MATCH_EVERYTHING$$")
                                    {
                                        assert_eq!(
                                            response[..index].to_string(),
                                            expected_response[..index].to_string(),
                                            "Response mismatch"
                                        );
                                    } else {
                                        assert_eq!(
                                            response, expected_response,
                                            "Response mismatch"
                                        );
                                    }
                                };
                                assert(
                                    serde_json::to_string(&expected).unwrap(),
                                    serde_json::to_string(&msg).unwrap(),
                                );
                                server_response_received_sender.send(id.clone()).unwrap();
                            }
                            Message::Notification(notification) => {
                                eprintln!("Received notification: {notification:?}");
                            }
                            Message::Request(Request { id, .. }) => {
                                let expected = responses.remove(0);
                                let assert = |expected_response: String, response: String| {
                                    if let Some(index) =
                                        expected_response.find("$$MATCH_EVERYTHING$$")
                                    {
                                        assert_eq!(
                                            response[..index].to_string(),
                                            expected_response[..index].to_string(),
                                            "Response mismatch"
                                        );
                                    } else {
                                        assert_eq!(
                                            response, expected_response,
                                            "Response mismatch"
                                        );
                                    }
                                };
                                assert(
                                    serde_json::to_string(&expected).unwrap(),
                                    serde_json::to_string(&msg).unwrap(),
                                );
                                client_request_received_sender.send(id.clone()).unwrap();
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        panic!("Timeout waiting for response. Expected {responses:?}.");
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        panic!("Channel disconnected. Expected {responses:?}.");
                    }
                }
            }
        });
    });
}

/// Substitute captured variables in outgoing messages
fn substitute_variables_in_message(
    msg: &Message,
    captured_variables: &HashMap<String, serde_json::Value>,
) -> Message {
    let msg_str = serde_json::to_string(msg).unwrap();
    let substituted_str = substitute_variables_in_string(&msg_str, captured_variables);
    serde_json::from_str(&substituted_str).unwrap()
}

/// Substitute variables in a JSON string
fn substitute_variables_in_string(
    text: &str,
    captured_variables: &HashMap<String, serde_json::Value>,
) -> String {
    let variable_regex = Regex::new(r#""?\$\$([A-Z_][A-Z0-9_]*)\$\$"?"#).unwrap();

    variable_regex
        .replace_all(text, |caps: &regex::Captures| {
            let var_name = &caps[1];
            if let Some(value) = captured_variables.get(var_name) {
                // Return the JSON representation of the value directly
                value.to_string()
            } else {
                // If variable not found, leave placeholder as-is
                caps[0].to_string()
            }
        })
        .to_string()
}

/// Advanced matching that supports both wildcards and variable capture
fn match_with_capture(
    expected: &Message,
    actual: &Message,
    captured_variables: &mut HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    let expected_str = serde_json::to_string(expected)
        .map_err(|e| format!("Failed to serialize expected: {e}"))?;
    let actual_str =
        serde_json::to_string(actual).map_err(|e| format!("Failed to serialize actual: {e}"))?;

    match_json_with_capture(&expected_str, &actual_str, captured_variables)
}

/// Core JSON matching logic with capture and wildcard support
fn match_json_with_capture(
    expected: &str,
    actual: &str,
    captured_variables: &mut HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    // Handle $$MATCH_EVERYTHING$$ wildcard only for simple cases where entire content is wildcard
    if expected.trim() == "\"$$MATCH_EVERYTHING$$\"" {
        return Ok(());
    }

    // Parse both as JSON for structured comparison
    let expected_json: serde_json::Value =
        serde_json::from_str(expected).map_err(|e| format!("Invalid expected JSON: {e}"))?;
    let actual_json: serde_json::Value =
        serde_json::from_str(actual).map_err(|e| format!("Invalid actual JSON: {e}"))?;

    match_json_values(&expected_json, &actual_json, captured_variables)
}

/// Recursively match JSON values with capture support
fn match_json_values(
    expected: &serde_json::Value,
    actual: &serde_json::Value,
    captured_variables: &mut HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    match expected {
        serde_json::Value::String(expected_str) => {
            // Check for capture pattern: $$CAPTURE_<NAME>$$
            let capture_regex = Regex::new(r"^\$\$CAPTURE_([A-Z_][A-Z0-9_]*)\$\$$").unwrap();
            if let Some(caps) = capture_regex.captures(expected_str) {
                let var_name = caps[1].to_string();
                captured_variables.insert(var_name.clone(), actual.clone());
                eprintln!("Captured variable {var_name}: {actual}");
                return Ok(());
            }

            // Check for $$MATCH_EVERYTHING$$ wildcard
            if expected_str == "$$MATCH_EVERYTHING$$" {
                return Ok(());
            }

            // Regular string comparison
            match actual {
                serde_json::Value::String(actual_str) => {
                    if expected_str == actual_str {
                        Ok(())
                    } else {
                        Err(format!(
                            "String mismatch: expected '{expected_str}', got '{actual_str}'"
                        ))
                    }
                }
                _ => Err(format!(
                    "Type mismatch: expected string '{expected_str}', got {actual:?}"
                )),
            }
        }
        serde_json::Value::Object(expected_obj) => match actual {
            serde_json::Value::Object(actual_obj) => {
                for (key, expected_value) in expected_obj {
                    match actual_obj.get(key) {
                        Some(actual_value) => {
                            match_json_values(expected_value, actual_value, captured_variables)?;
                        }
                        None => {
                            return Err(format!("Missing key '{key}' in actual object"));
                        }
                    }
                }
                Ok(())
            }
            _ => Err(format!("Type mismatch: expected object, got {actual:?}")),
        },
        serde_json::Value::Array(expected_arr) => match actual {
            serde_json::Value::Array(actual_arr) => {
                if expected_arr.len() != actual_arr.len() {
                    return Err(format!(
                        "Array length mismatch: expected {}, got {}",
                        expected_arr.len(),
                        actual_arr.len()
                    ));
                }
                for (i, (expected_item, actual_item)) in
                    expected_arr.iter().zip(actual_arr.iter()).enumerate()
                {
                    // Special case: if expected item is $$MATCH_EVERYTHING$$ string, allow any actual item
                    if let serde_json::Value::String(s) = expected_item
                        && s == "$$MATCH_EVERYTHING$$"
                    {
                        continue; // Skip validation for this array element
                    }

                    match_json_values(expected_item, actual_item, captured_variables)
                        .map_err(|e| format!("Array index {i}: {e}"))?;
                }
                Ok(())
            }
            _ => Err(format!("Type mismatch: expected array, got {actual:?}")),
        },
        _ => {
            // For numbers, booleans, null - direct comparison
            if expected == actual {
                Ok(())
            } else {
                Err(format!(
                    "Value mismatch: expected {expected:?}, got {actual:?}"
                ))
            }
        }
    }
}

pub fn build_did_open_notification(file_path: std::path::PathBuf) -> lsp_server::Notification {
    let content = std::fs::read_to_string(&file_path).unwrap();
    let uri = Url::from_file_path(file_path).unwrap();

    lsp_server::Notification {
        method: DidOpenTextDocument::METHOD.to_owned(),
        params: serde_json::to_value(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: "python".to_owned(),
                version: 1,
                text: content,
            },
        })
        .unwrap(),
    }
}
