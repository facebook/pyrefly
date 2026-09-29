/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A request the server cannot act on is answered with an error rather than
//! dropped, so the client is never left waiting.

use lsp_server::ErrorCode;
use lsp_server::RequestId;
use tempfile::TempDir;

use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Request;
use crate::test::tsp::tsp_interaction::object_model::TspInteraction;

#[test]
fn test_tsp_unknown_method_returns_method_not_found() {
    let temp_dir = TempDir::new().unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "typeServer/thisMethodDoesNotExist".to_owned(),
        params: serde_json::json!(null),
        activity_key: None,
    }));

    let response = tsp.client.receive_response_skip_notifications();
    assert_eq!(response.id, RequestId::from(2));
    assert_eq!(response.result, None);
    let error = response.error.expect("unknown method should be an error");
    assert_eq!(error.code, ErrorCode::MethodNotFound as i32);
    assert!(
        error.message.contains("typeServer/thisMethodDoesNotExist"),
        "error should name the unsupported method, got: {}",
        error.message
    );

    tsp.shutdown();
}

#[test]
fn test_tsp_malformed_params_return_invalid_params() {
    let temp_dir = TempDir::new().unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // A known method, but `arg` is missing the `uri` and `range` that
    // `GetTypeParams` requires.
    tsp.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "typeServer/getComputedType".to_owned(),
        params: serde_json::json!({ "arg": {}, "snapshot": 0 }),
        activity_key: None,
    }));

    let response = tsp.client.receive_response_skip_notifications();
    assert_eq!(response.id, RequestId::from(2));
    assert_eq!(response.result, None);
    let error = response.error.expect("malformed params should be an error");
    assert_eq!(error.code, ErrorCode::InvalidParams as i32);

    tsp.shutdown();
}
