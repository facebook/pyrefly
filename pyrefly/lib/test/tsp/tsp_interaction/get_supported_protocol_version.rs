/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;
use tempfile::TempDir;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;

#[test]
fn test_tsp_get_supported_protocol_version_interaction() {
    // Test retrieval of TSP protocol version with null params
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("version_test.py");

    let test_content = r#"# Simple test file for protocol version request
print("Hello, World!")
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Open the test file
    tsp.server.did_open("version_test.py");

    // Wait for any diagnostics
    tsp.client.expect_any_message();

    // Get supported protocol version - test with null params
    tsp.server.get_supported_protocol_version();

    // Expect protocol version response
    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!("0.2.0")),
        error: None,
    });

    tsp.shutdown();
}

#[test]
fn test_tsp_get_supported_protocol_version_interaction_empty_params() {
    // Test protocol version retrieval with empty object params
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("version_test_2.py");

    let test_content = r#"# Another test file
x = 42
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    // Open the test file
    tsp.server.did_open("version_test_2.py");

    // Wait for any diagnostics
    tsp.client.expect_any_message();

    // Get supported protocol version - test with empty object params
    tsp.server.get_supported_protocol_version();

    // Expect protocol version response
    tsp.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!("0.2.0")),
        error: None,
    });

    tsp.shutdown();
}
