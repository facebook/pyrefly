/*
 * TSP interaction tests for getSupportedProtocolVersion request handler
 *
 * These tests verify the full TSP message protocol for getSupportedProtocolVersion requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSupportedProtocolVersion
 * 3. Validating that the correct protocol version is returned
 * 4. Using real message passing to simulate end-to-end TSP interactions
 *
 * The getSupportedProtocolVersion request returns the current TSP protocol version.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use tempfile::TempDir;

use crate::commands::lsp::IndexingMode;
use crate::test::tsp::tsp_interaction::util::TestCase;
use crate::test::tsp::tsp_interaction::util::build_did_open_notification;
use crate::test::tsp::tsp_interaction::util::run_test_tsp;

#[test]
fn test_tsp_get_supported_protocol_version_interaction() {
    // Test retrieval of TSP protocol version
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("version_test.py");

    let test_content = r#"# Simple test file for protocol version request
print("Hello, World!")
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get supported protocol version - test with null params
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSupportedProtocolVersion".to_owned(),
                params: serde_json::json!(null),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Protocol version response - should return "0.1.0"
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!("0.1.0")),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
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

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get supported protocol version - test with empty object params
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSupportedProtocolVersion".to_owned(),
                params: serde_json::json!({}),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Protocol version response - should return "0.1.0"
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!("0.1.0")),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
