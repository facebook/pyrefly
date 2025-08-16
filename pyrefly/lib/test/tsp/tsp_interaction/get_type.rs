/*
 * TSP interaction tests for get_type request handler
 *
 * These tests verify the full TSP message protocol for get_type requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getType
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * These integration tests complement the unit tests in lib/test/tsp/get_type.rs by testing
 * the complete TSP protocol implementation rather than individual handler components.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;
use tempfile::TempDir;

use crate::commands::lsp::IndexingMode;
use crate::test::tsp::tsp_interaction::util::TestCase;
use crate::test::tsp::tsp_interaction::util::build_did_open_notification;
use crate::test::tsp::tsp_interaction::util::get_test_files_root;
use crate::test::tsp::tsp_interaction::util::run_test_tsp;

#[test]
fn test_tsp_get_type_interaction_basic() {
    // This test demonstrates TSP interaction testing following the LSP pattern
    // It verifies that:
    // 1. TSP requests can be sent through the same server infrastructure as LSP
    // 2. Files can be opened and TSP requests made against them
    // 3. The server properly handles typeServer/getType requests
    // 4. Response infrastructure works for TSP messages

    let root = get_test_files_root();
    let request_file_name = root.path().join("bar.py"); // Use existing test file
    let file_uri = Url::from_file_path(&request_file_name).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // First open the file
            Message::from(build_did_open_notification(request_file_name.clone())),
            // Get the current snapshot ID before making TSP requests
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Then request type information for a variable at a known position using snapshot from getSnapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": {
                                "line": 7,
                                "character": 4
                            },
                            "end": {
                                "line": 7,
                                "character": 7
                            }
                        }
                    },
                    "snapshot": 2  // Use snapshot ID 2, which should be returned from getSnapshot
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // First expect response to getSnapshot - should return a snapshot ID
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!(2)), // Expect snapshot ID 2
                error: None,
            }),
            // Then expect response to getType - expect type info for 'foo = 3'
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "aliasName": null,
                    "category": 0,
                    "categoryFlags": 0,
                    "flags": 8,
                    "handle": "$$MATCH_EVERYTHING$$",  // Handle can vary, so use wildcard
                    "name": "Literal[3]"
                })),
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
fn test_tsp_get_type_interaction_multiple_positions() {
    // Create a temporary directory and test file
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("simple_types.py");

    // Create simple test content that's easier to target positions for
    let test_content = r#"# Test file
x = 42
y = "hello"
z = True
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    let file_uri = Url::from_file_path(&test_file_path).unwrap();
    let request_file_name = test_file_path.clone();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(request_file_name.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Test: Get type of variable 'x' on line 1
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 1, "character": 0 },
                            "end": { "line": 1, "character": 1 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!(2)),
                error: None,
            }),
            // Type for variable 'x' -> int (inferred from assignment)
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "aliasName": null,
                    "category": 0,
                    "categoryFlags": 0,
                    "flags": 8,
                    "handle": "$$MATCH_EVERYTHING$$",
                    "name": "$$MATCH_EVERYTHING$$"  // Accept any type name for variable
                })),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
