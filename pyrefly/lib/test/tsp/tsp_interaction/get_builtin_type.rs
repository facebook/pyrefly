/*
 * TSP interaction tests for get_builtin_type request handler
 *
 * These tests verify the full TSP message protocol for get_builtin_type requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getBuiltinType
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * These integration tests complement the unit tests in lib/test/tsp/get_builtin_type.rs by testing
 * the complete TSP protocol implementation rather than individual handler components.
 */

use lsp_server::ErrorCode;
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
fn test_tsp_get_builtin_type_interaction_basic() {
    // This test demonstrates TSP interaction testing following the LSP pattern
    // It verifies that:
    // 1. TSP requests can be sent through the same server infrastructure as LSP
    // 2. Files can be opened and TSP requests made against them
    // 3. The server properly handles typeServer/getBuiltinType requests
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
            // Then request builtin type information for 'int' using snapshot from getSnapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": {
                                "line": 0,
                                "character": 0
                            },
                            "end": {
                                "line": 0,
                                "character": 1
                            }
                        }
                    },
                    "name": "int",
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
            // Then expect response to getBuiltinType - expect type info for 'int'
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "category": 3,  // Category 3 = CLASS
                    "categoryFlags": 0,
                    "flags": 2,
                    "handle": "$$MATCH_EVERYTHING$$",  // Handle can vary, so use wildcard
                    "moduleName": {"leadingDots": 0, "nameParts": ["builtins"]},
                    "name": "int",
                    "aliasName": null
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
fn test_tsp_get_builtin_type_interaction_multiple_types() {
    // Create a temporary directory and test file
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("builtin_types.py");

    // Create simple test content for scoping context
    let test_content = r#"# Test file for builtin types
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
            // Test: Get builtin type 'str'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 1, "character": 0 },
                            "end": { "line": 1, "character": 1 }
                        }
                    },
                    "name": "str",
                    "snapshot": 2
                }),
            }),
            // Test: Get builtin type 'list'
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 0 },
                            "end": { "line": 2, "character": 1 }
                        }
                    },
                    "name": "list",
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
            // Type for builtin 'str'
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "category": 3,
                    "categoryFlags": 0,
                    "flags": 2,
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": {"leadingDots": 0, "nameParts": ["builtins"]},
                    "name": "str",
                    "aliasName": null
                })),
                error: None,
            }),
            // Type for builtin 'list'
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": 3,
                    "categoryFlags": 0,
                    "flags": 2,
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": {"leadingDots": 0, "nameParts": ["builtins"]},
                    "name": "list[Any]",  // List types are parameterized
                    "aliasName": null
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
fn test_tsp_get_builtin_type_interaction_invalid_snapshot() {
    // Test error handling for invalid snapshots
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("builtin_types.py");

    let test_content = r#"# Test file
x = 42
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
            // Test: Get builtin type with invalid (old) snapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 1 }
                        }
                    },
                    "name": "int",
                    "snapshot": 1  // Use outdated snapshot ID (should be 2)
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
            // Error response for invalid snapshot
            Message::Response(Response {
                id: RequestId::from(3),
                result: None,
                error: Some(lsp_server::ResponseError {
                    code: ErrorCode::ServerCancelled as i32,
                    message: "Snapshot outdated".to_owned(),
                    data: None,
                }),
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}

#[test]
fn test_tsp_get_builtin_type_interaction_unknown_type() {
    // Test handling of unknown builtin types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("builtin_types.py");

    let test_content = r#"# Test file
x = 42
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
            // Test: Get unknown builtin type
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 1 }
                        }
                    },
                    "name": "unknown_builtin_type",
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
            // Null response for unknown type (this is valid - not an error)
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(null)), // Unknown types return null, not an error
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
fn test_tsp_get_builtin_type_interaction_protocol_structure() {
    // Test that the protocol structure matches expected TSP patterns
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("builtin_types.py");

    let test_content = r#"# Test protocol structure
def test_function():
    pass
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
            // Test: Validate that the getBuiltinType request follows proper TSP structure
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getBuiltinType".to_owned(),
                params: serde_json::json!({
                    "scopingNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 1, "character": 0 },
                            "end": { "line": 1, "character": 1 }
                        }
                    },
                    "name": "function",  // Test the 'function' builtin type
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
            // Function type response - validate structure
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "category": 3,
                    "categoryFlags": 0,
                    "flags": 2,
                    "handle": "$$MATCH_EVERYTHING$$",  // Handle format can vary
                    "moduleName": {"leadingDots": 0, "nameParts": ["types"]},  // Function type is in 'types' module
                    "name": "FunctionType",
                    "aliasName": null
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
