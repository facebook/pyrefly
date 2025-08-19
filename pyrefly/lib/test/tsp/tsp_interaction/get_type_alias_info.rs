/*
 * TSP interaction tests for get_type_alias_info request handler
 *
 * These tests verify the full TSP message protocol for get_type_alias_info requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getTypeAliasInfo
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * These integration tests complement the unit tests in lib/test/tsp/get_type_alias_info.rs by testing
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
use crate::test::tsp::tsp_interaction::util::run_test_tsp;

#[test]
fn test_tsp_get_type_alias_info_interaction_basic() {
    // This test demonstrates TSP interaction testing for type alias info
    // It verifies that:
    // 1. TSP typeServer/getTypeAliasInfo requests can be sent through the server infrastructure
    // 2. Files with type aliases can be opened and TSP requests made against them
    // 3. The server properly handles typeServer/getTypeAliasInfo requests
    // 4. Response infrastructure works for type alias TSP messages

    // Create a temporary directory and test file with type aliases
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_aliases.py");

    // Create test content with a simple type alias
    let test_content = r#"from typing import List

# Simple type alias
StringList = List[str]
"#;

    std::fs::write(&test_file_path, test_content).unwrap();

    let _file_uri = Url::from_file_path(&test_file_path).unwrap();
    let request_file_name = test_file_path.clone();

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
            // Try to get type alias info with a simple type - this should return null since
            // getTypeAliasInfo expects a type that is actually a TypeAlias, which may not
            // be what we get from variable assignments like "StringList = List[str]"
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeAliasInfo".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "category": 3,  // CLASS category
                        "categoryFlags": 0,
                        "flags": 2,
                        "handle": "test_handle",
                        "moduleName": {"leadingDots": 0, "nameParts": ["typing"]},
                        "name": "list[str]",
                        "aliasName": null
                    },
                    "snapshot": 2
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
            // Expect response to getTypeAliasInfo - should return null since this type is not a TypeAlias
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(null)),
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
fn test_tsp_get_type_alias_info_interaction_empty_params() {
    // Test error handling for missing required parameters
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("alias_test.py");

    let test_content = r#"from typing import Dict
MyDict = Dict[str, int]
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let request_file_name = test_file_path.clone();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the file
            Message::from(build_did_open_notification(request_file_name.clone())),
            // Try to call getTypeAliasInfo with empty params
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getTypeAliasInfo".to_owned(),
                params: serde_json::json!({}), // Empty params - should cause parsing error
            }),
        ],
        expected_messages_from_language_server: vec![
            // Error response for method not found (since empty params cause deserialization failure)
            Message::Response(Response {
                id: RequestId::from(2),
                result: None,
                error: Some(lsp_server::ResponseError {
                    code: lsp_server::ErrorCode::MethodNotFound as i32,
                    message: "$$MATCH_EVERYTHING$$".to_owned(), // Accept any error message
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
fn test_tsp_get_type_alias_info_interaction_protocol_structure() {
    // Test that the protocol properly handles well-formed requests
    // even if the business logic returns null for non-alias types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("complex_aliases.py");

    let test_content = r#"from typing import List, Dict, Optional, Union, Tuple, Callable

# Various type alias patterns for testing
SimpleList = List[str]
ComplexDict = Dict[str, Union[int, str, bool]]
OptionalType = Optional[Tuple[str, int]]
CallableType = Callable[[str, int], bool]
NestedGeneric = Dict[str, List[Optional[int]]]
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let request_file_name = test_file_path.clone();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the file
            Message::from(build_did_open_notification(request_file_name.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Test first type alias info request with a concrete type
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeAliasInfo".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "category": 3,  // CLASS category
                        "categoryFlags": 0,
                        "flags": 2,
                        "handle": "test_handle_1",
                        "moduleName": {"leadingDots": 0, "nameParts": ["typing"]},
                        "name": "list[str]",
                        "aliasName": null
                    },
                    "snapshot": 2
                }),
            }),
            // Test second type alias info request with a different type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeAliasInfo".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "category": 3,  // CLASS category
                        "categoryFlags": 0,
                        "flags": 2,
                        "handle": "test_handle_2",
                        "moduleName": {"leadingDots": 0, "nameParts": ["typing"]},
                        "name": "dict[str, Union[int, str, bool]]",
                        "aliasName": null
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
            // First type alias info response - should return null for non-alias types
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(null)),
                error: None,
            }),
            // Second type alias info response - should return null for non-alias types
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!(null)),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
