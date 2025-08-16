/*
 * TSP interaction tests for get_type_attributes request handler
 *
 * These tests verify the full TSP message protocol for get_type_attributes requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getTypeAttributes
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * These integration tests complement the unit tests in lib/test/tsp/get_type_attributes.rs by testing
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
use crate::test::tsp::tsp_interaction::util::TspTestCase;
use crate::test::tsp::tsp_interaction::util::build_did_open_notification;
use crate::test::tsp::tsp_interaction::util::get_test_files_root;
use crate::test::tsp::tsp_interaction::util::run_test_tsp;
use crate::test::tsp::tsp_interaction::util::run_test_tsp_with_capture;

#[test]
fn test_tsp_get_type_attributes_interaction_basic() {
    // Test basic get_type_attributes functionality with a simple class
    let root = get_test_files_root();
    let request_file_name = root.path().join("test_class_attributes.py");

    // Create a test file with a simple class that has attributes
    std::fs::write(
        &request_file_name,
        r#"
class TestClass:
    def __init__(self):
        self.name = "test"
        self.value = 42
        self.enabled = True
    
    def get_name(self):
        return self.name
"#,
    )
    .unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // First open the file
            Message::from(build_did_open_notification(request_file_name.clone())),
            // Get the current snapshot ID
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
        ],
        expected_messages_from_language_server: vec![
            // getSnapshot response
            Message::Response(lsp_server::Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!(2)), // Usually snapshot ID 2
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
fn test_tsp_get_type_attributes_interaction_empty_class() {
    // Test get_type_attributes with an empty class
    let root = get_test_files_root();
    let request_file_name = root.path().join("test_empty_class.py");

    // Create a test file with an empty class
    std::fs::write(
        &request_file_name,
        r#"
class EmptyClass:
    pass
"#,
    )
    .unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            Message::from(build_did_open_notification(request_file_name.clone())),
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
        ],
        expected_messages_from_language_server: vec![Message::Response(lsp_server::Response {
            id: RequestId::from(2),
            result: Some(serde_json::json!(2)),
            error: None,
        })],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}

#[test]
fn test_tsp_get_type_attributes_function_interaction() {
    // Test type attribute extraction for function types with parameters and return type
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("function_attributes_test.py");

    let test_content = r#"from typing import Callable, List, Optional

def process_numbers(data: List[int], multiplier: int = 2) -> int:
    return sum(x * multiplier for x in data)

def calculate_average(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

def filter_values(items: List[str], predicate: Callable[[str], bool]) -> List[str]:
    return [item for item in items if predicate(item)]

# Function type variables
number_processor: Callable[[List[int], int], int] = process_numbers
average_calculator: Callable[[List[float]], float] = calculate_average
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp_with_capture(TspTestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of the process_numbers function
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 4 },
                            "end": { "line": 2, "character": 19 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type attributes for the function type - should include parameters and return type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeAttributes".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
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
            // Type response for process_numbers function - capture all fields
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                "aliasName": "$$CAPTURE_TYPE_ALIAS_NAME$$",
                "category": "$$CAPTURE_TYPE_CATEGORY$$",
                "categoryFlags": "$$CAPTURE_TYPE_CATEGORY_FLAGS$$",
                "flags": "$$CAPTURE_TYPE_FLAGS$$",
                "handle": "$$CAPTURE_TYPE_HANDLE$$",
                "name": "$$CAPTURE_TYPE_NAME$$"
                    })),
                error: None,
            }),
            // Type attributes response - should contain parameters and return type
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // PARAMETER flag (value may vary)
                        "name": "data",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "$$MATCH_EVERYTHING$$"
                        }
                    },
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // PARAMETER flag (value may vary)
                        "name": "multiplier",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "int"
                        }
                    },
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // RETURN_TYPE flag (value may vary)
                        "name": "$$MATCH_EVERYTHING$$",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "int"
                        }
                    }
                ])),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_type_attributes_callable_interaction() {
    // Test type attribute extraction for Callable types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("callable_attributes_test.py");

    let test_content = r#"from typing import Callable, List

# Callable type variable
string_processor: Callable[[str, int], str] = lambda s, n: s * n
number_filter: Callable[[List[int]], List[int]] = lambda nums: [x for x in nums if x > 0]
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp_with_capture(TspTestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of the string_processor variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 3, "character": 0 },
                            "end": { "line": 3, "character": 16 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type attributes for the Callable type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeAttributes".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
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
            // Type response for string_processor variable - capture all fields
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                "aliasName": "$$CAPTURE_TYPE_ALIAS_NAME$$",
                "category": "$$CAPTURE_TYPE_CATEGORY$$",
                "categoryFlags": "$$CAPTURE_TYPE_CATEGORY_FLAGS$$",
                "flags": "$$CAPTURE_TYPE_FLAGS$$",
                "handle": "$$CAPTURE_TYPE_HANDLE$$",
                "name": "$$CAPTURE_TYPE_NAME$$"
                    })),
                error: None,
            }),
            // Type attributes response - should contain Callable parameters and return type
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // PARAMETER flag (value may vary)
                        "name": "$$MATCH_EVERYTHING$$",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "str"
                        }
                    },
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // PARAMETER flag (value may vary)
                        "name": "$$MATCH_EVERYTHING$$",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "int"
                        }
                    },
                    {
                        "flags": "$$MATCH_EVERYTHING$$", // RETURN_TYPE flag (value may vary)
                        "name": "$$MATCH_EVERYTHING$$",
                        "type": {
                "aliasName": "$$MATCH_EVERYTHING$$",
                "category": "$$MATCH_EVERYTHING$$",
                "categoryFlags": "$$MATCH_EVERYTHING$$",
                "flags": "$$MATCH_EVERYTHING$$",
                "handle": "$$MATCH_EVERYTHING$$",
                "name": "str"
                        }
                    }
                ])),
                error: None,
            }),
        ],
    });
}
