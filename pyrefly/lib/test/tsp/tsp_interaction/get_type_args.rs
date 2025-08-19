/*
 * TSP interaction tests for get_type_args request handler
 *
 * These tests verify the full TSP message protocol for get_type_args:
 * 1. Following the LSP interaction test pattern using run_test_tsp_with_capture
 * 2. Testing complete request/response flows including typeServer/getSnapshot, typeServer/getType, and typeServer/getTypeArgs
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_type_args request requires a type handle (obtained from get_type) and returns
 * the type arguments for generic types like List[int], Dict[str, float], Optional[T], etc.
 */

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;
use tempfile::TempDir;

use crate::test::tsp::tsp_interaction::util::TspTestCase;
use crate::test::tsp::tsp_interaction::util::build_did_open_notification;
use crate::test::tsp::tsp_interaction::util::run_test_tsp_with_capture;

#[test]
fn test_tsp_get_type_args_interaction_list() {
    // Test type argument extraction for List[T] types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_args_list_test.py");

    let test_content = r#"from typing import List, Dict, Tuple, Optional

numbers: List[int] = [1, 2, 3, 4, 5]
names: List[str] = ["Alice", "Bob", "Charlie"]
mixed_data: List[Tuple[str, int]] = [("a", 1), ("b", 2)]

def process_numbers(data: List[int]) -> int:
    return sum(data)

def process_names(data: List[str]) -> str:
    return ", ".join(data)
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
            // Get type of the List[int] variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 0 },
                            "end": { "line": 2, "character": 7 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type arguments for the List[int] type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeArgs".to_owned(),
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
            // Type response for List[int] variable - capture all fields
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
            // Type args response - should contain Type object for int
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([{
                    "category": "$$MATCH_EVERYTHING$$",
                    "categoryFlags": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": "$$MATCH_EVERYTHING$$",
                    "name": "int"
                }])),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_type_args_interaction_dict() {
    // Test type argument extraction for Dict[K, V] types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_args_dict_test.py");

    let test_content = r#"from typing import Dict, List, Optional

user_ages: Dict[str, int] = {"Alice": 30, "Bob": 25, "Charlie": 35}
scores: Dict[str, float] = {"math": 95.5, "science": 87.2, "english": 92.0}
nested_data: Dict[str, List[int]] = {"group1": [1, 2, 3], "group2": [4, 5, 6]}

def get_user_age(users: Dict[str, int], name: str) -> Optional[int]:
    return users.get(name)

def calculate_average(scores: Dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)
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
            // Get type of the Dict[str, int] variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 0 },
                            "end": { "line": 2, "character": 9 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type arguments for the Dict[str, int] type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeArgs".to_owned(),
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
            // Type response for Dict[str, int] variable - capture all fields
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
            // Type args response - should return [str, int] for Dict[str, int]
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([
                    {
                        "category": "$$MATCH_EVERYTHING$$",
                        "categoryFlags": "$$MATCH_EVERYTHING$$",
                        "flags": "$$MATCH_EVERYTHING$$",
                        "handle": "$$MATCH_EVERYTHING$$",
                        "moduleName": "$$MATCH_EVERYTHING$$",
                        "name": "str"
                    },
                    {
                        "category": "$$MATCH_EVERYTHING$$",
                        "categoryFlags": "$$MATCH_EVERYTHING$$",
                        "flags": "$$MATCH_EVERYTHING$$",
                        "handle": "$$MATCH_EVERYTHING$$",
                        "moduleName": "$$MATCH_EVERYTHING$$",
                        "name": "int"
                    }
                ])),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_type_args_interaction_optional() {
    // Test type argument extraction for Optional[T] types
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_args_optional_test.py");

    let test_content = r#"from typing import Optional, Union, Callable

maybe_number: Optional[int] = None
maybe_name: Optional[str] = "Alice"
maybe_callback: Optional[Callable[[int], str]] = None

def process_optional(value: Optional[int]) -> str:
    if value is not None:
        return str(value)
    return "No value"

def get_default(value: Optional[str], default: str = "unknown") -> str:
    return value if value is not None else default
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
            // Get type of the Optional[int] variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 0 },
                            "end": { "line": 2, "character": 12 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type arguments for the Optional[int] type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeArgs".to_owned(),
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
            // Type response for Optional[int] variable - capture all fields
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
            // Type args response - should return [int, None] for Optional[int] (Optional is Union[T, None])
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([
                    {
                        "category": "$$MATCH_EVERYTHING$$",
                        "categoryFlags": "$$MATCH_EVERYTHING$$",
                        "flags": "$$MATCH_EVERYTHING$$",
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "int"
                    },
                    {
                        "category": "$$MATCH_EVERYTHING$$",
                        "categoryFlags": "$$MATCH_EVERYTHING$$",
                        "flags": "$$MATCH_EVERYTHING$$",
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "None"
                    }
                ])),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_type_args_interaction_no_args() {
    // Test type argument request for non-generic types (should return empty or null)
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("no_type_args_test.py");

    let test_content = r#"simple_int: int = 42
simple_str: str = "hello"
simple_bool: bool = True

def simple_function(x: int) -> str:
    return str(x)

class SimpleClass:
    def __init__(self, value: int):
        self.value = value
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
            // Get type of the simple int variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 10 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get type arguments for the int type (should have none)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getTypeArgs".to_owned(),
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
            // Type response for int variable - capture all fields
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
            // Type args response - should return empty array or null for non-generic type
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!([])), // No type arguments for int
                error: None,
            }),
        ],
    });
}
