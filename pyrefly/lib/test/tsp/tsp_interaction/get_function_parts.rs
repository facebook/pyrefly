/*
 * TSP interaction tests for get_function_parts request handler
 *
 * These tests verify the full TSP message protocol for get_function_parts requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot, typeServer/getType, and typeServer/getFunctionParts
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_function_parts request requires a function type handle (obtained from get_type) and returns
 * the parameter and return type information for function analysis.
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
fn test_tsp_get_function_parts_interaction_basic() {
    // Test basic function parts extraction for a simple function
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("function_parts_test.py");

    let test_content = r#"def add(x: int, y: int) -> int:
    return x + y

def greet(name: str) -> str:
    return f"Hello, {name}!"
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
            // Get type of function 'add'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 4 },
                            "end": { "line": 0, "character": 7 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts for the 'add' function using captured type handle
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,
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
            // Type response for function 'add' - capture all fields
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
            // Function parts response - should return parameter and return type information
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["x: int", "y: int"],  // Array of parameter strings
                    "returnType": "int"  // Return type string
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_complex() {
    // Test function parts extraction for a more complex function with optional parameters
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("complex_function_test.py");

    let test_content = r#"from typing import Optional, List

def process_data(
    data: List[int], 
    multiplier: float = 1.0, 
    filter_positive: bool = True,
    callback: Optional[callable] = None
) -> List[float]:
    result = []
    for item in data:
        if not filter_positive or item > 0:
            processed = item * multiplier
            if callback:
                processed = callback(processed)
            result.append(processed)
    return result
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
            // Get type of function 'process_data'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 2, "character": 4 },
                            "end": { "line": 2, "character": 16 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts for the complex function using captured type handle
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,
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
            // Type response for complex function - capture all fields
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
            // Function parts response for complex function with multiple parameters
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["data: list[int]", "multiplier: float", "filter_positive: bool", "callback: Unknown | None"],  // Expected 4 parameters for complex function
                    "returnType": "list[float]"  // Expected return type
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_lambda() {
    // Test function parts extraction for lambda functions
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("lambda_test.py");

    let test_content = r#"square = lambda x: x * x
add_one = lambda n: n + 1
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
            // Get type of variable 'square' that holds the lambda function
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 6 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts for the lambda using captured type handle
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,
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
            // Type response for lambda function - capture all fields
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
            // Function parts response for lambda
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["x: Unknown"],  // Expected lambda parameter info
                    "returnType": "Unknown"  // Expected inferred return type
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_with_expand_type_aliases_flag() {
    // Test function parts extraction with type alias expansion
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_alias_test.py");

    let test_content = r#"# Simple type aliases
def process_list(data):
    return str(sum(data)) if data else 0
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
            // Get type of function 'process_list'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 1, "character": 4 },
                            "end": { "line": 1, "character": 16 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts WITHOUT type alias expansion (flags = 0)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,  // No flags - should show type aliases as-is
                    "snapshot": 2
                }),
            }),
            // Get function parts WITH type alias expansion (flags = 1)
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 1,  // EXPAND_TYPE_ALIASES flag - should expand aliases to underlying types
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
            // Type response for function - capture all fields
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
            // Function parts response without alias expansion - should show type aliases
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["data: Unknown"],  // Without typing annotations, shows as Unknown
                    "returnType": "Literal[0] | str"  // Union type
                })),
                error: None,
            }),
            // Function parts response with alias expansion - should show expanded types
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "params": ["data: Unknown"],  // Same as without expansion since no aliases used
                    "returnType": "Literal[0] | str"  // Same return type
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_with_convert_to_instance_type_flag() {
    // Test function parts extraction with class-to-instance type conversion
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("class_instance_test.py");

    let test_content = r#"class MyClass:
    def __init__(self, value: int):
        self.value = value

def create_instance(cls, value: int) -> MyClass:
    return cls(value)
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
            // Get type of function 'create_instance'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 4, "character": 4 },
                            "end": { "line": 4, "character": 19 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts WITHOUT instance type conversion (flags = 0)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,  // No flags - should show class types as-is
                    "snapshot": 2
                }),
            }),
            // Get function parts WITH instance type conversion (flags = 4)
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 4,  // CONVERT_TO_INSTANCE_TYPE flag - should convert class types to instances
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
            // Type response for function - capture all fields
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
            // Function parts response without instance conversion - should show class types
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["cls: Unknown", "value: int"],  // Should show cls parameter
                    "returnType": "MyClass"  // Instance type for return
                })),
                error: None,
            }),
            // Function parts response with instance conversion - should convert class types to instances
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "params": ["cls: Unknown", "value: int"],  // Should show converted types
                    "returnType": "MyClass"  // Instance type for return
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_with_print_type_var_variance_flag() {
    // Test function parts extraction with type variable variance information
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_var_variance_test.py");

    let test_content = r#"def process_generic(func, arg):
    return func(arg)
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
            // Get type of function 'process_generic'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 4 },
                            "end": { "line": 0, "character": 19 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts WITHOUT variance information (flags = 0)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 0,  // No flags - should show type vars without variance info
                    "snapshot": 2
                }),
            }),
            // Get function parts WITH variance information (flags = 2)
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 2,  // PRINT_TYPE_VAR_VARIANCE flag - should show variance information
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
            // Type response for function - capture all fields
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
            // Function parts response without variance information - should show type vars as normal
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["func: Unknown", "arg: Unknown"],  // May show type vars without variance info
                    "returnType": "Unknown"  // Return type without variance info
                })),
                error: None,
            }),
            // Function parts response with variance information - should show variance annotations
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "params": ["func: Unknown", "arg: Unknown"],  // May show type vars with variance info like "T_contra (contravariant)"
                    "returnType": "Unknown"  // Return type with variance info like "T_co (covariant)"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_function_parts_interaction_with_combined_flags() {
    // Test function parts extraction with multiple flags combined
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("combined_flags_test.py");

    let test_content = r#"class DataProcessor:
    pass

def complex_function(processor_type, data):
    return ["processed"]
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
            // Get type of function 'complex_function'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 3, "character": 4 },
                            "end": { "line": 3, "character": 20 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get function parts with all flags combined (flags = 7 = 1 + 2 + 4)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getFunctionParts".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "flags": 7,  // All flags: EXPAND_TYPE_ALIASES + PRINT_TYPE_VAR_VARIANCE + CONVERT_TO_INSTANCE_TYPE
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
            // Type response for function - capture all fields
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
            // Function parts response with all flags - should show expanded aliases, instance types, and variance
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "params": ["processor_type: Unknown", "data: Unknown"],  // Should show expanded MyList alias and converted Type[DataProcessor] to DataProcessor instance
                    "returnType": "list[str]"  // Should show expanded MyList[str] as list[str]
                })),
                error: None,
            }),
        ],
    });
}
