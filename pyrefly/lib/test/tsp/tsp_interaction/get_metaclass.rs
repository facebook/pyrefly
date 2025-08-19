/*
 * TSP interaction tests for get_metaclass request handler
 *
 * These tests verify the full TSP message protocol for get_metaclass requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp_with_capture
 * 2. Testing complete request/response flows including typeServer/getSnapshot, typeServer/getType, and typeServer/getMetaclass
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_metaclass request requires a type handle (obtained from get_type) and returns
 * the metaclass of a given class type. For most classes, this returns `type` (the default metaclass),
 * but for classes with custom metaclasses, it returns the specific metaclass.
 */

use lsp_server::ErrorCode;
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
fn test_tsp_get_metaclass_interaction_default_metaclass() {
    // Test get_metaclass for a regular class (should return the default 'type' metaclass)
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("default_metaclass_test.py");

    let test_content = r#"class RegularClass:
    """A regular class with the default metaclass."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name

# Create an instance to get the class type
instance = RegularClass("test")
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
            // Get type of the instance variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 10, "character": 0 },  // "instance" variable
                            "end": { "line": 10, "character": 8 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get metaclass for the instance type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getMetaclass".to_owned(),
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
            // Type response for RegularClass - capture all fields
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
            // Metaclass response - should be the default 'type' metaclass
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": "$$MATCH_EVERYTHING$$",
                    "categoryFlags": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["builtins"]
                    },
                    "name": "type"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_metaclass_interaction_custom_metaclass() {
    // Test get_metaclass for a class with a custom metaclass
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("custom_metaclass_test.py");

    let test_content = r#"class CustomMeta(type):
    """A custom metaclass."""
    
    def __new__(cls, name, bases, namespace):
        namespace['custom_attribute'] = True
        return super().__new__(cls, name, bases, namespace)

class CustomClass(metaclass=CustomMeta):
    """A class with a custom metaclass."""
    
    def __init__(self, value: int):
        self.value = value
    
    def get_value(self) -> int:
        return self.value

# Create an instance to work with
instance = CustomClass(42)
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
            // Get type of the instance variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 17, "character": 0 },  // "instance" variable
                            "end": { "line": 17, "character": 8 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get metaclass for the instance type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getMetaclass".to_owned(),
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
            // Type response for CustomClass - capture all fields
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
            // Metaclass response - should be the custom CustomMeta metaclass
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": "$$MATCH_EVERYTHING$$",
                    "categoryFlags": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": "$$MATCH_EVERYTHING$$",
                    "name": "CustomMeta"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_metaclass_interaction_inherited_metaclass() {
    // Test get_metaclass for a class that inherits from a class with a custom metaclass
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("inherited_metaclass_test.py");

    let test_content = r#"class Meta(type):
    """Base metaclass."""
    pass

class SubMeta(Meta):
    """Derived metaclass."""
    pass

class BaseWithMeta(metaclass=Meta):
    """Base class with Meta metaclass."""
    pass

class DerivedClass(BaseWithMeta):
    """Class inheriting from BaseWithMeta - should inherit Meta metaclass."""
    
    def __init__(self, name: str):
        self.name = name

instance = DerivedClass("test")
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
            // Get type of the instance variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 18, "character": 0 },  // "instance" variable
                            "end": { "line": 18, "character": 8 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Get metaclass for the instance type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getMetaclass".to_owned(),
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
            // Type response for DerivedClass - capture all fields
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
            // Metaclass response - should be the inherited Meta metaclass
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": "$$MATCH_EVERYTHING$$",
                    "categoryFlags": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": "$$MATCH_EVERYTHING$$",
                    "name": "Meta"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_metaclass_interaction_non_class_type() {
    // Test get_metaclass for a non-class type (should return None/null)
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("non_class_test.py");

    let test_content = r#"# Test with primitive types and functions
number = 42
text = "hello"

def some_function() -> str:
    return "test"

# These are not class types, so getting their metaclass should return None
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
            // Get type of the number variable (should be int literal or int)
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 1, "character": 0 },  // "number" variable
                            "end": { "line": 1, "character": 6 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Try to get metaclass for the int type (should return None)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getMetaclass".to_owned(),
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
            // Type response for number variable - capture all fields
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
            // Metaclass response - should be None/null for non-class types
            Message::Response(Response {
                id: RequestId::from(4),
                result: None,
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_get_metaclass_interaction_invalid_snapshot() {
    // Test get_metaclass with an invalid/outdated snapshot
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("snapshot_test.py");

    let test_content = r#"class TestClass:
    pass
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
            // Get type of the TestClass
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 6 },  // "TestClass" in class definition
                            "end": { "line": 0, "character": 15 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Try to get metaclass with an outdated snapshot
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/getMetaclass".to_owned(),
                params: serde_json::json!({
                    "type": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "snapshot": 1  // Outdated snapshot
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
            // Type response for TestClass - capture all fields
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
            // Metaclass response - should return an error for outdated snapshot
            Message::Response(Response {
                id: RequestId::from(4),
                result: None,
                error: Some(lsp_server::ResponseError {
                    code: ErrorCode::ServerCancelled as i32,
                    message: "Snapshot outdated".to_owned(),
                    data: None,
                }),
            }),
        ],
    });
}
