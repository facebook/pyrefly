/*
 * Integration tests for TSP search_for_type_attribute request
 *
 * These tests verify the full TSP message flow for search_for_type_attribute requests by:
 * 1. Testing class attribute search in real Python files
 * 2. Testing different access flag combinations (NONE, SKIP_INSTANCE_ATTRIBUTES, etc.)
 * 3. Testing instance vs class attribute access
 * 4. Testing method, property, and field attribute access
 * 5. Testing private and dunder attribute access
 * 6. Testing attribute inheritance from parent classes
 * 7. Testing attribute access with different expression contexts
 * 8. Testing error cases and non-existent attributes
 *
 * These integration tests complement the unit tests by verifying the actual
 * attribute resolution logic against real Python code files.
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
fn test_tsp_search_for_type_attribute_interaction_basic_class_method() {
    // Test searching for class methods in a basic class
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("class_methods_test.py");

    let test_content = r#"class MyClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
    
    def set_value(self, new_value):
        self.value = new_value

# Create an instance
instance = MyClass()
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
            // Get type of 'instance' variable to get a type handle for the class instance
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 11, "character": 0 },
                            "end": { "line": 11, "character": 8 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Search for get_value method using the instance type
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "get_value",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
                    "snapshot": 2
                }),
            }),
            // Search for non-existent method
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "nonexistent_method",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
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
            // Type response for 'instance' variable - capture handle
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
            // Search response for get_value method - should find it
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "name": "get_value",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
            // Search response for non-existent method - should return null
            Message::Response(Response {
                id: RequestId::from(5),
                result: None,
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_search_for_type_attribute_interaction_access_flags() {
    // Test different access flag combinations
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("access_flags_test.py");

    let test_content = r#"class TestClass:
    class_var = "I am a class variable"
    
    def __init__(self):
        self.instance_var = "I am an instance variable"
    
    @classmethod
    def class_method(cls):
        return cls.class_var
    
    @staticmethod
    def static_method():
        return "static"

# Test access flags
test_instance = TestClass()
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
            // Get type of 'test_instance' variable to get a type handle
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 15, "character": 0 },
                            "end": { "line": 15, "character": 13 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Search with NONE flags (should find both instance and class attributes)
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "class_var",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
                    "snapshot": 2
                }),
            }),
            // Search with SKIP_INSTANCE_ATTRIBUTES flag (should still find class attributes)
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "class_var",
                    "accessFlags": 1,
                    "expressionNode": null,
                    "instanceType": null,
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
            // Type response for 'test_instance' variable - capture handle
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
            // Search response with no flags - should find class_var
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "name": "class_var",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
            // Search response with SKIP_INSTANCE_ATTRIBUTES - should still find class_var
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "name": "class_var",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_search_for_type_attribute_interaction_dunder_methods() {
    // Test searching for dunder methods
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("dunder_methods_test.py");

    let test_content = r#"class DunderClass:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"DunderClass({self.value})"

# Test dunder methods
dunder_instance = DunderClass(42)
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
            // Get type of 'dunder_instance' variable to get a type handle
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 11, "character": 0 },
                            "end": { "line": 11, "character": 15 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Search for __init__ method
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "__init__",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
                    "snapshot": 2
                }),
            }),
            // Search for __str__ method
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "__str__",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
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
            // Type response for 'dunder_instance' variable - capture handle
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
            // Search response for __init__ method - should find it
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "name": "__init__",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
            // Search response for __str__ method - should find it
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "name": "__str__",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_search_for_type_attribute_interaction_inheritance() {
    // Test attribute search with inheritance
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("inheritance_test.py");

    let test_content = r#"class BaseClass:
    def base_method(self):
        return "base"
    
    def override_me(self):
        return "base version"

class DerivedClass(BaseClass):
    def derived_method(self):
        return "derived"
    
    def override_me(self):
        return "derived version"

# Test inheritance
derived_instance = DerivedClass()
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
            // Get type of 'derived_instance' variable to get a type handle
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 15, "character": 0 },
                            "end": { "line": 15, "character": 16 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Search for inherited method from base class
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "base_method",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
                    "snapshot": 2
                }),
            }),
            // Search for method defined in derived class
            Message::from(Request {
                id: RequestId::from(5),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "derived_method",
                    "accessFlags": 0,
                    "expressionNode": null,
                    "instanceType": null,
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
            // Type response for 'derived_instance' variable - capture handle
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
            // Search response for inherited base_method - should find it
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "name": "base_method",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
            // Search response for derived_method - should find it
            Message::Response(Response {
                id: RequestId::from(5),
                result: Some(serde_json::json!({
                    "name": "derived_method",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
        ],
    });
}

#[test]
fn test_tsp_search_for_type_attribute_interaction_with_expression_context() {
    // Test attribute search with expression context provided
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("expression_context_test.py");

    let test_content = r#"class ContextClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

# Test with expression context
context_instance = ContextClass()
result = context_instance.get_value()
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
            // Get type of 'context_instance' variable to get a type handle
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getType".to_owned(),
                params: serde_json::json!({
                    "node": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 8, "character": 0 },
                            "end": { "line": 8, "character": 16 }
                        }
                    },
                    "snapshot": 2
                }),
            }),
            // Search with expression context provided
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/searchForTypeAttribute".to_owned(),
                params: serde_json::json!({
                    "startType": {
                        "aliasName": "$$TYPE_ALIAS_NAME$$",
                        "category": "$$TYPE_CATEGORY$$",
                        "categoryFlags": "$$TYPE_CATEGORY_FLAGS$$",
                        "flags": "$$TYPE_FLAGS$$",
                        "handle": "$$TYPE_HANDLE$$",
                        "name": "$$TYPE_NAME$$"
                    },
                    "attributeName": "get_value",
                    "accessFlags": 0,
                    "expressionNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 9, "character": 9 },
                            "end": { "line": 9, "character": 38 }
                        }
                    },
                    "instanceType": null,
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
            // Type response for 'context_instance' variable - capture handle
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
            // Search response with expression context - should find get_value
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "name": "get_value",
                    "type": "$$MATCH_EVERYTHING$$",
                    "flags": "$$MATCH_EVERYTHING$$",
                    "decls": "$$MATCH_EVERYTHING$$"
                })),
                error: None,
            }),
        ],
    });
}
