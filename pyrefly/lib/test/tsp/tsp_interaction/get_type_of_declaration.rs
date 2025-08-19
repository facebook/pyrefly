/*
 * TSP interaction tests for get_type_of_declaration request handler
 *
 * These tests verify the full TSP message protocol for get_type_of_declaration requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getTypeOfDeclaration
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_type_of_declaration request requires a node location pointing to a declaration
 * and returns the type information for that declaration.
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
fn test_tsp_get_type_of_declaration_interaction_variable() {
    // Test type of declaration for variable declarations
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("declaration_type_test.py");

    let test_content = r#"from typing import List, Dict

# Variable declarations with type annotations
user_name: str = "Alice"
user_age: int = 30
user_scores: List[float] = [95.5, 87.2, 92.0]
user_data: Dict[str, int] = {"math": 95, "science": 87}

# Variable declarations without explicit type annotations
inferred_number = 42
inferred_text = "Hello, World!"
inferred_list = [1, 2, 3, 4, 5]
inferred_dict = {"key1": "value1", "key2": "value2"}
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of declaration for user_name variable
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeOfDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "handle": "decl_test_handle",
                        "category": 1,
                        "flags": 1,
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 3, "character": 0 },
                                "end": { "line": 3, "character": 9 }
                            }
                        },
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "user_name",
                        "uri": file_uri.to_string()
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
            // Type of declaration response for user_name: str
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                        "aliasName": null,
                        "category": 3,
                        "categoryFlags": 0,
                        "flags": 2,
                        "handle": "$$MATCH_EVERYTHING$$",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["builtins"]
                        },
                        "name": "str"
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
fn test_tsp_get_type_of_declaration_interaction_function() {
    // Test type of declaration for function declarations
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("function_declaration_type_test.py");

    let test_content = r#"from typing import List, Optional

def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of integers."""
    return sum(numbers)

def find_item(items: List[str], target: str) -> Optional[int]:
    """Find the index of an item in a list."""
    try:
        return items.index(target)
    except ValueError:
        return None

async def fetch_data(url: str) -> Dict[str, Any]:
    """Asynchronously fetch data from a URL."""
    # Implementation would go here
    return {}

def lambda_function():
    # Lambda with inferred types
    square = lambda x: x * x
    return square
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of declaration for calculate_sum function
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeOfDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "handle": "decl_test_handle",
                        "category": 5,
                        "flags": 0,
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 2, "character": 4 },
                                "end": { "line": 2, "character": 17 }
                            }
                        },
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "calculate_sum",
                        "uri": file_uri.to_string()
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
            // Type of declaration response for function
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                        "aliasName": null,
                        "category": 1,
                        "categoryFlags": 0,
                        "flags": 4,
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "$$MATCH_EVERYTHING$$"
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
fn test_tsp_get_type_of_declaration_interaction_class() {
    // Test type of declaration for class and method declarations
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("class_declaration_type_test.py");

    let test_content = r#"from typing import Generic, TypeVar, List

T = TypeVar('T')

class DataContainer(Generic[T]):
    """A generic container for data."""
    
    def __init__(self, data: T):
        self.data: T = data
    
    def get_data(self) -> T:
        return self.data
    
    def set_data(self, new_data: T) -> None:
        self.data = new_data

class StringContainer(DataContainer[str]):
    """A specialized container for strings."""
    
    def append_text(self, text: str) -> None:
        self.data += text

# Class attribute declarations
class Config:
    DEBUG: bool = True
    MAX_CONNECTIONS: int = 100
    DEFAULT_TIMEOUT: float = 30.0
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of declaration for DataContainer class
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeOfDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "handle": "decl_test_handle",
                        "category": 2,
                        "flags": 0,
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 4, "character": 6 },
                                "end": { "line": 4, "character": 19 }
                            }
                        },
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "DataContainer",
                        "uri": file_uri.to_string()
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
            // Type of declaration response for class
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "aliasName": null,
                    "category": 3,
                    "categoryFlags": 0,
                    "flags": 1,
                    "handle": "$$MATCH_EVERYTHING$$",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["__unknown__"]
                    },
                    "name": "$$MATCH_EVERYTHING$$"
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
fn test_tsp_get_type_of_declaration_interaction_parameter() {
    // Test type of declaration for function parameter declarations
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("parameter_declaration_type_test.py");

    let test_content = r#"from typing import Union, Optional, Callable

def process_data(
    data: List[int], 
    processor: Callable[[int], str],
    multiplier: float = 1.0,
    filter_func: Optional[Callable[[int], bool]] = None
) -> List[str]:
    """Process a list of integers with various parameters."""
    filtered_data = data if filter_func is None else [x for x in data if filter_func(x)]
    return [processor(int(x * multiplier)) for x in filtered_data]

def handle_union_type(value: Union[int, str, float]) -> str:
    """Handle different types of input values."""
    return str(value)

class Calculator:
    def compute(self, x: int, y: int, operation: str = "add") -> Union[int, float]:
        """Perform computation with typed parameters."""
        if operation == "add":
            return x + y
        elif operation == "divide":
            return x / y if y != 0 else float('inf')
        else:
            return 0
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get type of declaration for 'data' parameter
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getTypeOfDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "handle": "decl_test_handle",
                        "category": 3,
                        "flags": 0,
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 3, "character": 4 },
                                "end": { "line": 3, "character": 8 }
                            }
                        },
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "data",
                        "uri": file_uri.to_string()
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
            // Type of declaration response for parameter
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "aliasName": null,
                    "category": 0,
                    "categoryFlags": 0,
                    "flags": 0,
                    "handle": "$$MATCH_EVERYTHING$$",
                    "name": "Unknown"
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
