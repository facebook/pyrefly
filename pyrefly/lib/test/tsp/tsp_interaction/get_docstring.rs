/*
 * TSP interaction tests for get_docstring request handler
 *
 * These tests verify the full TSP message protocol for get_docstring requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getDocstring
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_docstring request requires a node location and returns the docstring content
 * for functions, classes, and modules.
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
fn test_tsp_get_docstring_interaction_function() {
    // Test docstring extraction for a function with docstring
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("docstring_function_test.py");

    let test_content = r#"def calculate_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle in units.
        
    Returns:
        The area of the circle in square units.
        
    Raises:
        ValueError: If radius is negative.
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 3.14159 * radius * radius
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
            // Get docstring for the function
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDocString".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 1,
                        "flags": 1,
                        "handle": "decl_function_handle",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "calculate_area",
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 4 },
                                "end": { "line": 0, "character": 18 }
                            }
                        },
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
            // Docstring response - should return the complete docstring
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::Value::String("Calculate the area of a circle.\n\nArgs:\n    radius: The radius of the circle in units.\n    \nReturns:\n    The area of the circle in square units.\n    \nRaises:\n    ValueError: If radius is negative.\n".to_owned())),
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
fn test_tsp_get_docstring_interaction_class() {
    // Test docstring extraction for a class with comprehensive documentation
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("docstring_class_test.py");

    let test_content = r#"class DataProcessor:
    """A class for processing and analyzing data.
    
    This class provides methods for loading, cleaning, and analyzing
    various types of data structures.
    
    Attributes:
        data: The raw data to be processed.
        cleaned_data: The processed and cleaned data.
        
    Example:
        processor = DataProcessor()
        processor.load_data(my_data)
        results = processor.analyze()
    """
    
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        
    def load_data(self, data):
        """Load raw data for processing."""
        self.data = data
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
            // Get docstring for the class
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDocString".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 1,
                        "flags": 1,
                        "handle": "decl_class_handle",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "DataProcessor",
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 6 },
                                "end": { "line": 0, "character": 19 }
                            }
                        },
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
            // Class docstring response
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::Value::String("A class for processing and analyzing data.\n\nThis class provides methods for loading, cleaning, and analyzing\nvarious types of data structures.\n\nAttributes:\n    data: The raw data to be processed.\n    cleaned_data: The processed and cleaned data.\n    \nExample:\n    processor = DataProcessor()\n    processor.load_data(my_data)\n    results = processor.analyze()\n".to_owned())),
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
fn test_tsp_get_docstring_interaction_method() {
    // Test docstring extraction for a method within a class
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("docstring_method_test.py");

    let test_content = r#"class Calculator:
    def add(self, x: int, y: int) -> int:
        """Add two integers together.
        
        Args:
            x: First integer to add.
            y: Second integer to add.
            
        Returns:
            The sum of x and y.
        """
        return x + y
        
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
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
            // Get docstring for the 'add' method
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDocString".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 1,
                        "flags": 1,
                        "handle": "decl_method_handle",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "add",
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 1, "character": 8 },
                                "end": { "line": 1, "character": 11 }
                            }
                        },
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
            // Method docstring response
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::Value::String("Add two integers together.\n\nArgs:\n    x: First integer to add.\n    y: Second integer to add.\n    \nReturns:\n    The sum of x and y.\n".to_owned())),
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
fn test_tsp_get_docstring_interaction_no_docstring() {
    // Test docstring request for a function without a docstring
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("no_docstring_test.py");

    let test_content = r#"def simple_function(x):
    return x * 2

class SimpleClass:
    def method(self):
        pass
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
            // Get docstring for function without docstring
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDocString".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 1,
                        "flags": 1,
                        "handle": "decl_no_docstring_handle",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["__unknown__"]
                        },
                        "name": "simple_function",
                        "node": {
                            "uri": file_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 4 },
                                "end": { "line": 0, "character": 19 }
                            }
                        },
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
            // Docstring response - should return null or empty for no docstring
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::Value::Null), // Or could be empty string
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
