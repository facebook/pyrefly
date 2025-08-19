/*
 * TSP interaction tests for get_matching_overloads request handler
 *
 * These tests verify the full TSP message protocol for get_matching_overloads requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getMatchingOverloads
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_matching_overloads request requires a node location and call arguments to return
 * only the overloads that match the provided argument types.
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
fn test_tsp_get_matching_overloads_interaction_basic_ignored() {
    // TODO: getMatchingOverloads currently returns null instead of filtering overloads
    // by argument types. This test documents the current limitation.
    // Expected behavior: Should return only the overload that matches convert(42) -> int -> str
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("matching_overloads_test.py");

    let test_content = r#"from typing import overload, Union

@overload
def convert(value: int) -> str: ...

@overload
def convert(value: str) -> int: ...

@overload
def convert(value: float) -> str: ...

def convert(value: Union[int, str, float]) -> Union[str, int]:
    """Convert between different types."""
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return int(value) if value.isdigit() else 0
    else:
        return str(value)

def test_conversions():
    result1 = convert(42)        # overload matching is unsupported - should match int -> str overload
    result2 = convert("123")     # Should match str -> int overload
    result3 = convert(3.14)      # Should match float -> str overload
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
            // Get matching overloads for convert(42) call
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getMatchingOverloads".to_owned(),
                params: serde_json::json!({
                    "callNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 19, "character": 14 },
                            "end": { "line": 19, "character": 25 }
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
            // TODO: Currently returns null instead of filtering overloads by argument types
            // Expected: Should return array with single overload matching int -> str
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
fn test_tsp_get_matching_overloads_interaction_method_ignored() {
    // TODO: getMatchingOverloads currently returns null for method calls with overloads
    // This test documents the current limitation for method overload matching.
    // Expected behavior: Should return only the overload matching compute(10, 20) -> int, int -> int
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("method_matching_overloads_test.py");

    let test_content = r#"from typing import overload, Union

class Calculator:
    @overload
    def compute(self, x: int, y: int) -> int: ...
    
    @overload
    def compute(self, x: float, y: float) -> float: ...
    
    @overload
    def compute(self, x: str, y: str) -> str: ...
    
    def compute(self, x: Union[int, float, str], y: Union[int, float, str]) -> Union[int, float, str]:
        """Compute based on argument types."""
        if isinstance(x, str) and isinstance(y, str):
            return x + y
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return x + y
        else:
            raise TypeError("Mismatched argument types")

def test_method_calls():
    calc = Calculator()
    result1 = calc.compute(10, 20)        # overload matching is unsupported - should match int, int -> int
    result2 = calc.compute(1.5, 2.5)      # Should match float, float -> float
    result3 = calc.compute("a", "b")      # Should match str, str -> str
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
            // Get matching overloads for calc.compute(10, 20)
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getMatchingOverloads".to_owned(),
                params: serde_json::json!({
                    "callNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 22, "character": 14 },
                            "end": { "line": 22, "character": 33 }
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
            // TODO: Currently returns null instead of filtering overloads by argument types
            // Expected: Should return array with single overload matching int, int -> int
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
fn test_tsp_get_matching_overloads_interaction_named_args() {
    // Test matching overloads with named arguments
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("named_args_overloads_test.py");

    let test_content = r#"from typing import overload, Optional

@overload
def process_file(path: str, encoding: str = "utf-8") -> str: ...

@overload
def process_file(path: str, binary: bool = True) -> bytes: ...

def process_file(path: str, encoding: Optional[str] = None, binary: bool = False) -> Union[str, bytes]:
    """Process a file with different options."""
    if binary:
        with open(path, 'rb') as f:
            return f.read()
    else:
        enc = encoding or "utf-8"
        with open(path, 'r', encoding=enc) as f:
            return f.read()

def test_file_processing():
    # Test with named argument
    result1 = process_file("test.txt", encoding="utf-8")
    result2 = process_file("data.bin", binary=True)
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
            // Get matching overloads for process_file with encoding argument
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getMatchingOverloads".to_owned(),
                params: serde_json::json!({
                    "callNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 18, "character": 14 },
                            "end": { "line": 18, "character": 56 }
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
            // Matching overloads response for named argument call - returns single function when found
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!([
                    {
                        "category": 1,
                        "categoryFlags": 0,
                        "flags": 4,
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "$$MATCH_EVERYTHING$$",
                        "aliasName": null
                    }
                ])),
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
fn test_tsp_get_matching_overloads_interaction_no_match() {
    // Test case where no overloads match the given arguments
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("no_match_overloads_test.py");

    let test_content = r#"from typing import overload

@overload
def strict_function(x: int) -> str: ...

@overload
def strict_function(x: str) -> int: ...

def strict_function(x):
    """Function with strict type overloads."""
    if isinstance(x, int):
        return str(x)
    elif isinstance(x, str):
        return len(x)
    else:
        raise TypeError("Unsupported type")

def test_type_mismatch():
    # This should not match any overload (float not supported)
    result = strict_function(3.14)
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
            // Get matching overloads for strict_function(3.14) - no match expected
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getMatchingOverloads".to_owned(),
                params: serde_json::json!({
                    "callNode": {
                        "uri": file_uri.to_string(),
                        "range": {
                            "start": { "line": 19, "character": 13 },
                            "end": { "line": 19, "character": 34 }
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
            // Matching overloads response - returns all available overloads even when no exact match
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!([
                    {
                        "category": 1,
                        "categoryFlags": 0,
                        "flags": 4,
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "$$MATCH_EVERYTHING$$",
                        "aliasName": null
                    },
                    {
                        "category": 1,
                        "categoryFlags": 0,
                        "flags": 4,
                        "handle": "$$MATCH_EVERYTHING$$",
                        "name": "$$MATCH_EVERYTHING$$",
                        "aliasName": null
                    }
                ])),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
