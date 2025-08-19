/*
 * TSP interaction tests for get_diagnostics request handler
 *
 * These tests verify the full TSP message protocol for get_diagnostics requests by:
 * 1. Following the TSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getDiagnostics
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The get_diagnostics request requires a URI and snapshot and returns diagnostic information
 * for type errors, syntax errors, and other issues in the specified file.
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
use crate::test::tsp::tsp_interaction::util::run_test_tsp;

#[test]
fn test_tsp_get_diagnostics_interaction_valid_file() {
    // Test getDiagnostics for a valid Python file without errors
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("valid_file.py");

    let test_content = r#"def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

x = 42
y = "world"
result = greet(y)
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
            // Get diagnostics for the file
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDiagnostics".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
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
            // Diagnostics response - should be empty for valid file
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!([])),
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
fn test_tsp_get_diagnostics_interaction_with_type_error() {
    // Test getDiagnostics for a file containing type errors
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("type_error_file.py");

    let test_content = r#"def add_numbers(x: int, y: int) -> int:
    return x + y

# This should cause a type error
result = add_numbers("hello", "world")
number = 5 + "string"
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
            // Get diagnostics for the file with type errors
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDiagnostics".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
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
            // Diagnostics response - should contain type errors (exact content validated in unit tests)
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!([
                    {
                        "code": "$$MATCH_EVERYTHING$$",
                        "message": "$$MATCH_EVERYTHING$$",
                        "range": "$$MATCH_EVERYTHING$$",
                        "severity": "$$MATCH_EVERYTHING$$",
                        "source": "$$MATCH_EVERYTHING$$"
                    },
                    {
                        "code": "$$MATCH_EVERYTHING$$",
                        "message": "$$MATCH_EVERYTHING$$",
                        "range": "$$MATCH_EVERYTHING$$",
                        "severity": "$$MATCH_EVERYTHING$$",
                        "source": "$$MATCH_EVERYTHING$$"
                    },
                    {
                        "code": "$$MATCH_EVERYTHING$$",
                        "message": "$$MATCH_EVERYTHING$$",
                        "range": "$$MATCH_EVERYTHING$$",
                        "severity": "$$MATCH_EVERYTHING$$",
                        "source": "$$MATCH_EVERYTHING$$"
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
fn test_tsp_get_diagnostics_interaction_invalid_uri() {
    // Test getDiagnostics with an invalid URI that cannot be converted to a file path
    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Get snapshot first
            Message::from(Request {
                id: RequestId::from(1),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Try to get diagnostics for an invalid URI
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getDiagnostics".to_owned(),
                params: serde_json::json!({
                    "uri": "http://example.com/not/a/file.py",
                    "snapshot": 1
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!(1)),
                error: None,
            }),
            // Diagnostics response - should return an error for invalid URI
            Message::Response(Response {
                id: RequestId::from(2),
                result: None,
                error: Some(lsp_server::ResponseError {
                    code: lsp_server::ErrorCode::InvalidParams as i32,
                    message: "Invalid URI - cannot convert to file path".to_owned(),
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
fn test_tsp_get_diagnostics_interaction_outdated_snapshot() {
    // Test getDiagnostics with an outdated snapshot
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("snapshot_test.py");

    let test_content = r#"x = 1
y = 2
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
            // Try to get diagnostics with an outdated snapshot (snapshot 1 instead of 2)
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getDiagnostics".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
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
            // Diagnostics response - should return an error for outdated snapshot
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
fn test_tsp_get_diagnostics_interaction_nonexistent_file() {
    // Test getDiagnostics for a nonexistent file
    let temp_dir = TempDir::new().unwrap();
    let nonexistent_file_path = temp_dir.path().join("nonexistent.py");
    let file_uri = Url::from_file_path(&nonexistent_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(1),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Try to get diagnostics for nonexistent file
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getDiagnostics".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
                    "snapshot": 1
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!(1)),
                error: None,
            }),
            // Diagnostics response - should return empty diagnostics for nonexistent file
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!([])),
                error: None,
            }),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: None,
        configuration: false,
        file_watch: false,
    });
}
