/*
 * TSP interaction tests for getSymbolsForFile request handler
 *
 * These tests verify the full TSP message protocol for getSymbolsForFile requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/getSymbolsForFile
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The getSymbolsForFile request returns all symbols in a specific file.
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
fn test_tsp_get_symbols_for_file_interaction_basic() {
    // Test basic symbol lookup functionality through the TSP protocol
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("symbols.py");

    let test_content = r#"# Test file for symbol tests
def my_function(param: int) -> str:
    return str(param)

class MyClass:
    def __init__(self):
        self.value = 42

x = 10
"#;

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(1),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get symbols for file
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSymbolsForFile".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
                    "snapshot": 2
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Response to getSnapshot
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!(2)),
                error: None,
            }),
            // Response to getSymbolsForFile - should contain symbols
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!({
                    "uri": "$$MATCH_EVERYTHING$$",
                    "symbols": [
                        // We expect symbols for the function, class, and variable
                        {
                            "node": {
                                "uri": "$$MATCH_EVERYTHING$$",
                                "range": {
                                    "start": {"line": 1, "character": 4},
                                    "end": {"line": 1, "character": 15}
                                }
                            },
                            "name": "my_function",
                            "decls": [
                                {
                                    "handle": "$$MATCH_EVERYTHING$$",
                                    "category": 5, // FUNCTION
                                    "flags": 0,
                                    "node": {
                                        "uri": "$$MATCH_EVERYTHING$$",
                                        "range": {
                                            "start": {"line": 1, "character": 4},
                                            "end": {"line": 1, "character": 15}
                                        }
                                    },
                                    "moduleName": {
                                        "leadingDots": 0,
                                        "nameParts": ["$$MATCH_EVERYTHING$$"]
                                    },
                                    "name": "my_function",
                                    "uri": "$$MATCH_EVERYTHING$$"
                                }
                            ],
                            "synthesizedTypes": []
                        }
                        // More symbols could be here...
                    ]
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
fn test_tsp_get_symbols_for_file_interaction_empty_file() {
    // Test with an empty Python file
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("empty.py");

    let test_content = "# Empty file\n";

    std::fs::write(&test_file_path, test_content).unwrap();
    let file_uri = Url::from_file_path(&test_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the test file
            Message::from(build_did_open_notification(test_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(1),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Get symbols for file
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSymbolsForFile".to_owned(),
                params: serde_json::json!({
                    "uri": file_uri.to_string(),
                    "snapshot": 2
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Response to getSnapshot
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!(2)),
                error: None,
            }),
            // Response to getSymbolsForFile - should be empty or minimal symbols
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!({
                    "uri": "$$MATCH_EVERYTHING$$",
                    "symbols": []
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
