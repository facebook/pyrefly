/*
 * TSP interaction tests for resolve_import request handler
 *
 * These tests verify the full TSP message protocol for resolve_import requests by:
 * 1. Following the LSP interaction test pattern using run_test_tsp
 * 2. Testing complete request/response flows including typeServer/getSnapshot and typeServer/resolveImport
 * 3. Validating proper snapshot management and protocol sequencing
 * 4. Using real file operations and message passing to simulate end-to-end TSP interactions
 *
 * The resolve_import request requires a sourceUri and moduleDescriptor
 * and returns the resolved import location or null if the import cannot be resolved.
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
fn test_tsp_resolve_import_interaction_basic() {
    // Test import resolution for standard library modules
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("import_resolution_test.py");

    let test_content = r#"import os
x = os.path.join('a', 'b')
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
            // Resolve import for 'import os'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImport".to_owned(),
                params: serde_json::json!({
                    "sourceUri": file_uri.to_string(),
                    "moduleDescriptor": {
                        "leadingDots": 0,
                        "nameParts": ["os"]
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
            // Import resolution response for os module
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!("file://$$MATCH_EVERYTHING$$")), // Should end with "pyrefly_bundled_typeshed/os/__init__.pyi"
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
fn test_tsp_resolve_import_interaction_from_import() {
    // Test import resolution for 'from ... import' statements
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("from_import_test.py");

    let test_content = r#"from typing import List
x: List[str] = []
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
            // Resolve import for 'typing' module
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImport".to_owned(),
                params: serde_json::json!({
                    "sourceUri": file_uri.to_string(),
                    "moduleDescriptor": {
                        "leadingDots": 0,
                        "nameParts": ["typing"]
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
            // Import resolution response for typing module
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!("file://$$MATCH_EVERYTHING$$")), // Should end with "pyrefly_bundled_typeshed/typing.pyi"
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
fn test_tsp_resolve_import_interaction_relative() {
    // Test import resolution for relative imports
    let temp_dir = TempDir::new().unwrap();

    // Create a package structure
    let package_dir = temp_dir.path().join("mypackage");
    std::fs::create_dir_all(&package_dir).unwrap();

    // Create __init__.py
    let init_file = package_dir.join("__init__.py");
    std::fs::write(&init_file, "# Package init").unwrap();

    // Create utils.py module
    let utils_file = package_dir.join("utils.py");
    std::fs::write(
        &utils_file,
        r#"def utility_function():
    return "utility"
"#,
    )
    .unwrap();

    // Create main test file with relative imports
    let test_file_path = package_dir.join("main.py");
    let test_content = r#"from .utils import utility_function
result = utility_function()
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
            // Resolve relative import '.utils'
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImport".to_owned(),
                params: serde_json::json!({
                    "sourceUri": file_uri.to_string(),
                    "moduleDescriptor": {
                        "leadingDots": 1,
                        "nameParts": ["utils"]
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
            // Import resolution response for relative import
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!("file://$$MATCH_EVERYTHING$$")), // Should end with "mypackage/utils.py"
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
fn test_tsp_resolve_import_interaction_unresolved() {
    // Test import resolution for imports that cannot be resolved
    let temp_dir = TempDir::new().unwrap();
    let test_file_path = temp_dir.path().join("unresolved_import_test.py");

    let test_content = r#"import nonexistent_module
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
            // Try to resolve nonexistent import
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImport".to_owned(),
                params: serde_json::json!({
                    "sourceUri": file_uri.to_string(),
                    "moduleDescriptor": {
                        "leadingDots": 0,
                        "nameParts": ["nonexistent_module"]
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
            // Import resolution response for unresolved import (should be null)
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
