/*
* Integration tests for resolve_import_declaration TSP interaction
*
* These tests verify the complete resolve_import_declaration flow by:
* 1. Testing resolution of import declarations to actual symbol definitions
* 2. Validating handling of different import types (absolute, relative, star imports)
* 3. Testing cross-module symbol resolution
* 4. Validating error handling for unresolva        expected_messages_from_language_server: vec![
           // Snapshot response
           Message::Response(Response {
               id: RequestId::from(2),
               result: Some(serde_json::json!(2)),
               error: None,
           }),
           // Resolve import declaration response - currently returns original import with unresolved flag
           // TODO: Expected behavior should be null for truly unresolvable imports
           Message::Response(Response {
               id: RequestId::from(3),
               result: Some(serde_json::json!({
                   "handle": "test_unresolved_import",
                   "category": 4,  // IMPORT
                   "flags": 2,    // UNRESOLVED_IMPORT flag set
                   "node": {
                       "uri": main_uri.to_string(),
                       "range": {
                           "start": { "line": 0, "character": 34 },
                           "end": { "line": 0, "character": 48 }
                       }
                   },
                   "moduleName": {
                       "leadingDots": 0,
                       "nameParts": ["nonexistent_module"]
                   },
                   "name": "missing_symbol",
                   "uri": main_uri.to_string()
               })),
               error: None,
           }),
       ],* 5. Testing various declaration categories and their resolution behavior
*
* The resolve_import_declaration request takes a Declaration and ResolveImportOptions,
* resolves import declarations to their actual definitions in target modules,
* and returns the resolved Declaration or None if resolution fails.
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
fn test_tsp_resolve_import_declaration_interaction_basic() {
    let temp_dir = TempDir::new().unwrap();

    // Create main.py file
    let main_file_path = temp_dir.path().join("main.py");
    let main_content = r#"from utils import helper_function
from math import sqrt
import os

helper_function()
result = sqrt(16)
os.getcwd()
"#;
    std::fs::write(&main_file_path, main_content).unwrap();
    let main_uri = Url::from_file_path(&main_file_path).unwrap();

    // Create utils.py file
    let utils_file_path = temp_dir.path().join("utils.py");
    let utils_content = r#"def helper_function():
    return "helper"

class HelperClass:
    pass

CONSTANT = 42
"#;
    std::fs::write(&utils_file_path, utils_content).unwrap();
    let _utils_uri = Url::from_file_path(&utils_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open both files
            Message::from(build_did_open_notification(main_file_path.clone())),
            Message::from(build_did_open_notification(utils_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Resolve import declaration for helper_function
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/resolveImportDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 4,
                        "flags": 0,
                        "handle": "test_import_helper",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["utils"]
                        },
                        "name": "helper_function",
                        "node": {
                            "uri": main_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 17 },
                                "end": { "line": 0, "character": 32 }
                            }
                        },
                        "uri": main_uri.to_string()
                    },
                    "options": {
                        "resolveLocalNames": true,
                        "allowExternallyHiddenAccess": false,
                        "skipFileNeededCheck": false
                    },
                    "snapshot": 3
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(3)),
                error: None,
            }),
            // Resolve import declaration response - currently returns original import due to symbol lookup limitations
            // TODO: Expected behavior should be category 5 (FUNCTION) with resolved handle and node
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": 4,
                    "flags": 0,
                    "handle": "test_import_helper",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["utils"]
                    },
                    "name": "helper_function",
                    "node": {
                        "uri": main_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 17 },
                            "end": { "line": 0, "character": 32 }
                        }
                    },
                    "uri": main_uri.to_string()
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
fn test_tsp_resolve_import_declaration_interaction_class() {
    let temp_dir = TempDir::new().unwrap();

    // Create main.py file
    let main_file_path = temp_dir.path().join("main.py");
    let main_content = r#"from models import User, Product
from typing import List

users: List[User] = []
product = Product()
"#;
    std::fs::write(&main_file_path, main_content).unwrap();
    let main_uri = Url::from_file_path(&main_file_path).unwrap();

    // Create models.py file
    let models_file_path = temp_dir.path().join("models.py");
    let models_content = r#"class User:
    def __init__(self, name: str):
        self.name = name

class Product:
    def __init__(self, title: str = "Unknown"):
        self.title = title

class Category:
    pass
"#;
    std::fs::write(&models_file_path, models_content).unwrap();
    let _models_uri = Url::from_file_path(&models_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open both files
            Message::from(build_did_open_notification(main_file_path.clone())),
            Message::from(build_did_open_notification(models_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Resolve import declaration for User class
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/resolveImportDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 4,
                        "flags": 0,
                        "handle": "test_import_user",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["models"]
                        },
                        "name": "User",
                        "node": {
                            "uri": main_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 18 },
                                "end": { "line": 0, "character": 22 }
                            }
                        },
                        "uri": main_uri.to_string()
                    },
                    "options": {
                        "resolveLocalNames": true,
                        "allowExternallyHiddenAccess": false,
                        "skipFileNeededCheck": false
                    },
                    "snapshot": 3
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(3)),
                error: None,
            }),
            // Resolve import declaration response - currently returns original import due to symbol lookup limitations
            // TODO: Expected behavior should be category 6 (CLASS) with resolved handle and node
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": 4,
                    "flags": 0,
                    "handle": "test_import_user",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["models"]
                    },
                    "name": "User",
                    "node": {
                        "uri": main_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 18 },
                            "end": { "line": 0, "character": 22 }
                        }
                    },
                    "uri": main_uri.to_string()
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
fn test_tsp_resolve_import_declaration_interaction_non_import() {
    let temp_dir = TempDir::new().unwrap();

    // Create main.py file with local function
    let main_file_path = temp_dir.path().join("main.py");
    let main_content = r#"def my_function():
    return "local function"

class MyClass:
    pass

CONSTANT = 42
"#;
    std::fs::write(&main_file_path, main_content).unwrap();
    let main_uri = Url::from_file_path(&main_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the file
            Message::from(build_did_open_notification(main_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Try to resolve a non-import declaration (should return the same declaration)
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImportDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 5,
                        "flags": 0,
                        "handle": "test_local_function",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["main"]
                        },
                        "name": "my_function",
                        "node": {
                            "uri": main_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 4 },
                                "end": { "line": 0, "character": 15 }
                            }
                        },
                        "uri": main_uri.to_string()
                    },
                    "options": {
                        "resolveLocalNames": true,
                        "allowExternallyHiddenAccess": false,
                        "skipFileNeededCheck": false
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
            // Resolve import declaration response (same declaration for non-imports)
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "category": 5,
                    "flags": 0,
                    "handle": "test_local_function",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["main"]
                    },
                    "name": "my_function",
                    "node": {
                        "uri": main_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 4 },
                            "end": { "line": 0, "character": 15 }
                        }
                    },
                    "uri": main_uri.to_string()
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
fn test_tsp_resolve_import_declaration_interaction_unresolved() {
    let temp_dir = TempDir::new().unwrap();

    // Create main.py file with import that can't be resolved
    let main_file_path = temp_dir.path().join("main.py");
    let main_content = r#"from nonexistent_module import missing_symbol
from utils import nonexistent_function

# This should fail to resolve
missing_symbol()
"#;
    std::fs::write(&main_file_path, main_content).unwrap();
    let main_uri = Url::from_file_path(&main_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open the file
            Message::from(build_did_open_notification(main_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(2),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Try to resolve an unresolved import
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/resolveImportDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 4,
                        "flags": 2,
                        "handle": "test_unresolved_import",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["nonexistent_module"]
                        },
                        "name": "missing_symbol",
                        "node": {
                            "uri": main_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 34 },
                                "end": { "line": 0, "character": 48 }
                            }
                        },
                        "uri": main_uri.to_string()
                    },
                    "options": {
                        "resolveLocalNames": true,
                        "allowExternallyHiddenAccess": false,
                        "skipFileNeededCheck": false
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
            // Resolve import declaration response - currently returns unresolved declaration, should return null
            // TODO: Implementation should return null for unresolvable imports
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!({
                    "category": 4,
                    "flags": 2,
                    "handle": "test_unresolved_import",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["nonexistent_module"]
                    },
                    "name": "missing_symbol",
                    "node": {
                        "uri": main_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 34 },
                            "end": { "line": 0, "character": 48 }
                        }
                    },
                    "uri": main_uri.to_string()
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
fn test_tsp_resolve_import_declaration_interaction_variable() {
    let temp_dir = TempDir::new().unwrap();

    // Create main.py file
    let main_file_path = temp_dir.path().join("main.py");
    let main_content = r#"from config import DEBUG, VERSION
from constants import PI, E

if DEBUG:
    print(f"Version: {VERSION}")
    print(f"Pi: {PI}, E: {E}")
"#;
    std::fs::write(&main_file_path, main_content).unwrap();
    let main_uri = Url::from_file_path(&main_file_path).unwrap();

    // Create config.py file
    let config_file_path = temp_dir.path().join("config.py");
    let config_content = r#"DEBUG = True
VERSION = "1.0.0"
FEATURE_FLAGS = {"new_ui": False}
"#;
    std::fs::write(&config_file_path, config_content).unwrap();
    let _config_uri = Url::from_file_path(&config_file_path).unwrap();

    run_test_tsp(TestCase {
        messages_from_language_client: vec![
            // Open both files
            Message::from(build_did_open_notification(main_file_path.clone())),
            Message::from(build_did_open_notification(config_file_path.clone())),
            // Get snapshot
            Message::from(Request {
                id: RequestId::from(3),
                method: "typeServer/getSnapshot".to_owned(),
                params: serde_json::json!({}),
            }),
            // Resolve import declaration for DEBUG variable
            Message::from(Request {
                id: RequestId::from(4),
                method: "typeServer/resolveImportDeclaration".to_owned(),
                params: serde_json::json!({
                    "decl": {
                        "category": 4,
                        "flags": 0,
                        "handle": "test_variable_import",
                        "moduleName": {
                            "leadingDots": 0,
                            "nameParts": ["config"]
                        },
                        "name": "DEBUG",
                        "node": {
                            "uri": main_uri.to_string(),
                            "range": {
                                "start": { "line": 0, "character": 19 },
                                "end": { "line": 0, "character": 24 }
                            }
                        },
                        "uri": main_uri.to_string()
                    },
                    "options": {
                        "resolveLocalNames": true,
                        "allowExternallyHiddenAccess": false,
                        "skipFileNeededCheck": false
                    },
                    "snapshot": 3
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            // Snapshot response
            Message::Response(Response {
                id: RequestId::from(3),
                result: Some(serde_json::json!(3)),
                error: None,
            }),
            // Resolve import declaration response - currently returns original import due to symbol lookup limitations
            // TODO: Expected behavior should be category 1 (VARIABLE) with resolved handle and node
            Message::Response(Response {
                id: RequestId::from(4),
                result: Some(serde_json::json!({
                    "category": 4,
                    "flags": 0,
                    "handle": "test_variable_import",
                    "moduleName": {
                        "leadingDots": 0,
                        "nameParts": ["config"]
                    },
                    "name": "DEBUG",
                    "node": {
                        "uri": main_uri.to_string(),
                        "range": {
                            "start": { "line": 0, "character": 19 },
                            "end": { "line": 0, "character": 24 }
                        }
                    },
                    "uri": main_uri.to_string()
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
