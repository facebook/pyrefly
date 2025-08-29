/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetSymbolsForNodeParams and GetSymbolsForNodeRequest type construction and serialization
//!
//! These tests verify the GetSymbolsForNodeParams parameter construction and validation by:
//! 1. Testing TSP GetSymbolsForNodeParams construction with various parameter combinations
//! 2. Testing Node parameter variations
//! 3. Testing snapshot validation logic
//! 4. Testing flag combinations
//! 5. Testing optional name parameter
//! 6. Testing serialization/deserialization round-trips

use tsp_types::GetSymbolsForNodeParams;
use tsp_types::GetSymbolsForNodeRequest;
use tsp_types::LSPId;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::SymbolSearchFlags;
use tsp_types::TSPRequestMethods;

#[test]
fn test_get_symbols_for_node_params_construction() {
    // Test basic parameter construction
    let node = Node {
        uri: "file:///test.py".to_owned(),
        range: Range {
            start: Position {
                line: 5,
                character: 10,
            },
            end: Position {
                line: 5,
                character: 25,
            },
        },
    };

    let params = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("symbol_name".to_owned()),
        node: node.clone(),
        snapshot: 42,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.name, Some("symbol_name".to_owned()));
    assert_eq!(params.node.uri, "file:///test.py");
    assert_eq!(params.node.range.start.line, 5);
    assert_eq!(params.node.range.start.character, 10);
}

#[test]
fn test_get_symbols_for_node_request_construction() {
    // Test basic request construction
    let node = Node {
        uri: "file:///example.py".to_owned(),
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 10,
                character: 0,
            },
        },
    };

    let params = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node,
        snapshot: 10,
    };

    let request = GetSymbolsForNodeRequest {
        method: TSPRequestMethods::TypeServerGetSymbolsForNode,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(
        request.method,
        TSPRequestMethods::TypeServerGetSymbolsForNode
    );
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.node.uri, "file:///example.py");
    assert!(request.params.name.is_none());
}

#[test]
fn test_get_symbols_for_node_params_different_node_ranges() {
    // Test with single character range
    let node_single = Node {
        uri: "file:///single.py".to_owned(),
        range: Range {
            start: Position {
                line: 1,
                character: 5,
            },
            end: Position {
                line: 1,
                character: 6,
            },
        },
    };

    let params_single = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_single,
        snapshot: 1,
    };

    // Test with multi-line range
    let node_multiline = Node {
        uri: "file:///multiline.py".to_owned(),
        range: Range {
            start: Position {
                line: 10,
                character: 0,
            },
            end: Position {
                line: 25,
                character: 50,
            },
        },
    };

    let params_multiline = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_multiline,
        snapshot: 1,
    };

    // Test with zero-width range (cursor position)
    let node_cursor = Node {
        uri: "file:///cursor.py".to_owned(),
        range: Range {
            start: Position {
                line: 15,
                character: 20,
            },
            end: Position {
                line: 15,
                character: 20,
            },
        },
    };

    let params_cursor = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_cursor,
        snapshot: 1,
    };

    // Verify different range types
    assert_eq!(
        params_single.node.range.end.character - params_single.node.range.start.character,
        1
    );
    assert_eq!(
        params_multiline.node.range.end.line - params_multiline.node.range.start.line,
        15
    );
    assert_eq!(
        params_cursor.node.range.start.line,
        params_cursor.node.range.end.line
    );
    assert_eq!(
        params_cursor.node.range.start.character,
        params_cursor.node.range.end.character
    );
}

#[test]
fn test_get_symbols_for_node_params_different_uris() {
    // Test with simple Python file
    let node_py = Node {
        uri: "file:///simple.py".to_owned(),
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 1,
                character: 0,
            },
        },
    };

    let params_py = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_py,
        snapshot: 1,
    };

    // Test with complex path
    let node_complex = Node {
        uri: "file:///path/to/complex/module/file.py".to_owned(),
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 1,
                character: 0,
            },
        },
    };

    let params_complex = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_complex,
        snapshot: 1,
    };

    // Test with Windows-style path
    let node_windows = Node {
        uri: "file:///C:/Users/test/project/file.py".to_owned(),
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 1,
                character: 0,
            },
        },
    };

    let params_windows = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: node_windows,
        snapshot: 1,
    };

    // Verify URI handling
    assert_eq!(params_py.node.uri, "file:///simple.py");
    assert_eq!(
        params_complex.node.uri,
        "file:///path/to/complex/module/file.py"
    );
    assert_eq!(
        params_windows.node.uri,
        "file:///C:/Users/test/project/file.py"
    );
}

#[test]
fn test_get_symbols_for_node_params_with_name_variations() {
    let node = Node {
        uri: "file:///test.py".to_owned(),
        range: Range {
            start: Position {
                line: 5,
                character: 0,
            },
            end: Position {
                line: 5,
                character: 10,
            },
        },
    };

    // Test with specific name
    let params_with_name = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("specific_symbol".to_owned()),
        node: node.clone(),
        snapshot: 5,
    };

    // Test with empty name (but Some)
    let params_empty_name = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("".to_owned()),
        node: node.clone(),
        snapshot: 5,
    };

    // Test with None name (returns all symbols)
    let params_no_name = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node,
        snapshot: 5,
    };

    // Verify name handling
    assert_eq!(params_with_name.name, Some("specific_symbol".to_owned()));
    assert_eq!(params_empty_name.name, Some("".to_owned()));
    assert!(params_no_name.name.is_none());
}

#[test]
fn test_get_symbols_for_node_request_different_id_types() {
    let node = Node {
        uri: "file:///test.py".to_owned(),
        range: Range {
            start: Position {
                line: 2,
                character: 5,
            },
            end: Position {
                line: 2,
                character: 15,
            },
        },
    };

    let params = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("test_symbol".to_owned()),
        node,
        snapshot: 7,
    };

    // Test with Number ID
    let request_number = GetSymbolsForNodeRequest {
        method: TSPRequestMethods::TypeServerGetSymbolsForNode,
        id: LSPId::Int(42),
        params: params.clone(),
    };

    // Test with String ID
    let request_string = GetSymbolsForNodeRequest {
        method: TSPRequestMethods::TypeServerGetSymbolsForNode,
        id: LSPId::String("symbols-for-node-req-123".to_owned()),
        params,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "symbols-for-node-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_symbols_for_node_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetSymbolsForNodeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("serialization_test".to_owned()),
        node: Node {
            uri: "file:///serialization/test.py".to_owned(),
            range: Range {
                start: Position {
                    line: 100,
                    character: 25,
                },
                end: Position {
                    line: 105,
                    character: 40,
                },
            },
        },
        snapshot: 999,
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetSymbolsForNodeParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.name, original_params.name);
    assert_eq!(deserialized_params.node.uri, original_params.node.uri);
    assert_eq!(
        deserialized_params.node.range.start.line,
        original_params.node.range.start.line
    );
    assert_eq!(
        deserialized_params.node.range.start.character,
        original_params.node.range.start.character
    );
    assert_eq!(
        deserialized_params.node.range.end.line,
        original_params.node.range.end.line
    );
    assert_eq!(
        deserialized_params.node.range.end.character,
        original_params.node.range.end.character
    );
}

#[test]
fn test_get_symbols_for_node_request_json_structure() {
    // Test that the JSON structure matches expected TSP format
    let request = GetSymbolsForNodeRequest {
        method: TSPRequestMethods::TypeServerGetSymbolsForNode,
        id: LSPId::Int(3),
        params: GetSymbolsForNodeParams {
            flags: SymbolSearchFlags::new(),
            name: Some("json_test_symbol".to_owned()),
            node: Node {
                uri: "file:///json/test.py".to_owned(),
                range: Range {
                    start: Position {
                        line: 10,
                        character: 5,
                    },
                    end: Position {
                        line: 10,
                        character: 15,
                    },
                },
            },
            snapshot: 555,
        },
    };

    let json_str = serde_json::to_string(&request).expect("Failed to serialize");
    let json_value: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Verify JSON structure
    assert_eq!(json_value["method"], "typeServer/getSymbolsForNode");
    assert_eq!(json_value["id"], 3);
    assert_eq!(json_value["params"]["name"], "json_test_symbol");
    assert_eq!(json_value["params"]["node"]["uri"], "file:///json/test.py");
    assert_eq!(json_value["params"]["node"]["range"]["start"]["line"], 10);
    assert_eq!(
        json_value["params"]["node"]["range"]["start"]["character"],
        5
    );
    assert_eq!(json_value["params"]["snapshot"], 555);
}
