/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for GetSymbolsForFileParams, FileSymbolInfo, and Symbol type construction and serialization

use tsp_types::FileSymbolInfo;
use tsp_types::GetSymbolsForFileParams;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::Symbol;

#[test]
fn test_get_symbols_for_file_params_construction() {
    let params = GetSymbolsForFileParams {
        uri: "file:///test.py".to_owned(),
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert_eq!(params.uri, "file:///test.py");
}

#[test]
fn test_get_symbols_for_file_serialization_deserialization() {
    let params = GetSymbolsForFileParams {
        uri: "file:///test_file.py".to_owned(),
        snapshot: 42,
    };

    // Test serialization
    let serialized = serde_json::to_string(&params).expect("Should serialize");

    // Test deserialization
    let deserialized: GetSymbolsForFileParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.uri, params.uri);
    assert_eq!(deserialized.snapshot, params.snapshot);
}

#[test]
fn test_file_symbol_info_construction() {
    let symbols = vec![Symbol {
        node: Node {
            uri: "file:///test.py".to_owned(),
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 5,
                },
            },
        },
        name: "test_symbol".to_owned(),
        decls: vec![],
        synthesized_types: vec![],
    }];

    let file_symbol_info = FileSymbolInfo {
        uri: "file:///test.py".to_owned(),
        symbols: symbols.clone(),
    };

    assert_eq!(file_symbol_info.uri, "file:///test.py");
    assert_eq!(file_symbol_info.symbols.len(), 1);
    assert_eq!(file_symbol_info.symbols[0].name, "test_symbol");
}

#[test]
fn test_symbol_construction() {
    let symbol = Symbol {
        node: Node {
            uri: "file:///example.py".to_owned(),
            range: Range {
                start: Position {
                    line: 5,
                    character: 10,
                },
                end: Position {
                    line: 5,
                    character: 20,
                },
            },
        },
        name: "example_function".to_owned(),
        decls: vec![],
        synthesized_types: vec![],
    };

    assert_eq!(symbol.name, "example_function");
    assert_eq!(symbol.node.uri, "file:///example.py");
    assert_eq!(symbol.node.range.start.line, 5);
    assert_eq!(symbol.node.range.start.character, 10);
    assert_eq!(symbol.node.range.end.line, 5);
    assert_eq!(symbol.node.range.end.character, 20);
    assert_eq!(symbol.decls.len(), 0);
    assert_eq!(symbol.synthesized_types.len(), 0);
}

#[test]
fn test_file_symbol_info_serialization() {
    let symbols = vec![
        Symbol {
            node: Node {
                uri: "file:///test_file.py".to_owned(),
                range: Range {
                    start: Position {
                        line: 1,
                        character: 0,
                    },
                    end: Position {
                        line: 1,
                        character: 8,
                    },
                },
            },
            name: "function1".to_owned(),
            decls: vec![],
            synthesized_types: vec![],
        },
        Symbol {
            node: Node {
                uri: "file:///test_file.py".to_owned(),
                range: Range {
                    start: Position {
                        line: 5,
                        character: 0,
                    },
                    end: Position {
                        line: 5,
                        character: 7,
                    },
                },
            },
            name: "MyClass".to_owned(),
            decls: vec![],
            synthesized_types: vec![],
        },
    ];

    let file_symbol_info = FileSymbolInfo {
        uri: "file:///test_file.py".to_owned(),
        symbols,
    };

    // Test serialization round-trip
    let serialized = serde_json::to_string(&file_symbol_info).expect("Should serialize");
    let deserialized: FileSymbolInfo =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.uri, file_symbol_info.uri);
    assert_eq!(deserialized.symbols.len(), file_symbol_info.symbols.len());
    assert_eq!(deserialized.symbols[0].name, "function1");
    assert_eq!(deserialized.symbols[1].name, "MyClass");
}
