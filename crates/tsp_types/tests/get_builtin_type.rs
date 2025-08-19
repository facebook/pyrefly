/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for GetBuiltinTypeParams type construction and serialization

use tsp_types::GetBuiltinTypeParams;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;

#[test]
fn test_get_builtin_type_params_construction() {
    let position = Position {
        line: 0,
        character: 0,
    };

    let params = GetBuiltinTypeParams {
        scoping_node: Node {
            uri: "file:///test.py".to_owned(),
            range: Range {
                start: position.clone(),
                end: position,
            },
        },
        name: "int".to_owned(),
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert_eq!(params.name, "int");
    assert_eq!(params.scoping_node.uri, "file:///test.py");
    assert_eq!(params.scoping_node.range.start.line, 0);
    assert_eq!(params.scoping_node.range.start.character, 0);
    assert_eq!(params.scoping_node.range.end.line, 0);
    assert_eq!(params.scoping_node.range.end.character, 0);
}

#[test]
fn test_get_builtin_type_params_with_different_types() {
    let builtin_types = vec![
        "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    ];

    for type_name in builtin_types {
        let params = GetBuiltinTypeParams {
            scoping_node: Node {
                uri: "file:///builtin_test.py".to_owned(),
                range: Range {
                    start: Position {
                        line: 1,
                        character: 5,
                    },
                    end: Position {
                        line: 1,
                        character: 10,
                    },
                },
            },
            name: type_name.to_owned(),
            snapshot: 42,
        };

        assert_eq!(params.name, type_name);
        assert_eq!(params.snapshot, 42);
        assert_eq!(params.scoping_node.uri, "file:///builtin_test.py");
    }
}

#[test]
fn test_get_builtin_type_params_serialization() {
    let params = GetBuiltinTypeParams {
        scoping_node: Node {
            uri: "file:///test_module.py".to_owned(),
            range: Range {
                start: Position {
                    line: 10,
                    character: 5,
                },
                end: Position {
                    line: 10,
                    character: 8,
                },
            },
        },
        name: "str".to_owned(),
        snapshot: 123,
    };

    // Test serialization round-trip
    let serialized = serde_json::to_string(&params).expect("Should serialize");
    let deserialized: GetBuiltinTypeParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.name, params.name);
    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.scoping_node.uri, params.scoping_node.uri);
    assert_eq!(
        deserialized.scoping_node.range.start.line,
        params.scoping_node.range.start.line
    );
    assert_eq!(
        deserialized.scoping_node.range.start.character,
        params.scoping_node.range.start.character
    );
    assert_eq!(
        deserialized.scoping_node.range.end.line,
        params.scoping_node.range.end.line
    );
    assert_eq!(
        deserialized.scoping_node.range.end.character,
        params.scoping_node.range.end.character
    );
}
