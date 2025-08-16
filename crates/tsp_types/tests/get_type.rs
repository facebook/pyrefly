/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for GetTypeParams type construction and serialization

use tsp_types::GetTypeParams;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;

#[test]
fn test_get_type_params_construction() {
    let position = Position {
        line: 0,
        character: 0,
    };

    let params = GetTypeParams {
        node: Node {
            uri: "file:///test.py".to_string(),
            range: Range {
                start: position.clone(),
                end: position,
            },
        },
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert_eq!(params.node.uri, "file:///test.py");
    assert_eq!(params.node.range.start.line, 0);
    assert_eq!(params.node.range.start.character, 0);
    assert_eq!(params.node.range.end.line, 0);
    assert_eq!(params.node.range.end.character, 0);
}

#[test]
fn test_get_type_params_serialization() {
    let params = GetTypeParams {
        node: Node {
            uri: "file:///example.py".to_string(),
            range: Range {
                start: Position {
                    line: 5,
                    character: 10,
                },
                end: Position {
                    line: 5,
                    character: 15,
                },
            },
        },
        snapshot: 42,
    };

    // Test serialization round-trip
    let serialized = serde_json::to_string(&params).expect("Should serialize");
    let deserialized: GetTypeParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.node.uri, params.node.uri);
    assert_eq!(
        deserialized.node.range.start.line,
        params.node.range.start.line
    );
    assert_eq!(
        deserialized.node.range.start.character,
        params.node.range.start.character
    );
    assert_eq!(deserialized.node.range.end.line, params.node.range.end.line);
    assert_eq!(
        deserialized.node.range.end.character,
        params.node.range.end.character
    );
}

#[test]
fn test_get_type_params_different_positions() {
    let test_cases = vec![
        (
            Position {
                line: 0,
                character: 0,
            },
            Position {
                line: 0,
                character: 5,
            },
        ),
        (
            Position {
                line: 10,
                character: 15,
            },
            Position {
                line: 10,
                character: 25,
            },
        ),
        (
            Position {
                line: 100,
                character: 0,
            },
            Position {
                line: 105,
                character: 0,
            },
        ),
    ];

    for (i, (start_pos, end_pos)) in test_cases.into_iter().enumerate() {
        let params = GetTypeParams {
            node: Node {
                uri: format!("file:///test_{i}.py"),
                range: Range {
                    start: start_pos.clone(),
                    end: end_pos.clone(),
                },
            },
            snapshot: i as i32 + 1,
        };

        assert_eq!(params.snapshot, i as i32 + 1);
        assert_eq!(params.node.uri, format!("file:///test_{i}.py"));
        assert_eq!(params.node.range.start.line, start_pos.line);
        assert_eq!(params.node.range.start.character, start_pos.character);
        assert_eq!(params.node.range.end.line, end_pos.line);
        assert_eq!(params.node.range.end.character, end_pos.character);
    }
}
