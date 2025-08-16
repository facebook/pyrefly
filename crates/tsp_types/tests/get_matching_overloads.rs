/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for GetMatchingOverloadsParams type construction and serialization

use tsp_types::GetMatchingOverloadsParams;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;

#[test]
fn test_get_matching_overloads_params_construction() {
    let params = GetMatchingOverloadsParams {
        call_node: Node {
            uri: "file:///test.py".to_string(),
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 1,
                },
            },
        },
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert_eq!(params.call_node.uri, "file:///test.py");
    assert_eq!(params.call_node.range.start.line, 0);
    assert_eq!(params.call_node.range.start.character, 0);
    assert_eq!(params.call_node.range.end.line, 0);
    assert_eq!(params.call_node.range.end.character, 1);
}

#[test]
fn test_get_matching_overloads_params_serialization() {
    let params = GetMatchingOverloadsParams {
        call_node: Node {
            uri: "file:///function_call.py".to_string(),
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
        snapshot: 42,
    };

    // Test serialization round-trip
    let serialized = serde_json::to_string(&params).expect("Should serialize");
    let deserialized: GetMatchingOverloadsParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.call_node.uri, params.call_node.uri);
    assert_eq!(
        deserialized.call_node.range.start.line,
        params.call_node.range.start.line
    );
    assert_eq!(
        deserialized.call_node.range.start.character,
        params.call_node.range.start.character
    );
    assert_eq!(
        deserialized.call_node.range.end.line,
        params.call_node.range.end.line
    );
    assert_eq!(
        deserialized.call_node.range.end.character,
        params.call_node.range.end.character
    );
}

#[test]
fn test_get_matching_overloads_params_different_positions() {
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
        let params = GetMatchingOverloadsParams {
            call_node: Node {
                uri: format!("file:///test_{i}.py"),
                range: Range {
                    start: start_pos.clone(),
                    end: end_pos.clone(),
                },
            },
            snapshot: i as i32 + 1,
        };

        assert_eq!(params.snapshot, i as i32 + 1);
        assert_eq!(params.call_node.uri, format!("file:///test_{i}.py"));
        assert_eq!(params.call_node.range.start.line, start_pos.line);
        assert_eq!(params.call_node.range.start.character, start_pos.character);
        assert_eq!(params.call_node.range.end.line, end_pos.line);
        assert_eq!(params.call_node.range.end.character, end_pos.character);
    }
}
