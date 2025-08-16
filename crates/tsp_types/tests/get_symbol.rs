/*
 * Unit tests for TSP GetSymbolParams type construction and serialization
 */

use tsp_types::GetSymbolParams;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;

#[test]
fn test_get_symbol_params_construction() {
    let params = GetSymbolParams {
        node: Node {
            uri: "file:///test.py".to_string(),
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
        name: None,
        skip_unreachable_code: false,
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert!(!params.skip_unreachable_code);
    assert!(params.name.is_none());
    assert_eq!(params.node.uri, "file:///test.py");
    assert_eq!(params.node.range.start.line, 5);
    assert_eq!(params.node.range.start.character, 10);
    assert_eq!(params.node.range.end.line, 5);
    assert_eq!(params.node.range.end.character, 20);
}

#[test]
fn test_get_symbol_params_with_name() {
    let params = GetSymbolParams {
        node: Node {
            uri: "file:///module.py".to_string(),
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 15,
                },
            },
        },
        name: Some("my_function".to_owned()),
        skip_unreachable_code: true,
        snapshot: 42,
    };

    // Verify parameter construction with optional fields set
    assert_eq!(params.snapshot, 42);
    assert!(params.skip_unreachable_code);
    assert_eq!(params.name.as_ref().unwrap(), "my_function");
    assert_eq!(params.node.uri, "file:///module.py");
}

#[test]
fn test_get_symbol_params_serialization() {
    let params = GetSymbolParams {
        node: Node {
            uri: "file:///serialization_test.py".to_string(),
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
        name: Some("test_symbol".to_owned()),
        skip_unreachable_code: true,
        snapshot: 999,
    };

    // Test serialization round-trip
    let json_str = serde_json::to_string(&params).expect("Failed to serialize");
    let deserialized: GetSymbolParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip
    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(
        deserialized.skip_unreachable_code,
        params.skip_unreachable_code
    );
    assert_eq!(deserialized.name, params.name);
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
