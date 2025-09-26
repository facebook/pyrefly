/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for TSP GetDiagnosticsParams type construction and serialization

use tsp_types::GetDiagnosticsParams;

#[test]
fn test_get_diagnostics_params_construction() {
    let params = GetDiagnosticsParams {
        uri: "file:///test.py".to_owned(),
        snapshot: 1,
    };

    assert_eq!(params.uri, "file:///test.py");
    assert_eq!(params.snapshot, 1);
}

#[test]
fn test_get_diagnostics_params_serialization() {
    let params = GetDiagnosticsParams {
        uri: "file:///test.py".to_owned(),
        snapshot: 1,
    };

    let serialized = serde_json::to_string(&params).expect("Failed to serialize params");
    // Field order may now be snapshot then uri (serde preserves struct field declaration order)
    let expected = r#"{"snapshot":1,"uri":"file:///test.py"}"#;
    assert_eq!(serialized, expected);

    let deserialized: GetDiagnosticsParams =
        serde_json::from_str(&serialized).expect("Failed to deserialize params");
    assert_eq!(deserialized.uri, params.uri);
    assert_eq!(deserialized.snapshot, params.snapshot);
}
