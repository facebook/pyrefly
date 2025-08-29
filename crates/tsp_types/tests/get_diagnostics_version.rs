/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetDiagnosticsVersionParams and GetDiagnosticsVersionRequest type construction and serialization
//!
//! These tests verify the GetDiagnosticsVersionParams parameter construction and validation by:
//! 1. Testing TSP GetDiagnosticsVersionParams construction with various parameter combinations
//! 2. Testing snapshot validation logic
//! 3. Testing URI handling and edge cases
//! 4. Testing serialization/deserialization round-trips

use tsp_types::GetDiagnosticsVersionParams;
use tsp_types::GetDiagnosticsVersionRequest;
use tsp_types::LSPId;
use tsp_types::TSPRequestMethods;

#[test]
fn test_get_diagnostics_version_params_construction() {
    // Test basic parameter construction
    let params = GetDiagnosticsVersionParams {
        snapshot: 42,
        uri: "file:///test.py".to_owned(),
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.uri, "file:///test.py");
}

#[test]
fn test_get_diagnostics_version_request_construction() {
    // Test basic request construction
    let params = GetDiagnosticsVersionParams {
        snapshot: 10,
        uri: "file:///example.py".to_owned(),
    };

    let request = GetDiagnosticsVersionRequest {
        method: TSPRequestMethods::TypeServerGetDiagnosticsVersion,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(
        request.method,
        TSPRequestMethods::TypeServerGetDiagnosticsVersion
    );
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.uri, "file:///example.py");
}

#[test]
fn test_get_diagnostics_version_params_different_snapshots() {
    // Test with zero snapshot
    let params_zero = GetDiagnosticsVersionParams {
        snapshot: 0,
        uri: "file:///zero.py".to_owned(),
    };

    // Test with positive snapshot
    let params_positive = GetDiagnosticsVersionParams {
        snapshot: 12345,
        uri: "file:///positive.py".to_owned(),
    };

    // Test with negative snapshot (should be valid in parameter construction)
    let params_negative = GetDiagnosticsVersionParams {
        snapshot: -1,
        uri: "file:///negative.py".to_owned(),
    };

    // Verify snapshot values
    assert_eq!(params_zero.snapshot, 0);
    assert_eq!(params_positive.snapshot, 12345);
    assert_eq!(params_negative.snapshot, -1);
}

#[test]
fn test_get_diagnostics_version_params_different_uris() {
    // Test with simple file URI
    let params_simple = GetDiagnosticsVersionParams {
        snapshot: 1,
        uri: "file:///simple.py".to_owned(),
    };

    // Test with complex file path
    let params_complex = GetDiagnosticsVersionParams {
        snapshot: 1,
        uri: "file:///path/to/complex/module.py".to_owned(),
    };

    // Test with relative-style URI
    let params_relative = GetDiagnosticsVersionParams {
        snapshot: 1,
        uri: "file:///../relative/path.py".to_owned(),
    };

    // Test with Windows-style path
    let params_windows = GetDiagnosticsVersionParams {
        snapshot: 1,
        uri: "file:///C:/Windows/path/file.py".to_owned(),
    };

    // Test with special characters in filename
    let params_special = GetDiagnosticsVersionParams {
        snapshot: 1,
        uri: "file:///path/with%20spaces/file-with-dashes_and_underscores.py".to_owned(),
    };

    // Verify URI handling
    assert_eq!(params_simple.uri, "file:///simple.py");
    assert_eq!(params_complex.uri, "file:///path/to/complex/module.py");
    assert_eq!(params_relative.uri, "file:///../relative/path.py");
    assert_eq!(params_windows.uri, "file:///C:/Windows/path/file.py");
    assert_eq!(
        params_special.uri,
        "file:///path/with%20spaces/file-with-dashes_and_underscores.py"
    );
}

#[test]
fn test_get_diagnostics_version_request_different_id_types() {
    // Test with Number ID
    let params = GetDiagnosticsVersionParams {
        snapshot: 5,
        uri: "file:///test.py".to_owned(),
    };

    let request_number = GetDiagnosticsVersionRequest {
        method: TSPRequestMethods::TypeServerGetDiagnosticsVersion,
        id: LSPId::Int(42),
        params: params.clone(),
    };

    // Test with String ID
    let request_string = GetDiagnosticsVersionRequest {
        method: TSPRequestMethods::TypeServerGetDiagnosticsVersion,
        id: LSPId::String("diag-version-req-123".to_owned()),
        params,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "diag-version-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_diagnostics_version_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetDiagnosticsVersionParams {
        snapshot: 999,
        uri: "file:///serialization/test.py".to_owned(),
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetDiagnosticsVersionParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.uri, original_params.uri);
}

#[test]
fn test_get_diagnostics_version_request_serialization() {
    // Test that request can be properly serialized and deserialized
    let original_request = GetDiagnosticsVersionRequest {
        method: TSPRequestMethods::TypeServerGetDiagnosticsVersion,
        id: LSPId::Int(777),
        params: GetDiagnosticsVersionParams {
            snapshot: 888,
            uri: "file:///request/serialization/test.py".to_owned(),
        },
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_request).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_request: GetDiagnosticsVersionRequest =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_request.method, original_request.method);
    assert_eq!(
        deserialized_request.params.snapshot,
        original_request.params.snapshot
    );
    assert_eq!(deserialized_request.params.uri, original_request.params.uri);

    match (&deserialized_request.id, &original_request.id) {
        (LSPId::Int(d), LSPId::Int(o)) => assert_eq!(d, o),
        _ => panic!("ID type mismatch"),
    }
}

#[test]
fn test_get_diagnostics_version_request_json_structure() {
    // Test that the JSON structure matches expected TSP format
    let request = GetDiagnosticsVersionRequest {
        method: TSPRequestMethods::TypeServerGetDiagnosticsVersion,
        id: LSPId::Int(3),
        params: GetDiagnosticsVersionParams {
            snapshot: 555,
            uri: "file:///json/structure/test.py".to_owned(),
        },
    };

    let json_str = serde_json::to_string(&request).expect("Failed to serialize");
    let json_value: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Verify JSON structure
    assert_eq!(json_value["method"], "typeServer/getDiagnosticsVersion");
    assert_eq!(json_value["id"], 3);
    assert_eq!(json_value["params"]["snapshot"], 555);
    assert_eq!(
        json_value["params"]["uri"],
        "file:///json/structure/test.py"
    );
}
