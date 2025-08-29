/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetSnapshotRequest type construction and serialization
//!
//! These tests verify the GetSnapshotRequest parameter construction and validation by:
//! 1. Testing TSP GetSnapshotRequest construction (no parameters required)
//! 2. Testing serialization/deserialization round-trips
//! 3. Testing method and id field validation

use tsp_types::GetSnapshotRequest;
use tsp_types::LSPId;
use tsp_types::TSPRequestMethods;

#[test]
fn test_get_snapshot_request_construction() {
    // Test basic request construction
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(1),
        params: None,
    };

    // Verify request construction
    assert_eq!(request.method, TSPRequestMethods::TypeServerGetSnapshot);
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert!(request.params.is_none());
}

#[test]
fn test_get_snapshot_request_different_id_types() {
    // Test with Number ID
    let request_number = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(42),
        params: None,
    };

    // Test with String ID
    let request_string = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::String("snapshot-req-123".to_owned()),
        params: None,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "snapshot-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_snapshot_request_serialization() {
    // Test that request can be properly serialized and deserialized
    let original_request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(999),
        params: None,
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_request).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_request: GetSnapshotRequest =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_request.method, original_request.method);
    assert_eq!(deserialized_request.params, original_request.params);

    match (&deserialized_request.id, &original_request.id) {
        (LSPId::Int(d), LSPId::Int(o)) => assert_eq!(d, o),
        _ => panic!("ID type mismatch"),
    }
}

#[test]
fn test_get_snapshot_request_json_structure() {
    // Test that the JSON structure matches expected TSP format
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(1),
        params: None,
    };

    let json_str = serde_json::to_string(&request).expect("Failed to serialize");
    let json_value: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Verify JSON structure
    assert_eq!(json_value["method"], "typeServer/getSnapshot");
    assert_eq!(json_value["id"], 1);
    assert!(json_value["params"].is_null());
}
