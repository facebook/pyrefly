/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for GetSnapshotRequest type construction and serialization

use tsp_types::GetSnapshotRequest;
use tsp_types::GetSnapshotResponse;
use tsp_types::LSPId;
use tsp_types::TSPRequestMethods;

#[test]
fn test_get_snapshot_request_construction() {
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(1),
        params: None,
    };

    assert_eq!(request.method, TSPRequestMethods::TypeServerGetSnapshot);
    assert_eq!(request.id, LSPId::Int(1));
    assert!(request.params.is_none());
}

#[test]
fn test_get_snapshot_request_with_string_id() {
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::String("request-1".to_owned()),
        params: None,
    };

    assert_eq!(request.method, TSPRequestMethods::TypeServerGetSnapshot);
    match &request.id {
        LSPId::String(s) => assert_eq!(s, "request-1"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_snapshot_request_serialization() {
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(42),
        params: None,
    };

    // Test serialization round-trip
    let json_str = serde_json::to_string(&request).expect("Failed to serialize");
    let deserialized: GetSnapshotRequest =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    assert_eq!(deserialized.method, request.method);
    assert_eq!(deserialized.id, request.id);
}

#[test]
fn test_get_snapshot_response_type() {
    // GetSnapshotResponse is an i32 (snapshot number)
    let response: GetSnapshotResponse = 123;
    assert_eq!(response, 123);

    // Test serialization
    let json_str = serde_json::to_string(&response).expect("Failed to serialize");
    assert_eq!(json_str, "123");

    let deserialized: GetSnapshotResponse =
        serde_json::from_str(&json_str).expect("Failed to deserialize");
    assert_eq!(deserialized, 123);
}

#[test]
fn test_get_snapshot_request_json_format() {
    let request = GetSnapshotRequest {
        method: TSPRequestMethods::TypeServerGetSnapshot,
        id: LSPId::Int(1),
        params: None,
    };

    let json_value = serde_json::to_value(&request).expect("Failed to serialize");

    // Verify the JSON structure
    assert_eq!(
        json_value["method"],
        serde_json::json!("typeServer/getSnapshot")
    );
    assert_eq!(json_value["id"], serde_json::json!(1));
}
