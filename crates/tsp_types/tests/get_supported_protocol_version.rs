/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetSupportedProtocolVersionRequest type construction and serialization
//!
//! These tests verify the GetSupportedProtocolVersionRequest parameter construction and validation by:
//! 1. Testing TSP GetSupportedProtocolVersionRequest construction (no parameters required)
//! 2. Testing serialization/deserialization round-trips
//! 3. Testing method and id field validation

use tsp_types::GetSupportedProtocolVersionRequest;
use tsp_types::LSPId;
use tsp_types::TSPRequestMethods;

#[test]
fn test_get_supported_protocol_version_request_construction() {
    // Test basic request construction
    let request = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::Int(1),
        params: None,
    };

    // Verify request construction
    assert_eq!(
        request.method,
        TSPRequestMethods::TypeServerGetSupportedProtocolVersion
    );
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert!(request.params.is_none());
}

#[test]
fn test_get_supported_protocol_version_request_different_id_types() {
    // Test with Number ID
    let request_number = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::Int(100),
        params: None,
    };

    // Test with String ID
    let request_string = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::String("protocol-version-req-456".to_owned()),
        params: None,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 100),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "protocol-version-req-456"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_supported_protocol_version_request_serialization() {
    // Test that request can be properly serialized and deserialized
    let original_request = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::Int(789),
        params: None,
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_request).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_request: GetSupportedProtocolVersionRequest =
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
fn test_get_supported_protocol_version_request_json_structure() {
    // Test that the JSON structure matches expected TSP format
    let request = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::Int(2),
        params: None,
    };

    let json_str = serde_json::to_string(&request).expect("Failed to serialize");
    let json_value: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Verify JSON structure
    assert_eq!(
        json_value["method"],
        "typeServer/getSupportedProtocolVersion"
    );
    assert_eq!(json_value["id"], 2);
    assert!(json_value["params"].is_null());
}
