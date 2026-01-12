/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Common utilities and helper functions for TSP request handling

use lsp_server::ErrorCode;
use lsp_server::Request;
use lsp_server::ResponseError;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::protocol as tsp;

// ---------------------------------------------------------------------------
// Backward compatibility shims (manually added)
// ---------------------------------------------------------------------------
// Provide the current protocol version.
// This references the generated TypeServerVersion::Current variant.
pub const TSP_PROTOCOL_VERSION: tsp::TypeServerVersion = tsp::TypeServerVersion::Current;

// Older handlers referenced GetSupportedProtocolVersionParams even though
// the generator only emits a Request with no params. Provide an empty params
// struct so existing handler signatures (before refactor) can compile or we
// can simplify handlers to omit it. This can be removed once all handlers
// are updated to not expect params.
#[derive(Serialize, PartialEq, Debug, Eq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct GetSupportedProtocolVersionParams {}

// Custom Deserialize to allow `null`, `{}`, or any object with unknown fields.
impl<'de> Deserialize<'de> for GetSupportedProtocolVersionParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept any JSON value (null / object / other) and ignore its contents.
        let _ignored = serde_json::Value::deserialize(deserializer)?;
        Ok(GetSupportedProtocolVersionParams {})
    }
}

// -------------------------------------------------------------------------------------------------
// Compatibility shims for legacy handwritten code expecting older enum shapes / helper builders.
// These adapt the generated protocol.rs API (do NOT modify the generated file).
// Keep this section minimal; remove once all call sites are migrated.
// -------------------------------------------------------------------------------------------------

// Lightweight debug macro used by request handlers (avoids pulling in tracing for generated-ish code)
#[macro_export]
macro_rules! tsp_debug {
    ($($arg:tt)*) => {{
        if cfg!(debug_assertions) { eprintln!("[TSP] {}:", module_path!()); eprintln!($($arg)*); }
    }};
}
pub use tsp_debug;

/// Provide a Default implementation shim for ResolveImportOptions (all None)
impl Default for tsp::ResolveImportOptions {
    fn default() -> Self {
        // Historical default behavior used explicit false values. Tests assert these are Some(false)
        // to ensure stable wire format and avoid Option omission during serialization.
        tsp::ResolveImportOptions {
            allow_externally_hidden_access: Some(false),
            resolve_local_names: Some(false),
            skip_file_needed_check: Some(false),
        }
    }
}

/// Handle TypeServer Protocol (TSP) requests that don't implement the LSP Request trait
#[allow(dead_code)]
pub fn as_tsp_request<T>(x: &Request, method_name: &str) -> Option<Result<T, serde_json::Error>>
where
    T: DeserializeOwned,
{
    if x.method == method_name {
        match serde_json::from_value(x.params.clone()) {
            Ok(request) => Some(Ok(request)),
            Err(err) => Some(Err(err)),
        }
    } else {
        None
    }
}

/// Helper to build a JSON-RPC error response for TSP handlers
#[allow(dead_code)]
pub fn error_response(
    id: lsp_server::RequestId,
    code: i32,
    message: String,
) -> lsp_server::Response {
    lsp_server::Response {
        id,
        result: None,
        error: Some(ResponseError {
            code,
            message,
            data: None,
        }),
    }
}

/// Creates a snapshot outdated error
#[allow(dead_code)]
pub fn snapshot_outdated_error() -> ResponseError {
    ResponseError {
        code: ErrorCode::ServerCancelled as i32,
        message: "Snapshot outdated".to_owned(),
        data: None,
    }
}

/// Creates a common error response for internal errors
#[allow(dead_code)]
pub(crate) fn create_internal_error(message: &str) -> ResponseError {
    ResponseError {
        code: ErrorCode::InternalError as i32,
        message: message.to_owned(),
        data: None,
    }
}

/// Creates a common error response for language services being disabled
#[allow(dead_code)]
pub(crate) fn language_services_disabled_error() -> ResponseError {
    ResponseError {
        code: ErrorCode::RequestFailed as i32,
        message: "Language services disabled".to_owned(),
        data: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json;

    use super::*;

    #[test]
    fn test_get_supported_protocol_version_params_deserialization() {
        // Test null case
        let null_json = serde_json::Value::Null;
        let result: Result<GetSupportedProtocolVersionParams, _> =
            serde_json::from_value(null_json);
        assert!(result.is_ok());

        // Test empty object case
        let empty_obj_json = serde_json::json!({});
        let result: Result<GetSupportedProtocolVersionParams, _> =
            serde_json::from_value(empty_obj_json);
        assert!(result.is_ok());

        // Test object with unknown fields (should be ignored)
        let obj_with_fields = serde_json::json!({"unknown_field": "value"});
        let result: Result<GetSupportedProtocolVersionParams, _> =
            serde_json::from_value(obj_with_fields);
        assert!(result.is_ok());
    }
}
