/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared validation and error-mapping helpers for TSP request handlers.
//!
//! Every TSP request handler should use these helpers to ensure consistent
//! error behavior across the protocol surface. The helpers cover:
//!
//! - Snapshot freshness checks (stale-snapshot rejection)
//! - Canonical TSP error construction (invalid params, internal, etc.)
//! - Response dispatch (success or error, routed through `TspInterface`)

use lsp_server::ErrorCode;
use lsp_server::RequestId;
use lsp_server::ResponseError;
use lsp_types::Url;
use serde::Serialize;

use crate::lsp::non_wasm::lsp::new_response;
use crate::lsp::non_wasm::protocol::Response;
use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspServer;

// ---------------------------------------------------------------------------
// Canonical TSP error constructors
// ---------------------------------------------------------------------------

/// Build a `ResponseError` for a stale snapshot.
///
/// Returned when the client supplies a snapshot version that no longer matches
/// the server's current snapshot. The client should re-fetch the snapshot and
/// retry.
pub fn snapshot_outdated_error(client_snapshot: i32, server_snapshot: i32) -> ResponseError {
    ResponseError {
        code: ErrorCode::ServerCancelled as i32,
        message: format!(
            "Snapshot outdated: client sent {client_snapshot}, server is at {server_snapshot}"
        ),
        data: None,
    }
}

/// Build a `ResponseError` for invalid / malformed request parameters.
pub fn invalid_params_error(detail: &str) -> ResponseError {
    ResponseError {
        code: ErrorCode::InvalidParams as i32,
        message: format!("Invalid params: {detail}"),
        data: None,
    }
}

/// Build a `ResponseError` for an unexpected internal failure.
pub fn internal_error(detail: &str) -> ResponseError {
    ResponseError {
        code: ErrorCode::InternalError as i32,
        message: format!("Internal error: {detail}"),
        data: None,
    }
}

// ---------------------------------------------------------------------------
// URI parsing
// ---------------------------------------------------------------------------

/// Parse a URI string, rejecting only malformed/unparseable input.
///
/// Returns the parsed [`Url`] for any valid URI regardless of scheme.
/// Use this when the handler can resolve non-file URIs (e.g. notebook
/// cell URIs) via [`TspInterface::resolve_uri_to_path`].
pub fn parse_uri(uri: &str) -> Result<Url, ResponseError> {
    Url::parse(uri).map_err(|_| invalid_params_error("URI is not valid"))
}

// ---------------------------------------------------------------------------
// Snapshot validation
// ---------------------------------------------------------------------------

impl<T: TspInterface> TspServer<T> {
    /// Validate that the client-supplied snapshot matches the server's current
    /// snapshot. Returns `Ok(())` on match or `Err(ResponseError)` on mismatch.
    pub fn validate_snapshot(&self, client_snapshot: i32) -> Result<(), ResponseError> {
        let current = self.get_snapshot();
        if client_snapshot != current {
            Err(snapshot_outdated_error(client_snapshot, current))
        } else {
            Ok(())
        }
    }

    /// Send a successful JSON-RPC response for `id` with `result`.
    pub fn send_ok<R: Serialize>(&self, id: RequestId, result: R) {
        self.inner.send_response(new_response(id, Ok(result)));
    }

    /// Send a JSON-RPC error response for `id`.
    pub fn send_err(&self, id: RequestId, error: ResponseError) {
        self.inner.send_response(Response {
            id,
            result: None,
            error: Some(error),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Error constructor unit tests ---

    #[test]
    fn test_snapshot_outdated_error_code() {
        let err = snapshot_outdated_error(1, 2);
        assert_eq!(err.code, ErrorCode::ServerCancelled as i32);
    }

    #[test]
    fn test_snapshot_outdated_error_message_contains_versions() {
        let err = snapshot_outdated_error(3, 7);
        assert!(err.message.contains("3"), "should mention client snapshot");
        assert!(err.message.contains("7"), "should mention server snapshot");
    }

    #[test]
    fn test_invalid_params_error_code() {
        let err = invalid_params_error("missing field");
        assert_eq!(err.code, ErrorCode::InvalidParams as i32);
    }

    #[test]
    fn test_invalid_params_error_message() {
        let err = invalid_params_error("sourceUri is required");
        assert!(err.message.contains("sourceUri is required"));
    }

    #[test]
    fn test_internal_error_code() {
        let err = internal_error("type resolution failed");
        assert_eq!(err.code, ErrorCode::InternalError as i32);
    }

    #[test]
    fn test_internal_error_message() {
        let err = internal_error("mutex poisoned");
        assert!(err.message.contains("mutex poisoned"));
    }

    #[test]
    fn test_error_data_is_none() {
        // All canonical errors should have data = None
        assert!(snapshot_outdated_error(0, 1).data.is_none());
        assert!(invalid_params_error("x").data.is_none());
        assert!(internal_error("x").data.is_none());
    }

    #[test]
    fn test_error_codes_are_distinct() {
        let snap = snapshot_outdated_error(0, 1).code;
        let params = invalid_params_error("x").code;
        let internal = internal_error("x").code;
        // ServerCancelled, InvalidParams, InternalError should all differ
        assert_ne!(snap, params);
        assert_ne!(snap, internal);
        assert_ne!(params, internal);
    }

    // --- parse_uri unit tests ---

    #[test]
    fn test_parse_uri_file() {
        let url = parse_uri("file:///home/user/project/main.py").unwrap();
        assert_eq!(url.scheme(), "file");
    }

    #[test]
    fn test_parse_uri_notebook_cell() {
        let url =
            parse_uri("vscode-notebook-cell:/Users/kylei/projects/test/test.ipynb#W0sZmlsZQ%3D%3D")
                .unwrap();
        assert_eq!(url.scheme(), "vscode-notebook-cell");
    }

    #[test]
    fn test_parse_uri_empty_is_error() {
        assert!(parse_uri("").is_err());
    }

    #[test]
    fn test_parse_uri_relative_path_is_error() {
        assert!(parse_uri("some/path").is_err());
    }

    #[test]
    fn test_parse_uri_http() {
        let url = parse_uri("http://example.com").unwrap();
        assert_eq!(url.scheme(), "http");
    }

    #[test]
    fn test_parse_uri_untitled() {
        let url = parse_uri("untitled:Untitled-1").unwrap();
        assert_eq!(url.scheme(), "untitled");
    }
}
