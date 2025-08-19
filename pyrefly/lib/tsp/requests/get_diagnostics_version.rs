/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get diagnostics version request implementation

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use tsp_types as tsp;

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_diagnostics_version(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetDiagnosticsVersionParams,
    ) -> Result<u32, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        let url = lsp_types::Url::parse(&params.uri).map_err(|_| ResponseError {
            code: ErrorCode::InvalidParams as i32,
            message: "Invalid URI".to_owned(),
            data: None,
        })?;
        // Convert URI to file path (validation only)
        if url.to_file_path().is_err() {
            return Err(ResponseError {
                code: ErrorCode::InvalidParams as i32,
                message: "Invalid URI - cannot convert to file path".to_owned(),
                data: None,
            });
        }

        // Validate language services; then create handle
        self.validate_language_services(&url)?;
        let Some(handle) = self.inner.make_handle_if_enabled(&url) else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Language services disabled for this workspace".to_owned(),
                data: None,
            });
        };

        // Try to get load data for this module
        let Some(load_data) = transaction.get_load(&handle) else {
            // If load data doesn't exist, return version 0 to indicate no diagnostics available
            return Ok(0);
        };

        // Return the current load version
        Ok(load_data.version())
    }
}
