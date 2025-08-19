/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getDiagnostics TSP request

use std::iter::once;

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use lsp_types::Diagnostic;
use pyrefly_python::module_name::ModuleName;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::lsp::module_helpers::make_open_handle;
use crate::lsp::module_helpers::to_real_path;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_diagnostics(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetDiagnosticsParams,
    ) -> Result<Option<Vec<Diagnostic>>, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        tsp_debug!("Getting diagnostics for URI: {}", params.uri);

        // Parse URI string
        let url = lsp_types::Url::parse(&params.uri).map_err(|_| ResponseError {
            code: ErrorCode::InvalidParams as i32,
            message: "Invalid URI".to_owned(),
            data: None,
        })?;

        // Convert URI to file path
        let file_path = match url.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                return Err(ResponseError {
                    code: ErrorCode::InvalidParams as i32,
                    message: "Invalid URI - cannot convert to file path".to_owned(),
                    data: None,
                });
            }
        };

        // Check if workspace has language services enabled
        let Some(_handle) = self.inner.make_handle_if_enabled(&url) else {
            tsp_debug!("Language services disabled for workspace");
            return Ok(Some(Vec::new()));
        };

        // Create handle for the file
        let handle = make_open_handle(&self.inner.state, &file_path);

        // Collect errors for this file
        let mut diagnostics = Vec::new();
        let open_files = self.inner.open_files.read();

        for error in transaction.get_errors(once(&handle)).collect_errors().shown {
            // Apply the same filtering logic as get_diag_if_shown
            if let Some(path) = to_real_path(error.path()) {
                // When no file covers this, we'll get the default configured config which includes "everything"
                // and excludes `.<file>`s.
                let config = self
                    .inner
                    .state
                    .config_finder()
                    .python_file(ModuleName::unknown(), error.path());
                if open_files.contains_key(&path)
                    && !config.project_excludes.covers(&path)
                    && !self
                        .inner
                        .workspaces
                        .get_with(path.to_path_buf(), |w| w.disable_type_errors)
                {
                    diagnostics.push(error.to_diagnostic());
                }
            }
        }

        tsp_debug!(
            "Found {} diagnostics for URI: {}",
            diagnostics.len(),
            params.uri
        );
        Ok(Some(diagnostics))
    }
}
