/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP resolve import request implementation

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use lsp_types::Url;
use pyrefly_util::absolutize::Absolutize;
use tsp_types as tsp;
use tsp_types::tsp_debug;

use crate::lsp::module_helpers::to_real_path;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn resolve_import(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::ResolveImportParams,
    ) -> Result<Option<lsp_types::Url>, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        let source_url = lsp_types::Url::parse(&params.source_uri).map_err(|_| ResponseError {
            code: ErrorCode::InvalidParams as i32,
            message: "Invalid source URI".to_owned(),
            data: None,
        })?;
        // Convert source URI to file path (validation only)
        if source_url.to_file_path().is_err() {
            return Err(ResponseError {
                code: ErrorCode::InvalidParams as i32,
                message: "Invalid source URI - cannot convert to file path".to_owned(),
                data: None,
            });
        }

        // Validate language services enabled for workspace
        self.validate_language_services(&source_url)?;
        // After validation, make the handle
        let Some(source_handle) = self.inner.make_handle_if_enabled(&source_url) else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Language services disabled for this workspace".to_owned(),
                data: None,
            });
        };

        // Use the transaction to resolve the import
        let pyrefly_module_name =
            crate::tsp::common::convert_tsp_module_name_to_pyrefly(&params.module_descriptor);
        match transaction.import_handle(&source_handle, pyrefly_module_name, None) {
            Ok(resolved_handle) => {
                // For import resolution, we don't need to load the module at all.
                // We can get the path directly from the resolved handle and convert it to a URI.
                // This avoids the expensive module loading operation.
                let path = match to_real_path(resolved_handle.path()) {
                    Some(path) => path,
                    None => {
                        tsp_debug!("Could not get real path for: {:?}", resolved_handle.path());
                        return Ok(None);
                    }
                };

                let final_path = path.absolutize();

                match Url::from_file_path(final_path) {
                    Ok(url) => Ok(Some(url)),
                    Err(_) => {
                        tsp_debug!(
                            "Could not convert path to URI for: {:?}",
                            resolved_handle.path()
                        );
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                // For debugging, use {:?} instead of {}
                tsp_debug!("Import resolution failed: {:?}", e);
                // Return None instead of an error if the import cannot be resolved
                Ok(None)
            }
        }
    }
}
