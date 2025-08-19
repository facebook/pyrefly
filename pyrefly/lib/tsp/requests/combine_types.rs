/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the combineTypes TSP request
//!
//! This request combines multiple type handles into a union type.
//!
//! Important: Type handles are tied to specific snapshots and are invalidated
//! when the snapshot changes. The request will:
//! 1. First validate that the provided snapshot is still current
//! 2. Attempt to resolve all provided type handles
//! 3. Skip any handles that can't be resolved (likely due to snapshot changes)
//! 4. Combine the remaining valid types into a union

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
use crate::types::simplify::unions;

impl TspServer {
    pub fn combine_types(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::CombineTypesParams,
    ) -> Result<Option<tsp::Type>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        tsp_debug!("Combining {} types", params.types.len());

        // Validate input
        if params.types.is_empty() {
            return Err(ResponseError {
                code: ErrorCode::InvalidParams as i32,
                message: "combineTypes requires at least one type".to_owned(),
                data: None,
            });
        }

        if params.types.len() == 1 {
            // Single type, just return it as-is
            return Ok(Some(params.types[0].clone()));
        }

        // Convert all TSP types to internal pyrefly types
        let mut py_types = Vec::new();
        for tsp_type in &params.types {
            let Some(py_type) = self.lookup_type_from_tsp_type(tsp_type) else {
                tsp_debug!(
                    "Warning: Could not resolve type handle: {:?}",
                    tsp_type.handle
                );
                // Skip unresolvable types (likely due to snapshot invalidation) rather than failing completely
                continue;
            };
            py_types.push(py_type);
        }

        if py_types.is_empty() {
            // No valid types found
            return Ok(None);
        }

        if py_types.len() == 1 {
            // Only one valid type found
            return Ok(Some(crate::tsp::common::convert_to_tsp_type(
                py_types.into_iter().next().unwrap(),
            )));
        }

        // Create a union of all the types using pyrefly's union simplification logic
        let union_type = unions(py_types);

        // Convert back to TSP type format
        let result = Some(crate::tsp::common::convert_to_tsp_type(union_type));

        tsp_debug!("combineTypes result: {:?}", result);
        Ok(result)
    }
}
