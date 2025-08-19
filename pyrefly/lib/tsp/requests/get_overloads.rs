/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get overloads request implementation

use lsp_server::ResponseError;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

/// Extract overload signatures from an overloaded type
///
/// This is the core logic for getting overloads that can be used independently
/// of the Server implementation for unit testing.
pub fn extract_overloads_from_type(
    internal_type: &crate::types::types::Type,
) -> Option<Vec<crate::types::types::Type>> {
    // Only process overloaded function types
    match internal_type {
        crate::types::types::Type::Overload(overload_type) => {
            let mut result_types = Vec::new();

            // Convert each overload signature to a Function type
            for signature in overload_type.signatures.iter() {
                match signature {
                    crate::types::types::OverloadType::Function(function) => {
                        // OverloadType::Function already contains a Function
                        let function_type =
                            crate::types::types::Type::Function(Box::new(function.clone()));
                        result_types.push(function_type);
                    }
                    crate::types::types::OverloadType::Forall(forall) => {
                        // Convert Forall<Function> to Function type
                        let function_type =
                            crate::types::types::Type::Function(Box::new(forall.body.clone()));
                        result_types.push(function_type);
                    }
                }
            }

            Some(result_types)
        }

        // Non-overloaded types return None
        _ => {
            tsp_debug!(
                "extract_overloads_from_type called on non-overloaded type: {:?}",
                internal_type
            );
            None
        }
    }
}

impl TspServer {
    pub fn get_overloads(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::GetOverloadsParams,
    ) -> Result<Option<Vec<tsp::Type>>, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        // Get the internal type from the type handle
        let internal_type = match self.lookup_type_from_tsp_type(&params.type_) {
            Some(t) => t,
            None => {
                tsp_debug!("Could not resolve type handle: {:?}", params.type_.handle);
                return Ok(None);
            }
        };

        // Extract overloads using standalone function
        let overload_types = extract_overloads_from_type(&internal_type);

        // Convert internal types to TSP types
        let result_types = overload_types.map(|types| {
            types
                .into_iter()
                .map(|t| self.convert_and_register_type(t))
                .collect()
        });

        Ok(result_types)
    }
}
