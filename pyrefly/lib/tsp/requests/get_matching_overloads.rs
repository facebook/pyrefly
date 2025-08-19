/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get matching overloads request implementation

use lsp_server::ResponseError;
use tsp_types as tsp;
use tsp_types::tsp_debug;

use crate::state::state::Transaction;
use crate::tsp::requests::common::node_start_position;
use crate::tsp::server::TspServer;

impl TspServer {
    /// Get matching overloads for a function call site
    ///
    /// This function analyzes a call site and returns the overloads of the function
    /// being called that could potentially match the call. Currently, it returns all
    /// available overloads since full argument analysis is not yet implemented.
    ///
    /// In a complete implementation, this would:
    /// 1. Parse the call arguments at the call site
    /// 2. Match argument types against each overload's parameter types
    /// 3. Return only the overloads that could accept the provided arguments
    pub fn get_matching_overloads(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetMatchingOverloadsParams,
    ) -> Result<Option<Vec<tsp::Type>>, ResponseError> {
        // Validate and get handle/module info; load if needed
        let call_url = lsp_types::Url::parse(&params.call_node.uri).map_err(|_| ResponseError {
            code: lsp_server::ErrorCode::InvalidParams as i32,
            message: "Invalid call_node.uri".to_owned(),
            data: None,
        })?;
        let (handle, module_info, maybe_fresh_tx) = self.with_active_transaction(
            transaction,
            &call_url,
            params.snapshot,
            crate::state::require::Require::Everything,
        )?;

        tsp_debug!(
            "Getting matching overloads for call node: {:?}",
            params.call_node
        );

        let active_tx = maybe_fresh_tx.as_ref().unwrap_or(transaction);

        // Convert the Range position to TextSize using helper
        let position = node_start_position(&module_info, &params.call_node);

        // Try to find the type at the call site position
        let Some(call_site_type) = active_tx.get_type_at(&handle, position) else {
            tsp_debug!("Could not determine type at call site position");
            return Ok(None);
        };

        // Check if the call site type is an overloaded function
        match &call_site_type {
            crate::types::types::Type::Overload(overload_type) => {
                // TODO: Implement sophisticated argument matching
                // For now, we return all overloads since we don't have argument analysis yet.
                // In a complete implementation, we would:
                // 1. Parse the call arguments from the AST at the call site
                // 2. Infer types of each argument
                // 3. Match argument types against each overload's parameter signature
                // 4. Return only overloads where arguments could match parameters

                let mut result_types = Vec::new();

                for signature in overload_type.signatures.iter() {
                    match signature {
                        crate::types::types::OverloadType::Function(function) => {
                            let function_type =
                                crate::types::types::Type::Function(Box::new(function.clone()));
                            result_types.push(self.convert_and_register_type(function_type));
                        }
                        crate::types::types::OverloadType::Forall(forall) => {
                            let function_type =
                                crate::types::types::Type::Function(Box::new(forall.body.clone()));
                            result_types.push(self.convert_and_register_type(function_type));
                        }
                    }
                }

                tsp_debug!("Found {} matching overloads", result_types.len());
                Ok(Some(result_types))
            }

            // If it's a regular function, return it as a single-item array
            crate::types::types::Type::Function(_) => {
                tsp_debug!("Found single function at call site");
                let tsp_type = self.convert_and_register_type(call_site_type);
                Ok(Some(vec![tsp_type]))
            }

            // Other types don't have overloads
            _ => {
                tsp_debug!(
                    "Type at call site is not a function or overload: {:?}",
                    call_site_type
                );
                Ok(None)
            }
        }
    }
}
