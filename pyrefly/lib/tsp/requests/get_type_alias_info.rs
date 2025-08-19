/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getTypeAliasInfo TSP request

use lsp_server::ResponseError;
use tsp_types as tsp;

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
use crate::types::types::Type as PyType;

/// Standalone get_type_alias_info function that can be used independently of the Server
/// This follows the same pattern as the get_type feature
pub fn get_type_alias_info(py_type: &PyType) -> Option<tsp::TypeAliasInfo> {
    // Check if this is a TypeAlias
    match py_type {
        PyType::TypeAlias(type_alias) => {
            let name = type_alias.name.to_string();

            // Get the aliased type and check if it has type arguments
            let aliased_type = type_alias.as_type();
            let type_args = extract_type_arguments(&aliased_type);

            Some(tsp::TypeAliasInfo { name, type_args })
        }
        _ => {
            // Not a type alias, return None
            None
        }
    }
}

/// Extract type arguments from a type if it's a generic type
pub fn extract_type_arguments(py_type: &PyType) -> Option<Vec<tsp::Type>> {
    match py_type {
        // If it's a ClassType with type arguments, extract them
        PyType::ClassType(class_type) => {
            let targs = class_type.targs();

            if targs.is_empty() {
                None
            } else {
                let type_args: Vec<tsp::Type> = targs
                    .as_slice()
                    .iter()
                    .map(|arg| {
                        // Convert each pyrefly type argument to TSP Type
                        crate::tsp::common::convert_to_tsp_type(arg.clone())
                    })
                    .collect();

                Some(type_args)
            }
        }
        // For other generic types, we could add more cases here
        _ => None,
    }
}

impl TspServer {
    pub fn get_type_alias_info(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::GetTypeAliasInfoParams,
    ) -> Result<Option<tsp::TypeAliasInfo>, ResponseError> {
        // Common validation logic
        self.validate_snapshot(params.snapshot)?;

        // Look up the pyrefly type from the TSP type handle
        let Some(py_type) = self.lookup_type_from_tsp_type(&params.type_) else {
            // If we can't find the type, return None
            return Ok(None);
        };

        // Call the standalone get_type_alias_info function
        Ok(get_type_alias_info(&py_type))
    }
}
