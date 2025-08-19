/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get function parts request implementation

use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

// Import shared type formatting utilities
use super::type_formatting;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

/// Extract function parts from a function type
///
/// This is the core logic for getting function parts that can be used independently
/// of the Server implementation for unit testing.
pub fn extract_function_parts_from_function(
    func_type: &crate::types::callable::Function,
    flags: tsp::TypeReprFlags,
    transaction: &Transaction<'_>,
) -> Option<tsp::FunctionParts> {
    // Extract parameter information from the function's signature
    let signature = &func_type.signature;
    extract_function_parts_from_callable(signature, flags, transaction)
}

/// Extract function parts from a callable type
///
/// This is the core logic for getting function parts from callables that can be used
/// independently of the Server implementation for unit testing.
pub fn extract_function_parts_from_callable(
    callable_type: &crate::types::callable::Callable,
    flags: tsp::TypeReprFlags,
    transaction: &Transaction<'_>,
) -> Option<tsp::FunctionParts> {
    // Extract parameter information from callable
    let mut params = Vec::new();

    // Handle different types of params
    match &callable_type.params {
        crate::types::callable::Params::List(param_list) => {
            for param in param_list.items() {
                let param_str = format_param_for_display(param, flags, transaction);
                params.push(param_str);
            }
        }
        crate::types::callable::Params::Ellipsis => {
            params.push("...".to_owned());
        }
        crate::types::callable::Params::ParamSpec(types, param_spec) => {
            // Handle concatenated parameters with a ParamSpec
            for (i, param_type) in types.iter().enumerate() {
                let type_str = type_formatting::format_type_for_display(
                    param_type.clone(),
                    flags,
                    transaction,
                );
                params.push(format!("param{i}: {type_str}"));
            }
            let param_spec_str =
                type_formatting::format_type_for_display(param_spec.clone(), flags, transaction);
            params.push(format!("*{param_spec_str}"));
        }
    }

    // Get return type
    let return_type_str =
        type_formatting::format_type_for_display(callable_type.ret.clone(), flags, transaction);

    Some(tsp::FunctionParts {
        params,
        return_type: return_type_str,
    })
}

/// Format a parameter for display
///
/// This is a helper function that can be used independently for formatting parameters.
pub fn format_param_for_display(
    param: &crate::types::callable::Param,
    flags: tsp::TypeReprFlags,
    transaction: &Transaction<'_>,
) -> String {
    use crate::types::callable::Param;

    match param {
        Param::PosOnly(name, param_type, _required) => {
            let type_str =
                type_formatting::format_type_for_display(param_type.clone(), flags, transaction);
            if let Some(name) = name {
                format!("{name}: {type_str}")
            } else {
                type_str
            }
        }
        Param::Pos(name, param_type, _required) => {
            let type_str =
                type_formatting::format_type_for_display(param_type.clone(), flags, transaction);
            format!("{name}: {type_str}")
        }
        Param::VarArg(name, param_type) => {
            let type_str =
                type_formatting::format_type_for_display(param_type.clone(), flags, transaction);
            if let Some(name) = name {
                format!("*{name}: {type_str}")
            } else {
                format!("*{type_str}")
            }
        }
        Param::KwOnly(name, param_type, _required) => {
            let type_str =
                type_formatting::format_type_for_display(param_type.clone(), flags, transaction);
            format!("{name}: {type_str}")
        }
        Param::Kwargs(name, param_type) => {
            let type_str =
                type_formatting::format_type_for_display(param_type.clone(), flags, transaction);
            if let Some(name) = name {
                format!("**{name}: {type_str}")
            } else {
                format!("**{type_str}")
            }
        }
    }
}

/// Format a type for display
impl TspServer {
    pub fn get_function_parts(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetFunctionPartsParams,
    ) -> Result<Option<tsp::FunctionParts>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        tsp_debug!("Getting function parts for type: {:?}", params.type_.handle);

        // Get the internal type from the type handle
        let internal_type = match self.lookup_type_from_tsp_type(&params.type_) {
            Some(t) => t,
            None => {
                tsp_debug!("Could not resolve type handle: {:?}", params.type_.handle);
                return Ok(None);
            }
        };

        // Extract function parts based on the type
        let flags = params.flags;
        match &internal_type {
            crate::types::types::Type::Function(func_type) => Ok(
                extract_function_parts_from_function(func_type, flags, transaction),
            ),
            crate::types::types::Type::Callable(callable_type) => Ok(
                extract_function_parts_from_callable(callable_type, flags, transaction),
            ),
            crate::types::types::Type::Overload(_overload_type) => {
                // For overloaded functions, we could return the signature of the first overload
                // or a combined representation. For now, let's return None as it's complex.
                tsp_debug!("Function parts for overloaded functions not yet implemented");
                Ok(None)
            }
            _ => {
                tsp_debug!(
                    "get_function_parts only works on function types, got: {:?}",
                    internal_type
                );
                Ok(None)
            }
        }
    }
}
