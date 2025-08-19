/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get repr request implementation

use lsp_server::ResponseError;
use tsp_types as tsp;
// Import the lsp_debug macro from common
use tsp_types::tsp_debug;

// Import shared type formatting utilities
use super::type_formatting;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_repr(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetReprParams,
    ) -> Result<String, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        // Use the handle mapping to get the actual pyrefly type
        let Some(internal_type) = self.lookup_type_from_tsp_type(&params.type_) else {
            // If we can't find the internal type, fall back to the basic formatter
            tsp_debug!(
                "Warning: Could not resolve type handle for repr: {:?}",
                params.type_.handle
            );
            let type_repr = format_type_representation(&params.type_, params.flags);
            return Ok(type_repr);
        };

        // Use pyrefly's native type formatting with flags support
        let type_repr =
            type_formatting::format_type_for_display(internal_type, params.flags, transaction);

        tsp_debug!(
            "Generated repr for type {:?}: {}",
            params.type_.handle,
            type_repr
        );
        Ok(type_repr)
    }
}

/// Format a type representation when internal type lookup fails
fn format_type_representation(type_param: &tsp::Type, flags: tsp::TypeReprFlags) -> String {
    use tsp::TypeCategory;

    let mut result = String::new();

    // Handle different type categories
    match type_param.category {
        TypeCategory::Any => result.push_str("Any"),
        TypeCategory::Function => {
            // For functions, show signature if available
            if type_param.name.is_empty() {
                result.push_str("Callable[..., Any]");
            } else {
                result.push_str(&type_param.name);
            }
        }
        TypeCategory::Overloaded => {
            result.push_str("Overload[");
            result.push_str(&type_param.name);
            result.push(']');
        }
        TypeCategory::Class => {
            // For classes, show the class name
            if flags.has_convert_to_instance_type() {
                // Convert to instance type representation
                result.push_str(&type_param.name);
            } else {
                // Show as type
                result.push_str("type[");
                result.push_str(&type_param.name);
                result.push(']');
            }
        }
        TypeCategory::Module => {
            result.push_str("Module[");
            result.push_str(&type_param.name);
            result.push(']');
        }
        TypeCategory::Union => {
            // For unions, we'd need to format multiple types
            result.push_str("Union[");
            result.push_str(&type_param.name);
            result.push(']');
        }
        TypeCategory::TypeVar => {
            result.push_str(&type_param.name);
            // Add variance information if requested
            if flags.has_print_type_var_variance() {
                // This would require additional metadata about variance
                // For now, just show the basic type var name
            }
        } // All remaining categories already handled; no default arm needed.
    }

    // Add module information if available and it's not a builtin
    if let Some(module_name) = &type_param.module_name
        && !module_name.name_parts.is_empty()
        && module_name.name_parts[0] != "builtins"
    {
        let module_path = module_name.name_parts.join(".");
        result = format!("{module_path}.{result}");
    }

    result
}
