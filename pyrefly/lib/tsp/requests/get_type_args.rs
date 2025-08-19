/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get type args request implementation

use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_type_args(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::GetTypeArgsParams,
    ) -> Result<Vec<tsp::Type>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        // Get the internal type from the type handle
        let internal_type = match self.lookup_type_from_tsp_type(&params.type_) {
            Some(t) => t,
            None => {
                tsp_debug!("Could not resolve type handle: {:?}", params.type_.handle);
                return Ok(Vec::new());
            }
        };

        // Extract type arguments based on the type
        match &internal_type {
            // Union types: return the constituent types
            crate::types::types::Type::Union(union_type) => {
                let mut result_types = Vec::new();
                for union_member in union_type.iter() {
                    result_types.push(self.convert_and_register_type(union_member.clone()));
                }
                Ok(result_types)
            }

            // Class types with generic arguments
            crate::types::types::Type::ClassType(class_type) => {
                let type_args = class_type.targs();
                let mut result_types = Vec::new();
                for arg_type in type_args.as_slice() {
                    result_types.push(self.convert_and_register_type(arg_type.clone()));
                }
                Ok(result_types)
            }

            // TypedDict types with generic arguments
            crate::types::types::Type::TypedDict(typed_dict) => {
                let type_args = typed_dict.targs();
                let mut result_types = Vec::new();
                for arg_type in type_args.as_slice() {
                    result_types.push(self.convert_and_register_type(arg_type.clone()));
                }
                Ok(result_types)
            }

            // Partial TypedDict types with generic arguments
            crate::types::types::Type::PartialTypedDict(typed_dict) => {
                let type_args = typed_dict.targs();
                let mut result_types = Vec::new();
                for arg_type in type_args.as_slice() {
                    result_types.push(self.convert_and_register_type(arg_type.clone()));
                }
                Ok(result_types)
            }

            // Tuple types
            crate::types::types::Type::Tuple(tuple_type) => {
                match tuple_type {
                    crate::types::tuple::Tuple::Concrete(element_types) => {
                        let mut result_types = Vec::new();
                        for element_type in element_types.iter() {
                            result_types.push(self.convert_and_register_type(element_type.clone()));
                        }
                        Ok(result_types)
                    }
                    crate::types::tuple::Tuple::Unbounded(element_type) => {
                        // For unbounded tuples like Tuple[int, ...], return the element type
                        Ok(vec![
                            self.convert_and_register_type(element_type.as_ref().clone()),
                        ])
                    }
                    crate::types::tuple::Tuple::Unpacked(unpacked) => {
                        // For unpacked tuples like Tuple[int, str, *T, bool], return all the types
                        let mut result_types = Vec::new();
                        // Add prefix types
                        for element_type in unpacked.0.iter() {
                            result_types.push(self.convert_and_register_type(element_type.clone()));
                        }
                        // Add the unpacked type (the variadic part)
                        result_types.push(self.convert_and_register_type(unpacked.1.clone()));
                        // Add suffix types
                        for element_type in unpacked.2.iter() {
                            result_types.push(self.convert_and_register_type(element_type.clone()));
                        }
                        Ok(result_types)
                    }
                }
            }

            // Generic class definitions might have type parameters
            crate::types::types::Type::ClassDef(_class_def) => {
                // For class definitions, we can't return type arguments since they aren't instantiated
                // Return empty array as this represents the uninstantiated generic
                Ok(Vec::new())
            }

            // Other types don't have type arguments
            _ => {
                tsp_debug!(
                    "get_type_args called on non-union, non-generic type: {:?}",
                    internal_type
                );
                Ok(Vec::new())
            }
        }
    }
}
