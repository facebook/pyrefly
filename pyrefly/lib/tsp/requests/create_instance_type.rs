/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the createInstanceType TSP request
//!
//! This request creates an instance type from a class/type handle.
//! For example: Type[int] -> int, Type[str] -> str
//!
//! Important: Type handles are tied to specific snapshots and are invalidated
//! when the snapshot changes. The request will:
//! 1. First validate that the provided snapshot is still current
//! 2. Attempt to resolve the provided type handle
//! 3. Return None if the handle can't be resolved (likely due to snapshot changes)
//! 4. Transform the resolved type to its instance type

use lsp_server::ResponseError;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
use crate::types::types::Type as PyType;

impl TspServer {
    pub fn create_instance_type(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::CreateInstanceTypeParams,
    ) -> Result<Option<tsp::Type>, ResponseError> {
        // Validate snapshot
        self.validate_snapshot(params.snapshot)?;

        tsp_debug!("Creating instance type for: {:?}", params.type_);

        // Use the handle mapping to get the actual pyrefly type
        let Some(py_type) = self.lookup_type_from_tsp_type(&params.type_) else {
            tsp_debug!(
                "Warning: Could not resolve type handle for createInstanceType (likely due to snapshot invalidation)"
            );
            return Ok(None);
        };

        // Create an instance type based on the input type
        let instance_type = match py_type {
            PyType::ClassType(class_type) => {
                // For a class type, the instance type is just the class type itself
                // e.g., given class `str`, the instance type is `str` (instances of the class)
                PyType::ClassType(class_type)
            }
            PyType::Type(inner_type) => {
                // For Type[X], the instance type is X
                // e.g., Type[int] -> int, Type[str] -> str
                *inner_type
            }
            PyType::Union(types) => {
                // For Union[Type[A], Type[B]], create Union[A, B]
                let mut instance_types = Vec::new();
                for ty in types {
                    match ty {
                        PyType::Type(inner) => {
                            instance_types.push(*inner);
                        }
                        PyType::ClassType(_) => {
                            // ClassType already represents instances
                            instance_types.push(ty);
                        }
                        _ => {
                            // For other types in the union, include them as-is
                            instance_types.push(ty);
                        }
                    }
                }

                if instance_types.len() == 1 {
                    instance_types.into_iter().next().unwrap()
                } else {
                    PyType::Union(instance_types)
                }
            }
            _ => {
                // For other types (literals, callables, etc.), they are already instance types
                // or don't have a meaningful "instance" concept
                py_type
            }
        };

        // Convert back to TSP type format
        let result = Some(crate::tsp::common::convert_to_tsp_type(instance_type));

        tsp_debug!("createInstanceType result: {:?}", result);
        Ok(result)
    }
}
