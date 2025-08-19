/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getMetaclass TSP request

use dupe::Dupe;
use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::binding::binding::KeyClassMetadata;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
use crate::types::types::Type as PyType;

impl TspServer {
    pub fn get_metaclass(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetMetaclassParams,
    ) -> Result<Option<tsp::Type>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        tsp_debug!("Getting metaclass for type: {:?}", params.type_);

        // Convert TSP type to internal pyrefly type
        let Some(py_type) = self.lookup_type_from_tsp_type(&params.type_) else {
            tsp_debug!("Warning: Could not resolve type handle for getMetaclass");
            return Ok(None);
        };

        // Extract the ClassType from the pyrefly type
        let class_type = match py_type {
            PyType::ClassType(class_type) => class_type,
            _ => {
                tsp_debug!("Type is not a class type, returning None");
                return Ok(None);
            }
        };

        // Get the class metadata using KeyClassMetadata, following the pattern from class_keywords test
        // First, construct the handle for the class's module
        let module_name = class_type.class_object().module_name();
        let module_path = class_type.class_object().module_path();
        let config = self
            .inner
            .state
            .config_finder()
            .python_file(module_name, module_path);
        let handle = crate::state::handle::Handle::new(
            module_name,
            module_path.clone(),
            config.get_sys_info(),
        );

        let solutions = match transaction.get_solutions(&handle) {
            Some(solutions) => solutions,
            None => {
                tsp_debug!("No solutions found for primary handle");
                return Ok(None);
            }
        };

        let class_metadata = solutions
            .get(&KeyClassMetadata(class_type.class_object().index()))
            .dupe();

        // Get the metaclass from the metadata
        let metaclass = match class_metadata.metaclass() {
            Some(metaclass) => metaclass.clone(),
            None => {
                // When no explicit metaclass is specified, Python uses `type` as the default metaclass
                tsp_debug!("No explicit metaclass found, returning default 'type' metaclass");

                // Use the built-in 'type' class as the default metaclass
                let stdlib = transaction.get_stdlib(&handle);
                stdlib.builtins_type().clone()
            }
        };

        // Convert the metaclass ClassType back to a TSP type
        let metaclass_type = PyType::ClassType(metaclass);
        let result = Some(crate::tsp::common::convert_to_tsp_type(metaclass_type));

        tsp_debug!("getMetaclass result: {:?}", result);
        Ok(result)
    }
}
