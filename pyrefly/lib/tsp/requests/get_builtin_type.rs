/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getBuiltinType TSP request

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;
use tsp_types::snapshot_outdated_error;
use tsp_types::{self as tsp};

use crate::state::handle::Handle;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
use crate::types::class::ClassType;

/// Standalone get_builtin_type function that can be used independently of the Server
/// This follows the same pattern as the get_type feature
pub fn get_builtin_type(
    transaction: &Transaction<'_>,
    scoping_handle: &Handle,
    type_name: &str,
) -> Option<Type> {
    // Create a handle for the builtins module using the same path and sys_info as the scoping node
    let builtins_handle = Handle::new(
        ModuleName::builtins(),
        scoping_handle.path().clone(),
        scoping_handle.sys_info().clone(),
    );

    // Get the stdlib for the scoping handle
    let stdlib = transaction.get_stdlib(scoping_handle);

    // Try to get the builtin type from the stdlib using known builtin type names
    match type_name {
        "int" => Some(stdlib.int().clone().to_type()),
        "str" => Some(stdlib.str().clone().to_type()),
        "float" => Some(stdlib.float().clone().to_type()),
        "bool" => Some(stdlib.bool().clone().to_type()),
        "bytes" => Some(stdlib.bytes().clone().to_type()),
        "complex" => Some(stdlib.complex().clone().to_type()),
        "object" => Some(stdlib.object().clone().to_type()),
        "type" => Some(stdlib.builtins_type().clone().to_type()),
        "list" => Some(stdlib.list(Type::any_implicit()).to_type()),
        "dict" => Some(
            stdlib
                .dict(Type::any_implicit(), Type::any_implicit())
                .to_type(),
        ),
        "set" => Some(stdlib.set(Type::any_implicit()).to_type()),
        "tuple" => Some(stdlib.tuple(Type::any_implicit()).to_type()),
        "slice" => {
            let slice = stdlib.slice_class_object();
            Some(ClassType::new(slice, TArgs::default()).to_type())
        }
        "BaseException" => Some(stdlib.base_exception().clone().to_type()),
        "NoneType" => Some(stdlib.none_type().clone().to_type()),
        "EllipsisType" => stdlib.ellipsis_type().map(|t| t.clone().to_type()),
        "function" => Some(stdlib.function_type().clone().to_type()),
        "property" => Some(stdlib.property().clone().to_type()),
        // Add more builtin types as needed
        _ => {
            // For types not directly available from stdlib, try to look them up in builtins exports
            let exports = transaction.get_exports(&builtins_handle);
            if let Some(export_location) = exports.get(&Name::new(type_name)) {
                match export_location {
                    crate::export::exports::ExportLocation::ThisModule(export) => {
                        // Get the type at the export location in the builtins module
                        transaction.get_type_at(&builtins_handle, export.location.start())
                    }
                    crate::export::exports::ExportLocation::OtherModule(_) => {
                        // Builtin type is imported from another module - this shouldn't happen for true builtins
                        None
                    }
                }
            } else {
                None
            }
        }
    }
}

impl TspServer {
    pub fn get_builtin_type(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetBuiltinTypeParams,
    ) -> Result<Option<tsp::Type>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        // Convert Node to URI to get the handle for the scoping node
        let uri = lsp_types::Url::parse(&params.scoping_node.uri).map_err(|_| ResponseError {
            code: ErrorCode::InvalidParams as i32,
            message: "Invalid scoping_node.uri".to_owned(),
            data: None,
        })?;

        // Check if workspace has language services enabled
        let Some(scoping_handle) = self.inner.make_handle_if_enabled(&uri) else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Language services disabled".to_owned(),
                data: None,
            });
        };

        // Call the standalone get_builtin_type function
        let builtin_type = get_builtin_type(transaction, &scoping_handle, &params.name);

        // Convert pyrefly Type to TSP Type format if we found the type
        if let Some(pyrefly_type) = builtin_type {
            // Register the type and get a stable handle, then create TSP type
            let handle_str = self.inner.state.register_type_handle(pyrefly_type.clone());
            let tsp_type =
                crate::tsp::common::convert_to_tsp_type_with_handle(pyrefly_type, handle_str);

            Ok(Some(tsp_type))
        } else {
            Ok(None)
        }
    }
}
