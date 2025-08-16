/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get type request implementation

// use lsp_server::ErrorCode; // removed unused import
use lsp_server::ResponseError;
use tsp_types as tsp;

use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::state::Transaction;
// use crate::tsp::common::lsp_debug; // removed unused import
use crate::tsp::requests::common::node_start_position;
use crate::tsp::server::TspServer;

/// Standalone get_type function that can be used independently of the Server
/// This follows the same pattern as the hover feature
#[allow(dead_code)]
pub fn get_type(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &ModuleInfo,
    params: &tsp::GetTypeParams,
) -> Option<tsp::Type> {
    // Convert range start to TextSize position using module_info
    let position = node_start_position(module_info, &params.node);

    // Try to get the type at the specified position
    let type_info = transaction.get_type_at(handle, position)?;

    // Convert pyrefly Type to TSP Type format using protocol helper
    Some(crate::tsp::common::convert_to_tsp_type(type_info))
}

impl TspServer {
    pub fn get_type(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetTypeParams,
    ) -> Result<Option<tsp::Type>, ResponseError> {
        // Use common helper to validate, get handle, module info and maybe a fresh transaction
        let node_url = lsp_types::Url::parse(&params.node.uri).map_err(|_| ResponseError {
            code: lsp_server::ErrorCode::InvalidParams as i32,
            message: "Invalid node.uri".to_owned(),
            data: None,
        })?;
        let (handle, module_info, transaction_to_use) = self.with_active_transaction(
            transaction,
            &node_url,
            params.snapshot,
            crate::state::require::Require::Everything,
        )?;

        // Use the appropriate transaction (fresh if module was loaded, original if already loaded)
        let active_transaction = transaction_to_use.as_ref().unwrap_or(transaction);

        // Compute position and get internal type once
        let position = node_start_position(&module_info, &params.node);
        let Some(internal_type) = active_transaction.get_type_at(&handle, position) else {
            return Ok(None);
        };

        // Convert and register via server helper
        let tsp_type = self.convert_and_register_type(internal_type);
        Ok(Some(tsp_type))
    }
}
