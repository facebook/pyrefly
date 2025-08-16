/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP get docstring request implementation

use lsp_server::ResponseError;
use pyrefly_python::docstring::Docstring;
use tsp_types as tsp;

use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

/// Extract docstring from a transaction at a specific position
///
/// This is the core logic for getting docstrings that can be used independently
/// of the Server implementation for unit testing.
pub fn get_docstring_at_position(
    transaction: &Transaction<'_>,
    handle: &crate::state::handle::Handle,
    node: &tsp::Node,
) -> Option<String> {
    // Get module info for position conversion
    let module_info = transaction.get_module_info(handle)?;

    // Convert Range position to TextSize using the module's line buffer
    let position = module_info
        .lined_buffer()
        .from_lsp_position(tsp_types::to_lsp_position(&node.range.start));

    // Try to find definition at the position - this is the same logic as hover
    let first_definition = transaction
        .find_definition(handle, position, &FindPreference::default())
        .into_iter()
        .next()?;

    let _definition_metadata = &first_definition.metadata;
    let _definition_range = first_definition.definition_range;
    let docstring_range = first_definition.docstring_range?;

    // Get the module info for this handle
    let module_info = transaction.get_load(handle)?;

    // Use the Docstring class to properly format the docstring, same as hover
    let docstring = Docstring(docstring_range, module_info.module_info.clone());
    Some(docstring.resolve())
}

impl TspServer {
    pub fn get_docstring(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetDocstringParams,
    ) -> Result<Option<String>, ResponseError> {
        // Validate and obtain handle/module info (and potentially a fresh transaction)
        let Some(node) = &params.decl.node else {
            return Ok(None);
        };

        let node_url = lsp_types::Url::parse(&node.uri).map_err(|_| ResponseError {
            code: lsp_server::ErrorCode::InvalidParams as i32,
            message: "Invalid node.uri".to_owned(),
            data: None,
        })?;
        let (handle, _module_info, maybe_fresh_tx) = self.with_active_transaction(
            transaction,
            &node_url,
            params.snapshot,
            crate::state::require::Require::Everything,
        )?;

        let active_tx = maybe_fresh_tx.as_ref().unwrap_or(transaction);

        // Use the common function for getting docstring at position
        Ok(get_docstring_at_position(active_tx, &handle, node))
    }
}
