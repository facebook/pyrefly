/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSupportedProtocolVersion TSP request

use tsp_types::TSP_PROTOCOL_VERSION;

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_supported_protocol_version(&self, _transaction: &Transaction<'_>) -> String {
        // Return the hardcoded protocol version (compat shim)
        TSP_PROTOCOL_VERSION.to_owned()
    }
}
