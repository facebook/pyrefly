/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSupportedProtocolVersion TSP request

use tsp_types::TSP_PROTOCOL_VERSION;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspServer;

impl<T: TspInterface> TspServer<T> {
    pub fn get_supported_protocol_version(&self) -> String {
        // Return the hardcoded protocol version (compat shim)
        TSP_PROTOCOL_VERSION.to_owned()
    }
}
