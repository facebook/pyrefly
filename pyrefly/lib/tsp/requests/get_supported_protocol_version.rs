/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSupportedProtocolVersion TSP request

use tsp_types::TSP_PROTOCOL_VERSION;
use tsp_types::protocol::TypeServerVersion;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspServer;

impl<T: TspInterface> TspServer<T> {
    pub fn get_supported_protocol_version(&self) -> TypeServerVersion {
        // Return the current protocol version from the generated enum
        TSP_PROTOCOL_VERSION
    }
}
