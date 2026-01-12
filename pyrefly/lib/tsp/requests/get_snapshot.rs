/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSnapshot TSP request

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspServer;

impl<T: TspInterface> TspServer<T> {
    /// Get the current snapshot version
    ///
    /// The snapshot represents the current epoch of the global state.
    /// It changes whenever files are modified, configuration changes,
    /// or any other event that would trigger a recomputation.
    pub fn get_snapshot(&self) -> i32 {
        *self.current_snapshot.lock().unwrap_or_else(|poisoned| {
            // In case of poisoned mutex, recover and return the value
            eprintln!("TSP: Warning - snapshot mutex was poisoned, recovering");
            poisoned.into_inner()
        })
    }
}
