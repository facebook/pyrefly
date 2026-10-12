/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSnapshot TSP request

use std::sync::atomic::Ordering;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspServer;

impl<T: TspInterface> TspServer<T> {
    /// Get the current snapshot version
    ///
    /// The snapshot changes whenever the answers of queries can change: after
    /// an open, an edit, or a close, and after a commit that changes what
    /// Pyrefly reads from outside the editor, such as a file on disk or the
    /// configuration.
    ///
    /// The snapshot is the sum of two counters that only grow: the open-file
    /// events of this server and `TspInterface::generation`. The protocol
    /// carries an `i32`, so the sum wraps. A client only compares snapshots for
    /// equality, so the snapshot still changes exactly when a counter changes.
    pub fn get_snapshot(&self) -> i32 {
        let sum = self
            .open_file_events
            .load(Ordering::Relaxed)
            .wrapping_add(self.inner().generation());
        sum as i32
    }
}
