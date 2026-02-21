/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use dupe::Dupe;

#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq)]
pub struct Epoch(u32);

impl Epoch {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn next(&mut self) {
        self.0 += 1;
    }
}

/// Invariant: checked >= computed >= changed
///
/// Whenever we confirm this data is up to date, we bump checked.
/// If to do so, we have to recompute everything, we bump computed.
/// If when computing it we change the data, we bump changed.
///
/// If A depends on B, and A.computed < B.changed, then A is stale.
#[derive(Debug, Clone, Copy)]
pub struct Epochs {
    /// The point at which we have validated that everything is correct.
    pub checked: Epoch,
    /// The point at which we last computed everything.
    pub computed: Epoch,
}

#[derive(Debug)]
pub struct AtomicEpoch(AtomicU32);

impl AtomicEpoch {
    pub fn new(epoch: Epoch) -> Self {
        Self(AtomicU32::new(epoch.0))
    }

    pub fn load(&self) -> Epoch {
        Epoch(self.0.load(Ordering::Acquire))
    }

    pub fn store(&self, epoch: Epoch) {
        self.0.store(epoch.0, Ordering::Release);
    }

    /// Used when the write is followed by a Release store on a
    /// different variable that readers check first (e.g., `checked`
    /// synchronizes reads of `current_step` and `computed`).
    pub fn store_relaxed(&self, epoch: Epoch) {
        self.0.store(epoch.0, Ordering::Relaxed);
    }
}
