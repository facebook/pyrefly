/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

/// Bit flags for the four dirty states, packed into a single `u8`.
const LOAD: u8 = 1 << 0;
const FIND: u8 = 1 << 1;
const DEPS: u8 = 1 << 2;
const REQUIRE: u8 = 1 << 3;

/// Tracks which parts of a module's state are potentially stale and need
/// recomputation. Each flag is stored as a single bit in a `u8`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Dirty(u8);

impl Dirty {
    /// The result from loading has potentially changed, either
    /// `load_from_memory` on `Loader` (if a memory file path) or
    /// the underlying disk if a disk file path.
    pub fn load(self) -> bool {
        self.0 & LOAD != 0
    }

    /// The result from finding has potentially changed.
    /// Given all data is indexed by `Handle`, the path in the `Handle` can't
    /// change or it would simply represent a different `Handle`.
    /// This instead represents the modules I found from my imports have changed.
    pub fn find(self) -> bool {
        self.0 & FIND != 0
    }

    /// The result I got from my dependencies have potentially changed.
    pub fn deps(self) -> bool {
        self.0 & DEPS != 0
    }

    /// I have increased the amount of data I `Require`.
    pub fn require(self) -> bool {
        self.0 & REQUIRE != 0
    }
}

/// Atomic mutable version of `Dirty`, storing all flags in a single `AtomicU8`.
/// Setting uses `fetch_or` to atomically set individual bits. Cleaning uses
/// a single `swap(0)` to atomically read and clear all flags at once, preventing
/// a race where a flag set between check and clear is silently lost.
#[derive(Debug)]
pub struct AtomicDirty(AtomicU8);

impl AtomicDirty {
    pub fn new(dirty: Dirty) -> Self {
        Self(AtomicU8::new(dirty.0))
    }

    pub fn set_load(&self) {
        self.0.fetch_or(LOAD, Ordering::Release);
    }

    pub fn set_find(&self) {
        self.0.fetch_or(FIND, Ordering::Release);
    }

    pub fn set_deps(&self) {
        self.0.fetch_or(DEPS, Ordering::Release);
    }

    pub fn set_require(&self) {
        self.0.fetch_or(REQUIRE, Ordering::Release);
    }

    /// Try to set the deps flag atomically via CAS.
    /// Returns true if we were the one to set it (it was previously unset).
    pub fn try_set_deps(&self) -> bool {
        let mut old = self.0.load(Ordering::Acquire);
        loop {
            if old & DEPS != 0 {
                return false; // deps already set
            }
            match self
                .0
                .compare_exchange_weak(old, old | DEPS, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(x) => old = x,
            }
        }
    }

    /// Atomically clear the deps flag.
    pub fn clear_deps(&self) {
        self.0.fetch_and(!DEPS, Ordering::Release);
    }

    /// Atomically read all flags and clear them in a single swap.
    /// Returns a `Dirty` snapshot of the flags that were set.
    pub fn take_all(&self) -> Dirty {
        Dirty(self.0.swap(0, Ordering::AcqRel))
    }

    /// Read a snapshot of all flags without clearing.
    pub fn snapshot(&self) -> Dirty {
        Dirty(self.0.load(Ordering::Acquire))
    }
}
