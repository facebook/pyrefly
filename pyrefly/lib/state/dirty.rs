/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::state::epoch::Epoch;

/// Bit flags for the four dirty states, packed into a single `u8`.
const LOAD: u8 = 1 << 0;
const FIND: u8 = 1 << 1;
const DEPS: u8 = 1 << 2;
const REQUIRE: u8 = 1 << 3;

/// Bit shift for dirty flags within the packed u64.
const DIRTY_SHIFT: u32 = 32;

/// Mask for the epoch portion (bits 0-31).
const EPOCH_MASK: u64 = 0xFFFF_FFFF;

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

// ---------------------------------------------------------------------------
// AtomicComputedDirty — combined atomic storage for epoch + dirty flags
// ---------------------------------------------------------------------------

/// Combined atomic storage for `computed` epoch (u32) and `dirty` flags (u8).
/// Layout: bits 0–31 = epoch (u32), bits 32–39 = dirty flags (u8).
///
/// These are combined into a single `AtomicU64` so that `try_mark_deps_dirty`
/// can atomically check `computed != now` and set the DEPS flag in a single
/// CAS operation. With separate atomics, these would be two operations with
/// no way to make them atomic together.
#[derive(Debug)]
pub struct AtomicComputedDirty(AtomicU64);

impl AtomicComputedDirty {
    pub fn new(epoch: Epoch, dirty: Dirty) -> Self {
        Self(AtomicU64::new(Self::pack(epoch, dirty)))
    }

    /// Load both epoch and dirty atomically.
    pub fn load(&self) -> (Epoch, Dirty) {
        let packed = self.0.load(Ordering::Acquire);
        (Self::unpack_epoch(packed), Self::unpack_dirty(packed))
    }

    /// Atomically set the LOAD dirty flag.
    pub fn set_load(&self) {
        self.0
            .fetch_or((LOAD as u64) << DIRTY_SHIFT, Ordering::Release);
    }

    /// Atomically set the FIND dirty flag.
    pub fn set_find(&self) {
        self.0
            .fetch_or((FIND as u64) << DIRTY_SHIFT, Ordering::Release);
    }

    /// Atomically set the DEPS dirty flag.
    pub fn set_deps(&self) {
        self.0
            .fetch_or((DEPS as u64) << DIRTY_SHIFT, Ordering::Release);
    }

    /// Atomically set the REQUIRE dirty flag.
    pub fn set_require(&self) {
        self.0
            .fetch_or((REQUIRE as u64) << DIRTY_SHIFT, Ordering::Release);
    }

    /// Try to mark deps dirty atomically.
    ///
    /// Returns true if we were the one to set the DEPS flag (meaning the
    /// caller should add this module to the dirty set).
    ///
    /// This is the key race-free implementation: in a single CAS loop we
    /// check `computed != now` AND set the DEPS flag. If another thread
    /// updates `computed` to `now` concurrently, the CAS fails (epoch bits
    /// changed), we retry, see `computed == now`, and return `false`.
    pub fn try_mark_deps_dirty(&self, now: Epoch) -> bool {
        let now_u32 = now.as_u32();
        let mut packed = self.0.load(Ordering::Acquire);
        loop {
            let epoch = packed as u32;
            let dirty = (packed >> DIRTY_SHIFT) as u8;

            // If already computed in this epoch, skip.
            if epoch == now_u32 {
                return false;
            }

            // If DEPS already set, another thread already marked it.
            if dirty & DEPS != 0 {
                return false;
            }

            // Try to set DEPS atomically.
            let new_packed = packed | ((DEPS as u64) << DIRTY_SHIFT);
            match self.0.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(x) => packed = x,
            }
        }
    }

    /// Atomically read all dirty flags and clear them, keeping the epoch.
    /// Returns a `Dirty` snapshot of the flags that were set.
    pub fn take_dirty(&self) -> Dirty {
        let mut packed = self.0.load(Ordering::Acquire);
        loop {
            let dirty = Self::unpack_dirty(packed);
            let new_packed = packed & EPOCH_MASK; // keep epoch, zero dirty bits
            match self.0.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return dirty,
                Err(x) => packed = x,
            }
        }
    }

    /// Update the epoch portion, preserving dirty flags.
    /// Uses Relaxed ordering — relies on a subsequent release-store of
    /// `checked` for synchronization.
    pub fn store_computed_relaxed(&self, epoch: Epoch) {
        let new_epoch = epoch.as_u32() as u64;
        let mut packed = self.0.load(Ordering::Relaxed);
        loop {
            let new_packed = (packed & !EPOCH_MASK) | new_epoch;
            match self.0.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(x) => packed = x,
            }
        }
    }

    // --- Helper functions ---

    /// Pack epoch and dirty into a single u64.
    fn pack(epoch: Epoch, dirty: Dirty) -> u64 {
        (epoch.as_u32() as u64) | ((dirty.0 as u64) << DIRTY_SHIFT)
    }

    /// Unpack epoch from the packed u64.
    fn unpack_epoch(packed: u64) -> Epoch {
        Epoch::from_u32(packed as u32)
    }

    /// Unpack dirty flags from the packed u64.
    fn unpack_dirty(packed: u64) -> Dirty {
        Dirty((packed >> DIRTY_SHIFT) as u8)
    }
}
