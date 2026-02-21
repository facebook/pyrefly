/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;

/// Bit flags for the four dirty states, packed into a single `u8`.
const LOAD: u8 = 1 << 0;
const FIND: u8 = 1 << 1;
const DEPS: u8 = 1 << 2;
const REQUIRE: u8 = 1 << 3;

/// Tracks which parts of a module's state are potentially stale and need
/// recomputation. Each flag is stored as a single bit in a `u8`.
#[derive(Debug, Default, Dupe, Clone, Copy)]
pub struct Dirty(u8);

impl Dirty {
    /// Reset all dirty flags to false.
    pub fn clean(&mut self) {
        self.0 = 0;
    }

    /// The result from loading has potentially changed, either
    /// `load_from_memory` on `Loader` (if a memory file path) or
    /// the underlying disk if a disk file path.
    pub fn load(self) -> bool {
        self.0 & LOAD != 0
    }

    pub fn set_load(&mut self) {
        self.0 |= LOAD;
    }

    /// The result from finding has potentially changed.
    /// Given all data is indexed by `Handle`, the path in the `Handle` can't
    /// change or it would simply represent a different `Handle`.
    /// This instead represents the modules I found from my imports have changed.
    pub fn find(self) -> bool {
        self.0 & FIND != 0
    }

    pub fn set_find(&mut self) {
        self.0 |= FIND;
    }

    /// The result I got from my dependencies have potentially changed.
    pub fn deps(self) -> bool {
        self.0 & DEPS != 0
    }

    pub fn set_deps(&mut self) {
        self.0 |= DEPS;
    }

    /// I have increased the amount of data I `Require`.
    pub fn require(self) -> bool {
        self.0 & REQUIRE != 0
    }

    pub fn set_require(&mut self) {
        self.0 |= REQUIRE;
    }
}
