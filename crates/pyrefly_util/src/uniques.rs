/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unique values produced by a factory.
//! Typically used to produce fresh variables.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use dupe::Dupe;

/// Vend fresh unique `Var`s.
/// Deliberately NOT Clone.
#[derive(Debug)]
pub struct UniqueFactory {
    unique: AtomicUsize,
    /// Global cache for callers that need idempotent Unique allocation.
    /// Given the same key, `get_or_fresh` always returns the same `Unique`
    /// regardless of which thread calls it.
    keyed_cache: Mutex<HashMap<usize, Unique>>,
}

impl Default for UniqueFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique value, provided two given values were produced by the same factory.
#[derive(Debug, Copy, Dupe, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unique(usize);

impl Display for Unique {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "_")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Unique {
    pub const ZERO: Self = Self(0);
}

impl UniqueFactory {
    pub fn new() -> Self {
        Self {
            unique: AtomicUsize::new(1),
            keyed_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn fresh(&self) -> Unique {
        Unique(self.unique.fetch_add(1, Ordering::Relaxed))
    }

    /// Return a cached `Unique` for `key`, allocating a fresh one on first
    /// access. Thread-safe: all threads see the same `Unique` for a given key.
    pub fn get_or_fresh(&self, key: usize) -> Unique {
        let mut cache = self.keyed_cache.lock().unwrap();
        *cache.entry(key).or_insert_with(|| self.fresh())
    }
}
