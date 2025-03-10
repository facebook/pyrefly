/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A trait for values that have a mutable and immutable part.

use std::hash::Hasher;

/// A trait for values that have a mutable and immutable part.
pub trait Mutable {
    /// Equality for the immutable part of the value.
    /// Values which return `true` must also produce the same hash.
    fn immutable_eq(&self, other: &Self) -> bool;

    /// Hash for the immutable part of the value.
    fn immutable_hash<H: Hasher>(&self, state: &mut H);

    /// Mutate the mutable part of the value.
    /// If `immutable_eq` returns `true` for both values,
    /// then after `mutate` the values should be fully equal.
    fn mutate(&self, x: &Self);
}
