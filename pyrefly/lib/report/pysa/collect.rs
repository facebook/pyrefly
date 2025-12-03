/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::hash::Hash;

/// Collects all key-value pairs from the iterator into a HashMap,
/// but fails if there are duplicate keys.
/// This is similar to `.collect::<HashMap<K, V>>()`.
pub fn collect_no_duplicate_keys<K, V>(
    mut iter: impl Iterator<Item = (K, V)>,
) -> Result<HashMap<K, V>, ()>
where
    K: Eq + Hash,
{
    let result = HashMap::with_capacity(iter.size_hint().0);
    iter.try_fold(result, |mut result, (k, v)| {
        if result.insert(k, v).is_some() {
            Err(())
        } else {
            Ok(result)
        }
    })
}

pub trait CollectNoDuplicateKeys {
    type Key;
    type Value;

    /// Collects all key-value pairs from the iterator into a HashMap,
    /// but fails if there are duplicate keys.
    /// This is similar to `.collect::<HashMap<K, V>>()`.
    fn collect_no_duplicate_keys(self) -> Result<HashMap<Self::Key, Self::Value>, ()>
    where
        Self::Key: Eq + Hash;
}

impl<K, V, I> CollectNoDuplicateKeys for I
where
    I: Iterator<Item = (K, V)>,
    K: Eq + Hash,
{
    type Key = K;
    type Value = V;
    fn collect_no_duplicate_keys(self) -> Result<HashMap<Self::Key, Self::Value>, ()> {
        collect_no_duplicate_keys(self)
    }
}
