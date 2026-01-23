/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::hash::Hash;
use std::iter;
use std::mem;

use itertools::Either;
use starlark_map::small_map;
use starlark_map::small_map::SmallMap;

/// SmallMap but with at least one element.
/// If only one element is inserted, we won't ever hash it.
#[derive(Debug, Clone)]
pub struct SmallMap1<K, V>(SmallMap1Inner<K, V>);

#[derive(Debug, Clone)]
enum SmallMap1Inner<K, V> {
    One(K, V),
    Map(SmallMap<K, V>),
}

impl<K, V> SmallMap1<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self(SmallMap1Inner::One(key, value))
    }

    /// Insert the key-value pair into the map.
    /// Return the previous value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Hash + Eq,
    {
        match &mut self.0 {
            SmallMap1Inner::One(existing_key, existing_value) => {
                if existing_key == &key {
                    Some(mem::replace(existing_value, value))
                } else {
                    let mut old = SmallMap1Inner::Map(SmallMap::new());
                    mem::swap(&mut self.0, &mut old);
                    let map = match &mut self.0 {
                        SmallMap1Inner::Map(map) => map,
                        _ => unreachable!(),
                    };
                    let (old_key, old_value) = match old {
                        SmallMap1Inner::One(k, v) => (k, v),
                        _ => unreachable!(),
                    };
                    map.insert(old_key, old_value);
                    map.insert(key, value);
                    debug_assert_eq!(map.len(), 2);
                    None
                }
            }
            SmallMap1Inner::Map(map) => map.insert(key, value),
        }
    }

    pub fn first(&self) -> (&K, &V) {
        match &self.0 {
            SmallMap1Inner::One(key, value) => (key, value),
            SmallMap1Inner::Map(map) => map.iter().next().unwrap(),
        }
    }

    /// Get a reference to the value associated with the given key.
    pub fn get(&self, key: &K) -> Option<&V>
    where
        K: Hash + Eq,
    {
        match &self.0 {
            SmallMap1Inner::One(existing_key, value) => {
                if existing_key == key {
                    Some(value)
                } else {
                    None
                }
            }
            SmallMap1Inner::Map(map) => map.get(key),
        }
    }

    /// Get a mutable reference to the value associated with the given key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V>
    where
        K: Hash + Eq,
    {
        match &mut self.0 {
            SmallMap1Inner::One(existing_key, value) => {
                if existing_key == key {
                    Some(value)
                } else {
                    None
                }
            }
            SmallMap1Inner::Map(map) => map.get_mut(key),
        }
    }

    /// Returns an iterator over the keys in the map.
    pub fn iter_keys(&self) -> impl Iterator<Item = &K> {
        match &self.0 {
            SmallMap1Inner::One(key, _) => Either::Left(iter::once(key)),
            SmallMap1Inner::Map(map) => Either::Right(map.keys()),
        }
    }
}

impl<'a, K, V> IntoIterator for &'a SmallMap1<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Either<iter::Once<(&'a K, &'a V)>, small_map::Iter<'a, K, V>>;

    fn into_iter(self) -> Self::IntoIter {
        match &self.0 {
            SmallMap1Inner::One(key, value) => Either::Left(iter::once((key, value))),
            SmallMap1Inner::Map(map) => Either::Right(map.iter()),
        }
    }
}

impl<'a, K, V> IntoIterator for &'a mut SmallMap1<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = Either<iter::Once<(&'a K, &'a mut V)>, small_map::IterMut<'a, K, V>>;
    fn into_iter(self) -> Self::IntoIter {
        match &mut self.0 {
            SmallMap1Inner::One(key, value) => Either::Left(iter::once((key, value))),
            SmallMap1Inner::Map(map) => Either::Right(map.iter_mut()),
        }
    }
}
