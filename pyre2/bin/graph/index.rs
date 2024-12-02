/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use dupe::Dupe;
use starlark_map::small_map::SmallMap;

#[derive(Clone, Debug)]
pub struct Index<K> {
    map: SmallMap<K, Idx<K>>,
}

impl<K> Default for Index<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Idx<K> {
    // We use a NonZeroUsize to have an optimised representation for Option<Idx>.
    // We treat it as usize, and inc/dec as we store.
    idx: NonZeroUsize,
    phantom: PhantomData<K>,
}

impl<K> Clone for Idx<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for Idx<K> {}

impl<K> Dupe for Idx<K> {}

impl<K> Idx<K> {
    /// Should be used cautiously - make sure this is really a valid index first.
    pub fn new(idx: usize) -> Self {
        Idx {
            idx: NonZeroUsize::new(idx + 1).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn idx(self) -> usize {
        self.idx.get() - 1
    }
}

impl<K> Index<K> {
    pub fn new() -> Self {
        Index {
            map: SmallMap::new(),
        }
    }

    pub fn items(&self) -> impl ExactSizeIterator<Item = (Idx<K>, &K)> {
        self.map.iter().map(|(k, v)| (*v, k))
    }
}

impl<K: Eq + Hash + Debug> Index<K> {
    #[expect(dead_code)] // Logically part of the API
    pub fn insert(&mut self, k: K) -> Idx<K>
    where
        K: Clone,
    {
        let idx = Idx::new(self.map.len());
        let res = self.map.insert(k.clone(), idx);
        if res.is_some() {
            panic!("Duplicate key: {k:?}");
        }
        idx
    }

    pub fn insert_if_missing(&mut self, k: K) -> Idx<K> {
        let idx = Idx::new(self.map.len());
        *self.map.entry(k).or_insert(idx)
    }

    pub fn key_to_idx(&self, k: &K) -> Idx<K>
    where
        K: Clone,
    {
        let idx = self.map.get(k);
        if let Some(idx) = idx {
            *idx
        } else {
            panic!("Key not found: {k:?}");
        }
    }

    pub fn idx_to_key(&self, idx: Idx<K>) -> &K {
        self.map.get_index(idx.idx()).unwrap().0
    }

    /// Does the index contain an element. Should be used very rarely.
    #[expect(dead_code)] // Logically part of the API
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
}
