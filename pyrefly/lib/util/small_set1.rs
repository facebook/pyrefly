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
use starlark_map::small_set;
use starlark_map::small_set::SmallSet;

/// SmallSet but with at least one element.
/// If only one element is inserted, we won't ever hash it.
#[derive(Debug, Clone)]
pub struct SmallSet1<T>(SmallSet1Inner<T>);

#[derive(Debug, Clone)]
enum SmallSet1Inner<T> {
    One(T),
    Set(SmallSet<T>),
}

impl<T> SmallSet1<T> {
    pub fn new(x: T) -> Self {
        Self(SmallSet1Inner::One(x))
    }

    /// Insert the element into the set.
    /// Return `true` iff the element was inserted.
    pub fn insert(&mut self, x: T) -> bool
    where
        T: Hash + Eq,
    {
        match &mut self.0 {
            SmallSet1Inner::One(item) => {
                if item == &x {
                    false
                } else {
                    let mut old = SmallSet1Inner::Set(SmallSet::new());
                    mem::swap(&mut self.0, &mut old);
                    let set = match &mut self.0 {
                        SmallSet1Inner::Set(set) => set,
                        _ => unreachable!(),
                    };
                    let old = match old {
                        SmallSet1Inner::One(item) => item,
                        _ => unreachable!(),
                    };
                    set.insert(old);
                    set.insert(x);
                    debug_assert_eq!(set.len(), 2);
                    true
                }
            }
            SmallSet1Inner::Set(set) => set.insert(x),
        }
    }

    pub fn first(&self) -> &T {
        match &self.0 {
            SmallSet1Inner::One(item) => item,
            SmallSet1Inner::Set(set) => set.iter().next().unwrap(),
        }
    }
}

impl<'a, T> IntoIterator for &'a SmallSet1<T> {
    type Item = &'a T;
    type IntoIter = Either<iter::Once<&'a T>, small_set::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        match &self.0 {
            SmallSet1Inner::One(item) => Either::Left(iter::once(item)),
            SmallSet1Inner::Set(set) => Either::Right(set.iter()),
        }
    }
}
