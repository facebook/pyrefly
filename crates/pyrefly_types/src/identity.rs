/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ord;
use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::types::Type;

/// A wrapper for auxiliary data whose identity should be completely ignored
/// in equality, hashing, ordering, and type-equality comparisons.
/// `IdentityIgnored<T>` always compares as equal, hashes as a no-op, and
/// orders as `Equal` — making it transparent to all identity checks.
///
/// This is useful for attaching auxiliary data (e.g. closure caches) to
/// types that derive `PartialEq`, `Hash`, `Ord`, etc. without affecting
/// their logical identity.
#[derive(Debug, Clone)]
pub struct IdentityIgnored<T>(pub T);

impl<T> PartialEq for IdentityIgnored<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> Eq for IdentityIgnored<T> {}

impl<T> Hash for IdentityIgnored<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<T> PartialOrd for IdentityIgnored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for IdentityIgnored<T> {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl<T> Visit<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl<T> VisitMut<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl<T> TypeEq for IdentityIgnored<T> {
    fn type_eq(&self, _other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        true
    }
}

impl<T> Deref for IdentityIgnored<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
