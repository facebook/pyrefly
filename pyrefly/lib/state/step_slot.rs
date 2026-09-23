/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use arc_swap::ArcSwapOption;

/// An ArcSwap slot whose debt-bearing guards never outlive a
/// [`StepSlot::with`] callback.
#[derive(Debug)]
pub(super) struct StepSlot<T> {
    inner: ArcSwapOption<T>,
}

impl<T> StepSlot<T> {
    pub fn new(value: Option<Arc<T>>) -> Self {
        Self {
            inner: ArcSwapOption::new(value),
        }
    }

    pub fn clone_arc(&self) -> Option<Arc<T>> {
        self.inner.load_full()
    }

    pub fn with<R>(&self, f: impl for<'a> FnOnce(Option<&'a T>) -> R) -> R {
        let guard = self.inner.load();
        f(guard.as_deref())
    }

    pub fn store(&self, value: Option<Arc<T>>) {
        self.inner.store(value);
    }

    pub fn swap(&self, value: Option<Arc<T>>) -> Option<Arc<T>> {
        self.inner.swap(value)
    }

    pub fn into_inner(self) -> Option<Arc<T>> {
        self.inner.into_inner()
    }
}
