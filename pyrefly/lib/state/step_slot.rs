/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;
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

    /// Consume the slot without ArcSwap's global debt handoff.
    ///
    /// `ArcSwapAny::into_inner` calls the strategy's `wait_for_readers` [1]; the
    /// default hybrid strategy pays every debt in the process-global list [2].
    /// `StepSlot` never exposes a guard — every borrowed read is
    /// callback-scoped — so consuming `self` proves all such guards have been
    /// dropped, and the stored reference can be released without waiting.
    ///
    /// [1]: https://github.com/vorner/arc-swap/blob/147d6c0319d389a0aaa134a67abaa00106122f7d/src/lib.rs#L406-L412
    /// [2]: https://github.com/vorner/arc-swap/blob/147d6c0319d389a0aaa134a67abaa00106122f7d/src/debt/mod.rs#L80-L110
    pub fn into_inner(self) -> Option<Arc<T>> {
        let value = self.inner.load_full();
        mem::forget(self.inner);
        value.map(|value| {
            // `load_full` created one reference while the forgotten slot retains
            // another. Round-trip the owned reference through `into_raw` so the
            // pointer satisfies the raw Arc API's provenance contract, then
            // release the reference retained by the slot.
            let ptr = Arc::into_raw(value);
            // SAFETY: `ptr` came from `Arc::into_raw` immediately above.
            let value = unsafe { Arc::from_raw(ptr) };
            // SAFETY: `ptr` came from `Arc::into_raw`, and `value` keeps the
            // allocation alive while the forgotten slot's reference is released.
            unsafe { Arc::decrement_strong_count(ptr) };
            value
        })
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;
    use std::sync::Arc;

    use super::StepSlot;

    #[test]
    fn into_inner_transfers_the_slot_reference() {
        let value = Arc::new(17);
        let slot = StepSlot::new(Some(value.clone()));

        assert_eq!(slot.with(|current| current.copied()), Some(17));
        assert_eq!(Arc::strong_count(&value), 2);

        let taken = slot
            .into_inner()
            .expect("the populated slot should yield its value");
        assert!(Arc::ptr_eq(&value, &taken));
        assert_eq!(Arc::strong_count(&value), 2);
        drop(taken);
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    fn callback_guard_is_dropped_during_unwind() {
        let value = Arc::new(17);
        let slot = StepSlot::new(Some(value.clone()));

        let result = catch_unwind(AssertUnwindSafe(|| {
            slot.with::<()>(|_| panic!("stop inside callback"));
        }));
        assert!(result.is_err());

        let taken = slot
            .into_inner()
            .expect("unwinding should drop the internal guard");
        assert!(Arc::ptr_eq(&value, &taken));
        assert_eq!(Arc::strong_count(&value), 2);
    }

    #[test]
    fn into_inner_preserves_an_empty_slot() {
        let slot = StepSlot::<u8>::new(None);
        assert!(slot.into_inner().is_none());
    }
}
