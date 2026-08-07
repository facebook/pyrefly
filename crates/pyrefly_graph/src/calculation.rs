/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::num::NonZeroU8;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use dupe::Dupe;
use pyrefly_util::lock::Condvar;
use pyrefly_util::lock::Mutex;
use starlark_map::small_set::SmallSet;

/// Recursive calculations by the same thread return None, but
/// if they are different threads they may start calculating.
///
/// We have to allow multiple threads to calculate the same value
/// in parallel, as you may have A, B that mutually recurse.
/// If thread 1 starts on A, then thread 2 starts on B, they will
/// deadlock if they both wait for the other to finish.
///
/// Assumes we don't use async (where recursive context may change
/// which thread is being used).
///
/// The type `T` is the final result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Status {
    /// This value has not yet been calculated.
    NotCalculated,
    /// This value is currently being calculated.
    Calculating,
    /// This value has been calculated.
    Calculated,
}

#[derive(Debug)]
struct AtomicStatus(AtomicU8);

impl AtomicStatus {
    fn new(status: Status) -> Self {
        Self(AtomicU8::new(status as u8))
    }

    /// Acquire-load the status, synchronizing with result publication.
    fn load(&self) -> Status {
        Self::decode(self.0.load(Ordering::Acquire))
    }

    /// Attempt to mark the calculation as started. Returns true if the caller
    /// may calculate, including when another thread is already calculating — so
    /// a `true` result does not imply this call is the only one calculating.
    /// Returns false only if the result is already calculated.
    fn start_calculating(&self) -> bool {
        match self.load() {
            Status::NotCalculated => {
                // Failure may observe the release-published result, so it must
                // acquire. Success must be at least as strong as failure.
                match self.0.compare_exchange(
                    Status::NotCalculated as u8,
                    Status::Calculating as u8,
                    Ordering::Acquire,
                    Ordering::Acquire,
                ) {
                    Ok(_) => true,
                    Err(status) => Self::decode(status) != Status::Calculated,
                }
            }
            Status::Calculating => true,
            Status::Calculated => false,
        }
    }

    /// Publish the initialized result to readers.
    fn store_calculated(&self) {
        self.0.store(Status::Calculated as u8, Ordering::Release);
    }

    fn get_mut(&mut self) -> Status {
        Self::decode(*self.0.get_mut())
    }

    fn decode(status: u8) -> Status {
        match status {
            x if x == Status::NotCalculated as u8 => Status::NotCalculated,
            x if x == Status::Calculating as u8 => Status::Calculating,
            x if x == Status::Calculated as u8 => Status::Calculated,
            _ => unreachable!("invalid calculation status: {status}"),
        }
    }
}

/// The result of proposing a calculation in the current thread. See
/// `propose_calculation` for more details on how it is used.
#[derive(Clone, Debug)]
pub enum ProposalResult<T> {
    /// The current thread may proceed with the calculation.
    Calculatable,
    /// A final result is already available.
    Calculated(T),
}

thread_local! {
    /// Calculations entered through `Calculation::calculate` on this thread.
    static CALCULATING: RefCell<SmallSet<usize>> = const { RefCell::new(SmallSet::new()) };
}

struct CalculationGuard(usize);

impl CalculationGuard {
    fn enter<T>(calculation: &Calculation<T>) -> Option<Self> {
        let key = calculation as *const Calculation<T> as usize;
        CALCULATING.with(|calculating| {
            if calculating.borrow_mut().insert(key) {
                Some(Self(key))
            } else {
                None
            }
        })
    }
}

impl Drop for CalculationGuard {
    fn drop(&mut self) {
        CALCULATING.with(|calculating| {
            assert_eq!(
                calculating.borrow_mut().pop(),
                Some(self.0),
                "calculation guards must be dropped in LIFO order"
            );
        });
    }
}

/// A cached calculation where recursive calculation returns None.
pub struct Calculation<T> {
    /// The monotonic status is the initialization marker for `result`.
    status: AtomicStatus,
    /// True when an SCC batch commit has locked this cell for writing.
    /// `record_value` blocks while this is set; reads are unaffected.
    write_locked: Mutex<bool>,
    /// The final result is written once before `status` becomes `Calculated`.
    result: UnsafeCell<MaybeUninit<T>>,
    condvar: Condvar,
    /// Keeps `Option<Calculation<T>>` the same size as `Calculation<T>`.
    _niche: NonZeroU8,
}

// SAFETY: `Calculation` writes `result` exactly once while holding `write_locked`,
// then publishes terminal `Status::Calculated`. After that status is visible, the
// result is never mutated again, so concurrent readers only take shared
// references to initialized data.
unsafe impl<T: Send + Sync> Sync for Calculation<T> {}

impl<T: fmt::Debug> fmt::Debug for Calculation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = self.status.load();
        let write_locked = self.write_locked.lock();
        // SAFETY: The acquire load observed the release publication of
        // `Status::Calculated`, after which `result` is never mutated.
        let result: &dyn fmt::Debug = if status == Status::Calculated {
            unsafe { (*self.result.get()).assume_init_ref() }
        } else {
            &"<uninitialized>"
        };
        f.debug_struct("Calculation")
            .field("status", &status)
            .field("write_locked", &*write_locked)
            .field("result", result)
            .field("condvar", &self.condvar)
            .finish()
    }
}

impl<T> Default for Calculation<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Calculation<T> {
    pub fn new() -> Self {
        Self {
            status: AtomicStatus::new(Status::NotCalculated),
            write_locked: Mutex::new(false),
            result: UnsafeCell::new(MaybeUninit::uninit()),
            condvar: Condvar::new(),
            _niche: NonZeroU8::MIN,
        }
    }
}

impl<T> Drop for Calculation<T> {
    fn drop(&mut self) {
        let initialized = self.status.get_mut() == Status::Calculated;
        if initialized {
            // SAFETY: `Status::Calculated` is published only after `result` is
            // initialized, and `drop` has exclusive access to the calculation.
            unsafe {
                self.result.get_mut().assume_init_drop();
            }
        }
    }
}

impl<T: Dupe> Calculation<T> {
    /// Get the value if it has been calculated, otherwise `None`.
    /// Does not block on write locks — reads are unaffected.
    pub fn get(&self) -> Option<T> {
        match self.status.load() {
            Status::Calculated => {
                // SAFETY: The acquire load observed the release publication of
                // `Status::Calculated`, after which `result` is never mutated.
                Some(unsafe { (*self.result.get()).assume_init_ref() }.dupe())
            }
            _ => None,
        }
    }

    /// Look up the current status of the calculation as a `ProposalResult`, under
    /// the assumption that the current thread will begin the calculation if
    /// the result is `Calculatable`.
    /// - If no other thread has already computed a result, return `Calculatable`.
    ///   Multiple threads may calculate the value concurrently.
    /// - If the calculation has already been completed, return `Calculated(value)`.
    ///
    /// Does not block on write locks — proposal is unaffected.
    ///
    /// # Safety
    ///
    /// The caller must detect same-thread calculation cycles before recursively
    /// evaluating a `Calculatable` result. A cell that is already `Calculating`
    /// deliberately returns `Calculatable` so different threads can calculate
    /// concurrently; without caller-side cycle detection, a cyclic calculation
    /// will recurse indefinitely.
    pub unsafe fn propose_calculation(&self) -> ProposalResult<T> {
        if self.status.start_calculating() {
            ProposalResult::Calculatable
        } else {
            // SAFETY: start_calculating returned false after an acquire operation
            // observed the release publication of `Status::Calculated`, after
            // which `result` is never mutated.
            ProposalResult::Calculated(unsafe { (*self.result.get()).assume_init_ref() }.dupe())
        }
    }

    /// Attempt to record a calculated value.
    ///
    /// Blocks while the cell is write-locked by an SCC batch commit.
    ///
    /// Returns `(final_value, did_write)` where:
    /// - `final_value` is the value that was recorded (which may be different from
    ///   the value passed in if another thread finished the calculation first)
    /// - `did_write` is `true` if this call was the one that wrote the value,
    ///   `false` if another thread had already written it
    pub fn record_value(&self, value: T) -> (T, bool) {
        if let Some(value) = self.get() {
            return (value, false);
        }
        let mut lock = self.write_locked.lock();
        lock = self.condvar.wait_while(lock, |write_locked| *write_locked);
        match self.status.load() {
            Status::NotCalculated => {
                unreachable!("Should not record a result before calculating")
            }
            Status::Calculating => {
                // SAFETY: We hold `write_locked`, and `Status::Calculating` means no
                // final result has been written yet. This write happens before
                // publishing terminal `Status::Calculated`.
                unsafe {
                    (*self.result.get()).write(value);
                }
                self.status.store_calculated();
                drop(lock);
                // SAFETY: This call initialized `result` before publishing
                // `Status::Calculated`; the value will not be mutated.
                (
                    unsafe { (*self.result.get()).assume_init_ref() }.dupe(),
                    true,
                )
            }
            Status::Calculated => {
                // The first thread to write a value wins
                drop(lock);
                // SAFETY: The acquire load observed the release publication of
                // `Status::Calculated`, after which `result` is never mutated.
                (
                    unsafe { (*self.result.get()).assume_init_ref() }.dupe(),
                    false,
                )
            }
        }
    }

    /// Lock this cell for an SCC batch commit. Blocks if another SCC commit
    /// already holds the lock. Returns false (no lock acquired) if the cell
    /// is already `Calculated`, since `record_value` would be a no-op anyway.
    pub fn write_lock(&self) -> bool {
        if self.status.load() == Status::Calculated {
            return false;
        }
        let mut lock = self.write_locked.lock();
        lock = self.condvar.wait_while(lock, |write_locked| *write_locked);
        if self.status.load() == Status::Calculated {
            false
        } else {
            *lock = true;
            true
        }
    }

    /// Write a value to a write-locked cell and release the lock.
    /// Panics if the cell is not write-locked.
    pub fn write_unlock(&self, value: T) -> (T, bool) {
        let mut lock = self.write_locked.lock();
        assert!(*lock, "write_unlock called on non-locked cell");
        *lock = false;
        let result = match self.status.load() {
            Status::NotCalculated => {
                unreachable!("write_unlock called before calculating")
            }
            Status::Calculating => {
                // SAFETY: We hold the `write_locked` mutex and the SCC write
                // lock, and `Status::Calculating` means no final result has been written
                // yet. This write happens before publishing terminal
                // `Status::Calculated`.
                unsafe {
                    (*self.result.get()).write(value);
                }
                self.status.store_calculated();
                true
            }
            Status::Calculated => false,
        };
        // The predicate change and notification happen under the mutex used by
        // wait_while, so a waiter cannot observe the old value and miss this wakeup.
        self.condvar.notify_all();
        drop(lock);
        // SAFETY: Either this call wrote `result` and published terminal
        // `Status::Calculated`, or it observed that another writer had already
        // done so while holding `inner`.
        (
            unsafe { (*self.result.get()).assume_init_ref() }.dupe(),
            result,
        )
    }

    /// Release the write lock without writing a value.
    /// Used by the RAII guard for panic cleanup.
    pub fn write_unlock_empty(&self) {
        let mut lock = self.write_locked.lock();
        if *lock {
            *lock = false;
            // Keep the predicate change and notification under the wait mutex
            // to avoid a lost wakeup.
            self.condvar.notify_all();
        }
    }

    /// Perform or use the cached result of a calculation without using the full
    /// power of cycle-breaking plumbing.
    ///
    /// Returns `None` if we encounter a cycle.
    pub fn calculate(&self, calculate: impl FnOnce() -> T) -> Option<T> {
        // SAFETY: CalculationGuard::enter rejects same-thread re-entry before
        // the calculation callback is evaluated.
        match unsafe { self.propose_calculation() } {
            ProposalResult::Calculatable => {
                let _guard = CalculationGuard::enter(self)?;
                let value = calculate();
                let (value, _did_write) = self.record_value(value);
                Some(value)
            }
            ProposalResult::Calculated(v) => Some(v.dupe()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn option_has_no_size_overhead() {
        assert_eq!(
            size_of::<Calculation<Arc<usize>>>(),
            size_of::<Option<Calculation<Arc<usize>>>>(),
            "the explicit niche should make Option storage free",
        );
    }

    #[test]
    fn record_value_publishes_one_final_result() {
        let calculation = Calculation::new();

        assert!(matches!(
            // SAFETY: This test publishes a value without recursively calculating.
            unsafe { calculation.propose_calculation() },
            ProposalResult::Calculatable
        ));
        assert!(calculation.get().is_none());

        let (value, did_write) = calculation.record_value(Arc::new(1));
        assert!(did_write);
        assert_eq!(*value, 1);
        assert_eq!(*calculation.get().unwrap(), 1);

        let (value, did_write) = calculation.record_value(Arc::new(2));
        assert!(!did_write);
        assert_eq!(*value, 1);

        // SAFETY: The calculation is already complete, so no recursion can occur.
        match unsafe { calculation.propose_calculation() } {
            ProposalResult::Calculated(value) => assert_eq!(*value, 1),
            result => panic!("expected calculated result, got {result:?}"),
        }
    }

    #[test]
    fn concurrent_proposals_publish_one_final_result() {
        let calculation = Calculation::new();

        assert!(matches!(
            // SAFETY: This test makes a proposal without evaluating dependencies.
            unsafe { calculation.propose_calculation() },
            ProposalResult::Calculatable
        ));
        assert!(matches!(
            // SAFETY: This test makes a concurrent proposal without recursing.
            unsafe { calculation.propose_calculation() },
            ProposalResult::Calculatable
        ));

        let (value, did_write) = calculation.record_value(Arc::new(1));
        assert!(did_write);
        assert_eq!(*value, 1);

        let (value, did_write) = calculation.record_value(Arc::new(2));
        assert!(!did_write);
        assert_eq!(*value, 1);
        // SAFETY: The calculation is already complete, so no recursion can occur.
        match unsafe { calculation.propose_calculation() } {
            ProposalResult::Calculated(value) => assert_eq!(*value, 1),
            result => panic!("expected calculated result, got {result:?}"),
        }
    }

    #[test]
    fn calculate_detects_recursion() {
        let calculation: Calculation<Arc<usize>> = Calculation::new();
        let detected_cycle = Cell::new(false);

        let value = calculation.calculate(|| {
            detected_cycle.set(calculation.calculate(|| Arc::new(1)).is_none());
            Arc::new(2)
        });

        assert!(detected_cycle.get());
        assert_eq!(*value.unwrap(), 2);
    }

    #[test]
    fn write_unlock_publishes_one_final_result() {
        let calculation = Calculation::new();

        assert!(matches!(
            // SAFETY: This test writes the value directly without recursively calculating.
            unsafe { calculation.propose_calculation() },
            ProposalResult::Calculatable
        ));
        assert!(calculation.write_lock());

        let (value, did_write) = calculation.write_unlock(Arc::new(1));
        assert!(did_write);
        assert_eq!(*value, 1);
        assert_eq!(*calculation.get().unwrap(), 1);
        assert!(!calculation.write_lock());
    }
}
