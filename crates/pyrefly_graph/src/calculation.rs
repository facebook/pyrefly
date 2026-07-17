/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::thread;
use std::thread::ThreadId;

use dupe::Dupe;
use pyrefly_util::lock::Condvar;
use pyrefly_util::lock::Mutex;
use starlark_map::small_set::SmallSet;
use starlark_map::smallset;

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
enum Status {
    /// This value has not yet been calculated.
    NotCalculated,
    /// This value is currently being calculated by `CalcInner::calculating_threads`.
    Calculating,
    /// This value has been calculated.
    Calculated,
}

/// Interior state protected by the mutex.
#[derive(Clone, Debug)]
struct CalcInner {
    // Use a Box so the mutex state stays small. The option is present iff
    // `status` is `Calculating`.
    calculating_threads: Option<Box<SmallSet<ThreadId>>>,
    status: Status,
    /// True when an SCC batch commit has locked this cell for writing.
    /// `record_value` blocks while this is set; reads are unaffected.
    write_locked: bool,
}

/// The result of proposing a calculation in the current thread. See
/// `propose_calculation` for more details on how it is used.
#[derive(Clone, Debug)]
pub enum ProposalResult<T> {
    /// The current thread may proceed with the calculation.
    Calculatable,
    /// The current thread has encountered a cycle.
    CycleDetected,
    /// A final result is already available.
    Calculated(T),
}

/// A cached calculation where recursive calculation returns None.
pub struct Calculation<T> {
    inner: Mutex<CalcInner>,
    /// The final result is written once before the state becomes `Calculated`;
    /// the status inside `inner` is the initialization marker for this cell.
    result: UnsafeCell<MaybeUninit<T>>,
    condvar: Condvar,
}

// SAFETY: `Calculation` writes `result` exactly once while holding `inner`, then
// publishes terminal `Status::Calculated`. After that status is visible, the
// result is never mutated again, so concurrent readers only take shared
// references to initialized data.
unsafe impl<T: Send + Sync> Sync for Calculation<T> {}

impl<T: fmt::Debug> fmt::Debug for Calculation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lock = self.inner.lock();
        // SAFETY: `result` is initialized iff `Status::Calculated` is published,
        // which we observe here under `inner`; it is never mutated afterward.
        let result: &dyn fmt::Debug = if lock.status == Status::Calculated {
            unsafe { (*self.result.get()).assume_init_ref() }
        } else {
            &"<uninitialized>"
        };
        f.debug_struct("Calculation")
            .field("inner", &*lock)
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
            inner: Mutex::new(CalcInner {
                calculating_threads: None,
                status: Status::NotCalculated,
                write_locked: false,
            }),
            result: UnsafeCell::new(MaybeUninit::uninit()),
            condvar: Condvar::new(),
        }
    }
}

impl<T> Drop for Calculation<T> {
    fn drop(&mut self) {
        let initialized = self.inner.get_mut().status == Status::Calculated;
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
        let lock = self.inner.lock();
        match &lock.status {
            Status::Calculated => {
                drop(lock);
                // SAFETY: We observed terminal `Status::Calculated` under
                // `inner`. The result was written before that status was
                // published and is never mutated afterward.
                Some(unsafe { (*self.result.get()).assume_init_ref() }.dupe())
            }
            _ => None,
        }
    }

    /// Look up the current status of the calculation as a `ProposalResult`, under
    /// the assumption that the current thread will begin the calculation if
    /// the result is `Calculatable`.
    /// - If the calculation can proceed (the current thread has not encountered
    ///   a cycle and no other thread has already computed a result), we will
    ///   mark the current thread as active and return `Calculatable`.
    /// - If the current thread encountered a cycle, return `CycleDetected`.
    /// - If the calculation has already been completed, return `Calculated(value)`.
    ///
    /// Does not block on write locks — proposal is unaffected.
    pub fn propose_calculation(&self) -> ProposalResult<T> {
        let mut lock = self.inner.lock();
        match &mut lock.status {
            Status::NotCalculated => {
                lock.calculating_threads = Some(Box::new(smallset! {thread::current().id()}));
                lock.status = Status::Calculating;
                ProposalResult::Calculatable
            }
            Status::Calculating => {
                let threads = lock
                    .calculating_threads
                    .as_mut()
                    .expect("calculating status without calculating threads");
                if threads.insert(thread::current().id()) {
                    ProposalResult::Calculatable
                } else {
                    ProposalResult::CycleDetected
                }
            }
            Status::Calculated => {
                drop(lock);
                // SAFETY: We observed terminal `Status::Calculated` under
                // `inner`. The result was written before that status was
                // published and is never mutated afterward.
                ProposalResult::Calculated(unsafe { (*self.result.get()).assume_init_ref() }.dupe())
            }
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
        let mut lock = self.inner.lock();
        lock = self.condvar.wait_while(lock, |inner| inner.write_locked);
        match &mut lock.status {
            Status::NotCalculated => {
                unreachable!("Should not record a result before calculating")
            }
            Status::Calculating => {
                // SAFETY: We hold `inner`, and `Status::Calculating` means no
                // final result has been written yet. This write happens before
                // publishing terminal `Status::Calculated`.
                unsafe {
                    (*self.result.get()).write(value);
                }
                lock.calculating_threads = None;
                lock.status = Status::Calculated;
                drop(lock);
                // SAFETY: This call just initialized `result` and published
                // terminal `Status::Calculated`; the value will not be mutated.
                (
                    unsafe { (*self.result.get()).assume_init_ref() }.dupe(),
                    true,
                )
            }
            Status::Calculated => {
                // The first thread to write a value wins
                drop(lock);
                // SAFETY: We observed terminal `Status::Calculated` under
                // `inner`. The result was written before that status was
                // published and is never mutated afterward.
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
        let mut lock = self.inner.lock();
        lock = self.condvar.wait_while(lock, |inner| inner.write_locked);
        if matches!(&lock.status, Status::Calculated) {
            false
        } else {
            lock.write_locked = true;
            true
        }
    }

    /// Write a value to a write-locked cell and release the lock.
    /// Panics if the cell is not write-locked.
    pub fn write_unlock(&self, value: T) -> (T, bool) {
        let mut lock = self.inner.lock();
        assert!(lock.write_locked, "write_unlock called on non-locked cell");
        lock.write_locked = false;
        let result = match &mut lock.status {
            Status::NotCalculated => {
                unreachable!("write_unlock called before calculating")
            }
            Status::Calculating => {
                // SAFETY: We hold `inner` and the SCC write lock, and
                // `Status::Calculating` means no final result has been written
                // yet. This write happens before publishing terminal
                // `Status::Calculated`.
                unsafe {
                    (*self.result.get()).write(value);
                }
                lock.calculating_threads = None;
                lock.status = Status::Calculated;
                true
            }
            Status::Calculated => false,
        };
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
        let mut lock = self.inner.lock();
        if lock.write_locked {
            lock.write_locked = false;
            self.condvar.notify_all();
        }
    }

    /// Perform or use the cached result of a calculation without using the full
    /// power of cycle-breaking plumbing.
    ///
    /// Returns `None` if we encounter a cycle.
    pub fn calculate(&self, calculate: impl FnOnce() -> T) -> Option<T> {
        match self.propose_calculation() {
            ProposalResult::Calculatable => {
                let value = calculate();
                let (value, _did_write) = self.record_value(value);
                Some(value)
            }
            ProposalResult::Calculated(v) => Some(v.dupe()),
            ProposalResult::CycleDetected => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn record_value_publishes_one_final_result() {
        let calculation = Calculation::new();

        assert!(matches!(
            calculation.propose_calculation(),
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

        match calculation.propose_calculation() {
            ProposalResult::Calculated(value) => assert_eq!(*value, 1),
            result => panic!("expected calculated result, got {result:?}"),
        }
    }

    #[test]
    fn write_unlock_publishes_one_final_result() {
        let calculation = Calculation::new();

        assert!(matches!(
            calculation.propose_calculation(),
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
