/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
#[derive(Clone, Debug)]
enum Status<T> {
    /// This value has not yet been calculated.
    NotCalculated,
    /// This value is currently being calculated by the following threads.
    // Use a Box so the size of the struct stays small
    Calculating(Box<SmallSet<ThreadId>>),
    /// This value has been calculated.
    Calculated(T),
}

/// Interior state protected by the mutex.
#[derive(Clone, Debug)]
struct CalcInner<T> {
    status: Status<T>,
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
#[derive(Debug)]
pub struct Calculation<T> {
    inner: Mutex<CalcInner<T>>,
    condvar: Condvar,
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
                status: Status::NotCalculated,
                write_locked: false,
            }),
            condvar: Condvar::new(),
        }
    }
}

impl<T: Dupe> Calculation<T> {
    /// Get the value if it has been calculated, otherwise `None`.
    /// Does not block on write locks — reads are unaffected.
    pub fn get(&self) -> Option<T> {
        let lock = self.inner.lock();
        match &lock.status {
            Status::Calculated(v) => Some(v.dupe()),
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
                lock.status = Status::Calculating(Box::new(smallset! {thread::current().id()}));
                ProposalResult::Calculatable
            }
            Status::Calculating(threads) => {
                if threads.insert(thread::current().id()) {
                    ProposalResult::Calculatable
                } else {
                    ProposalResult::CycleDetected
                }
            }
            Status::Calculated(v) => ProposalResult::Calculated(v.dupe()),
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
            Status::Calculating(_) => {
                lock.status = Status::Calculated(value.dupe());
                (value, true)
            }
            Status::Calculated(v) => {
                // The first thread to write a value wins
                (v.dupe(), false)
            }
        }
    }

    /// Lock this cell for an SCC batch commit. Blocks if another SCC commit
    /// already holds the lock. Returns false (no lock acquired) if the cell
    /// is already `Calculated`, since `record_value` would be a no-op anyway.
    pub fn write_lock(&self) -> bool {
        let mut lock = self.inner.lock();
        lock = self.condvar.wait_while(lock, |inner| inner.write_locked);
        match lock.status {
            Status::Calculated(_) => false,
            _ => {
                lock.write_locked = true;
                true
            }
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
            Status::Calculating(_) => {
                lock.status = Status::Calculated(value.dupe());
                (value, true)
            }
            Status::Calculated(v) => (v.dupe(), false),
        };
        self.condvar.notify_all();
        result
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
