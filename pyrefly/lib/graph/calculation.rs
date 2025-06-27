/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::thread;
use std::thread::ThreadId;

use dupe::Dupe;
use itertools::Either;
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
/// The type `T` is the final result, while `R` is the value if you
/// hit a recursive loop.
#[derive(Clone, Debug)]
enum Status<T, R> {
    /// This value has not yet been calculated.
    NotCalculated,
    /// This value is currently being calculated by the following threads.
    // Use a Box so the size of the struct stays small
    Calculating(Box<(Option<R>, SmallSet<ThreadId>)>),
    /// This value has been calculated.
    Calculated(T),
}

/// The result of proposing a calculation in the current thread. See
/// `propose_calculation` for more details on how it is used.
#[derive(Clone, Debug)]
pub enum ProposalResult<T, R> {
    /// The current thread may proceed with the calculation.
    Calculatable,
    /// The current thread has encountered a cycle; no recursive placeholder exists yet.
    CycleDetected,
    /// The current thread has encountered a cycle, this is the recursive placeholder.
    CycleBroken(R),
    /// A final reasult is already available.
    Calculated(T),
}

/// A cached calculation where recursive calculation returns None.
#[derive(Debug)]
pub struct Calculation<T, R = ()>(Mutex<Status<T, R>>);

impl<T, R> Default for Calculation<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> Calculation<T, R> {
    pub fn new() -> Self {
        Self(Mutex::new(Status::NotCalculated))
    }
}

impl<T: Dupe, R: Dupe> Calculation<T, R> {
    /// Get the value if it has been calculated, otherwise `None`.
    /// Does not block.
    pub fn get(&self) -> Option<T> {
        let lock = self.0.lock();
        match &*lock {
            Status::Calculated(v) => Some(v.dupe()),
            _ => None,
        }
    }

    /// Look up the current status of the calculation as a `LookupResult`, under
    /// the assumption that the current thread will begin the calculation if
    /// the result is `Status::Calculatable`.
    /// - If the calculation can proceed (the current thread has not encountered
    ///   a cycle and no other thread has already computed a result), we will
    ///   mark the current thread as active and return `Calculatable`.
    /// - If the current thread encountered a cycle and no recursive placeholder
    ///   is yet recorded, return `CycleDetected`.
    /// - If the current thread encountered a cycle and a recursive placeholder
    ///   exists, return `CycleBroken(recursive_placeholder)`.
    /// - If the calculation has already be completed, return `Calculated(value)`.
    pub fn propose_calculation(&self) -> ProposalResult<T, R> {
        let mut lock = self.0.lock();
        match &mut *lock {
            Status::NotCalculated => {
                *lock = Status::Calculating(Box::new((None, smallset! {thread::current().id()})));
                ProposalResult::Calculatable
            }
            Status::Calculating(box (rec, threads)) => {
                if threads.insert(thread::current().id()) {
                    ProposalResult::Calculatable
                } else {
                    match rec {
                        None => ProposalResult::CycleDetected,
                        Some(r) => ProposalResult::CycleBroken(r.dupe()),
                    }
                }
            }
            Status::Calculated(v) => ProposalResult::Calculated(v.dupe()),
        }
    }

    /// Attempt to record a cycle we want to break. Returns:
    /// - The recursive placeholder if there is one; it may not be the one
    ///   we passed in, because the first thread to write a placeholder wins.
    /// - Or, if another thread has already completed the calculation, return
    ///   the final value.
    pub fn record_cycle(&self, placeholder: R) -> Either<T, R> {
        let mut lock = self.0.lock();
        match &mut *lock {
            Status::NotCalculated => {
                unreachable!("Should not record a recursive result before calculating")
            }
            Status::Calculating(box (rec, _)) => {
                if rec.is_none() {
                    // The first thread to write a cycle placeholder wins
                    *rec = Some(placeholder.dupe());
                }
                Either::Right(rec.dupe().unwrap())
            }
            Status::Calculated(v) => Either::Left(v.dupe()),
        }
    }

    /// Attempt to record a calculated value.
    ///
    /// Returns the final value (which may be different from the value passed
    /// in if another thread finished the calculation first) along with
    /// the recursive placeholder, if this thread was the first to write and
    /// one was recorded (the caller, in some cases, may be responsible for
    /// recording a mapping between the placeholder and the final value).
    pub fn record_value(&self, value: T) -> (T, Option<R>) {
        let mut lock = self.0.lock();
        match &mut *lock {
            Status::NotCalculated => {
                unreachable!("Should not record a result before calculating")
            }
            Status::Calculating(box (rec, _)) => {
                let rec = rec.take();
                *lock = Status::Calculated(value.dupe());
                (value, rec)
            }
            Status::Calculated(v) => {
                // The first thread to write a value wins
                (v.dupe(), None)
            }
        }
    }

    /// Perform or use the cached result of a calculation without using the full
    /// power of cycle-breaking plumbing.
    ///
    /// Returns `None` if we encounter a cycle.
    pub fn calculate(&self, calculate: impl FnOnce() -> T) -> Option<T>
    where
        R: Default,
    {
        match self.propose_calculation() {
            ProposalResult::Calculatable => {
                let value = calculate();
                let (value, _) = self.record_value(value);
                Some(value)
            }
            ProposalResult::Calculated(v) => Some(v.dupe()),
            ProposalResult::CycleDetected | ProposalResult::CycleBroken(..) => None,
        }
    }
}
