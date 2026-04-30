/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Lock-free module computation state.
//!
//! This module defines `ModuleState` (frozen, committed form) and `ModuleStateMut`
//! (lock-free mutable form used during transactions). The lock-free design eliminates
//! read-side contention on the module data state lock during transactions.
//!
//! # Ordering Protocol
//!
//! Writer (step computation):
//!   1. Store step data (ArcSwap::store)
//!   2. Release-store `current_step`
//!
//! Clean:
//!   1. Reset step data and `current_step` (relaxed — not yet visible)
//!   2. Release-store `checked` epoch (makes everything visible)
//!
//! Reader:
//!   1. Acquire-load `checked` epoch (synchronizes with clean)
//!   2. Acquire-load `current_step` (synchronizes with writer)
//!   3. Load step data from ArcSwap (returns `Arc` — refcounted, safe)
//!
//! # Key Invariant: No Epoch Changes During Reads
//!
//! A new epoch is only started by `run_step()`. All read paths
//! (`lookup_answer`, `lookup_export`, `demand`) execute within
//! a single step. Therefore, once a reader observes `checked == now`,
//! this condition remains true for the duration of the read.

use std::sync::Arc;

use arc_swap::Guard;
use dupe::Dupe;
use pyrefly_util::lock::Condvar;
use pyrefly_util::lock::Mutex;
use ruff_python_ast::ModModule;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::Solutions;
use crate::binding::bindings::Bindings;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::state::dirty::AtomicComputedDirty;
use crate::state::dirty::Dirty;
use crate::state::epoch::AtomicEpoch;
use crate::state::epoch::Epoch;
use crate::state::epoch::Epochs;
use crate::state::load::Load;
use crate::state::require::AtomicRequire;
use crate::state::require::Require;
use crate::state::steps::Context;
use crate::state::steps::Step;
use crate::state::steps::Steps;
use crate::state::steps::StepsMut;

// ---------------------------------------------------------------------------
// ModuleState — frozen form, stored in committed StateData
// ---------------------------------------------------------------------------

/// Frozen module computation state. Stored in committed `StateData`.
/// Contains plain (non-atomic) values. Created from `ModuleStateMut`
/// via `take_and_freeze`, created into `ModuleStateMut` via `clone_for_mutation`.
#[derive(Debug, Clone)]
pub struct ModuleState {
    pub require: Require,
    pub epochs: Epochs,
    pub dirty: Dirty,
    pub steps: Steps,
}

impl ModuleState {
    pub fn line_count(&self) -> usize {
        self.steps.line_count()
    }

    /// Create a mutable version for use during a transaction.
    pub fn clone_for_mutation(&self) -> ModuleStateMut {
        ModuleStateMut {
            steps: StepsMut::from_frozen(&self.steps),
            checked: AtomicEpoch::new(self.epochs.checked),
            computed_dirty: AtomicComputedDirty::new(self.epochs.computed, self.dirty),
            require: AtomicRequire::new(self.require),
            computing: Mutex::new(false),
            computing_condvar: Condvar::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ModuleStateMut — lock-free mutable form used during transactions
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ModuleStateMut {
    steps: StepsMut,
    checked: AtomicEpoch,
    computed_dirty: AtomicComputedDirty,
    require: AtomicRequire,
    /// True when a thread is computing or cleaning this module.
    computing: Mutex<bool>,
    /// Signaled when `computing` becomes false.
    computing_condvar: Condvar,
}

impl ModuleStateMut {
    pub fn new(require: Require, now: Epoch) -> Self {
        Self {
            steps: StepsMut::new(),
            checked: AtomicEpoch::new(now),
            computed_dirty: AtomicComputedDirty::new(now, Dirty::default()),
            require: AtomicRequire::new(require),
            computing: Mutex::new(false),
            computing_condvar: Condvar::new(),
        }
    }

    // --- Read API ---

    pub fn is_checked(&self, now: Epoch) -> bool {
        self.checked.load() == now
    }

    pub fn next_step(&self) -> Option<Step> {
        self.steps.next_step()
    }

    pub fn last_step(&self) -> Option<Step> {
        self.steps.current_step.load()
    }

    pub fn require(&self) -> Require {
        self.require.load()
    }

    pub fn get_load(&self) -> Option<Arc<Load>> {
        self.steps.load.load_full()
    }

    pub fn get_ast(&self) -> Option<Arc<ModModule>> {
        self.steps.ast.load_full()
    }

    pub fn get_exports(&self) -> Option<Arc<Exports>> {
        self.steps.exports.load_full()
    }

    pub fn get_answers(&self) -> Option<Arc<(Bindings, Arc<Answers>)>> {
        self.steps.answers.load_full()
    }

    /// Borrow the answers via a Guard, avoiding Arc refcount operations.
    /// The Guard keeps the data alive without incrementing the Arc refcount.
    pub fn load_answers(&self) -> Guard<Option<Arc<(Bindings, Arc<Answers>)>>> {
        self.steps.answers.load()
    }

    pub fn get_solutions(&self) -> Option<Arc<Solutions>> {
        self.steps.solutions.load_full()
    }

    /// Borrow the solutions via a Guard, avoiding Arc refcount operations.
    /// The Guard keeps the data alive without incrementing the Arc refcount.
    pub fn load_solutions(&self) -> Guard<Option<Arc<Solutions>>> {
        self.steps.solutions.load()
    }

    pub fn line_count(&self) -> usize {
        self.steps.line_count()
    }

    // --- Compute Guards ---

    /// Blocks until `needed` is computed or this thread should make progress
    /// toward computing it.
    ///
    /// Returns `None` when `needed` is already computed.
    /// Returns `Some(guard)` when this thread should compute the next step.
    pub fn try_start_compute(&self, needed: Step) -> Option<ComputeGuard<'_>> {
        let mut computing = self.computing.lock();
        loop {
            if let Some(todo) = self.steps.next_step()
                && todo <= needed
            {
                if !*computing {
                    *computing = true;
                    return Some(ComputeGuard {
                        state: self,
                        todo,
                        _computing: ComputingFlag { state: self },
                    });
                }
            } else {
                return None;
            }
            computing = self.computing_condvar.wait(computing);
        }
    }

    /// Blocks until this module is checked for `now` or this thread should clean it.
    ///
    /// Returns `None` when the module is already checked for this epoch.
    /// Returns `Some(guard)` when this thread should clean the module.
    pub fn try_start_clean(&self, now: Epoch) -> Option<CleanGuard<'_>> {
        let mut computing = self.computing.lock();
        loop {
            if self.is_checked(now) {
                return None;
            }
            if !*computing {
                *computing = true;
                return Some(CleanGuard {
                    state: self,
                    _computing: ComputingFlag { state: self },
                });
            }
            computing = self.computing_condvar.wait(computing);
        }
    }

    // --- Dirty Marking ---

    /// Set the LOAD dirty flag.
    pub fn set_dirty_load(&self) {
        self.computed_dirty.set_load();
    }

    /// Set the FIND dirty flag.
    pub fn set_dirty_find(&self) {
        self.computed_dirty.set_find();
    }

    /// Set the DEPS dirty flag.
    pub fn set_dirty_deps(&self) {
        self.computed_dirty.set_deps();
    }

    /// Try to mark this module's deps as dirty.
    /// Returns true if we were the one to set the flag (CAS succeeded),
    /// meaning the caller should add this module to the dirty set.
    pub fn try_mark_deps_dirty(&self, now: Epoch) -> bool {
        self.computed_dirty.try_mark_deps_dirty(now)
    }

    /// Increase the require level and set dirty.require if increased.
    /// Returns true if require was actually increased.
    pub fn increase_require(&self, require: Require) -> bool {
        let dirty_require = self.require.increase(require);
        if dirty_require {
            self.computed_dirty.set_require();
        }
        dirty_require
    }

    /// Consume and produce a frozen `ModuleState`.
    pub fn take_and_freeze(self) -> ModuleState {
        let (computed, dirty) = self.computed_dirty.load();
        ModuleState {
            require: self.require(),
            epochs: Epochs {
                checked: self.checked.load(),
                computed,
            },
            dirty,
            steps: self.steps.take_and_freeze(),
        }
    }
}

// ---------------------------------------------------------------------------
// ComputingFlag — RAII handle for the `computing` flag
// ---------------------------------------------------------------------------

/// Clears the `computing` flag and notifies waiters on drop.
struct ComputingFlag<'a> {
    state: &'a ModuleStateMut,
}

impl Drop for ComputingFlag<'_> {
    fn drop(&mut self) {
        let mut computing = self.state.computing.lock();
        *computing = false;
        self.state.computing_condvar.notify_all();
    }
}

// ---------------------------------------------------------------------------
// ComputeGuard — held while computing a step
// ---------------------------------------------------------------------------

/// Guard held while computing a step. The `computing` flag is set while
/// this guard is alive via the `ComputingFlag`.
pub struct ComputeGuard<'a> {
    state: &'a ModuleStateMut,
    /// The next step to compute, determined atomically with the lock
    /// acquisition in `try_start_compute`.
    todo: Step,
    _computing: ComputingFlag<'a>,
}

impl<'a> ComputeGuard<'a> {
    /// The step to compute, determined when the guard was acquired.
    pub fn todo(&self) -> Step {
        self.todo
    }

    pub fn require(&self) -> Require {
        self.state.require()
    }

    /// Compute the step, then release the computing flag and notify waiters.
    /// Returns a `PostComputeGuard` for post-compute work (diffing, eviction).
    pub fn compute<Lookup: LookupExport + LookupAnswer>(
        self,
        ctx: &Context<Lookup>,
    ) -> PostComputeGuard<'a> {
        let ComputeGuard {
            state,
            todo,
            _computing,
        } = self;
        state.steps.compute(todo, ctx);
        drop(_computing);
        PostComputeGuard { state }
    }
}

// ---------------------------------------------------------------------------
// PostComputeGuard — held after computing, lock released
// ---------------------------------------------------------------------------

/// Guard returned by `ComputeGuard::compute()`. The computing flag has been
/// released, so other threads can proceed. Provides access to diffing and
/// eviction operations that are safe without exclusion (step data is in
/// ArcSwap, old data slots are only written during clean).
pub struct PostComputeGuard<'a> {
    state: &'a ModuleStateMut,
}

impl PostComputeGuard<'_> {
    /// Take old exports saved before rebuild for diffing. Clears the slot.
    pub fn take_old_exports(&self) -> Option<Arc<Exports>> {
        self.state.steps.old_exports.swap(None)
    }

    /// Take old answers saved before rebuild for diffing. Clears the slot.
    pub fn take_old_answers(&self) -> Option<Arc<(Bindings, Arc<Answers>)>> {
        self.state.steps.old_answers.swap(None)
    }

    /// Take old solutions saved before rebuild for diffing. Clears the slot.
    pub fn take_old_solutions(&self) -> Option<Arc<Solutions>> {
        self.state.steps.old_solutions.swap(None)
    }

    /// Evict the AST after computing answers (if not needed for retention).
    pub fn evict_ast(&self) {
        debug_assert!(
            self.state.steps.current_step.load() >= Some(Step::Answers),
            "evict_ast called before answers computed"
        );
        self.state.steps.ast.store(None);
    }

    /// Evict answers after computing solutions (if not needed for retention).
    pub fn evict_answers(&self) {
        debug_assert!(
            self.state.steps.current_step.load() >= Some(Step::Solutions),
            "evict_answers called before solutions computed"
        );
        self.state.steps.answers.store(None);
    }
}

// ---------------------------------------------------------------------------
// CleanGuard — held while cleaning a module
// ---------------------------------------------------------------------------

/// Guard held while cleaning a module. The `computing` flag is set while
/// this guard is alive, preventing concurrent computation from starting.
pub struct CleanGuard<'a> {
    state: &'a ModuleStateMut,
    _computing: ComputingFlag<'a>,
}

impl CleanGuard<'_> {
    /// Atomically read and clear all dirty flags in a single operation.
    /// Any flag set after this operation remains set for the next clean cycle.
    pub fn take_dirty(&self) -> Dirty {
        self.state.computed_dirty.take_dirty()
    }

    /// Read load data (under exclusive, for comparison during clean).
    pub fn get_load(&self) -> Option<Arc<Load>> {
        self.state.steps.load.load_full()
    }

    /// Replace the load data. Used during clean to store a new load
    /// before calling `rebuild`.
    pub fn store_load(&self, load: Option<Arc<Load>>) {
        self.state.steps.load.store(load);
    }

    /// Rebuild: reset steps for recomputation, update epochs.
    ///
    /// Uses relaxed writes for `current_step` and step data (not yet visible),
    /// then release-stores `checked = now` (making everything visible).
    /// This works because readers must acquire-load `checked` BEFORE reading
    /// `current_step`.
    ///
    /// `clear_ast`: if true, also clear the AST (e.g., load contents changed).
    pub fn rebuild(&self, clear_ast: bool, now: Epoch) {
        self.state.steps.reset_for_rebuild(clear_ast);

        // Atomically set computed = now and clear all dirty flags.
        //
        // This closes a race window between `take_dirty()` at the start of
        // `clean` and this `rebuild` call: another thread computing our
        // dependency's Solutions step can call `try_mark_deps_dirty`, which
        // checks `computed != now` and sets the DEPS flag. Without clearing
        // here, that DEPS flag would persist and cause a redundant recheck
        // in the next epoch.
        //
        // Clearing DEPS is safe because we are rebuilding: the module will
        // re-demand all its dependencies and get fresh data, making any
        // concurrent DEPS notification redundant.
        //
        // Clearing LOAD, FIND, and REQUIRE is safe because those flags are
        // only set during invalidation (set_memory, config changes, etc.),
        // which happens before transactions run and never races with clean.
        self.state
            .computed_dirty
            .store_computed_and_clear_dirty_relaxed(now);

        // Release-store checked: this is the synchronization point.
        // Any reader that subsequently observes `checked == now` via
        // acquire-load is guaranteed to see all the writes above.
        self.state.checked.store(now);
    }

    /// Finish clean without rebuild: module was not actually dirty.
    /// Release-stores `checked = now`. Dirty flags were already atomically
    /// consumed via `take_dirty_*` by the caller.
    pub fn finish_clean(&self, now: Epoch) {
        // Release-store checked: synchronization point for readers.
        self.state.checked.store(now);
    }
}

// ---------------------------------------------------------------------------
// ModuleStateReader trait — unified read access
// ---------------------------------------------------------------------------

/// Unified read access for both frozen (`ModuleState`) and mutable
/// (`ModuleStateMut`) module state. Allows `with_module_inner` to work
/// with both committed and in-transaction modules.
pub trait ModuleStateReader {
    fn get_load(&self) -> Option<Arc<Load>>;
    fn get_ast(&self) -> Option<Arc<ModModule>>;
    fn get_answers(&self) -> Option<Arc<(Bindings, Arc<Answers>)>>;
    fn get_solutions(&self) -> Option<Arc<Solutions>>;
}

impl ModuleStateReader for ModuleState {
    fn get_load(&self) -> Option<Arc<Load>> {
        self.steps.load.dupe()
    }

    fn get_ast(&self) -> Option<Arc<ModModule>> {
        self.steps.ast.dupe()
    }

    fn get_answers(&self) -> Option<Arc<(Bindings, Arc<Answers>)>> {
        self.steps.answers.dupe()
    }

    fn get_solutions(&self) -> Option<Arc<Solutions>> {
        self.steps.solutions.dupe()
    }
}

impl ModuleStateReader for ModuleStateMut {
    fn get_load(&self) -> Option<Arc<Load>> {
        self.get_load()
    }

    fn get_ast(&self) -> Option<Arc<ModModule>> {
        self.get_ast()
    }

    fn get_answers(&self) -> Option<Arc<(Bindings, Arc<Answers>)>> {
        self.get_answers()
    }

    fn get_solutions(&self) -> Option<Arc<Solutions>> {
        self.get_solutions()
    }
}
