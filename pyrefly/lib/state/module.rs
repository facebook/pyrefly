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

use dupe::Dupe;
use pyrefly_util::exclusive_lock::ExclusiveLock;
use pyrefly_util::exclusive_lock::ExclusiveLockGuard;
use ruff_python_ast::ModModule;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::Solutions;
use crate::binding::bindings::Bindings;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::state::dirty::AtomicDirty;
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
            computed: AtomicEpoch::new(self.epochs.computed),
            dirty: AtomicDirty::new(self.dirty),
            require: AtomicRequire::new(self.require),
            compute_lock: ExclusiveLock::default(),
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
    computed: AtomicEpoch,
    pub dirty: AtomicDirty,
    require: AtomicRequire,
    compute_lock: ExclusiveLock<Step>,
}

impl ModuleStateMut {
    pub fn new(require: Require, now: Epoch) -> Self {
        Self {
            steps: StepsMut::new(),
            checked: AtomicEpoch::new(now),
            computed: AtomicEpoch::new(now),
            dirty: AtomicDirty::new(Dirty::default()),
            require: AtomicRequire::new(require),
            compute_lock: ExclusiveLock::default(),
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

    pub fn get_solutions(&self) -> Option<Arc<Solutions>> {
        self.steps.solutions.load_full()
    }

    pub fn line_count(&self) -> usize {
        self.steps.line_count()
    }

    // --- Compute Guards ---

    /// Try to start computing a step. Returns `None` if another thread is
    /// already computing the same step (or this step is being cleaned).
    pub fn try_start_compute(&self, step: Step) -> Option<ComputeGuard<'_>> {
        let exclusive = self.compute_lock.lock(step)?;
        Some(ComputeGuard {
            state: self,
            _exclusive: exclusive,
        })
    }

    /// Try to start clean. Uses `Step::first()` as the exclusive key,
    /// preventing concurrent computation from starting while clean is in progress.
    pub fn try_start_clean(&self) -> Option<CleanGuard<'_>> {
        let exclusive = self.compute_lock.lock(Step::first())?;
        Some(CleanGuard {
            state: self,
            _exclusive: exclusive,
        })
    }

    // --- Dirty Marking ---

    /// Try to mark this module's deps as dirty.
    /// Returns true if we were the one to set the flag (CAS succeeded),
    /// meaning the caller should add this module to the dirty set.
    pub fn try_mark_deps_dirty(&self, now: Epoch) -> bool {
        // If already computed in this epoch, skip.
        if self.computed.load() == now {
            return false;
        }
        if !self.dirty.try_set_deps() {
            return false;
        }
        // Re-check after setting. If the module was computed between our
        // epoch check and the CAS, the dirty flag is spurious — undo it.
        // Safe because try_set_deps uses CAS: no other thread can set DEPS
        // between our successful CAS and this check (they'd see it already set).
        if self.computed.load() == now {
            self.dirty.clear_deps();
            return false;
        }
        true
    }

    /// Increase the require level and set dirty.require if increased.
    /// Returns true if require was actually increased.
    pub fn increase_require(&self, require: Require) -> bool {
        let dirty_require = self.require.increase(require);
        if dirty_require {
            self.dirty.set_require();
        }
        dirty_require
    }

    /// Drain into a read-only snapshot for committed state.
    /// The `ModuleStateMut` should not be reused after this call.
    pub fn take_and_freeze(&self) -> ModuleState {
        ModuleState {
            require: self.require(),
            epochs: Epochs {
                checked: self.checked.load(),
                computed: self.computed.load(),
            },
            dirty: self.dirty.snapshot(),
            steps: self.steps.take_and_freeze(),
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeGuard — held while computing a step
// ---------------------------------------------------------------------------

/// Guard held while computing a step. The `ExclusiveLock` ensures only one
/// thread computes a given step at a time.
pub struct ComputeGuard<'a> {
    state: &'a ModuleStateMut,
    _exclusive: ExclusiveLockGuard<'a, Step>,
}

impl ComputeGuard<'_> {
    /// Re-check the next step under exclusive access. Another thread may have
    /// computed it between our initial check and acquiring the lock.
    pub fn next_step(&self) -> Option<Step> {
        self.state.next_step()
    }

    pub fn require(&self) -> Require {
        self.state.require()
    }

    /// Compute a step under exclusive access, delegating to `StepsMut::compute`.
    pub fn compute<Lookup: LookupExport + LookupAnswer>(
        &self,
        step: Step,
        old: &mut Steps,
        ctx: &Context<Lookup>,
    ) {
        self.state.steps.compute(step, old, ctx)
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

/// Guard held while cleaning a module. Uses `Step::first()` as the exclusive key,
/// preventing concurrent computation from starting while clean is in progress.
pub struct CleanGuard<'a> {
    state: &'a ModuleStateMut,
    _exclusive: ExclusiveLockGuard<'a, Step>,
}

impl CleanGuard<'_> {
    /// Atomically read and clear all dirty flags in a single `swap(0)`.
    /// Any flag set after this swap remains set for the next clean cycle.
    pub fn take_dirty(&self) -> Dirty {
        self.state.dirty.take_all()
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
        self.state.computed.store_relaxed(now);

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
