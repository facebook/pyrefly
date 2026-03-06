/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::Any;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use dupe::IterDupedExt;
use fxhash::FxHashMap;
use itertools::Itertools;
use pyrefly_graph::calculation::Calculation;
use pyrefly_graph::calculation::ProposalResult;
use pyrefly_graph::index::Idx;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_alias::TypeAlias;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_var::PreInferenceVariance;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Union;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::recurser::Guard;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::AnswerEntry;
use crate::alt::answers::AnswerTable;
use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::SolutionsEntry;
use crate::alt::answers::SolutionsTable;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::Binding;
use crate::binding::binding::Exported;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTypeAlias;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::config::base::RecursionLimitConfig;
use crate::config::base::RecursionOverflowHandler;
use crate::config::base::SccMode;
use crate::config::error_kind::ErrorKind;
use crate::dispatch_anyidx;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::style::ErrorStyle;
use crate::export::exports::LookupExport;
use crate::module::module_info::ModuleInfo;
use crate::solver::solver::VarRecurser;
use crate::solver::type_order::TypeOrder;
use crate::types::class::Class;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;
use crate::types::stdlib::Stdlib;
use crate::types::type_info::TypeInfo;
use crate::types::types::Type;
use crate::types::types::Var;

/// Compactly represents the identity of a binding, for the purposes of
/// understanding the calculation stack.
#[derive(Clone, Dupe)]
pub struct CalcId(pub Bindings, pub AnyIdx);

impl Debug for CalcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CalcId({}, {}, {:?})",
            self.0.module().name(),
            self.0.module().path(),
            self.1,
        )
    }
}

impl Display for CalcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CalcId({}, {}, {})",
            self.0.module().name(),
            self.0.module().path(),
            self.1.display_with(&self.0),
        )
    }
}

impl PartialEq for CalcId {
    fn eq(&self, other: &Self) -> bool {
        (self.0.module().name(), self.0.module().path(), &self.1)
            == (other.0.module().name(), other.0.module().path(), &other.1)
    }
}

impl Eq for CalcId {}

impl Ord for CalcId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.1.cmp(&other.1) {
            Ordering::Equal => match self.0.module().name().cmp(&other.0.module().name()) {
                Ordering::Equal => self.0.module().path().cmp(other.0.module().path()),
                not_equal => not_equal,
            },
            not_equal => not_equal,
        }
    }
}

impl PartialOrd for CalcId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for CalcId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.module().name().hash(state);
        self.0.module().path().hash(state);
        self.1.hash(state);
    }
}

impl CalcId {
    /// Create a CalcId for testing purposes.
    ///
    /// The `module_name` creates a distinguishable module, and `idx` creates
    /// a distinguishable index within that module. CalcIds with different
    /// (module_name, idx) pairs will compare as not equal.
    #[cfg(test)]
    pub fn for_test(module_name: &str, idx: usize) -> Self {
        use pyrefly_graph::index::Idx;

        let bindings = Bindings::for_test(module_name);
        // Create a fake Key index - the actual key doesn't matter for test purposes,
        // only that different idx values produce different CalcIds
        let key_idx: Idx<Key> = Idx::new(idx);
        CalcId(bindings, AnyIdx::Key(key_idx))
    }
}

/// Represent a stack of in-progress calculations in an `AnswersSolver`.
///
/// This is useful for debugging, particularly for debugging scc handling.
///
/// The stack is per-thread; we create a new `AnswersSolver` every time
/// we change modules when resolving exports, but the stack is passed
/// down because sccs can cross module boundaries.
pub struct CalcStack {
    stack: RefCell<Vec<CalcId>>,
    scc_stack: RefCell<Vec<Scc>>,
    /// Reverse lookup of `stack`, to enable O(1) access for a given CalcId.
    position_of: RefCell<FxHashMap<CalcId, Vec1<usize>>>,
    /// SCCs that completed during `on_calculation_finished` but haven't been
    /// batch-committed yet. Drained by `get_idx` after each frame completes.
    pending_completed_sccs: RefCell<Vec<Scc>>,
    /// The SCC solving mode, propagated from `ThreadState` at construction time.
    /// This makes the mode accessible during SCC operations without needing a
    /// back-reference to `ThreadState`.
    scc_solving_mode: SccSolvingMode,
}

impl CalcStack {
    fn new(scc_solving_mode: SccSolvingMode) -> Self {
        Self {
            stack: RefCell::new(Vec::new()),
            scc_stack: RefCell::new(Vec::new()),
            position_of: RefCell::new(FxHashMap::default()),
            pending_completed_sccs: RefCell::new(Vec::new()),
            scc_solving_mode,
        }
    }

    /// Returns true when the SCC solving mode is `Iterative`.
    fn is_iterative(&self) -> bool {
        self.scc_solving_mode == SccSolvingMode::Iterative
    }

    /// Pop the current frame and drain any SCCs that completed during it.
    ///
    /// These two operations are always paired: every `pop` must be followed by
    /// draining and committing completed SCCs.
    ///
    /// We pop before draining (not after) for two reasons:
    /// - Lifecycle correctness: committed answers should correspond to fully
    ///   unwound computations. Popping first ensures the stack no longer
    ///   contains the completing frame when results are written to Calculation.
    /// - `pop()` decrements `segment_size` on the top SCC. If we drained first,
    ///   the completed SCC would already be gone from `scc_stack`, and `pop()`
    ///   could incorrectly decrement a parent SCC's segment_size instead.
    ///
    /// Note that the `+ 1` in `on_calculation_finished`'s completion check
    /// (`stack_len <= anchor_pos + 1`) is unrelated to this ordering — it
    /// exists because completion is detected during calculation, while the
    /// frame is still on the stack, well before we reach this method.
    ///
    /// Safety: `drain_completed_sccs` uses `std::mem::take` which drops the
    /// `RefCell` borrow before returning the owned `Vec<Scc>`. By the time the
    /// caller iterates the returned SCCs, no borrow on `self` is live.
    fn pop_and_drain_completed_sccs(&self) -> Vec<Scc> {
        self.pop();
        self.drain_completed_sccs()
    }

    /// Drain and return all completed SCCs that were collected during
    /// `on_calculation_finished`. Used by `pop_and_drain_completed_sccs`
    /// to batch-commit answers after a frame completes.
    fn drain_completed_sccs(&self) -> Vec<Scc> {
        std::mem::take(&mut *self.pending_completed_sccs.borrow_mut())
    }

    /// Push a CalcId onto the stack and compute the binding action.
    ///
    /// This combines the push operation with computing what action to take,
    /// performing all SCC state checks and mutations (like `on_scc_detected`,
    /// `on_calculation_finished`). SCC merging (`merge_sccs`) is handled
    /// inside `pre_calculate_state` when a node is found in a previous SCC.
    fn push<T: Dupe>(&self, current: CalcId, calculation: &Calculation<T>) -> BindingAction<T> {
        let position = {
            let mut stack = self.stack.borrow_mut();
            let pos = stack.len();
            stack.push(current.dupe());
            pos
        };
        self.position_of
            .borrow_mut()
            .entry(current.dupe())
            .and_modify(|positions| positions.push(position))
            .or_insert_with(|| Vec1::new(position));

        // Membership back-edge detection: when iterative mode is active, check
        // if the target is a member of a *non-top* iterating SCC. If so, this
        // is a cross-SCC back-edge that must merge all SCCs from that index to
        // the top and demote (restart at iteration 1).
        //
        // This check runs BEFORE the top-SCC iterative bypass because cross-SCC
        // back-edges must be caught first. SCCs still in Phase 0 discovery
        // (iterative: None) are handled by the existing `on_scc_detected` path.
        //
        // Borrow safety: `find_iterating_scc_containing` returns an owned
        // `Option<usize>`, so the shared borrow on `scc_stack` is released
        // before the exclusive borrow needed for merging.
        if self.is_iterative()
            && let Some(scc_idx) = self.find_iterating_scc_containing(&current)
        {
            let is_non_top = {
                let scc_stack = self.scc_stack.borrow();
                scc_idx < scc_stack.len() - 1
            };
            if is_non_top {
                // Merge all SCCs from scc_idx to the top of the stack.
                // This produces a single merged SCC with demoted = true
                // (via Scc::merge's iteration state merge logic).
                {
                    let stack_len = self.stack.borrow().len();
                    let mut scc_stack = self.scc_stack.borrow_mut();
                    let sccs_to_merge: Vec<Scc> = scc_stack.drain(scc_idx..).collect();
                    let sccs_to_merge = Vec1::try_from_vec(sccs_to_merge)
                        .expect("membership back-edge: at least the found SCC must be present");
                    let detected_at = sccs_to_merge.first().detected_at.dupe();
                    let mut merged = Scc::merge_many(sccs_to_merge, detected_at);
                    // Recompute segment_size after merge.
                    merged.segment_size = stack_len - merged.anchor_pos;
                    scc_stack.push(merged);
                }
                // The target is now in the top SCC's iteration state.
                // Determine the appropriate action based on iteration state.
                // After merge with demotion, the iteration state is fresh
                // (iteration 1, all nodes Fresh, no previous answers), so the
                // target will typically be Fresh. But we handle all cases for
                // robustness.
                if let Some(kind) = self.get_iteration_node_state(&current) {
                    return match kind {
                        IterationNodeStateKind::Fresh => {
                            self.set_iteration_node_in_progress(&current);
                            BindingAction::Calculate
                        }
                        IterationNodeStateKind::InProgressWithPreviousAnswer => {
                            let answer = self.get_previous_answer(&current).expect(
                                "InProgressWithPreviousAnswer but no previous answer found",
                            );
                            BindingAction::SccLocalAnswer(answer)
                        }
                        IterationNodeStateKind::InProgressWithPlaceholder => {
                            let var = self
                                .get_iteration_placeholder(&current)
                                .expect("InProgressWithPlaceholder but no placeholder found");
                            BindingAction::CycleBroken(var)
                        }
                        IterationNodeStateKind::InProgressCold => {
                            BindingAction::NeedsColdPlaceholder
                        }
                        IterationNodeStateKind::Done => {
                            let answer = self
                                .get_iteration_done_answer(&current)
                                .expect("Done iteration node state but no answer found");
                            BindingAction::SccLocalAnswer(answer)
                        }
                    };
                }
                // If we merged but the target is somehow not in iteration state,
                // this is a bug: the merge should have included it.
                unreachable!(
                    "membership back-edge: target {} was in iterating SCC but \
                     not found in merged SCC's iteration state",
                    current,
                );
            }
            // The target is now in the top SCC's iteration state.
            // Determine the appropriate action based on iteration state.
            // After merge with demotion, the iteration state is fresh
            // (iteration 1, all nodes Fresh, no previous answers), so the
            // target will typically be Fresh. But we handle all cases for
            // robustness.
            if let Some(kind) = self.get_iteration_node_state(&current) {
                return match kind {
                    IterationNodeStateKind::Fresh => {
                        self.set_iteration_node_in_progress(&current);
                        BindingAction::Calculate
                    }
                    IterationNodeStateKind::InProgressWithPreviousAnswer => {
                        let answer = self
                            .get_previous_answer(&current)
                            .expect("InProgressWithPreviousAnswer but no previous answer found");
                        BindingAction::SccLocalAnswer(answer)
                    }
                    IterationNodeStateKind::InProgressWithPlaceholder => {
                        let var = self
                            .get_iteration_placeholder(&current)
                            .expect("InProgressWithPlaceholder but no placeholder found");
                        BindingAction::CycleBroken(var)
                    }
                    IterationNodeStateKind::InProgressCold => BindingAction::NeedsColdPlaceholder,
                    IterationNodeStateKind::Done => {
                        let answer = self
                            .get_iteration_done_answer(&current)
                            .expect("Done iteration node state but no answer found");
                        BindingAction::SccLocalAnswer(answer)
                    }
                };
            }
            // If we merged but the target is somehow not in iteration state,
            // this is a bug: the merge should have included it.
            unreachable!(
                "membership back-edge: target {} was in iterating SCC but \
                     not found in merged SCC's iteration state",
                current,
            );
        }

        // Iterative bypass: when iterative mode is active and the top SCC is
        // iterating, check if the target is a member of the top SCC's iteration
        // state. If so, use SCC-scoped iteration state to determine the action
        // instead of falling through to the legacy SCC logic.
        //
        // Borrow safety: `get_iteration_node_state` returns an owned
        // `IterationNodeStateKind`, so the shared borrow on `scc_stack` is
        // released before any exclusive borrow for mutation.
        if self.is_iterative()
            && let Some(kind) = self.get_iteration_node_state(&current)
        {
            // The node was unconditionally pushed onto the raw CalcStack
            // above, and pop() will decrement segment_size for any node
            // in the top SCC's node_state. We must increment here to
            // keep segment_size symmetric, since the early return below
            // bypasses the segment_size += 1 in the SccState::Participant
            // arm.
            if let Some(top_scc) = self.scc_stack.borrow_mut().last_mut() {
                top_scc.segment_size += 1;
            }
            return match kind {
                IterationNodeStateKind::Fresh => {
                    // First encounter in this iteration: mark InProgress
                    // and proceed to calculate.
                    self.set_iteration_node_in_progress(&current);
                    BindingAction::Calculate
                }
                IterationNodeStateKind::InProgressWithPreviousAnswer => {
                    // Back-edge with a warm-start answer from prior iteration.
                    let answer = self
                        .get_previous_answer(&current)
                        .expect("InProgressWithPreviousAnswer but no previous answer found");
                    BindingAction::SccLocalAnswer(answer)
                }
                IterationNodeStateKind::InProgressWithPlaceholder => {
                    // Back-edge with a placeholder already allocated.
                    let var = self
                        .get_iteration_placeholder(&current)
                        .expect("InProgressWithPlaceholder but no placeholder found");
                    BindingAction::CycleBroken(var)
                }
                IterationNodeStateKind::InProgressCold => {
                    // Cold-start back-edge: no placeholder, no previous answer.
                    // Return NeedsColdPlaceholder so the caller (get_idx) can
                    // allocate via K::create_recursive.
                    BindingAction::NeedsColdPlaceholder
                }
                IterationNodeStateKind::Done => {
                    // Already solved in this iteration; return the answer.
                    let answer = self
                        .get_iteration_done_answer(&current)
                        .expect("Done iteration node state but no answer found");
                    BindingAction::SccLocalAnswer(answer)
                }
            };
        }

        match self.pre_calculate_state(&current) {
            SccState::NotInScc | SccState::RevisitingInProgress => {
                match calculation.propose_calculation() {
                    ProposalResult::Calculated(v) => BindingAction::Calculated(v),
                    // Use the thread-local stack as the source of truth for
                    // cycle detection: `position_of` tells us definitively
                    // whether this CalcId has a live frame on the current stack.
                    ProposalResult::Calculatable => {
                        if let Some(current_cycle) = self.current_cycle() {
                            match self.on_scc_detected(current_cycle) {
                                SccDetectedResult::BreakHere => BindingAction::Unwind,
                                SccDetectedResult::Continue => BindingAction::Calculate,
                            }
                        } else {
                            // No cycle on the stack: this node simply needs
                            // computing. This is the normal case for both
                            // iterative and legacy modes.
                            BindingAction::Calculate
                        }
                    }
                    ProposalResult::CycleDetected => {
                        if let Some(current_cycle) = self.current_cycle() {
                            match self.on_scc_detected(current_cycle) {
                                SccDetectedResult::BreakHere => BindingAction::Unwind,
                                SccDetectedResult::Continue => BindingAction::Calculate,
                            }
                        } else if self.is_iterative()
                            && self.get_iteration_node_state(&current).is_some()
                        {
                            // In iterative mode, CycleDetected without an active
                            // cycle means Phase 0 left stale Calculating state
                            // (the thread started this node in a previous
                            // iteration but the stack was unwound). Treat as a
                            // cold-start back-edge: the caller (get_idx) will
                            // allocate a placeholder via K::create_recursive.
                            BindingAction::NeedsColdPlaceholder
                        } else {
                            // Legacy mode: CycleDetected without a stack cycle is
                            // surprising but has been observed in LSP. The thread
                            // started a calculation but never saved an answer, and
                            // the stack frame is gone. Proceed with Calculate as a
                            // best-effort fallback.
                            //
                            // TODO: This may indicate a bug in SCC merging, state
                            // transition tracking, or batch commit.
                            BindingAction::Calculate
                        }
                    }
                }
            }
            SccState::RevisitingDone => {
                // Try to read from the SCC-local NodeState::Done first.
                // If the answer is available, return it without touching Calculation.
                if let Some(answer) = self.get_scc_done_answer(&current) {
                    BindingAction::SccLocalAnswer(answer)
                } else {
                    // Fallback: answer is None (another path computed it).
                    // Check if another thread already committed a final answer.
                    match calculation.get() {
                        Some(v) => BindingAction::Calculated(v),
                        None => unreachable!(
                            "RevisitingDone node has no SCC-local answer and no global answer"
                        ),
                    }
                }
            }
            SccState::BreakAt => BindingAction::Unwind,
            SccState::HasPlaceholder => {
                // Read placeholder from SCC-local NodeState::HasPlaceholder.
                // No need to touch the Calculation cell — placeholders are never
                // stored there.
                if let Some(v) = calculation.get() {
                    // Another thread already committed a final answer.
                    BindingAction::Calculated(v)
                } else {
                    let var = self
                        .get_scc_placeholder_var(&current)
                        .expect("HasPlaceholder state but no placeholder in NodeState");
                    BindingAction::CycleBroken(var)
                }
            }
            SccState::Participant => {
                if let Some(top_scc) = self.scc_stack.borrow_mut().last_mut() {
                    top_scc.segment_size += 1;
                }
                match calculation.propose_calculation() {
                    ProposalResult::Calculatable => {
                        unreachable!(
                            "Participant nodes must have Calculating state, not NotCalculated"
                        )
                    }
                    ProposalResult::CycleDetected => BindingAction::Calculate,
                    ProposalResult::Calculated(v) => {
                        // Participant already computed: no data to store.
                        self.on_calculation_finished(&current, None, None);
                        BindingAction::Calculated(v)
                    }
                }
            }
        }
    }

    /// Pop a binding frame from the raw binding-level CalcId stack.
    /// - Update both the direct stack and the `position_of` reverse index.
    /// - Also check whether the popped frame was part of the top Scc in the
    ///   Scc stack; if so, decrement the segment_size to account for the fact
    ///   that this frame has completed.
    fn pop(&self) -> Option<CalcId> {
        let popped = self.stack.borrow_mut().pop();
        if let Some(ref calc_id) = popped {
            let mut position_of = self.position_of.borrow_mut();
            if let Some(positions) = position_of.get_mut(calc_id) {
                // Try to pop from Vec1 - if it fails (Size0Error), this was the last position
                if positions.pop().is_err() {
                    // Vec1 only has one element, so remove the entire entry
                    position_of.remove(calc_id);
                }
            }
            let mut scc_stack = self.scc_stack.borrow_mut();
            if let Some(top_scc) = scc_stack.last_mut()
                && top_scc.node_state.contains_key(calc_id)
            {
                top_scc.segment_size = top_scc.segment_size.saturating_sub(1);
            }
        }
        popped
    }

    /// Check if a CalcId is on the stack and return its first (earliest) position if so.
    #[allow(dead_code)]
    fn find_on_stack(&self, calc_id: &CalcId) -> Option<usize> {
        self.position_of
            .borrow()
            .get(calc_id)
            .map(|positions| *positions.first())
    }

    /// Check if a CalcId is an SCC participant (exists in the top SCC's node_state).
    fn is_scc_participant(&self, current: &CalcId) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .is_some_and(|top_scc| top_scc.node_state.contains_key(current))
    }

    /// Retrieve the placeholder Var from NodeState::HasPlaceholder in the top SCC.
    /// Returns `Some(var)` if the node is a break_at node with a placeholder,
    /// `None` otherwise. Used during calculate_and_record_answer to determine
    /// whether finalize_recursive_answer needs to be called.
    fn get_scc_placeholder_var(&self, current: &CalcId) -> Option<Var> {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .and_then(|top_scc| match top_scc.node_state.get(current)? {
                NodeState::HasPlaceholder(var) => Some(*var),
                _ => None,
            })
    }

    /// Retrieve the type-erased answer from NodeState::Done in the top SCC.
    /// Returns `Some(answer)` if the node is Done with data, `None` otherwise
    /// (node not in SCC, not Done, or Done with answer: None).
    fn get_scc_done_answer(&self, current: &CalcId) -> Option<Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .and_then(|top_scc| match top_scc.node_state.get(current)? {
                NodeState::Done { answer, .. } => answer.clone(),
                _ => None,
            })
    }

    /// Push a CalcId onto the stack without computing the binding action, for tests
    #[cfg(test)]
    fn push_for_test(&self, current: CalcId) {
        let position = {
            let mut stack = self.stack.borrow_mut();
            let pos = stack.len();
            stack.push(current.dupe());
            pos
        };
        self.position_of
            .borrow_mut()
            .entry(current)
            .and_modify(|positions| positions.push(position))
            .or_insert_with(|| Vec1::new(position));
    }

    pub fn peek(&self) -> Option<CalcId> {
        self.stack.borrow().last().cloned()
    }

    pub fn into_vec(&self) -> Vec<CalcId> {
        self.stack.borrow().clone()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.borrow().is_empty()
    }

    /// Return the current stack depth (number of entries on the stack).
    pub fn len(&self) -> usize {
        self.stack.borrow().len()
    }

    /// Return the current cycle, if we are at a (module, idx) that we've already seen in this thread.
    ///
    /// The answer will have the form
    /// - if there is no cycle, `None`
    /// - if there is a cycle, `Some(vec![(m0, i0), (m2, i2)...])`
    ///   where the order of (module, idx) pairs is recency (so starting with current
    ///   module and idx, and ending with the oldest).
    pub fn current_cycle(&self) -> Option<Vec1<CalcId>> {
        let stack = self.stack.borrow();
        let current = stack.last()?;
        let positions = self.position_of.borrow();
        let target_positions = positions.get(current)?;
        // If there are is now more than one position,we have encountered a cycle.
        if target_positions.len() == 1 {
            None
        } else {
            // The actual cycle is the set of nodes between the occurrence we just pushed
            // and the most recent *previous* occurrence of this CaclId (i.e. the second-to-last)
            let cycle_start = target_positions[target_positions.len() - 2];
            let cycle_entries: Vec<CalcId> =
                stack[cycle_start + 1..].iter().rev().duped().collect();
            Vec1::try_from_vec(cycle_entries).ok()
        }
    }

    // SCC methods - these manage the scc_stack

    fn sccs_is_empty(&self) -> bool {
        self.scc_stack.borrow().is_empty()
    }

    /// Borrow the SCC stack for iteration (used in debug output).
    fn borrow_scc_stack(&self) -> std::cell::Ref<'_, Vec<Scc>> {
        self.scc_stack.borrow()
    }

    /// Check if an existing SCC overlaps with a newly detected cycle.
    ///
    /// Uses O(1) position arithmetic: if the existing SCC's segment upper bound
    /// (anchor_pos + segment_size) is greater than the cycle start position,
    /// the segments overlap and must be merged.
    ///
    /// This works because segments are contiguous - all frames between anchor_pos
    /// and anchor_pos + segment_size belong to this SCC.
    #[cfg_attr(test, allow(dead_code))]
    fn check_overlap(existing: &Scc, cycle_start_pos: usize) -> bool {
        // O(1) overlap check using segment bounds.
        // If the existing SCC's upper bound <= cycle start, there's no overlap.
        // Upper bound = anchor_pos + segment_size (exact count of live frames in segment)
        existing.anchor_pos + existing.segment_size > cycle_start_pos
    }

    /// Handle an SCC we just detected.
    ///
    /// Return whether to break immediately (which is relatively common, since
    /// we break on the minimal idx which is often where we detect the problem)
    /// or continue recursing.
    ///
    /// When a new SCC overlaps with existing SCCs (shares participants),
    /// we merge them to form a larger SCC. This preserves behavioral equivalence
    /// because all break points are retained in the merged break_at set.
    ///
    /// Optimization: We use stack depth to efficiently find overlapping SCCs.
    /// The cycle spans CalcStack positions [N, M] where M = stack_depth - 1 and
    /// N = M - cycle_length + 1. Any SCC with max_stack_depth < N cannot overlap.
    /// Once we find the first overlapping SCC, all subsequent SCCs must also
    /// overlap (due to LIFO ordering of the SCC stack).
    #[allow(clippy::mutable_key_type)] // CalcId's Hash impl doesn't depend on mutable parts
    fn on_scc_detected(&self, raw: Vec1<CalcId>) -> SccDetectedResult {
        let calc_stack_vec = self.into_vec();

        // Create the new SCC
        let new_scc = Scc::new(raw, &calc_stack_vec);
        let detected_at = new_scc.detected_at.dupe();
        let cycle_start_pos = new_scc.anchor_pos;

        // Check for overlapping SCCs and merge if needed
        let mut scc_stack = self.scc_stack.borrow_mut();

        // Find the first (oldest) SCC that overlaps with the new cycle.
        // Overlap is determined by O(1) segment arithmetic: if the existing SCC's
        // upper bound (anchor_pos + segment_size) exceeds cycle_start_pos, they overlap.
        // Due to LIFO ordering, once we find one overlapping SCC, all subsequent ones
        // on the stack must also overlap.
        let mut first_merge_idx: Option<usize> = None;

        for (i, existing) in scc_stack.iter().enumerate() {
            if Self::check_overlap(existing, cycle_start_pos) {
                first_merge_idx = Some(i);
                break; // All subsequent SCCs will also overlap
            }
        }

        let result = if let Some(first_idx) = first_merge_idx {
            // Merge all SCCs from first_idx to end, plus the new SCC
            let sccs_from_stack: Vec<Scc> = scc_stack.drain(first_idx..).collect();
            let sccs_to_merge = Vec1::from_vec_push(sccs_from_stack, new_scc);

            // Use the helper method to merge SCCs
            let mut merged_scc = Scc::merge_many(sccs_to_merge, detected_at.dupe());

            // After a merge, everything from the merged anchor to the current stack top
            // is part of this single SCC. Recompute segment_size from scratch.
            merged_scc.segment_size = calc_stack_vec.len() - merged_scc.anchor_pos;

            let result = if merged_scc.break_at.contains(&detected_at) {
                SccDetectedResult::BreakHere
            } else {
                SccDetectedResult::Continue
            };
            scc_stack.push(merged_scc);
            result
        } else {
            // No overlap - just push the new SCC
            let result = if new_scc.break_at.contains(&detected_at) {
                SccDetectedResult::BreakHere
            } else {
                SccDetectedResult::Continue
            };
            scc_stack.push(new_scc);
            result
        };

        // Iterative mode never uses min-idx breaking: every back-edge breaks
        // immediately. This ensures Phase 0 is purely membership discovery and
        // that no frame continues past its own cycle detection point.
        if self.is_iterative() {
            return SccDetectedResult::BreakHere;
        }

        result
    }

    /// Check the SCC state for a node before calculating it.
    ///
    /// We check ALL SCCs on the stack, not just the top one, because a node
    /// might be a participant in an SCC that's not at the top of the stack.
    /// This is especially important after merging, where nodes from previously
    /// separate SCCs are now in the same merged SCC.
    ///
    /// Invariant: After merging, each node appears in at most one SCC on the
    /// stack. We return the first non-NotInScc result when scanning
    /// top-to-bottom, which will be the unique SCC containing this node (if any).
    ///
    /// Special case: If we find a node in the top SCC but we've pushed frames
    /// above the SCC's segment (i.e., we exited and are now re-entering), or
    /// in a non-top SCC, we call `merge_sccs` immediately to merge all
    /// intervening SCCs and return the underlying state. `Participant` is
    /// converted to `RevisitingInProgress` after merge since segment_size
    /// is already correct (merge recalculates it).
    fn pre_calculate_state(&self, current: &CalcId) -> SccState {
        let stack_len = self.stack.borrow().len();

        // Scan SCCs top-to-bottom to find one containing this node.
        // If found in the top SCC within its segment, return directly.
        // Otherwise, save the info needed for merge and break out.
        let merge_info: Option<(CalcId, SccState)> = {
            let mut scc_stack = self.scc_stack.borrow_mut();
            let mut result = None;
            for (rev_idx, scc) in scc_stack.iter_mut().rev().enumerate() {
                let is_top_scc = rev_idx == 0;
                let state = scc.pre_calculate_state(current);

                match state {
                    SccState::NotInScc => continue,
                    // For the top SCC, check if we're still within its segment.
                    _ if is_top_scc && is_within_scc_segment(stack_len, scc) => {
                        // Normal case: still within the top SCC's segment
                        return state;
                    }
                    _ => {}
                }
                // Node is in a non-top SCC, or in the top SCC but outside its
                // segment. Save the detected_at and state, then break so we can
                // drop the borrow and call merge_sccs.
                result = Some((scc.detected_at(), state));
                break;
            }
            result
        };
        // scc_stack borrow is now dropped

        if let Some((detected_at, state)) = merge_info {
            self.merge_sccs(&detected_at);
            // After merge, segment_size is recalculated. Participant would
            // increment segment_size again in push(), so convert it to
            // RevisitingInProgress to avoid double-counting.
            match state {
                SccState::Participant | SccState::RevisitingInProgress => {
                    SccState::RevisitingInProgress
                }
                other => other,
            }
        } else {
            SccState::NotInScc
        }
    }

    /// Handle the completion of a calculation. Mark the node as Done in the
    /// top SCC (if it's a participant), then push any completed SCCs to the
    /// `pending_completed_sccs` buffer for later batch-commit by `get_idx`.
    ///
    /// Only the top SCC is checked because each node appears in at most one
    /// SCC, and active calculations are always in the top SCC.
    fn on_calculation_finished(
        &self,
        current: &CalcId,
        answer: Option<Arc<dyn Any + Send + Sync>>,
        errors: Option<Arc<ErrorCollector>>,
    ) -> Option<Arc<dyn Any + Send + Sync>> {
        let stack_len = self.stack.borrow().len();
        let mut scc_stack = self.scc_stack.borrow_mut();
        let canonical = if let Some(top_scc) = scc_stack.last_mut() {
            let canonical = top_scc.on_calculation_finished(current, answer, errors);
            // Debug-only check: verify the node isn't in any other SCC.
            debug_assert!(
                scc_stack
                    .iter()
                    .rev()
                    .skip(1)
                    .all(|scc| !scc.node_state.contains_key(current)),
                "on_calculation_finished: CalcId {} found in multiple SCCs",
                current,
            );
            canonical
        } else {
            // No active SCC; return the provided answer unchanged.
            answer
        };
        // Pop all SCCs whose anchor position indicates completion.
        // An SCC is complete when the stack has unwound to (or past) its
        // anchor: at that point all participants' frames have been popped
        // and their answers recorded. Push them to the pending buffer
        // so that `get_idx` can batch-commit them after the frame completes.
        while let Some(scc) = scc_stack.last() {
            if stack_len <= scc.anchor_pos + 1 {
                self.pending_completed_sccs
                    .borrow_mut()
                    .push(scc_stack.pop().unwrap());
            } else {
                break;
            }
        }
        canonical
    }

    /// Track that a placeholder has been recorded for a break_at node.
    ///
    /// Only the top SCC is checked because each node appears in at most one
    /// SCC, and placeholder recording happens during active cycle breaking
    /// in the top SCC.
    fn on_placeholder_recorded(&self, current: &CalcId, var: Var) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        if let Some(top_scc) = scc_stack.last_mut() {
            top_scc.on_placeholder_recorded(current, var);
            // Debug-only check: verify the node isn't in any other SCC.
            debug_assert!(
                scc_stack
                    .iter()
                    .rev()
                    .skip(1)
                    .all(|scc| !scc.node_state.contains_key(current)),
                "on_placeholder_recorded: CalcId {} found in multiple SCCs",
                current,
            );
        }
    }

    /// Merge all SCCs from the target SCC to the top of the stack, and add
    /// any free-floating CalcStack nodes between the target SCC's min_stack_depth
    /// and the current stack position.
    ///
    /// This is called from `pre_calculate_state` when a node is found in a
    /// non-top SCC, or in the top SCC but outside its segment. After this call,
    /// the SCC stack will have one merged SCC at the top containing all
    /// participants from the merged SCCs plus any free-floating nodes from the
    /// CalcStack.
    ///
    /// The oldest previously-known Scc we should merge is identified based on its
    /// `detected_at`; this has the potentially-useful property of being a valid
    /// identifier of the merged Scc *after* the merge, since we always use the
    /// very first cycle detected for `detected_at`.
    #[allow(clippy::mutable_key_type)]
    fn merge_sccs(&self, detected_at_of_scc: &CalcId) {
        let calc_stack_vec = self.into_vec();
        let mut scc_stack = self.scc_stack.borrow_mut();

        // Pop SCCs until we find the target component (identified by detected_at).
        //
        // Push them to a vec we will merge; in addition, when we reach the last component
        // use it to determine how much of the CalcStack needs to be merged in order
        // to ensure bindings that weren't yet part of a known SCC are included.
        let mut sccs_to_merge: Vec<Scc> = Vec::new();
        let mut target_anchor_pos: Option<usize> = None;
        while let Some(scc) = scc_stack.pop() {
            let is_target = scc.detected_at() == *detected_at_of_scc;
            if is_target {
                target_anchor_pos = Some(scc.anchor_pos);
            }
            sccs_to_merge.push(scc);
            if is_target {
                break;
            }
        }
        let min_depth = target_anchor_pos
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");
        let sccs_to_merge = Vec1::try_from_vec(sccs_to_merge)
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");

        // Perform the merge, then add any free-floating bindings that weren't previously part
        // of a known SCC. These nodes are already on the call stack (they have active frames),
        // so they are InProgress, not Fresh.
        let mut merged = Scc::merge_many(sccs_to_merge, detected_at_of_scc.dupe());
        for calc_id in calc_stack_vec.iter().skip(min_depth) {
            merged
                .node_state
                .entry(calc_id.dupe())
                .or_insert(NodeState::InProgress);
        }

        // After a merge, everything from the merged anchor to the current stack top
        // is part of this single SCC. Recompute segment_size from scratch.
        merged.segment_size = calc_stack_vec.len() - merged.anchor_pos;

        scc_stack.push(merged);
    }

    /// Find the index in `scc_stack` of an iterating SCC that contains `target`.
    ///
    /// Scans the SCC stack for an SCC with `iterative: Some(...)` whose
    /// `node_state` (legacy membership map) contains the target. Returns the index in the stack
    /// (not the SCC's `anchor_pos`). Used for membership-based back-edge
    /// detection: a request for a CalcId in a non-top iterating SCC is a
    /// back-edge that must trigger merge + demotion.
    fn find_iterating_scc_containing(&self, target: &CalcId) -> Option<usize> {
        let scc_stack = self.scc_stack.borrow();
        for (i, scc) in scc_stack.iter().enumerate() {
            if scc.iterative.is_some() && scc.node_state.contains_key(target) {
                return Some(i);
            }
        }
        None
    }

    /// Returns true if `target` is a member of any iterating SCC on the stack.
    #[allow(dead_code)]
    fn is_iterating_member(&self, target: &CalcId) -> bool {
        self.find_iterating_scc_containing(target).is_some()
    }

    /// Returns true if the top SCC is iterating at iteration 1 (cold start).
    ///
    /// During cold-start iteration, back-edges allocate placeholders rather
    /// than reusing previous answers.
    fn is_cold_iteration(&self) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .and_then(|scc| scc.iterative.as_ref())
            .is_some_and(|iter_state| iter_state.iteration == 1)
    }

    /// Returns true if the top SCC is iterating at iteration >= 2.
    ///
    /// This is used to decide whether to suppress errors: errors are
    /// swallowed during iteration 1 and collected from iteration 2 onward.
    /// The name "final" is a misnomer since more iterations may follow;
    /// it means "past cold start."
    #[allow(dead_code)]
    fn is_final_iteration(&self) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .and_then(|scc| scc.iterative.as_ref())
            .is_some_and(|iter_state| iter_state.iteration >= 2)
    }

    /// Get the lightweight summary of a target's iteration node state in
    /// the top SCC.
    ///
    /// Returns `None` if the top SCC is not iterating or the target is not
    /// found in the iteration node states. The summary is safe to use for
    /// read-then-act patterns because it does not borrow the SCC.
    fn get_iteration_node_state(&self, target: &CalcId) -> Option<IterationNodeStateKind> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let iter_state = top_scc.iterative.as_ref()?;
        let node_state = iter_state.node_states.get(target)?;
        let has_previous_answer = iter_state.previous_answers.contains_key(target);
        Some(node_state.kind(has_previous_answer))
    }

    /// Mark a target node as `InProgress` in the top SCC's iteration state.
    ///
    /// Panics if the top SCC is not iterating, the target is not a member,
    /// or the target is not `Fresh`.
    fn set_iteration_node_in_progress(&self, target: &CalcId) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        let iter_state = top_scc
            .iterative
            .as_mut()
            .expect("top SCC is not iterating");
        let node_state = iter_state
            .node_states
            .get_mut(target)
            .expect("target is not a member of the iterating SCC");
        assert!(
            matches!(node_state, IterationNodeState::Fresh),
            "set_iteration_node_in_progress called on non-Fresh node: {target:?}"
        );
        *node_state = IterationNodeState::InProgress { placeholder: None };
    }

    /// Set the placeholder variable on an existing `InProgress` iteration
    /// node state for the target.
    ///
    /// Panics if the target is not found or is not `InProgress`.
    fn set_iteration_placeholder(&self, target: &CalcId, var: Var) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        let iter_state = top_scc
            .iterative
            .as_mut()
            .expect("top SCC is not iterating");
        let node_state = iter_state
            .node_states
            .get_mut(target)
            .expect("target is not a member of the iterating SCC");
        match node_state {
            IterationNodeState::InProgress { placeholder } => {
                *placeholder = Some(var);
            }
            _ => panic!(
                "set_iteration_placeholder called on a node that is not InProgress: {:?}",
                node_state
            ),
        }
    }

    /// Get the placeholder variable from the target's `InProgress` state,
    /// if one exists.
    ///
    /// Returns `None` if the top SCC is not iterating, the target is not
    /// found, the target is not `InProgress`, or no placeholder has been set.
    fn get_iteration_placeholder(&self, target: &CalcId) -> Option<Var> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let iter_state = top_scc.iterative.as_ref()?;
        match iter_state.node_states.get(target)? {
            IterationNodeState::InProgress {
                placeholder: Some(var),
            } => Some(*var),
            _ => None,
        }
    }

    /// Mark a target node as `Done` in the top SCC's iteration state.
    ///
    /// Panics if the top SCC is not iterating.
    fn set_iteration_node_done(
        &self,
        target: &CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
    ) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        let iter_state = top_scc
            .iterative
            .as_mut()
            .expect("top SCC is not iterating");
        iter_state
            .node_states
            .insert(target.dupe(), IterationNodeState::Done { answer, errors });
    }

    /// Set `has_changed = true` on the top SCC's iteration state.
    ///
    /// Called when a node's answer differs from its previous-iteration answer,
    /// indicating the fixpoint has not yet converged.
    ///
    /// Panics if the top SCC is not iterating.
    fn mark_iteration_changed(&self) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        let iter_state = top_scc
            .iterative
            .as_mut()
            .expect("top SCC is not iterating");
        iter_state.has_changed = true;
    }

    /// Look up the previous-iteration answer for a target in the top SCC.
    ///
    /// Returns `None` if the top SCC is not iterating or there is no
    /// previous answer for the target (e.g., during cold-start iteration 1).
    fn get_previous_answer(&self, target: &CalcId) -> Option<Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let iter_state = top_scc.iterative.as_ref()?;
        iter_state.previous_answers.get(target).cloned()
    }

    /// Retrieve the type-erased answer from `IterationNodeState::Done` in the
    /// top SCC's iteration state.
    ///
    /// Returns `None` if the top SCC is not iterating, the target is not
    /// found, or the target is not `Done`.
    fn get_iteration_done_answer(&self, target: &CalcId) -> Option<Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let iter_state = top_scc.iterative.as_ref()?;
        match iter_state.node_states.get(target)? {
            IterationNodeState::Done { answer, .. } => Some(answer.clone()),
            _ => None,
        }
    }

    /// Find the first member in the top SCC's iteration state that is `Fresh`.
    ///
    /// Returns `None` if all members have been processed or the top SCC is
    /// not iterating. BTreeMap iteration order gives deterministic results.
    fn next_fresh_member(&self) -> Option<CalcId> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let iter_state = top_scc.iterative.as_ref()?;
        for (calc_id, state) in &iter_state.node_states {
            if matches!(state, IterationNodeState::Fresh) {
                return Some(calc_id.dupe());
            }
        }
        None
    }

    /// Extract done answers from the top SCC's iteration state.
    ///
    /// Iterates over `node_states`, collecting answers from `Done` variants.
    /// Used to build `previous_answers` for the next iteration. Returns an
    /// empty map if the top SCC is not iterating.
    #[allow(dead_code, clippy::mutable_key_type)]
    fn extract_previous_answers(&self) -> BTreeMap<CalcId, Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        let Some(top_scc) = scc_stack.last() else {
            return BTreeMap::new();
        };
        let Some(iter_state) = top_scc.iterative.as_ref() else {
            return BTreeMap::new();
        };
        let mut answers = BTreeMap::new();
        for (calc_id, state) in &iter_state.node_states {
            if let IterationNodeState::Done { answer, .. } = state {
                answers.insert(calc_id.dupe(), answer.clone());
            }
        }
        answers
    }

    /// Read the demotion and convergence flags from the top SCC's iteration state.
    ///
    /// Returns `(demoted, has_changed)`. Panics if the top SCC is not iterating.
    #[allow(dead_code)]
    fn read_iteration_outcome(&self) -> (bool, bool) {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last().expect("no SCC on the stack");
        let iter_state = top_scc
            .iterative
            .as_ref()
            .expect("top SCC is not iterating");
        (iter_state.demoted, iter_state.has_changed)
    }

    /// Push an SCC onto the SCC stack.
    ///
    /// Used by the iteration driver between iterations: the SCC is popped,
    /// mutated (iteration state updated), and pushed back for the next
    /// iteration.
    fn push_scc(&self, scc: Scc) {
        self.scc_stack.borrow_mut().push(scc);
    }

    /// Pop the top SCC from the SCC stack and return it.
    ///
    /// Used by the iteration driver between iterations to extract the SCC
    /// for mutation before pushing it back with updated iteration state.
    ///
    /// Panics if the SCC stack is empty.
    fn pop_scc(&self) -> Scc {
        self.scc_stack
            .borrow_mut()
            .pop()
            .expect("pop_scc: SCC stack is empty")
    }

    /// Return the `detected_at` of the top SCC on the stack.
    ///
    /// Used by the iteration driver for absorption detection: if the top
    /// SCC's `detected_at` changed, this SCC was merged into an ancestor.
    ///
    /// Panics if the SCC stack is empty.
    fn top_scc_detected_at(&self) -> CalcId {
        self.scc_stack
            .borrow()
            .last()
            .expect("top_scc_detected_at: SCC stack is empty")
            .detected_at
            .dupe()
    }
}

/// Tracks the state of a node within an active SCC.
///
/// This replaces the previous stack-based tracking (recursion_stack, unwind_stack)
/// with explicit state tracking. The state transitions are:
/// - Fresh → InProgress (when we first encounter the node as a Participant)
/// - InProgress → HasPlaceholder (when this is a break_at node and we record a placeholder)
/// - InProgress/HasPlaceholder → Done (when the node's calculation completes)
///
/// The variants are ordered by "advancement" (Fresh < InProgress < HasPlaceholder < Done).
/// The `advancement_rank()` method encodes this ordering for use during SCC merge.
#[derive(Debug, Clone)]
enum NodeState {
    /// Node hasn't been processed yet as part of SCC handling.
    Fresh,
    /// Node is currently being processed (on the Rust call stack).
    InProgress,
    /// This is a break_at node: we've recorded a placeholder in SCC-local state
    /// but haven't computed the real answer yet.
    /// The Var is the placeholder variable recorded for this break_at node.
    HasPlaceholder(Var),
    /// Node's calculation has completed. Stores the type-erased answer and
    /// error collector for thread-local SCC isolation.
    ///
    /// For SCC participants, the answer is stored here until the entire SCC
    /// completes, at which point batch_commit_scc writes all answers to
    /// their respective Calculation cells.
    ///
    /// The data is `None` when the node was already computed by another
    /// path (e.g. a Participant revisit) and only the state transition
    /// to Done matters.
    Done {
        answer: Option<Arc<dyn Any + Send + Sync>>,
        errors: Option<Arc<ErrorCollector>>,
    },
}

impl NodeState {
    /// Returns a numeric rank for the advancement level of this state.
    /// Used during SCC merge to keep the more advanced state.
    /// Fresh(0) < InProgress(1) < HasPlaceholder(2) < Done(3)
    fn advancement_rank(&self) -> u8 {
        match self {
            NodeState::Fresh => 0,
            NodeState::InProgress => 1,
            NodeState::HasPlaceholder(_) => 2,
            NodeState::Done { .. } => 3,
        }
    }
}

/// Represents the current SCC state prior to attempting a particular calculation.
enum SccState {
    /// The current idx is not participating in any currently detected SCC (though it
    /// remains possible we will detect one here).
    ///
    /// Note that this does not necessarily mean there is no active SCC: the
    /// graph solve will frequently branch out from an SCC into other parts of
    /// the dependency graph, and in those cases we are not in a currently-known
    /// SCC.
    NotInScc,
    /// The current idx is in an active SCC but is already being processed
    /// (NodeState::InProgress). This represents a back-edge through an in-progress
    /// calculation - we've hit this node via a different path while it's still computing.
    ///
    /// This will trigger new cycle detection via propose_calculation().
    RevisitingInProgress,
    /// The current idx is in an active SCC but its calculation has already completed
    /// (NodeState::Done). A preliminary answer should be available.
    RevisitingDone,
    /// This idx is part of the active SCC, and we are either (if this is a pre-calculation
    /// check) recursing out toward `break_at` or unwinding back toward `break_at`.
    Participant,
    /// This idx has already recorded a placeholder but hasn't computed the real
    /// answer yet. We should return the placeholder value.
    HasPlaceholder,
    /// This idx is the `break_at` for the active SCC (in the break_at set but
    /// hasn't recorded a placeholder yet), which means we have reached the end
    /// of the recursion and should return a placeholder to our parent frame.
    BreakAt,
}

/// Check if the given stack length is within an SCC's segment.
///
/// Returns true if stack_len < anchor_pos + segment_size, meaning
/// we're currently inside the SCC's segment (haven't exited).
/// The segment covers positions [anchor_pos, anchor_pos + segment_size),
/// so at exactly anchor_pos + segment_size we've exited.
fn is_within_scc_segment(stack_len: usize, scc: &Scc) -> bool {
    stack_len < scc.anchor_pos + scc.segment_size
}

enum SccDetectedResult {
    /// Break immediately at the idx where we detected the SCC, so that we
    /// unwind back to the same idx.
    BreakHere,
    /// Continue recursing until we hit some other idx that is the minimal `break_at` idx.
    Continue,
}

/// The action to take for a binding after checking SCC state and calculation proposal.
///
/// This flattens the nested match on `SccState` and `ProposalResult` into a single
/// discriminated union. The `CalcStack::push` method performs all state checks and
/// SCC mutations (like `merge_sccs`, `on_scc_detected`, `on_calculation_finished`),
/// returning the action that `get_idx` should take.
enum BindingAction<T> {
    /// Calculate the binding and record the answer.
    /// Action: call `calculate_and_record_answer`
    Calculate,
    /// We are at a break point and need to unwind the cycle with a placeholder.
    /// Action: call `attempt_to_unwind_cycle_from_here`
    Unwind,
    /// A final answer is already available.
    /// Action: return `v`
    Calculated(T),
    /// A recursive placeholder exists (in SCC-local `NodeState::HasPlaceholder`)
    /// and we should return it.
    /// Action: return `Arc::new(K::promote_recursive(heap, r))`
    CycleBroken(Var),
    /// An answer is available from NodeState::Done in the top SCC.
    /// Type-erased; will be downcast to `Arc<K::Answer>` in `get_idx`.
    /// Action: downcast and return
    SccLocalAnswer(Arc<dyn Any + Send + Sync>),
    /// An iterating SCC member is InProgress with no placeholder and no
    /// previous answer (cold-start back-edge). Two-step protocol: `push`
    /// returns this because it lacks `K: Solve`; the caller (`get_idx`)
    /// allocates the placeholder via `K::create_recursive`, stores it in
    /// iteration state, and returns `K::promote_recursive`.
    NeedsColdPlaceholder,
}

/// Per-SCC iteration state for iterative fixpoint solving.
///
/// This tracks the current iteration number, per-node progress within the
/// iteration, warm-start answers from the previous iteration, and flags
/// for demotion (membership expansion) and convergence (answer stability).
///
/// Iteration state is SCC-scoped so that disjoint SCCs can iterate
/// independently.
#[derive(Debug, Clone)]
pub struct SccIterationState {
    /// Current iteration number (starts at 1).
    pub iteration: u32,
    /// Per-node iteration tracking. Membership is `node_states.keys()`.
    pub node_states: BTreeMap<CalcId, IterationNodeState>,
    /// Answers from the prior iteration, used for warm-start on back-edges.
    /// Empty on iteration 1 (cold start).
    pub previous_answers: BTreeMap<CalcId, Arc<dyn Any + Send + Sync>>,
    /// Whether SCC membership expanded during this iteration (requires
    /// restarting at iteration 1 with fresh state).
    pub demoted: bool,
    /// Whether any answer changed compared to `previous_answers` during
    /// this iteration. When `false` after iteration >= 2, the SCC has
    /// converged.
    pub has_changed: bool,
}

/// Tracks the state of a node within a single iteration of iterative SCC solving.
///
/// This is separate from the legacy `NodeState` used during Phase 0 discovery.
/// State transitions within one iteration:
/// - `Fresh` -> `InProgress` (when we start solving this node)
/// - `InProgress` -> `Done` (when the node's calculation completes)
///
/// The `placeholder` in `InProgress` is set when a cold-start back-edge
/// allocates a recursive variable for cycle breaking.
#[derive(Debug, Clone)]
pub enum IterationNodeState {
    /// Not yet processed in this iteration.
    Fresh,
    /// Currently being solved; may have a placeholder for cycle breaking.
    InProgress {
        /// Placeholder variable allocated for cold-start cycle breaking.
        /// `None` until a back-edge triggers `NeedsColdPlaceholder`.
        placeholder: Option<Var>,
    },
    /// Solved in this iteration. Stores the type-erased answer and errors.
    Done {
        /// The computed answer for this node in this iteration.
        answer: Arc<dyn Any + Send + Sync>,
        /// Errors collected during this iteration. None for cold-start.
        errors: Option<Arc<ErrorCollector>>,
    },
}

/// Lightweight summary of an `IterationNodeState` for borrow-safe read-then-act
/// patterns.
///
/// Reading the full `IterationNodeState` requires borrowing the SCC, but we
/// often need to drop that borrow before mutating. This enum captures just
/// enough information to decide what action to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationNodeStateKind {
    /// Node has not been processed yet in this iteration.
    Fresh,
    /// Node is in progress and a previous answer is available for warm-start.
    InProgressWithPreviousAnswer,
    /// Node is in progress and a placeholder variable exists for cycle breaking.
    InProgressWithPlaceholder,
    /// Node is in progress with neither a previous answer nor a placeholder
    /// (cold start, first encounter).
    InProgressCold,
    /// Node has been solved in this iteration.
    Done,
}

impl IterationNodeState {
    /// Compute the lightweight summary kind from this state plus whether a
    /// previous answer exists for the same node.
    pub fn kind(&self, has_previous_answer: bool) -> IterationNodeStateKind {
        match self {
            IterationNodeState::Fresh => IterationNodeStateKind::Fresh,
            IterationNodeState::InProgress {
                placeholder: Some(_),
            } => IterationNodeStateKind::InProgressWithPlaceholder,
            IterationNodeState::InProgress { placeholder: None } => {
                if has_previous_answer {
                    IterationNodeStateKind::InProgressWithPreviousAnswer
                } else {
                    IterationNodeStateKind::InProgressCold
                }
            }
            IterationNodeState::Done { .. } => IterationNodeStateKind::Done,
        }
    }
}

/// Represent an SCC (Strongly Connected Component) we are currently solving.
///
/// This simplified model tracks SCC participants with explicit state rather than
/// using separate recursion and unwind stacks. The Rust call stack naturally
/// enforces LIFO ordering, so we only need to track:
/// - Which idx is the anchor where we break the SCC
/// - The state of each participant (Fresh/InProgress/Done)
#[derive(Debug, Clone)]
pub struct Scc {
    /// Where do we want to break the SCC.
    /// TODO(stroxler):
    /// - This is a set because when SCCs overlap and are merged, we preserve
    ///   all the original break points to maintain behavioral equivalence with
    ///   solving each cycle independently, which is what Pyrefly used to do.
    /// - One goal of solving at the SCC granularity is to eventually eliminate
    ///   this behavior, which can cost excessive stack space, in favor of
    ///   an algorithm that breaks recursion faster.
    break_at: BTreeSet<CalcId>,
    /// State of each participant in this SCC.
    /// Keys are all participants; values track their computation state.
    node_state: BTreeMap<CalcId, NodeState>,
    /// Where we detected the SCC (for debugging only)
    detected_at: CalcId,
    /// Stack position of the SCC anchor (the position of the detected_at CalcId).
    /// The detected_at CalcId is the one that was pushed twice, triggering cycle
    /// detection; its first occurrence is at the deepest position in the cycle
    /// (cycle_start), making it a robust anchor.
    /// When the stack length drops to anchor_pos, the SCC is complete.
    /// This enables O(1) completion checking instead of iterating all participants.
    anchor_pos: usize,
    /// Number of CalcIds in this SCC segment.
    /// This is the count of stack frames that belong to this SCC.
    /// Initially the cycle size; grows on merge.
    segment_size: usize,
    /// Iteration state for iterative fixpoint solving.
    /// `None` during Phase 0 discovery (legacy SCC tracking).
    /// `Some(...)` when the SCC is being iteratively solved.
    iterative: Option<SccIterationState>,
}

impl Display for Scc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let states: Vec<_> = self.node_state.iter().collect();
        write!(
            f,
            "Scc{{break_at: [{}], node_state: {:?}, detected_at: {}}}",
            self.break_at.iter().format(", "),
            states,
            self.detected_at,
        )
    }
}

impl Scc {
    #[allow(clippy::mutable_key_type)] // CalcId's Hash impl doesn't depend on mutable parts
    fn new(raw: Vec1<CalcId>, calc_stack_vec: &[CalcId]) -> Self {
        let detected_at = raw.first().dupe();
        let (_, break_at) = raw.iter().enumerate().min_by_key(|(_, c)| *c).unwrap();

        // Initialize all nodes as Fresh
        let node_state: BTreeMap<CalcId, NodeState> =
            raw.iter().duped().map(|c| (c, NodeState::Fresh)).collect();

        let mut break_at_set = BTreeSet::new();
        break_at_set.insert(break_at.dupe());

        // The anchor is the detected_at CalcId (the one pushed twice, triggering cycle
        // detection). Its first occurrence is at the deepest position in the cycle
        // (cycle_start), making it a more robust anchor than break_at.
        //
        // The initial segment size is the number of frames from anchor to top of stack.
        let anchor_pos = calc_stack_vec
            .iter()
            .position(|c| c == &detected_at)
            .unwrap_or(0);
        let segment_size = calc_stack_vec.len() - anchor_pos;

        Scc {
            break_at: break_at_set,
            node_state,
            detected_at,
            anchor_pos,
            segment_size,
            iterative: None,
        }
    }

    /// Check if the current idx is a participant in this SCC and determine its state.
    ///
    /// Returns the appropriate SccState:
    /// - BreakAt if this is the anchor where we produce a placeholder
    /// - Participant if this is a Fresh node (marks it as InProgress)
    /// - RevisitingInProgress if this idx is InProgress (back-edge through in-progress node)
    /// - RevisitingDone if this idx is Done (preliminary answer should exist)
    /// - NotInScc if this idx is not in the SCC
    ///
    /// When a Fresh node is encountered, it transitions to InProgress.
    fn pre_calculate_state(&mut self, current: &CalcId) -> SccState {
        if self.break_at.contains(current) {
            // For break_at nodes that already have a placeholder or are Done,
            // return the state-appropriate response. BreakAt (which triggers
            // Unwind -> attempt_to_unwind_cycle_from_here -> on_placeholder_recorded)
            // should only fire when the node is Fresh or InProgress, i.e. the
            // break has not happened yet.
            //
            // Without this guard, revisiting a Done break_at node would cause
            // on_placeholder_recorded to overwrite Done back to HasPlaceholder,
            // losing the stored answer needed for batch commit.
            if let Some(state) = self.node_state.get(current) {
                match state {
                    NodeState::HasPlaceholder(_) => return SccState::HasPlaceholder,
                    NodeState::Done { .. } => return SccState::RevisitingDone,
                    NodeState::Fresh | NodeState::InProgress => {}
                }
            }
            SccState::BreakAt
        } else if let Some(state) = self.node_state.get_mut(current) {
            match state {
                NodeState::Fresh => {
                    *state = NodeState::InProgress;
                    SccState::Participant
                }
                NodeState::InProgress => {
                    // Back-edge: we're hitting a node currently on the call stack
                    // via a different path. This will trigger new cycle detection.
                    SccState::RevisitingInProgress
                }
                NodeState::HasPlaceholder(_) => {
                    // Already has placeholder, return it
                    SccState::HasPlaceholder
                }
                NodeState::Done { .. } => {
                    // Node completed within this SCC - preliminary answer should exist.
                    SccState::RevisitingDone
                }
            }
        } else {
            SccState::NotInScc
        }
    }

    /// Track that a calculation has finished, marking it as Done.
    /// Stores the type-erased answer and error collector in NodeState.
    /// For SCC participants, this is the primary storage until batch commit.
    ///
    /// This method implements first-answer-wins semantics: once a node is marked
    /// as Done, subsequent calculations (from duplicate stack frames within an SCC)
    /// do not overwrite the state. This ensures that the first computed answer is
    /// the one that persists, consistent with Calculation::record_value semantics.
    ///
    /// Returns the canonical answer: the one that is (or was already) stored in
    /// NodeState::Done. If the node was already Done, returns the pre-existing
    /// answer without overwriting. If the node was not yet Done, stores the
    /// provided answer and returns a clone of it. If the node is not tracked
    /// by this SCC at all, returns the provided answer unchanged.
    ///
    /// The data is `None` when the node was already computed by another
    /// path and only the state transition matters.
    fn on_calculation_finished(
        &mut self,
        current: &CalcId,
        answer: Option<Arc<dyn Any + Send + Sync>>,
        errors: Option<Arc<ErrorCollector>>,
    ) -> Option<Arc<dyn Any + Send + Sync>> {
        if let Some(state) = self.node_state.get_mut(current) {
            if let NodeState::Done {
                answer: existing_answer,
                ..
            } = state
            {
                // Already Done: return the canonical (first-written) answer.
                existing_answer.clone()
            } else {
                *state = NodeState::Done {
                    answer: answer.clone(),
                    errors,
                };
                answer
            }
        } else {
            // Node not tracked by this SCC; return the provided answer as-is.
            answer
        }
    }

    /// Track that a placeholder has been recorded for a break_at node.
    fn on_placeholder_recorded(&mut self, current: &CalcId, var: Var) {
        if let Some(state) = self.node_state.get_mut(current) {
            // Only upgrade: do not overwrite Done back to HasPlaceholder.
            // This is defense-in-depth; pre_calculate_state should prevent
            // this path from being reached for Done nodes.
            if state.advancement_rank() < NodeState::HasPlaceholder(var).advancement_rank() {
                *state = NodeState::HasPlaceholder(var);
            }
        }
    }

    /// Get the detection point of this SCC (stable identifier for merging).
    fn detected_at(&self) -> CalcId {
        self.detected_at.dupe()
    }

    /// Merge two SCCs into one, preserving all break points and taking the
    /// most advanced state for each participant.
    ///
    /// If either SCC has iteration state (`iterative: Some(...)`), the merged
    /// SCC restarts at iteration 1 with fresh node states, cleared previous
    /// answers, and `demoted = true`. The members of the merged iteration state
    /// come from the already-merged `node_state.keys()` (the legacy SCC
    /// membership), which is the union of both SCCs' members. This is
    /// important because a non-iterating SCC has `iterative: None` but still
    /// has members in `node_state`.
    #[allow(clippy::mutable_key_type)]
    fn merge(mut self, other: Scc) -> Self {
        // Union break_at sets
        self.break_at.extend(other.break_at);
        // Union node_state maps (keep the more advanced state)
        for (k, v) in other.node_state {
            self.node_state
                .entry(k)
                .and_modify(|existing| {
                    if v.advancement_rank() > existing.advancement_rank() {
                        *existing = v.clone();
                    }
                })
                .or_insert(v);
        }
        // Keep the smallest detected_at for consistency/determinism
        self.detected_at = self.detected_at.min(other.detected_at);
        // Keep the minimum anchor position
        self.anchor_pos = self.anchor_pos.min(other.anchor_pos);
        // Note: segment_size is NOT updated here. After a merge, everything from
        // the merged anchor to the current stack top is part of this single SCC.
        // The caller must recompute segment_size = stack.len() - anchor_pos.

        // Merge iteration state: if either SCC is iterating, the merged SCC
        // restarts at iteration 1 with all members fresh and demoted = true.
        // Members come from self.node_state.keys() (the already-merged legacy
        // membership), NOT from iterative.node_states.keys(), because a
        // non-iterating SCC has iterative: None but still has members.
        self.iterative = match (self.iterative.take(), other.iterative) {
            (None, None) => None,
            (Some(_), _) | (_, Some(_)) => {
                // At least one SCC is iterating. Build fresh iteration state
                // with ALL members from the merged node_state map.
                let all_members: BTreeMap<CalcId, IterationNodeState> = self
                    .node_state
                    .keys()
                    .duped()
                    .map(|k| (k, IterationNodeState::Fresh))
                    .collect();
                Some(SccIterationState {
                    iteration: 1,
                    node_states: all_members,
                    previous_answers: BTreeMap::new(),
                    demoted: true,
                    has_changed: false,
                })
            }
        };

        self
    }

    /// Merge multiple SCCs into one.
    ///
    /// The `detected_at` parameter is an additional candidate for the minimum
    /// detected_at, used when the detection point may not be represented in
    /// any of the SCCs being merged.
    #[cfg_attr(test, allow(dead_code))]
    fn merge_many(sccs: Vec1<Scc>, detected_at: CalcId) -> Self {
        let (first, rest) = sccs.split_off_first();
        let mut result = rest.into_iter().fold(first, Scc::merge);
        if detected_at < result.detected_at {
            result.detected_at = detected_at;
        }
        result
    }

    /// Extract done answers from the current iteration state.
    ///
    /// Iterates over the iteration `node_states`, collecting answers from
    /// `Done` variants into a `BTreeMap`. Used to build `previous_answers`
    /// for the next iteration. Returns an empty map if the SCC has no
    /// iteration state.
    #[allow(clippy::mutable_key_type)]
    fn extract_done_answers(&self) -> BTreeMap<CalcId, Arc<dyn Any + Send + Sync>> {
        let Some(iter_state) = self.iterative.as_ref() else {
            return BTreeMap::new();
        };
        let mut answers = BTreeMap::new();
        for (calc_id, state) in &iter_state.node_states {
            if let IterationNodeState::Done { answer, .. } = state {
                answers.insert(calc_id.dupe(), answer.clone());
            }
        }
        answers
    }

    /// Set up fresh iteration state for the next iteration.
    ///
    /// All members from the legacy `node_state` map are reset to `Fresh`
    /// in the iteration `node_states` map. This is called between iterations
    /// in the pop-mutate-push cycle.
    #[allow(clippy::mutable_key_type)]
    fn set_fresh_iteration_state(
        &mut self,
        iteration: u32,
        previous_answers: BTreeMap<CalcId, Arc<dyn Any + Send + Sync>>,
    ) {
        let all_members: BTreeMap<CalcId, IterationNodeState> = self
            .node_state
            .keys()
            .duped()
            .map(|k| (k, IterationNodeState::Fresh))
            .collect();
        self.iterative = Some(SccIterationState {
            iteration,
            node_states: all_members,
            previous_answers,
            demoted: false,
            has_changed: false,
        });
    }

    /// Returns true if this SCC has break_at entries, indicating a true
    /// cycle with self-loops. A singleton SCC without self-loops has an
    /// empty break_at set and should not be iteratively solved.
    fn has_break_at(&self) -> bool {
        !self.break_at.is_empty()
    }
}

/// Represents thread-local state for the current `AnswersSolver` and any
/// `AnswersSolver`s waiting for the results that we are currently computing.
///
/// This state is initially created by some top-level `AnswersSolver` - when
/// we're calculating results for bindings, we started at either:
/// - a solver that is type-checking some module end-to-end, or
/// - an ad-hoc solver (used in some LSP functionality) solving one specific binding
///
/// We'll create a new `AnswersSolver` will change every time we switch modules,
/// which happens as we resolve types of imported names, but when this happens
/// we always pass the current `ThreadState`.
pub struct ThreadState {
    stack: CalcStack,
    /// For debugging only: thread-global that allows us to control debug logging across components.
    debug: RefCell<bool>,
    /// Configuration for recursion depth limiting. None means disabled.
    recursion_limit_config: Option<RecursionLimitConfig>,
    /// How SCC participants store answers during solving.
    scc_solving_mode: SccSolvingMode,
    /// Partial answers for inline first-use pinning, keyed by (NameAssign def_idx, CalcStack height).
    /// The height ensures that only ForwardToFirstUse bindings at the same CalcStack depth
    /// as the NameAssign's solve_binding can see the partial answer (offset 0 in get_idx,
    /// which checks before pushing its own frame).
    partial_answers: RefCell<FxHashMap<(Idx<Key>, usize), Arc<TypeInfo>>>,
}

/// Internal SCC-solving modes controlled via `PYREFLY_SCC_SOLVING_MODE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SccSolvingMode {
    /// Write SCC participant answers to Calculation immediately.
    #[default]
    CyclesDualWrite,
    /// Thread-local SCC solving with batch commits to Calculation.
    CyclesThreadLocal,
    /// Iterative fixpoint: re-solve SCC members until answers converge.
    Iterative,
}

impl ThreadState {
    pub fn new(recursion_limit_config: Option<RecursionLimitConfig>, scc_mode: SccMode) -> Self {
        let scc_solving_mode = SccSolvingMode::resolve(scc_mode);
        Self {
            stack: CalcStack::new(scc_solving_mode),
            debug: RefCell::new(false),
            recursion_limit_config,
            scc_solving_mode,
            partial_answers: RefCell::new(FxHashMap::default()),
        }
    }

    fn scc_solving_mode(&self) -> SccSolvingMode {
        self.scc_solving_mode
    }
}

impl SccSolvingMode {
    /// Resolve the SCC solving mode from a config enum, with an env var override.
    ///
    /// The `PYREFLY_SCC_SOLVING_MODE` env var takes precedence over the config
    /// value, allowing runtime experimentation without config changes.
    fn resolve(mode: SccMode) -> Self {
        match env::var("PYREFLY_SCC_SOLVING_MODE") {
            Ok(value) => match value.as_str() {
                "cycles-dual-write" => Self::CyclesDualWrite,
                "cycles-thread-local" => Self::CyclesThreadLocal,
                "iterative-fixpoint" => Self::Iterative,
                _ => panic!(
                    "$PYREFLY_SCC_SOLVING_MODE must be one of \
                     `cycles-dual-write`, `cycles-thread-local`, \
                     or `iterative-fixpoint`, got `{value}`"
                ),
            },
            Err(_) => match mode {
                SccMode::CyclesDualWrite => Self::CyclesDualWrite,
                SccMode::CyclesThreadLocal => Self::CyclesThreadLocal,
                SccMode::IterativeFixpoint => Self::Iterative,
            },
        }
    }
}

/// Maximum number of fixpoint iterations before the iterative SCC solver
/// gives up and commits the last answers. Exceeding this threshold logs a
/// warning but does not panic, since the answers may still be approximately
/// correct.
const MAX_ITERATIONS: u32 = 5;

/// Maximum number of demotion restarts (SCC membership expansions) before
/// the iterative SCC solver panics. Exceeding this threshold almost
/// certainly indicates an infinite membership expansion loop rather than
/// legitimate growth.
const MAX_DEMOTIONS: u32 = 10;

/// Check whether the demotion count has exceeded `MAX_DEMOTIONS`, and panic
/// if so. Extracted from the `iterative_resolve_scc` loop to allow direct
/// unit testing of the safety limit.
///
/// Uses `Debug` formatting for `scc_identity` rather than `Display` because
/// `CalcId::Display` requires a populated bindings table (which panics in
/// test contexts), while `CalcId::Debug` prints the raw index safely.
fn check_demotion_limit(demotions: u32, scc_identity: &CalcId) {
    if demotions > MAX_DEMOTIONS {
        panic!(
            "iterative_resolve_scc: SCC {:?} exceeded {} demotions; \
             likely infinite membership expansion",
            scc_identity, MAX_DEMOTIONS,
        );
    }
}

pub struct AnswersSolver<'a, Ans: LookupAnswer> {
    answers: &'a Ans,
    current: &'a Answers,
    thread_state: &'a ThreadState,
    // The base solver is only used to reset the error collector at binding
    // boundaries. Answers code should generally use the error collector passed
    // along the call stack instead.
    base_errors: &'a ErrorCollector,
    bindings: &'a Bindings,
    pub exports: &'a dyn LookupExport,
    pub uniques: &'a UniqueFactory,
    pub recurser: &'a VarRecurser,
    pub stdlib: &'a Stdlib,
    pub heap: &'a TypeHeap,
    /// Cache for jaxtyping dimension name → Quantified type mappings.
    /// Module-scoped: the same dimension name always maps to the same Quantified,
    /// which is correct because each function independently wraps its signature
    /// in a Forall (just like legacy TypeVars defined at module scope).
    jaxtyping_dims: RefCell<FxHashMap<Name, Quantified>>,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn new(
        answers: &'a Ans,
        current: &'a Answers,
        base_errors: &'a ErrorCollector,
        bindings: &'a Bindings,
        exports: &'a dyn LookupExport,
        uniques: &'a UniqueFactory,
        recurser: &'a VarRecurser,
        stdlib: &'a Stdlib,
        thread_state: &'a ThreadState,
        heap: &'a TypeHeap,
    ) -> AnswersSolver<'a, Ans> {
        AnswersSolver {
            stdlib,
            uniques,
            answers,
            bindings,
            base_errors,
            exports,
            recurser,
            current,
            thread_state,
            heap,
            jaxtyping_dims: RefCell::new(FxHashMap::default()),
        }
    }

    /// Is the debug flag set? Intended to support print debugging.
    pub fn is_debug(&self) -> bool {
        *self.thread_state.debug.borrow()
    }

    /// Set the debug flag. Intended to support print debugging.
    #[allow(dead_code)]
    pub fn set_debug(&self, value: bool) {
        *self.thread_state.debug.borrow_mut() = value;
    }

    /// Get or create a Quantified type for a jaxtyping dimension name.
    /// Cached per module: the same name always returns the same Quantified.
    pub fn get_or_create_jaxtyping_dim(&self, name: Name, kind: QuantifiedKind) -> Quantified {
        let mut dims = self.jaxtyping_dims.borrow_mut();
        dims.entry(name.clone())
            .or_insert_with(|| match kind {
                QuantifiedKind::TypeVar => Quantified::type_var(
                    name,
                    self.uniques,
                    None,
                    Restriction::Unrestricted,
                    PreInferenceVariance::Invariant,
                ),
                QuantifiedKind::TypeVarTuple => {
                    Quantified::type_var_tuple(name, self.uniques, None)
                }
                QuantifiedKind::ParamSpec => {
                    unreachable!("jaxtyping dimensions cannot be ParamSpec")
                }
            })
            .clone()
    }

    /// Check if a Quantified type was created by jaxtyping dimension parsing.
    pub fn is_jaxtyping_dim(&self, q: &Quantified) -> bool {
        self.jaxtyping_dims.borrow().values().any(|v| v == q)
    }

    pub fn current(&self) -> &Answers {
        self.current
    }

    pub fn bindings(&self) -> &Bindings {
        self.bindings
    }

    pub fn base_errors(&self) -> &ErrorCollector {
        self.base_errors
    }

    pub fn module(&self) -> &ModuleInfo {
        self.bindings.module()
    }

    pub fn stack(&self) -> &CalcStack {
        &self.thread_state.stack
    }

    /// Store a partial answer for inline first-use pinning.
    /// `def_idx` is the Key::Definition idx of the NameAssign.
    /// Keyed by (def_idx, current CalcStack height).
    pub(crate) fn store_partial_answer(&self, def_idx: Idx<Key>, type_info: Arc<TypeInfo>) {
        let height = self.stack().len();
        self.thread_state
            .partial_answers
            .borrow_mut()
            .insert((def_idx, height), type_info);
    }

    /// Remove the partial answer for a NameAssign at the current height.
    pub(crate) fn clear_partial_answer(&self, def_idx: Idx<Key>) {
        let height = self.stack().len();
        self.thread_state
            .partial_answers
            .borrow_mut()
            .remove(&(def_idx, height));
    }

    /// Check for a matching partial answer at the current CalcStack height.
    ///
    /// The height check ensures that only a ForwardToFirstUse resolved at the same
    /// CalcStack depth as the NameAssign's solve_binding can see the partial answer.
    /// This is offset 0 because the check runs in `get_idx` BEFORE pushing the
    /// ForwardToFirstUse's own frame. Bindings at deeper heights (e.g., a ClassField
    /// that indirectly depends on the same variable) correctly miss the partial answer
    /// and go through normal resolution.
    pub(crate) fn check_partial_answer(&self, def_idx: Idx<Key>) -> Option<Arc<TypeInfo>> {
        let height = self.stack().len();
        self.thread_state
            .partial_answers
            .borrow()
            .get(&(def_idx, height))
            .cloned()
    }

    /// Given the target idx of a ForwardToFirstUse binding, find the NameAssign's
    /// def_idx for partial answer lookup.
    ///
    /// ForwardToFirstUse always points to a NameAssign with `def_idx.is_some()`.
    pub(crate) fn def_idx_for_forward_to_first_use(&self, target: Idx<Key>) -> Option<Idx<Key>> {
        let binding = self.bindings().get(target);
        match binding {
            Binding::NameAssign(na) if na.def_idx.is_some() => Some(target),
            _ => None,
        }
    }

    fn recursion_limit_config(&self) -> Option<RecursionLimitConfig> {
        self.thread_state.recursion_limit_config
    }

    pub fn for_display(&self, t: Type) -> Type {
        self.solver().for_display(t)
    }

    pub fn type_order(&self) -> TypeOrder<'_, Ans> {
        TypeOrder::new(self)
    }

    pub fn validate_final_thread_state(&self) {
        assert!(
            self.thread_state.stack.is_empty(),
            "The calculation stack should be empty in the final thread state"
        );
        assert!(
            self.thread_state.stack.sccs_is_empty(),
            "The SCC stack should be empty in the final thread state"
        );
    }

    pub fn get_idx<K: Solve<Ans>>(&self, idx: Idx<K>) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        // Check for a partial answer shortcut before pushing to the CalcStack.
        // This is used by ForwardToFirstUse during inline first-use pinning to
        // return the raw type without caching it in shared Answers and without
        // triggering cycle detection against the NameAssign's CalcStack frame.
        let binding = self.bindings().get(idx);
        if let Some(answer) = K::check_shortcut(self, binding) {
            return answer;
        }

        let current = CalcId(self.bindings().dupe(), K::to_anyidx(idx));
        let calculation = self.get_calculation(idx);

        // Check depth limit before any calculation
        if let Some(config) = self.recursion_limit_config()
            && self.stack().len() > config.limit as usize
        {
            let result = self.handle_depth_overflow(&current, idx, calculation, config);
            return result;
        }

        let result = match self.stack().push(current.dupe(), calculation) {
            BindingAction::Calculate => self.calculate_and_record_answer(current, idx, calculation),
            BindingAction::Unwind => self
                .attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r))),
            BindingAction::Calculated(v) => v,
            BindingAction::CycleBroken(r) => Arc::new(K::promote_recursive(self.heap, r)),
            BindingAction::SccLocalAnswer(type_erased) => {
                // Downcast the type-erased answer back to Arc<K::Answer>.
                // The answer was stored as Arc::new(answer.dupe()) where answer: Arc<K::Answer>,
                // so the concrete type inside Arc<dyn Any> is Arc<K::Answer>.
                // downcast() returns Arc<Arc<K::Answer>>; unwrap_or_clone extracts the inner Arc.
                Arc::unwrap_or_clone(
                    type_erased
                        .downcast::<Arc<K::Answer>>()
                        .expect("SccLocalAnswer downcast failed: type mismatch"),
                )
            }
            BindingAction::NeedsColdPlaceholder => {
                // Two-step protocol: push() returned NeedsColdPlaceholder because
                // it lacks K: Solve. We allocate the placeholder here, store it
                // in iteration state so subsequent back-edges return CycleBroken,
                // and return the promoted value.
                let binding = self.bindings().get(idx);
                let var = K::create_recursive(self, binding);
                self.stack().set_iteration_placeholder(&current, var);
                Arc::new(K::promote_recursive(self.heap, var))
            }
        };
        for scc in self.stack().pop_and_drain_completed_sccs() {
            if self.stack().is_iterative() && scc.has_break_at() {
                // True cycle in iterative mode: run the iterative fixpoint
                // driver instead of the one-shot batch commit.
                self.iterative_resolve_scc(scc);
            } else {
                // Legacy mode or singleton without self-loop: commit directly.
                self.batch_commit_scc(scc);
            }
        }
        result
    }
    /// Calculate the answer for a binding using `K::solve` and record it.
    ///
    /// This is called when the `push` method determines we need to actually compute the value.
    ///
    /// For SCC participants, the answer is stored in `NodeState::Done` and will be
    /// batch-committed to the `Calculation` cell when the entire SCC completes.
    /// For non-SCC nodes, the answer is written directly to `Calculation` as before.
    ///
    /// In iterative mode, if the current CalcId is a member of the top SCC's
    /// iteration state, we use a separate iterative path that:
    /// - Suppresses errors during cold-start (iteration 1) and collects them
    ///   from iteration 2 onward.
    /// - Deep-forces the answer to avoid Var-ID inequality in convergence
    ///   comparisons.
    /// - Finalizes any placeholder created for this node.
    /// - Compares to the previous iteration's answer to track convergence.
    /// - Stores the answer in SCC-local iteration state (not in Calculation)
    ///   until the final commit.
    ///
    /// Completed SCCs are pushed to the `pending_completed_sccs` buffer
    /// inside `on_calculation_finished`; `get_idx` drains them after the
    /// frame completes.
    fn calculate_and_record_answer<K: Solve<Ans>>(
        &self,
        current: CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>>,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        // Iterative path: when iterative mode is active and the current CalcId
        // is in the top SCC's iteration state, use the iterative code path
        // instead of the legacy SCC or non-SCC paths.
        if self.stack().is_iterative() && self.stack().get_iteration_node_state(&current).is_some()
        {
            return self.calculate_and_record_answer_iterative(current, idx);
        }

        let binding = self.bindings().get(idx);
        // Note that we intentionally do not pass in the key when solving the binding,
        // as the result of a binding should not depend on the key it was bound to.
        // We use the range for error reporting.
        let range = K::range_with(idx, self.bindings());

        let local_errors = self.error_collector();
        let raw_answer = K::solve(self, binding, range, &local_errors);

        // For exported keys, eagerly resolve all type variables in the answer.
        // This avoids redundant clone+force work in solve_exported_key and post_solve,
        // which would otherwise repeat this work on every cross-module lookup.
        // Arc::unwrap_or_clone avoids cloning since the refcount is 1 here.
        let raw_answer = if K::EXPORTED {
            let mut forced = Arc::unwrap_or_clone(raw_answer);
            forced.visit_mut(&mut |x| self.current.solver().deep_force_mut(x));
            Arc::new(forced)
        } else {
            raw_answer
        };

        if self.stack().is_scc_participant(&current) {
            // SCC path: store in NodeState::Done and (optionally) write to Calculation.
            //
            // If this is a break_at node (has a placeholder Var), we must finalize
            // the recursive answer now, before storing. Finalization mutates solver
            // state (force_var) and must happen during computation, not at batch commit.
            let answer = if let Some(var) = self.stack().get_scc_placeholder_var(&current) {
                self.finalize_recursive_answer::<K>(idx, var, raw_answer, &local_errors)
            } else {
                raw_answer
            };
            let (calc_answer, errors) =
                if self.thread_state.scc_solving_mode() == SccSolvingMode::CyclesDualWrite {
                    // Write to Calculation immediately for cross-thread visibility.
                    // Without this, the Calculation stays in Calculating status during
                    // SCC processing, allowing other threads to independently re-compute
                    // this binding via propose_calculation() → Calculatable. Since
                    // cycle-oriented solving is not entrypoint-invariant, independent
                    // re-computation can produce different results depending on thread
                    // scheduling, causing non-determinism.
                    let (calc_answer, did_write) = calculation.record_value(answer.dupe());
                    if did_write {
                        self.base_errors.extend(local_errors);
                    }
                    (calc_answer, None)
                } else {
                    (answer, Some(Arc::new(local_errors)))
                };
            // Also store in NodeState::Done for SCC-local isolation (the SCC
            // uses these answers via SccLocalAnswer without touching Calculation).
            let answer_erased: Arc<dyn Any + Send + Sync> = Arc::new(calc_answer.dupe());
            let canonical_erased =
                self.stack()
                    .on_calculation_finished(&current, Some(answer_erased), errors);
            // Use the canonical answer from thread-local state, mirroring how
            // Calculation::record_value returns the first-written answer.
            match canonical_erased {
                Some(erased) => Arc::unwrap_or_clone(
                    erased
                        .downcast::<Arc<K::Answer>>()
                        .expect("on_calculation_finished canonical answer downcast failed"),
                ),
                None => calc_answer,
            }
        } else {
            // Non-SCC path: write directly to Calculation as before.
            // No recursive placeholder can exist in the Calculation cell because
            // placeholders are stored only in SCC-local NodeState::HasPlaceholder.
            let (answer, did_write) = calculation.record_value(raw_answer);
            if did_write {
                self.base_errors.extend(local_errors);
            }
            self.stack().on_calculation_finished(&current, None, None);
            answer
        }
    }

    /// Iterative path for `calculate_and_record_answer`.
    ///
    /// Called when the current CalcId is a member of the top SCC's iteration
    /// state. Unlike the legacy path, this:
    /// - During cold-start iteration 1, bypasses `LoopPhi` bindings by
    ///   resolving only the prior/default index. This prevents LoopPhi from
    ///   creating its own `LoopRecursive` placeholder, which would conflict
    ///   with the iterative placeholder system.
    /// - Uses `error_swallower()` during cold-start (iteration 1) because
    ///   cold-start answers are based on placeholders and produce spurious
    ///   errors. From iteration 2 onward, uses `error_collector()` to capture
    ///   errors that will be committed if this is the final iteration.
    /// - Deep-forces the answer before storage to avoid Var-ID inequality
    ///   in convergence comparisons.
    /// - Finalizes any placeholder created for this node during cycle breaking.
    /// - Compares the answer to `previous_answers` via `answers_equal` and
    ///   calls `mark_iteration_changed` if they differ.
    /// - Stores the answer in `IterationNodeState::Done` (SCC-local), NOT in
    ///   `Calculation`. The answer is only committed to `Calculation` when
    ///   the iteration driver commits the final converged answers.
    fn calculate_and_record_answer_iterative<K: Solve<Ans>>(
        &self,
        current: CalcId,
        idx: Idx<K>,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        // LoopPhi cold-start bypass: during iteration 1, LoopPhi's normal
        // solve path would resolve loop-body branches, hit cycle breaks, and
        // create its own LoopRecursive placeholder. That conflicts with the
        // iterative placeholder system. Instead, resolve only the prior/default
        // value (the value from before the loop body) and use it as the answer.
        // On warm-start iterations (>= 2), LoopPhi goes through the normal
        // path and gets the previous iteration's answer via the iterative
        // bypass, which converges correctly.
        if self.stack().is_cold_iteration()
            && let AnyIdx::Key(key_idx) = current.1
        {
            // Use explicit Key type parameter because `key_idx` is `Idx<Key>`
            // (from the AnyIdx::Key match), not `Idx<K>`. The function is
            // generic over K, but we know K = Key here; Rust's type system
            // requires the concrete type to resolve the binding table lookup.
            let key_binding = self.bindings().get::<Key>(key_idx);
            if let Binding::LoopPhi(prior_idx, _) = key_binding {
                // Resolve the prior/default index (value from above the loop).
                // Uses get_idx::<Key> explicitly since prior_idx is Idx<Key>.
                let prior_answer: Arc<TypeInfo> = self.get_idx::<Key>(*prior_idx);

                // Deep-force to resolve all type variables, matching the
                // invariant that all iterative answers are deep-forced.
                let mut forced = Arc::unwrap_or_clone(prior_answer);
                forced.visit_mut(&mut |x| self.current.solver().deep_force_mut(x));
                let answer = Arc::new(forced);

                // Type-erase for storage. The concrete type inside the outer
                // Arc<dyn Any> is Arc<TypeInfo> (= Arc<Key::Answer>).
                let answer_erased: Arc<dyn Any + Send + Sync> = Arc::new(answer.dupe());

                // Cold start has no previous answer; always mark changed so
                // iteration 1 never appears converged.
                self.stack().mark_iteration_changed();

                // Store as Done in iteration state. Errors are None because
                // this is cold-start iteration 1 (errors are swallowed).
                self.stack()
                    .set_iteration_node_done(&current, answer_erased.clone(), None);

                // Downcast back to Arc<K::Answer>. This code path only
                // executes when K = Key (guarded by AnyIdx::Key match), so
                // K::Answer = TypeInfo and the downcast always succeeds.
                return Arc::unwrap_or_clone(
                    answer_erased
                        .downcast::<Arc<K::Answer>>()
                        .expect("LoopPhi bypass: K must be Key when AnyIdx::Key matches"),
                );
            }
        }

        let binding = self.bindings().get(idx);
        let range = K::range_with(idx, self.bindings());

        // Error handling strategy:
        // - Iteration 1 (cold): suppress all errors because cold-start answers
        //   (from placeholders) produce spurious diagnostics.
        // - Iteration >= 2: collect errors normally. Only the final iteration's
        //   errors are committed.
        let local_errors = if self.stack().is_cold_iteration() {
            self.error_swallower()
        } else {
            self.error_collector()
        };
        let raw_answer = K::solve(self, binding, range, &local_errors);

        // If a placeholder was created for this node during cycle breaking,
        // finalize the recursive answer (unify the placeholder with the actual
        // answer via record_recursive + force_var). This must happen BEFORE
        // deep-forcing: finalization sets the placeholder Var's answer in the
        // solver, so a subsequent deep-force correctly resolves it. Reversing
        // the order would leave the placeholder Var unresolved during forcing.
        let answer = if let Some(var) = self.stack().get_iteration_placeholder(&current) {
            self.finalize_recursive_answer::<K>(idx, var, raw_answer, &local_errors)
        } else {
            raw_answer
        };

        // Deep-force the answer to resolve all type variables. This is required
        // for convergence comparisons: without forcing, structurally identical
        // answers can appear different due to unresolved Var IDs.
        let mut forced = Arc::unwrap_or_clone(answer);
        forced.visit_mut(&mut |x| self.current.solver().deep_force_mut(x));
        let answer = Arc::new(forced);

        // Type-erase the answer for storage in iteration state.
        // Wrap in Arc::new() so the concrete type inside Arc<dyn Any> is
        // Arc<K::Answer>, matching downcasts in answers_equal_typed and
        // SccLocalAnswer handling.
        let answer_erased: Arc<dyn Any + Send + Sync> = Arc::new(answer.dupe());

        // Compare to the previous iteration's answer (if any) to detect
        // convergence. If the answer has changed, the fixpoint has not yet
        // converged and the iteration driver must run another iteration.
        if let Some(previous) = self.stack().get_previous_answer(&current) {
            if !self.answers_equal(&current.1, &previous, &answer_erased) {
                self.stack().mark_iteration_changed();
            }
        } else {
            // No previous answer (cold start): always mark changed so that
            // iteration 1 never appears converged.
            self.stack().mark_iteration_changed();
        }

        // Store in IterationNodeState::Done. Do NOT write to Calculation;
        // that happens only when the iteration driver commits final answers.
        let errors = if self.stack().is_cold_iteration() {
            None
        } else {
            Some(Arc::new(local_errors))
        };
        self.stack()
            .set_iteration_node_done(&current, answer_erased, errors);

        answer
    }

    /// Commit a type-erased answer to the Calculation cell for a same-module binding.
    /// Used during batch commit when an SCC completes.
    ///
    /// The answer was already finalized (force_var called) during calculate_and_record_answer,
    /// so we use a no-op finalizer here. Errors are only extended into base_errors if
    /// this thread's write wins (did_write = true).
    fn commit_typed<K: Solve<Ans>>(
        &self,
        idx: Idx<K>,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
    ) where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let typed_answer: Arc<K::Answer> = Arc::unwrap_or_clone(
            answer
                .downcast::<Arc<K::Answer>>()
                .expect("commit_typed: type mismatch in batch commit"),
        );
        let calculation = self.get_calculation(idx);
        // No recursive placeholder can exist in the Calculation cell because
        // placeholders are stored only in SCC-local NodeState::HasPlaceholder.
        // The answer was already finalized (force_var called) during
        // calculate_and_record_answer's SCC path.
        let (_answer, did_write) = calculation.record_value(typed_answer);
        if did_write && let Some(errors) = errors {
            // ErrorCollector::extend takes ownership. Arc::try_unwrap succeeds
            // because the Arc refcount is 1: the ErrorCollector is moved (not
            // cloned) at every step from creation through NodeState::Done, SCC
            // completion, and into this consuming iteration. (Scc::merge may
            // transiently clone a NodeState, but the original is dropped
            // immediately, so the refcount returns to 1 before we get here.)
            let errors = Arc::try_unwrap(errors).expect(
                "Arc<ErrorCollector> refcount > 1 during batch commit; \
                 errors would be silently lost. This indicates a bug in SCC lifecycle management.",
            );
            self.base_errors.extend(errors);
        }
    }

    /// Commit a single preliminary result from a completed SCC.
    /// Uses dispatch_anyidx! to recover the concrete key type from AnyIdx,
    /// then delegates to commit_typed<K> for same-module commits or
    /// LookupAnswer::commit_to_module for cross-module commits.
    fn commit_single_result(
        &self,
        calc_id: CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
    ) {
        let CalcId(ref bindings, ref any_idx) = calc_id;
        if bindings.module().name() != self.bindings().module().name()
            || bindings.module().path() != self.bindings().module().path()
        {
            // Cross-module: delegate to the LookupAnswer trait to route
            // the commit to the correct module's Answers.
            assert!(
                self.answers
                    .commit_to_module(calc_id.dupe(), answer, errors),
                "commit_single_result: cross-module commit failed for {}. \
                 The target module's Answers may not be loaded, which would \
                 leave its Calculation cell stuck in Calculating state.",
                calc_id,
            );
            return;
        }
        dispatch_anyidx!(any_idx, self, commit_typed, answer, errors)
    }

    /// Batch-commit all preliminary answers from a completed SCC.
    /// Iterates the SCC's node_state map and commits each Done entry
    /// to the appropriate Calculation cell.
    ///
    /// Invariant: all SCC participants should be in `Done` state at this point.
    /// Nodes with `answer: None` are skipped (they were already committed by
    /// another thread via the Participant revisit path).
    fn batch_commit_scc(&self, completed_scc: Scc) {
        for (calc_id, node_state) in completed_scc.node_state {
            match node_state {
                NodeState::Done {
                    answer: Some(answer),
                    errors,
                } => {
                    self.commit_single_result(calc_id, answer, errors);
                }
                NodeState::Done { answer: None, .. } => {
                    // Already committed by another thread via the Participant revisit path.
                }
                NodeState::HasPlaceholder(_) => {
                    panic!(
                        "batch_commit_scc: node {} is still HasPlaceholder at commit time. \
                         This means its calculate_and_record_answer never completed, \
                         which would leave its Calculation cell stuck in Calculating state.",
                        calc_id,
                    );
                }
                NodeState::Fresh | NodeState::InProgress => {
                    panic!(
                        "batch_commit_scc: node {} is {:?} at commit time",
                        calc_id, node_state,
                    );
                }
            }
        }
    }

    /// Commit all final converged answers from an iteratively-solved SCC
    /// to their respective Calculation cells.
    ///
    /// Called after the fixpoint iteration converges (or max iterations are
    /// reached). Iterates the SCC's iteration `node_states` and commits each
    /// `Done` entry's answer and errors via `commit_single_result`.
    fn commit_final_answers(&self, scc: Scc) {
        let iter_state = scc
            .iterative
            .expect("commit_final_answers: SCC has no iteration state");
        for (calc_id, node_state) in iter_state.node_states {
            match node_state {
                IterationNodeState::Done { answer, errors } => {
                    self.commit_single_result(calc_id, answer, errors);
                }
                IterationNodeState::Fresh | IterationNodeState::InProgress { .. } => {
                    // All members should be Done after driving all iteration
                    // members. If not, something went wrong in the iteration.
                    panic!(
                        "commit_final_answers: node {} is {:?} at commit time",
                        calc_id, node_state,
                    );
                }
            }
        }
    }

    /// Drive a single iteration member by calling `get_idx` for its typed index.
    ///
    /// The member is a `CalcId` containing `(Bindings, AnyIdx)`. For same-module
    /// members (where the member's module matches this solver's module), we
    /// dispatch through `dispatch_anyidx!` to call `get_idx` with the concrete
    /// key type. Cross-module members are driven via `solve_idx_erased`, which
    /// constructs a temporary `AnswersSolver` in the target module's context
    /// using the shared `ThreadState` (and therefore the shared `CalcStack`).
    fn drive_member(&self, calc_id: &CalcId) {
        let CalcId(ref bindings, ref any_idx) = *calc_id;
        if bindings.module().name() != self.bindings().module().name()
            || bindings.module().path() != self.bindings().module().path()
        {
            // Cross-module member: drive via LookupAnswer::solve_idx_erased,
            // which routes to the target module's Answers and constructs a
            // temporary AnswersSolver there with the shared ThreadState.
            assert!(
                self.answers.solve_idx_erased(calc_id, self.thread_state),
                "drive_member: cross-module driving failed for {}. \
                 The target module's Answers may not be loaded.",
                calc_id,
            );
            return;
        }
        dispatch_anyidx!(any_idx, self, drive_member_typed);
    }

    /// Type-specialized helper for `drive_member`. Calls `get_idx` for the
    /// concrete key type, discarding the result (the answer is stored in
    /// iteration state by `calculate_and_record_answer_iterative`).
    fn drive_member_typed<K: Solve<Ans>>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let _ = self.get_idx(idx);
    }

    /// Type-specialized helper for `Answers::solve_idx_erased`. Calls `get_idx`
    /// for the concrete key type, discarding the result. Used for cross-module
    /// iterative driving where the answer is stored in iteration state on the
    /// shared `CalcStack`.
    pub(crate) fn solve_idx_erased_typed<K: Solve<Ans>>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let _ = self.get_idx(idx);
    }

    /// Drive all fresh iteration members in the top SCC until none remain.
    ///
    /// Because every back-edge breaks immediately in iterative mode, a single
    /// DFS from one member may not reach all members. This method loops until
    /// `next_fresh_member` returns `None`, ensuring every member is driven.
    fn drive_all_iteration_members(&self) {
        while let Some(id) = self.stack().next_fresh_member() {
            self.drive_member(&id);
        }
    }

    /// Iterative fixpoint driver for a completed SCC.
    ///
    /// Implements the two-loop algorithm:
    /// - Outer loop (demotion): if SCC membership expands during iteration
    ///   (detected by `demoted` flag), restart at iteration 1.
    /// - Inner loop (fixpoint): iterate until answers converge (no change
    ///   between iterations) or `MAX_ITERATIONS` is exceeded.
    ///
    /// Between iterations, the SCC is popped from the stack, its iteration
    /// state is updated (previous answers extracted, fresh state set), and
    /// it is pushed back for the next iteration (pop-mutate-push pattern).
    ///
    /// Absorption detection: if the top SCC's `detected_at` changes during
    /// iteration (because this SCC was merged into an ancestor), the driver
    /// returns without committing. The merged SCC is now part of the
    /// ancestor's iteration.
    #[allow(clippy::mutable_key_type)]
    fn iterative_resolve_scc(&self, mut scc: Scc) {
        let scc_identity = scc.detected_at.dupe();
        let mut iteration: u32 = 1;
        let mut demotions: u32 = 0;

        loop {
            if iteration > MAX_ITERATIONS {
                tracing::warn!(
                    "iterative_resolve_scc: SCC {} exceeded {} iterations; \
                     committing last answers",
                    scc_identity,
                    MAX_ITERATIONS,
                );
                break;
            }

            // Extract previous answers from the prior iteration (if any).
            let previous_answers = if iteration > 1 {
                scc.extract_done_answers()
            } else {
                BTreeMap::new()
            };

            // Set up fresh iteration state for this iteration.
            scc.set_fresh_iteration_state(iteration, previous_answers);

            // Push the SCC back onto the stack for this iteration.
            self.stack().push_scc(scc);

            // Drive all fresh members until none remain.
            self.drive_all_iteration_members();

            // Absorption detection: if the top SCC's detected_at no longer
            // matches our identity, this SCC was merged into an ancestor
            // during iteration. Return without committing; the ancestor's
            // iteration driver will handle it.
            if self.stack().top_scc_detected_at() != scc_identity {
                return;
            }

            // Pop the SCC to inspect its iteration outcome.
            scc = self.stack().pop_scc();

            // Check for demotion: if SCC membership expanded, restart at
            // iteration 1 with fresh state.
            let iter_state = scc
                .iterative
                .as_ref()
                .expect("iterative_resolve_scc: SCC lost iteration state after pop");
            let demoted = iter_state.demoted;
            let has_changed = iter_state.has_changed;

            if demoted {
                demotions += 1;
                check_demotion_limit(demotions, &scc_identity);
                iteration = 1;
                continue;
            }

            // Convergence check: if this is iteration >= 2 and no answers
            // changed, the fixpoint has converged.
            if iteration >= 2 && !has_changed {
                break;
            }

            iteration += 1;
        }

        self.commit_final_answers(scc);
    }

    /// Finalize a recursive answer. This takes the raw value produced by `K::solve` and calls
    /// `K::record_recursive` in order to:
    /// - ensure that the `Variables` map in `solver.rs` is updated
    /// - possibly simplify the result; in particular a recursive solution that comes out to be
    ///   a union that includes the recursive solution is simplified, which is important for
    ///   some kinds of cycles, particularly those coming from LoopPhi
    /// - force the recursive var if necessary; we skip Var::ZERO (which is an unforcable
    ///   placeholder used by some kinds of bindings that aren't Types) in this step.
    fn finalize_recursive_answer<K: Solve<Ans>>(
        &self,
        idx: Idx<K>,
        var: Var,
        answer: Arc<K::Answer>,
        errors: &ErrorCollector,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let range = K::range_with(idx, self.bindings());
        let final_answer = K::record_recursive(self, range, answer, var, errors);
        if var != Var::ZERO {
            self.solver().force_var(var);
        }
        final_answer
    }

    /// Attempt to record a cycle placeholder result to unwind a cycle from here.
    ///
    /// Returns a `Result` where `Err(var)` is the normal case (placeholder created,
    /// cycle should be unwound), and `Ok(value)` means another thread has already
    /// committed a final answer so we can skip the cycle-breaking entirely.
    ///
    /// Note: The placeholder is recorded in SCC-local state (NodeState::HasPlaceholder),
    /// not in the Calculation cell. Each thread that hits the same cycle creates its
    /// own placeholder. The final answer IS written thread-locally via NodeState::Done
    /// and only committed to Calculation during batch commit when the SCC completes.
    fn attempt_to_unwind_cycle_from_here<K: Solve<Ans>>(
        &self,
        current: &CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>>,
    ) -> Result<Arc<K::Answer>, Var>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        // Check if another thread already committed a final answer.
        if let Some(v) = calculation.get() {
            return Ok(v);
        }
        // Create a recursive placeholder and store it only in SCC-local state.
        let binding = self.bindings().get(idx);
        let rec = K::create_recursive(self, binding);
        self.stack().on_placeholder_recorded(current, rec);
        Err(rec)
    }

    /// Handle depth overflow based on the configured handler.
    fn handle_depth_overflow<K: Solve<Ans>>(
        &self,
        current: &CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>>,
        config: RecursionLimitConfig,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        match config.handler {
            RecursionOverflowHandler::BreakWithPlaceholder => self
                .handle_depth_overflow_break_with_placeholder(
                    current,
                    idx,
                    calculation,
                    config.limit,
                ),
            RecursionOverflowHandler::PanicWithDebugInfo => {
                self.handle_depth_overflow_panic_with_debug_info(idx, config.limit)
            }
        }
    }

    /// BreakWithPlaceholder handler: emit an internal error and return a recursive placeholder.
    fn handle_depth_overflow_break_with_placeholder<K: Solve<Ans>>(
        &self,
        current: &CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>>,
        limit: u32,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let range = K::range_with(idx, self.bindings());
        self.base_errors.add(
            range,
            ErrorInfo::Kind(ErrorKind::InternalError),
            vec1![format!(
                "Recursion depth limit ({}) exceeded; possible stack overflow prevented",
                limit
            )],
        );
        // Return recursive placeholder (same pattern as cycle handling)
        self.attempt_to_unwind_cycle_from_here(current, idx, calculation)
            .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r)))
    }

    /// PanicWithDebugInfo handler: dump debug info to stderr and panic.
    fn handle_depth_overflow_panic_with_debug_info<K: Solve<Ans>>(
        &self,
        idx: Idx<K>,
        limit: u32,
    ) -> !
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        eprintln!("=== RECURSION DEPTH OVERFLOW DEBUG ===");
        eprintln!("Depth limit: {}", limit);
        eprintln!("Current depth: {}", self.stack().len());

        eprintln!("\n--- CalcStack ---");
        let stack_vec = self.stack().into_vec();
        for (i, calc_id) in stack_vec.iter().rev().enumerate() {
            eprintln!("  [{}] {}", i, calc_id);
        }

        eprintln!("\n--- Scc Stack ---");
        if self.stack().sccs_is_empty() {
            eprintln!("  None");
        } else {
            for scc in self.stack().borrow_scc_stack().iter().rev() {
                eprintln!("  {}", scc);
            }
        }

        eprintln!("\n--- Triggering Idx Details ---");
        let key = self.bindings().idx_to_key(idx);
        let range = K::range_with(idx, self.bindings());
        let display_range = self.bindings().module().display_range(range);
        eprintln!("  Module: {}", self.module().name());
        eprintln!("  Range: {}", display_range);
        eprintln!("  Key: {}", key.display_with(self.bindings().module()));

        panic!("Recursion depth limit exceeded - stack overflow prevented");
    }

    fn get_from_module<K: Solve<Ans> + Exported>(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &K,
    ) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if module == self.module().name() && path == Some(self.module().path()) {
            // We are working in our own module, so don't have to go back to the `LookupAnswer` trait.
            // But even though we are looking at our own module, we might be using our own type via an import
            // from a mutually recursive module, so have to deal with key_to_idx finding nothing due to incremental.
            Some(self.get_idx(self.bindings().key_to_idx_hashed_opt(Hashed::new(k))?))
        } else {
            self.answers.get(module, path, k, self.thread_state)
        }
    }

    pub fn get_from_export(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &KeyExport,
    ) -> Arc<Type> {
        self.get_from_module(module, path, k).unwrap_or_else(|| {
            panic!("We should have checked Exports before calling this, {module} {k:?}")
        })
    }

    /// Might return None if the class is no longer present on the underlying module.
    pub fn get_from_class<K: Solve<Ans> + Exported>(
        &self,
        cls: &Class,
        k: &K,
    ) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_from_module(cls.module_name(), Some(cls.module_path()), k)
    }

    pub fn get_type_alias(&self, data: &TypeAliasData) -> Arc<TypeAlias> {
        match data {
            TypeAliasData::Ref(r) => {
                let ta = self.get_from_module(
                    r.module_name,
                    Some(&r.module_path),
                    &KeyTypeAlias(r.index),
                );
                let Some(ta) = ta else {
                    return Arc::new(TypeAlias::unknown(r.name.clone()));
                };
                if let Some(args) = &r.args {
                    let mut ta = (*ta).clone();
                    args.substitute_into_mut(ta.as_type_mut());
                    Arc::new(ta)
                } else {
                    ta
                }
            }
            TypeAliasData::Value(ta) => Arc::new(ta.clone()),
        }
    }

    pub fn get<K: Solve<Ans>>(&self, k: &K) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.get_hashed(Hashed::new(k))
    }

    pub fn get_hashed<K: Solve<Ans>>(&self, k: Hashed<&K>) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.get_idx(self.bindings().key_to_idx_hashed(k))
    }

    pub fn get_hashed_opt<K: Solve<Ans>>(&self, k: Hashed<&K>) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        Some(self.get_idx(self.bindings().key_to_idx_hashed_opt(k)?))
    }

    pub fn create_recursive(&self, binding: &Binding) -> Var {
        match binding {
            Binding::LoopPhi(prior_idx, _) => self.solver().fresh_loop_recursive(
                self.uniques,
                self.get_idx(*prior_idx)
                    .arc_clone_ty()
                    .promote_implicit_literals(self.stdlib),
            ),
            _ => self.solver().fresh_recursive(self.uniques),
        }
    }

    pub fn recurse(&'a self, var: Var) -> Option<Guard<'a, Var>> {
        self.solver().recurse(var, self.recurser)
    }

    pub fn record_recursive(
        &self,
        loc: TextRange,
        ty: Type,
        recursive: Var,
        errors: &ErrorCollector,
    ) -> Type {
        self.solver()
            .record_recursive::<Ans>(recursive, ty, self.type_order(), errors, loc)
    }

    /// Check if `got` matches `want`, returning `want` if the check fails.
    pub fn check_and_return_type_info(
        &self,
        got: TypeInfo,
        want: &Type,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> TypeInfo {
        if self.check_type(got.ty(), want, loc, errors, tcc) {
            got
        } else {
            got.with_ty(want.clone())
        }
    }

    /// Check if `got` matches `want`, returning `want` if the check fails.
    pub fn check_and_return_type(
        &self,
        got: Type,
        want: &Type,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> Type {
        if self.check_type(&got, want, loc, errors, tcc) {
            got
        } else {
            want.clone()
        }
    }

    /// Check if `got` matches `want`, returning `true` on success and `false` on failure.
    pub fn check_type(
        &self,
        got: &Type,
        want: &Type,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> bool {
        if got.is_error() {
            true
        } else {
            match self.is_subset_eq_with_reason(got, want) {
                Ok(()) => true,
                Err(error) => {
                    self.solver().error(got, want, errors, loc, tcc, error);
                    false
                }
            }
        }
    }

    pub fn distribute_over_union(&self, ty: &Type, mut f: impl FnMut(&Type) -> Type) -> Type {
        let mut res = Vec::new();
        self.map_over_union(ty, |ty| {
            res.push(f(ty));
        });
        self.unions(res)
    }

    pub fn map_over_union(&self, ty: &Type, f: impl FnMut(&Type)) {
        struct Data<'a, 'b, Ans: LookupAnswer, F: FnMut(&Type)> {
            /// The `self` of `AnswersSolver`
            me: &'b AnswersSolver<'a, Ans>,
            /// The function to apply on each call
            f: F,
            /// Arguments we have already used for the function.
            /// If we see the same element twice in a union (perhaps due to nested Var expansion),
            /// we only need to process it once. Avoids O(n^2) for certain flow patterns.
            done: SmallSet<Type>,
            /// Have we seen a union node? If not, we can skip the cache
            /// as there will only be exactly one call to `f` (the common case).
            seen_union: bool,
        }

        impl<Ans: LookupAnswer, F: FnMut(&Type)> Data<'_, '_, Ans, F> {
            fn go(&mut self, ty: &Type, in_type: bool) {
                match ty {
                    Type::Never(_) if !in_type => (),
                    Type::Union(box Union { members, .. }) => {
                        self.seen_union = true;
                        members.iter().for_each(|ty| self.go(ty, in_type))
                    }
                    Type::Type(box Type::Union(box Union { members, .. })) if !in_type => {
                        members.iter().for_each(|ty| self.go(ty, true))
                    }
                    Type::Var(v) if let Some(_guard) = self.me.recurse(*v) => {
                        self.go(&self.me.solver().force_var(*v), in_type)
                    }
                    _ if in_type => (self.f)(&self.me.heap.mk_type(ty.clone())),
                    _ => {
                        // If we haven't encountered a union this must be the only type, no need to cache it.
                        // Otherwise, if inserting succeeds (we haven't processed this type before) actually do it.
                        if !self.seen_union || self.done.insert(ty.clone()) {
                            (self.f)(ty)
                        }
                    }
                }
            }
        }
        Data {
            me: self,
            f,
            done: SmallSet::new(),
            seen_union: false,
        }
        .go(ty, false)
    }

    pub fn unions(&self, xs: Vec<Type>) -> Type {
        self.solver().unions(xs, self.type_order())
    }

    pub fn union(&self, x: Type, y: Type) -> Type {
        self.unions(vec![x, y])
    }

    pub fn error(
        &self,
        errors: &ErrorCollector,
        range: TextRange,
        info: ErrorInfo,
        msg: String,
    ) -> Type {
        errors.add(range, info, vec1![msg]);
        self.heap.mk_any_error()
    }

    /// Create a new error collector. Useful when a caller wants to decide whether or not to report
    /// errors from an operation.
    pub fn error_collector(&self) -> ErrorCollector {
        ErrorCollector::new(self.module().dupe(), ErrorStyle::Delayed)
    }

    /// Create an error collector that simply swallows errors. Useful when a caller wants to try an
    /// operation that may error but never report errors from it.
    pub fn error_swallower(&self) -> ErrorCollector {
        ErrorCollector::new(self.module().dupe(), ErrorStyle::Never)
    }

    /// Add an implicit-any error for a generic entity without explicit type arguments.
    pub fn add_implicit_any_error(
        errors: &ErrorCollector,
        range: TextRange,
        generic_entity: String,
        tparam_name: Option<&str>,
    ) {
        let msg = if let Some(tparam) = tparam_name {
            format!(
                "Cannot determine the type parameter `{}` for generic {}",
                tparam, generic_entity,
            )
        } else {
            format!(
                "Cannot determine the type parameter for generic {}",
                generic_entity
            )
        };
        errors.add(
            range,
            ErrorInfo::Kind(ErrorKind::ImplicitAny),
            vec1![
                msg,
                "Either specify the type argument explicitly, or specify a default for the type variable.".to_owned(),
            ],
        );
    }

    /// Compare two type-erased answers for equality, dispatching through
    /// the concrete answer type based on the `AnyIdx` variant.
    ///
    /// Used for convergence detection in the iterative fixpoint solver:
    /// if answers haven't changed between iterations, the SCC has converged.
    /// Assumes both answers have already been deep-forced (no unresolved Vars).
    fn answers_equal(
        &self,
        idx: &AnyIdx,
        old: &Arc<dyn Any + Send + Sync>,
        new: &Arc<dyn Any + Send + Sync>,
    ) -> bool {
        dispatch_anyidx!(idx, self, answers_equal_typed, old, new)
    }

    /// Type-specialized answer comparison. Downcasts both type-erased answers
    /// to `Arc<K::Answer>` and compares using `TypeEq`, which correctly handles
    /// identity-based equality for `Unique`, `TypeVar`, etc.
    fn answers_equal_typed<K: Solve<Ans>>(
        &self,
        _idx: Idx<K>,
        old: &Arc<dyn Any + Send + Sync>,
        new: &Arc<dyn Any + Send + Sync>,
    ) -> bool {
        let old_typed = old
            .downcast_ref::<Arc<K::Answer>>()
            .expect("answers_equal_typed: type mismatch on old answer");
        let new_typed = new
            .downcast_ref::<Arc<K::Answer>>()
            .expect("answers_equal_typed: type mismatch on new answer");
        let mut ctx = TypeEqCtx::default();
        old_typed.type_eq(new_typed, &mut ctx)
    }
}

#[cfg(test)]
mod scc_tests {
    use super::*;

    /// Create a dummy `NodeState::Done` for testing.
    fn done_for_test() -> NodeState {
        NodeState::Done {
            answer: None,
            errors: None,
        }
    }

    /// Helper to create a test Scc with given parameters.
    ///
    /// This bypasses the normal Scc::new constructor to allow direct construction
    /// for testing merge logic.
    ///
    /// Note: segment_size is set to node_state.len() which approximates the number
    /// of live frames. In production, segment_size may differ from participant count
    /// due to duplicate CalcIds during cycle breaking.
    #[allow(clippy::mutable_key_type)]
    fn make_test_scc(
        break_at: Vec<CalcId>,
        node_state: BTreeMap<CalcId, NodeState>,
        detected_at: CalcId,
        anchor_pos: usize,
    ) -> Scc {
        let segment_size = node_state.len();
        Scc {
            break_at: break_at.into_iter().collect(),
            node_state,
            detected_at,
            anchor_pos,
            segment_size,
            iterative: None,
        }
    }

    /// Helper to create a CalcStack for testing.
    fn make_calc_stack(entries: &[CalcId]) -> CalcStack {
        make_calc_stack_with_mode(entries, SccSolvingMode::default())
    }

    /// Helper to create a CalcStack for testing in iterative mode.
    #[allow(dead_code)]
    fn make_iterative_calc_stack(entries: &[CalcId]) -> CalcStack {
        make_calc_stack_with_mode(entries, SccSolvingMode::Iterative)
    }

    /// Helper to create a CalcStack for testing with a specific solving mode.
    fn make_calc_stack_with_mode(entries: &[CalcId], mode: SccSolvingMode) -> CalcStack {
        let stack = CalcStack::new(mode);
        for entry in entries {
            stack.push_for_test(entry.dupe());
        }
        stack
    }

    /// Helper to create node_state map with all nodes Fresh.
    #[allow(clippy::mutable_key_type)]
    fn fresh_nodes(ids: &[CalcId]) -> BTreeMap<CalcId, NodeState> {
        ids.iter().map(|id| (id.dupe(), NodeState::Fresh)).collect()
    }

    #[test]
    fn test_current_cycle_no_cycle() {
        // Stack with unique entries: no cycle
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe()]);
        assert!(calc_stack.current_cycle().is_none());
    }

    #[test]
    fn test_current_cycle_simple_cycle() {
        // Stack [A, B, C, A] - A appears twice, creating a cycle
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), a.dupe()]);
        let cycle = calc_stack.current_cycle().expect("Should detect cycle");

        // Cycle should be in recency order: [A(newest), C, B]
        // (excludes the duplicate A at position 0)
        assert_eq!(cycle.len(), 3);
        assert_eq!(cycle[0], a); // Newest A
        assert_eq!(cycle[1], c);
        assert_eq!(cycle[2], b);
    }

    #[test]
    fn test_current_cycle_longer_cycle() {
        // Stack [A, B, C, D, E, A] - cycle from position 1 to 5
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);

        let calc_stack =
            make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe(), a.dupe()]);
        let cycle = calc_stack.current_cycle().expect("Should detect cycle");

        // Cycle should be [A(newest), E, D, C, B] in recency order
        assert_eq!(cycle.len(), 5);
        assert_eq!(cycle[0], a); // Newest A
        assert_eq!(cycle[1], e);
        assert_eq!(cycle[2], d);
        assert_eq!(cycle[3], c);
        assert_eq!(cycle[4], b);
    }

    #[test]
    fn test_current_cycle_empty_stack() {
        let calc_stack = CalcStack::new(SccSolvingMode::default());
        assert!(calc_stack.current_cycle().is_none());
    }

    #[test]
    fn test_initial_cycle_detection() {
        // Setup: CalcStack = [M0, M1, M2], detect a cycle [M2, M1, M0]
        // Expected: New SCC with participants {M0, M1, M2}, break_at = M0 (minimal)
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe()]);

        // Simulate detecting cycle - raw cycle order is from detection point to back-edge target
        let raw_cycle = vec1![c.dupe(), b.dupe(), a.dupe()];
        let result = calc_stack.on_scc_detected(raw_cycle);

        // Should not break immediately since break_at is A (minimal) but detected_at is C
        assert!(matches!(result, SccDetectedResult::Continue));

        // Verify SCC was created
        let stack = calc_stack.borrow_scc_stack();
        assert_eq!(stack.len(), 1);

        let scc = &stack[0];
        assert!(scc.break_at.contains(&a));
        assert_eq!(scc.node_state.len(), 3);
        assert!(scc.node_state.contains_key(&a));
        assert!(scc.node_state.contains_key(&b));
        assert!(scc.node_state.contains_key(&c));
    }

    #[test]
    fn test_subcycle_within_active_cycle() {
        // Setup: CalcStack = [M0, M1, M2, M3], existing SCC with {M0, M1, M2, M3}
        // New cycle detected: [M3, M2, M1] (sub-cycle within the existing SCC)
        // Expected: Merged into same SCC
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);

        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe()]);

        // Create initial SCC with A, B, C, D
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe(), a.dupe()];
        calc_stack.on_scc_detected(initial_cycle);

        // Now detect sub-cycle D -> B
        let sub_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        calc_stack.on_scc_detected(sub_cycle);

        // The sub-cycle overlaps with existing SCC, so they merge
        let stack = calc_stack.borrow_scc_stack();
        assert_eq!(
            stack.len(),
            1,
            "Should still have exactly one SCC after merging"
        );

        // All nodes should be in the merged SCC
        let scc = &stack[0];
        assert!(scc.node_state.contains_key(&a));
        assert!(scc.node_state.contains_key(&b));
        assert!(scc.node_state.contains_key(&c));
        assert!(scc.node_state.contains_key(&d));
    }

    #[test]
    fn test_back_edge_into_existing_cycle() {
        // CalcStack: [M0, M1, M2, M3, M4, M5]
        // Existing SCC: {M1, M2, M3}
        // New cycle: [M5, M4, M3, M2] (back-edge from M5 to M2)
        // Expected: Merge creates SCC with {M1, M2, M3, M4, M5}
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);
        let f = CalcId::for_test("m", 5);

        let calc_stack =
            make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe(), f.dupe()]);

        // Create initial SCC with B, C, D (detected from D going back to B)
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        calc_stack.on_scc_detected(initial_cycle);

        // Verify initial state
        {
            let stack = calc_stack.borrow_scc_stack();
            assert_eq!(stack.len(), 1);
            assert_eq!(stack[0].node_state.len(), 3);
        }

        // Now detect cycle [F, E, D, C] - overlaps with existing at C and D
        let new_cycle = vec1![f.dupe(), e.dupe(), d.dupe(), c.dupe()];
        calc_stack.on_scc_detected(new_cycle);

        // Should merge because new cycle overlaps with existing SCC
        let stack = calc_stack.borrow_scc_stack();
        assert_eq!(stack.len(), 1, "Should have merged into one SCC");

        let scc = &stack[0];
        // B, C, D, E, F should all be in the merged SCC
        assert!(scc.node_state.contains_key(&b));
        assert!(scc.node_state.contains_key(&c));
        assert!(scc.node_state.contains_key(&d));
        assert!(scc.node_state.contains_key(&e));
        assert!(scc.node_state.contains_key(&f));
    }

    #[test]
    fn test_back_edge_before_existing_cycle() {
        // CalcStack: [M0, M1, M2, M3, M4, M5]
        // Existing SCC: {M1, M2, M3}
        // New cycle: [M5, M4, M3, M2, M1, M0] (back-edge from M5 to M0)
        // Expected: Merge creates SCC with {M0, M1, M2, M3, M4, M5}
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);
        let f = CalcId::for_test("m", 5);

        let calc_stack =
            make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe(), f.dupe()]);

        // Create initial SCC with B, C, D
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        calc_stack.on_scc_detected(initial_cycle);

        // Now detect cycle [F, E, D, C, B, A] - includes everything from A to F
        let new_cycle = vec1![f.dupe(), e.dupe(), d.dupe(), c.dupe(), b.dupe(), a.dupe()];
        calc_stack.on_scc_detected(new_cycle);

        // Should merge because new cycle contains the existing SCC
        let stack = calc_stack.borrow_scc_stack();
        assert_eq!(stack.len(), 1, "Should have merged into one SCC");

        let scc = &stack[0];
        // All nodes should be in the merged SCC
        assert!(scc.node_state.contains_key(&a));
        assert!(scc.node_state.contains_key(&b));
        assert!(scc.node_state.contains_key(&c));
        assert!(scc.node_state.contains_key(&d));
        assert!(scc.node_state.contains_key(&e));
        assert!(scc.node_state.contains_key(&f));
    }

    #[test]
    fn test_merge_many_preserves_break_points() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);

        // Create two SCCs with different break points
        let scc1 = make_test_scc(
            vec![a.dupe()],
            fresh_nodes(&[a.dupe(), b.dupe()]),
            a.dupe(),
            0, // anchor_pos
        );
        let scc2 = make_test_scc(
            vec![c.dupe()],
            fresh_nodes(&[c.dupe(), d.dupe()]),
            c.dupe(),
            2, // anchor_pos
        );

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Both break points should be preserved
        assert!(merged.break_at.contains(&a));
        assert!(merged.break_at.contains(&c));
        assert_eq!(merged.break_at.len(), 2);

        // All nodes should be present
        assert_eq!(merged.node_state.len(), 4);

        // anchor_pos should be the minimum (0)
        assert_eq!(merged.anchor_pos, 0);
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn test_merge_many_takes_most_advanced_state() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);

        // SCC1 has M0 as Done, M1 as Fresh
        let mut scc1_state = BTreeMap::new();
        scc1_state.insert(a.dupe(), done_for_test());
        scc1_state.insert(b.dupe(), NodeState::Fresh);
        let scc1 = make_test_scc(vec![a.dupe()], scc1_state, a.dupe(), 0);

        // SCC2 has M0 as Fresh, M1 as InProgress
        let mut scc2_state = BTreeMap::new();
        scc2_state.insert(a.dupe(), NodeState::Fresh);
        scc2_state.insert(b.dupe(), NodeState::InProgress);
        let scc2 = make_test_scc(vec![a.dupe()], scc2_state, a.dupe(), 0);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should take the most advanced state for each node
        assert!(matches!(
            merged.node_state.get(&a),
            Some(NodeState::Done { .. })
        ));
        assert!(matches!(
            merged.node_state.get(&b),
            Some(NodeState::InProgress)
        ));
    }

    #[test]
    fn test_merge_many_keeps_smallest_detected_at() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        // SCC1 detected at M1
        let scc1 = make_test_scc(
            vec![a.dupe()],
            fresh_nodes(&[a.dupe(), b.dupe()]),
            b.dupe(),
            0,
        );
        // SCC2 detected at M2
        let scc2 = make_test_scc(
            vec![a.dupe()],
            fresh_nodes(&[a.dupe(), c.dupe()]),
            c.dupe(),
            0,
        );
        // When merging with M0 as the new detected_at, should keep M0 (smallest)
        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());
        assert_eq!(merged.detected_at, a);
    }

    #[test]
    fn test_merge_many_keeps_minimum_anchor_pos() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        // SCC1 with anchor_pos = 5
        let scc1 = make_test_scc(
            vec![a.dupe()],
            fresh_nodes(&[a.dupe(), b.dupe()]),
            a.dupe(),
            5,
        );
        // SCC2 with anchor_pos = 2
        let scc2 = make_test_scc(vec![c.dupe()], fresh_nodes(&[c.dupe()]), c.dupe(), 2);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should keep the minimum anchor_pos
        assert_eq!(merged.anchor_pos, 2);
    }

    #[test]
    fn test_stale_calculation_panic() {
        // Reproduces the panic where Calculation has stale state but CalcStack is fresh.
        let calc_id = CalcId::for_test("m", 0);
        let calculation: Calculation<usize> = Calculation::new();

        // 1. Simulate stale state: propose calculation on this thread.
        // This sets the thread bit in calculation.
        match calculation.propose_calculation() {
            ProposalResult::Calculatable => {}
            _ => panic!("Expected Calculatable"),
        }

        // 2. Create a fresh stack (simulating a new request/thread reuse).
        let stack = CalcStack::new(SccSolvingMode::default());

        // 3. Push the same calculation.
        // This should NOT panic.
        let action = stack.push(calc_id, &calculation);

        // 4. Expect Calculate action (to recover).
        match action {
            BindingAction::Calculate => {}
            _ => panic!("Expected Calculate action to recover from stale state"),
        }
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn test_membership_back_edge_merge_and_demotion() {
        // Verify that pushing a CalcId which is a member of a non-top iterating
        // SCC causes the SCCs to merge and the result to have demoted = true.
        //
        // Setup:
        //   CalcStack = [A, B, C, D, E]
        //   SCC0 (non-top): members {A, B}, iterating at iteration 2
        //   SCC1 (top):     members {D, E}, iterating at iteration 1
        //   C is between the two SCCs but not a member of either.
        //
        // Action: push(A, ...) -- A is a member of SCC0, the non-top SCC.
        //
        // Expected:
        //   - SCCs merge into one (stack length goes from 2 to 1)
        //   - Merged SCC has iterative.demoted == true
        //   - Merged SCC contains members from both original SCCs {A, B, D, E}
        //   - push returns Calculate (since after demotion all nodes are Fresh)
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);

        // Build the iterative CalcStack with [A, B, C, D, E].
        let calc_stack =
            make_iterative_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe()]);

        // Manually construct SCC0 with iterative state (iteration 2).
        let scc0 = {
            let mut node_state = BTreeMap::new();
            node_state.insert(a.dupe(), NodeState::Fresh);
            node_state.insert(b.dupe(), NodeState::Fresh);
            let iter_nodes: BTreeMap<CalcId, IterationNodeState> = [
                (a.dupe(), IterationNodeState::Fresh),
                (b.dupe(), IterationNodeState::Fresh),
            ]
            .into_iter()
            .collect();
            Scc {
                break_at: [a.dupe()].into_iter().collect(),
                node_state,
                detected_at: a.dupe(),
                anchor_pos: 0,
                segment_size: 2,
                iterative: Some(SccIterationState {
                    iteration: 2,
                    node_states: iter_nodes,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                }),
            }
        };

        // Manually construct SCC1 with iterative state (iteration 1).
        let scc1 = {
            let mut node_state = BTreeMap::new();
            node_state.insert(d.dupe(), NodeState::Fresh);
            node_state.insert(e.dupe(), NodeState::Fresh);
            let iter_nodes: BTreeMap<CalcId, IterationNodeState> = [
                (d.dupe(), IterationNodeState::Fresh),
                (e.dupe(), IterationNodeState::Fresh),
            ]
            .into_iter()
            .collect();
            Scc {
                break_at: [d.dupe()].into_iter().collect(),
                node_state,
                detected_at: d.dupe(),
                anchor_pos: 3,
                segment_size: 2,
                iterative: Some(SccIterationState {
                    iteration: 1,
                    node_states: iter_nodes,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                }),
            }
        };

        // Push both SCCs onto the scc_stack: SCC0 at bottom, SCC1 on top.
        {
            let mut scc_stack = calc_stack.scc_stack.borrow_mut();
            scc_stack.push(scc0);
            scc_stack.push(scc1);
        }

        // Verify initial state: two SCCs.
        assert_eq!(calc_stack.borrow_scc_stack().len(), 2);

        // Push A: A is a member of SCC0 (the non-top iterating SCC).
        // This should trigger a membership back-edge merge.
        let calculation: Calculation<usize> = Calculation::new();
        let action = calc_stack.push(a.dupe(), &calculation);

        // After merge, there should be exactly one SCC.
        let scc_stack = calc_stack.borrow_scc_stack();
        assert_eq!(
            scc_stack.len(),
            1,
            "SCCs should have merged into one after membership back-edge"
        );

        let merged = &scc_stack[0];

        // The merged SCC must have demoted = true.
        let iter_state = merged
            .iterative
            .as_ref()
            .expect("merged SCC should have iterative state");
        assert!(
            iter_state.demoted,
            "merged SCC should have demoted = true after membership back-edge merge"
        );

        // Iteration should be reset to 1 after demotion.
        assert_eq!(
            iter_state.iteration, 1,
            "merged SCC iteration should be reset to 1 after demotion"
        );

        // All members from both original SCCs should be in the merged SCC's
        // legacy node_state.
        assert!(
            merged.node_state.contains_key(&a),
            "A should be in merged SCC"
        );
        assert!(
            merged.node_state.contains_key(&b),
            "B should be in merged SCC"
        );
        assert!(
            merged.node_state.contains_key(&d),
            "D should be in merged SCC"
        );
        assert!(
            merged.node_state.contains_key(&e),
            "E should be in merged SCC"
        );

        // All members should also be in the iteration node_states.
        assert!(
            iter_state.node_states.contains_key(&a),
            "A should be in iteration node_states"
        );
        assert!(
            iter_state.node_states.contains_key(&b),
            "B should be in iteration node_states"
        );
        assert!(
            iter_state.node_states.contains_key(&d),
            "D should be in iteration node_states"
        );
        assert!(
            iter_state.node_states.contains_key(&e),
            "E should be in iteration node_states"
        );

        // The push should return Calculate because after demotion all nodes
        // are Fresh, and A (the pushed target) transitions to Calculate.
        assert!(
            matches!(action, BindingAction::Calculate),
            "push should return Calculate for a Fresh member after merge"
        );
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn test_absorption_detection() {
        // Verify the absorption detection mechanism used by iterative_resolve_scc:
        // when an iterating inner SCC is merged into an ancestor SCC during
        // iteration, top_scc_detected_at() changes, allowing the driver to
        // detect that absorption occurred and return without committing.
        //
        // Setup:
        //   CalcStack = [A, B, C, D, E]
        //   SCC_outer (ancestor): members {A, B}, detected_at = A, iterating at iteration 2
        //   SCC_inner (top):      members {D, E}, detected_at = D, iterating at iteration 1
        //   C is between the two SCCs but not a member of either.
        //
        // The iterative_resolve_scc driver for SCC_inner would have saved
        // scc_identity = D (the inner SCC's detected_at) before pushing it
        // onto the stack and driving members.
        //
        // Action: push(A, ...) -- simulates a dependency on A discovered during
        //   driving of SCC_inner. A is a member of SCC_outer, triggering a
        //   membership back-edge merge that absorbs SCC_inner into SCC_outer.
        //
        // Expected:
        //   - After merge, only one SCC remains on the stack.
        //   - top_scc_detected_at() returns A (the ancestor's detected_at),
        //     NOT D (the inner SCC's original detected_at).
        //   - This mismatch (top_scc_detected_at() != scc_identity) is the
        //     absorption detection condition in iterative_resolve_scc.
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);

        // Build the iterative CalcStack with [A, B, C, D, E].
        let calc_stack =
            make_iterative_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe()]);

        // Manually construct SCC_outer (ancestor) with iterative state at iteration 2.
        let scc_outer = {
            let mut node_state = BTreeMap::new();
            node_state.insert(a.dupe(), NodeState::Fresh);
            node_state.insert(b.dupe(), NodeState::Fresh);
            let iter_nodes: BTreeMap<CalcId, IterationNodeState> = [
                (a.dupe(), IterationNodeState::Fresh),
                (b.dupe(), IterationNodeState::Fresh),
            ]
            .into_iter()
            .collect();
            Scc {
                break_at: [a.dupe()].into_iter().collect(),
                node_state,
                detected_at: a.dupe(),
                anchor_pos: 0,
                segment_size: 2,
                iterative: Some(SccIterationState {
                    iteration: 2,
                    node_states: iter_nodes,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                }),
            }
        };

        // Manually construct SCC_inner (top) with iterative state at iteration 1.
        let scc_inner = {
            let mut node_state = BTreeMap::new();
            node_state.insert(d.dupe(), NodeState::Fresh);
            node_state.insert(e.dupe(), NodeState::Fresh);
            let iter_nodes: BTreeMap<CalcId, IterationNodeState> = [
                (d.dupe(), IterationNodeState::Fresh),
                (e.dupe(), IterationNodeState::Fresh),
            ]
            .into_iter()
            .collect();
            Scc {
                break_at: [d.dupe()].into_iter().collect(),
                node_state,
                detected_at: d.dupe(),
                anchor_pos: 3,
                segment_size: 2,
                iterative: Some(SccIterationState {
                    iteration: 1,
                    node_states: iter_nodes,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                }),
            }
        };

        // Save the inner SCC's identity, as iterative_resolve_scc would.
        let scc_identity = scc_inner.detected_at.dupe();

        // Verify scc_identity is D (not A).
        assert_eq!(
            scc_identity, d,
            "scc_identity should be D (the inner SCC's detected_at)"
        );

        // Push both SCCs: SCC_outer at bottom, SCC_inner on top.
        {
            let mut scc_stack = calc_stack.scc_stack.borrow_mut();
            scc_stack.push(scc_outer);
            scc_stack.push(scc_inner);
        }

        // Verify initial state: two SCCs, top detected_at == D.
        assert_eq!(calc_stack.borrow_scc_stack().len(), 2);
        assert_eq!(
            calc_stack.top_scc_detected_at(),
            d,
            "before merge, top_scc_detected_at should be D"
        );

        // No absorption yet: the identity matches the top SCC's detected_at.
        assert_eq!(
            calc_stack.top_scc_detected_at(),
            scc_identity,
            "before merge, no absorption should be detected"
        );

        // Simulate what happens during driving: push(A) triggers a
        // membership back-edge merge because A is in SCC_outer.
        let calculation: Calculation<usize> = Calculation::new();
        let _action = calc_stack.push(a.dupe(), &calculation);

        // After merge, there should be exactly one SCC.
        assert_eq!(
            calc_stack.borrow_scc_stack().len(),
            1,
            "SCCs should have merged into one after membership back-edge"
        );

        // The absorption detection condition: top_scc_detected_at() != scc_identity.
        // After merging, the top SCC's detected_at should be A (the ancestor's),
        // which differs from D (the saved scc_identity).
        let top_detected_at = calc_stack.top_scc_detected_at();
        assert_eq!(
            top_detected_at, a,
            "after merge, top_scc_detected_at should be A (the ancestor's detected_at)"
        );
        assert_ne!(
            top_detected_at, scc_identity,
            "absorption detection: top_scc_detected_at should differ from the \
             inner SCC's saved identity, signaling that the inner SCC was absorbed"
        );

        // Verify the merged SCC has demoted = true (a side effect of merging
        // iterating SCCs, confirming the merge actually happened).
        let scc_stack = calc_stack.borrow_scc_stack();
        let merged = &scc_stack[0];
        let iter_state = merged
            .iterative
            .as_ref()
            .expect("merged SCC should have iterative state");
        assert!(iter_state.demoted, "merged SCC should have demoted = true");
    }

    #[test]
    fn test_demotion_limit_constants() {
        // Verify the safety-limit constants have the expected values.
        // These constants guard against infinite membership expansion in the
        // iterative SCC solver. Changing them without updating tests should
        // be a deliberate decision.
        assert_eq!(
            MAX_DEMOTIONS, 10,
            "MAX_DEMOTIONS should be 10; changing this limit affects \
             how many SCC membership expansions are tolerated before panic"
        );
        assert_eq!(
            MAX_ITERATIONS, 5,
            "MAX_ITERATIONS should be 5; changing this limit affects \
             how many fixpoint iterations are attempted before giving up"
        );
    }

    #[test]
    fn test_check_demotion_limit_allows_demotions_at_limit() {
        // Demotions at exactly MAX_DEMOTIONS should NOT panic.
        // The check is `demotions > MAX_DEMOTIONS`, so 10 is the last
        // allowed value.
        let id = CalcId::for_test("m", 0);
        check_demotion_limit(MAX_DEMOTIONS, &id); // should not panic
    }

    #[test]
    fn test_check_demotion_limit_allows_demotions_below_limit() {
        // Any demotion count below the limit should be fine.
        let id = CalcId::for_test("m", 0);
        for count in 0..MAX_DEMOTIONS {
            check_demotion_limit(count, &id); // should not panic
        }
    }

    #[test]
    #[should_panic(expected = "exceeded 10 demotions")]
    fn test_check_demotion_limit_panics_above_limit() {
        // One demotion past the limit should trigger the panic with the
        // expected message substring.
        let id = CalcId::for_test("m", 0);
        check_demotion_limit(MAX_DEMOTIONS + 1, &id);
    }

    #[test]
    #[should_panic(expected = "likely infinite membership expansion")]
    fn test_check_demotion_limit_panic_message() {
        // Verify the panic message contains the diagnostic hint so that
        // developers investigating a crash can identify the root cause.
        let id = CalcId::for_test("m", 0);
        check_demotion_limit(MAX_DEMOTIONS + 1, &id);
    }
}
