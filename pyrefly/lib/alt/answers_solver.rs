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
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use dupe::Dupe;
use dupe::IterDupedExt;
use fxhash::FxHashMap;
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
use crate::alt::answers::OverloadedCallee;
use crate::alt::answers::SolutionsEntry;
use crate::alt::answers::SolutionsTable;
use crate::alt::answers::TraceSideEffects;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::Binding;
use crate::binding::binding::Exported;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTypeAlias;
use crate::binding::binding::LambdaParamId;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::config::base::RecursionLimitConfig;
use crate::config::base::RecursionOverflowHandler;
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
use crate::types::class::ClassFields;
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
    /// The SCC (if any) that completed during `on_calculation_finished` but
    /// hasn't been committed yet. Taken by `get_idx` after each frame completes.
    /// At most one SCC can complete per completion point.
    pending_completed_scc: RefCell<Option<Scc>>,
}

impl CalcStack {
    fn new() -> Self {
        Self {
            stack: RefCell::new(Vec::new()),
            scc_stack: RefCell::new(Vec::new()),
            position_of: RefCell::new(FxHashMap::default()),
            pending_completed_scc: RefCell::new(None),
        }
    }

    /// Pop the current frame and take the completed SCC (if any).
    ///
    /// These two operations are always paired: every `pop` must be followed by
    /// taking and committing the completed SCC.
    ///
    /// We pop before taking (not after) for two reasons:
    /// - Lifecycle correctness: committed answers should correspond to fully
    ///   unwound computations. Popping first ensures the stack no longer
    ///   contains the completing frame when results are written to Calculation.
    /// - `pop()` decrements `top_pos_exclusive` on the top SCC. If we took first,
    ///   the completed SCC would already be gone from `scc_stack`, and `pop()`
    ///   could incorrectly decrement a parent SCC's top_pos_exclusive instead.
    ///
    /// Note that the `+ 1` in `on_calculation_finished`'s completion check
    /// (`stack_len <= bottom_pos_inclusive + 1`) is unrelated to this ordering — it
    /// exists because completion is detected during calculation, while the
    /// frame is still on the stack, well before we reach this method.
    fn pop_and_take_completed_scc(&self) -> Option<Scc> {
        self.pop();
        self.pending_completed_scc.borrow_mut().take()
    }

    /// Push a CalcId onto the stack and determine the binding action.
    ///
    /// This is purely thread-local: it manages the CalcStack and SCC state
    /// without touching the cross-thread Calculation cell. Cycle detection
    /// uses the thread-local stack exclusively; propose_calculation() is
    /// called by the caller (get_idx) before push.
    fn push(&self, current: CalcId) -> BindingAction {
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

        // Membership back-edge detection: check if the target is a member of
        // a *non-top* SCC. If so, this is a cross-SCC back-edge that must merge
        // all SCCs from that index to the top and demote (restart at iteration 1).
        //
        // This check runs BEFORE the top-SCC membership check because cross-SCC
        // back-edges must be caught first.
        //
        // Borrow safety: `find_scc_containing` returns an owned
        // `Option<usize>`, so the shared borrow on `scc_stack` is released
        // before the exclusive borrow needed for merging.
        if let Some(scc_idx) = self.find_scc_containing(&current) {
            let is_non_top = {
                let scc_stack = self.scc_stack.borrow();
                scc_idx < scc_stack.len() - 1
            };
            if is_non_top {
                // Merge all SCCs from scc_idx to the top of the stack.
                // This produces a single merged SCC with merge_happened = true
                // (via Scc::merge's iteration state merge logic). Demotion
                // is deferred to drive_all_iteration_members.
                {
                    let calc_stack_vec = self.into_vec();
                    let mut scc_stack = self.scc_stack.borrow_mut();
                    let mut sccs_to_merge: Vec<Scc> = scc_stack.drain(scc_idx..).collect();
                    // Reverse so the top SCC (last in drain order) appears first,
                    // matching merge_sccs order: merge_many gives priority to the
                    // first element's iteration states.
                    sccs_to_merge.reverse();
                    let sccs_to_merge = Vec1::try_from_vec(sccs_to_merge)
                        .expect("membership back-edge: at least the found SCC must be present");
                    // detected_at is just an extra min-candidate; merge_many
                    // takes min across all SCCs regardless of which we pass here.
                    let detected_at = sccs_to_merge.first().detected_at.dupe();
                    let mut merged = Scc::merge_many(sccs_to_merge, detected_at);
                    // Recompute top_pos_exclusive after merge.
                    merged.top_pos_exclusive = calc_stack_vec.len();

                    // Add free-floating CalcStack nodes (between merged SCCs)
                    // to node_state, mirroring merge_sccs.
                    merged.absorb_calc_stack_members(&calc_stack_vec, merged.bottom_pos_inclusive);

                    scc_stack.push(merged);
                }
                // The target is now in the top SCC's iteration state.
                // Determine the appropriate action based on iteration state.
                // After merge, existing iteration states are preserved (Done/
                // InProgress stay as-is) and new members are Fresh. The target
                // will typically be Fresh or InProgress. Handle all cases.
                return self.binding_action_for_top_scc_member(&current);
            }
            // The target is in the top SCC's iteration state (not a cross-SCC
            // back-edge). If we've exited the SCC segment, merge from the top
            // SCC anchor so intervening nodes/SCC fragments are absorbed.
            return self.binding_action_for_top_scc_member(&current);
        }

        // Top-SCC membership check: if the target is already a member of the
        // top SCC's iteration state, determine the action from its node state.
        // This catches back-edges within the top SCC and any member that is
        // already tracked in the top SCC's iteration state.
        //
        // Borrow safety: `get_iteration_node_state` returns an owned
        // `SccNodeStateKind`, so the shared borrow on `scc_stack` is
        // released before any exclusive borrow for mutation.
        if self.get_iteration_node_state(&current).is_some() {
            // Top-SCC member handling is shared across all re-entry paths.
            return self.binding_action_for_top_scc_member(&current);
        }

        // At this point, the node is not a known member of any SCC. But it
        // may still *become* one. If this push itself closes a cycle,
        // `current_cycle` below triggers immediate SCC creation. Otherwise,
        // a dependency chain explored during `K::solve` can still cycle back
        // and create an SCC that includes this node before computation returns.
        // Because of that, this node's "not in any SCC" status is only final
        // after computation finishes (see `is_scc_participant` in
        // `calculate_and_record_answer`).
        //
        // Check whether this push itself completes a cycle (i.e., this CalcId
        // already appears lower on the stack). If so, create a new SCC.
        if let Some(current_cycle) = self.current_cycle() {
            self.on_scc_detected(current_cycle);
            BindingAction::NeedsColdPlaceholder
        } else {
            BindingAction::Calculate
        }
    }

    /// Pop a binding frame from the raw binding-level CalcId stack.
    /// - Update both the direct stack and the `position_of` reverse index.
    /// - Also check whether the popped frame was part of the top Scc in the
    ///   Scc stack; if so, decrement the top_pos_exclusive to account for the fact
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
                top_scc.top_pos_exclusive = top_scc.top_pos_exclusive.saturating_sub(1);
            }
        }
        popped
    }

    /// Check if a CalcId is an SCC participant (exists in the top SCC's node_state).
    ///
    /// This is used in `calculate_and_record_answer` to detect nodes that
    /// became SCC members *during* computation. At push time, a node may not
    /// be in any SCC yet (so `get_iteration_node_state` returns `None` and
    /// `push` returns `Calculate`). But during `K::solve`, a dependency chain
    /// can cycle back to this node, creating an SCC that includes it. After
    /// `K::solve` returns, this check catches that case so the answer is
    /// stored in SCC-local state rather than written directly to Calculation.
    fn is_scc_participant(&self, current: &CalcId) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .is_some_and(|top_scc| top_scc.node_state.contains_key(current))
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
    /// Uses O(1) position arithmetic: if the existing SCC's exclusive upper bound
    /// (top_pos_exclusive) is greater than the cycle start position,
    /// the segments overlap and must be merged.
    ///
    /// This works because segments are contiguous - all frames between bottom_pos_inclusive
    /// and top_pos_exclusive belong to this SCC.
    fn check_overlap(existing: &Scc, cycle_start_pos: usize) -> bool {
        // O(1) overlap check using segment bounds.
        // If the existing SCC's upper bound <= cycle start, there's no overlap.
        existing.top_pos_exclusive > cycle_start_pos
    }

    /// Handle an SCC we just detected.
    ///
    /// When a new SCC overlaps with existing SCCs (shares participants),
    /// we merge them to form a larger SCC.
    ///
    /// Optimization: We use stack depth to efficiently find overlapping SCCs.
    /// The cycle spans CalcStack positions [N, M] where M = stack_depth - 1 and
    /// N = M - cycle_length + 1. Any SCC with max_stack_depth < N cannot overlap.
    /// Once we find the first overlapping SCC, all subsequent SCCs must also
    /// overlap (due to LIFO ordering of the SCC stack).
    #[allow(clippy::mutable_key_type)] // CalcId's Hash impl doesn't depend on mutable parts
    fn on_scc_detected(&self, raw: Vec1<CalcId>) {
        let calc_stack_vec = self.into_vec();

        // Create the new SCC
        let new_scc = Scc::new(raw, &calc_stack_vec);
        let detected_at = new_scc.detected_at.dupe();
        let cycle_start_pos = new_scc.bottom_pos_inclusive;

        // Check for overlapping SCCs and merge if needed
        let mut scc_stack = self.scc_stack.borrow_mut();

        // Find the first (oldest) SCC that overlaps with the new cycle.
        // Overlap is determined by O(1) segment arithmetic: if the existing SCC's
        // upper bound (top_pos_exclusive) exceeds cycle_start_pos, they overlap.
        // Due to LIFO ordering, once we find one overlapping SCC, all subsequent ones
        // on the stack must also overlap.
        let mut first_merge_idx: Option<usize> = None;

        for (i, existing) in scc_stack.iter().enumerate() {
            if Self::check_overlap(existing, cycle_start_pos) {
                first_merge_idx = Some(i);
                break; // All subsequent SCCs will also overlap
            }
        }

        if let Some(first_idx) = first_merge_idx {
            // Merge all SCCs from first_idx to end, plus the new SCC
            let sccs_from_stack: Vec<Scc> = scc_stack.drain(first_idx..).collect();
            let sccs_to_merge = Vec1::from_vec_push(sccs_from_stack, new_scc);

            // Use the helper method to merge SCCs
            let mut merged_scc = Scc::merge_many(sccs_to_merge, detected_at.dupe());

            // After a merge, everything from the merged anchor to the current stack top
            // is part of this single SCC. Recompute top_pos_exclusive from scratch.
            merged_scc.top_pos_exclusive = calc_stack_vec.len();

            scc_stack.push(merged_scc);
        } else {
            // No overlap - just push the new SCC
            scc_stack.push(new_scc);
        };
    }

    /// Handle the completion of a calculation. Mark the node as Done in the
    /// top SCC (if it's a participant), then store the completed SCC (if any)
    /// in `pending_completed_scc` for later commit by `get_idx`.
    ///
    /// Only the top SCC is checked because each node appears in at most one
    /// SCC, and active calculations are always in the top SCC.
    fn on_calculation_finished(
        &self,
        current: &CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
        traces: Option<TraceSideEffects>,
    ) -> Arc<dyn Any + Send + Sync> {
        let canonical = {
            let mut scc_stack = self.scc_stack.borrow_mut();
            if let Some(top_scc) = scc_stack.last_mut() {
                let canonical = top_scc.on_calculation_finished(current, answer, errors, traces);
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
            }
        }; // scc_stack borrow dropped here
        self.check_scc_completion();
        canonical
    }

    /// Check whether the top SCC has completed and, if so, pop it into
    /// `pending_completed_scc` for later retrieval by `pop_and_take_completed_scc`.
    ///
    /// An SCC is complete when the stack has unwound to (or past) its anchor
    /// position: `stack_len <= bottom_pos_inclusive + 1`. The `+ 1` exists
    /// because this check runs during calculation, while the completing
    /// frame is still on the stack.
    ///
    /// This is called after recording a node's answer (via either
    /// `on_calculation_finished` or `set_iteration_node_done`) to detect
    /// when the last SCC member has finished.
    fn check_scc_completion(&self) {
        let stack_len = self.stack.borrow().len();
        let mut scc_stack = self.scc_stack.borrow_mut();
        if let Some(scc) = scc_stack.last()
            && stack_len <= scc.bottom_pos_inclusive + 1
        {
            let completed = scc_stack.pop().unwrap();
            // At most one SCC can complete per completion point: verify
            // the next SCC (if any) is not also complete.
            debug_assert!(
                scc_stack
                    .last()
                    .is_none_or(|next| stack_len > next.bottom_pos_inclusive + 1),
                "Multiple SCCs completed at stack_len={stack_len}",
            );
            let mut slot = self.pending_completed_scc.borrow_mut();
            assert!(
                slot.is_none(),
                "pending_completed_scc was not taken before a new SCC completed",
            );
            *slot = Some(completed);
        }
    }

    /// Merge all SCCs from the target SCC to the top of the stack, and add
    /// any free-floating CalcStack nodes between the target SCC's min_stack_depth
    /// and the current stack position.
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
        let mut sccs_to_merge: Vec<Scc> = Vec::new();
        let mut target_bottom_pos_inclusive: Option<usize> = None;
        while let Some(scc) = scc_stack.pop() {
            let is_target = scc.detected_at == *detected_at_of_scc;
            if is_target {
                target_bottom_pos_inclusive = Some(scc.bottom_pos_inclusive);
            }
            sccs_to_merge.push(scc);
            if is_target {
                break;
            }
        }
        let min_depth = target_bottom_pos_inclusive
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");
        let sccs_to_merge = Vec1::try_from_vec(sccs_to_merge)
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");

        // Perform the merge, then add any free-floating bindings that weren't previously part
        // of a known SCC. These nodes are already on the call stack (they have active frames),
        // so they are InProgress, not Fresh.
        let mut merged = Scc::merge_many(sccs_to_merge, detected_at_of_scc.dupe());
        merged.absorb_calc_stack_members(&calc_stack_vec, min_depth);

        // After a merge, everything from the merged anchor to the current stack top
        // is part of this single SCC. Recompute top_pos_exclusive from scratch.
        merged.top_pos_exclusive = calc_stack_vec.len();

        scc_stack.push(merged);
    }

    /// Find the index in `scc_stack` of an SCC that contains `target`.
    ///
    /// Scans the SCC stack for an SCC whose `node_state` membership map
    /// contains the target. Returns the index in the stack (not the SCC's
    /// `bottom_pos_inclusive`). Used for membership-based back-edge detection:
    /// a request for a CalcId in a non-top SCC is a back-edge that must trigger
    /// merge + demotion.
    fn find_scc_containing(&self, target: &CalcId) -> Option<usize> {
        let scc_stack = self.scc_stack.borrow();
        for (i, scc) in scc_stack.iter().enumerate() {
            if scc.node_state.contains_key(target) {
                return Some(i);
            }
        }
        None
    }

    /// Merge from the top SCC anchor when a back-edge re-enters from outside
    /// the top SCC segment.
    ///
    /// A plain top-SCC absorb only handles free-floating nodes and can miss
    /// full SCC merge semantics when SCC fragments are involved. Using
    /// `merge_sccs` here ensures all phase-0 and phase-1+ re-entry paths share
    /// the same merge+absorb behavior and consistent demotion signaling
    /// (`merge_happened`).
    fn merge_top_scc_on_outside_reentry(&self) {
        let detected_at = {
            let stack_len = self.stack.borrow().len();
            let scc_stack = self.scc_stack.borrow();
            if let Some(top_scc) = scc_stack.last() {
                if stack_len > top_scc.top_pos_exclusive {
                    Some(top_scc.detected_at.dupe())
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(detected_at) = detected_at {
            self.merge_sccs(&detected_at);
        }
    }

    /// Shared top-SCC member handling for back-edge re-entry paths in `push`.
    ///
    /// Ensures outside-segment re-entry merge runs first, then restores
    /// top_pos_exclusive symmetry with `pop`, then dispatches using the current
    /// iteration node state.
    fn binding_action_for_top_scc_member(&self, current: &CalcId) -> BindingAction {
        self.merge_top_scc_on_outside_reentry();
        // Increment top_pos_exclusive because pop() will decrement
        // top_pos_exclusive for any node in node_state, so push must
        // balance it with an increment.
        if let Some(top_scc) = self.scc_stack.borrow_mut().last_mut() {
            top_scc.top_pos_exclusive += 1;
        }
        if let Some(kind) = self.get_iteration_node_state(current) {
            return self.binding_action_for_node_state(current, kind);
        }
        // If we merged but the target is somehow not in iteration state,
        // this is a bug: the merge should have included it.
        unreachable!(
            "membership back-edge: target {} was in iterating SCC but \
             not found in merged SCC's iteration state",
            current,
        );
    }

    /// Returns true if the top SCC is iterating at iteration 0 (Phase 0
    /// discovery) or iteration 1 (first iterative cold start).
    ///
    /// During cold-start iteration, back-edges allocate placeholders rather
    /// than reusing previous answers.
    fn is_cold_iteration(&self) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .is_some_and(|scc| scc.iterative.iteration <= 1)
    }

    /// Get the lightweight summary of a target's iteration node state in
    /// the top SCC.
    ///
    /// Returns `None` if the target is not found in the iteration node
    /// states. The summary is safe to use for read-then-act patterns because
    /// it does not borrow the SCC.
    fn get_iteration_node_state(&self, target: &CalcId) -> Option<SccNodeStateKind> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        let node_state = top_scc.node_state.get(target)?;
        let has_previous_answer = top_scc.iterative.previous_answers.contains_key(target);
        Some(node_state.kind(has_previous_answer))
    }

    /// Convert an `SccNodeStateKind` into the appropriate `BindingAction`.
    ///
    /// This is the shared logic for all paths in `push` that find a node in
    /// an SCC's iteration state: cross-SCC merge, same-top-SCC membership,
    /// and the iterative bypass. The mapping is:
    /// - Fresh → mark InProgress, Calculate
    /// - InProgressWithPreviousAnswer → mark recursion break, return previous answer
    /// - InProgressWithPlaceholder → return CycleBroken with the placeholder Var
    /// - InProgressCold → NeedsColdPlaceholder (caller allocates)
    /// - Done → return the SCC-local answer
    fn binding_action_for_node_state(
        &self,
        current: &CalcId,
        kind: SccNodeStateKind,
    ) -> BindingAction {
        match kind {
            SccNodeStateKind::Fresh => {
                self.set_iteration_node_in_progress(current);
                BindingAction::Calculate
            }
            SccNodeStateKind::InProgressWithPreviousAnswer => {
                self.mark_recursion_break(current);
                let answer = self
                    .get_previous_answer(current)
                    .expect("InProgressWithPreviousAnswer but no previous answer found");
                BindingAction::SccLocalAnswer(answer)
            }
            SccNodeStateKind::InProgressWithPlaceholder => {
                let var = self
                    .get_iteration_placeholder(current)
                    .expect("InProgressWithPlaceholder but no placeholder found");
                BindingAction::CycleBroken(var)
            }
            SccNodeStateKind::InProgressCold => BindingAction::NeedsColdPlaceholder,
            SccNodeStateKind::Done => {
                let answer = self
                    .get_iteration_done_answer(current)
                    .expect("Done iteration node state but no answer found");
                BindingAction::SccLocalAnswer(answer)
            }
        }
    }

    /// Mark a target node as `InProgress` in the top SCC's `node_state`.
    ///
    /// Panics if the SCC stack is empty, the target is not a member, or the
    /// target is not `Fresh`.
    fn set_iteration_node_in_progress(&self, target: &CalcId) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        let node_state = top_scc
            .node_state
            .get_mut(target)
            .expect("target is not a member of the iterating SCC");
        assert!(
            matches!(node_state, SccNodeState::Fresh),
            "set_iteration_node_in_progress called on non-Fresh node: {target:?}"
        );
        *node_state = SccNodeState::InProgress;
    }

    /// Set the placeholder variable for a cycle-breaking node in the top SCC's
    /// `node_state`.
    ///
    /// This is used by both Phase 0 (initial cycle detection in
    /// `attempt_to_unwind_cycle_from_here`) and Phase 1+ (iteration via
    /// `NeedsColdPlaceholder` in `get_idx`).
    ///
    /// The write is lenient: it delegates to `Scc::on_placeholder_recorded`,
    /// which uses an advancement rank check so that a `Done` state is never
    /// overwritten back to `HasPlaceholder`. If the top SCC does not contain
    /// the target (e.g. during `handle_depth_overflow` where the node may not
    /// be in any SCC), the call is a no-op.
    fn set_iteration_placeholder(&self, target: &CalcId, var: Var) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        if let Some(top_scc) = scc_stack.last_mut() {
            top_scc.on_placeholder_recorded(target, var);
            // Debug-only check: verify the node isn't in any other SCC.
            debug_assert!(
                scc_stack
                    .iter()
                    .rev()
                    .skip(1)
                    .all(|scc| !scc.node_state.contains_key(target)),
                "set_iteration_placeholder: CalcId {} found in multiple SCCs",
                target,
            );
        }
    }

    /// Retrieve the placeholder Var from SccNodeState::HasPlaceholder in the top SCC.
    /// Returns `Some(var)` if the node has a placeholder, `None` otherwise.
    /// Used during calculate_and_record_answer to determine whether
    /// finalize_recursive_answer needs to be called.
    fn get_iteration_placeholder(&self, target: &CalcId) -> Option<Var> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        match top_scc.node_state.get(target)? {
            SccNodeState::HasPlaceholder(var) => Some(*var),
            _ => None,
        }
    }

    /// Mark a target node as `Done` in the top SCC's `node_state`.
    ///
    /// Silently does nothing if there is no top SCC, which has never been
    /// observed but seems to occur in the LSP (possibly related to indexing).
    ///
    /// This shouldn't be a correctness bug, because if no SCC is found then
    /// there's nothing to set. The SCC likely already finished, and skipping
    /// the update is fine.
    ///
    /// TODO(stroxler): while I'm fairly confident that it's not a correctness bug
    /// to skip this update, it would be good to understand more clearly what the
    /// flow is where we try to update an iteration state on an Scc that does not
    /// exist. It's likely related to the discovery phase and possibly something
    /// in our handling of `bottom_pos_inclusive`.
    fn set_iteration_node_done(
        &self,
        target: &CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
        traces: Option<TraceSideEffects>,
    ) {
        let needs_completion_check = {
            let mut scc_stack = self.scc_stack.borrow_mut();
            let Some(top_scc) = scc_stack.last_mut() else {
                // TODO(stroxler): Consider panicking here once we're confident this
                // path is unreachable in the LSP. The silent no-op may mask bugs.
                debug_assert!(
                    false,
                    "set_iteration_node_done: no iterating SCC on the stack for {:?}",
                    target
                );
                return;
            };
            let is_iteration_0 = top_scc.iterative.iteration == 0;
            top_scc.node_state.insert(
                target.dupe(),
                SccNodeState::Done {
                    answer,
                    errors,
                    traces,
                },
            );
            is_iteration_0
        };
        // During iteration 0 (Phase 0 discovery), SCC members are driven by the
        // normal recursive call chain, not by drive_all_iteration_members. We need
        // completion detection to trigger iterative_resolve_scc when the last
        // member finishes. During iteration >= 1, completion is managed by the
        // iteration loop in iterative_resolve_scc.
        if needs_completion_check {
            self.check_scc_completion();
        }
    }

    /// Set `has_changed = true` on the top SCC's iteration state.
    ///
    /// Called when a node's answer differs from its previous-iteration answer,
    /// indicating the fixpoint has not yet converged.
    ///
    /// Silently does nothing if there is no top SCC. This can occur in the
    /// LSP when the SCC is prematurely popped from the stack due to a
    /// stale `bottom_pos_inclusive` (see pyrefly-docs/scc-stack-invariants/v0-doc.md).
    /// In that case the SCC has already been committed by a nested driver, so
    /// there is no top SCC left to update and skipping is safe.
    fn mark_iteration_changed(&self) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let Some(top_scc) = scc_stack.last_mut() else {
            // TODO(stroxler): Consider panicking here once we're confident this
            // path is unreachable in the LSP. The silent no-op may mask bugs.
            debug_assert!(
                false,
                "mark_iteration_changed: no iterating SCC on the stack"
            );
            return;
        };
        top_scc.iterative.has_changed = true;
    }

    /// Record `target` as a recursion break point in the top SCC's iteration state.
    ///
    /// Called when a back-edge hits `InProgressWithPreviousAnswer` — i.e., when
    /// the cycle is broken by returning the previous-iteration answer. These
    /// break points are where non-convergence errors should be reported, since
    /// other non-converging members are downstream consequences.
    ///
    /// Panics if the SCC stack is empty.
    fn mark_recursion_break(&self, target: &CalcId) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        let top_scc = scc_stack.last_mut().expect("no SCC on the stack");
        top_scc.iterative.recursion_breaks.insert(target.dupe());
    }

    /// Look up the previous-iteration answer for a target in the top SCC.
    ///
    /// Returns `None` if there is no top SCC or there is no previous answer
    /// for the target (e.g., during cold-start iteration 1).
    fn get_previous_answer(&self, target: &CalcId) -> Option<Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        top_scc.iterative.previous_answers.get(target).cloned()
    }

    /// Retrieve the type-erased answer from SccNodeState::Done in the top SCC.
    /// Returns `Some(answer)` if the node is Done, `None` otherwise
    /// (node not in SCC or not Done).
    fn get_iteration_done_answer(&self, target: &CalcId) -> Option<Arc<dyn Any + Send + Sync>> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        match top_scc.node_state.get(target)? {
            SccNodeState::Done { answer, .. } => Some(answer.dupe()),
            _ => None,
        }
    }

    /// Find the first member in the top SCC's iteration state that is `Fresh`.
    ///
    /// Returns `None` if all members have been processed or there is no top
    /// SCC. BTreeMap iteration order gives deterministic results.
    fn next_fresh_member(&self) -> Option<CalcId> {
        let scc_stack = self.scc_stack.borrow();
        let top_scc = scc_stack.last()?;
        for (calc_id, state) in &top_scc.node_state {
            if matches!(state, SccNodeState::Fresh) {
                return Some(calc_id.dupe());
            }
        }
        None
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

    /// Returns true if any SCC exists below the top of the stack.
    ///
    /// Used by the absorption check in `iterative_resolve_scc`: when the top
    /// SCC's `detected_at` has changed (indicating a merge), the driver can
    /// only return early if an ancestor SCC is still on the stack to own the
    /// merged SCC. If no ancestor SCC exists, the current driver must continue
    /// with the merged SCC to avoid orphaning it.
    fn has_ancestor_scc(&self) -> bool {
        let scc_stack = self.scc_stack.borrow();
        // Skip the last element (the top SCC) and check the rest.
        let len = scc_stack.len();
        len >= 2
    }

    /// Returns true if the top SCC's `node_state` contains the given CalcId.
    /// Returns false if the stack is empty (callers should guard with
    /// `sccs_is_empty` first).
    ///
    /// Used after nested absorption to distinguish two cases:
    /// - Our SCC was merged into the top SCC (detected_at changed, but our
    ///   members are in the top SCC's node_state) → continue driving.
    /// - Our SCC was committed by a nested driver, and a pre-existing SCC
    ///   (e.g. a Phase 0 SCC that was below us on the stack) is now the top
    ///   → return.
    ///
    /// This works because `detected_at` is always a member of the SCC's
    /// `node_state`, and merges union the `node_state` maps. Within a single
    /// thread's `scc_stack`, SCCs are disjoint (overlapping membership
    /// triggers a merge), so an unrelated SCC will not contain our CalcId.
    fn top_scc_contains_member(&self, calc_id: &CalcId) -> bool {
        self.scc_stack
            .borrow()
            .last()
            .map(|scc| scc.node_state.contains_key(calc_id))
            .unwrap_or(false)
    }

    /// Returns true if the top SCC's iteration state has `merge_happened` set.
    ///
    /// Used by `drive_all_iteration_members` to detect whether a merge occurred
    /// during the drive loop, so it can defer demotion until after the loop.
    fn top_scc_merge_happened(&self) -> bool {
        let scc_stack = self.scc_stack.borrow();
        scc_stack
            .last()
            .is_some_and(|scc| scc.iterative.merge_happened)
    }

    /// Set the `demoted` flag on the top SCC's iteration state.
    ///
    /// Used by `drive_all_iteration_members` to defer demotion: if a merge
    /// occurred during the drive loop, the demotion is applied after the loop
    /// completes rather than mid-loop (which would cause re-driving of
    /// already-done members).
    fn set_top_scc_demoted(&self, demoted: bool) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        if let Some(scc) = scc_stack.last_mut() {
            scc.iterative.demoted = demoted;
        }
    }

    /// Removes a CalcId from the top SCC's `node_state`.
    ///
    /// Used when `drive_member` was a no-op (e.g., the target module's
    /// Answers were evicted by another thread). Removing the member from
    /// node state prevents `next_fresh_member` from returning it
    /// again, breaking what would otherwise be an infinite loop.
    ///
    /// If a merge or iteration restart rebuilds node states, the member
    /// may be re-added as Fresh and re-detected on the next drive loop
    /// (which is harmless — the eviction is persistent, so the member
    /// is immediately removed again).
    fn remove_from_iteration_state(&self, calc_id: &CalcId) {
        let mut scc_stack = self.scc_stack.borrow_mut();
        if let Some(scc) = scc_stack.last_mut() {
            scc.node_state.remove(calc_id);
        } else {
            // TODO(stroxler): Consider panicking here once we're confident this
            // path is unreachable in the LSP. The silent no-op may mask bugs.
            debug_assert!(
                false,
                "remove_from_iteration_state: no iterating SCC on the stack for {:?}",
                calc_id
            );
        }
    }
}

/// Tracks the state of a node within an active SCC.
///
/// This replaces the previous stack-based tracking (recursion_stack, unwind_stack)
/// with explicit state tracking. The state transitions are:
/// - Fresh → InProgress (when we first encounter the node as a Participant)
/// - InProgress → HasPlaceholder (when a placeholder is recorded for cycle breaking)
/// - InProgress/HasPlaceholder → Done (when the node's calculation completes)
///
/// The variants are ordered by "advancement" (Fresh < InProgress < HasPlaceholder < Done).
/// The `advancement_rank()` method encodes this ordering for use during SCC merge.
#[derive(Debug, Clone)]
pub enum SccNodeState {
    /// Node hasn't been processed yet as part of SCC handling.
    Fresh,
    /// Node is currently being processed (on the Rust call stack).
    InProgress,
    /// A placeholder has been recorded in SCC-local state for cycle breaking,
    /// but we haven't computed the real answer yet.
    /// The Var is the placeholder variable recorded for this node.
    HasPlaceholder(Var),
    /// Node's calculation has completed. Stores the type-erased answer and
    /// error collector for thread-local SCC isolation.
    ///
    /// For SCC participants, the answer is stored here until the entire SCC
    /// completes, at which point answers are committed to their respective
    /// Calculation cells.
    Done {
        answer: Arc<dyn Any + Send + Sync>,
        /// Errors collected during solving. None during Phase 0 (cold start).
        errors: Option<Arc<ErrorCollector>>,
        /// Trace side effects collected during solving. None during Phase 0.
        traces: Option<TraceSideEffects>,
    },
}

impl SccNodeState {
    /// Returns a numeric rank for the advancement level of this state.
    /// Used during SCC merge to keep the more advanced state.
    /// Fresh(0) < InProgress(1) < HasPlaceholder(2) < Done(3)
    fn advancement_rank(&self) -> u8 {
        match self {
            SccNodeState::Fresh => 0,
            SccNodeState::InProgress => 1,
            SccNodeState::HasPlaceholder(_) => 2,
            SccNodeState::Done { .. } => 3,
        }
    }
}

/// The action to take for a binding after checking CalcStack and SCC state.
///
/// This flattens the nested match on `SccState` into a single discriminated
/// union. The `CalcStack::push` method performs all state checks and SCC
/// mutations (like `merge_sccs`, `on_scc_detected`, `on_calculation_finished`),
/// returning the action that `get_idx` should take. Push is purely thread-local
/// and never touches the cross-thread Calculation cell.
enum BindingAction {
    /// Calculate the binding and record the answer.
    /// Action: call `calculate_and_record_answer`
    Calculate,
    /// A recursive placeholder exists (in SCC-local `SccNodeState::HasPlaceholder`)
    /// and we should return it.
    /// Action: return `Arc::new(K::promote_recursive(heap, r))`
    CycleBroken(Var),
    /// An answer is available from SccNodeState::Done in the top SCC.
    /// Type-erased; will be downcast to `Arc<K::Answer>` in `get_idx`.
    /// Action: downcast and return
    SccLocalAnswer(Arc<dyn Any + Send + Sync>),
    /// A cycle break point where a placeholder is needed. The caller (`get_idx`)
    /// calls `attempt_to_unwind_cycle_from_here` to check if another thread
    /// already committed an answer, and if not, allocates a placeholder via
    /// `K::create_recursive` and stores it in SCC-local state.
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
    /// Whether an SCC merge occurred during the current drive loop.
    /// When set, `drive_all_iteration_members` defers demotion until after
    /// the loop completes, ensuring each member is visited at most once
    /// per iteration regardless of how many merges occur.
    pub merge_happened: bool,
    /// Members whose cycle was broken by returning a previous-iteration answer
    /// (i.e., hit `InProgressWithPreviousAnswer`). These are the actual recursion
    /// break points; other non-converging members are downstream consequences.
    /// Used to limit non-convergence error reporting to only the break points.
    pub recursion_breaks: BTreeSet<CalcId>,
}

// `SccNodeState` is used by both Phase 0 discovery and iterative fixpoint solving.

/// Lightweight summary of an `SccNodeState` for borrow-safe read-then-act
/// patterns.
///
/// Reading the full `SccNodeState` requires borrowing the SCC, but we
/// often need to drop that borrow before mutating. This enum captures just
/// enough information to decide what action to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SccNodeStateKind {
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

impl SccNodeState {
    /// Compute the lightweight summary kind from this state plus whether a
    /// previous answer exists for the same node.
    pub fn kind(&self, has_previous_answer: bool) -> SccNodeStateKind {
        match self {
            SccNodeState::Fresh => SccNodeStateKind::Fresh,
            SccNodeState::HasPlaceholder(_) => SccNodeStateKind::InProgressWithPlaceholder,
            SccNodeState::InProgress => {
                if has_previous_answer {
                    SccNodeStateKind::InProgressWithPreviousAnswer
                } else {
                    SccNodeStateKind::InProgressCold
                }
            }
            SccNodeState::Done { .. } => SccNodeStateKind::Done,
        }
    }
}

/// Represent an SCC (Strongly Connected Component) we are currently solving.
///
/// This simplified model tracks SCC participants with explicit state rather than
/// using separate recursion and unwind stacks. The Rust call stack naturally
/// enforces LIFO ordering, so we only need to track the state of each
/// participant (Fresh/InProgress/Done).
#[derive(Debug, Clone)]
pub struct Scc {
    /// State of each participant in this SCC.
    /// Keys are all participants; values track their computation state.
    node_state: BTreeMap<CalcId, SccNodeState>,
    /// Where we detected the SCC (for debugging only)
    detected_at: CalcId,
    /// Stack position of the SCC anchor (the position of the detected_at CalcId).
    /// The detected_at CalcId is the one that was pushed twice, triggering cycle
    /// detection; its first occurrence is at the deepest position in the cycle
    /// (cycle_start), making it a robust anchor.
    /// When the stack length drops to bottom_pos_inclusive, the SCC is complete.
    /// This enables O(1) completion checking instead of iterating all participants.
    bottom_pos_inclusive: usize,
    /// Exclusive upper bound of this SCC's segment on the calc stack.
    /// The segment is [bottom_pos_inclusive, top_pos_exclusive).
    /// Initially set to the stack length when the SCC is created; updated on merge.
    top_pos_exclusive: usize,
    /// Iteration state for iterative fixpoint solving.
    /// Invariant: every active SCC has iteration state, initialized to
    /// iteration 0 on creation (Phase 0 discovery), then reset to
    /// iteration 1 by `reset_for_cold_start` when entering iterative solving.
    iterative: SccIterationState,
}

impl Display for Scc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let states: Vec<_> = self.node_state.iter().collect();
        write!(
            f,
            "Scc{{node_state: {:?}, detected_at: {}}}",
            states, self.detected_at,
        )
    }
}

impl Scc {
    #[allow(clippy::mutable_key_type)] // CalcId's Hash impl doesn't depend on mutable parts
    fn new(raw: Vec1<CalcId>, calc_stack_vec: &[CalcId]) -> Self {
        let detected_at = raw.first().dupe();

        // Initialize all nodes as Fresh
        let node_state: BTreeMap<CalcId, SccNodeState> = raw
            .iter()
            .duped()
            .map(|c| (c, SccNodeState::Fresh))
            .collect();

        // The anchor is the detected_at CalcId (the one pushed twice, triggering cycle
        // detection). Its first occurrence is at the deepest position in the cycle
        // (cycle_start), making it a more robust anchor.
        //
        // The segment spans from the anchor to the top of the stack.
        let bottom_pos_inclusive = calc_stack_vec
            .iter()
            .position(|c| c == &detected_at)
            .unwrap_or(0);

        Scc {
            node_state,
            detected_at,
            bottom_pos_inclusive,
            top_pos_exclusive: calc_stack_vec.len(),
            iterative: SccIterationState {
                iteration: 0,
                previous_answers: BTreeMap::new(),
                demoted: false,
                has_changed: false,
                merge_happened: false,
                recursion_breaks: BTreeSet::new(),
            },
        }
    }

    /// Track that a calculation has finished, marking it as Done.
    /// Stores the type-erased answer and error collector in SccNodeState.
    /// For SCC participants, this is the primary storage until batch commit.
    ///
    /// This method implements first-answer-wins semantics: once a node is marked
    /// as Done, subsequent calculations (from duplicate stack frames within an SCC)
    /// do not overwrite the state. This ensures that the first computed answer is
    /// the one that persists, consistent with Calculation::record_value semantics.
    ///
    /// Returns the canonical answer: the one that is (or was already) stored in
    /// SccNodeState::Done. If the node was already Done, returns the pre-existing
    /// answer without overwriting. If the node was not yet Done, stores the
    /// provided answer and returns a clone of it. If the node is not tracked
    /// by this SCC at all, returns the provided answer unchanged.
    fn on_calculation_finished(
        &mut self,
        current: &CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
        traces: Option<TraceSideEffects>,
    ) -> Arc<dyn Any + Send + Sync> {
        if let Some(state) = self.node_state.get_mut(current) {
            if let SccNodeState::Done {
                answer: existing_answer,
                ..
            } = state
            {
                // Already Done: return the canonical (first-written) answer.
                existing_answer.dupe()
            } else {
                *state = SccNodeState::Done {
                    answer: answer.dupe(),
                    errors,
                    traces,
                };
                answer
            }
        } else {
            // Node not tracked by this SCC; return the provided answer as-is.
            answer
        }
    }

    /// Track that a placeholder has been recorded for a cycle-breaking node.
    fn on_placeholder_recorded(&mut self, current: &CalcId, var: Var) {
        if let Some(state) = self.node_state.get_mut(current) {
            // Only upgrade: do not overwrite Done back to HasPlaceholder.
            // This is defense-in-depth; placeholder recording should not
            // regress a completed node.
            if state.advancement_rank() < SccNodeState::HasPlaceholder(var).advancement_rank() {
                *state = SccNodeState::HasPlaceholder(var);
            }
        }
    }

    /// Merge two SCCs into one, taking the most advanced state for each
    /// participant.
    ///
    /// Node states are merged via `node_state` (keeping the more advanced
    /// state). If either SCC has iteration state (`iterative: Some(...)`),
    /// the merged SCC preserves iteration metadata (iteration number,
    /// previous answers). The `merge_happened` flag is set so that
    /// `drive_all_iteration_members` can defer demotion until after the
    /// current drive loop completes (bounding per-iteration work to O(N)).
    #[allow(clippy::mutable_key_type)]
    fn merge(mut self, other: Scc) -> Self {
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
        self.bottom_pos_inclusive = self.bottom_pos_inclusive.min(other.bottom_pos_inclusive);
        // Note: top_pos_exclusive is NOT updated here. After a merge, everything from
        // the merged anchor to the current stack top is part of this single SCC.
        // The caller must recompute top_pos_exclusive = stack.len().

        // Merge iteration state. Node states are already merged via `node_state`
        // above; the iteration state only carries metadata (iteration number,
        // previous answers, flags).
        // Set merge_happened so the drive loop defers demotion until
        // after the current iteration completes, bounding per-iteration work
        // to O(N) regardless of how many merges occur.
        // Take max iteration from either SCC: if one has progressed further,
        // we should not regress to iteration 1.
        let iteration = self.iterative.iteration.max(other.iterative.iteration);
        // Union previous_answers from both SCCs. Start with other's answers,
        // then extend with self's (self is the older/lower SCC so its answers
        // take priority on overlap).
        let mut previous_answers = other.iterative.previous_answers;
        previous_answers.extend(self.iterative.previous_answers);
        self.iterative = SccIterationState {
            iteration,
            previous_answers,
            demoted: false,
            has_changed: false,
            merge_happened: true,
            recursion_breaks: BTreeSet::new(),
        };

        self
    }

    /// Merge multiple SCCs into one.
    ///
    /// The `detected_at` parameter is an additional candidate for the minimum
    /// detected_at, used when the detection point may not be represented in
    /// any of the SCCs being merged.
    fn merge_many(sccs: Vec1<Scc>, detected_at: CalcId) -> Self {
        let (first, rest) = sccs.split_off_first();
        let mut result = rest.into_iter().fold(first, Scc::merge);
        if detected_at < result.detected_at {
            result.detected_at = detected_at;
        }
        result
    }

    /// Absorb CalcStack members from `calc_stack[from_pos..]` into this SCC.
    ///
    /// Adds each CalcId as `SccNodeState::InProgress` to `node_state` (if not already
    /// present). Sets `merge_happened = true` on the iteration state if any new
    /// entries are added.
    ///
    /// This is used for free-floating nodes: CalcIds that are on the call stack
    /// (their frames are active) but were not previously tracked by any SCC.
    /// They must be `InProgress` (not `Fresh`) because their computation has
    /// already started — a revisit of a `Fresh` node would incorrectly trigger
    /// the `Participant → InProgress` transition again.
    #[allow(clippy::mutable_key_type)]
    fn absorb_calc_stack_members(&mut self, calc_stack: &[CalcId], from_pos: usize) {
        let mut added_new = false;
        for calc_id in calc_stack.iter().skip(from_pos) {
            self.node_state.entry(calc_id.dupe()).or_insert_with(|| {
                added_new = true;
                SccNodeState::InProgress
            });
        }
        if added_new {
            self.iterative.merge_happened = true;
        }
    }

    /// Extract done answers from `node_state`.
    ///
    /// Iterates over `node_state`, collecting answers from `Done` variants
    /// into a `BTreeMap`. Used to build `previous_answers` for the next
    /// iteration.
    #[allow(clippy::mutable_key_type)]
    fn extract_done_answers(&self) -> BTreeMap<CalcId, Arc<dyn Any + Send + Sync>> {
        let mut answers = BTreeMap::new();
        for (calc_id, state) in &self.node_state {
            if let SccNodeState::Done { answer, .. } = state {
                answers.insert(calc_id.dupe(), answer.dupe());
            }
        }
        answers
    }

    /// Reset the SCC for a cold start at iteration 1.
    ///
    /// Used for Phase 0 → iteration 1 and for demotion restarts. Clears all
    /// iteration metadata (previous answers, recursion breaks, flags) and
    /// resets every member state to Fresh.
    fn reset_for_cold_start(&mut self) {
        for state in self.node_state.values_mut() {
            *state = SccNodeState::Fresh;
        }
        self.iterative = SccIterationState {
            iteration: 1,
            previous_answers: BTreeMap::new(),
            demoted: false,
            has_changed: false,
            merge_happened: false,
            recursion_breaks: BTreeSet::new(),
        };
        debug_assert!(
            self.node_state
                .values()
                .all(|s| matches!(s, SccNodeState::Fresh)),
            "reset_for_cold_start: not all nodes are Fresh after reset"
        );
        debug_assert!(
            self.iteration() == 1,
            "reset_for_cold_start: iteration should be 1 after cold start"
        );
    }

    /// Advance to the next warm iteration during fixpoint progression.
    ///
    /// Moves current Done answers into `previous_answers` (via
    /// `extract_done_answers`), resets all member states to Fresh, increments
    /// the iteration counter, and clears flags.
    #[allow(clippy::mutable_key_type)]
    fn advance_to_next_warm_iteration(&mut self) {
        let previous_answers = self.extract_done_answers();
        let current_iteration = self.iterative.iteration;
        for state in self.node_state.values_mut() {
            *state = SccNodeState::Fresh;
        }
        self.iterative = SccIterationState {
            iteration: current_iteration + 1,
            previous_answers,
            demoted: false,
            has_changed: false,
            merge_happened: false,
            recursion_breaks: BTreeSet::new(),
        };
        debug_assert!(
            self.node_state
                .values()
                .all(|s| matches!(s, SccNodeState::Fresh)),
            "advance_to_next_warm_iteration: not all nodes are Fresh after advance"
        );
        debug_assert!(
            self.iteration() >= 2,
            "advance_to_next_warm_iteration: iteration should be >= 2 after warm advance"
        );
    }

    /// Returns the current iteration number.
    fn iteration(&self) -> u32 {
        self.iterative.iteration
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
    /// Partial answers for inline first-use pinning, keyed by (NameAssign def_idx, CalcStack height).
    /// The height ensures that only ForwardToFirstUse bindings at the same CalcStack depth
    /// as the NameAssign's solve_binding can see the partial answer (offset 0 in get_idx,
    /// which checks before pushing its own frame).
    partial_answers: RefCell<FxHashMap<(Idx<Key>, usize), Arc<TypeInfo>>>,
    /// Solve-time mapping from per-module lambda parameter IDs to the
    /// thread-local Var that represents that parameter in the current solve.
    lambda_param_vars: RefCell<FxHashMap<(ModuleName, LambdaParamId), Var>>,
    /// Active trace side-effect sink for the current calculation.
    /// Set before `K::solve`, taken after. `None` when tracing is disabled
    /// or between calculations. Saved sinks form a stack to handle recursive
    /// calls to `calculate_and_record_answer`.
    trace_sink: RefCell<Option<TraceSideEffects>>,
    /// Stack of saved trace sinks from outer calculations. When a nested
    /// `calculate_and_record_answer` installs a new sink, the current sink
    /// is pushed here. When the nested call takes its sink, the previous
    /// one is restored.
    trace_sink_stack: RefCell<Vec<Option<TraceSideEffects>>>,
}

impl ThreadState {
    pub fn new(recursion_limit_config: Option<RecursionLimitConfig>) -> Self {
        Self {
            stack: CalcStack::new(),
            debug: RefCell::new(false),
            recursion_limit_config,
            partial_answers: RefCell::new(FxHashMap::default()),
            lambda_param_vars: RefCell::new(FxHashMap::default()),
            trace_sink: RefCell::new(None),
            trace_sink_stack: RefCell::new(Vec::new()),
        }
    }

    /// Install a fresh trace sink for the current calculation, saving any
    /// existing sink for later restoration.
    pub(crate) fn install_trace_sink(&self) {
        let previous = self.trace_sink.borrow_mut().take();
        self.trace_sink_stack.borrow_mut().push(previous);
        *self.trace_sink.borrow_mut() = Some(TraceSideEffects::default());
    }

    /// Take the accumulated trace side effects, restoring any saved sink
    /// from an outer calculation.
    pub(crate) fn take_trace_sink(&self) -> Option<TraceSideEffects> {
        let result = self.trace_sink.borrow_mut().take();
        let restored = self.trace_sink_stack.borrow_mut().pop().flatten();
        *self.trace_sink.borrow_mut() = restored;
        result
    }

    /// Append a type trace to the active sink. No-op if no sink is installed.
    pub(crate) fn record_type_trace(&self, loc: TextRange, ty: Arc<Type>) {
        if let Some(sink) = self.trace_sink.borrow_mut().as_mut() {
            sink.types.insert(loc, ty);
        }
    }

    /// Append a resolved callee trace to the active sink.
    pub(crate) fn record_resolved_trace(&self, loc: TextRange, callee: OverloadedCallee) {
        if let Some(sink) = self.trace_sink.borrow_mut().as_mut() {
            sink.overloaded_callees.insert(loc, callee);
        }
    }

    /// Append an overload trace to the active sink.
    pub(crate) fn record_overload_trace(&self, loc: TextRange, callee: OverloadedCallee) {
        if let Some(sink) = self.trace_sink.borrow_mut().as_mut() {
            sink.overloaded_callees.insert(loc, callee);
        }
    }

    /// Append a property getter trace to the active sink.
    pub(crate) fn record_property_getter_trace(&self, loc: TextRange, ty: Arc<Type>) {
        if let Some(sink) = self.trace_sink.borrow_mut().as_mut() {
            sink.invoked_properties.insert(loc, ty);
        }
    }
}

/// Maximum number of fixpoint iterations before the iterative SCC solver
/// gives up and commits the last answers. Exceeding this threshold produces
/// a type error but accepts the result as-is, since the answer will usually
/// still be approximately correct.
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

/// RAII guard that releases write locks on panic during SCC batch commit.
///
/// Holds a list of `CalcId`s whose Calculation cells have been write-locked.
/// On drop (panic), calls `write_unlock_empty` on each to release the locks
/// without writing values, preventing deadlocks. On success, call `disarm()`
/// to clear the list so `drop` is a no-op.
struct SccWriteLockGuard<'a, 'b, Ans: LookupAnswer> {
    solver: &'a AnswersSolver<'b, Ans>,
    locked: Vec<CalcId>,
}

impl<Ans: LookupAnswer> Drop for SccWriteLockGuard<'_, '_, Ans> {
    fn drop(&mut self) {
        for calc_id in &self.locked {
            self.solver.write_unlock_empty_single(calc_id);
        }
    }
}

impl<Ans: LookupAnswer> SccWriteLockGuard<'_, '_, Ans> {
    /// Disarm the guard after a successful commit. Clears the locked list
    /// so `drop` does nothing.
    fn disarm(mut self) {
        self.locked.clear();
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn fixpoint_details_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("PYREFLY_FIXPOINT_DETAILS")
                .is_some_and(|value| !value.is_empty() && value != "0")
        })
    }

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
                    self.uniques.fresh(),
                    None,
                    Restriction::Unrestricted,
                    PreInferenceVariance::Invariant,
                ),
                QuantifiedKind::TypeVarTuple => {
                    Quantified::type_var_tuple(name, self.uniques.fresh(), None)
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

    /// Look up the fields of a class from binding metadata.
    ///
    /// For same-module classes, reads directly from local bindings metadata.
    /// For cross-module classes, delegates to `LookupAnswer::get_class_fields`
    /// which caches metadata per module and registers class-level dependencies
    /// for proper incremental invalidation.
    ///
    /// Returns `None` if the `ClassDefIndex` is stale (cross-module only;
    /// same-module indices are always valid).
    pub fn get_class_fields(&self, cls: &Class) -> Option<&ClassFields> {
        if cls.module_path() == self.module().path() {
            return Some(&self.bindings.metadata().get_class(cls.index()).fields);
        }
        self.answers.get_class_fields(cls)
    }

    pub(crate) fn set_lambda_param_var(&self, id: LambdaParamId, var: Var) {
        self.thread_state
            .lambda_param_vars
            .borrow_mut()
            .insert((self.module().name(), id), var);
    }

    pub(crate) fn get_lambda_param_var(&self, id: LambdaParamId) -> Option<Var> {
        self.thread_state
            .lambda_param_vars
            .borrow()
            .get(&(self.module().name(), id))
            .copied()
    }

    pub(crate) fn get_or_create_lambda_param_var(&self, id: LambdaParamId) -> Var {
        if let Some(var) = self.get_lambda_param_var(id) {
            var
        } else {
            let var = self.solver().fresh_unwrap(self.uniques);
            self.set_lambda_param_var(id, var);
            var
        }
    }

    /// Resolve a lambda parameter Var from thread-local state.
    ///
    /// If owner exists, force owner evaluation first so this binding
    /// participates in the same SCC/fixpoint dynamics as the containing
    /// lambda expression.
    pub(crate) fn resolve_lambda_param_var(
        &self,
        id: LambdaParamId,
        owner: Option<Idx<Key>>,
    ) -> Var {
        if let Some(owner_idx) = owner {
            let _ = self.get_idx(owner_idx);
        }
        self.get_or_create_lambda_param_var(id)
    }

    pub fn stack(&self) -> &CalcStack {
        &self.thread_state.stack
    }

    /// Access the thread-local state for trace recording.
    pub(crate) fn trace_state(&self) -> &ThreadState {
        self.thread_state
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

        let calculation = self.get_calculation(idx);

        // Fast path: if the value is already calculated, return it immediately
        // without constructing a CalcId or touching the CalcStack. This avoids
        // the CalcId Arc increment, position_of hash map insert/remove, RefCell
        // borrows, and SCC checks for the common case of re-reading an already-
        // solved binding.
        if let Some(v) = calculation.get() {
            return v;
        }

        let current = CalcId(self.bindings().dupe(), K::to_anyidx(idx));

        // Check depth limit before any calculation
        if let Some(config) = self.recursion_limit_config()
            && self.stack().len() > config.limit as usize
        {
            let result = self.handle_depth_overflow(&current, idx, calculation, config);
            return result;
        }

        // Register this thread's intent to calculate with the Calculation cell.
        // answers_solver intentionally ignores Calculation's cycle semantics
        // (CycleDetected vs Calculatable) and uses the thread-local CalcStack
        // as the sole source of truth for cycle detection.
        match calculation.propose_calculation() {
            ProposalResult::Calculated(v) => return v,
            ProposalResult::Calculatable | ProposalResult::CycleDetected => {
                // Both cases proceed into push, which uses thread-local
                // CalcStack cycle detection exclusively.
            }
        }

        let mut result = match self.stack().push(current.dupe()) {
            BindingAction::Calculate => self.calculate_and_record_answer(current, idx, calculation),
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
            BindingAction::NeedsColdPlaceholder => self
                .attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r))),
        };
        if let Some(scc) = self.stack().pop_and_take_completed_scc() {
            self.iterative_resolve_scc(scc);
        }
        // After SCC iteration, the Calculation cell may hold a newer answer
        // than what `calculate_and_record_answer` returned. This happens when
        // the current CalcId is an SCC member: in iterative mode, the answer
        // is stored in SCC-local SccNodeState::Done (not in the Calculation cell)
        // and `calculate_and_record_answer` returns the first-iteration answer.
        // After `iterative_resolve_scc` commits the final iterated answer to
        // the Calculation cell, we must re-read it so that callers (like
        // KeyExport nodes that depend on SCC members) see the SCC's final
        // answer rather than the stale pre-iteration answer.
        if let Some(v) = calculation.get() {
            result = v;
        }
        result
    }
    /// Calculate the answer for a binding using `K::solve` and record it.
    ///
    /// This is called when the `push` method determines we need to actually
    /// compute the value (i.e., `push` returned `BindingAction::Calculate`).
    ///
    /// There are three recording paths, reflecting the fact that SCC membership
    /// can be discovered at two different times:
    ///
    /// - **Already-known SCC member** (iterative path): If the node is already
    ///   in the top SCC's iteration state at the start of this function, we
    ///   delegate to `calculate_and_record_answer_iterative`. This applies
    ///   whenever SCC membership is known before computation begins.
    ///
    /// - **Became an SCC member during computation** (SCC discovery path): A
    ///   node may not be in any SCC when `push` returns `Calculate`, but a
    ///   dependency chain explored during `K::solve` can cycle back to it,
    ///   creating an SCC mid-computation. After `K::solve` returns, we check
    ///   `is_scc_participant` to catch this case and store the answer in
    ///   SCC-local state via `on_calculation_finished`.
    ///
    /// - **Not an SCC member** (direct path): The node is not in any SCC even
    ///   after computation. The answer is written directly to `Calculation`.
    ///
    /// Key invariant: at push time, we can determine that a node IS in an SCC
    /// (because its identity is tracked in an SCC's `node_state`), but we
    /// cannot determine that it is NOT in an SCC until after computation
    /// completes — because any dependency chain can cycle back to this node
    /// during `K::solve`, creating an SCC that includes it.
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
        // Already-known SCC member: the node was in the top SCC's
        // iteration state when `push` was called.
        // Delegate to the iterative path, which handles error suppression,
        // convergence comparison, and SCC-local storage.
        if self.stack().get_iteration_node_state(&current).is_some() {
            return self.calculate_and_record_answer_iterative(current, idx);
        }

        // Skip trace sink setup during cold-start iterations: their answers and
        // diagnostics are intentionally discarded, so collecting trace side
        // effects would only add avoidable allocation churn.
        let tracing_enabled = self.current().tracing_enabled();
        if tracing_enabled {
            self.thread_state.install_trace_sink();
        }

        let binding = self.bindings().get(idx);
        // Note that we intentionally do not pass in the key when solving the binding,
        // as the result of a binding should not depend on the key it was bound to.
        // We use the range for error reporting.
        let range = K::range_with(idx, self.bindings());

        let local_errors = self.error_collector();
        let raw_answer = K::solve(self, binding, range, &local_errors);

        // Take accumulated traces.
        let trace_side_effects = if tracing_enabled {
            self.thread_state.take_trace_sink()
        } else {
            None
        };

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
            // Became an SCC member during computation: an SCC was discovered by
            // a dependency chain during K::solve above, and this node is now in
            // the top SCC's node_state. Store the answer in SCC-local state for
            // batch commit when the SCC completes.
            //
            // If this node has a placeholder Var (from cycle breaking), we must
            // finalize the recursive answer now, before storing. Finalization
            // mutates solver state (force_var) and must happen during computation,
            // not at batch commit.
            let answer = if let Some(var) = self.stack().get_iteration_placeholder(&current) {
                self.finalize_recursive_answer::<K>(idx, var, raw_answer, &local_errors)
            } else {
                raw_answer
            };
            // Also store in SccNodeState::Done for SCC-local isolation (the SCC
            // uses these answers via SccLocalAnswer without touching Calculation).
            let answer_erased: Arc<dyn Any + Send + Sync> = Arc::new(answer.dupe());
            let canonical_erased =
                self.stack()
                    .on_calculation_finished(&current, answer_erased, None, None);
            // Use the canonical answer from thread-local state, mirroring how
            // Calculation::record_value returns the first-written answer.
            Arc::unwrap_or_clone(
                canonical_erased
                    .downcast::<Arc<K::Answer>>()
                    .expect("on_calculation_finished canonical answer downcast failed"),
            )
        } else {
            // Not an SCC member even after computation: write directly to
            // Calculation. No recursive placeholder can exist because
            // placeholders are stored only in SCC-local SccNodeState::HasPlaceholder.
            let (answer, did_write) = calculation.record_value(raw_answer);
            if did_write {
                self.base_errors.extend(local_errors);
                // Publish trace side effects alongside errors.
                if let Some(traces) = trace_side_effects {
                    self.current().merge_trace_side_effects(traces);
                }
            }
            answer
        }
    }

    /// Iterative path for `calculate_and_record_answer`.
    ///
    /// Called when the current CalcId is a member of the top SCC's iteration
    /// state. Unlike the legacy path, this:
    /// - During cold-start iteration 1, bypasses `LoopPhi` bindings by
    ///   resolving only the prior/default index. This prevents LoopPhi from
    ///   creating its own recursive placeholder, which would conflict with
    ///   the iterative placeholder system.
    /// - Uses `error_swallower()` during cold-start (iteration 1) because
    ///   cold-start answers are based on placeholders and produce spurious
    ///   errors. From iteration 2 onward, uses `error_collector()` to capture
    ///   errors that will be committed if this is the final iteration.
    /// - Deep-forces the answer before storage to avoid Var-ID inequality
    ///   in convergence comparisons.
    /// - Finalizes any placeholder created for this node during cycle breaking.
    /// - Compares the answer to `previous_answers` via `answers_equal` and
    ///   calls `mark_iteration_changed` if they differ.
    /// - Stores the answer in `IterationSccNodeState::Done` (SCC-local), NOT in
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
        // create a recursive placeholder. That conflicts with the iterative
        // placeholder system. Instead, resolve only the prior/default value
        // (the value from before the loop body) and use it as the answer.
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
                // Traces are None because this is cold-start (traces are swallowed).
                self.stack()
                    .set_iteration_node_done(&current, answer_erased.clone(), None, None);

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

        // Install trace sink if tracing is enabled for this module.
        // We must always install a sink (even during cold iteration) to prevent
        // traces from leaking into an outer trace sink owned by a different module.
        // During cold iteration, the traces are discarded (just like errors).
        let tracing_enabled = self.current().tracing_enabled();
        if tracing_enabled {
            self.thread_state.install_trace_sink();
        }

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

        // Take accumulated traces. Discard during cold iteration (like errors).
        let trace_side_effects = if tracing_enabled {
            let traces = self.thread_state.take_trace_sink();
            if self.stack().is_cold_iteration() {
                None
            } else {
                traces
            }
        } else {
            None
        };

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

        // Store in IterationSccNodeState::Done. Do NOT write to Calculation;
        // that happens only when the iteration driver commits final answers.
        let errors = if self.stack().is_cold_iteration() {
            None
        } else {
            Some(Arc::new(local_errors))
        };
        self.stack()
            .set_iteration_node_done(&current, answer_erased, errors, trace_side_effects);

        answer
    }

    /// Returns true if the cell is same-module.
    fn is_same_module(&self, calc_id: &CalcId) -> bool {
        let CalcId(ref bindings, _) = *calc_id;
        bindings.module().name() == self.bindings().module().name()
            && bindings.module().path() == self.bindings().module().path()
    }

    /// Acquire a write lock on a single Calculation cell.
    /// Routes to same-module or cross-module depending on the CalcId.
    fn write_lock_single(&self, calc_id: &CalcId) -> bool {
        let CalcId(_, ref any_idx) = *calc_id;
        if self.is_same_module(calc_id) {
            dispatch_anyidx!(any_idx, self, write_lock_same_module)
        } else {
            self.answers.write_lock_in_module(calc_id)
        }
    }

    /// Same-module write lock: acquire the write lock on a typed Calculation cell.
    fn write_lock_same_module<K: Solve<Ans>>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.get_calculation(idx).write_lock()
    }

    /// Write a value to a write-locked cell and release the lock, with error handling.
    /// Routes to same-module or cross-module depending on the CalcId.
    fn write_unlock_single(
        &self,
        calc_id: CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
        traces: Option<TraceSideEffects>,
    ) -> bool {
        let CalcId(_, ref any_idx) = calc_id;
        if self.is_same_module(&calc_id) {
            dispatch_anyidx!(
                any_idx,
                self,
                write_unlock_same_module,
                answer,
                errors,
                traces
            )
        } else {
            self.answers
                .write_unlock_in_module(calc_id, answer, errors, traces)
        }
    }

    /// Same-module write unlock: write the answer and handle errors.
    fn write_unlock_same_module<K: Solve<Ans>>(
        &self,
        idx: Idx<K>,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
        traces: Option<TraceSideEffects>,
    ) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let typed_answer: Arc<K::Answer> = Arc::unwrap_or_clone(
            answer
                .downcast::<Arc<K::Answer>>()
                .expect("write_unlock_same_module: type mismatch in batch commit"),
        );
        let calculation = self.get_calculation(idx);
        let (_answer, did_write) = calculation.write_unlock(typed_answer);
        if did_write {
            if let Some(errors) = errors {
                let errors = Arc::try_unwrap(errors).expect(
                    "Arc<ErrorCollector> refcount > 1 during write_unlock; \
                     errors would be silently lost.",
                );
                self.base_errors.extend(errors);
            }
            if let Some(traces) = traces {
                self.current().merge_trace_side_effects(traces);
            }
        }
        did_write
    }

    /// Release a write lock without writing a value (panic cleanup).
    /// Routes to same-module or cross-module depending on the CalcId.
    fn write_unlock_empty_single(&self, calc_id: &CalcId) {
        let CalcId(_, ref any_idx) = *calc_id;
        if self.is_same_module(calc_id) {
            dispatch_anyidx!(any_idx, self, write_unlock_empty_same_module)
        } else {
            self.answers.write_unlock_empty_in_module(calc_id);
        }
    }

    /// Same-module write unlock empty: release the lock without writing.
    fn write_unlock_empty_same_module<K: Solve<Ans>>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.get_calculation(idx).write_unlock_empty();
    }

    /// Commit all final converged answers from an iteratively-solved SCC
    /// to their respective Calculation cells using two-phase commit.
    ///
    /// Phase 1: Lock all member Calculation cells (in CalcId order).
    /// Phase 2: Write all answers via `write_unlock`.
    ///
    /// Called after the fixpoint iteration converges (or max iterations are
    /// reached).
    fn commit_final_answers(&self, scc: Scc) -> bool {
        // Collect Done members from node_state. BTreeMap iteration is already sorted by CalcId.
        let members: Vec<(
            CalcId,
            Arc<dyn Any + Send + Sync>,
            Option<Arc<ErrorCollector>>,
            Option<TraceSideEffects>,
        )> = scc
            .node_state
            .into_iter()
            .map(|(calc_id, node_state)| match node_state {
                SccNodeState::Done {
                    answer,
                    errors,
                    traces,
                } => (calc_id, answer, errors, traces),
                SccNodeState::Fresh
                | SccNodeState::InProgress
                | SccNodeState::HasPlaceholder(_) => {
                    panic!(
                        "commit_final_answers: node {} is {:?} at commit time",
                        calc_id, node_state,
                    );
                }
            })
            .collect();

        // Phase 1: Lock all cells.
        let mut guard = SccWriteLockGuard {
            solver: self,
            locked: Vec::new(),
        };
        for (calc_id, _, _, _) in &members {
            if self.write_lock_single(calc_id) {
                guard.locked.push(calc_id.dupe());
            }
        }

        // Phase 2: Write answers to locked cells + publish traces.
        // Cells that weren't locked are already Calculated (write_lock
        // returned false), so writing would be a no-op — skip them.
        let mut did_write_any = false;
        for (calc_id, answer, errors, traces) in members {
            if guard.locked.contains(&calc_id) {
                did_write_any |= self.write_unlock_single(calc_id, answer, errors, traces);
            }
        }
        guard.disarm();
        did_write_any
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
    ///
    /// If a merge occurred during the drive loop (detected via `merge_happened`),
    /// the demotion is deferred: instead of resetting all states to Fresh
    /// mid-loop (which would re-drive already-done members), we set the
    /// `demoted` flag so `iterative_resolve_scc` restarts the iteration.
    fn drive_all_iteration_members(&self) {
        while let Some(id) = self.stack().next_fresh_member() {
            self.drive_member(&id);
            // If the member is still Fresh after driving, the drive was a
            // no-op. This happens when solve_idx_erased encounters an
            // Evicted module (another thread ran Solutions and freed
            // Answers). The member's answer is already committed globally,
            // so remove it from iteration state to prevent infinite looping.
            if matches!(
                self.stack().get_iteration_node_state(&id),
                Some(SccNodeStateKind::Fresh)
            ) {
                self.stack().remove_from_iteration_state(&id);
            }
        }
        // If a merge happened during this drive loop, defer the demotion:
        // set the demoted flag so iterative_resolve_scc will restart the
        // iteration with all members Fresh. This ensures each member is
        // visited at most once per drive loop, bounding work to O(N).
        if self.stack().top_scc_merge_happened() {
            self.stack().set_top_scc_demoted(true);
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
    /// returns without committing if an ancestor SCC still exists on stack to
    /// own the merged SCC. If no ancestor SCC exists, the current driver
    /// continues with the updated identity to avoid orphaning.
    #[allow(clippy::mutable_key_type)]
    fn iterative_resolve_scc(&self, mut scc: Scc) {
        let mut scc_identity = scc.detected_at.dupe();
        let mut demotions: u32 = 0;
        let mut exceeded_max_iterations = false;

        // Initial cold start at iteration 1.
        scc.reset_for_cold_start();

        loop {
            // Push the SCC back onto the stack for this iteration.
            self.stack().push_scc(scc);

            // Drive all fresh members until none remain.
            self.drive_all_iteration_members();

            // Defensive guard: if the SCC stack is empty here, another driver
            // (via nested absorption) has already committed all members.
            // We investigated this path during cleanup but could not confirm
            // it is unreachable, so we keep the defensive return rather than
            // asserting.
            if self.stack().sccs_is_empty() {
                return;
            }

            // Absorption detection: if the top SCC's detected_at no longer
            // matches our identity, this SCC was absorbed during iteration.
            // Three cases:
            // 1. An ancestor SCC still exists on stack → that owner context
            //    will handle the merged SCC, so return.
            // 2. No ancestor, but the top SCC contains our original
            //    detected_at as a member → our SCC was merged into the top
            //    SCC (merge changes detected_at to min). Continue driving
            //    with the updated identity.
            // 3. No ancestor, and the top SCC does NOT contain our member
            //    → our SCC was committed by a nested driver, and a
            //    pre-existing SCC (e.g. Phase 0) remains on top. Return
            //    to avoid taking ownership of an unrelated SCC.
            if self.stack().top_scc_detected_at() != scc_identity {
                if self.stack().has_ancestor_scc() {
                    return;
                }
                if !self.stack().top_scc_contains_member(&scc_identity) {
                    // The top SCC doesn't contain our member. Our SCC was
                    // committed by a nested driver; the remaining SCC is
                    // unrelated.
                    return;
                }
                // The top SCC absorbed our SCC via merge. Continue driving
                // with the updated identity.
                scc_identity = self.stack().top_scc_detected_at();
            }

            // Pop the SCC to inspect its iteration outcome.
            scc = self.stack().pop_scc();

            // Check for demotion: if SCC membership expanded, restart at
            // iteration 1 with fresh state.
            let demoted = scc.iterative.demoted;
            let has_changed = scc.iterative.has_changed;

            if demoted {
                demotions += 1;
                check_demotion_limit(demotions, &scc_identity);
                scc.reset_for_cold_start();
                continue;
            }

            // Max iterations check: must happen after pop (so nodes are still
            // Done) but before advance (which resets nodes to Fresh).
            if scc.iteration() >= MAX_ITERATIONS {
                exceeded_max_iterations = true;
                break;
            }

            // Convergence check: if this is iteration >= 2 and no answers
            // changed, the fixpoint has converged.
            if scc.iteration() >= 2 && !has_changed {
                break;
            }

            scc.advance_to_next_warm_iteration();
        }

        // Report non-convergence errors only at the recursion break points —
        // the bindings where `InProgressWithPreviousAnswer` was hit, i.e., where
        // the cycle was broken by returning a previous-iteration answer. Other
        // non-converging members are downstream consequences and would produce
        // noisy duplicate errors.
        let non_convergent_members: Vec<(
            CalcId,
            Arc<dyn Any + Send + Sync>,
            Option<Arc<dyn Any + Send + Sync>>,
        )> = if exceeded_max_iterations {
            scc.node_state
                .iter()
                .filter_map(|(calc_id, node_state)| match node_state {
                    SccNodeState::Done { answer, .. }
                        if scc.iterative.recursion_breaks.contains(calc_id) =>
                    {
                        Some((
                            calc_id.dupe(),
                            answer.dupe(),
                            scc.iterative.previous_answers.get(calc_id).cloned(),
                        ))
                    }
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };

        let did_commit = self.commit_final_answers(scc);
        if did_commit {
            for (calc_id, answer, previous) in &non_convergent_members {
                let member_bindings = &calc_id.0;
                if self.is_same_module(calc_id) {
                    dispatch_anyidx!(
                        &calc_id.1,
                        self,
                        check_and_report_non_convergent_member,
                        answer,
                        previous.as_ref(),
                        member_bindings,
                        self.base_errors
                    );
                } else {
                    let cross_errors =
                        ErrorCollector::new(calc_id.0.module().dupe(), ErrorStyle::Delayed);
                    dispatch_anyidx!(
                        &calc_id.1,
                        self,
                        check_and_report_non_convergent_member,
                        answer,
                        previous.as_ref(),
                        member_bindings,
                        &cross_errors
                    );
                    self.base_errors.extend(cross_errors);
                }
            }
        }
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
    /// Note: The placeholder is recorded in SCC-local state (SccNodeState::HasPlaceholder),
    /// not in the Calculation cell. Each thread that hits the same cycle creates its
    /// own placeholder. The final answer IS written thread-locally via SccNodeState::Done
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
        self.stack().set_iteration_placeholder(current, rec);
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
        let _ = binding; // Used only during phase 0 discovery, whose results are discarded.
        self.solver().fresh_recursive(self.uniques)
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
        match self.is_subset_eq_with_reason(got, want) {
            Ok(()) => true,
            Err(error) => {
                self.solver().error(got, want, errors, loc, tcc, error);
                false
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

    /// Report a `NonConvergentRecursion` error on a single SCC member whose
    /// answer changed in the final iteration. Called via `dispatch_anyidx!`
    /// so that the concrete `K` (and therefore `K::Answer`) is known.
    ///
    /// `member_bindings` and `member_errors` must come from the member's own
    /// module, not necessarily `self`. SCCs can span modules, so `self.bindings()`
    /// and `self.base_errors` are only correct for same-module members.
    ///
    /// The message distinguishes `TypeInfo` answers ("inferred type") from
    /// other answer kinds ("inferred result") for clarity in diagnostics.
    ///
    /// If `PYREFLY_FIXPOINT_DETAILS` is set (to any non-empty value besides `0`),
    /// append internal debug details for bug reports (key/binding and both
    /// previous/current answers in Debug format).
    fn check_and_report_non_convergent_member<K: Solve<Ans>>(
        &self,
        idx: Idx<K>,
        current: &Arc<dyn Any + Send + Sync>,
        previous: Option<&Arc<dyn Any + Send + Sync>>,
        member_bindings: &Bindings,
        member_errors: &ErrorCollector,
    ) where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        K::Answer: Debug,
        K::Value: Debug,
    {
        // Only report if the answer actually changed from the previous iteration.
        if let Some(prev) = previous
            && self.answers_equal_typed::<K>(idx, prev, current)
        {
            return;
        }
        let typed_answer = current
            .downcast_ref::<Arc<K::Answer>>()
            .expect("check_and_report_non_convergent_member: type mismatch");
        // TypeInfo answers represent inferred types; other answer kinds are
        // internal results (class fields, metadata, etc.).
        let noun = if current.downcast_ref::<Arc<TypeInfo>>().is_some() {
            "type"
        } else {
            "result"
        };
        let mut messages = vec1![format!(
            "Fixpoint iteration did not converge. \
             Inferred {} `{}`. Adding annotations may help.",
            noun, typed_answer,
        )];
        // If PYREFLY_FIXPOINT_DETAILS=1 is set, we output much more detailed information useful
        // for explaining or debugging nonconvergence in terms of Pyrefly internals.
        if Self::fixpoint_details_enabled() {
            let binding = member_bindings.get(idx);
            let previous_debug = previous
                .map(|prev| {
                    let prev_typed = prev.downcast_ref::<Arc<K::Answer>>().expect(
                        "check_and_report_non_convergent_member: previous answer type mismatch",
                    );
                    format!("{prev_typed:?}")
                })
                .unwrap_or_else(|| "<none>".to_owned());
            messages.push(format!(
                "[PYREFLY_FIXPOINT_DETAILS] key={:?} key_idx={idx:?}",
                K::to_anyidx(idx),
            ));
            messages.push(format!(
                "[PYREFLY_FIXPOINT_DETAILS] module={} path={}",
                member_bindings.module().name(),
                member_bindings.module().path(),
            ));
            messages.push(format!("[PYREFLY_FIXPOINT_DETAILS] binding={binding:?}",));
            messages.push(format!(
                "[PYREFLY_FIXPOINT_DETAILS] answer_type={}",
                std::any::type_name::<K::Answer>(),
            ));
            messages.push(format!(
                "[PYREFLY_FIXPOINT_DETAILS] previous={previous_debug}",
            ));
            messages.push(format!(
                "[PYREFLY_FIXPOINT_DETAILS] current={typed_answer:?}",
            ));
        }
        let range = K::range_with(idx, member_bindings);
        member_errors.add(
            range,
            ErrorInfo::Kind(ErrorKind::NonConvergentRecursion),
            messages,
        );
    }
}

#[cfg(test)]
mod scc_tests {
    use super::*;

    /// Create a dummy `SccNodeState::Done` for testing.
    fn done_for_test() -> SccNodeState {
        SccNodeState::Done {
            answer: Arc::new(()) as Arc<dyn Any + Send + Sync>,
            errors: None,
            traces: None,
        }
    }

    /// Helper to create a test Scc with given parameters.
    ///
    /// This bypasses the normal Scc::new constructor to allow direct construction
    /// for testing merge logic.
    ///
    /// Note: top_pos_exclusive is set to bottom_pos_inclusive + node_state.len()
    /// which approximates the segment span. In production, top_pos_exclusive may
    /// differ from bottom_pos_inclusive + participant count due to duplicate
    /// CalcIds during cycle breaking.
    #[allow(clippy::mutable_key_type)]
    fn make_test_scc(
        node_state: BTreeMap<CalcId, SccNodeState>,
        detected_at: CalcId,
        bottom_pos_inclusive: usize,
    ) -> Scc {
        let top_pos_exclusive = bottom_pos_inclusive + node_state.len();
        Scc {
            node_state,
            detected_at,
            bottom_pos_inclusive,
            top_pos_exclusive,
            iterative: SccIterationState {
                iteration: 0,
                previous_answers: BTreeMap::new(),
                demoted: false,
                has_changed: false,
                merge_happened: false,
                recursion_breaks: BTreeSet::new(),
            },
        }
    }

    /// Helper to create a CalcStack for testing.
    fn make_calc_stack(entries: &[CalcId]) -> CalcStack {
        let stack = CalcStack::new();
        for entry in entries {
            stack.push_for_test(entry.dupe());
        }
        stack
    }

    /// Helper to create node_state map with all nodes Fresh.
    #[allow(clippy::mutable_key_type)]
    fn fresh_nodes(ids: &[CalcId]) -> BTreeMap<CalcId, SccNodeState> {
        ids.iter()
            .map(|id| (id.dupe(), SccNodeState::Fresh))
            .collect()
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
        let calc_stack = CalcStack::new();
        assert!(calc_stack.current_cycle().is_none());
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
    fn test_merge_many_preserves_members() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);

        let scc1 = make_test_scc(
            fresh_nodes(&[a.dupe(), b.dupe()]),
            a.dupe(),
            0, // bottom_pos_inclusive
        );
        let scc2 = make_test_scc(
            fresh_nodes(&[c.dupe(), d.dupe()]),
            c.dupe(),
            2, // bottom_pos_inclusive
        );

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // All nodes should be present
        assert_eq!(merged.node_state.len(), 4);

        // bottom_pos_inclusive should be the minimum (0)
        assert_eq!(merged.bottom_pos_inclusive, 0);
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn test_merge_many_takes_most_advanced_state() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);

        // SCC1 has M0 as Done, M1 as Fresh
        let mut scc1_state = BTreeMap::new();
        scc1_state.insert(a.dupe(), done_for_test());
        scc1_state.insert(b.dupe(), SccNodeState::Fresh);
        let scc1 = make_test_scc(scc1_state, a.dupe(), 0);

        // SCC2 has M0 as Fresh, M1 as InProgress
        let mut scc2_state = BTreeMap::new();
        scc2_state.insert(a.dupe(), SccNodeState::Fresh);
        scc2_state.insert(b.dupe(), SccNodeState::InProgress);
        let scc2 = make_test_scc(scc2_state, a.dupe(), 0);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should take the most advanced state for each node
        assert!(matches!(
            merged.node_state.get(&a),
            Some(SccNodeState::Done { .. })
        ));
        assert!(matches!(
            merged.node_state.get(&b),
            Some(SccNodeState::InProgress)
        ));
    }

    #[test]
    fn test_merge_many_keeps_smallest_detected_at() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        // SCC1 detected at M1
        let scc1 = make_test_scc(fresh_nodes(&[a.dupe(), b.dupe()]), b.dupe(), 0);
        // SCC2 detected at M2
        let scc2 = make_test_scc(fresh_nodes(&[a.dupe(), c.dupe()]), c.dupe(), 0);
        // When merging with M0 as the new detected_at, should keep M0 (smallest)
        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());
        assert_eq!(merged.detected_at, a);
    }

    #[test]
    fn test_merge_many_keeps_minimum_bottom_pos_inclusive() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        // SCC1 with bottom_pos_inclusive = 5
        let scc1 = make_test_scc(fresh_nodes(&[a.dupe(), b.dupe()]), a.dupe(), 5);
        // SCC2 with bottom_pos_inclusive = 2
        let scc2 = make_test_scc(fresh_nodes(&[c.dupe()]), c.dupe(), 2);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should keep the minimum bottom_pos_inclusive
        assert_eq!(merged.bottom_pos_inclusive, 2);
    }

    #[test]
    #[should_panic(expected = "pending_completed_scc was not taken before a new SCC completed")]
    fn test_pending_completed_scc_must_be_taken_before_overwrite() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);

        // Stack has one live frame; this makes an SCC with bottom_pos_inclusive=0
        // eligible for completion in on_calculation_finished.
        let calc_stack = make_calc_stack(&[a.dupe()]);

        // Active top SCC that will complete.
        let active_scc = make_test_scc(fresh_nodes(&[a.dupe()]), a.dupe(), 0);
        calc_stack.scc_stack.borrow_mut().push(active_scc);

        // Simulate a bug where a previous completed SCC wasn't taken yet.
        let already_pending = make_test_scc(fresh_nodes(&[b.dupe()]), b.dupe(), 0);
        *calc_stack.pending_completed_scc.borrow_mut() = Some(already_pending);

        let answer: Arc<dyn Any + Send + Sync> = Arc::new(Arc::new(42usize));
        let _ = calc_stack.on_calculation_finished(&a, answer, None, None);
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
        let stack = CalcStack::new();

        // 3. Push the same calculation.
        // This should NOT panic.
        let action = stack.push(calc_id);

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
        // SCC causes the SCCs to merge and the result to have merge_happened = true.
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
        //   - Merged SCC has iterative.merge_happened == true
        //   - Merged SCC contains members from both original SCCs {A, B, D, E}
        //   - push returns Calculate (since new members are Fresh)
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);

        // Build the iterative CalcStack with [A, B, C, D, E].
        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe()]);

        // Manually construct SCC0 with iterative state (iteration 2).
        let scc0 = {
            let mut node_state = BTreeMap::new();
            node_state.insert(a.dupe(), SccNodeState::Fresh);
            node_state.insert(b.dupe(), SccNodeState::Fresh);
            Scc {
                node_state,
                detected_at: a.dupe(),
                bottom_pos_inclusive: 0,
                top_pos_exclusive: 2,
                iterative: SccIterationState {
                    iteration: 2,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                    merge_happened: false,
                    recursion_breaks: BTreeSet::new(),
                },
            }
        };

        // Manually construct SCC1 with iterative state (iteration 1).
        let scc1 = {
            let mut node_state = BTreeMap::new();
            node_state.insert(d.dupe(), SccNodeState::Fresh);
            node_state.insert(e.dupe(), SccNodeState::Fresh);
            Scc {
                node_state,
                detected_at: d.dupe(),
                bottom_pos_inclusive: 3,
                top_pos_exclusive: 5,
                iterative: SccIterationState {
                    iteration: 1,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                    merge_happened: false,
                    recursion_breaks: BTreeSet::new(),
                },
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
        let action = calc_stack.push(a.dupe());

        // After merge, there should be exactly one SCC.
        let scc_stack = calc_stack.borrow_scc_stack();
        assert_eq!(
            scc_stack.len(),
            1,
            "SCCs should have merged into one after membership back-edge"
        );

        let merged = &scc_stack[0];

        // The merged SCC must have merge_happened = true (demotion is deferred
        // until after drive_all_iteration_members completes).
        assert!(
            merged.iterative.merge_happened,
            "merged SCC should have merge_happened = true after membership back-edge merge"
        );
        assert!(
            !merged.iterative.demoted,
            "merged SCC should have demoted = false (demotion is deferred)"
        );

        // Iteration should be preserved from self (the more advanced SCC).
        assert_eq!(
            merged.iterative.iteration, 2,
            "merged SCC iteration should be preserved from self"
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
        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe(), d.dupe(), e.dupe()]);

        // Manually construct SCC_outer (ancestor) with iterative state at iteration 2.
        let scc_outer = {
            let mut node_state = BTreeMap::new();
            node_state.insert(a.dupe(), SccNodeState::Fresh);
            node_state.insert(b.dupe(), SccNodeState::Fresh);
            Scc {
                node_state,
                detected_at: a.dupe(),
                bottom_pos_inclusive: 0,
                top_pos_exclusive: 2,
                iterative: SccIterationState {
                    iteration: 2,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                    merge_happened: false,
                    recursion_breaks: BTreeSet::new(),
                },
            }
        };

        // Manually construct SCC_inner (top) with iterative state at iteration 1.
        let scc_inner = {
            let mut node_state = BTreeMap::new();
            node_state.insert(d.dupe(), SccNodeState::Fresh);
            node_state.insert(e.dupe(), SccNodeState::Fresh);
            Scc {
                node_state,
                detected_at: d.dupe(),
                bottom_pos_inclusive: 3,
                top_pos_exclusive: 5,
                iterative: SccIterationState {
                    iteration: 1,
                    previous_answers: BTreeMap::new(),
                    demoted: false,
                    has_changed: false,
                    merge_happened: false,
                    recursion_breaks: BTreeSet::new(),
                },
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
        let _action = calc_stack.push(a.dupe());

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

        // Verify the merged SCC has merge_happened = true (confirming the
        // merge actually happened; demotion is deferred to drive_all_iteration_members).
        let scc_stack = calc_stack.borrow_scc_stack();
        let merged = &scc_stack[0];
        assert!(
            merged.iterative.merge_happened,
            "merged SCC should have merge_happened = true"
        );
        assert!(
            !merged.iterative.demoted,
            "merged SCC should have demoted = false (demotion is deferred)"
        );
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
