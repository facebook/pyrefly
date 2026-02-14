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
use pyrefly_types::type_alias::TypeAlias;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_alias::TypeAliasRef;
use pyrefly_types::types::Union;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::recurser::Guard;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::VisitMut;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
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
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::class_metadata::DjangoReverseRelationIndex;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::Binding;
use crate::binding::binding::Exported;
use crate::binding::binding::KeyDjangoRelations;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTypeAlias;
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
        write!(f, "CalcId({}, {:?})", self.0.module().name(), self.1)
    }
}

impl Display for CalcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CalcId({}, {})",
            self.0.module().name(),
            self.1.display_with(&self.0),
        )
    }
}

impl PartialEq for CalcId {
    fn eq(&self, other: &Self) -> bool {
        (self.0.module().name(), &self.1) == (other.0.module().name(), &other.1)
    }
}

impl Eq for CalcId {}

impl Ord for CalcId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.1.cmp(&other.1) {
            Ordering::Equal => self.0.module().name().cmp(&other.0.module().name()),
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

        use crate::binding::binding::Key;

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
}

impl CalcStack {
    pub fn new() -> Self {
        Self {
            stack: RefCell::new(Vec::new()),
            scc_stack: RefCell::new(Vec::new()),
            position_of: RefCell::new(FxHashMap::default()),
            pending_completed_sccs: RefCell::new(Vec::new()),
        }
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
        match self.pre_calculate_state(&current) {
            SccState::NotInScc | SccState::RevisitingInProgress => {
                match calculation.propose_calculation() {
                    ProposalResult::Calculated(v) => BindingAction::Calculated(v),
                    ProposalResult::CycleDetected => {
                        let current_cycle = self.current_cycle().unwrap();
                        match self.on_scc_detected(current_cycle) {
                            SccDetectedResult::BreakHere => BindingAction::Unwind,
                            SccDetectedResult::Continue => BindingAction::Calculate,
                        }
                    }
                    ProposalResult::Calculatable => BindingAction::Calculate,
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

        if let Some(first_idx) = first_merge_idx {
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
        }
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
    django_reverse_relations: RefCell<Option<Arc<DjangoReverseRelationIndex>>>,
    /// Configuration for recursion depth limiting. None means disabled.
    recursion_limit_config: Option<RecursionLimitConfig>,
}

impl ThreadState {
    pub fn new(recursion_limit_config: Option<RecursionLimitConfig>) -> Self {
        Self {
            stack: CalcStack::new(),
            debug: RefCell::new(false),
            django_reverse_relations: RefCell::new(None),
            recursion_limit_config,
        }
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

    pub fn django_reverse_relations_index(&self) -> Arc<DjangoReverseRelationIndex> {
        if let Some(index) = self.thread_state.django_reverse_relations.borrow().clone() {
            return index;
        }

        let mut combined: SmallMap<Class, ClassSynthesizedFields> = SmallMap::new();
        for module in self.answers.modules() {
            let Some(index) =
                self.answers
                    .get(module, None, &KeyDjangoRelations, self.thread_state)
            else {
                continue;
            };
            for (class, fields) in index.iter() {
                match combined.get_mut(class) {
                    Some(existing) => {
                        let merged = existing.clone().combine(fields.clone());
                        *existing = merged;
                    }
                    None => {
                        combined.insert(class.clone(), fields.clone());
                    }
                }
            }
        }

        let index = Arc::new(DjangoReverseRelationIndex::new(combined));
        *self.thread_state.django_reverse_relations.borrow_mut() = Some(index.clone());
        index
    }

    fn cycles(&self) -> &Cycles {
        &self.thread_state.cycles
    fn cycles(&self) -> &Cycles {
        &self.thread_state.cycles
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
        };
        for scc in self.stack().pop_and_drain_completed_sccs() {
            self.batch_commit_scc(scc);
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
        let binding = self.bindings().get(idx);
        // Note that we intentionally do not pass in the key when solving the binding,
        // as the result of a binding should not depend on the key it was bound to.
        // We use the range for error reporting.
        let range = self.bindings().idx_to_key(idx).range();

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
            // SCC path: store in NodeState::Done, defer Calculation write to batch commit.
            //
            // If this is a break_at node (has a placeholder Var), we must finalize
            // the recursive answer now, before storing. Finalization mutates solver
            // state (force_var) and must happen during computation, not at batch commit.
            let answer = if let Some(var) = self.stack().get_scc_placeholder_var(&current) {
                self.finalize_recursive_answer::<K>(idx, var, raw_answer, &local_errors)
            } else {
                raw_answer
            };
            let answer_erased: Arc<dyn Any + Send + Sync> = Arc::new(answer.dupe());
            let canonical_erased = self.stack().on_calculation_finished(
                &current,
                Some(answer_erased),
                Some(Arc::new(local_errors)),
            );
            // Use the canonical answer from thread-local state, mirroring how
            // Calculation::record_value returns the first-written answer.
            match canonical_erased {
                Some(erased) => Arc::unwrap_or_clone(
                    erased
                        .downcast::<Arc<K::Answer>>()
                        .expect("on_calculation_finished canonical answer downcast failed"),
                ),
                None => answer,
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
        if bindings.module().name() != self.bindings().module().name() {
            // Cross-module: delegate to the LookupAnswer trait to route
            // the commit to the correct module's Answers.
            self.answers.commit_to_module(calc_id, answer, errors);
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
        // Validate invariant: no Fresh or InProgress nodes should remain.
        debug_assert!(
            completed_scc.node_state.values().all(|state| matches!(
                state,
                NodeState::Done { .. } | NodeState::HasPlaceholder(_)
            )),
            "batch_commit_scc: SCC has participants that are not Done or HasPlaceholder: {:?}",
            completed_scc
                .node_state
                .iter()
                .filter(|(_, s)| matches!(s, NodeState::Fresh | NodeState::InProgress))
                .map(|(id, s)| format!("{}: {:?}", id, s))
                .collect::<Vec<_>>(),
        );
        for (calc_id, node_state) in completed_scc.node_state {
            if let NodeState::Done {
                answer: Some(answer),
                errors,
            } = node_state
            {
                self.commit_single_result(calc_id, answer, errors);
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
        let range = self.bindings().idx_to_key(idx).range();
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
        let range = self.bindings().idx_to_key(idx).range();
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
        let range = key.range();
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
            Binding::TypeAlias {
                name,
                tparams,
                key_type_alias,
                range: _,
            } => self.solver().fresh_alias_recursive(
                self.uniques,
                TypeAliasRef {
                    name: name.clone(),
                    args: None,
                    module_name: self.module().name(),
                    module_path: self.module().path().clone(),
                    index: self.bindings().idx_to_key(*key_type_alias).0,
                },
                self.create_type_alias_params_recursive(tparams),
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
        let calc_stack = CalcStack::new();
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
}
