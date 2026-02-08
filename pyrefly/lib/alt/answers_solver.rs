/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use dupe::IterDupedExt;
use itertools::Either;
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
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTypeAlias;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::config::base::RecursionLimitConfig;
use crate::config::base::RecursionOverflowHandler;
use crate::config::error_kind::ErrorKind;
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
pub struct CalcStack(RefCell<Vec<CalcId>>);

impl CalcStack {
    pub fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    fn push(&self, current: CalcId) {
        self.0.borrow_mut().push(current);
    }

    fn pop(&self) -> Option<CalcId> {
        self.0.borrow_mut().pop()
    }

    pub fn peek(&self) -> Option<CalcId> {
        self.0.borrow().last().cloned()
    }

    pub fn into_vec(&self) -> Vec<CalcId> {
        self.0.borrow().clone()
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    /// Return the current stack depth (number of entries on the stack).
    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }

    /// Return the current cycle, if we are at a (module, idx) that we've already seen in this thread.
    ///
    /// The answer will have the form
    /// - if there is no cycle, `None`
    /// - if there is a cycle, `Some(vec![(m0, i0), (m2, i2)...])`
    ///   where the order of (module, idx) pairs is recency (so starting with current
    ///   module and idx, and ending with the oldest).
    pub fn current_cycle(&self) -> Option<Vec1<CalcId>> {
        let stack = self.0.borrow();
        let mut rev_stack = stack.iter().rev();
        let current = rev_stack.next()?;
        let mut cycle = Vec1::with_capacity(current.dupe(), rev_stack.len());
        for c in rev_stack {
            if c == current {
                return Some(cycle);
            }
            cycle.push(c.dupe());
        }
        None
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
/// The variants are ordered by "advancement" (Fresh < InProgress < HasPlaceholder < Done)
/// so that when merging SCCs we can use `max()` to keep the more advanced state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NodeState {
    /// Node hasn't been processed yet as part of SCC handling.
    Fresh,
    /// Node is currently being processed (on the Rust call stack).
    InProgress,
    /// This is a break_at node: we've recorded a placeholder in Calculation
    /// but haven't computed the real answer yet.
    HasPlaceholder,
    /// Node's calculation has completed.
    Done,
}

/// The state of a target node when revisiting a previous SCC.
///
/// When we read back into a previous (non-top) SCC, we need to know the
/// target node's state to determine how to handle it after merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RevisitingTargetState {
    /// Node completed within the SCC, has preliminary answer.
    /// No new break_at needed - just merge and return the answer.
    Done,
    /// Node is still being computed (on the Rust call stack).
    /// May need new break_at via normal cycle detection after merge.
    InProgress,
    /// Node has recorded a placeholder but hasn't computed the real answer yet.
    /// No new break_at needed - merge and break here.
    HasPlaceholder,
    /// Node is a break point for that SCC (in the break_at set but hasn't
    /// recorded a placeholder yet).
    /// No new break_at needed - merge and break here.
    BreakAt,
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
    /// Read back into a PREVIOUS SCC (not the top of the stack).
    /// This occurs when the current computation reads a node that belongs to
    /// an SCC lower in the SCC stack. Requires merging all intervening nodes
    /// and SCCs before proceeding.
    RevisitingPreviousScc {
        /// Stable identifier for the target SCC (the detected_at field of that SCC).
        detected_at_of_scc: CalcId,
        /// The state of the target node within that SCC.
        target_state: RevisitingTargetState,
    },
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

enum SccDetectedResult {
    /// Break immediately at the idx where we detected the SCC, so that we
    /// unwind back to the same idx.
    BreakHere,
    /// Continue recursing until we hit some other idx that is the minimal `break_at` idx.
    Continue,
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
    node_state: HashMap<CalcId, NodeState>,
    /// Where we detected the SCC (for debugging only)
    detected_at: CalcId,
    /// The minimum CalcStack depth of any participant - specifically, the
    /// position of the anchor (minimal CalcId) on the CalcStack when the SCC
    /// was created. This is well-defined because the anchor is the last
    /// participant to finish during unwinding.
    /// Used as a fast filter to skip SCCs that can't possibly overlap with
    /// a newly detected cycle.
    min_stack_depth: usize,
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
        let node_state: HashMap<CalcId, NodeState> =
            raw.iter().duped().map(|c| (c, NodeState::Fresh)).collect();

        let mut break_at_set = BTreeSet::new();
        break_at_set.insert(break_at.dupe());

        // Find the anchor's position on the CalcStack.
        // The anchor (minimal CalcId) is where the cycle will complete during unwinding.
        let min_stack_depth = calc_stack_vec
            .iter()
            .position(|c| c == break_at)
            .unwrap_or(0);

        Scc {
            break_at: break_at_set,
            node_state,
            detected_at,
            min_stack_depth,
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
                NodeState::HasPlaceholder => {
                    // Already has placeholder, return it
                    SccState::HasPlaceholder
                }
                NodeState::Done => {
                    // Node completed within this SCC - preliminary answer should exist.
                    SccState::RevisitingDone
                }
            }
        } else {
            SccState::NotInScc
        }
    }

    /// Track that a calculation has finished, marking it as Done.
    fn on_calculation_finished(&mut self, current: &CalcId) {
        if let Some(state) = self.node_state.get_mut(current) {
            *state = NodeState::Done;
        }
    }

    /// Track that a placeholder has been recorded for a break_at node.
    fn on_placeholder_recorded(&mut self, current: &CalcId) {
        if let Some(state) = self.node_state.get_mut(current) {
            *state = NodeState::HasPlaceholder;
        }
    }

    /// Check if the SCC is complete (all participants are Done).
    fn is_complete(&self) -> bool {
        self.node_state
            .values()
            .all(|state| *state == NodeState::Done)
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
                .and_modify(|existing| *existing = (*existing).max(v))
                .or_insert(v);
        }
        // Keep the smallest detected_at for consistency/determinism
        self.detected_at = self.detected_at.min(other.detected_at);
        // Keep the minimum stack depth (the earliest anchor position)
        self.min_stack_depth = self.min_stack_depth.min(other.min_stack_depth);
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

/// Represent the current thread's SCCs, which form a stack
/// because we can encounter a new one while solving another.
pub struct Sccs(RefCell<Vec<Scc>>);

impl Sccs {
    pub fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    /// Check if an existing SCC overlaps with a newly detected cycle.
    ///
    /// Uses a fast filter based on min_stack_depth + cardinality, then verifies
    /// by checking if any CalcStack entry in the cycle range is a participant
    /// in the existing SCC.
    ///
    /// This is extracted for testability - overlap detection can be tested
    /// independently of the full merge logic.
    #[cfg_attr(test, allow(dead_code))]
    fn check_overlap(
        existing: &Scc,
        cycle_start_pos: usize,
        stack_depth: usize,
        calc_stack_vec: &[CalcId],
    ) -> bool {
        // Fast filter: if existing SCC's upper bound < new cycle's min, definitely no overlap
        // Upper bound = min_stack_depth + cardinality (number of participants)
        let existing_max_bound = existing.min_stack_depth + existing.node_state.len();
        if existing_max_bound <= cycle_start_pos {
            return false;
        }

        // Check if any CalcStack entry in the cycle range is in this SCC
        (cycle_start_pos..stack_depth).any(|pos| {
            calc_stack_vec
                .get(pos)
                .map(|calc_id| existing.node_state.contains_key(calc_id))
                .unwrap_or(false)
        })
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
    fn on_scc_detected(&self, raw: Vec1<CalcId>, calc_stack: &CalcStack) -> SccDetectedResult {
        let stack_depth = calc_stack.len();
        let calc_stack_vec = calc_stack.into_vec();

        // Create the new SCC - this computes min_stack_depth as the anchor's position
        let new_scc = Scc::new(raw, &calc_stack_vec);
        let detected_at = new_scc.detected_at.dupe();
        let cycle_start_pos = new_scc.min_stack_depth;

        // Check for overlapping SCCs and merge if needed
        let mut scc_stack = self.0.borrow_mut();

        // Find the first (oldest) SCC that overlaps with the new cycle.
        // Use min_stack_depth + cardinality as a bound for the SCC's max position.
        // If this bound < cycle_start_pos, the SCC is entirely below the new cycle.
        // Due to LIFO ordering, once we find one overlapping SCC, all subsequent ones
        // on the stack must also overlap.
        let mut first_merge_idx: Option<usize> = None;

        for (i, existing) in scc_stack.iter().enumerate() {
            if Self::check_overlap(existing, cycle_start_pos, stack_depth, &calc_stack_vec) {
                first_merge_idx = Some(i);
                break; // All subsequent SCCs will also overlap
            }
        }

        if let Some(first_idx) = first_merge_idx {
            // Merge all SCCs from first_idx to end, plus the new SCC
            let sccs_from_stack: Vec<Scc> = scc_stack.drain(first_idx..).collect();
            let sccs_to_merge = Vec1::from_vec_push(sccs_from_stack, new_scc);

            // Use the helper method to merge SCCs
            let merged_scc = Scc::merge_many(sccs_to_merge, detected_at.dupe());

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
    fn pre_calculate_state(&self, current: &CalcId) -> SccState {
        let mut stack = self.0.borrow_mut();

        // Check from top to bottom (rev gives us index 0 = top)
        for (rev_idx, scc) in stack.iter_mut().rev().enumerate() {
            let is_top_scc = rev_idx == 0;
            let state = scc.pre_calculate_state(current);

            match state {
                SccState::NotInScc => continue,
                // Only return the raw scc state when this is the top SCC
                _ if is_top_scc => return state,
                // Otherwise, remap it to a suitable RevisitingPreviousScc value, because
                // we are going to need to merge SCCs when this happens.
                SccState::RevisitingInProgress | SccState::Participant => {
                    return SccState::RevisitingPreviousScc {
                        detected_at_of_scc: scc.detected_at(),
                        target_state: RevisitingTargetState::InProgress,
                    };
                }
                SccState::RevisitingDone => {
                    return SccState::RevisitingPreviousScc {
                        detected_at_of_scc: scc.detected_at(),
                        target_state: RevisitingTargetState::Done,
                    };
                }
                SccState::HasPlaceholder => {
                    return SccState::RevisitingPreviousScc {
                        detected_at_of_scc: scc.detected_at(),
                        target_state: RevisitingTargetState::HasPlaceholder,
                    };
                }
                SccState::BreakAt => {
                    return SccState::RevisitingPreviousScc {
                        detected_at_of_scc: scc.detected_at(),
                        target_state: RevisitingTargetState::BreakAt,
                    };
                }
                // RevisitingPreviousScc shouldn't be returned by Scc::pre_calculate_state
                SccState::RevisitingPreviousScc { .. } => unreachable!(),
            }
        }
        SccState::NotInScc
    }

    /// Handle the completion of a calculation. This might involve progress on
    /// the remaining participants of one or more SCCs.
    ///
    /// Return `true` if there are active SCCs after finishing this calculation,
    /// `false` if there are not.
    fn on_calculation_finished(&self, current: &CalcId) -> bool {
        let mut stack = self.0.borrow_mut();
        for scc in stack.iter_mut() {
            scc.on_calculation_finished(current);
        }
        while let Some(scc) = stack.last() {
            if scc.is_complete() {
                stack.pop();
            } else {
                break;
            }
        }
        // Do we still have active SCCs?
        !stack.is_empty()
    }

    /// Track that a placeholder has been recorded for a break_at node.
    fn on_placeholder_recorded(&self, current: &CalcId) {
        let mut stack = self.0.borrow_mut();
        for scc in stack.iter_mut() {
            scc.on_placeholder_recorded(current);
        }
    }

    /// Merge all SCCs from the target SCC to the top of the stack, and add
    /// any free-floating CalcStack nodes between the target SCC's min_stack_depth
    /// and the current stack position.
    ///
    /// This is called when we detect a read into a previous (non-top) SCC via
    /// `RevisitingPreviousScc`. After this call, the SCC stack will have one
    /// merged SCC at the top containing all participants from the merged SCCs
    /// plus any free-floating nodes from the CalcStack.
    ///
    /// The oldest previously-known Scc we should merge is identified based on its
    /// `detected_at`; this has the potentially-useful property of being a valid
    /// identifier of the merged Scc *after* the merge, since we always use the
    /// very first cycle detected for `detected_at`.
    #[allow(clippy::mutable_key_type)]
    fn merge_sccs(&self, detected_at_of_scc: &CalcId, calc_stack: &CalcStack) {
        let calc_stack_vec = calc_stack.into_vec();
        let mut stack = self.0.borrow_mut();

        // Pop SCCs until we find the target component (identified by detected_at).
        //
        // Push them to a vec we will merge; in addition, when we reach the last component
        // use it to determine how much of the CalcStack needs to be merged in order
        // to ensure bindings that weren't yet part of a known SCC are included.
        let mut sccs_to_merge: Vec<Scc> = Vec::new();
        let mut target_min_stack_depth: Option<usize> = None;
        while let Some(scc) = stack.pop() {
            let is_target = scc.detected_at() == *detected_at_of_scc;
            if is_target {
                target_min_stack_depth = Some(scc.min_stack_depth);
            }
            sccs_to_merge.push(scc);
            if is_target {
                break;
            }
        }
        let min_depth = target_min_stack_depth
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");
        let sccs_to_merge = Vec1::try_from_vec(sccs_to_merge)
            .expect("Target SCC not found during merge - this indicates a bug in SCC tracking");

        // Perform the merge, then add any free-floating bindings that weren't previously part
        // of a known SCC.
        let mut merged = Scc::merge_many(sccs_to_merge, detected_at_of_scc.dupe());
        for calc_id in calc_stack_vec.iter().skip(min_depth) {
            merged
                .node_state
                .entry(calc_id.dupe())
                .or_insert(NodeState::Fresh);
        }

        stack.push(merged);
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
    sccs: Sccs,
    stack: CalcStack,
    /// For debugging only: thread-global that allows us to control debug logging across components.
    debug: RefCell<bool>,
    /// Configuration for recursion depth limiting. None means disabled.
    recursion_limit_config: Option<RecursionLimitConfig>,
}

impl ThreadState {
    pub fn new(recursion_limit_config: Option<RecursionLimitConfig>) -> Self {
        Self {
            sccs: Sccs::new(),
            stack: CalcStack::new(),
            debug: RefCell::new(false),
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

    fn sccs(&self) -> &Sccs {
        &self.thread_state.sccs
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
            self.thread_state.sccs.is_empty(),
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
        self.stack().push(current.dupe());

        // Check depth limit before any calculation
        if let Some(config) = self.recursion_limit_config()
            && self.stack().len() > config.limit as usize
        {
            let result = self.handle_depth_overflow(&current, idx, calculation, config);
            self.stack().pop();
            return result;
        }

        let result = match self.sccs().pre_calculate_state(&current) {
            SccState::NotInScc | SccState::RevisitingInProgress | SccState::RevisitingDone => {
                match calculation.propose_calculation() {
                    ProposalResult::Calculated(v) => v,
                    ProposalResult::CycleBroken(r) => Arc::new(K::promote_recursive(self.heap, r)),
                    ProposalResult::CycleDetected => {
                        let current_cycle = self.stack().current_cycle().unwrap();
                        match self.sccs().on_scc_detected(current_cycle, self.stack()) {
                            SccDetectedResult::BreakHere => self
                                .attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                                .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r))),
                            SccDetectedResult::Continue => {
                                self.calculate_and_record_answer(current, idx, calculation)
                            }
                        }
                    }
                    ProposalResult::Calculatable => {
                        self.calculate_and_record_answer(current, idx, calculation)
                    }
                }
            }
            SccState::BreakAt => {
                // Begin unwinding the cycle using a recursive placeholder
                self.attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                    .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r)))
            }
            SccState::HasPlaceholder => {
                // We already recorded a placeholder for this node; return it.
                // The Calculation should have a CycleBroken result.
                match calculation.propose_calculation() {
                    ProposalResult::CycleBroken(r) => Arc::new(K::promote_recursive(self.heap, r)),
                    ProposalResult::Calculated(v) => v,
                    ProposalResult::CycleDetected | ProposalResult::Calculatable => {
                        unreachable!(
                            "HasPlaceholder node must have CycleBroken or Calculated result"
                        )
                    }
                }
            }
            SccState::RevisitingPreviousScc {
                detected_at_of_scc,
                target_state,
            } => {
                // Current node was found in a previous (non-top) SCC.
                // Handle based on the target node's state in that SCC.
                match target_state {
                    // Node was already a break-at index. Merge SCCs without any new break_at, and break here
                    RevisitingTargetState::BreakAt => {
                        self.sccs().merge_sccs(&detected_at_of_scc, self.stack());
                        self.attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                            .unwrap_or_else(|r| Arc::new(K::promote_recursive(self.heap, r)))
                    }
                    // Node already has a placeholder. Merge SCCs and return it.
                    RevisitingTargetState::HasPlaceholder => {
                        self.sccs().merge_sccs(&detected_at_of_scc, self.stack());
                        match calculation.propose_calculation() {
                            ProposalResult::CycleBroken(r) => {
                                Arc::new(K::promote_recursive(self.heap, r))
                            }
                            ProposalResult::Calculated(v) => v,
                            ProposalResult::CycleDetected | ProposalResult::Calculatable => {
                                unreachable!(
                                    "HasPlaceholder node in previous SCC must have CycleBroken or Calculated result"
                                )
                            }
                        }
                    }
                    // Node already completed within the SCC. Merge SCCs without new break_at, and get preliminary answer.
                    RevisitingTargetState::Done => {
                        self.sccs().merge_sccs(&detected_at_of_scc, self.stack());
                        match calculation.propose_calculation() {
                            ProposalResult::Calculated(v) => v,
                            ProposalResult::CycleBroken(r) => {
                                Arc::new(K::promote_recursive(self.heap, r))
                            }
                            ProposalResult::CycleDetected | ProposalResult::Calculatable => {
                                unreachable!(
                                    "Done node in previous SCC must have Calculated or CycleBroken result"
                                )
                            }
                        }
                    }
                    // This binding has an in-flight computation. This case should be handled the same as SccState::NotInScc;
                    // in most cases we're going to wind up detecting a cycle although because of races between threads
                    // we have to handle the possibility of Calculated or CycleBroken.
                    RevisitingTargetState::InProgress => match calculation.propose_calculation() {
                        ProposalResult::CycleDetected => {
                            let current_cycle = self.stack().current_cycle().unwrap();
                            match self.sccs().on_scc_detected(current_cycle, self.stack()) {
                                SccDetectedResult::BreakHere => self
                                    .attempt_to_unwind_cycle_from_here(&current, idx, calculation)
                                    .unwrap_or_else(|r| {
                                        Arc::new(K::promote_recursive(self.heap, r))
                                    }),
                                SccDetectedResult::Continue => {
                                    self.calculate_and_record_answer(current, idx, calculation)
                                }
                            }
                        }
                        ProposalResult::Calculated(v) => {
                            self.sccs().merge_sccs(&detected_at_of_scc, self.stack());
                            v
                        }
                        ProposalResult::CycleBroken(r) => {
                            self.sccs().merge_sccs(&detected_at_of_scc, self.stack());
                            Arc::new(K::promote_recursive(self.heap, r))
                        }
                        ProposalResult::Calculatable => {
                            unreachable!(
                                "InProgress node in previous SCC must be Calculating, not NotCalculated"
                            )
                        }
                    },
                }
            }
            SccState::Participant => {
                match calculation.propose_calculation() {
                    // Participant nodes were on the CalcStack when the cycle was detected,
                    // so their Calculation must be Calculating, not NotCalculated.
                    ProposalResult::Calculatable => unreachable!(
                        "Participant nodes must have Calculating state, not NotCalculated"
                    ),
                    ProposalResult::CycleDetected => {
                        self.calculate_and_record_answer(current, idx, calculation)
                    }
                    // Short circuit if another thread has already written an answer or recursive placeholder.
                    //
                    // In either case, we need to call `on_calculation_finished` to make sure that
                    // we accurately reflect that this idx is no longer a remaining participant in
                    // active components.
                    ProposalResult::Calculated(v) => {
                        self.sccs().on_calculation_finished(&current);
                        v
                    }
                    ProposalResult::CycleBroken(r) => {
                        self.sccs().on_calculation_finished(&current);
                        Arc::new(K::promote_recursive(self.heap, r))
                    }
                }
            }
        };
        self.stack().pop();
        result
    }

    /// Calculate the value for a `K::Value`, and record it in the `Calculation`.
    ///
    /// Return the final result from the `Calculation`, which potentially might
    /// be coming from another thread because the first write wins.
    ///
    /// Errors are collected into a local error collector during solving, and
    /// only transferred to `base_errors` if this thread is the one that writes
    /// the answer. This prevents duplicate errors when multiple threads compute
    /// the same binding.
    fn calculate_and_record_answer<K: Solve<Ans>>(
        &self,
        current: CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>, Var>,
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

        // Solve the binding with a local error collector.
        //
        // Only write the errors if we actually write the result - if another thread
        // or cycle unwinding already wrote the result, we discard the errors.
        let local_errors = self.error_collector();
        let (answer, did_write) = calculation.record_value(
            K::solve(self, binding, range, &local_errors),
            |var, answer| self.finalize_recursive_answer::<K>(idx, var, answer, &local_errors),
        );
        if did_write {
            self.base_errors.extend(local_errors);
        }

        // Handle cycle unwinding, if applicable.
        //
        // TODO(stroxler): we eventually need to use is-a-cycle-active information to isolate
        // placeholder values.
        self.sccs().on_calculation_finished(&current);
        answer
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
    /// Returns a `Result` where the normal case is `Err`, because another thread
    /// might have already finished the cycle in which case we can just use that result
    /// (which will come in an `Ok(result)` form)
    ///
    /// TODO: eventually we should be recording this answer in a thread-local place rather
    /// than in the Calculation for better isolation against data races. Once that plumbing
    /// is in place, this code can probably be simplified to just return the recursive result;
    /// we are doing extra work here to get partial protection against races through the mutex.
    fn attempt_to_unwind_cycle_from_here<K: Solve<Ans>>(
        &self,
        current: &CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>, Var>,
    ) -> Result<Arc<K::Answer>, Var>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let binding = self.bindings().get(idx);
        let rec = K::create_recursive(self, binding);
        match calculation.record_cycle(rec) {
            Either::Right(rec) => {
                // No final answer is available, so we'll unwind the cycle using `rec`.
                // Track that we've recorded a placeholder for this break_at node.
                self.sccs().on_placeholder_recorded(current);
                Err(rec)
            }
            Either::Left(v) => {
                // Another thread already completed a final result, we can just use it.
                Ok(v)
            }
        }
    }

    /// Handle depth overflow based on the configured handler.
    fn handle_depth_overflow<K: Solve<Ans>>(
        &self,
        current: &CalcId,
        idx: Idx<K>,
        calculation: &Calculation<Arc<K::Answer>, Var>,
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
        calculation: &Calculation<Arc<K::Answer>, Var>,
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
        if self.sccs().is_empty() {
            eprintln!("  None");
        } else {
            for scc in self.sccs().0.borrow().iter().rev() {
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
            TypeAliasData::Ref(r) => self
                .get_from_module(r.module, None, &KeyTypeAlias(r.index))
                .unwrap_or_else(|| Arc::new(TypeAlias::unknown(r.name.clone()))),
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
                tparams: _,
                key_type_alias,
                range: _,
            } => self.solver().fresh_alias_recursive(
                self.uniques,
                TypeAliasRef {
                    name: name.clone(),
                    module: self.module().name(),
                    index: self.bindings().idx_to_key(*key_type_alias).0,
                },
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

    /// Add an implicit-any error for a generic class without explicit type arguments.
    pub fn add_implicit_any_error(
        errors: &ErrorCollector,
        range: TextRange,
        class_name: &str,
        tparam_name: Option<&str>,
    ) {
        let msg = if let Some(tparam) = tparam_name {
            format!(
                "Cannot determine the type parameter `{}` for generic class `{}`",
                tparam, class_name,
            )
        } else {
            format!(
                "Cannot determine the type parameter for generic class `{}`",
                class_name
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

    /// Helper to create a test Scc with given parameters.
    ///
    /// This bypasses the normal Scc::new constructor to allow direct construction
    /// for testing merge logic.
    #[allow(clippy::mutable_key_type)]
    fn make_test_scc(
        break_at: Vec<CalcId>,
        node_state: HashMap<CalcId, NodeState>,
        detected_at: CalcId,
        min_stack_depth: usize,
    ) -> Scc {
        Scc {
            break_at: break_at.into_iter().collect(),
            node_state,
            detected_at,
            min_stack_depth,
        }
    }

    /// Helper to create a CalcStack for testing.
    fn make_calc_stack(entries: &[CalcId]) -> CalcStack {
        let stack = CalcStack::new();
        for entry in entries {
            stack.push(entry.dupe());
        }
        stack
    }

    /// Helper to create node_state map with all nodes Fresh.
    #[allow(clippy::mutable_key_type)]
    fn fresh_nodes(ids: &[CalcId]) -> HashMap<CalcId, NodeState> {
        ids.iter().map(|id| (id.dupe(), NodeState::Fresh)).collect()
    }

    #[test]
    fn test_initial_cycle_detection() {
        // Setup: CalcStack = [M0, M1, M2], detect a cycle [M2, M1, M0]
        // Expected: New SCC with participants {M0, M1, M2}, break_at = M0 (minimal)
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        let calc_stack = make_calc_stack(&[a.dupe(), b.dupe(), c.dupe()]);
        let sccs = Sccs::new();

        // Simulate detecting cycle - raw cycle order is from detection point to back-edge target
        let raw_cycle = vec1![c.dupe(), b.dupe(), a.dupe()];
        let result = sccs.on_scc_detected(raw_cycle, &calc_stack);

        // Should not break immediately since break_at is A (minimal) but detected_at is C
        assert!(matches!(result, SccDetectedResult::Continue));

        // Verify SCC was created
        let stack = sccs.0.borrow();
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
        let sccs = Sccs::new();

        // Create initial SCC with A, B, C, D
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe(), a.dupe()];
        sccs.on_scc_detected(initial_cycle, &calc_stack);

        // Now detect sub-cycle D -> B
        let sub_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        sccs.on_scc_detected(sub_cycle, &calc_stack);

        // The sub-cycle overlaps with existing SCC, so they merge
        let stack = sccs.0.borrow();
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
        let sccs = Sccs::new();

        // Create initial SCC with B, C, D (detected from D going back to B)
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        sccs.on_scc_detected(initial_cycle, &calc_stack);

        // Verify initial state
        {
            let stack = sccs.0.borrow();
            assert_eq!(stack.len(), 1);
            assert_eq!(stack[0].node_state.len(), 3);
        }

        // Now detect cycle [F, E, D, C] - overlaps with existing at C and D
        let new_cycle = vec1![f.dupe(), e.dupe(), d.dupe(), c.dupe()];
        sccs.on_scc_detected(new_cycle, &calc_stack);

        // Should merge because new cycle overlaps with existing SCC
        let stack = sccs.0.borrow();
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
        let sccs = Sccs::new();

        // Create initial SCC with B, C, D
        let initial_cycle = vec1![d.dupe(), c.dupe(), b.dupe()];
        sccs.on_scc_detected(initial_cycle, &calc_stack);

        // Now detect cycle [F, E, D, C, B, A] - includes everything from A to F
        let new_cycle = vec1![f.dupe(), e.dupe(), d.dupe(), c.dupe(), b.dupe(), a.dupe()];
        sccs.on_scc_detected(new_cycle, &calc_stack);

        // Should merge because new cycle contains the existing SCC
        let stack = sccs.0.borrow();
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
    fn test_merge_two_components_with_gap() {
        // CalcStack: [M0, M1, M2, M3, M4, M5, M6, M7]
        // SCC1: {M0, M1, M2}
        // SCC2: {M5, M6}
        // Gap: M3, M4, M7 are not in any SCC
        //
        // New cycle: [M7, M6, M5, M4, M3, M2, M1] (M7 back to M1)
        // Overlaps with SCC1 at {M1, M2} and SCC2 at {M5, M6}
        //
        // Expected: All merge into {M0, M1, M2, M3, M4, M5, M6, M7}
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);
        let f = CalcId::for_test("m", 5);
        let g = CalcId::for_test("m", 6);
        let h = CalcId::for_test("m", 7);

        let calc_stack = make_calc_stack(&[
            a.dupe(),
            b.dupe(),
            c.dupe(),
            d.dupe(),
            e.dupe(),
            f.dupe(),
            g.dupe(),
            h.dupe(),
        ]);
        let sccs = Sccs::new();

        // Create SCC1: {A, B, C}
        sccs.on_scc_detected(vec1![c.dupe(), b.dupe(), a.dupe()], &calc_stack);

        // Create SCC2: {F, G}
        sccs.on_scc_detected(vec1![g.dupe(), f.dupe()], &calc_stack);

        // Verify initial state: two separate SCCs
        {
            let stack = sccs.0.borrow();
            assert_eq!(stack.len(), 2, "Should have two SCCs before merge");
        }

        // New cycle: [H, G, F, E, D, C, B] - spans both SCCs and the gap
        let merge_cycle = vec1![
            h.dupe(),
            g.dupe(),
            f.dupe(),
            e.dupe(),
            d.dupe(),
            c.dupe(),
            b.dupe(),
        ];
        sccs.on_scc_detected(merge_cycle, &calc_stack);

        // Should merge into one SCC
        let stack = sccs.0.borrow();
        assert_eq!(stack.len(), 1, "Should have merged into one SCC");

        let scc = &stack[0];
        // All nodes should be in the merged SCC
        // A is included because it was in SCC1 which got merged
        assert!(
            scc.node_state.contains_key(&a),
            "A should be in merged SCC (from SCC1)"
        );
        assert!(scc.node_state.contains_key(&b));
        assert!(scc.node_state.contains_key(&c));
        assert!(scc.node_state.contains_key(&d));
        assert!(scc.node_state.contains_key(&e));
        assert!(scc.node_state.contains_key(&f));
        assert!(scc.node_state.contains_key(&g));
        assert!(scc.node_state.contains_key(&h));
    }

    // Make sure that we only merge overlapping components
    //
    // This is critical for the optimization: we must not incorrectly merge
    // SCC1 just because there's a cycle that spans a large range.
    #[test]
    fn test_selective_merge_three_components() {
        // CalcStack: [M0, M1, M2, M3, M4, M5, M6, M7, M8, M9]
        //
        // Three separate SCCs, each with 2 members:
        //   SCC1: {M0, M1} - does NOT overlap with new cycle
        //   SCC2: {M3, M4} - overlaps with new cycle
        //   SCC3: {M7, M8} - overlaps with new cycle
        //
        // New cycle: [M9, M8, M7, M6, M5, M4, M3]
        //   Overlaps with SCC2 at {M3, M4} and SCC3 at {M7, M8}
        //   Does NOT overlap with SCC1 (M0, M1 not in cycle)
        //
        // Expected: SCC2 and SCC3 merge with cycle, but SCC1 stays separate
        // Result: Two SCCs - {M0, M1} and {M3, M4, M5, M6, M7, M8, M9}
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);
        let d = CalcId::for_test("m", 3);
        let e = CalcId::for_test("m", 4);
        let f = CalcId::for_test("m", 5);
        let g = CalcId::for_test("m", 6);
        let h = CalcId::for_test("m", 7);
        let i = CalcId::for_test("m", 8);
        let j = CalcId::for_test("m", 9);

        let calc_stack = make_calc_stack(&[
            a.dupe(),
            b.dupe(),
            c.dupe(),
            d.dupe(),
            e.dupe(),
            f.dupe(),
            g.dupe(),
            h.dupe(),
            i.dupe(),
            j.dupe(),
        ]);
        let sccs = Sccs::new();

        // Create SCC1: {A, B}
        sccs.on_scc_detected(vec1![b.dupe(), a.dupe()], &calc_stack);

        // Create SCC2: {D, E}
        sccs.on_scc_detected(vec1![e.dupe(), d.dupe()], &calc_stack);

        // Create SCC3: {H, I}
        sccs.on_scc_detected(vec1![i.dupe(), h.dupe()], &calc_stack);

        // Verify: three separate SCCs
        {
            let stack = sccs.0.borrow();
            assert_eq!(stack.len(), 3, "Should have three SCCs initially");
        }

        // New cycle: [J, I, H, G, F, E, D]
        // Should merge SCC2 and SCC3 but NOT SCC1
        let merge_cycle = vec1![
            j.dupe(),
            i.dupe(),
            h.dupe(),
            g.dupe(),
            f.dupe(),
            e.dupe(),
            d.dupe(),
        ];
        sccs.on_scc_detected(merge_cycle, &calc_stack);

        // Should now have exactly 2 SCCs
        let stack = sccs.0.borrow();
        assert_eq!(stack.len(), 2, "Should have two SCCs after selective merge");

        // Find the SCC containing A (should be separate)
        let scc_with_a = stack.iter().find(|scc| scc.node_state.contains_key(&a));
        assert!(scc_with_a.is_some(), "SCC1 should still exist");
        let scc1 = scc_with_a.unwrap();

        // SCC1 should only contain A and B
        assert_eq!(scc1.node_state.len(), 2, "SCC1 should only have A and B");
        assert!(scc1.node_state.contains_key(&a));
        assert!(scc1.node_state.contains_key(&b));

        // Find the merged SCC (contains D)
        let scc_with_d = stack.iter().find(|scc| scc.node_state.contains_key(&d));
        assert!(scc_with_d.is_some(), "Merged SCC should exist");
        let merged = scc_with_d.unwrap();

        // Merged SCC should contain D, E, F, G, H, I, J (7 nodes)
        assert_eq!(merged.node_state.len(), 7, "Merged SCC should have 7 nodes");
        assert!(
            !merged.node_state.contains_key(&a),
            "A should not be in merged SCC"
        );
        assert!(
            !merged.node_state.contains_key(&b),
            "B should not be in merged SCC"
        );
        assert!(merged.node_state.contains_key(&d));
        assert!(merged.node_state.contains_key(&e));
        assert!(merged.node_state.contains_key(&f));
        assert!(merged.node_state.contains_key(&g));
        assert!(merged.node_state.contains_key(&h));
        assert!(merged.node_state.contains_key(&i));
        assert!(merged.node_state.contains_key(&j));
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
            0, // min_stack_depth
        );
        let scc2 = make_test_scc(
            vec![c.dupe()],
            fresh_nodes(&[c.dupe(), d.dupe()]),
            c.dupe(),
            2, // min_stack_depth
        );

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Both break points should be preserved
        assert!(merged.break_at.contains(&a));
        assert!(merged.break_at.contains(&c));
        assert_eq!(merged.break_at.len(), 2);

        // All nodes should be present
        assert_eq!(merged.node_state.len(), 4);

        // min_stack_depth should be the minimum (0)
        assert_eq!(merged.min_stack_depth, 0);
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn test_merge_many_takes_most_advanced_state() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);

        // SCC1 has M0 as Done, M1 as Fresh
        let mut scc1_state = HashMap::new();
        scc1_state.insert(a.dupe(), NodeState::Done);
        scc1_state.insert(b.dupe(), NodeState::Fresh);
        let scc1 = make_test_scc(vec![a.dupe()], scc1_state, a.dupe(), 0);

        // SCC2 has M0 as Fresh, M1 as InProgress
        let mut scc2_state = HashMap::new();
        scc2_state.insert(a.dupe(), NodeState::Fresh);
        scc2_state.insert(b.dupe(), NodeState::InProgress);
        let scc2 = make_test_scc(vec![a.dupe()], scc2_state, a.dupe(), 0);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should take the most advanced state for each node
        assert_eq!(merged.node_state.get(&a), Some(&NodeState::Done));
        assert_eq!(merged.node_state.get(&b), Some(&NodeState::InProgress));
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
    fn test_merge_many_keeps_minimum_stack_depth() {
        let a = CalcId::for_test("m", 0);
        let b = CalcId::for_test("m", 1);
        let c = CalcId::for_test("m", 2);

        // SCC1 with min_stack_depth = 5
        let scc1 = make_test_scc(
            vec![a.dupe()],
            fresh_nodes(&[a.dupe(), b.dupe()]),
            a.dupe(),
            5,
        );
        // SCC2 with min_stack_depth = 2
        let scc2 = make_test_scc(vec![c.dupe()], fresh_nodes(&[c.dupe()]), c.dupe(), 2);

        let merged = Scc::merge_many(vec1![scc1, scc2], a.dupe());

        // Should keep the minimum stack depth
        assert_eq!(merged.min_stack_depth, 2);
    }
}
