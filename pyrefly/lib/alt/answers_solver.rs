/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::cmp::Ordering;
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
use pyrefly_graph::calculation::Calculation;
use pyrefly_graph::calculation::ProposalResult;
use pyrefly_graph::index::Idx;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
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
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
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
        (self.0.module(), &self.1) == (other.0.module(), &other.1)
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
        self.0.module().hash(state);
        self.1.hash(state);
    }
}

/// Represent a stack of in-progress calculations in an `AnswersSolver`.
///
/// This is useful for debugging, particularly for debugging cycle handling.
///
/// The stack is per-thread; we create a new `AnswersSolver` every time
/// we change modules when resolving exports, but the stack is passed
/// down because cycles can cross module boundaries.
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

    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
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

const MAXIMUM_CYCLE_DEPTH: usize = 100;

/// Normalize a raw cycle for comparison by sorting and deduplicating.
fn normalize_raw_cycle(raw: &Vec1<CalcId>) -> Vec<CalcId> {
    let mut normalized: Vec<CalcId> = raw.iter().duped().collect();
    normalized.sort();
    normalized.dedup();
    normalized
}

/// Tracks the state of a node within an active cycle.
///
/// This replaces the previous stack-based tracking (recursion_stack, unwind_stack)
/// with explicit state tracking. The state transitions are:
/// - Fresh → InProgress (when we first encounter the node as a Participant)
/// - InProgress → Done (when the node's calculation completes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    /// Node hasn't been processed yet as part of cycle handling.
    Fresh,
    /// Node is currently being processed (on the Rust call stack).
    InProgress,
    /// Node's calculation has completed.
    Done,
}

/// Represent a cycle we are currently solving.
///
/// This simplified model tracks cycle participants with explicit state rather than
/// using separate recursion and unwind stacks. The Rust call stack naturally
/// enforces LIFO ordering, so we only need to track:
/// - Which idx is the anchor where we break the cycle
/// - The state of each participant (Fresh/InProgress/Done)
#[derive(Debug, Clone)]
pub struct Cycle {
    /// Where do we want to break the cycle (the minimal CalcId in the cycle)
    break_at: CalcId,
    /// State of each participant in this cycle.
    /// Keys are all participants; values track their computation state.
    node_state: HashMap<CalcId, NodeState>,
    /// Where we detected the cycle (for debugging only)
    detected_at: CalcId,
}

impl Display for Cycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let states: Vec<_> = self.node_state.iter().collect();
        write!(
            f,
            "Cycle{{break_at: {}, node_state: {:?}, detected_at: {}}}",
            self.break_at, states, self.detected_at,
        )
    }
}

impl Cycle {
    #[allow(clippy::mutable_key_type)] // CalcId's Hash impl doesn't depend on mutable parts
    fn new(raw: Vec1<CalcId>) -> Self {
        let detected_at = raw.first().dupe();
        let (_, break_at) = raw.iter().enumerate().min_by_key(|(_, c)| *c).unwrap();

        // Initialize all nodes as Fresh
        let node_state: HashMap<CalcId, NodeState> =
            raw.iter().duped().map(|c| (c, NodeState::Fresh)).collect();

        Cycle {
            break_at: break_at.dupe(),
            node_state,
            detected_at,
        }
    }

    /// Check if the current idx is a participant in this cycle and determine its state.
    ///
    /// Returns the appropriate CycleState:
    /// - BreakAt if this is the anchor where we produce a placeholder
    /// - Participant if this is a Fresh node (marks it as InProgress)
    /// - NoDetectedCycle if this idx is InProgress or Done (or not in cycle)
    ///
    /// When a Fresh node is encountered, it transitions to InProgress.
    /// If an InProgress node is hit again (via a different path like A→X→C),
    /// we return NoDetectedCycle which triggers proper new cycle detection.
    fn pre_calculate_state(&mut self, current: &CalcId) -> CycleState {
        if *current == self.break_at {
            CycleState::BreakAt
        } else if let Some(state) = self.node_state.get_mut(current) {
            match state {
                NodeState::Fresh => {
                    *state = NodeState::InProgress;
                    CycleState::Participant
                }
                NodeState::InProgress | NodeState::Done => {
                    // Already being processed or finished - treat as if not in cycle.
                    // If InProgress, a back edge through this node will trigger new cycle detection.
                    CycleState::NoDetectedCycle
                }
            }
        } else {
            CycleState::NoDetectedCycle
        }
    }

    /// Track that a calculation has finished, marking it as Done.
    fn on_calculation_finished(&mut self, current: &CalcId) {
        if let Some(state) = self.node_state.get_mut(current) {
            *state = NodeState::Done;
        }
    }

    /// Check if the cycle is complete (all participants are Done).
    fn is_complete(&self) -> bool {
        self.node_state
            .values()
            .all(|state| *state == NodeState::Done)
    }

    /// Get all participants in this cycle as a sorted vector for comparison.
    ///
    /// This normalizes the cycle representation so cycles can be compared regardless
    /// of where they were detected or how far they've been processed.
    fn participants_normalized(&self) -> Vec<CalcId> {
        let mut participants: Vec<CalcId> = self.node_state.keys().duped().collect();
        participants.sort();
        participants.dedup();
        participants
    }
}

/// Represents the current cycle state prior to attempting a particular calculation.
enum CycleState {
    /// The current idx is not participating in any currently detected cycle (though it
    /// remains possible we will detect one here).
    ///
    /// Note that this does not necessarily mean there is no active cycle: the
    /// graph solve will frequently branch out from a cycle into other parts of
    /// the dependency graph, and in those cases we are not in a currently-known
    /// cycle.
    NoDetectedCycle,
    /// This idx is part of the active cycle, and we are either (if this is a pre-calculation
    /// check) recursing out toward `break_at` or unwinding back toward `break_at`.
    Participant,
    /// This idx is the `break_at` for the active cycle, which means we have
    /// reached the end of the recursion and should return a placeholder to our
    /// parent frame.
    BreakAt,
}

enum CycleDetectedResult {
    /// Break immediately at the idx where we detected the cycle, so that we
    /// unwind back to the same idx.
    BreakHere,
    /// Continue recursing until we hit some other idx that is the minimal `break_at` idx.
    Continue,
    /// Duplicate cycle detected in the stack - this indicates infinite recursion.
    /// Raise a Pyrefly error and produce a placeholder result.
    DuplicateCycleDetected,
}

/// Represent the current thread's cycles, which form a stack
/// because we can encounter a new one while solving another.
pub struct Cycles(RefCell<Vec<Cycle>>);

impl Cycles {
    pub fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    /// Handle a cycle we just detected.
    ///
    /// Return whether to break immediately (which is relatively common, since
    /// we break on the minimal idx which is often where we detect the problem)
    /// or continue recursing.
    fn on_cycle_detected(&self, raw: Vec1<CalcId>) -> CycleDetectedResult {
        if self.0.borrow().len() > MAXIMUM_CYCLE_DEPTH {
            // Check if this is a duplicate of an existing cycle (indicating infinite recursion)
            let normalized_raw = normalize_raw_cycle(&raw);
            let has_duplicate =
                self.0.borrow().iter().any(|existing_cycle| {
                    existing_cycle.participants_normalized() == normalized_raw
                });
            if has_duplicate {
                // Don't push the duplicate cycle - just return DuplicateCycleDetected
                return CycleDetectedResult::DuplicateCycleDetected;
            }
            // High depth but no duplicate - treat as normal cycle
        }

        // Normal cycle detection logic
        let cycle = Cycle::new(raw);
        let result = if cycle.break_at == cycle.detected_at {
            CycleDetectedResult::BreakHere
        } else {
            CycleDetectedResult::Continue
        };
        self.0.borrow_mut().push(cycle);
        result
    }

    fn pre_calculate_state(&self, current: &CalcId) -> CycleState {
        if let Some(active_cycle) = self.0.borrow_mut().last_mut() {
            active_cycle.pre_calculate_state(current)
        } else {
            CycleState::NoDetectedCycle
        }
    }

    /// Handle the completion of a calculation. This might involve progress on
    /// the remaining participants of one or more cycles.
    ///
    /// Return `true` if there are active cycles after finishing this calculation,
    /// `false` if there are not.
    fn on_calculation_finished(&self, current: &CalcId) -> bool {
        let mut stack = self.0.borrow_mut();
        for cycle in stack.iter_mut() {
            cycle.on_calculation_finished(current);
        }
        while let Some(cycle) = stack.last() {
            if cycle.is_complete() {
                stack.pop();
            } else {
                break;
            }
        }
        // Do we still have active cycles?
        !stack.is_empty()
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
    cycles: Cycles,
    stack: CalcStack,
    /// For debugging only: thread-global that allows us to control debug logging across components.
    debug: RefCell<bool>,
}

impl ThreadState {
    pub fn new() -> Self {
        Self {
            cycles: Cycles::new(),
            stack: CalcStack::new(),
            debug: RefCell::new(false),
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

    fn cycles(&self) -> &Cycles {
        &self.thread_state.cycles
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
            self.thread_state.cycles.is_empty(),
            "The cycle stack should be empty in the final thread state"
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
        let result = match self.cycles().pre_calculate_state(&current) {
            CycleState::NoDetectedCycle => match calculation.propose_calculation() {
                ProposalResult::Calculated(v) => v,
                ProposalResult::CycleBroken(r) => Arc::new(K::promote_recursive(r)),
                ProposalResult::CycleDetected => {
                    let current_cycle = self.stack().current_cycle().unwrap();
                    match self.cycles().on_cycle_detected(current_cycle) {
                        CycleDetectedResult::BreakHere => self
                            .attempt_to_unwind_cycle_from_here(idx, calculation)
                            .unwrap_or_else(|r| Arc::new(K::promote_recursive(r))),
                        CycleDetectedResult::Continue => {
                            self.calculate_and_record_answer(current, idx, calculation)
                        }
                        CycleDetectedResult::DuplicateCycleDetected => {
                            let range = self.bindings().idx_to_key(idx).range();
                            self.base_errors.internal_error(
                                range,
                                vec1![format!(
                                    "Duplicate cycle detected at {current}; likely infinite recursion in type resolution"
                                )],
                            );
                            self.attempt_to_unwind_cycle_from_here(idx, calculation)
                                .unwrap_or_else(|r| Arc::new(K::promote_recursive(r)))
                        }
                    }
                }
                ProposalResult::Calculatable => {
                    self.calculate_and_record_answer(current, idx, calculation)
                }
            },
            CycleState::BreakAt => {
                // Begin unwinding the cycle using a recursive placeholder
                self.attempt_to_unwind_cycle_from_here(idx, calculation)
                    .unwrap_or_else(|r| Arc::new(K::promote_recursive(r)))
            }
            CycleState::Participant => {
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
                    // active cycles.
                    ProposalResult::Calculated(v) => {
                        self.cycles().on_calculation_finished(&current);
                        v
                    }
                    ProposalResult::CycleBroken(r) => {
                        self.cycles().on_calculation_finished(&current);
                        Arc::new(K::promote_recursive(r))
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
        self.cycles().on_calculation_finished(&current);
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
                Err(rec)
            }
            Either::Left(v) => {
                // Another thread already completed a final result, we can just use it.
                Ok(v)
            }
        }
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
                    _ if in_type => (self.f)(&Type::Type(Box::new(ty.clone()))),
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
        Type::any_error()
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
