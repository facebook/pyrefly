/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::Any;
use std::any::type_name;
use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hint::spin_loop;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::ops::Deref;
use std::ops::DerefMut;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;
use std::sync::atomic::fence;
use std::thread::yield_now;
use std::time::Duration;
use std::time::Instant;

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::lock::Mutex;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;

use crate::alt::answers_solver::AnswerScope;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::CalcId;
use crate::alt::answers_solver::ReservedSlot;
use crate::alt::answers_solver::ThreadState;
use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::Exported;
use crate::binding::binding::Key;
use crate::binding::binding::Keyed;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::metadata::BindingsMetadata;
use crate::binding::table::TableKeyed;
use crate::config::base::RecursionLimitConfig;
use crate::dispatch_anyidx;
use crate::error::collector::ErrorCollector;
use crate::error::style::ErrorStyle;
use crate::export::exports::LookupExport;
use crate::module::module_info::ModuleInfo;
use crate::report::cinderx::CinderxSolutions;
use crate::report::pysa::PysaSolutions;
use crate::solver::solver::Solver;
use crate::solver::solver::VarRecurser;
use crate::state::errors::ModuleRanges;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::state::ModuleChanges;
use crate::table;
use crate::table_for_each;
use crate::table_mut_for_each;
use crate::table_try_for_each;
use crate::types::callable::Callable;
use crate::types::class::Class;
use crate::types::class::ClassFields;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;
use crate::types::heap::TypeHeap;
use crate::types::stdlib::Stdlib;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::TParams;
use crate::types::types::Type;

/// The index stores reference edges that cannot be recovered by scanning the current module's AST.
/// This includes references to external definitions and implicit constructor-protocol references.
#[derive(Debug, Default)]
pub struct Index {
    /// A map from (import specifier (ModuleName), imported symbol (Name)) to all references to it
    /// in the current module.
    pub externally_defined_variable_references: SmallMap<(ModuleName, Name), Vec<TextRange>>,
    /// A map from (import specifier (ModuleName), imported symbol (Name)) to all references to it
    /// in the current module.
    pub renamed_imports: SmallMap<(ModuleName, Name), Vec<TextRange>>,
    /// A map from (attribute definition module) to a list of pairs of
    /// (range of attribute definition in the definition, range of reference in the current module).
    pub externally_defined_attribute_references: SmallMap<ModulePath, Vec<(TextRange, TextRange)>>,
    /// A map from (constructor definition module) to a list of pairs of
    /// (range of the constructor definition, range of the call site in the current module).
    /// The call-site range spells the class name, not the definition's own name, so these are
    /// gated behind `ReferenceOptions::include_constructor_call_sites`.
    pub constructor_references: SmallMap<ModulePath, Vec<(TextRange, TextRange)>>,
    /// A map from (child method range) to a list of parent method definitions (ModulePath, parent method range).
    /// This is used to find reimplementations when doing find-references on parent methods.
    pub parent_methods_map: SmallMap<TextRange, Vec<(ModulePath, TextRange)>>,
}

/// How the source text at a reference range relates to the attribute it resolves to.
#[derive(Debug, Clone, Copy)]
pub enum AttributeReferenceKind {
    /// The reference spells the attribute's own name, as in `x.attr`.
    Textual,
    /// The reference is a call site that reaches the attribute implicitly through the
    /// constructor protocol, as in `Foo()` reaching `Foo.__init__`. Only class construction
    /// is recorded this way; calling an instance through its `__call__` is not.
    ConstructorCall,
}

/// Visiting a trace store reaches every type recorded in it.
impl VisitMut<Type> for Traces {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        for map in [
            &mut self.types,
            &mut self.invoked_properties,
            &mut self.expected_types,
        ] {
            for ty in map.values_mut() {
                f(Arc::make_mut(ty));
            }
        }
        for callee in self.overloaded_callees.values_mut() {
            // Note: `tparams` does not need to be visited, as this comes from an answer
            // which is already visited. `TArgs` skips tparams for the same reason.
            match callee {
                OverloadedCallee::Resolved { callable: trace } => {
                    let OverloadTrace {
                        callable,
                        tparams: _,
                    } = trace;
                    callable.visit_mut(f);
                }
                OverloadedCallee::Candidates { all, closest, .. } => {
                    for OverloadTrace {
                        callable,
                        tparams: _,
                    } in all.iter_mut().chain(std::iter::once(closest))
                    {
                        callable.visit_mut(f);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct OverloadTrace {
    callable: Callable,
    tparams: Option<Arc<TParams>>,
}

impl OverloadTrace {
    pub(crate) fn new(callable: Callable, tparams: Option<Arc<TParams>>) -> Self {
        Self { callable, tparams }
    }

    fn as_type(&self) -> Type {
        match &self.tparams {
            Some(tparams) if !tparams.is_empty() => Type::Forall(Box::new(Forall {
                tparams: tparams.clone(),
                body: Forallable::Callable(self.callable.clone()),
            })),
            _ => Type::Callable(Box::new(self.callable.clone())),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OverloadedCallee {
    Resolved {
        callable: OverloadTrace,
    },
    Candidates {
        all: Vec<OverloadTrace>,
        closest: OverloadTrace,
        is_closest_chosen: bool,
    },
}

#[derive(Debug, Default)]
pub struct Traces {
    types: SmallMap<TextRange, Arc<Type>>,
    /// A map from (range of callee, overload information)
    overloaded_callees: SmallMap<TextRange, OverloadedCallee>,
    /// A map of text ranges that correspond to 'b' portion in expressions a.b where b is a property access -> getter type
    invoked_properties: SmallMap<TextRange, Arc<Type>>,
    /// A map from expression range to expected type at that position (for type checking)
    expected_types: SmallMap<TextRange, Arc<Type>>,
}

impl Traces {
    /// Merge accumulated side effects into the persisted trace store.
    fn merge(&mut self, side_effects: TraceSideEffects) {
        for (k, v) in side_effects.types {
            self.types.insert(k, v);
        }
        for (k, v) in side_effects.overloaded_callees {
            self.overloaded_callees.insert(k, v);
        }
        for (k, v) in side_effects.invoked_properties {
            self.invoked_properties.insert(k, v);
        }
        for (k, v) in side_effects.expected_types {
            self.expected_types.insert(k, v);
        }
    }
}

/// Accumulates trace events during a single calculation.
/// Published to `Traces` only when the calculation result is committed.
#[derive(Debug, Default, Clone)]
pub struct TraceSideEffects {
    pub types: SmallMap<TextRange, Arc<Type>>,
    pub overloaded_callees: SmallMap<TextRange, OverloadedCallee>,
    pub invoked_properties: SmallMap<TextRange, Arc<Type>>,
    pub expected_types: SmallMap<TextRange, Arc<Type>>,
}

/// Invariants:
///
/// * Every module name referenced anywhere MUST be present
///   in the `exports` and `bindings` map.
/// * Every key referenced in `bindings`/`answers` MUST be present.
///
/// We never issue contains queries on these maps.
#[derive(Debug)]
pub struct Answers {
    solver: Solver,
    table: AnswerTable,
    solutions: Arc<SolutionsData>,
    index: Option<Arc<Mutex<Index>>>,
    trace: Option<Mutex<Traces>>,
}

const PUBLISH_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PUBLISH_BACKOFF_STEP: u32 = 6;
const PENDING_TAG: usize = 0b01;
const ALIAS_TAG: usize = 0b10;
const TAG_MASK: usize = PENDING_TAG | ALIAS_TAG;

/// Guarantees two free low pointer bits for the pending and alias tags.
#[repr(align(4))]
struct AlignedAnswer<T>(T);

/// An answer allocation owned by its result table.
#[doc(hidden)]
pub struct AnswerBox<T>(Box<AlignedAnswer<T>>);

/// An answer whose key type has been erased, mirroring `std::any::Any`.
///
/// Construction names the key that produced the answer, while downcasts name
/// its answer type. How answers are allocated stays private to this module.
#[doc(hidden)]
pub struct AnyAnswer(Box<dyn Any + Send + Sync>);

impl Debug for AnyAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyAnswer").finish_non_exhaustive()
    }
}

impl AnyAnswer {
    pub(crate) fn new<K: Keyed>(answer: AnswerBox<K::Answer>) -> Self {
        Self(answer.0)
    }

    pub(crate) fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0
            .downcast_ref::<AlignedAnswer<T>>()
            .map(|answer| &answer.0)
    }

    pub(crate) fn downcast<T: Any + Send + Sync>(self) -> Result<AnswerBox<T>, Self> {
        match self.0.downcast::<AlignedAnswer<T>>() {
            Ok(answer) => Ok(AnswerBox(answer)),
            Err(answer) => Err(Self(answer)),
        }
    }
}

impl<T> AnswerBox<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(Box::new(AlignedAnswer(value)))
    }
}

impl<T> Deref for AnswerBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<T> DerefMut for AnswerBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T: Debug> Debug for AnswerBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: Display> Display for AnswerBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

/// A first-write-wins result slot containing an aligned answer pointer.
/// Null is unpublished, `PENDING_TAG` marks a pending pointer, and `ALIAS_TAG`
/// marks a pointer to another slot's allocation. An untagged non-null pointer is
/// published and owned by this slot.
///
/// The tags are independent, so a pending alias carries both. No caller reserves
/// an alias today, but the slot supports it because a tag combination that costs
/// nothing to handle is cheaper than a rule that every caller has to remember.
pub struct AnswerSlot<T: Send> {
    ptr: AtomicPtr<AlignedAnswer<T>>,
    _owner: PhantomData<AnswerBox<T>>,
}

impl<T: Send> Default for AnswerSlot<T> {
    fn default() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
            _owner: PhantomData,
        }
    }
}

impl<T: Send> AnswerSlot<T> {
    fn is_pending(ptr: *mut AlignedAnswer<T>) -> bool {
        ptr.addr() & PENDING_TAG != 0
    }

    fn is_alias(ptr: *mut AlignedAnswer<T>) -> bool {
        ptr.addr() & ALIAS_TAG != 0
    }

    fn answer_ptr(ptr: *mut AlignedAnswer<T>) -> *mut AlignedAnswer<T> {
        ptr.map_addr(|addr| addr & !TAG_MASK)
    }

    fn pending(ptr: *mut AlignedAnswer<T>) -> *mut AlignedAnswer<T> {
        ptr.map_addr(|addr| addr | PENDING_TAG)
    }

    fn published(ptr: *mut AlignedAnswer<T>) -> *mut AlignedAnswer<T> {
        ptr.map_addr(|addr| addr & !PENDING_TAG)
    }

    fn alias(ptr: *mut AlignedAnswer<T>) -> *mut AlignedAnswer<T> {
        Self::answer_ptr(ptr).map_addr(|addr| addr | ALIAS_TAG)
    }

    fn into_raw(value: AnswerBox<T>) -> *mut AlignedAnswer<T> {
        Box::into_raw(value.0)
    }

    /// # Safety
    ///
    /// `ptr` must identify a live answer allocation owned by this result table.
    unsafe fn get_raw(&self, ptr: *mut AlignedAnswer<T>) -> &T {
        // SAFETY: Forwarded from the caller after removing the status tags.
        unsafe { &(*Self::answer_ptr(ptr)).0 }
    }

    /// Wait for a pending writer to publish its result. Returns `None` only if
    /// panic unwinding rolls back an SCC reservation before publication.
    #[cold]
    fn wait_for_publish(&self) -> Option<*mut AlignedAnswer<T>> {
        let deadline = Instant::now() + PUBLISH_TIMEOUT;
        let mut backoff_step = 0;
        loop {
            let ptr = self.ptr.load(Ordering::Relaxed);
            if ptr.is_null() {
                // A pending slot can return to Unpublished only when panic
                // unwinding rolls back an SCC reservation.
                return None;
            } else if Self::is_pending(ptr) {
                // Keep waiting for the reservation to be published or rolled back.
            } else {
                // The relaxed load observed the release publication, so
                // this fence acquires initialization of the result.
                fence(Ordering::Acquire);
                return Some(ptr);
            }

            if backoff_step <= MAX_PUBLISH_BACKOFF_STEP {
                let spins = 1 << backoff_step;
                for _ in 0..spins {
                    spin_loop();
                }
                backoff_step += 1;
            } else {
                if Instant::now() >= deadline {
                    // Pending does not include answer calculation. Ordinary publication
                    // takes a few cache-resident atomic operations (roughly 10-100 ns on
                    // current server CPUs); even SCC batches should finish in microseconds.
                    // Reaching 30 seconds therefore indicates a stuck publisher, so panic
                    // intentionally rather than waiting forever.
                    panic!("answer publication remained pending for {PUBLISH_TIMEOUT:?}");
                }
                yield_now();
                backoff_step = 0;
            }
        }
    }

    /// Decide what a failed publish CAS means, which is the same for every
    /// publisher regardless of what it was trying to store.
    ///
    /// `Break` carries the published pointer the winning writer installed.
    /// `Continue` means the slot held a reservation that panic unwinding then
    /// rolled back, leaving the slot free again, so the caller should retry.
    ///
    /// A caller that owns a candidate allocation must keep it owned across this
    /// call, which waits and can panic, so that unwinding drops it.
    #[cold]
    fn handle_failed_cas(
        &self,
        actual: *mut AlignedAnswer<T>,
    ) -> ControlFlow<*mut AlignedAnswer<T>> {
        if Self::is_pending(actual) {
            match self.wait_for_publish() {
                Some(actual) => ControlFlow::Break(actual),
                None => ControlFlow::Continue(()),
            }
        } else {
            ControlFlow::Break(actual)
        }
    }

    /// Called only after `get` observes that this slot is pending.
    #[cold]
    fn get_pending(&self) -> Option<&T> {
        let ptr = self.wait_for_publish()?;
        // SAFETY: `wait_for_publish` observed a published pointer retained by
        // this result table.
        Some(unsafe { self.get_raw(ptr) })
    }

    fn get(&self) -> Option<&T> {
        let ptr = self.ptr.load(Ordering::Acquire);
        if Self::is_pending(ptr) {
            self.get_pending()
        } else if ptr.is_null() {
            None
        } else {
            // SAFETY: A published pointer is retained by this result table.
            Some(unsafe { self.get_raw(ptr) })
        }
    }

    fn get_published(&self) -> &T {
        let ptr = self.ptr.load(Ordering::Acquire);
        assert!(!ptr.is_null(), "solution result is unpublished");
        assert!(!Self::is_pending(ptr), "solution result is pending");
        // SAFETY: The checks above prove that this is a published pointer
        // retained by this result table.
        unsafe { self.get_raw(ptr) }
    }

    fn record(&self, value: AnswerBox<T>) -> (&T, bool) {
        let mut candidate = Self::into_raw(value);
        loop {
            match self.ptr.compare_exchange(
                ptr::null_mut(),
                candidate,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: This slot owns the newly published allocation.
                    return (unsafe { self.get_raw(candidate) }, true);
                }
                Err(actual) => {
                    // SAFETY: The failed CAS left the candidate allocation with us.
                    // Owning it here keeps it live across the wait below, so an
                    // unwind drops it, and the losing path frees it on return.
                    let value = AnswerBox(unsafe { Box::from_raw(candidate) });
                    match self.handle_failed_cas(actual) {
                        ControlFlow::Break(actual) => {
                            // SAFETY: The failed acquire CAS, or the subsequent wait,
                            // observed a published pointer retained by this table.
                            return (unsafe { self.get_raw(actual) }, false);
                        }
                        ControlFlow::Continue(()) => candidate = Self::into_raw(value),
                    }
                }
            }
        }
    }

    /// Publish the same allocation as another slot in the same result table.
    ///
    /// The caller decides to record an alias by observing the target published,
    /// and publication is permanent, so the target is still published here.
    ///
    /// # Safety
    ///
    /// `target` must belong to the same result table as `self`. The table must
    /// retain the owning canonical slot until both slots can no longer be read.
    unsafe fn record_alias(&self, target: &Self) -> (&T, bool) {
        let ptr = target.ptr.load(Ordering::Acquire);
        assert!(
            !ptr.is_null() && !Self::is_pending(ptr),
            "alias target must be published"
        );
        // The alias tags `target`'s pointer rather than owning an allocation of
        // its own, so unlike `record` there is nothing to reclaim on a failed CAS.
        let alias = Self::alias(ptr);
        loop {
            match self.ptr.compare_exchange(
                ptr::null_mut(),
                alias,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: `target` retains the allocation for the lifetime
                    // guaranteed by the caller.
                    return (unsafe { self.get_raw(alias) }, true);
                }
                Err(actual) => match self.handle_failed_cas(actual) {
                    ControlFlow::Break(actual) => {
                        // SAFETY: The failed acquire CAS, or the subsequent wait,
                        // observed a published pointer retained by this table.
                        return (unsafe { self.get_raw(actual) }, false);
                    }
                    ControlFlow::Continue(()) => {}
                },
            }
        }
    }

    fn reserve(&self, value: AnswerBox<T>) -> bool {
        let mut candidate = Self::into_raw(value);
        loop {
            match self.ptr.compare_exchange(
                ptr::null_mut(),
                Self::pending(candidate),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => {
                    // SAFETY: The failed CAS left the candidate allocation with us.
                    // Owning it here keeps it live across the wait below, so an
                    // unwind drops it, and the losing path frees it on return.
                    let value = AnswerBox(unsafe { Box::from_raw(candidate) });
                    match self.handle_failed_cas(actual) {
                        ControlFlow::Break(_) => return false,
                        ControlFlow::Continue(()) => candidate = Self::into_raw(value),
                    }
                }
            }
        }
    }

    /// Make a reserved result visible. Calling this again after publication is
    /// harmless so an unwind guard can complete an interrupted commit.
    ///
    /// # Safety
    ///
    /// The caller must own this slot's reservation.
    unsafe fn publish_reserved(&self) {
        let pending = self.ptr.load(Ordering::Acquire);
        if !Self::is_pending(pending) {
            assert!(!pending.is_null(), "reserved SCC result disappeared");
            return;
        }
        self.ptr.store(Self::published(pending), Ordering::Release);
    }

    /// Cancel this owner's reservation if it is still pending.
    ///
    /// # Safety
    ///
    /// The caller must own this slot's reservation.
    unsafe fn rollback_reserved_if_pending(&self) -> bool {
        let pending = self.ptr.load(Ordering::Acquire);
        if !Self::is_pending(pending) {
            return false;
        }
        self.ptr.store(ptr::null_mut(), Ordering::Release);
        if !Self::is_alias(pending) {
            // SAFETY: The caller owns this pending allocation and removed it
            // before any reader could access it.
            unsafe { drop(Box::from_raw(Self::answer_ptr(pending))) };
        }
        true
    }
}

/// A published view of an answer slot owned by a complete `Solutions`.
/// `SolutionsData::new` creates slots only for exported bindings. Before
/// constructing `Solutions`, `Answers::solve` calls the infallible
/// `AnswersSolver::get_idx` for every exported binding. That call does not
/// return until the final result, including an SCC result, has been published.
/// Therefore every slot reachable through this view is published.
#[repr(transparent)]
struct SolutionSlot<'a, T: Send>(&'a AnswerSlot<T>);

impl<'a, T: Send> SolutionSlot<'a, T> {
    fn get(&self) -> &'a T {
        self.0.get_published()
    }
}

impl<T: Send> Drop for AnswerSlot<T> {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() && !Self::is_alias(ptr) {
            // SAFETY: Exclusive access proves that this slot owns the allocation.
            unsafe { drop(Box::from_raw(Self::answer_ptr(ptr))) };
        }
    }
}

impl<T: Send> Debug for AnswerSlot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = self.ptr.load(Ordering::Relaxed);
        let state = if ptr.is_null() {
            "unpublished"
        } else if Self::is_pending(ptr) {
            "pending"
        } else if Self::is_alias(ptr) {
            "alias"
        } else {
            "published"
        };
        f.debug_tuple("AnswerSlot").field(&state).finish()
    }
}

/// Result slots indexed identically to `Bindings`.
#[derive(Debug)]
pub struct AnswerEntry<K: Keyed>(Vec<AnswerSlot<K::Answer>>);

impl<K: Keyed> Default for AnswerEntry<K> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

/// `Answers::new` gives every binding a slot, so a lookup only fails when `idx`
/// came from different `Bindings` than the answers being indexed.
fn missing_answer_slot<K: Keyed>(idx: Idx<K>) -> ! {
    panic!(
        "no answer slot for {} at index {}; the index must come from the bindings these answers were built from",
        type_name::<K>(),
        idx.idx(),
    )
}

impl<K: Keyed> AnswerEntry<K> {
    fn answer_slot(&self, idx: Idx<K>) -> &AnswerSlot<K::Answer> {
        assert!(!K::EXPORTED, "exported answers live in SolutionsData");
        self.0
            .get(idx.idx())
            .unwrap_or_else(|| missing_answer_slot(idx))
    }

    fn get(&self, idx: Idx<K>) -> Option<&K::Answer> {
        self.answer_slot(idx).get()
    }

    fn record(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> (&K::Answer, bool) {
        self.answer_slot(idx).record(answer)
    }

    fn record_alias(&self, idx: Idx<K>, target: Idx<K>) -> (&K::Answer, bool) {
        let slot = self.answer_slot(idx);
        let target = self.answer_slot(target);
        // SAFETY: Both slots belong to this entry and are dropped together.
        unsafe { slot.record_alias(target) }
    }

    fn reserve(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> bool {
        self.answer_slot(idx).reserve(answer)
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn publish_reserved(&self, idx: Idx<K>) {
        // SAFETY: Forwarded from the caller.
        unsafe { self.answer_slot(idx).publish_reserved() }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn rollback_reserved_if_pending(&self, idx: Idx<K>) -> bool {
        // SAFETY: Forwarded from the caller.
        unsafe { self.answer_slot(idx).rollback_reserved_if_pending() }
    }
}

table!(
    #[derive(Debug, Default)]
    pub struct AnswerTable(pub AnswerEntry)
);

impl DisplayWith<Bindings> for Answers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, bindings: &Bindings) -> fmt::Result {
        fn go<K: Keyed>(
            answers: &Answers,
            bindings: &Bindings,
            _entry: &AnswerEntry<K>,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result
        where
            AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            for idx in bindings.keys::<K>() {
                let key = bindings.idx_to_key(idx);
                let value = bindings.get(idx);
                writeln!(
                    f,
                    "{} = {} = {}",
                    bindings.module().display(key),
                    value.display_with(bindings),
                    match answers.get_idx(idx) {
                        Some(v) => v.to_string(),
                        None => "(unsolved)".to_owned(),
                    },
                )?;
            }
            Ok(())
        }

        table_try_for_each!(self.table, |x| go(self, bindings, x, f));
        Ok(())
    }
}

#[derive(Debug)]
pub struct SolutionsEntry<K: Keyed>(SmallMap<K, AnswerSlot<K::Answer>>);

impl<K: Keyed> Default for SolutionsEntry<K> {
    fn default() -> Self {
        Self(SmallMap::new())
    }
}

impl<K: Keyed> Deref for SolutionsEntry<K> {
    type Target = SmallMap<K, AnswerSlot<K::Answer>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: Keyed> SolutionsEntry<K> {
    fn answer_slot(&self, idx: Idx<K>) -> &AnswerSlot<K::Answer> {
        match self.0.get_index(idx.idx()) {
            Some((_, slot)) => slot,
            None => missing_answer_slot(idx),
        }
    }

    fn get(&self, idx: Idx<K>) -> Option<&K::Answer> {
        self.answer_slot(idx).get()
    }

    fn record(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> (&K::Answer, bool) {
        self.answer_slot(idx).record(answer)
    }

    fn record_alias(&self, idx: Idx<K>, target: Idx<K>) -> (&K::Answer, bool) {
        let slot = self.answer_slot(idx);
        let target = self.answer_slot(target);
        // SAFETY: Both slots belong to this entry and are dropped together.
        unsafe { slot.record_alias(target) }
    }

    fn reserve(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> bool {
        self.answer_slot(idx).reserve(answer)
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn publish_reserved(&self, idx: Idx<K>) {
        // SAFETY: Forwarded from the caller.
        unsafe { self.answer_slot(idx).publish_reserved() }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn rollback_reserved_if_pending(&self, idx: Idx<K>) -> bool {
        // SAFETY: Forwarded from the caller.
        unsafe { self.answer_slot(idx).rollback_reserved_if_pending() }
    }

    fn answer_slot_hashed(&self, key: Hashed<&K>) -> Option<SolutionSlot<'_, K::Answer>> {
        Some(SolutionSlot(self.0.get_hashed(key)?))
    }

    fn answer_slot_key(&self, key: &K) -> Option<SolutionSlot<'_, K::Answer>> {
        Some(SolutionSlot(self.0.get(key)?))
    }

    fn answer_slots(&self) -> impl Iterator<Item = (&K, SolutionSlot<'_, K::Answer>)> {
        self.0.iter().map(|(key, slot)| (key, SolutionSlot(slot)))
    }
}

table!(
    // Only the exported keys are stored in the solutions table.
    #[derive(Default, Debug)]
    pub struct SolutionsTable(pub SolutionsEntry)
);

#[derive(Debug)]
pub(crate) struct SolutionsData {
    table: SolutionsTable,
}

impl SolutionsData {
    fn new(bindings: &Bindings) -> Self {
        fn presize<K: Keyed>(items: &mut SolutionsEntry<K>, bindings: &Bindings)
        where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        {
            if K::EXPORTED {
                let keys = bindings.keys::<K>();
                let len = keys.len();
                items.0.reserve(len);
                // `Bindings::keys` enumerates the binding storage and creates
                // each `Idx` from that same position. The table is never
                // structurally mutated after this loop, so each `SmallMap`
                // position remains identical to its binding `Idx`.
                for idx in keys {
                    items
                        .0
                        .insert(bindings.idx_to_key(idx).clone(), AnswerSlot::default());
                }
                assert_eq!(items.len(), len, "exported solution keys must be unique");
            }
        }

        let mut table = SolutionsTable::default();
        table_mut_for_each!(&mut table, |items| presize(items, bindings));
        Self { table }
    }

    fn get_idx<K: Keyed>(&self, idx: Idx<K>) -> Option<&K::Answer>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get::<K>().get(idx)
    }

    pub(crate) fn record<K: Keyed>(
        &self,
        idx: Idx<K>,
        answer: AnswerBox<K::Answer>,
    ) -> (&K::Answer, bool)
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get::<K>().record(idx, answer)
    }

    fn record_alias<K: Keyed>(&self, idx: Idx<K>, target: Idx<K>) -> (&K::Answer, bool)
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get::<K>().record_alias(idx, target)
    }

    fn reserve<K: Keyed>(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> bool
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get::<K>().reserve(idx, answer)
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn publish_reserved<K: Keyed>(&self, idx: Idx<K>)
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // SAFETY: Forwarded from the caller.
        unsafe { self.table.get::<K>().publish_reserved(idx) }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    unsafe fn rollback_reserved_if_pending<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // SAFETY: Forwarded from the caller.
        unsafe { self.table.get::<K>().rollback_reserved_if_pending(idx) }
    }
}

#[derive(Debug, Clone)]
pub struct Solutions {
    module_info: ModuleInfo,
    data: Arc<SolutionsData>,
    metadata: Arc<BindingsMetadata>,
    /// Multi-line ranges and ignore-all directives.
    module_ranges: Arc<ModuleRanges>,
    index: Option<Arc<Mutex<Index>>>,
    /// Per-module pysa data, populated when pysa reporting is enabled.
    pysa_solutions: Option<Arc<PysaSolutions>>,
    /// Per-module cinderx data, populated when cinderx reporting is enabled.
    cinderx_solutions: Option<Arc<CinderxSolutions>>,
}

impl Display for Solutions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn go<K: Keyed>(
            _entry: &SolutionsEntry<K>,
            solutions: &Solutions,
            f: &mut fmt::Formatter<'_>,
            ctx: &ModuleInfo,
        ) -> fmt::Result
        where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            for (key, slot) in solutions.solution_slots::<K>() {
                let answer = slot.get();
                writeln!(f, "{} = {}", ctx.display(key), answer)?;
            }
            Ok(())
        }

        table_try_for_each!(&self.data.table, |x| go(x, self, f, &self.module_info));
        Ok(())
    }
}

pub struct SolutionsDifference<'a> {
    key: (&'a dyn DisplayWith<ModuleInfo>, &'a dyn Debug),
    lhs: Option<(&'a dyn Display, &'a dyn Debug)>,
    rhs: Option<(&'a dyn Display, &'a dyn Debug)>,
}

impl Debug for SolutionsDifference<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SolutionsDifference")
            .field("key", self.key.1)
            .field("lhs", &self.lhs.map(|x| x.1))
            .field("rhs", &self.rhs.map(|x| x.1))
            .finish()
    }
}

impl Display for SolutionsDifference<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let missing = |f: &mut fmt::Formatter, x: Option<(&dyn Display, &dyn Debug)>| match x {
            None => write!(f, "missing"),
            Some(x) => write!(f, "`{}`", x.0),
        };

        // The key has type DisplayWith<ModuleInfo>.
        // We don't know if the key originates on the LHS or RHS, so we don't know which is the appropriate ModuleInfo.
        // However, we do know it is exported, and exported things can't rely on locations, so regardless
        // of the ModuleInfo, it should display the same. Therefore, we fake one up.
        let fake_module_info = ModuleInfo::new(
            ModuleName::from_str("Fake.Module.For.SolutionsDifference.Display"),
            ModulePath::memory(PathBuf::new()),
            Default::default(),
        );

        write!(f, "`")?;
        self.key.0.fmt(f, &fake_module_info)?;
        write!(f, "` was ")?;
        missing(f, self.lhs)?;
        write!(f, " now ")?;
        missing(f, self.rhs)?;
        Ok(())
    }
}

impl Solutions {
    #[inline]
    fn solution_slot_hashed<K: Keyed>(&self, key: Hashed<&K>) -> Option<SolutionSlot<'_, K::Answer>>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.data.table.get::<K>().answer_slot_hashed(key)
    }

    #[inline]
    fn solution_slots<K: Keyed>(&self) -> impl Iterator<Item = (&K, SolutionSlot<'_, K::Answer>)>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.data.table.get::<K>().answer_slots()
    }

    pub fn metadata(&self) -> &Arc<BindingsMetadata> {
        &self.metadata
    }

    pub fn module_ranges(&self) -> &Arc<ModuleRanges> {
        &self.module_ranges
    }

    /// Access per-module pysa data, if pysa reporting was enabled.
    pub fn pysa_solutions(&self) -> Option<&Arc<PysaSolutions>> {
        self.pysa_solutions.as_ref()
    }

    /// Access per-module cinderx data, if cinderx reporting was enabled.
    pub fn cinderx_solutions(&self) -> Option<&Arc<CinderxSolutions>> {
        self.cinderx_solutions.as_ref()
    }

    pub fn get<K: Exported>(&self, key: &K) -> &<K as Keyed>::Answer
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_hashed(Hashed::new(key))
    }

    pub fn get_hashed_opt<K: Exported>(&self, key: Hashed<&K>) -> Option<&<K as Keyed>::Answer>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        Some(self.solution_slot_hashed(key)?.get())
    }

    pub fn get_hashed<K: Exported>(&self, key: Hashed<&K>) -> &<K as Keyed>::Answer
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_hashed_opt(key).unwrap_or_else(|| {
            panic!(
                "Internal error: solution not found, module {}, path {}, key {:?}",
                self.module_info.name(),
                self.module_info.path(),
                key.key(),
            )
        })
    }

    /// Helper to create a difference for a key only in rhs.
    #[inline]
    fn make_only_in_rhs<'a, K: Keyed>(k: &'a K, v: &'a K::Answer) -> SolutionsDifference<'a> {
        SolutionsDifference {
            key: (k, k),
            lhs: None,
            rhs: Some((v, v)),
        }
    }

    /// Helper to create a difference for a key only in lhs.
    #[inline]
    fn make_only_in_lhs<'a, K: Keyed>(k: &'a K, v: &'a K::Answer) -> SolutionsDifference<'a> {
        SolutionsDifference {
            key: (k, k),
            lhs: Some((v, v)),
            rhs: None,
        }
    }

    /// Helper to create a difference for differing values.
    #[inline]
    fn make_value_differs<'a, K: Keyed>(
        k: &'a K,
        v1: &'a K::Answer,
        v2: &'a K::Answer,
    ) -> SolutionsDifference<'a> {
        SolutionsDifference {
            key: (k, k),
            lhs: Some((v1, v1)),
            rhs: Some((v2, v2)),
        }
    }

    /// Find the first key that differs between two solutions, with the two values.
    pub fn first_difference<'a>(&'a self, other: &'a Self) -> Option<SolutionsDifference<'a>> {
        fn f<'a, K: Keyed>(
            x: &'a SolutionsEntry<K>,
            y: &'a Solutions,
            ctx: &mut TypeEqCtx,
        ) -> Option<SolutionsDifference<'a>>
        where
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            if !K::EXPORTED {
                assert_eq!(x.len(), 0, "Expect no non-exported keys in Solutions");
                return None;
            }

            let y_table = y.data.table.get::<K>();
            if y_table.len() > x.len() {
                for (k, slot) in y_table.answer_slots() {
                    if !x.contains_key(k) {
                        let v = slot.get();
                        return Some(Solutions::make_only_in_rhs(k, v));
                    }
                }
                unreachable!();
            }
            for (k, slot) in x.answer_slots() {
                let v = slot.get();
                match y_table.answer_slot_key(k) {
                    Some(slot2) => {
                        let v2 = slot2.get();
                        if !v.type_eq(v2, ctx) {
                            return Some(Solutions::make_value_differs(k, v, v2));
                        }
                    }
                    None => {
                        return Some(Solutions::make_only_in_lhs(k, v));
                    }
                }
            }
            None
        }

        let mut difference = None;
        // Important we have a single TypeEqCtx, so that we don't have
        // types used in different ways.
        let mut ctx = TypeEqCtx::default();
        table_for_each!(self.data.table, |x| {
            if difference.is_none() {
                difference = f(x, other, &mut ctx);
            }
        });
        difference
    }

    /// Diff two solutions and merge changed keys into `changed`.
    ///
    /// For each exported key, records the change with the correct semantics:
    /// - Added/removed keys: existence change (default NameDep for name keys).
    /// - Value changed: type/metadata change (name still exists).
    pub fn changed_exports(&self, other: &Self, changed: &mut ModuleChanges) {
        fn check_table<K: Keyed>(
            x: &SolutionsEntry<K>,
            y: &Solutions,
            ctx: &mut TypeEqCtx,
            changed: &mut ModuleChanges,
        ) where
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            if !K::EXPORTED {
                return;
            }

            let y_table = y.data.table.get::<K>();

            // Check for items only in y (added keys) — existence change.
            for (k, _v) in y_table.answer_slots() {
                if !x.contains_key(k)
                    && let Some(anykey) = k.try_to_anykey()
                {
                    changed.add_key_existence(anykey);
                }
            }

            // Check for differences in x
            for (k, slot) in x.answer_slots() {
                let v = slot.get();
                match y_table.answer_slot_key(k) {
                    Some(slot2) => {
                        let v2 = slot2.get();
                        if v.type_eq(v2, ctx) {
                            continue;
                        }
                        // Value changed — type/metadata change, key still exists.
                        if let Some(anykey) = k.try_to_anykey() {
                            changed.add_key(anykey);
                        }
                    }
                    None => {
                        // Key removed — existence change.
                        if let Some(anykey) = k.try_to_anykey() {
                            changed.add_key_existence(anykey);
                        }
                    }
                }
            }
        }

        // Important we have a single TypeEqCtx, so that we don't have
        // types used in different ways.
        let mut ctx = TypeEqCtx::default();

        // Check all tables
        table_for_each!(self.data.table, |x| {
            check_table(x, other, &mut ctx, changed);
        });
    }

    /// Record exports that changed between new solutions (self) and old answers
    /// (bindings + answers) into `changed`. This is used when the old solutions
    /// were None but old answers exist — e.g., the module was previously only
    /// computed up to Answers and is now computed to Solutions for the first time.
    ///
    /// If a calculation in old answers was never forced, we skip it — nothing
    /// could have depended on it, so there's no change to propagate.
    pub fn changed_exports_vs_answers(
        &self,
        old_bindings: &Bindings,
        old_answers: &Answers,
        changed: &mut ModuleChanges,
    ) {
        fn check_table_vs_answers<K: Keyed>(
            new_solutions: &SolutionsEntry<K>,
            old_bindings: &Bindings,
            old_answers: &Answers,
            ctx: &mut TypeEqCtx,
            changed: &mut ModuleChanges,
        ) where
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
            AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        {
            if !K::EXPORTED {
                return;
            }

            for (k, slot) in new_solutions.answer_slots() {
                let new_val = slot.get();
                let Some(anykey) = k.try_to_anykey() else {
                    continue;
                };
                let hashed_k = Hashed::new(k);
                match old_bindings.key_to_idx_hashed_opt::<K>(hashed_k) {
                    Some(idx) => {
                        // Key existed in old answers — compare values.
                        match old_answers.get_idx::<K>(idx) {
                            Some(old_val) if !old_val.type_eq(new_val, ctx) => {
                                changed.add_key(anykey);
                            }
                            // None means the old answer was never computed, so
                            // no downstream module ever depended on this value.
                            // No change to propagate.
                            _ => {}
                        }
                    }
                    None => {
                        // Key didn't exist in old bindings — new export, treat as changed.
                        changed.add_key_existence(anykey);
                    }
                }
            }
        }

        let mut ctx = TypeEqCtx::default();

        table_for_each!(self.data.table, |x| {
            check_table_vs_answers(x, old_bindings, old_answers, &mut ctx, changed);
        });
    }

    pub fn get_index(&self) -> Option<Arc<Mutex<Index>>> {
        let index = self.index.as_ref()?;
        Some(index.dupe())
    }
}

pub trait LookupAnswer: Sized {
    /// Look up the value. If present, the `path` is a hint which can optimize certain cases.
    ///
    /// Return None if the file is undergoing concurrent modification.
    fn get<'answer, K: Solve<Self> + Exported>(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &K,
        stack: &'answer ThreadState,
        answer_scope: &'answer AnswerScope,
    ) -> Option<&'answer K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>;

    /// Drive a cross-module iteration member by calling `get_idx` in the
    /// target module's context.
    ///
    /// Used during iterative SCC solving when a member belongs to a different
    /// module than the current solver. The answer from `get_idx` is stored in
    /// iteration state on the shared `CalcStack` (via the shared `ThreadState`),
    /// so no return value is needed.
    ///
    /// Returns true if the driving was performed, false if the implementation
    /// does not support cross-module driving.
    ///
    /// Default implementation returns false (not supported).
    fn solve_idx_erased(
        &self,
        _calc_id: &CalcId,
        _thread_state: &ThreadState,
        _answer_scope: &AnswerScope,
    ) -> bool {
        false
    }

    /// Reserve a cross-module result slot for SCC batch publication.
    ///
    /// Returns the target Answers when reserved so the slot remains reachable
    /// until publication or rollback. The default implementation returns
    /// `None` (not supported).
    fn reserve_in_module(&self, _calc_id: &CalcId, _answer: AnyAnswer) -> Option<Arc<Answers>> {
        None
    }

    /// Publish a cross-module result slot previously reserved by this SCC.
    ///
    /// Default implementation returns false (not supported).
    fn publish_reserved_in_module(&self, _reserved: &mut ReservedSlot<'_, '_, '_, Self>) -> bool {
        false
    }

    /// Look up the class fields for a class, which may be defined in another
    /// module. The fields are populated during the binding phase and can be
    /// queried without going through the solve code path.
    ///
    /// Returns `None` if the `ClassDefIndex` is stale (e.g., the target module
    /// was rebuilt with fewer classes during incremental recompilation).
    ///
    /// Implementations must register a class-level dependency so that
    /// incremental rebuilds properly invalidate dependents when class
    /// fields change.
    fn get_class_fields(&self, cls: &Class) -> Option<&ClassFields>;
}

impl Answers {
    pub fn new(
        bindings: &Bindings,
        solver: Solver,
        enable_index: bool,
        enable_trace: bool,
    ) -> Self {
        fn presize<K: Keyed>(items: &mut AnswerEntry<K>, bindings: &Bindings)
        where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        {
            if !K::EXPORTED {
                let len = bindings.keys::<K>().len();
                items.0.resize_with(len, AnswerSlot::default);
            }
        }
        let mut table = AnswerTable::default();
        table_mut_for_each!(&mut table, |items| presize(items, bindings));
        let solutions = Arc::new(SolutionsData::new(bindings));
        let index = if enable_index {
            Some(Arc::new(Mutex::new(Index::default())))
        } else {
            None
        };
        let trace = if enable_trace {
            Some(Mutex::new(Traces::default()))
        } else {
            None
        };

        Self {
            solver,
            table,
            solutions,
            index,
            trace,
        }
    }

    pub fn table(&self) -> &AnswerTable {
        &self.table
    }

    pub(crate) fn record<K: Keyed>(
        &self,
        idx: Idx<K>,
        answer: AnswerBox<K::Answer>,
    ) -> (&K::Answer, bool)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            self.solutions.record(idx, answer)
        } else {
            self.table.get::<K>().record(idx, answer)
        }
    }

    pub(crate) fn record_alias<K: Keyed>(&self, idx: Idx<K>, target: Idx<K>) -> (&K::Answer, bool)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            self.solutions.record_alias(idx, target)
        } else {
            self.table.get::<K>().record_alias(idx, target)
        }
    }

    pub(crate) fn reserve<K: Keyed>(&self, idx: Idx<K>, answer: AnswerBox<K::Answer>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            self.solutions.reserve(idx, answer)
        } else {
            self.table.get::<K>().reserve(idx, answer)
        }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    pub(crate) unsafe fn publish_reserved<K: Keyed>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            // SAFETY: Forwarded from the caller.
            unsafe { self.solutions.publish_reserved(idx) }
        } else {
            // SAFETY: Forwarded from the caller.
            unsafe { self.table.get::<K>().publish_reserved(idx) }
        }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot.
    pub(crate) unsafe fn rollback_reserved_if_pending<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            // SAFETY: Forwarded from the caller.
            unsafe { self.solutions.rollback_reserved_if_pending(idx) }
        } else {
            // SAFETY: Forwarded from the caller.
            unsafe { self.table.get::<K>().rollback_reserved_if_pending(idx) }
        }
    }

    pub fn heap(&self) -> &TypeHeap {
        &self.solver.heap
    }

    pub fn solve<Ans: LookupAnswer>(
        &self,
        exports: &dyn LookupExport,
        answers: &Ans,
        bindings: &Bindings,
        errors: &ErrorCollector,
        stdlib: &Stdlib,
        uniques: &UniqueFactory,
        compute_everything: bool,
        recursion_limit_config: Option<RecursionLimitConfig>,
        pysa_context: Option<&crate::report::pysa::context::ModuleAnswersContext>,
        enable_cinderx_solutions: bool,
    ) -> Solutions {
        fn pre_solve<Ans: LookupAnswer, K: Solve<Ans>>(
            _items: &SolutionsEntry<K>,
            answers: &AnswersSolver<Ans>,
            compute_everything: bool,
        ) where
            AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            if !K::EXPORTED
                && !compute_everything
                && answers.base_errors().style() == ErrorStyle::Never
            {
                // No point doing anything here.
                return;
            }
            for idx in answers.bindings().keys::<K>() {
                let answer_scope = AnswerScope::new();
                answers.for_answer_scope(&answer_scope).get_idx(idx);
            }
        }
        let recurser = &VarRecurser::new();
        let thread_state = &ThreadState::new(recursion_limit_config);
        let answer_scope = &AnswerScope::new();
        let jaxtyping_dims = RefCell::default();
        let answers_solver = AnswersSolver::new(
            answers,
            self,
            errors,
            bindings,
            exports,
            uniques,
            recurser,
            stdlib,
            thread_state,
            answer_scope,
            self.heap(),
            &jaxtyping_dims,
        );
        table_for_each!(&self.solutions.table, |items| pre_solve(
            items,
            &answers_solver,
            compute_everything
        ));
        // Every binding is solved, so every variable a trace mentions now has an answer. Force
        // them here, once, rather than on every read. This must precede any trace consumer below.
        if let Some(trace_store) = &self.trace {
            trace_store
                .lock()
                .visit_mut(&mut |ty| self.solver.force_mut(ty));
        }
        // `pre_solve` has published every exported slot. From this point on,
        // the preallocated solutions table represents a complete result set.
        if let Some(index) = &self.index {
            let mut index = index.lock();
            // Index bindings with external definitions.
            for idx in bindings.keys::<Key>() {
                let key = bindings.idx_to_key(idx);
                let (imported_module_name, imported_name) =
                    match key_to_intermediate_definition(bindings, key) {
                        None => continue,
                        Some(IntermediateDefinition::Local(_)) => continue,
                        Some(IntermediateDefinition::Module(..)) => continue,
                        Some(IntermediateDefinition::NamedImport(
                            _import_key,
                            module_name,
                            name,
                            original_name_range,
                        )) => {
                            if let Some(original_name_range) = original_name_range {
                                index
                                    .renamed_imports
                                    .entry((module_name, name))
                                    .or_default()
                                    .push(original_name_range);
                                continue;
                            } else {
                                (module_name, name)
                            }
                        }
                    };

                let reference_range = bindings.idx_to_key(idx).range();
                // Sanity check: the reference should have the same text as the definition.
                // This check helps to filter out synthetic bindings.
                if bindings.module().code_at(reference_range) == imported_name.as_str() {
                    index
                        .externally_defined_variable_references
                        .entry((imported_module_name, imported_name))
                        .or_default()
                        .push(reference_range);
                }
            }
        }

        let pysa_solutions = pysa_context.map(PysaSolutions::build);
        let cinderx_solutions =
            enable_cinderx_solutions.then(|| CinderxSolutions::build(bindings, &answers_solver));

        answers_solver.validate_final_thread_state();

        Solutions {
            module_info: bindings.module().dupe(),
            data: self.solutions.dupe(),
            metadata: bindings.metadata().dupe(),
            module_ranges: bindings.module_ranges().dupe(),
            index: self.index.dupe(),
            pysa_solutions,
            cinderx_solutions,
        }
    }

    pub fn solve_exported_key<'ctx, 'answer, Ans: LookupAnswer, K: Solve<Ans> + Exported>(
        &'answer self,
        exports: &'ctx dyn LookupExport,
        answers: &'ctx Ans,
        bindings: &'answer Bindings,
        errors: &'ctx ErrorCollector,
        stdlib: &'ctx Stdlib,
        uniques: &'ctx UniqueFactory,
        key: Hashed<&K>,
        thread_state: &'answer ThreadState,
        answer_scope: &'answer AnswerScope,
    ) -> Option<&'answer K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // Fast path: check if the answer has already been published in its result slot.
        // This avoids constructing a VarRecurser and AnswersSolver when the value is cached.
        if let Some(idx) = bindings.key_to_idx_hashed_opt(key)
            && let Some(v) = self.get_idx(idx)
        {
            return Some(v);
        }
        // Slow path: need to compute the answer.
        let recurser = &VarRecurser::new();
        let jaxtyping_dims = RefCell::default();
        let solver = AnswersSolver::new(
            answers,
            self,
            errors,
            bindings,
            exports,
            uniques,
            recurser,
            stdlib,
            thread_state,
            answer_scope,
            self.heap(),
            &jaxtyping_dims,
        );
        solver.get_hashed_opt(key)
    }

    /// Borrow a published answer retained by this `Answers` instance.
    pub(crate) fn get_idx<K: Keyed>(&self, k: Idx<K>) -> Option<&K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if K::EXPORTED {
            self.solutions.get_idx(k)
        } else {
            self.table.get::<K>().get(k)
        }
    }

    /// Drive a cross-module iteration member by constructing a temporary
    /// `AnswersSolver` for this module and calling `get_idx` on the member.
    ///
    /// Target-side entry point for cross-module iterative driving. The answer
    /// is stored in SCC iteration state on the shared `CalcStack` (via
    /// `thread_state`), so the `get_idx` result is discarded.
    pub fn solve_idx_erased<Ans: LookupAnswer>(
        &self,
        any_idx: &AnyIdx,
        answers: &Ans,
        bindings: &Bindings,
        exports: &dyn LookupExport,
        errors: &ErrorCollector,
        stdlib: &Stdlib,
        uniques: &UniqueFactory,
        thread_state: &ThreadState,
        answer_scope: &AnswerScope,
    ) {
        let recurser = &VarRecurser::new();
        let jaxtyping_dims = RefCell::default();
        let solver = AnswersSolver::new(
            answers,
            self,
            errors,
            bindings,
            exports,
            uniques,
            recurser,
            stdlib,
            thread_state,
            answer_scope,
            self.heap(),
            &jaxtyping_dims,
        );
        dispatch_anyidx!(any_idx, solver, solve_idx_erased_typed);
    }

    /// Reserve a result slot for SCC batch publication.
    pub fn reserve_preliminary(&self, any_idx: &AnyIdx, answer: AnyAnswer) -> bool {
        dispatch_anyidx!(any_idx, self, reserve_typed, answer)
    }

    /// Publish a slot previously reserved by an SCC batch.
    pub fn publish_reserved_preliminary<Ans: LookupAnswer>(
        &self,
        reserved: &mut ReservedSlot<'_, '_, '_, Ans>,
    ) {
        let CalcId(_, any_idx) = reserved.calc_id().dupe();
        // SAFETY: `reserved` proves that this SCC owns the pending slot.
        unsafe { dispatch_anyidx!(&any_idx, self, publish_reserved_typed) }
    }

    /// Roll back a slot reserved by an SCC batch if it is still pending.
    pub fn rollback_reserved_if_pending_preliminary<Ans: LookupAnswer>(
        &self,
        reserved: &mut ReservedSlot<'_, '_, '_, Ans>,
    ) -> bool {
        let CalcId(_, any_idx) = reserved.calc_id().dupe();
        // SAFETY: `reserved` proves that this SCC owns the pending slot.
        unsafe { dispatch_anyidx!(&any_idx, self, rollback_reserved_if_pending_typed) }
    }

    fn reserve_typed<K: Keyed>(&self, idx: Idx<K>, answer: AnyAnswer) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        let typed_answer = answer
            .downcast::<K::Answer>()
            .expect("Answers::reserve_typed: type mismatch");
        self.reserve(idx, typed_answer)
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot, which a
    /// `&mut ReservedSlot` proves.
    unsafe fn publish_reserved_typed<K: Keyed>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // SAFETY: The caller derives `idx` from its exclusive `&mut ReservedSlot`,
        // which proves ownership of the reservation.
        unsafe { self.publish_reserved(idx) }
    }

    /// # Safety
    ///
    /// The caller must own the reservation for `idx`'s slot, which a
    /// `&mut ReservedSlot` proves.
    unsafe fn rollback_reserved_if_pending_typed<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // SAFETY: The caller derives `idx` from its exclusive `&mut ReservedSlot`,
        // which proves ownership of the reservation.
        unsafe { self.rollback_reserved_if_pending(idx) }
    }

    pub fn solver(&self) -> &Solver {
        &self.solver
    }

    /// Returns `true` if tracing is enabled for this module.
    pub(crate) fn tracing_enabled(&self) -> bool {
        self.trace.is_some()
    }

    /// Merge accumulated trace side effects into the persisted trace store.
    /// No-op if tracing is not enabled.
    pub(crate) fn merge_trace_side_effects(&self, side_effects: TraceSideEffects) {
        if let Some(trace_store) = &self.trace {
            trace_store.lock().merge(side_effects);
        }
    }

    pub fn get_type_at(&self, idx: Idx<Key>) -> Option<Type> {
        Some(self.get_idx(idx)?.ty().clone())
    }

    pub fn get_type_at_for_display(&self, idx: Idx<Key>) -> Option<Type> {
        Some(self.get_idx(idx)?.ty().clone())
    }

    pub fn get_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.types.get(&range)?.as_ref().clone())
    }

    pub fn get_expected_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.expected_types.get(&range)?.as_ref().clone())
    }

    pub fn get_type_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.types.get(&range)?.as_ref().clone())
    }

    pub fn get_expected_type_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.expected_types.get(&range)?.as_ref().clone())
    }

    pub fn try_get_getter_for_range(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.invoked_properties.get(&range)?.as_ref().clone())
    }

    pub fn get_chosen_overload_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => Some(callable.as_type()),
            OverloadedCallee::Candidates {
                closest,
                is_closest_chosen,
                ..
            } if *is_closest_chosen => Some(closest.as_type()),
            _ => None,
        }
    }

    pub fn get_chosen_overload_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => Some(callable.as_type()),
            OverloadedCallee::Candidates {
                closest,
                is_closest_chosen,
                ..
            } if *is_closest_chosen => Some(closest.as_type()),
            _ => None,
        }
    }

    /// Returns all the overload, and the index of a chosen one
    pub fn get_all_overload_trace(
        &self,
        range: TextRange,
    ) -> Option<(Vec<Callable>, Option<usize>)> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => {
                Some((vec![callable.callable.clone()], Some(0)))
            }
            OverloadedCallee::Candidates { all, closest, .. } => {
                let chosen_index = all
                    .iter()
                    .position(|signature| signature.callable == closest.callable);
                let signatures = all.iter().map(|trace| trace.callable.clone()).collect();
                Some((signatures, chosen_index))
            }
        }
    }

    pub fn add_parent_method_mapping(
        &self,
        child_range: TextRange,
        parent_module: ModulePath,
        parent_range: TextRange,
    ) {
        if let Some(index) = &self.index {
            index
                .lock()
                .parent_methods_map
                .entry(child_range)
                .or_default()
                .push((parent_module, parent_range));
        }
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    pub fn solver(&self) -> &Solver {
        &self.current().solver
    }

    pub fn record_resolved_trace(&self, loc: TextRange, ty: &Type) {
        if self.current().trace.is_some()
            && let Some(callable) = ty.clone().to_callable()
        {
            self.trace_state().record_resolved_trace(
                loc,
                OverloadedCallee::Resolved {
                    callable: OverloadTrace::new(callable, None),
                },
            );
        }
    }

    /// Record all the overloads and the chosen overload.
    /// The trace will be used to power signature help and hover for overloaded functions.
    pub(crate) fn record_overload_trace(
        &self,
        loc: TextRange,
        all_overloads: Vec<OverloadTrace>,
        closest_overload: OverloadTrace,
        is_closest_overload_chosen: bool,
    ) {
        if self.current().trace.is_some() {
            self.trace_state().record_overload_trace(
                loc,
                OverloadedCallee::Candidates {
                    all: all_overloads,
                    closest: closest_overload,
                    is_closest_chosen: is_closest_overload_chosen,
                },
            );
        }
    }

    pub fn record_attribute_definition_index(
        &self,
        base: &Type,
        attribute_name: &Name,
        attribute_reference_range: TextRange,
        reference_kind: AttributeReferenceKind,
    ) {
        if let Some(index) = &self.current().index {
            for AttrInfo {
                name: _,
                ty: _,
                is_deprecated: _,
                definition,
                is_reexport: _,
            } in self.completions(base.clone(), Some(attribute_name), false)
            {
                match definition {
                    AttrDefinition::FullyResolved {
                        cls,
                        range,
                        docstring_range: _,
                    } => match reference_kind {
                        AttributeReferenceKind::ConstructorCall => index
                            .lock()
                            .constructor_references
                            .entry(cls.module_path().dupe())
                            .or_default()
                            .push((range, attribute_reference_range)),
                        AttributeReferenceKind::Textual => {
                            // Textual references to an attribute defined in this module are
                            // recovered by scanning the AST for `<expr>.<name>`, so only
                            // out-of-module definitions need an index entry.
                            if cls.module_path() != self.bindings().module().path() {
                                index
                                    .lock()
                                    .externally_defined_attribute_references
                                    .entry(cls.module_path().dupe())
                                    .or_default()
                                    .push((range, attribute_reference_range))
                            }
                        }
                    },
                    AttrDefinition::PartiallyResolvedImportedModuleAttribute { module_name } => {
                        index
                            .lock()
                            .externally_defined_variable_references
                            .entry((module_name, attribute_name.clone()))
                            .or_default()
                            .push(attribute_reference_range);
                    }
                    AttrDefinition::Submodule { module_name } => {
                        // For submodule access (e.g., `b` in `a.b`), record as a reference to
                        // the submodule. The last component of module_name is the attribute name.
                        if let Some(parent) = module_name.parent() {
                            index
                                .lock()
                                .externally_defined_variable_references
                                .entry((parent, attribute_name.clone()))
                                .or_default()
                                .push(attribute_reference_range);
                        }
                    }
                }
            }
        }
    }

    pub fn record_property_getter(&self, loc: TextRange, getter_ty: &Type) {
        if self.current().trace.is_some() {
            self.trace_state()
                .record_property_getter_trace(loc, Arc::new(getter_ty.clone()));
        }
    }

    pub fn record_type_trace(&self, loc: TextRange, ty: &Type) {
        if self.current().trace.is_some() && !loc.is_empty() {
            self.trace_state()
                .record_type_trace(loc, Arc::new(ty.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::thread;

    use super::*;

    #[test]
    fn answer_slot_publishes_one_value() {
        let slot = AnswerSlot::default();

        let (answer, did_write) = slot.record(AnswerBox::new(1));
        assert!(did_write);
        assert_eq!(*answer, 1);

        let (answer, did_write) = slot.record(AnswerBox::new(2));
        assert!(!did_write);
        assert_eq!(*answer, 1);
        assert_eq!(
            *slot
                .get()
                .expect("the winning answer should remain published"),
            1,
        );
    }

    #[test]
    fn answer_slot_concurrent_publish_has_one_winner() {
        const WRITERS: usize = 8;

        let slot = Arc::new(AnswerSlot::default());
        let barrier = Arc::new(Barrier::new(WRITERS));
        let writers = (0..WRITERS)
            .map(|value| {
                let slot = slot.dupe();
                let barrier = barrier.dupe();
                thread::spawn(move || {
                    barrier.wait();
                    let (answer, did_write) = slot.record(AnswerBox::new(value));
                    (*answer, did_write)
                })
            })
            .collect::<Vec<_>>();
        let results = writers
            .into_iter()
            .map(|writer| writer.join().expect("writer thread should not panic"))
            .collect::<Vec<_>>();

        assert_eq!(
            results.iter().filter(|(_, did_write)| *did_write).count(),
            1
        );
        let winner = results
            .iter()
            .find_map(|(answer, did_write)| did_write.then_some(*answer))
            .expect("one writer must win");
        assert!(results.iter().all(|(answer, _)| *answer == winner));
    }

    #[test]
    fn answer_slot_owns_one_value() {
        let value = Arc::new(1);
        let weak = Arc::downgrade(&value);
        let slot = AnswerSlot::default();
        let (answer, did_write) = slot.record(AnswerBox::new(value));
        assert!(did_write);
        assert_eq!(**answer, 1);
        assert_eq!(
            weak.strong_count(),
            1,
            "only the slot should retain the published answer"
        );

        assert!(weak.upgrade().is_some());
        drop(slot);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn answer_slot_alias_drops_shared_answer() {
        struct SetOnDrop<'a>(&'a AtomicBool);

        impl Drop for SetOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }

        let dropped = AtomicBool::new(false);
        {
            let slots = [AnswerSlot::default(), AnswerSlot::default()];
            let _ = slots[0].record(AnswerBox::new(SetOnDrop(&dropped)));
            // SAFETY: The array models one result table, and both slots remain
            // live until the returned reference is discarded.
            let _ = unsafe { slots[1].record_alias(&slots[0]) };
        }
        assert!(dropped.load(Ordering::Relaxed));
    }

    #[test]
    fn answer_slot_pending_alias_does_not_own_the_box() {
        let target = AnswerSlot::default();
        let published_alias = AnswerSlot::default();
        let rolled_back_alias = AnswerSlot::default();
        let value = Arc::new(1);
        let weak = Arc::downgrade(&value);

        target.record(AnswerBox::new(value));
        let target_ptr = target.ptr.load(Ordering::Acquire);
        published_alias.ptr.store(
            AnswerSlot::pending(AnswerSlot::alias(target_ptr)),
            Ordering::Release,
        );
        rolled_back_alias.ptr.store(
            AnswerSlot::pending(AnswerSlot::alias(target_ptr)),
            Ordering::Release,
        );

        // SAFETY: This test installed both pending reservations above.
        unsafe { published_alias.publish_reserved() };
        // SAFETY: This test installed both pending reservations above.
        unsafe { rolled_back_alias.rollback_reserved_if_pending() };
        assert_eq!(
            **published_alias.get().expect("alias should be published"),
            1
        );
        assert!(rolled_back_alias.get().is_none());

        drop(published_alias);
        drop(rolled_back_alias);
        assert!(weak.upgrade().is_some());
        drop(target);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn answer_slot_can_be_reused_after_reservation_rollback() {
        let slot = AnswerSlot::default();
        assert!(slot.reserve(AnswerBox::new(1)));
        // SAFETY: This test successfully reserved the slot above.
        unsafe { slot.rollback_reserved_if_pending() };
        assert!(slot.get().is_none());

        let (answer, did_write) = slot.record(AnswerBox::new(2));
        assert!(did_write);
        assert_eq!(*answer, 2);
    }
}
