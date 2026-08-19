/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::Any;
use std::cell::UnsafeCell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hint::spin_loop;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
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
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;

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
    index: Option<Arc<Mutex<Index>>>,
    trace: Option<Mutex<Traces>>,
}

const PUBLISH_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PUBLISH_BACKOFF_STEP: u32 = 6;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerStatus {
    /// No writer owns the slot and `result` is uninitialized. A writer may
    /// claim the slot by changing this state to `Pending`.
    Unpublished,
    /// One writer exclusively owns `result`. An ordinary writer may still be
    /// initializing it, while an SCC writer may be withholding an initialized
    /// result until the entire SCC is ready. Readers must wait; an SCC rollback
    /// returns the slot to `Unpublished`.
    Pending,
    /// `result` is initialized, immutable, and visible to readers. The release
    /// transition to this state synchronizes readers before they clone the Arc.
    Published,
}

impl AnswerStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            value if value == Self::Unpublished as u8 => Self::Unpublished,
            value if value == Self::Pending as u8 => Self::Pending,
            value if value == Self::Published as u8 => Self::Published,
            _ => unreachable!("invalid answer status {value}"),
        }
    }
}

/// A first-write-wins result slot. `Pending` reserves the separate result
/// storage while an ordinary result or complete SCC batch is being published.
pub struct AnswerSlot<T> {
    status: AtomicU8,
    result: UnsafeCell<MaybeUninit<Arc<T>>>,
}

impl<T> Default for AnswerSlot<T> {
    fn default() -> Self {
        Self {
            status: AtomicU8::new(AnswerStatus::Unpublished as u8),
            result: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// SAFETY: A writer must reserve the slot before initializing `result`, then
// publish it with a release store. The result is immutable while published.
unsafe impl<T: Send + Sync> Sync for AnswerSlot<T> {}

impl<T> AnswerSlot<T> {
    #[inline]
    fn load(&self, ordering: Ordering) -> AnswerStatus {
        AnswerStatus::from_u8(self.status.load(ordering))
    }

    /// Borrow the initialized result stored in this slot.
    ///
    /// # Safety
    ///
    /// `result` must be initialized and remain immutable for this call.
    unsafe fn result(&self) -> &Arc<T> {
        // SAFETY: The function contract guarantees that the result is initialized.
        unsafe { (*self.result.get()).assume_init_ref() }
    }

    /// Initialize storage after this thread changes the status from unpublished
    /// to pending.
    ///
    /// # Safety
    ///
    /// The slot must be pending and its result storage uninitialized.
    unsafe fn initialize(&self, value: Arc<T>) {
        // SAFETY: The function contract guarantees exclusive access to
        // uninitialized result storage.
        unsafe { (*self.result.get()).write(value) };
    }

    /// Wait for a pending writer to publish its result. Returns `false` only if
    /// panic unwinding rolls back an SCC reservation before publication.
    #[cold]
    #[inline(never)]
    fn wait_for_publish(&self) -> bool {
        let deadline = Instant::now() + PUBLISH_TIMEOUT;
        let mut backoff_step = 0;
        loop {
            match self.load(Ordering::Relaxed) {
                AnswerStatus::Pending => {}
                AnswerStatus::Published => {
                    // The relaxed load observed the release publication, so
                    // this fence acquires initialization of the result.
                    fence(Ordering::Acquire);
                    return true;
                }
                AnswerStatus::Unpublished => {
                    // A pending slot can return to Unpublished only when panic
                    // unwinding rolls back an SCC reservation.
                    return false;
                }
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

    /// Return the published value, waiting only when a writer already owns the slot.
    #[inline]
    pub(crate) fn get(&self) -> Option<&T> {
        match self.load(Ordering::Acquire) {
            AnswerStatus::Unpublished => None,
            AnswerStatus::Pending => self
                .wait_for_publish()
                // SAFETY: A true result proves that publication initialized the result.
                .then(|| unsafe { self.result() }.as_ref()),
            AnswerStatus::Published => {
                // SAFETY: Published status proves that the result is initialized.
                Some(unsafe { self.result() }.as_ref())
            }
        }
    }

    /// Clone the `Arc` owned by a published slot.
    #[inline]
    pub(crate) fn get_arc(&self) -> Option<Arc<T>> {
        self.get()?;
        // SAFETY: `get` returned a reference only after publication initialized
        // the result, which remains immutable.
        Some(unsafe { self.result() }.dupe())
    }

    /// Publish an ordinary, non-SCC result or return the competing winner.
    pub(crate) fn record(&self, value: Arc<T>) -> (Arc<T>, bool) {
        loop {
            match self.status.compare_exchange(
                AnswerStatus::Unpublished as u8,
                AnswerStatus::Pending as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: The successful CAS reserved uninitialized storage.
                    unsafe { self.initialize(value) };
                    self.status
                        .store(AnswerStatus::Published as u8, Ordering::Release);
                    // SAFETY: This thread initialized the now-published result.
                    return (unsafe { self.result() }.dupe(), true);
                }
                Err(actual) => {
                    let actual = AnswerStatus::from_u8(actual);
                    match actual {
                        AnswerStatus::Unpublished => {
                            unreachable!("failed CAS cannot observe the expected state")
                        }
                        AnswerStatus::Pending => {
                            if !self.wait_for_publish() {
                                continue;
                            }
                            // SAFETY: A true result proves that publication
                            // initialized the result.
                            return (unsafe { self.result() }.dupe(), false);
                        }
                        AnswerStatus::Published => {
                            // SAFETY: Published status proves that the result is initialized.
                            return (unsafe { self.result() }.dupe(), false);
                        }
                    }
                }
            }
        }
    }

    /// Reserve this result slot, waiting for a concurrent publisher.
    pub(crate) fn reserve(&self, value: Arc<T>) -> bool {
        loop {
            match self.status.compare_exchange(
                AnswerStatus::Unpublished as u8,
                AnswerStatus::Pending as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: The successful CAS reserved uninitialized storage.
                    unsafe { self.initialize(value) };
                    return true;
                }
                Err(actual) => {
                    let actual = AnswerStatus::from_u8(actual);
                    match actual {
                        AnswerStatus::Unpublished => {
                            unreachable!("failed CAS cannot observe the expected state")
                        }
                        AnswerStatus::Pending => {
                            if !self.wait_for_publish() {
                                continue;
                            }
                            return false;
                        }
                        AnswerStatus::Published => return false,
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
    /// The caller must own this slot's pending reservation. Only the owner may
    /// access the result while the slot is pending; other threads wait for
    /// publication. Ownership therefore proves that initialization has finished
    /// and no other thread can publish or roll back the result concurrently.
    pub(crate) unsafe fn publish_reserved(&self) {
        match self.load(Ordering::Acquire) {
            AnswerStatus::Pending => self
                .status
                .store(AnswerStatus::Published as u8, Ordering::Release),
            AnswerStatus::Published => {}
            AnswerStatus::Unpublished => panic!("reserved SCC result disappeared"),
        }
    }

    /// Cancel this owner's reservation if it is still pending.
    ///
    /// # Safety
    ///
    /// If the slot is pending, the caller must own its reservation. Only the
    /// owner may access the pending result; other threads wait for publication.
    /// Ownership therefore permits moving the initialized result out without
    /// racing another access to its storage.
    pub(crate) unsafe fn rollback_reserved_if_pending(&self) -> bool {
        match self.load(Ordering::Acquire) {
            AnswerStatus::Pending => {
                // SAFETY: Only the SCC owner changes a pending slot, so it owns
                // the initialized result. Move it out before making the slot reusable.
                let value = unsafe { (*self.result.get()).assume_init_read() };
                self.status
                    .store(AnswerStatus::Unpublished as u8, Ordering::Release);
                drop(value);
                true
            }
            AnswerStatus::Unpublished | AnswerStatus::Published => false,
        }
    }
}

impl<T> Drop for AnswerSlot<T> {
    fn drop(&mut self) {
        match AnswerStatus::from_u8(*self.status.get_mut()) {
            AnswerStatus::Unpublished => {}
            AnswerStatus::Pending | AnswerStatus::Published => {
                // SAFETY: Both states have initialized result storage, and
                // exclusive access proves that no reader can use it.
                unsafe { self.result.get_mut().assume_init_drop() };
            }
        }
    }
}

impl<T> Debug for AnswerSlot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AnswerSlot")
            .field(&self.load(Ordering::Relaxed))
            .finish()
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

impl<K: Keyed> AnswerEntry<K> {
    fn answer_slot(&self, idx: Idx<K>) -> Option<&AnswerSlot<K::Answer>> {
        self.0.get(idx.idx())
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

pub type SolutionsEntry<K> = SmallMap<K, Arc<<K as Keyed>::Answer>>;

table!(
    // Only the exported keys are stored in the solutions table.
    #[derive(Default, Debug, Clone, PartialEq, Eq)]
    pub struct SolutionsTable(pub SolutionsEntry)
);

#[derive(Debug, Clone)]
pub struct Solutions {
    module_info: ModuleInfo,
    table: SolutionsTable,
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
            entry: &SolutionsEntry<K>,
            f: &mut fmt::Formatter<'_>,
            ctx: &ModuleInfo,
        ) -> fmt::Result
        where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        {
            for (key, answer) in entry {
                writeln!(f, "{} = {}", ctx.display(key), answer)?;
            }
            Ok(())
        }

        table_try_for_each!(&self.table, |x| go(x, f, &self.module_info));
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
        self.table.get().get_hashed(key).map(Arc::as_ref)
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

    pub fn get_arc<K: Exported>(&self, key: &K) -> Arc<<K as Keyed>::Answer>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_hashed_arc(Hashed::new(key))
    }

    pub fn get_hashed_arc_opt<K: Exported>(
        &self,
        key: Hashed<&K>,
    ) -> Option<Arc<<K as Keyed>::Answer>>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get().get_hashed(key).map(|value| value.dupe())
    }

    pub fn get_hashed_arc<K: Exported>(&self, key: Hashed<&K>) -> Arc<<K as Keyed>::Answer>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_hashed_arc_opt(key).unwrap_or_else(|| {
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
    fn make_only_in_rhs<'a, K: Keyed>(k: &'a K, v: &'a Arc<K::Answer>) -> SolutionsDifference<'a> {
        SolutionsDifference {
            key: (k, k),
            lhs: None,
            rhs: Some((v, v)),
        }
    }

    /// Helper to create a difference for a key only in lhs.
    #[inline]
    fn make_only_in_lhs<'a, K: Keyed>(k: &'a K, v: &'a Arc<K::Answer>) -> SolutionsDifference<'a> {
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
        v1: &'a Arc<K::Answer>,
        v2: &'a Arc<K::Answer>,
    ) -> SolutionsDifference<'a> {
        SolutionsDifference {
            key: (k, k),
            lhs: Some((v1, v1)),
            rhs: Some((v2, v2)),
        }
    }

    /// Find the first key that differs between two solutions, with the two values.
    ///
    /// Don't love that we always allocate String's for the result, but it's rare that
    /// there is a difference, and if there is, we'll do quite a lot of computation anyway.
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

            let y_table = y.table.get::<K>();
            if y_table.len() > x.len() {
                for (k, v) in y_table {
                    if !x.contains_key(k) {
                        return Some(Solutions::make_only_in_rhs(k, v));
                    }
                }
                unreachable!();
            }
            for (k, v) in x {
                match y_table.get(k) {
                    Some(v2) if !v.type_eq(v2, ctx) => {
                        return Some(Solutions::make_value_differs(k, v, v2));
                    }
                    None => {
                        return Some(Solutions::make_only_in_lhs(k, v));
                    }
                    _ => {}
                }
            }
            None
        }

        let mut difference = None;
        // Important we have a single TypeEqCtx, so that we don't have
        // types used in different ways.
        let mut ctx = TypeEqCtx::default();
        table_for_each!(self.table, |x| {
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

            let y_table = y.table.get::<K>();

            // Check for items only in y (added keys) — existence change.
            for (k, _v) in y_table {
                if !x.contains_key(k)
                    && let Some(anykey) = k.try_to_anykey()
                {
                    changed.add_key_existence(anykey);
                }
            }

            // Check for differences in x
            for (k, v) in x {
                match y_table.get(k) {
                    Some(v2) if !v.type_eq(v2, ctx) => {
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
                    _ => {}
                }
            }
        }

        // Important we have a single TypeEqCtx, so that we don't have
        // types used in different ways.
        let mut ctx = TypeEqCtx::default();

        // Check all tables
        table_for_each!(self.table, |x| {
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

            for (k, new_val) in new_solutions {
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

        table_for_each!(self.table, |x| {
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
    fn get<K: Solve<Self> + Exported>(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &K,
        stack: &ThreadState,
    ) -> Option<Arc<K::Answer>>
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
    fn solve_idx_erased(&self, _calc_id: &CalcId, _thread_state: &ThreadState) -> bool {
        false
    }

    /// Reserve a cross-module result slot for SCC batch publication.
    ///
    /// Returns the target Answers when reserved so the slot remains reachable
    /// until publication or rollback. The default implementation returns
    /// `None` (not supported).
    fn reserve_in_module(
        &self,
        _calc_id: &CalcId,
        _answer: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<Answers>> {
        None
    }

    /// Publish a cross-module result slot previously reserved by this SCC.
    ///
    /// Default implementation returns false (not supported).
    fn publish_reserved_in_module(&self, _reserved: &mut ReservedSlot<'_, '_, Self>) -> bool {
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
            items
                .0
                .resize_with(bindings.keys::<K>().len(), AnswerSlot::default);
        }
        let mut table = AnswerTable::default();
        table_mut_for_each!(&mut table, |items| presize(items, bindings));
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
            index,
            trace,
        }
    }

    pub fn table(&self) -> &AnswerTable {
        &self.table
    }

    pub(crate) fn answer_slot<K: Keyed>(&self, idx: Idx<K>) -> Option<&AnswerSlot<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        self.table.get::<K>().answer_slot(idx)
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
        let mut res = SolutionsTable::default();

        fn pre_solve<Ans: LookupAnswer, K: Solve<Ans>>(
            items: &mut SolutionsEntry<K>,
            answers: &AnswersSolver<Ans>,
            compute_everything: bool,
        ) where
            AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        {
            if K::EXPORTED {
                items.reserve(answers.bindings().keys::<K>().len());
            }
            if !K::EXPORTED
                && !compute_everything
                && answers.base_errors().style() == ErrorStyle::Never
            {
                // No point doing anything here.
                return;
            }
            for idx in answers.bindings().keys::<K>() {
                let v = answers.get_idx(idx);
                if K::EXPORTED {
                    let k = answers.bindings().idx_to_key(idx);
                    items.insert(k.clone(), v.dupe());
                }
            }
        }
        let recurser = &VarRecurser::new();
        let thread_state = &ThreadState::new(recursion_limit_config);
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
            self.heap(),
        );
        table_mut_for_each!(&mut res, |items| pre_solve(
            items,
            &answers_solver,
            compute_everything
        ));
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
            table: res,
            metadata: bindings.metadata().dupe(),
            module_ranges: bindings.module_ranges().dupe(),
            index: self.index.dupe(),
            pysa_solutions,
            cinderx_solutions,
        }
    }

    pub fn solve_exported_key<Ans: LookupAnswer, K: Solve<Ans> + Exported>(
        &self,
        exports: &dyn LookupExport,
        answers: &Ans,
        bindings: &Bindings,
        errors: &ErrorCollector,
        stdlib: &Stdlib,
        uniques: &UniqueFactory,
        key: Hashed<&K>,
        thread_state: &ThreadState,
    ) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
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
            self.heap(),
        );
        solver.get_hashed_opt(key)
    }

    pub fn get_idx<K: Keyed>(&self, k: Idx<K>) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        self.answer_slot(k)?.get_arc()
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
    ) {
        let recurser = &VarRecurser::new();
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
            self.heap(),
        );
        dispatch_anyidx!(any_idx, solver, solve_idx_erased_typed);
    }

    /// Reserve a result slot for SCC batch publication.
    pub fn reserve_preliminary(
        &self,
        any_idx: &AnyIdx,
        answer: Arc<dyn Any + Send + Sync>,
    ) -> bool {
        dispatch_anyidx!(any_idx, self, reserve_typed, answer)
    }

    /// Publish a slot previously reserved by an SCC batch.
    pub fn publish_reserved_preliminary<Ans: LookupAnswer>(
        &self,
        reserved: &mut ReservedSlot<'_, '_, Ans>,
    ) -> bool {
        let CalcId(_, any_idx) = reserved.calc_id().dupe();
        // SAFETY: `reserved` proves that this SCC owns the pending slot.
        unsafe { dispatch_anyidx!(&any_idx, self, publish_reserved_typed) }
    }

    /// Roll back a slot reserved by an SCC batch if it is still pending.
    pub fn rollback_reserved_if_pending_preliminary<Ans: LookupAnswer>(
        &self,
        reserved: &mut ReservedSlot<'_, '_, Ans>,
    ) -> bool {
        let CalcId(_, any_idx) = reserved.calc_id().dupe();
        // SAFETY: `reserved` proves that this SCC owns the pending slot.
        unsafe { dispatch_anyidx!(&any_idx, self, rollback_reserved_if_pending_typed) }
    }

    fn reserve_typed<K: Keyed>(&self, idx: Idx<K>, answer: Arc<dyn Any + Send + Sync>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        let typed_answer: Arc<K::Answer> = Arc::unwrap_or_clone(
            answer
                .downcast::<Arc<K::Answer>>()
                .expect("Answers::reserve_typed: type mismatch"),
        );
        let Some(slot) = self.answer_slot(idx) else {
            return false;
        };
        slot.reserve(typed_answer)
    }

    unsafe fn publish_reserved_typed<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        let Some(slot) = self.answer_slot(idx) else {
            return false;
        };
        // SAFETY: The caller derives `idx` from its exclusive `&mut ReservedSlot`,
        // which proves ownership of this pending reservation.
        unsafe { slot.publish_reserved() };
        true
    }

    unsafe fn rollback_reserved_if_pending_typed<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        let Some(slot) = self.answer_slot(idx) else {
            return false;
        };
        // SAFETY: The caller derives `idx` from its exclusive `&mut ReservedSlot`,
        // which proves ownership of this pending reservation.
        unsafe { slot.rollback_reserved_if_pending() }
    }

    fn force_for_export_boundary(&self, t: Type) -> Type {
        self.solver.for_export_boundary(t)
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
        Some(self.force_for_export_boundary(self.get_idx(idx)?.arc_clone_ty()))
    }

    pub fn get_type_at_for_display(&self, idx: Idx<Key>) -> Option<Type> {
        Some(self.solver.for_display(self.get_idx(idx)?.arc_clone_ty()))
    }

    pub fn get_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(self.force_for_export_boundary(lock.types.get(&range)?.as_ref().clone()))
    }

    pub fn get_expected_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(self.force_for_export_boundary(lock.expected_types.get(&range)?.as_ref().clone()))
    }

    pub fn get_type_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(
            self.solver
                .for_display(lock.types.get(&range)?.as_ref().clone()),
        )
    }

    pub fn get_expected_type_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(
            self.solver
                .for_display(lock.expected_types.get(&range)?.as_ref().clone()),
        )
    }

    pub fn try_get_getter_for_range(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(self.force_for_export_boundary(lock.invoked_properties.get(&range)?.as_ref().clone()))
    }

    pub fn get_chosen_overload_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => {
                Some(self.force_for_export_boundary(callable.as_type()))
            }
            OverloadedCallee::Candidates {
                closest,
                is_closest_chosen,
                ..
            } if *is_closest_chosen => Some(self.force_for_export_boundary(closest.as_type())),
            _ => None,
        }
    }

    pub fn get_chosen_overload_trace_for_display(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => {
                Some(self.solver.for_display(callable.as_type()))
            }
            OverloadedCallee::Candidates {
                closest,
                is_closest_chosen,
                ..
            } if *is_closest_chosen => Some(self.solver.for_display(closest.as_type())),
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

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn get_answer_slot<K: Solve<Ans>>(&self, idx: Idx<K>) -> &AnswerSlot<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.current().answer_slot(idx).unwrap_or_else(|| {
            // Do not fix a panic by removing this error.
            // We should always be sure before calling `get`.
            panic!(
                "Internal error: answer not found, module {}, path {}, key {:?}",
                self.module().name(),
                self.module().path(),
                self.bindings().idx_to_key(idx),
            )
        })
    }

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
    use std::thread;

    use super::*;

    #[test]
    fn answer_slot_publishes_one_arc() {
        let slot = AnswerSlot::default();

        let (answer, did_write) = slot.record(Arc::new(1));
        assert!(did_write);
        assert_eq!(*answer, 1);

        let (answer, did_write) = slot.record(Arc::new(2));
        assert!(!did_write);
        assert_eq!(*answer, 1);
        assert_eq!(
            *slot
                .get()
                .expect("the winning answer should remain published"),
            1
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
                    let (answer, did_write) = slot.record(Arc::new(value));
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
    fn answer_slot_owns_one_arc_reference() {
        let value = Arc::new(1);
        let weak = Arc::downgrade(&value);
        let slot = AnswerSlot::default();
        let (answer, did_write) = slot.record(value);
        assert!(did_write);
        assert_eq!(
            Arc::strong_count(&answer),
            2,
            "slot and returned answer own references"
        );

        drop(slot);
        assert_eq!(
            Arc::strong_count(&answer),
            1,
            "dropping the slot releases its reference"
        );
        drop(answer);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn answer_slot_can_be_reused_after_reservation_rollback() {
        let slot = AnswerSlot::default();
        assert!(slot.reserve(Arc::new(1)));
        // SAFETY: This test successfully reserved the slot above.
        unsafe { slot.rollback_reserved_if_pending() };
        assert!(slot.get().is_none());

        let (answer, did_write) = slot.record(Arc::new(2));
        assert!(did_write);
        assert_eq!(*answer, 2);
    }
}
