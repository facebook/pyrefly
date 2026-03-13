/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_graph::calculation::Calculation;
use pyrefly_graph::index::Idx;
use pyrefly_graph::index_map::IndexMap;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::lock::Mutex;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;

use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::CalcId;
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
use crate::binding::table::TableKeyed;
use crate::config::base::RecursionLimitConfig;
use crate::dispatch_anyidx;
use crate::error::collector::ErrorCollector;
use crate::error::style::ErrorStyle;
use crate::export::exports::LookupExport;
use crate::module::module_info::ModuleInfo;
use crate::solver::solver::Solver;
use crate::solver::solver::VarRecurser;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::state::ModuleChanges;
use crate::table;
use crate::table_for_each;
use crate::table_mut_for_each;
use crate::table_try_for_each;
use crate::types::callable::Callable;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;
use crate::types::heap::TypeHeap;
use crate::types::stdlib::Stdlib;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::TParams;
use crate::types::types::Type;

/// The index stores all the references where the definition is external to the current module.
/// This is useful for fast references computation.
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
    /// A map from (child method range) to a list of parent method definitions (ModulePath, parent method range).
    /// This is used to find reimplementations when doing find-references on parent methods.
    pub parent_methods_map: SmallMap<TextRange, Vec<(ModulePath, TextRange)>>,
}

#[derive(Debug, Clone)]
pub struct OverloadTrace {
    pub(crate) callable: Callable,
    pub(crate) tparams: Option<Arc<TParams>>,
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
}

impl Traces {
    /// Merge accumulated side effects into the persisted trace store.
    pub(crate) fn merge(&mut self, side_effects: TraceSideEffects) {
        for (k, v) in side_effects.types {
            self.types.insert(k, v);
        }
        for (k, v) in side_effects.overloaded_callees {
            self.overloaded_callees.insert(k, v);
        }
        for (k, v) in side_effects.invoked_properties {
            self.invoked_properties.insert(k, v);
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
}

impl TraceSideEffects {
    /// Deep-force all embedded types, resolving any remaining `Type::Var`
    /// references using the given solver. Must be called before publishing
    /// to the persisted `Traces` store so that read APIs can return
    /// fully-resolved types without touching the solver.
    pub(crate) fn finalize(&mut self, solver: &Solver) {
        for ty in self.types.values_mut() {
            solver.deep_force_mut(Arc::make_mut(ty));
        }
        for callee in self.overloaded_callees.values_mut() {
            match callee {
                OverloadedCallee::Resolved { callable } => {
                    callable
                        .callable
                        .visit_mut(&mut |t: &mut Type| solver.deep_force_mut(t));
                }
                OverloadedCallee::Candidates { all, closest, .. } => {
                    for trace in all.iter_mut() {
                        trace
                            .callable
                            .visit_mut(&mut |t: &mut Type| solver.deep_force_mut(t));
                    }
                    closest
                        .callable
                        .visit_mut(&mut |t: &mut Type| solver.deep_force_mut(t));
                }
            }
        }
        for ty in self.invoked_properties.values_mut() {
            solver.deep_force_mut(Arc::make_mut(ty));
        }
    }

    /// Assert that no trace payload contains unresolved `Type::Var`.
    /// Only runs in debug builds to avoid production overhead.
    /// Panics if any Var is found, indicating a bug in finalization.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_var_free(&self) {
        fn has_var(ty: &Type) -> bool {
            let mut found = false;
            ty.visit(&mut |t: &Type| {
                if matches!(t, Type::Var(_)) {
                    found = true;
                }
            });
            found
        }

        for (range, ty) in &self.types {
            assert!(
                !has_var(ty),
                "Type trace at {range:?} contains unresolved Var after finalization: {ty}"
            );
        }
        for (range, ty) in &self.invoked_properties {
            assert!(
                !has_var(ty),
                "Property getter trace at {range:?} contains unresolved Var after finalization: {ty}"
            );
        }
        for (range, callee) in &self.overloaded_callees {
            match callee {
                OverloadedCallee::Resolved { callable } => {
                    let ty = callable.as_type();
                    assert!(
                        !has_var(&ty),
                        "Resolved callee trace at {range:?} contains unresolved Var: {ty}"
                    );
                }
                OverloadedCallee::Candidates { all, closest, .. } => {
                    for trace in all {
                        let ty = trace.as_type();
                        assert!(
                            !has_var(&ty),
                            "Overload candidate trace at {range:?} contains unresolved Var: {ty}"
                        );
                    }
                    let ty = closest.as_type();
                    assert!(
                        !has_var(&ty),
                        "Closest overload trace at {range:?} contains unresolved Var: {ty}"
                    );
                }
            }
        }
    }

    /// No-op in release builds.
    #[cfg(not(debug_assertions))]
    pub(crate) fn debug_assert_var_free(&self) {}
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

pub type AnswerEntry<K> = IndexMap<K, Calculation<Arc<<K as Keyed>::Answer>>>;

table!(
    #[derive(Debug, Default)]
    pub struct AnswerTable(pub AnswerEntry)
);

impl DisplayWith<Bindings> for Answers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, bindings: &Bindings) -> fmt::Result {
        fn go<K: Keyed>(
            bindings: &Bindings,
            entry: &AnswerEntry<K>,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result
        where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        {
            for (idx, answer) in entry.iter() {
                let key = bindings.idx_to_key(idx);
                let value = bindings.get(idx);
                writeln!(
                    f,
                    "{} = {} = {}",
                    bindings.module().display(key),
                    value.display_with(bindings),
                    match answer.get() {
                        Some(v) => v.to_string(),
                        None => "(unsolved)".to_owned(),
                    },
                )?;
            }
            Ok(())
        }

        table_try_for_each!(self.table, |x| go(bindings, x, f));
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
    index: Option<Arc<Mutex<Index>>>,
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
    #[allow(dead_code)] // Used in tests.
    pub fn get<K: Exported>(&self, key: &K) -> &Arc<<K as Keyed>::Answer>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.get_hashed(Hashed::new(key))
    }

    pub fn get_hashed_opt<K: Exported>(&self, key: Hashed<&K>) -> Option<&Arc<<K as Keyed>::Answer>>
    where
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        self.table.get().get_hashed(key)
    }

    pub fn get_hashed<K: Exported>(&self, key: Hashed<&K>) -> &Arc<<K as Keyed>::Answer>
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

    /// Commit a preliminary answer to a specific module's Calculation cell.
    /// Used for cross-module batch commit when an SCC spans module boundaries.
    ///
    /// Returns true if the commit was performed, false if the implementation
    /// does not support cross-module commits.
    ///
    /// Default implementation returns false (not supported).
    fn commit_to_module(
        &self,
        _calc_id: CalcId,
        _answer: Arc<dyn Any + Send + Sync>,
        _errors: Option<Arc<ErrorCollector>>,
    ) -> bool {
        false
    }

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

    /// Acquire a write lock on a cross-module Calculation cell for SCC
    /// batch commit. Returns true if the lock was acquired.
    ///
    /// Default implementation returns false (not supported).
    fn write_lock_in_module(&self, _calc_id: &CalcId) -> bool {
        false
    }

    /// Write a value to a write-locked cross-module Calculation cell and
    /// release the lock. Also extends errors and publishes traces if the
    /// write wins.
    ///
    /// Default implementation is a no-op.
    fn write_unlock_in_module(
        &self,
        _calc_id: CalcId,
        _answer: Arc<dyn Any + Send + Sync>,
        _errors: Option<Arc<ErrorCollector>>,
        _traces: Option<TraceSideEffects>,
    ) -> bool {
        false
    }

    /// Release a write lock on a cross-module Calculation cell without
    /// writing a value. Used for panic cleanup.
    ///
    /// Default implementation is a no-op.
    fn write_unlock_empty_in_module(&self, _calc_id: &CalcId) {}
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
            let ks = bindings.keys::<K>();
            items.reserve(ks.len());
            for k in ks {
                items.insert_once(k, Calculation::new());
            }
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

    pub fn heap(&self) -> &TypeHeap {
        &self.solver.heap
    }

    #[expect(dead_code)]
    fn len(&self) -> usize {
        let mut res = 0;
        table_for_each!(&self.table, |x: &AnswerEntry<_>| res += x.len());
        res
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
        answers_solver.validate_final_thread_state();

        Solutions {
            module_info: bindings.module().dupe(),
            table: res,
            index: self.index.dupe(),
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
        // Fast path: check if the answer has already been computed in the Calculation cell.
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
        self.table.get::<K>().get(k)?.get()
    }

    /// Commit a type-erased answer to this module's Calculation cell.
    /// Target-side entry point for cross-module batch commit.
    /// Returns true if the write won the first-write-wins race.
    pub fn commit_preliminary(&self, any_idx: &AnyIdx, answer: Arc<dyn Any + Send + Sync>) -> bool {
        dispatch_anyidx!(any_idx, self, commit_typed, answer)
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

    /// Typed commit for a specific key type. Downcasts the answer and writes
    /// to the Calculation cell. Returns true if this write won the first-write-wins
    /// race (i.e., the answer was actually stored).
    fn commit_typed<K: Keyed>(&self, idx: Idx<K>, answer: Arc<dyn Any + Send + Sync>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        let typed_answer: Arc<K::Answer> = Arc::unwrap_or_clone(
            answer
                .downcast::<Arc<K::Answer>>()
                .expect("Answers::commit_typed: type mismatch in cross-module batch commit"),
        );
        // Get the calculation cell from the answer table
        if let Some(calculation) = self.table.get::<K>().get(idx) {
            // No recursive placeholder can exist in the Calculation cell because
            // placeholders are stored only in SCC-local NodeState::HasPlaceholder.
            let (_answer, did_write) = calculation.record_value(typed_answer);
            did_write
        } else {
            false
        }
    }

    /// Acquire a write lock on a cell for SCC batch commit.
    /// Returns true if the lock was acquired, false if the cell is already
    /// `Calculated` (no lock needed since writes would be no-ops).
    pub fn write_lock_preliminary(&self, any_idx: &AnyIdx) -> bool {
        dispatch_anyidx!(any_idx, self, write_lock_typed)
    }

    /// Write a value to a write-locked cell and release the lock.
    /// Returns true if this write stored the value (first-write-wins).
    pub fn write_unlock_preliminary(
        &self,
        any_idx: &AnyIdx,
        answer: Arc<dyn Any + Send + Sync>,
    ) -> bool {
        dispatch_anyidx!(any_idx, self, write_unlock_typed, answer)
    }

    /// Release a write lock without writing a value (panic cleanup).
    pub fn write_unlock_empty_preliminary(&self, any_idx: &AnyIdx) {
        dispatch_anyidx!(any_idx, self, write_unlock_empty_typed)
    }

    fn write_lock_typed<K: Keyed>(&self, idx: Idx<K>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        if let Some(calculation) = self.table.get::<K>().get(idx) {
            calculation.write_lock()
        } else {
            false
        }
    }

    fn write_unlock_typed<K: Keyed>(&self, idx: Idx<K>, answer: Arc<dyn Any + Send + Sync>) -> bool
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        let typed_answer: Arc<K::Answer> = Arc::unwrap_or_clone(
            answer
                .downcast::<Arc<K::Answer>>()
                .expect("Answers::write_unlock_typed: type mismatch"),
        );
        if let Some(calculation) = self.table.get::<K>().get(idx) {
            let (_answer, did_write) = calculation.write_unlock(typed_answer);
            did_write
        } else {
            false
        }
    }

    fn write_unlock_empty_typed<K: Keyed>(&self, idx: Idx<K>)
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
    {
        if let Some(calculation) = self.table.get::<K>().get(idx) {
            calculation.write_unlock_empty();
        }
    }

    fn deep_force(&self, t: Type) -> Type {
        self.solver.deep_force(t)
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
        Some(self.deep_force(self.get_idx(idx)?.arc_clone_ty()))
    }

    pub fn get_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(lock.types.get(&range)?.as_ref().clone())
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
    pub fn get_calculation<K: Solve<Ans>>(&self, idx: Idx<K>) -> &Calculation<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.current().table.get::<K>().get(idx).unwrap_or_else(|| {
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

    pub fn record_resolved_trace(&self, loc: TextRange, ty: Type) {
        if self.current().trace.is_some()
            && let Some(callable) = ty.to_callable()
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

    pub fn record_external_attribute_definition_index(
        &self,
        base: &Type,
        attribute_name: &Name,
        attribute_reference_range: TextRange,
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
                    } => {
                        if cls.module_path() != self.bindings().module().path() {
                            index
                                .lock()
                                .externally_defined_attribute_references
                                .entry(cls.module_path().dupe())
                                .or_default()
                                .push((range, attribute_reference_range))
                        }
                    }
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
