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
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::CalcId;
use crate::alt::answers_solver::ThreadState;
use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::ChangedExport;
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
use crate::table;
use crate::table_for_each;
use crate::table_mut_for_each;
use crate::table_try_for_each;
use crate::types::callable::Callable;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;
use crate::types::heap::TypeHeap;
use crate::types::stdlib::Stdlib;
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

#[derive(Debug)]
enum OverloadedCallee {
    Resolved {
        callable: Callable,
    },
    Candidates {
        all: Vec<Callable>,
        closest: Callable,
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

    /// Compute the set of exports that have changed between two solutions.
    /// Returns a set of `ChangedExport` values: either `Name` (for `KeyExport`) or
    /// `ClassDefIndex` (for class-related keys).
    pub fn changed_exports(
        &self,
        other: &Self,
    ) -> starlark_map::small_set::SmallSet<ChangedExport> {
        use starlark_map::small_set::SmallSet;

        fn check_table<K: Keyed>(
            x: &SolutionsEntry<K>,
            y: &Solutions,
            ctx: &mut TypeEqCtx,
            changed: &mut SmallSet<ChangedExport>,
        ) where
            SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
        {
            if !K::EXPORTED {
                return;
            }

            let y_table = y.table.get::<K>();

            // Check for items only in y (added keys)
            for (k, _v) in y_table {
                if !x.contains_key(k)
                    && let Some(anykey) = k.try_to_anykey()
                {
                    changed.insert(anykey.to_changed_export());
                }
            }

            // Check for differences in x
            for (k, v) in x {
                match y_table.get(k) {
                    Some(v2) if !v.type_eq(v2, ctx) => {
                        // Value changed
                        if let Some(anykey) = k.try_to_anykey() {
                            changed.insert(anykey.to_changed_export());
                        }
                    }
                    None => {
                        // Key removed
                        if let Some(anykey) = k.try_to_anykey() {
                            changed.insert(anykey.to_changed_export());
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut changed = SmallSet::new();
        // Important we have a single TypeEqCtx, so that we don't have
        // types used in different ways.
        let mut ctx = TypeEqCtx::default();

        // Check all tables
        table_for_each!(self.table, |x| {
            check_table(x, other, &mut ctx, &mut changed);
        });

        changed
    }

    pub fn get_index(&self) -> Option<Arc<Mutex<Index>>> {
        let index = self.index.as_ref()?;
        Some(index.dupe())
    }
}

pub trait LookupAnswer: Sized {
    /// Return all modules that are available for lookup.
    fn modules(&self) -> SmallSet<ModuleName>;

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

    fn deep_force(&self, t: Type) -> Type {
        self.solver.deep_force(t)
    }

    pub fn solver(&self) -> &Solver {
        &self.solver
    }

    pub fn get_type_at(&self, idx: Idx<Key>) -> Option<Type> {
        Some(self.deep_force(self.get_idx(idx)?.arc_clone_ty()))
    }

    pub fn get_type_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(self.deep_force(lock.types.get(&range)?.as_ref().clone()))
    }

    pub fn try_get_getter_for_range(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        Some(self.deep_force(lock.invoked_properties.get(&range)?.as_ref().clone()))
    }

    pub fn get_chosen_overload_trace(&self, range: TextRange) -> Option<Type> {
        let lock = self.trace.as_ref()?.lock();
        match lock.overloaded_callees.get(&range)? {
            OverloadedCallee::Resolved { callable } => {
                Some(self.deep_force(self.heap().mk_callable_from(callable.clone())))
            }
            OverloadedCallee::Candidates {
                closest,
                is_closest_chosen,
                ..
            } if *is_closest_chosen => {
                Some(self.deep_force(self.heap().mk_callable_from(closest.clone())))
            }
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
            OverloadedCallee::Resolved { callable } => Some((vec![callable.clone()], Some(0))),
            OverloadedCallee::Candidates { all, closest, .. } => {
                let chosen_index = all.iter().position(|signature| signature == closest);
                Some((all.clone(), chosen_index))
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
        if let Some(trace) = &self.current().trace
            && let Some(callable) = ty.to_callable()
        {
            trace
                .lock()
                .overloaded_callees
                .insert(loc, OverloadedCallee::Resolved { callable });
        }
    }

    /// Record all the overloads and the chosen overload.
    /// The trace will be used to power signature help and hover for overloaded functions.
    pub fn record_overload_trace(
        &self,
        loc: TextRange,
        all_overloads: Vec<&Callable>,
        closest_overload: &Callable,
        is_closest_overload_chosen: bool,
    ) {
        if let Some(trace) = &self.current().trace {
            trace.lock().overloaded_callees.insert(
                loc,
                OverloadedCallee::Candidates {
                    all: all_overloads.into_iter().cloned().collect(),
                    closest: closest_overload.clone(),
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
        if let Some(trace) = &self.current().trace {
            trace
                .lock()
                .invoked_properties
                .insert(loc, Arc::new(getter_ty.clone()));
        }
    }

    pub fn record_type_trace(&self, loc: TextRange, ty: &Type) {
        if let Some(trace) = &self.current().trace
            && !loc.is_empty()
        {
            trace.lock().types.insert(loc, Arc::new(ty.clone()));
        }
    }
}
