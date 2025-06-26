/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use pyrefly_util::recurser::Recurser;
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
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::kind::ErrorKind;
use crate::error::style::ErrorStyle;
use crate::export::exports::LookupExport;
use crate::graph::calculation::ProposalResult;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::solver::type_order::TypeOrder;
use crate::types::class::Class;
use crate::types::stdlib::Stdlib;
use crate::types::type_info::TypeInfo;
use crate::types::types::Type;
use crate::types::types::Var;

/// Compactly represents the identity of a binding, for the purposes of
/// understanding the calculation stack.
#[derive(Debug, Clone, Dupe)]
pub struct CalcId(pub Bindings, pub AnyIdx);

impl PartialEq for CalcId {
    fn eq(&self, other: &Self) -> bool {
        (self.0.module_info(), &self.1) == (other.0.module_info(), &other.1)
    }
}

impl Eq for CalcId {}

impl Ord for CalcId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.1.cmp(&other.1) {
            Ordering::Equal => self
                .0
                .module_info()
                .name()
                .cmp(&other.0.module_info().name()),
            not_equal => not_equal,
        }
    }
}

impl PartialOrd for CalcId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

    fn push(&self, bindings: Bindings, idx: AnyIdx) {
        self.0.borrow_mut().push(CalcId(bindings, idx));
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

/// Represent a cycle we are currently solving.
pub struct Cycle;

/// Represent the current thread's cycles, which form a stack
/// because we can encounter a new one while solving another.
#[expect(dead_code)]
pub struct Cycles(RefCell<Vec<Cycle>>);

impl Cycles {
    pub fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }
}

pub struct AnswersSolver<'a, Ans: LookupAnswer> {
    answers: &'a Ans,
    current: &'a Answers,
    stack: &'a CalcStack,
    cycles: &'a Cycles,
    // The base solver is only used to reset the error collector at binding
    // boundaries. Answers code should generally use the error collector passed
    // along the call stack instead.
    base_errors: &'a ErrorCollector,
    bindings: &'a Bindings,
    pub exports: &'a dyn LookupExport,
    pub uniques: &'a UniqueFactory,
    pub recurser: &'a Recurser<Var>,
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
        recurser: &'a Recurser<Var>,
        stdlib: &'a Stdlib,
        stack: &'a CalcStack,
        cycles: &'a Cycles,
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
            stack,
            cycles,
        }
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

    pub fn module_info(&self) -> &ModuleInfo {
        self.bindings.module_info()
    }

    pub fn stack(&self) -> &CalcStack {
        self.stack
    }

    pub fn for_display(&self, t: Type) -> Type {
        self.solver().for_display(t)
    }

    pub fn type_order(&self) -> TypeOrder<Ans> {
        TypeOrder::new(self)
    }

    pub fn get_idx<K: Solve<Ans>>(&self, idx: Idx<K>) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let calculation = self.get_calculation(idx);
        self.stack.push(self.bindings().dupe(), K::to_anyidx(idx));

        let result = match calculation.propose_calculation() {
            ProposalResult::Calculated(v) => Ok((v, None)),
            ProposalResult::CycleBroken(rec) => Err(rec),
            ProposalResult::CycleDetected => {
                let binding = self.bindings().get(idx);
                let rec = K::create_recursive(self, binding);
                match calculation.record_cycle(rec) {
                    Either::Left(v) => {
                        // Another thread finished, treat it just like `Caluculated`
                        Ok((v, None))
                    }
                    Either::Right(rec) => Err(rec),
                }
            }
            ProposalResult::Calculatable => {
                let binding = self.bindings().get(idx);
                let value = K::solve(self, binding, self.base_errors);
                Ok(calculation.record_value(value))
            }
        };
        if let Ok((v, Some(r))) = &result {
            let k = self.bindings().idx_to_key(idx).range();
            K::record_recursive(self, k, v, r, self.base_errors);
        }
        self.stack.pop();
        match result {
            Ok((v, _)) => v,
            Err(r) => Arc::new(K::promote_recursive(r)),
        }
    }

    pub fn get_from_module<K: Solve<Ans> + Exported>(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &K,
    ) -> Arc<K::Answer>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        if module == self.module_info().name()
            && path.is_none_or(|path| path == self.module_info().path())
        {
            self.get(k)
        } else {
            self.answers.get(module, path, k, self.stack, self.cycles)
        }
    }

    pub fn get_from_class<K: Solve<Ans> + Exported>(&self, cls: &Class, k: &K) -> Arc<K::Answer>
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

    pub fn create_recursive(&self, binding: &Binding) -> Var {
        let t = if let Binding::Default(default, _) = binding {
            self.get_calculation(*default)
                .get()
                .map(|t| t.arc_clone_ty().promote_literals(self.stdlib))
        } else {
            None
        };
        self.solver().fresh_recursive(self.uniques, t)
    }

    pub fn record_recursive(
        &self,
        loc: TextRange,
        answer: Type,
        recursive: Var,
        errors: &ErrorCollector,
    ) {
        self.solver()
            .record_recursive::<Ans>(recursive, answer, self.type_order(), errors, loc);
    }

    /// Check if `want` matches `got` returning `want` if the check fails.
    pub fn check_and_return_type_info(
        &self,
        want: &Type,
        got: TypeInfo,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> TypeInfo {
        if self.check_type(want, got.ty(), loc, errors, tcc) {
            got
        } else {
            got.with_ty(want.clone())
        }
    }

    /// Check if `want` matches `got` returning `want` if the check fails.
    pub fn check_and_return_type(
        &self,
        want: &Type,
        got: Type,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> Type {
        if self.check_type(want, &got, loc, errors, tcc) {
            got
        } else {
            want.clone()
        }
    }

    /// Check if `want` matches `got`, returning `true` on success and `false` on failure.
    pub fn check_type(
        &self,
        want: &Type,
        got: &Type,
        loc: TextRange,
        errors: &ErrorCollector,
        tcc: &dyn Fn() -> TypeCheckContext,
    ) -> bool {
        if got.is_error() || self.is_subset_eq(got, want) {
            true
        } else {
            self.solver().error(want, got, errors, loc, tcc);
            false
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
                    Type::Union(tys) => {
                        self.seen_union = true;
                        tys.iter().for_each(|ty| self.go(ty, in_type))
                    }
                    Type::Type(box Type::Union(tys)) if !in_type => {
                        tys.iter().for_each(|ty| self.go(ty, true))
                    }
                    Type::Var(v) if let Some(_guard) = self.me.recurser.recurse(*v) => {
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
        kind: ErrorKind,
        context: Option<&dyn Fn() -> ErrorContext>,
        msg: String,
    ) -> Type {
        errors.add(range, kind, context, vec1![msg]);
        Type::any_error()
    }

    /// Create a new error collector. Useful when a caller wants to decide whether or not to report
    /// errors from an operation.
    pub fn error_collector(&self) -> ErrorCollector {
        ErrorCollector::new(self.module_info().dupe(), ErrorStyle::Delayed)
    }

    /// Create an error collector that simply swallows errors. Useful when a caller wants to try an
    /// operation that may error but never report errors from it.
    pub fn error_swallower(&self) -> ErrorCollector {
        ErrorCollector::new(self.module_info().dupe(), ErrorStyle::Never)
    }
}
