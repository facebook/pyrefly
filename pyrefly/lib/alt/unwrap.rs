/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::error::collector::ErrorCollector;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::ClassType;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::types::types::Union;
use crate::types::types::Var;

#[derive(Clone, Debug)]
pub struct Hint<'a> {
    union: Type,
    branches: Vec1<Type>,
    errors: Option<&'a ErrorCollector>,
    source_branches: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct HintRef<'a, 'b> {
    union: &'b Type,
    branches: &'b [Type],
    errors: Option<&'a ErrorCollector>,
    source_branches: usize,
}

impl<'a> Hint<'a> {
    pub fn new(union: Type, branches: Vec1<Type>, errors: Option<&'a ErrorCollector>) -> Self {
        let source_branches = branches.len();
        Self {
            union,
            branches,
            errors,
            source_branches,
        }
    }

    pub fn as_ref(&self) -> HintRef<'a, '_> {
        HintRef {
            union: &self.union,
            branches: self.branches.as_slice(),
            errors: self.errors,
            source_branches: self.source_branches,
        }
    }

    pub fn errors(&self) -> Option<&'a ErrorCollector> {
        self.errors
    }

    pub fn to_type(&self) -> Type {
        self.union.clone()
    }

    pub fn union(&self) -> &Type {
        &self.union
    }

    pub fn source_branches(&self) -> usize {
        self.source_branches
    }

    pub fn with_source_branches(mut self, count: usize) -> Self {
        self.source_branches = count.max(1);
        self
    }
}

impl<'a, 'b> HintRef<'a, 'b> {
    pub fn ty(&self) -> &Type {
        self.union
    }

    pub fn errors(&self) -> Option<&'a ErrorCollector> {
        self.errors
    }

    pub fn branches(&self) -> &'b [Type] {
        self.branches
    }

    pub fn source_branches(&self) -> usize {
        self.source_branches.max(1)
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn fresh_var(&self) -> Var {
        self.solver().fresh_unwrap(self.uniques)
    }

    /// Resolve a var to a type, but only if it was pinned by the subtype
    /// check we just ran. If it was not, return `None`.
    fn resolve_var_opt(&self, ty: &Type, var: Var) -> Option<Type> {
        let res = self.resolve_var(ty, var);
        // TODO: Really want to check if the Var is constrained in any way.
        // No way to do that currently, but this is close.
        if matches!(res, Type::Var(..)) {
            None
        } else {
            Some(res)
        }
    }

    /// Resolve a var to a type. This function assumes that the caller has just
    /// run a successful subtype check of `ty` against a type we are trying to
    /// decompose (for example `Awaitable[_]` or `Iterable[_]`).
    ///
    /// It is an error to call this if the subtype check failed. If the subtype
    /// check succeeded, in most cases the solver will have pinned the Var to
    /// the correct type argument.
    ///
    /// One tricky issue is that there are some scenarios where a subtype
    /// check can pass without pinning vars; this function needs to handle
    /// those as edge cases.
    ///
    /// As an example of how this works, if `x` is `CustomSubtypeOfAwaitable[int]`,
    /// we will synthesize an `Awaitable[@v]` and when we do a subtype check of
    /// `x`, the solver will pin `@v` to `int` and we will use that.
    ///
    /// Special cases we handle thus far (there may be bugs where we need more):
    /// - if `ty` is `Any`, the stubtype check passes without pinning, and the
    ///   right thing to do is propagate the `Any`, preserving its `AnyStyle`.
    /// - TODO: if `ty` is bottom (`Never` or `NoReturn`), the subtype check
    ///   will pass and we should propagate the type.
    /// - TODO: all edge cases probably need to also be handled when they are
    ///   the first entry in a union.
    fn resolve_var(&self, ty: &Type, var: Var) -> Type {
        match ty {
            Type::Any(style) => Type::Any(*style),
            Type::Never(style) => Type::Never(*style),
            _ => self.solver().expand_vars(var.to_type()),
        }
    }

    pub fn behaves_like_any(&self, ty: &Type) -> bool {
        ty.is_any() || (!ty.is_never() && self.is_subset_eq(ty, &Type::never()))
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_mapping(&self, ty: &Type) -> Option<(Type, Type)> {
        // TODO: Ideally, we would handle this inside of the subset check
        // Handle Type::Var and Type::Union explicitly, similar to iterate() in solve.rs.
        match ty {
            Type::Var(v) if let Some(_guard) = self.recurse(*v) => {
                self.unwrap_mapping(&self.solver().force_var(*v))
            }
            Type::Union(box Union { members, .. }) => {
                let results: Option<Vec<_>> =
                    members.iter().map(|t| self.unwrap_mapping(t)).collect();
                let (keys, values): (Vec<_>, Vec<_>) = results?.into_iter().unzip();
                Some((self.unions(keys), self.unions(values)))
            }
            _ => {
                let key = self.fresh_var();
                let value = self.fresh_var();
                let dict_type = self
                    .stdlib
                    .mapping(key.to_type(), value.to_type())
                    .to_type();
                if self.is_subset_eq(ty, &dict_type) {
                    Some((self.resolve_var(ty, key), self.resolve_var(ty, value)))
                } else {
                    None
                }
            }
        }
        branches.sort_by_key(|branch| self.type_contains_var(branch));
        branches
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_awaitable(&self, ty: &Type) -> Option<Type> {
        let var = self.fresh_var();
        let awaitable_ty = self.stdlib.awaitable(var.to_type()).to_type();
        if self.is_subset_eq(ty, &awaitable_ty) {
            Some(self.resolve_var(ty, var))
        } else {
            None
        }
    }

    /// Warning: this returns `true` if the type is `Any` or a class that extends `Any`
    pub fn is_coroutine(&self, ty: &Type) -> bool {
        let var1 = self.fresh_var();
        let var2 = self.fresh_var();
        let var3 = self.fresh_var();
        let coroutine_ty = self
            .stdlib
            .coroutine(var1.to_type(), var2.to_type(), var3.to_type())
            .to_type();
        self.is_subset_eq(ty, &coroutine_ty)
    }

    /// Check if a type is a sequence type for pattern matching purposes (PEP 634).
    ///
    /// Per PEP 634, sequence patterns match:
    /// - Builtins with Py_TPFLAGS_SEQUENCE: list, tuple, range, memoryview,
    ///   collections.deque, array.array
    /// - Classes that inherit from collections.abc.Sequence
    /// - Classes registered as collections.abc.Sequence (cannot detect statically)
    ///
    /// Explicitly excluded (even though they're sequences in other contexts):
    /// - str, bytes, bytearray
    ///
    /// Warning: this returns `true` if the type is `Any` or a class that extends `Any`
    pub fn is_sequence_for_pattern(&self, ty: &Type) -> bool {
        // Handle special exclusions first - str, bytes, bytearray are NOT sequences
        // for pattern matching per PEP 634
        match ty {
            Type::ClassType(cls)
                if cls.is_builtin("str")
                    || cls.is_builtin("bytes")
                    || cls.is_builtin("bytearray") =>
            {
                return false;
            }
            Type::LiteralString(_) => return false,
            // Tuples are always sequences for pattern matching
            Type::Tuple(_) => return true,
            _ => {}
        }

        // Check if the type is a subtype of Sequence[T] for some T
        let var = self.fresh_var();
        let sequence_ty = self.stdlib.sequence(var.to_type()).to_type();
        self.is_subset_eq(ty, &sequence_ty)
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_coroutine(&self, ty: &Type) -> Option<(Type, Type, Type)> {
        let yield_ty = self.fresh_var();
        let send_ty = self.fresh_var();
        let return_ty = self.fresh_var();
        let coroutine_ty = self
            .stdlib
            .coroutine(yield_ty.to_type(), send_ty.to_type(), return_ty.to_type())
            .to_type();
        if self.is_subset_eq(ty, &coroutine_ty) {
            let yield_ty: Type = self.resolve_var(ty, yield_ty);
            let send_ty = self.resolve_var(ty, send_ty);
            let return_ty = self.resolve_var(ty, return_ty);
            Some((yield_ty, send_ty, return_ty))
        } else {
            None
        }
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_generator(&self, ty: &Type) -> Option<(Type, Type, Type)> {
        let yield_ty = self.fresh_var();
        let send_ty = self.fresh_var();
        let return_ty = self.fresh_var();
        let generator_ty = self
            .stdlib
            .generator(yield_ty.to_type(), send_ty.to_type(), return_ty.to_type())
            .to_type();
        if self.is_subset_eq(ty, &generator_ty) {
            let yield_ty: Type = self.resolve_var(ty, yield_ty);
            let send_ty = self.resolve_var(ty, send_ty);
            let return_ty = self.resolve_var(ty, return_ty);
            Some((yield_ty, send_ty, return_ty))
        } else {
            None
        }
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_iterable(&self, ty: &Type) -> Option<Type> {
        let iter_ty = self.fresh_var();
        let iterable_ty = self.stdlib.iterable(iter_ty.to_type()).to_type();
        if self.is_subset_eq(ty, &iterable_ty) {
            Some(self.resolve_var(ty, iter_ty))
        } else {
            None
        }
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_async_iterable(&self, ty: &Type) -> Option<Type> {
        let iter_ty = self.fresh_var();
        let iterable_ty = self.stdlib.async_iterable(iter_ty.to_type()).to_type();
        if self.is_subset_eq(ty, &iterable_ty) {
            Some(self.resolve_var(ty, iter_ty))
        } else {
            None
        }
    }

    /// Warning: this returns `Some` if the type is `Any` or a class that extends `Any`
    pub fn unwrap_async_iterator(&self, ty: &Type) -> Option<Type> {
        let var = self.fresh_var();
        let iterator_ty = self.stdlib.async_iterator(var.to_type()).to_type();
        if self.is_subset_eq(ty, &iterator_ty) {
            Some(self.resolve_var(ty, var))
        } else {
            None
        }
    }

    pub fn decompose_dict<'b>(
        &self,
        hint: HintRef<'a, 'b>,
    ) -> (Option<Hint<'a>>, Option<Hint<'a>>) {
        let mut key_types = Vec::new();
        let mut value_types = Vec::new();
        let source_branches = hint.source_branches();
        for branch in hint.branches() {
            let key = self.fresh_var();
            let value = self.fresh_var();
            let dict_type = self.stdlib.dict(key.to_type(), value.to_type()).to_type();
            if self.is_subset_eq(&dict_type, branch) {
                if let (Some(key_ty), Some(value_ty)) = (
                    self.resolve_var_opt(branch, key),
                    self.resolve_var_opt(branch, value),
                ) {
                    key_types.push(key_ty);
                    value_types.push(value_ty);
                }
            }
        }
        let key = self
            .hint_from_branches_vec(key_types, hint.errors())
            .map(|hint| hint.with_source_branches(source_branches));
        let value = self
            .hint_from_branches_vec(value_types, hint.errors())
            .map(|hint| hint.with_source_branches(source_branches));
        (key, value)
    }

    pub fn decompose_set<'b>(&self, hint: HintRef<'a, 'b>) -> Option<Hint<'a>> {
        self.hint_filter_map(hint, move |branch| {
            let elem = self.fresh_var();
            let set_type = self.stdlib.set(elem.to_type()).to_type();
            if self.is_subset_eq(&set_type, branch) {
                self.resolve_var_opt(branch, elem)
            } else {
                None
            }
        })
    }

    pub fn decompose_iterable<'b>(&self, hint: HintRef<'a, 'b>) -> Option<Hint<'a>> {
        self.hint_filter_map(hint, move |branch| {
            let elem = self.fresh_var();
            let iterable_type = self.stdlib.iterable(elem.to_type()).to_type();
            if self.is_subset_eq(&iterable_type, branch) {
                self.resolve_var_opt(branch, elem)
            } else {
                None
            }
        })
    }

    pub fn decompose_list<'b>(&self, hint: HintRef<'a, 'b>) -> Option<Hint<'a>> {
        self.hint_filter_map(hint, move |branch| {
            let elem = self.fresh_var();
            let list_type = self.stdlib.list(elem.to_type()).to_type();
            if self.is_subset_eq(&list_type, branch) {
                self.resolve_var_opt(branch, elem)
            } else {
                None
            }
        })
    }

    pub fn decompose_tuple<'b>(&self, hint: HintRef<'a, 'b>) -> Option<Hint<'a>> {
        self.hint_filter_map(hint, move |branch| {
            let elem = self.fresh_var();
            let tuple_type = self.stdlib.tuple(elem.to_type()).to_type();
            if self.is_subset_eq(&tuple_type, branch) {
                self.resolve_var_opt(branch, elem)
            } else {
                None
            }
        })
    }

    pub fn decompose_lambda<'b>(
        &self,
        hint: HintRef<'a, 'b>,
        param_vars: &[(Name, Var)],
    ) -> Option<Hint<'a>> {
        let return_ty = self.fresh_var();
        let params = param_vars
            .iter()
            .map(|(name, var)| Param::Pos(name.clone(), var.to_type(), Required::Required))
            .collect::<Vec<_>>();
        let callable_ty = Type::callable(params, return_ty.to_type());

        self.hint_filter_map(hint, move |branch| {
            if self.is_subset_eq(&callable_ty, branch) {
                self.resolve_var_opt(branch, return_ty)
            } else {
                None
            }
        })
    }

    pub fn decompose_generator_yield<'b>(&self, hint: HintRef<'a, 'b>) -> Option<Hint<'a>> {
        let yield_ty = self.fresh_var();
        let generator_ty = self
            .stdlib
            .generator(
                yield_ty.to_type(),
                self.fresh_var().to_type(),
                self.fresh_var().to_type(),
            )
            .to_type();
        self.hint_filter_map(hint, move |branch| {
            if self.is_subset_eq(&generator_ty, branch) {
                self.resolve_var_opt(branch, yield_ty)
            } else {
                None
            }
        })
    }

    pub fn decompose_generator(&self, ty: &Type) -> Option<(Type, Type, Type)> {
        let yield_ty = self.fresh_var();
        let send_ty = self.fresh_var();
        let return_ty = self.fresh_var();
        let generator_ty = self
            .stdlib
            .generator(yield_ty.to_type(), send_ty.to_type(), return_ty.to_type())
            .to_type();
        if self.is_subset_eq(&generator_ty, ty) {
            let yield_ty: Type = self.resolve_var_opt(ty, yield_ty)?;
            let send_ty = self.resolve_var_opt(ty, send_ty).unwrap_or(Type::None);
            let return_ty = self.resolve_var_opt(ty, return_ty).unwrap_or(Type::None);
            Some((yield_ty, send_ty, return_ty))
        } else {
            None
        }
    }

    pub fn decompose_async_generator(&self, ty: &Type) -> Option<(Type, Type)> {
        let yield_ty = self.fresh_var();
        let send_ty = self.fresh_var();
        let async_generator_ty = self
            .stdlib
            .async_generator(yield_ty.to_type(), send_ty.to_type())
            .to_type();
        if self.is_subset_eq(&async_generator_ty, ty) {
            let yield_ty: Type = self.resolve_var_opt(ty, yield_ty)?;
            let send_ty = self.resolve_var_opt(ty, send_ty).unwrap_or(Type::None);
            Some((yield_ty, send_ty))
        } else if ty.is_any() {
            Some((Type::any_explicit(), Type::any_explicit()))
        } else {
            None
        }
    }

    /// Erase the structural information (length, ordering) Type::Tuple return the union of the contents
    /// Use to generate the type parameters for the Type::ClassType representation of tuple
    pub fn erase_tuple_type(&self, tuple: Tuple) -> ClassType {
        match tuple {
            Tuple::Unbounded(element) => self.stdlib.tuple(*element),
            Tuple::Concrete(elements) => {
                if elements.is_empty() {
                    self.stdlib.tuple(Type::any_implicit())
                } else {
                    self.stdlib.tuple(self.unions(elements))
                }
            }
            Tuple::Unpacked(box (prefix, middle, suffix)) => {
                let mut elements = prefix;
                match middle {
                    Type::Tuple(Tuple::Unbounded(unbounded_middle)) => {
                        elements.push(*unbounded_middle);
                    }
                    Type::Quantified(q) if q.is_type_var_tuple() => {
                        elements.push(Type::ElementOfTypeVarTuple(q))
                    }
                    _ => {
                        // We can't figure out the middle, fall back to `object`
                        elements.push(self.stdlib.object().clone().to_type())
                    }
                }
                elements.extend(suffix);
                self.stdlib.tuple(self.unions(elements))
            }
        }
    }
}
