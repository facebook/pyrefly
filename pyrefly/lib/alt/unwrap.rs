/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::ClassType;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::types::types::Var;

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
            _ => self.solver().expand(var.to_type()),
        }
    }

    pub fn unwrap_mapping(&self, ty: &Type) -> Option<(Type, Type)> {
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

    pub fn unwrap_awaitable(&self, ty: &Type) -> Option<Type> {
        let var = self.fresh_var();
        let awaitable_ty = self.stdlib.awaitable(var.to_type()).to_type();
        if self.is_subset_eq(ty, &awaitable_ty) {
            Some(self.resolve_var(ty, var))
        } else {
            None
        }
    }

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

    pub fn unwrap_iterable(&self, ty: &Type) -> Option<Type> {
        let iter_ty = self.fresh_var();
        let iterable_ty = self.stdlib.iterable(iter_ty.to_type()).to_type();
        if self.is_subset_eq(ty, &iterable_ty) {
            Some(self.resolve_var(ty, iter_ty))
        } else {
            None
        }
    }

    pub fn unwrap_async_iterable(&self, ty: &Type) -> Option<Type> {
        let iter_ty = self.fresh_var();
        let iterable_ty = self.stdlib.async_iterable(iter_ty.to_type()).to_type();
        if self.is_subset_eq(ty, &iterable_ty) {
            Some(self.resolve_var(ty, iter_ty))
        } else {
            None
        }
    }

    pub fn decompose_dict(&self, ty: &Type) -> (Option<Type>, Option<Type>) {
        let key = self.fresh_var();
        let value = self.fresh_var();
        let dict_type = self.stdlib.dict(key.to_type(), value.to_type()).to_type();
        if self.is_subset_eq(&dict_type, ty) {
            let key = self.resolve_var_opt(ty, key);
            let value = self.resolve_var_opt(ty, value);
            (key, value)
        } else {
            (None, None)
        }
    }

    pub fn decompose_set(&self, ty: &Type) -> Option<Type> {
        let elem = self.fresh_var();
        let set_type = self.stdlib.set(elem.to_type()).to_type();
        if self.is_subset_eq(&set_type, ty) {
            self.resolve_var_opt(ty, elem)
        } else {
            None
        }
    }

    pub fn decompose_list(&self, ty: &Type) -> Option<Type> {
        let elem = self.fresh_var();
        let list_type = self.stdlib.list(elem.to_type()).to_type();
        if self.is_subset_eq(&list_type, ty) {
            self.resolve_var_opt(ty, elem)
        } else {
            None
        }
    }

    pub fn decompose_lambda(&self, ty: &Type, param_vars: &[(&Name, Var)]) -> Option<Type> {
        let return_ty = self.fresh_var();
        let params = param_vars
            .iter()
            .map(|(name, var)| Param::Pos((*name).clone(), var.to_type(), Required::Required))
            .collect::<Vec<_>>();
        let callable_ty = Type::callable(params, return_ty.to_type());

        if self.is_subset_eq(&callable_ty, ty) {
            self.resolve_var_opt(ty, return_ty)
        } else {
            None
        }
    }

    pub fn decompose_generator_yield(&self, ty: &Type) -> Option<Type> {
        let yield_ty = self.fresh_var();
        let generator_ty = self
            .stdlib
            .generator(
                yield_ty.to_type(),
                self.fresh_var().to_type(),
                self.fresh_var().to_type(),
            )
            .to_type();
        if self.is_subset_eq(&generator_ty, ty) {
            self.resolve_var_opt(ty, yield_ty)
        } else {
            None
        }
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
                if let Type::Tuple(Tuple::Unbounded(unbounded_middle)) = middle {
                    elements.push(*unbounded_middle);
                } else {
                    // We can't figure out the middle, fall back to `object`
                    elements.push(self.stdlib.object().clone().to_type())
                }
                elements.extend(suffix);
                self.stdlib.tuple(self.unions(elements))
            }
        }
    }
}
