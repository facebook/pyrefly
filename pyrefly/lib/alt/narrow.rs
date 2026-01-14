/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use num_traits::ToPrimitive;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_types::class::Class;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::facet::FacetChain;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::facet::UnresolvedFacetChain;
use pyrefly_types::facet::UnresolvedFacetKind;
use pyrefly_types::simplify::intersect;
use pyrefly_types::type_info::JoinStyle;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::Number;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallTargetLookup;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::binding::binding::Key;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::FacetOrigin;
use crate::binding::narrow::FacetSubject;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowingSubject;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
use crate::types::callable::FunctionKind;
use crate::types::class::ClassType;
use crate::types::lit_int::LitInt;
use crate::types::literal::Lit;
use crate::types::literal::Literal;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

/// Beyond this size, don't try and narrow an enum.
///
/// If we have over 100 fields, the odds of the negative-type being useful is vanishingly small.
/// But the cost to create such a type (and then probably knock individual elements out of it)
/// is very high.
const NARROW_ENUM_LIMIT: usize = 100;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // Get the union of all members of an enum, minus the specified member
    fn subtract_enum_member(&self, cls: &ClassType, name: &Name) -> Type {
        if cls.class_object().fields().len() > NARROW_ENUM_LIMIT {
            return Type::ClassType(cls.clone());
        }
        let e = self.get_enum_from_class(cls.class_object()).unwrap();
        // Enums derived from enum.Flag cannot be treated as a union of their members
        if e.is_flag {
            return Type::ClassType(cls.clone());
        }
        self.unions(
            self.get_enum_members(cls.class_object())
                .into_iter()
                .filter_map(|f| {
                    if let Lit::Enum(lit_enum) = &f
                        && &lit_enum.member == name
                    {
                        None
                    } else {
                        Some(f.to_implicit_type())
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn disjoint_base<'b>(&'b self, t: &'b Type) -> &'b Class {
        // TODO: Implement the full disjoint base spec: https://peps.python.org/pep-0800/#specification.
        match t {
            Type::ClassType(cls)
                if let cls = cls.class_object()
                    && self.get_metadata_for_class(cls).is_disjoint_base() =>
            {
                cls
            }
            Type::Tuple(_) => self.stdlib.tuple_object(),
            _ => self.stdlib.object().class_object(),
        }
    }

    fn intersect_impl(&self, left: &Type, right: &Type, fallback: &dyn Fn() -> Type) -> Type {
        let is_literal =
            |t: &Type| matches!(t, Type::Literal(_) | Type::LiteralString(_) | Type::None);
        if self.is_subset_eq(right, left) {
            if left.is_toplevel_callable()
                && right.is_toplevel_callable()
                && self.is_subset_eq(left, right)
            {
                // If is_subset_eq checks succeed in both directions, we typically want to
                // return `right`, which corresponds to more recently encountered type info.
                // The exception is that, for callables, it's common to intersect a callable
                // with `(...) -> object` via `builtins.callable`, so we return the original
                // callable type.
                left.clone()
            } else {
                right.clone()
            }
        } else if self.is_subset_eq(left, right) {
            left.clone()
        } else if is_literal(left) || is_literal(right) {
            // The only inhabited intersections of literals are things like
            // `Literal[0] & Literal[0]` or `Literal[0] & int` that would have already been
            // intercepted by the is_subset_eq checks above. type(None) cannot be subclassed.
            Type::never()
        } else {
            let fallback = fallback();
            if fallback.is_never() {
                fallback
            } else {
                let left_base = self.disjoint_base(left);
                let right_base = self.disjoint_base(right);
                if self.has_superclass(left_base, right_base)
                    || self.has_superclass(right_base, left_base)
                {
                    intersect(vec![left.clone(), right.clone()], fallback)
                } else {
                    // A common subclass of these two classes cannot exist.
                    Type::never()
                }
            }
        }
    }

    /// Get our best approximation of ty & right.
    ///
    /// If the intersection is empty - which does not necessarily indicate
    /// an actual empty set because of multiple inheritance - use `fallback`
    fn intersect_with_fallback(
        &self,
        left: &Type,
        right: &Type,
        fallback: &dyn Fn() -> Type,
    ) -> Type {
        self.distribute_over_union(left, |l| {
            self.distribute_over_union(right, |r| self.intersect_impl(l, r, fallback))
        })
    }

    fn intersect(&self, left: &Type, right: &Type) -> Type {
        self.intersect_with_fallback(left, right, &Type::never)
    }

    /// Calculate the intersection of a number of types
    pub fn intersects(&self, ts: &[Type]) -> Type {
        match ts {
            [] => Type::ClassType(self.stdlib.object().clone()),
            [ty] => ty.clone(),
            [ty0, ty1] => self.intersect(ty0, ty1),
            [ty0, ts @ ..] => self.intersect(ty0, &self.intersects(ts)),
        }
    }

    fn subtract(&self, left: &Type, right: &Type) -> Type {
        self.distribute_over_union(left, |left| {
            // Special is_any check because `Any <: int` as a special case, but would mess up this.
            if !left.is_any() && self.is_subset_eq(left, right) {
                Type::never()
            } else {
                left.clone()
            }
        })
    }

    fn resolve_narrowing_call(
        &self,
        func: &Expr,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<AtomicNarrowOp> {
        let func_ty = self.expr_infer(func, errors);
        if args.args.len() > 1 {
            let second_arg = &args.args[1];
            let op = match func_ty.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::IsInstance)) => {
                    Some(AtomicNarrowOp::IsInstance(second_arg.clone()))
                }
                Some(CalleeKind::Function(FunctionKind::IsSubclass)) => {
                    Some(AtomicNarrowOp::IsSubclass(second_arg.clone()))
                }
                _ => None,
            };
            if op.is_some() {
                return op;
            }
        }
        if func_ty.is_typeis() {
            Some(AtomicNarrowOp::TypeIs(func_ty.clone(), args.clone()))
        } else if func_ty.is_typeguard() {
            Some(AtomicNarrowOp::TypeGuard(func_ty.clone(), args.clone()))
        } else {
            None
        }
    }

    fn narrow_isinstance(&self, left: &Type, right: &Type) -> Type {
        let mut res = Vec::new();
        for right in self.as_class_info(right.clone()) {
            if let Some(right) = self.unwrap_class_object_silently(&right) {
                res.push(self.intersect_with_fallback(left, &right, &|| right.clone()))
            } else {
                res.push(left.clone());
            }
        }
        self.unions(res)
    }

    fn narrow_is_not_instance(&self, left: &Type, right: &Type) -> Type {
        let mut res = Vec::new();
        for right in self.as_class_info(right.clone()) {
            if let Some(right) = self.unwrap_class_object_silently(&right) {
                res.push(self.subtract(left, &right))
            } else {
                res.push(left.clone())
            }
        }
        self.intersects(&res)
    }

    fn issubclass_result(&self, instance_result: Type, original: &Type) -> Type {
        // If a ClassDef is not narrowed by an `issubclass` call,
        // preserve the information that this is a bare class reference.
        if matches!(original, Type::ClassDef(cls) if instance_result == self.promote_silently(cls))
        {
            original.clone()
        } else {
            Type::type_form(instance_result)
        }
    }

    fn narrow_issubclass(
        &self,
        left: &Type,
        right: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut res = Vec::new();

        let narrow = |left: &Type, right| {
            if let Some(left_untyped) = self.untype_opt(left.clone(), range, errors) {
                self.issubclass_result(
                    self.intersect_with_fallback(&left_untyped, &right, &|| right.clone()),
                    left,
                )
            } else {
                left.clone()
            }
        };

        for right in self.as_class_info(right.clone()) {
            if let Some(right_unwrapped) = self.unwrap_class_object_silently(&right) {
                // Handle type vars specially: we need to enforce restrictions and avoid
                // simplifying them away.
                let mut quantifieds = Vec::new();
                let mut nonquantifieds = Vec::new();
                self.map_over_union(left, |left| {
                    if let Type::Quantified(q) = left {
                        quantifieds.push((**q).clone());
                    } else {
                        nonquantifieds.push(left.clone());
                    }
                });
                for q in quantifieds {
                    // The only time it's safe to simplify a quantified away is when the entire intersection is Never.
                    let intersection = narrow(
                        &q.restriction().as_type(self.stdlib),
                        right_unwrapped.clone(),
                    );
                    res.push(if matches!(&intersection, Type::Type(t) if t.is_never()) {
                        intersection
                    } else {
                        intersect(vec![q.to_type(), right.clone()], right.clone())
                    })
                }
                if !nonquantifieds.is_empty() {
                    res.push(narrow(&self.unions(nonquantifieds), right_unwrapped))
                }
            } else {
                res.push(left.clone())
            }
        }
        self.unions(res)
    }

    fn narrow_is_not_subclass(
        &self,
        left: &Type,
        right: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut res = Vec::new();
        for right in self.as_class_info(right.clone()) {
            if let Some(left_untyped) = self.untype_opt(left.clone(), range, errors)
                && let Some(right) = self.unwrap_class_object_silently(&right)
            {
                res.push(self.issubclass_result(self.subtract(&left_untyped, &right), left))
            } else {
                res.push(left.clone())
            }
        }
        self.intersects(&res)
    }

    fn narrow_length_greater(&self, ty: &Type, len: usize) -> Type {
        self.distribute_over_union(ty, |ty| match ty {
            Type::Tuple(Tuple::Concrete(elts)) if elts.len() <= len => Type::never(),
            Type::Literal(lit)
                if let Lit::Str(x) = &lit.value
                    && x.len() <= len =>
            {
                Type::never()
            }
            Type::ClassType(class)
                if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                    && elts.len() <= len =>
            {
                Type::never()
            }
            _ => ty.clone(),
        })
    }

    fn narrow_length_less_than(&self, ty: &Type, len: usize) -> Type {
        // TODO: simplify some tuple forms
        // - unbounded tuples can be narrowed to empty tuple if len==1
        // - unpacked tuples can be narrowed to concrete prefix+suffix if len==prefix.len()+suffix.len()+1
        // this needs to be done in conjunction with https://github.com/facebook/pyrefly/issues/273
        // otherwise the narrowed forms make weird unions when used with control flow
        self.distribute_over_union(ty, |ty| match ty {
            Type::Tuple(Tuple::Concrete(elts)) if elts.len() >= len => Type::never(),
            Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                if prefix.len() + suffix.len() >= len =>
            {
                Type::never()
            }
            Type::ClassType(class) if let Some(tuple) = self.as_tuple(class) => match tuple {
                Tuple::Concrete(elts) if elts.len() >= len => Type::never(),
                Tuple::Unpacked(box (prefix, _, suffix)) if prefix.len() + suffix.len() >= len => {
                    Type::never()
                }
                _ => ty.clone(),
            },
            _ => ty.clone(),
        })
    }

    // Try to narrow a type based on the type of its facet.
    // For example, if we have a `x.y == 0` check and `x` is some union,
    // we can eliminate cases from the union where `x.y` is some other
    // literal.
    pub fn atomic_narrow_for_facet(
        &self,
        base: &Type,
        facet: &FacetKind,
        op: &AtomicNarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        match op {
            AtomicNarrowOp::Is(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    match right {
                        Type::None
                        | Type::Literal(box Literal {
                            value: Lit::Bool(_) | Lit::Enum(_),
                            ..
                        }) => {
                            if self.is_subset_eq(&right, &facet_ty) {
                                t.clone()
                            } else {
                                Type::never()
                            }
                        }
                        _ => t.clone(),
                    }
                }))
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    match (&facet_ty, &right) {
                        (
                            Type::None
                            | Type::Literal(box Literal {
                                value: Lit::Bool(_) | Lit::Enum(_),
                                ..
                            }),
                            Type::None
                            | Type::Literal(box Literal {
                                value: Lit::Bool(_) | Lit::Enum(_),
                                ..
                            }),
                        ) if self.literal_equal(&right, &facet_ty) => Type::never(),
                        _ => t.clone(),
                    }
                }))
            }
            AtomicNarrowOp::Eq(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    match right {
                        Type::None | Type::Literal(_) => {
                            if self.is_subset_eq(&right, &facet_ty) {
                                t.clone()
                            } else {
                                Type::never()
                            }
                        }
                        _ => t.clone(),
                    }
                }))
            }
            AtomicNarrowOp::NotEq(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    match (&facet_ty, &right) {
                        (Type::None | Type::Literal(_), Type::None | Type::Literal(_))
                            if self.literal_equal(&right, &facet_ty) =>
                        {
                            Type::never()
                        }
                        _ => t.clone(),
                    }
                }))
            }
            _ => None,
        }
    }

    fn atomic_narrow(
        &self,
        ty: &Type,
        op: &AtomicNarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match op {
            AtomicNarrowOp::Placeholder => ty.clone(),
            AtomicNarrowOp::LenEq(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.distribute_over_union(ty, |ty| match ty {
                    Type::Tuple(Tuple::Concrete(elts)) if elts.len() != len => Type::never(),
                    Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                        if prefix.len() + suffix.len() > len =>
                    {
                        Type::never()
                    }
                    Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                        if prefix.len() + suffix.len() == len =>
                    {
                        Type::concrete_tuple(prefix.iter().cloned().chain(suffix.clone()).collect())
                    }
                    Type::Tuple(Tuple::Unpacked(box (
                        prefix,
                        Type::Tuple(Tuple::Unbounded(middle)),
                        suffix,
                    ))) if prefix.len() + suffix.len() < len => {
                        let middle_elements =
                            vec![(**middle).clone(); len - prefix.len() - suffix.len()];
                        Type::concrete_tuple(
                            prefix
                                .iter()
                                .cloned()
                                .chain(middle_elements)
                                .chain(suffix.clone())
                                .collect(),
                        )
                    }
                    Type::Tuple(Tuple::Unbounded(elements)) => {
                        Type::concrete_tuple(vec![(**elements).clone(); len])
                    }
                    Type::ClassType(class)
                        if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                            && elts.len() != len =>
                    {
                        Type::never()
                    }
                    _ => ty.clone(),
                })
            }
            AtomicNarrowOp::LenNotEq(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.distribute_over_union(ty, |ty| match ty {
                    Type::Tuple(Tuple::Concrete(elts)) if elts.len() == len => Type::never(),
                    Type::ClassType(class)
                        if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                            && elts.len() == len =>
                    {
                        Type::never()
                    }
                    _ => ty.clone(),
                })
            }
            AtomicNarrowOp::LenGt(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.narrow_length_greater(ty, len)
            }
            AtomicNarrowOp::LenGte(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                if len == 0 {
                    return ty.clone();
                }
                self.narrow_length_greater(ty, len - 1)
            }
            AtomicNarrowOp::LenLt(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return Type::never();
                };
                if len == 0 {
                    return Type::never();
                }
                self.narrow_length_less_than(ty, len)
            }
            AtomicNarrowOp::LenLte(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(box Literal {
                    value: Lit::Int(lit),
                    ..
                }) = &right
                else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.narrow_length_less_than(ty, len + 1)
            }
            AtomicNarrowOp::In(v) => {
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                let Some(exprs) = exprs else {
                    return ty.clone();
                };
                let mut literal_types = Vec::new();
                for expr in exprs {
                    let expr_ty = self.expr_infer(&expr, errors);
                    if matches!(expr_ty, Type::Literal(_) | Type::None) {
                        literal_types.push(expr_ty);
                    } else {
                        return ty.clone();
                    }
                }
                self.intersect(ty, &self.unions(literal_types))
            }
            AtomicNarrowOp::NotIn(v) => {
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                let Some(exprs) = exprs else {
                    return ty.clone();
                };
                let mut literal_types = Vec::new();
                for expr in exprs {
                    let expr_ty = self.expr_infer(&expr, errors);
                    if matches!(expr_ty, Type::Literal(_) | Type::None) {
                        literal_types.push(expr_ty);
                    } else {
                        return ty.clone();
                    }
                }
                self.distribute_over_union(ty, |t| {
                    let mut result = t.clone();
                    for right in &literal_types {
                        match (t, right) {
                            (_, _) if self.literal_equal(t, right) => {
                                result = Type::never();
                            }
                            (Type::ClassType(cls), Type::Literal(lit))
                                if cls.is_builtin("bool")
                                    && let Lit::Bool(b) = &lit.value =>
                            {
                                result = Lit::Bool(!b).to_implicit_type();
                            }
                            (Type::ClassType(left_cls), Type::Literal(right))
                                if let Lit::Enum(right) = &right.value
                                    && left_cls == &right.class =>
                            {
                                result = self.subtract_enum_member(left_cls, &right.member);
                            }
                            _ => {}
                        }
                    }
                    result
                })
            }
            AtomicNarrowOp::Is(v) => {
                let right = self.expr_infer(v, errors);
                // Get our best approximation of ty & right.
                self.intersect(ty, &right)
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self.expr_infer(v, errors);
                // Get our best approximation of ty - right.
                self.distribute_over_union(ty, |t| {
                    // Only certain literal types can be compared by identity.
                    match (t, &right) {
                        (
                            _,
                            Type::None
                            | Type::Literal(box Literal {
                                value: Lit::Bool(_) | Lit::Enum(_),
                                ..
                            }),
                        ) if self.literal_equal(t, &right) => Type::never(),
                        (Type::ClassType(cls), Type::Literal(lit))
                            if cls.is_builtin("bool")
                                && let Lit::Bool(b) = &lit.value =>
                        {
                            Lit::Bool(!b).to_implicit_type()
                        }
                        (Type::ClassType(left_cls), Type::Literal(right))
                            if let Lit::Enum(right) = &right.value
                                && left_cls == &right.class =>
                        {
                            self.subtract_enum_member(left_cls, &right.member)
                        }
                        _ => t.clone(),
                    }
                })
            }
            AtomicNarrowOp::IsInstance(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_isinstance(ty, &right)
            }
            AtomicNarrowOp::IsNotInstance(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_is_not_instance(ty, &right)
            }
            AtomicNarrowOp::TypeEq(v) => {
                // If type(X) == Y then X can't be a subclass of Y
                // We can't model that, so we narrow it exactly like isinstance(X, Y)
                let right = self.expr_infer(v, errors);
                self.narrow_isinstance(ty, &right)
            }
            // Even if type(X) != Y, X can still be a subclass of Y so we can't do any negative refinement
            AtomicNarrowOp::TypeNotEq(_) => ty.clone(),
            AtomicNarrowOp::IsSubclass(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_issubclass(ty, &right, v.range(), errors)
            }
            AtomicNarrowOp::IsNotSubclass(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_is_not_subclass(ty, &right, v.range(), errors)
            }
            // `hasattr` and `getattr` are handled in `narrow`
            AtomicNarrowOp::HasAttr(_) => ty.clone(),
            AtomicNarrowOp::NotHasAttr(_) => ty.clone(),
            AtomicNarrowOp::HasKey(_) => ty.clone(),
            AtomicNarrowOp::NotHasKey(_) => ty.clone(),
            AtomicNarrowOp::GetAttr(_, _) => ty.clone(),
            AtomicNarrowOp::NotGetAttr(_, _) => ty.clone(),
            AtomicNarrowOp::TypeGuard(t, arguments) => {
                if let CallTargetLookup::Ok(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret =
                        self.call_infer(*call_target, &args, &kws, range, errors, None, None, None);
                    if let Type::TypeGuard(t) = ret {
                        return *t;
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeGuard(_, _) => ty.clone(),
            AtomicNarrowOp::TypeIs(t, arguments) => {
                if let CallTargetLookup::Ok(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret =
                        self.call_infer(*call_target, &args, &kws, range, errors, None, None, None);
                    if let Type::TypeIs(t) = ret {
                        return self.distribute_over_union(&t, |right| {
                            self.intersect_with_fallback(ty, right, &|| {
                                // TODO: falling back to Never when the lhs is a union is a hack to get
                                // reasonable behavior in cases like this:
                                //     def f(x: int | Callable[[], int]):
                                //         if callable(x):
                                //             reveal_type(x)
                                // Both mypy and pyright say that the type of `x` on the last line is
                                // `() -> int`, whereas if we didn't fall back to Never, pyrefly would
                                // say `(int & (...) -> object) | () -> int`. A naive implementation of
                                // calling an intersection type would then lead to the type of `x()`
                                // being `object | int`. This is a surprising and unhelpful type, so we
                                // use Never as the fallback for now.
                                if ty.is_union() {
                                    Type::never()
                                } else {
                                    (*t).clone()
                                }
                            })
                        });
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeIs(t, arguments) => {
                if let CallTargetLookup::Ok(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret =
                        self.call_infer(*call_target, &args, &kws, range, errors, None, None, None);
                    if let Type::TypeIs(t) = ret {
                        return self.subtract(ty, &t);
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::IsTruthy | AtomicNarrowOp::IsFalsy => {
                self.distribute_over_union(ty, |t| {
                    let boolval = matches!(op, AtomicNarrowOp::IsTruthy);
                    // Do not emit errors here: the narrowed range doesn't always correspond to a valid expression
                    // For example, narrowing generated for implicit else branches.
                    if self.as_bool(
                        t,
                        range,
                        &ErrorCollector::new(errors.module().clone(), ErrorStyle::Never),
                    ) == Some(!boolval)
                    {
                        return Type::never();
                    } else if let Type::ClassType(cls) = t {
                        if cls.is_builtin("bool") {
                            return Lit::Bool(boolval).to_implicit_type();
                        }
                        if !boolval {
                            if cls.is_builtin("int") {
                                return LitInt::new(0).to_implicit_type();
                            } else if cls.is_builtin("str") {
                                return Lit::Str("".into()).to_implicit_type();
                            } else if cls.is_builtin("bytes") {
                                let empty = Vec::new();
                                return Lit::Bytes(empty.into_boxed_slice()).to_implicit_type();
                            }
                        }
                    }

                    t.clone()
                })
            }
            AtomicNarrowOp::Eq(v) => {
                let right = self.expr_infer(v, errors);
                if matches!(right, Type::Literal(_) | Type::None) {
                    self.intersect(ty, &right)
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::NotEq(v) => {
                let right = self.expr_infer(v, errors);
                if matches!(right, Type::Literal(_) | Type::None) {
                    self.distribute_over_union(ty, |t| match (t, &right) {
                        (_, _) if self.literal_equal(t, &right) => Type::never(),
                        (Type::ClassType(cls), Type::Literal(lit))
                            if cls.is_builtin("bool")
                                && let Lit::Bool(b) = &lit.value =>
                        {
                            Lit::Bool(!b).to_implicit_type()
                        }
                        (Type::ClassType(left_cls), Type::Literal(right))
                            if let Lit::Enum(right) = &right.value
                                && left_cls == &right.class =>
                        {
                            self.subtract_enum_member(left_cls, &right.member)
                        }
                        _ => t.clone(),
                    })
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::Call(func, args) | AtomicNarrowOp::NotCall(func, args) => {
                if let Some(resolved_op) = self.resolve_narrowing_call(func, args, errors) {
                    if matches!(op, AtomicNarrowOp::Call(..)) {
                        self.atomic_narrow(ty, &resolved_op, range, errors)
                    } else {
                        self.atomic_narrow(ty, &resolved_op.negate(), range, errors)
                    }
                } else {
                    ty.clone()
                }
            }
        }
    }

    fn get_facet_chain_type(
        &self,
        base: &TypeInfo,
        facet_chain: &FacetChain,
        range: TextRange,
    ) -> Type {
        // We don't want to throw any attribute access or indexing errors when narrowing - the same code is traversed
        // separately for type checking, and there might be error context then we don't have here.
        let ignore_errors = self.error_swallower();
        let (first_facet, remaining_facets) = facet_chain.facets().clone().split_off_first();
        self.narrowable_for_facet_chain(
            base,
            &first_facet,
            &remaining_facets,
            range,
            &ignore_errors,
        )
    }

    fn narrowable_for_facet_chain(
        &self,
        base: &TypeInfo,
        first_facet: &FacetKind,
        remaining_facets: &[FacetKind],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match first_facet {
            FacetKind::Attribute(first_attr_name) => match remaining_facets.split_first() {
                None => match base.type_at_facet(first_facet) {
                    Some(ty) => self.force_for_narrowing(ty, range, errors),
                    None => self.narrowable_for_attr(base.ty(), first_attr_name, range, errors),
                },
                Some((next_name, remaining_facets)) => {
                    let base = self.attr_infer(base, first_attr_name, range, errors, None);
                    self.narrowable_for_facet_chain(
                        &base,
                        next_name,
                        remaining_facets,
                        range,
                        errors,
                    )
                }
            },
            FacetKind::Index(idx) => {
                // We synthesize a slice expression for the subscript here
                // Use a synthesized fake range to avoid overwriting typing traces
                let synthesized_slice = Expr::NumberLiteral(ExprNumberLiteral {
                    node_index: AtomicNodeIndex::default(),
                    range: TextRange::empty(TextSize::from(0)),
                    value: Number::Int(Int::from(*idx as u64)),
                });
                match remaining_facets.split_first() {
                    None => match base.type_at_facet(first_facet) {
                        Some(ty) => self.force_for_narrowing(ty, range, errors),
                        None => self.subscript_infer_for_type(
                            base.ty(),
                            &synthesized_slice,
                            range,
                            errors,
                        ),
                    },
                    Some((next_name, remaining_facets)) => {
                        let base_ty = self.subscript_infer(base, &synthesized_slice, range, errors);
                        self.narrowable_for_facet_chain(
                            &base_ty,
                            next_name,
                            remaining_facets,
                            range,
                            errors,
                        )
                    }
                }
            }
            FacetKind::Key(key) => {
                // We synthesize a slice expression for the subscript here
                // Use a synthesized fake range to avoid overwriting typing traces
                let synthesized_slice = Ast::str_expr(key, TextRange::empty(TextSize::from(0)));
                match remaining_facets.split_first() {
                    None => match base.type_at_facet(first_facet) {
                        Some(ty) => self.force_for_narrowing(ty, range, errors),
                        None => self.subscript_infer_for_type(
                            base.ty(),
                            &synthesized_slice,
                            range,
                            errors,
                        ),
                    },
                    Some((next_name, remaining_facets)) => {
                        let base_ty = self.subscript_infer(base, &synthesized_slice, range, errors);
                        self.narrowable_for_facet_chain(
                            &base_ty,
                            next_name,
                            remaining_facets,
                            range,
                            errors,
                        )
                    }
                }
            }
        }
    }

    pub fn narrow(
        &self,
        type_info: &TypeInfo,
        op: &NarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        match op {
            NarrowOp::Atomic(subject, AtomicNarrowOp::HasKey(key)) => {
                let resolved_chain = subject
                    .as_ref()
                    .and_then(|s| self.resolve_facet_chain(s.chain.clone()));
                let base_ty = match (&subject, &resolved_chain) {
                    (Some(_), Some(chain)) => self.get_facet_chain_type(type_info, chain, range),
                    (Some(_), None) => return type_info.clone(),
                    (None, _) => self.force_for_narrowing(type_info.ty(), range, errors),
                };
                if matches!(base_ty, Type::TypedDict(_)) {
                    let key_facet = FacetKind::Key(key.to_string());
                    let facets = match resolved_chain {
                        Some(chain) => {
                            let mut new_facets = chain.facets().clone();
                            new_facets.push(key_facet);
                            new_facets
                        }
                        None => Vec1::new(key_facet),
                    };
                    let chain = FacetChain::new(facets);
                    // Apply a facet narrow w/ that key's type, so that the usual subscript inference
                    // code path which raises a warning for NotRequired keys does not execute later
                    let value_ty = self.get_facet_chain_type(type_info, &chain, range);
                    type_info.with_narrow(chain.facets(), value_ty)
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Atomic(subject, AtomicNarrowOp::NotHasKey(key)) => {
                let resolved_chain = subject
                    .as_ref()
                    .and_then(|s| self.resolve_facet_chain(s.chain.clone()));
                let base_ty = match (&subject, &resolved_chain) {
                    (Some(_), Some(chain)) => self.get_facet_chain_type(type_info, chain, range),
                    (Some(_), None) => return type_info.clone(),
                    (None, _) => self.force_for_narrowing(type_info.ty(), range, errors),
                };
                if matches!(base_ty, Type::TypedDict(_)) {
                    let key_facet = FacetKind::Key(key.to_string());
                    let facets = match resolved_chain {
                        Some(chain) => {
                            let mut new_facets = chain.facets().clone();
                            new_facets.push(key_facet);
                            new_facets
                        }
                        None => Vec1::new(key_facet),
                    };
                    // Invalidate existing facet narrows
                    let mut type_info = type_info.clone();
                    type_info.update_for_assignment(&facets, None);
                    type_info
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Atomic(subject, AtomicNarrowOp::HasAttr(attr)) => {
                let resolved_chain = subject
                    .as_ref()
                    .and_then(|s| self.resolve_facet_chain(s.chain.clone()));
                let base_ty = match (&subject, &resolved_chain) {
                    (Some(_), Some(chain)) => self.get_facet_chain_type(type_info, chain, range),
                    (Some(_), None) => return type_info.clone(),
                    (None, _) => self.force_for_narrowing(type_info.ty(), range, errors),
                };
                // We only narrow the attribute to `Any` if the attribute does not exist
                if !self.has_attr(&base_ty, attr) {
                    let attr_facet = FacetKind::Attribute(attr.clone());
                    let facets = match resolved_chain {
                        Some(chain) => {
                            let mut new_facets = chain.facets().clone();
                            new_facets.push(attr_facet);
                            new_facets
                        }
                        None => Vec1::new(attr_facet),
                    };
                    type_info.with_narrow(&facets, Type::any_implicit())
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Atomic(subject, AtomicNarrowOp::GetAttr(attr, default)) => {
                let suppress_errors =
                    ErrorCollector::new(errors.module().clone(), ErrorStyle::Never);
                let default_ty = default
                    .as_ref()
                    .map_or(Type::None, |v| self.expr_infer(v, &suppress_errors));
                // We can't narrow the type if the specified default is not falsy
                if self.as_bool(&default_ty, range, &suppress_errors) != Some(false) {
                    return type_info.clone();
                }
                let resolved_chain = subject
                    .as_ref()
                    .and_then(|s| self.resolve_facet_chain(s.chain.clone()));
                let base_ty = match (&subject, &resolved_chain) {
                    (Some(_), Some(chain)) => self.get_facet_chain_type(type_info, chain, range),
                    (Some(_), None) => return type_info.clone(),
                    (None, _) => self.force_for_narrowing(type_info.ty(), range, errors),
                };
                let attr_ty =
                    self.attr_infer_for_type(&base_ty, attr, range, &suppress_errors, None);
                let attr_facet = FacetKind::Attribute(attr.clone());
                let facets = match resolved_chain {
                    Some(chain) => {
                        let mut new_facets = chain.facets().clone();
                        new_facets.push(attr_facet);
                        new_facets
                    }
                    None => Vec1::new(attr_facet),
                };
                // Given that the default is falsy:
                // If the attribute does not exist we narrow to `Any`
                // If the attribute exists we narrow it to be truthy
                if attr_ty == Type::any_error() {
                    type_info.with_narrow(&facets, Type::any_implicit())
                } else {
                    let narrowed_ty = self.atomic_narrow(
                        &attr_ty,
                        &AtomicNarrowOp::IsTruthy,
                        range,
                        &suppress_errors,
                    );
                    type_info.with_narrow(&facets, narrowed_ty)
                }
            }
            NarrowOp::Atomic(None, op) => {
                let ty = self.atomic_narrow(
                    &self.force_for_narrowing(type_info.ty(), range, errors),
                    op,
                    range,
                    errors,
                );
                type_info.clone().with_ty(ty)
            }
            NarrowOp::Atomic(Some(facet_subject), op) => {
                let Some(resolved_chain) = self.resolve_facet_chain(facet_subject.chain.clone())
                else {
                    return type_info.clone();
                };
                let Some(op_for_narrow) = (match op {
                    AtomicNarrowOp::Call(func, args) => {
                        self.resolve_narrowing_call(func.as_ref(), args, errors)
                    }
                    AtomicNarrowOp::NotCall(func, args) => self
                        .resolve_narrowing_call(func.as_ref(), args, errors)
                        .map(|resolved_op| resolved_op.negate()),
                    _ => Some(op.clone()),
                }) else {
                    return type_info.clone();
                };
                if facet_subject.origin == FacetOrigin::GetMethod
                    && !self.supports_dict_get_subject(type_info, facet_subject, range)
                {
                    return type_info.clone();
                }
                let ty = self.atomic_narrow(
                    &self.get_facet_chain_type(type_info, &resolved_chain, range),
                    &op_for_narrow,
                    range,
                    errors,
                );
                let mut narrowed = type_info.with_narrow(resolved_chain.facets(), ty);
                // For certain types of narrows, we can also narrow the parent of the current subject
                // If `.get()` on a dict or TypedDict is falsy, the key may not be present at all
                // We should invalidate any existing narrows
                if let Some((last, prefix)) = resolved_chain.facets().split_last() {
                    match Vec1::try_from(prefix) {
                        Ok(prefix_facets) => {
                            let prefix_chain = FacetChain::new(prefix_facets);
                            let base_ty =
                                self.get_facet_chain_type(type_info, &prefix_chain, range);
                            let dict_get_key_falsy =
                                matches!(op_for_narrow, AtomicNarrowOp::IsFalsy)
                                    && matches!(last, FacetKind::Key(_));
                            if dict_get_key_falsy {
                                narrowed.update_for_assignment(resolved_chain.facets(), None);
                            } else if let Some(narrowed_ty) = self.atomic_narrow_for_facet(
                                &base_ty,
                                last,
                                &op_for_narrow,
                                range,
                                errors,
                            ) && narrowed_ty != base_ty
                            {
                                narrowed = narrowed.with_narrow(prefix_chain.facets(), narrowed_ty);
                            }
                        }
                        _ => {
                            let base_ty = type_info.ty();
                            let dict_get_key_falsy =
                                matches!(op_for_narrow, AtomicNarrowOp::IsFalsy)
                                    && matches!(last, FacetKind::Key(_));
                            if dict_get_key_falsy {
                                narrowed.update_for_assignment(resolved_chain.facets(), None);
                            } else if let Some(narrowed_ty) = self.atomic_narrow_for_facet(
                                base_ty,
                                last,
                                &op_for_narrow,
                                range,
                                errors,
                            ) && narrowed_ty != *base_ty
                            {
                                narrowed = narrowed.clone().with_ty(narrowed_ty);
                            }
                        }
                    };
                }
                narrowed
            }
            NarrowOp::And(ops) => {
                let mut ops_iter = ops.iter();
                if let Some(first_op) = ops_iter.next() {
                    let mut ret = self.narrow(type_info, first_op, range, errors);
                    for next_op in ops_iter {
                        ret = self.narrow(&ret, next_op, range, errors);
                    }
                    ret
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Or(ops) => TypeInfo::join(
                ops.map(|op| self.narrow(type_info, op, range, errors)),
                &|tys| self.unions(tys),
                &|got, want| self.is_subset_eq(got, want),
                JoinStyle::SimpleMerge,
            ),
        }
    }

    /// We only narrow `x.get("key")` if `x` resolves to a `dict`
    fn supports_dict_get_subject(
        &self,
        type_info: &TypeInfo,
        subject: &FacetSubject,
        range: TextRange,
    ) -> bool {
        let Some(resolved_chain) = self.resolve_facet_chain(subject.chain.clone()) else {
            return false;
        };
        let base_ty = if resolved_chain.facets().len() == 1 {
            type_info.ty().clone()
        } else {
            let prefix: Vec<_> = resolved_chain
                .facets()
                .iter()
                .take(resolved_chain.facets().len() - 1)
                .cloned()
                .collect();
            match Vec1::try_from_vec(prefix) {
                Ok(vec1) => {
                    let prefix_chain = FacetChain::new(vec1);
                    self.get_facet_chain_type(type_info, &prefix_chain, range)
                }
                Err(_) => return false,
            }
        };
        self.is_dict_like(&base_ty)
    }

    fn is_non_flag_enum(&self, cls: &ClassType) -> bool {
        self.get_metadata_for_class(cls.class_object())
            .enum_metadata()
            .is_some_and(|meta| !meta.is_flag)
    }

    fn is_enum_class_or_literal_union(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) | Type::SelfType(cls) => self.is_non_flag_enum(cls),
            Type::Union(union) => {
                let union = union.as_ref();
                !union.members.is_empty()
                    && union
                        .members
                        .iter()
                        .all(|member| matches!(member, Type::Literal(_)))
            }
            _ => false,
        }
    }

    fn format_missing_literal_cases(&self, ty: &Type) -> Option<String> {
        fn collect_cases(ty: &Type, acc: &mut Vec<String>) -> bool {
            match ty {
                Type::Literal(lit) => {
                    acc.push(format!("{}", lit.value));
                    true
                }
                Type::Union(union) => {
                    let union = union.as_ref();
                    union
                        .members
                        .iter()
                        .all(|member| collect_cases(member, acc))
                }
                _ => false,
            }
        }

        let mut cases = Vec::new();
        if collect_cases(ty, &mut cases) {
            Some(cases.join(", "))
        } else {
            None
        }
    }

    pub fn check_match_exhaustiveness(
        &self,
        subject_idx: &Idx<Key>,
        narrowing_subject: &NarrowingSubject,
        narrow_ops_for_fall_through: &(Box<NarrowOp>, TextRange),
        subject_range: &TextRange,
        errors: &ErrorCollector,
    ) {
        let (op, narrow_range) = narrow_ops_for_fall_through;
        let subject_info = self.get_idx(*subject_idx);
        let mut subject_ty = subject_info.ty().clone();
        self.expand_vars_mut(&mut subject_ty);
        // We only check match exhaustiveness if the subject is an enum or a union of enum literals
        if !self.is_enum_class_or_literal_union(&subject_ty) {
            return;
        }
        let ignore_errors = self.error_swallower();
        // Get the narrowed type of the match subject when none of the cases match
        let mut remaining_ty = match narrowing_subject {
            NarrowingSubject::Name(_) => self
                .narrow(&subject_info, op.as_ref(), *narrow_range, &ignore_errors)
                .ty()
                .clone(),
            NarrowingSubject::Facets(_, facets) => {
                let Some(resolved_chain) = self.resolve_facet_chain(facets.chain.clone()) else {
                    return;
                };
                // If the narrowing subject is the facet of some variable like `x.foo`,
                // We need to make a `TypeInfo` rooted at `x` using the type of `x.foo`
                let type_info = TypeInfo::of_ty(Type::any_implicit());
                let narrowing_subject_info =
                    type_info.with_narrow(resolved_chain.facets(), subject_ty.clone());
                let narrowed = self.narrow(
                    &narrowing_subject_info,
                    op.as_ref(),
                    *narrow_range,
                    &ignore_errors,
                );
                self.get_facet_chain_type(&narrowed, &resolved_chain, *subject_range)
            }
        };
        self.expand_vars_mut(&mut remaining_ty);
        // If the result is `Never` then the cases were exhaustive
        if remaining_ty.is_never() || remaining_ty.is_any() {
            return;
        }
        let subject_display = self.for_display(subject_ty);
        let remaining_display = self.for_display(remaining_ty.clone());
        let ctx = TypeDisplayContext::new(&[&subject_display, &remaining_display]);
        let mut msg = vec1![format!(
            "Match on `{}` is not exhaustive",
            ctx.display(&subject_display)
        )];
        if let Some(missing_cases) = self.format_missing_literal_cases(&remaining_ty) {
            msg.push(format!("Missing cases: {}", missing_cases));
        }
        errors.add(
            *subject_range,
            ErrorInfo::Kind(ErrorKind::NonExhaustiveMatch),
            msg,
        );
    }

    pub fn resolve_facet_chain(&self, unresolved: UnresolvedFacetChain) -> Option<FacetChain> {
        let resolved: Option<Vec<FacetKind>> = unresolved
            .facets()
            .iter()
            .map(|kind| self.resolve_facet_kind(kind.clone()))
            .collect();
        resolved.map(|facets| FacetChain::new(Vec1::try_from_vec(facets).unwrap()))
    }

    pub fn resolve_facet_kind(&self, unresolved: UnresolvedFacetKind) -> Option<FacetKind> {
        match unresolved {
            UnresolvedFacetKind::Attribute(name) => Some(FacetKind::Attribute(name)),
            UnresolvedFacetKind::Index(idx) => Some(FacetKind::Index(idx)),
            UnresolvedFacetKind::Key(key) => Some(FacetKind::Key(key)),
            UnresolvedFacetKind::VariableSubscript(expr_name) => {
                let suppress_errors = self.error_swallower();
                let ty = self.expr_infer(&Expr::Name(expr_name), &suppress_errors);
                match &ty {
                    Type::Literal(lit) if let Lit::Int(lit_int) = &lit.value => lit_int
                        .as_i64()
                        .and_then(|i| i.to_usize())
                        .map(FacetKind::Index),
                    Type::Literal(lit) if let Lit::Str(s) = &lit.value => {
                        Some(FacetKind::Key(s.to_string()))
                    }
                    _ => None,
                }
            }
        }
    }

    fn literal_equal(&self, left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::None, Type::None) => true,
            (Type::Literal(left), Type::Literal(right)) => left.value == right.value,
            _ => false,
        }
    }
}
