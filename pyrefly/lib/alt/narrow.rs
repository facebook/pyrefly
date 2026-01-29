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
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::Class;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::facet::FacetChain;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::facet::UnresolvedFacetChain;
use pyrefly_types::facet::UnresolvedFacetKind;
use pyrefly_types::simplify::intersect;
use pyrefly_types::simplify::simplify_tuples;
use pyrefly_types::type_info::JoinStyle;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Arguments;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallTargetLookup;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::FacetOrigin;
use crate::binding::narrow::FacetSubject;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowSource;
use crate::binding::narrow::NarrowingSubject;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
use crate::types::callable::FunctionKind;
use crate::types::class::ClassType;
use crate::types::lit_int::LitInt;
use crate::types::literal::Lit;
use crate::types::literal::LitSentinel;
use crate::types::literal::LitStyle;
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
            return self.heap.mk_class_type(cls.clone());
        }
        let e = self.get_enum_from_class(cls.class_object()).unwrap();
        // Enums derived from enum.Flag cannot be treated as a union of their members
        if e.is_flag {
            return self.heap.mk_class_type(cls.clone());
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

    fn is_final(&self, cls: &ClassType) -> bool {
        let class = cls.class_object();
        self.get_metadata_for_class(class).is_final()
            || (self.get_enum_from_class(class).is_some()
                && !self.get_enum_members(class).is_empty())
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
            self.heap.mk_never()
        } else {
            let fallback = fallback();
            if fallback.is_never() {
                fallback
            } else if let Type::ClassType(left_cls) = left
                && let Type::ClassType(right_cls) = right
                && (self.is_final(left_cls) || self.is_final(right_cls))
            {
                // The only way for `left & right` to exist is if it is an instance of a class that
                // multiply inherits from both `left` and `right`'s classes. But at least one of
                // the classes is final, so such a class does not exist.
                self.heap.mk_never()
            } else {
                let left_base = self.disjoint_base(left);
                let right_base = self.disjoint_base(right);
                if self.has_superclass(left_base, right_base)
                    || self.has_superclass(right_base, left_base)
                {
                    intersect(vec![left.clone(), right.clone()], fallback, self.heap)
                } else {
                    // A common subclass of these two classes cannot exist.
                    self.heap.mk_never()
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
        self.intersect_with_fallback(left, right, &|| self.heap.mk_never())
    }

    /// Calculate the intersection of a number of types
    pub fn intersects(&self, ts: &[Type]) -> Type {
        match ts {
            [] => self.heap.mk_class_type(self.stdlib.object().clone()),
            [ty] => ty.clone(),
            [ty0, ty1] => self.intersect(ty0, ty1),
            [ty0, ts @ ..] => self.intersect(ty0, &self.intersects(ts)),
        }
    }

    fn subtract(&self, left: &Type, right: &Type) -> Type {
        self.distribute_over_union(left, |left| {
            // Special is_any check because `Any <: int` as a special case, but would mess up this.
            if !left.is_any() && self.is_subset_eq(left, right) {
                self.heap.mk_never()
            } else {
                left.clone()
            }
        })
    }

    fn original_binding_for_idx(&self, mut idx: Idx<Key>) -> Option<(Idx<Key>, &Binding)> {
        let mut gas = 100;
        loop {
            if gas == 0 {
                return None;
            }
            gas -= 1;
            let binding = self.bindings().get(idx);
            match binding {
                Binding::Forward(fwd_idx)
                | Binding::CompletedPartialType(fwd_idx, _)
                | Binding::PartialTypeWithUpstreamsCompleted(fwd_idx, _)
                | Binding::Phi(JoinStyle::NarrowOf(fwd_idx), _) => {
                    idx = *fwd_idx;
                }
                _ => return Some((idx, binding)),
            }
        }
    }

    fn final_sentinel_literal_for_expr(
        &self,
        expr: &Expr,
        _errors: &ErrorCollector,
    ) -> Option<Type> {
        let Expr::Name(name) = expr else {
            return None;
        };
        if Ast::is_synthesized_empty_name(name) {
            return None;
        }
        let key = Key::BoundName(ShortIdentifier::expr_name(name));
        let idx = self.bindings().key_to_idx_hashed_opt(Hashed::new(&key))?;
        let (origin_idx, binding) = self.original_binding_for_idx(idx)?;
        let Binding::NameAssign {
            annotation: Some((_, annot_idx)),
            expr,
            ..
        } = binding
        else {
            return None;
        };
        let annotation = self.get_idx(*annot_idx);
        if !annotation.annotation.is_final() {
            return None;
        }
        let init_ty = self.expr_infer(expr.as_ref(), &self.error_swallower());
        match init_ty {
            Type::Literal(_) | Type::None => Some(init_ty),
            Type::ClassType(class) => {
                let sentinel = Lit::Sentinel(Box::new(LitSentinel {
                    module: self.module().name(),
                    name: name.id.clone(),
                    range: Some(self.bindings().idx_to_key::<Key>(origin_idx).range()),
                    class,
                }));
                Some(Type::Literal(Box::new(Literal {
                    value: sentinel,
                    style: LitStyle::Implicit,
                })))
            }
            _ => None,
        }
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
                Some(CalleeKind::Function(FunctionKind::IsInstance)) => Some(
                    AtomicNarrowOp::IsInstance(second_arg.clone(), NarrowSource::Call),
                ),
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
            res.push(self.distribute_over_union(left, |l| {
                if let Some((tparams, right)) = self.unwrap_class_object_silently(&right) {
                    let (vs, right) = self
                        .solver()
                        .fresh_quantified(&tparams, right, self.uniques);
                    let result = self.intersect_with_fallback(l, &right, &|| {
                        // TODO: falling back to Never when the lhs is a union is a hack to get
                        // reasonable behavior in cases like this:
                        //     def f(x: int | list[int]):
                        //         if isinstance(x, Iterable):
                        //             reveal_type(x)
                        // We want to narrow x to just `list[int]`, rather than `(int & Iterable[Unknown]) | list[int]`
                        if left.is_union() {
                            self.heap.mk_never()
                        } else {
                            right.clone()
                        }
                    });
                    // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
                    // * If `left` is an invalid specialization, the error has already been reported at its definition site.
                    // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
                    let _specialization_errors = self.solver().finish_quantified(vs, false);
                    result
                } else {
                    l.clone()
                }
            }));
        }
        self.unions(res)
    }

    fn narrow_is_not_instance(
        &self,
        left: &Type,
        right_expr: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        let mut res = Vec::new();
        for (right, allows_negative_narrow) in self.expr_as_class_info(right_expr, errors) {
            res.push(self.distribute_over_union(left, |l| {
                if allows_negative_narrow
                    && let Some((tparams, right)) = self.unwrap_class_object_silently(&right)
                {
                    let (vs, right) = self
                        .solver()
                        .fresh_quantified(&tparams, right, self.uniques);
                    let result = self.subtract(l, &right);
                    // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
                    // * If `left` is an invalid specialization, the error has already been reported at its definition site.
                    // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
                    let _specialization_errors = self.solver().finish_quantified(vs, false);
                    result
                } else {
                    l.clone()
                }
            }));
        }
        self.intersects(&res)
    }

    /// Turn an expression into a list of (type, allows_negative_narrow) pairs.
    /// allows_negative_narrow means that we can do `not isinstance`/`not issubclass` narrowing
    /// with the type. We allow negative narrowing as long as it is not definitely unsafe - that
    /// is, if we're unsure, we allow it.
    fn expr_as_class_info(&self, e: &Expr, errors: &ErrorCollector) -> Vec<(Type, bool)> {
        fn f<'a, Ans: LookupAnswer>(
            me: &AnswersSolver<'a, Ans>,
            e: &Expr,
            res: &mut Vec<(Type, bool)>,
            errors: &ErrorCollector,
        ) {
            match e {
                Expr::BinOp(ExprBinOp {
                    left,
                    op: Operator::BitOr,
                    right,
                    ..
                }) => {
                    f(me, left, res, errors);
                    f(me, right, res, errors);
                }
                Expr::Tuple(tuple) if !tuple.elts.iter().any(|e| matches!(e, Expr::Starred(_))) => {
                    for e in &tuple.elts {
                        f(me, e, res, errors);
                    }
                }
                _ => {
                    let t = me.expr_infer(e, errors);
                    if let Type::Type(box Type::ClassType(cls)) = &t {
                        // If `C` is not final, `type[C]` may be a subclass of `C`,
                        // making negative narrowing unsafe.
                        let allows_negative_narrow = me.is_final(cls);
                        res.push((t, allows_negative_narrow));
                    } else {
                        for t in me.as_class_info(t) {
                            res.push((t, true));
                        }
                    }
                }
            }
        }
        let mut res = Vec::new();
        f(self, e, &mut res, errors);
        res
    }

    fn issubclass_result(&self, instance_result: Type, original: &Type) -> Type {
        // If a ClassDef is not narrowed by an `issubclass` call,
        // preserve the information that this is a bare class reference.
        if matches!(original, Type::ClassDef(cls) if instance_result == self.promote_silently(cls))
        {
            original.clone()
        } else {
            self.heap.mk_type_form(instance_result)
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
            if let Some((tparams, right_unwrapped)) = self.unwrap_class_object_silently(&right) {
                // Handle type vars specially: we need to enforce restrictions and avoid
                // simplifying them away.
                let (vs, right_unwrapped) =
                    self.solver()
                        .fresh_quantified(&tparams, right_unwrapped, self.uniques);
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
                        &q.restriction().as_type(self.stdlib, self.heap),
                        right_unwrapped.clone(),
                    );
                    res.push(if matches!(&intersection, Type::Type(t) if t.is_never()) {
                        intersection
                    } else {
                        intersect(
                            vec![q.to_type(self.heap), right.clone()],
                            right.clone(),
                            self.heap,
                        )
                    })
                }
                if !nonquantifieds.is_empty() {
                    res.push(narrow(&self.unions(nonquantifieds), right_unwrapped));
                }
                // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
                // * If `left` is an invalid specialization, the error has already been reported at its definition site.
                // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
                let _specialization_errors = self.solver().finish_quantified(vs, false);
            } else {
                res.push(left.clone())
            }
        }
        self.unions(res)
    }

    fn narrow_is_not_subclass(
        &self,
        left: &Type,
        right_expr: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        let mut res = Vec::new();
        for (right, allows_negative_narrow) in self.expr_as_class_info(right_expr, errors) {
            if allows_negative_narrow
                && let Some(left_untyped) =
                    self.untype_opt(left.clone(), right_expr.range(), errors)
                && let Some((tparams, right)) = self.unwrap_class_object_silently(&right)
            {
                let (vs, right) = self
                    .solver()
                    .fresh_quantified(&tparams, right, self.uniques);
                res.push(self.issubclass_result(self.subtract(&left_untyped, &right), left));
                // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
                // * If `left` is an invalid specialization, the error has already been reported at its definition site.
                // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
                let _specialization_errors = self.solver().finish_quantified(vs, false);
            } else {
                res.push(left.clone())
            }
        }
        self.intersects(&res)
    }

    fn narrow_length_greater(&self, ty: &Type, len: usize) -> Type {
        self.distribute_over_union(ty, |ty| match ty {
            Type::Tuple(Tuple::Concrete(elts)) if elts.len() <= len => self.heap.mk_never(),
            Type::Literal(lit)
                if let Lit::Str(x) = &lit.value
                    && x.len() <= len =>
            {
                self.heap.mk_never()
            }
            Type::ClassType(class)
                if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                    && elts.len() <= len =>
            {
                self.heap.mk_never()
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
            Type::Tuple(Tuple::Concrete(elts)) if elts.len() >= len => self.heap.mk_never(),
            Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                if prefix.len() + suffix.len() >= len =>
            {
                self.heap.mk_never()
            }
            Type::ClassType(class) if let Some(tuple) = self.as_tuple(class) => match tuple {
                Tuple::Concrete(elts) if elts.len() >= len => self.heap.mk_never(),
                Tuple::Unpacked(box (prefix, _, suffix)) if prefix.len() + suffix.len() >= len => {
                    self.heap.mk_never()
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
                let right = self
                    .final_sentinel_literal_for_expr(v, errors)
                    .unwrap_or_else(|| self.expr_infer(v, errors));
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
                            value: Lit::Bool(_) | Lit::Enum(_) | Lit::Sentinel(_),
                            ..
                        }) => {
                            if self.is_subset_eq(&right, &facet_ty) {
                                t.clone()
                            } else {
                                self.heap.mk_never()
                            }
                        }
                        _ => t.clone(),
                    }
                }))
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self
                    .final_sentinel_literal_for_expr(v, errors)
                    .unwrap_or_else(|| self.expr_infer(v, errors));
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
                                value: Lit::Bool(_) | Lit::Enum(_) | Lit::Sentinel(_),
                                ..
                            }),
                            Type::None
                            | Type::Literal(box Literal {
                                value: Lit::Bool(_) | Lit::Enum(_) | Lit::Sentinel(_),
                                ..
                            }),
                        ) if self.literal_equal(&right, &facet_ty) => self.heap.mk_never(),
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
                                self.heap.mk_never()
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
                            self.heap.mk_never()
                        }
                        _ => t.clone(),
                    }
                }))
            }
            _ => None,
        }
    }

    fn tuple_len_eq(
        &self,
        tuple: &Tuple,
        len: usize,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match tuple {
            Tuple::Concrete(elts) if elts.len() != len => self.heap.mk_never(),
            Tuple::Unpacked(box (prefix, _, suffix)) if prefix.len() + suffix.len() > len => {
                self.heap.mk_never()
            }
            Tuple::Unpacked(box (prefix, _, suffix)) if prefix.len() + suffix.len() == len => self
                .heap
                .mk_concrete_tuple(prefix.iter().cloned().chain(suffix.clone()).collect()),
            Tuple::Unpacked(box (prefix, Type::Tuple(Tuple::Unbounded(middle)), suffix))
                if prefix.len() + suffix.len() < len =>
            {
                let middle_elements = vec![(**middle).clone(); len - prefix.len() - suffix.len()];
                self.heap.mk_concrete_tuple(
                    prefix
                        .iter()
                        .cloned()
                        .chain(middle_elements)
                        .chain(suffix.clone())
                        .collect(),
                )
            }
            Tuple::Unpacked(box (prefix, middle_var @ Type::Var(_), suffix)) => {
                let forced_middle = self.force_for_narrowing(middle_var, range, errors);
                let new_tuple =
                    Tuple::Unpacked(Box::new((prefix.clone(), forced_middle, suffix.clone())));
                self.tuple_len_eq(&simplify_tuples(new_tuple, self.heap), len, range, errors)
            }
            Tuple::Unbounded(elements) => {
                self.heap.mk_concrete_tuple(vec![(**elements).clone(); len])
            }
            _ => self.heap.mk_tuple(tuple.clone()),
        }
    }

    fn tuple_len_not_eq(
        &self,
        tuple: &Tuple,
        len: usize,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match tuple {
            Tuple::Concrete(elts) if elts.len() == len => self.heap.mk_never(),
            Tuple::Unpacked(box (prefix, middle_var @ Type::Var(_), suffix)) => {
                let forced_middle = self.force_for_narrowing(middle_var, range, errors);
                let new_tuple =
                    Tuple::Unpacked(Box::new((prefix.clone(), forced_middle, suffix.clone())));
                self.tuple_len_not_eq(&simplify_tuples(new_tuple, self.heap), len, range, errors)
            }
            _ => self.heap.mk_tuple(tuple.clone()),
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
                    Type::ClassType(class)
                        if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                            && elts.len() != len =>
                    {
                        self.heap.mk_never()
                    }
                    Type::Tuple(tuple) => self.tuple_len_eq(tuple, len, range, errors),
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
                    Type::ClassType(class)
                        if let Some(Tuple::Concrete(elts)) = self.as_tuple(class)
                            && elts.len() == len =>
                    {
                        self.heap.mk_never()
                    }
                    Type::Tuple(tuple) => self.tuple_len_not_eq(tuple, len, range, errors),
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
                    return self.heap.mk_never();
                };
                if len == 0 {
                    return self.heap.mk_never();
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
            AtomicNarrowOp::IsSequence => {
                self.is_type_for_pattern(ty, |t| self.is_sequence_for_pattern(t))
            }
            AtomicNarrowOp::IsNotSequence => {
                self.is_not_type_for_pattern(ty, |t| self.is_sequence_for_pattern(t))
            }
            AtomicNarrowOp::IsMapping => {
                let mapping = self.heap.mk_class_type(
                    self.stdlib
                        .mapping(self.heap.mk_any_implicit(), self.heap.mk_any_implicit()),
                );
                self.is_type_for_pattern(ty, |t| self.is_subset_eq(t, &mapping))
            }
            AtomicNarrowOp::IsNotMapping => {
                let mapping = self.heap.mk_class_type(
                    self.stdlib
                        .mapping(self.heap.mk_any_implicit(), self.heap.mk_any_implicit()),
                );
                self.is_not_type_for_pattern(ty, |t| self.is_subset_eq(t, &mapping))
            }
            AtomicNarrowOp::In(v) => {
                // First, check for List, Tuple, and Set literal expressions (syntactic check,
                // avoids type inference on the container itself)
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                if let Some(exprs) = exprs {
                    // Bail out if any element is a starred expression (e.g., `x in [*y, 1]`).
                    // We can't know all values at compile time when unpacking occurs.
                    if exprs.iter().any(|e| matches!(e, Expr::Starred(_))) {
                        return ty.clone();
                    }
                    let mut literal_types = Vec::new();
                    for expr in exprs {
                        let expr_ty = self.expr_infer(&expr, errors);
                        match expr_ty {
                            Type::Literal(_) | Type::None => {
                                literal_types.push(expr_ty);
                            }
                            // Bare class names (e.g., `int`) infer to ClassDef.
                            // Convert to type[...] so `x in (int, float)` can
                            // narrow x to type[int] | type[float].
                            Type::ClassDef(cls) => {
                                literal_types.push(Type::type_form(self.promote_silently(&cls)));
                            }
                            // Already-wrapped type[X] expressions pass through.
                            Type::Type(box Type::ClassType(_)) => {
                                literal_types.push(expr_ty);
                            }
                            _ => {
                                return ty.clone();
                            }
                        }
                    }
                    return self.intersect(ty, &self.unions(literal_types));
                }

                // Check if the right operand is a TypedDict.
                // If so, we can narrow the left operand to the union of the TypedDict's keys.
                let right_ty = self.expr_infer(v, errors);
                if let Type::TypedDict(typed_dict) = &right_ty {
                    let fields = self.typed_dict_fields(typed_dict);
                    if fields.is_empty() {
                        // Empty TypedDict - the `in` check is always false
                        return self.heap.mk_never();
                    }
                    let key_types: Vec<Type> = fields
                        .keys()
                        .map(|name| Lit::Str(name.as_str().into()).to_implicit_type())
                        .collect();
                    return self.intersect(ty, &self.unions(key_types));
                }

                ty.clone()
            }
            AtomicNarrowOp::NotIn(v) => {
                // First, check for List, Tuple, and Set literal expressions (syntactic check,
                // avoids type inference on the container itself)
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                if let Some(exprs) = exprs {
                    // Bail out if any element is a starred expression (e.g., `x not in [*y, 1]`).
                    // We can't know all values at compile time when unpacking occurs.
                    if exprs.iter().any(|e| matches!(e, Expr::Starred(_))) {
                        return ty.clone();
                    }
                    let mut literal_types = Vec::new();
                    for expr in exprs {
                        let expr_ty = self.expr_infer(&expr, errors);
                        match expr_ty {
                            Type::Literal(_) | Type::None => {
                                literal_types.push(expr_ty);
                            }
                            // Accept class objects so they don't trigger the
                            // bail-out below — this allows mixed containers
                            // like `(int, None)` to still narrow the non-class
                            // elements. Class objects themselves are not
                            // subtracted in the `not in` case (see comment in
                            // distribute_over_union below).
                            Type::ClassDef(cls) => {
                                literal_types.push(Type::type_form(self.promote_silently(&cls)));
                            }
                            Type::Type(box Type::ClassType(_)) => {
                                literal_types.push(expr_ty);
                            }
                            _ => {
                                return ty.clone();
                            }
                        }
                    }
                    return self.distribute_over_union(ty, |t| {
                        let mut result = t.clone();
                        for right in &literal_types {
                            match (t, right) {
                                (_, _) if self.literal_equal(t, right) => {
                                    result = self.heap.mk_never();
                                }
                                // We intentionally do NOT subtract class objects
                                // (type[X]) here. `x not in (int, float)` does
                                // not imply x is not type[int], because x could
                                // be type[MyInt] (a subclass of int) which
                                // satisfies type[int] but is not identity-equal
                                // to `int` at runtime.
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
                    });
                }

                // Check if the right operand is a TypedDict.
                // If so, we can narrow the left operand if it's exactly one of the TypedDict's keys.
                let right_ty = self.expr_infer(v, errors);
                if let Type::TypedDict(typed_dict) = &right_ty {
                    let fields = self.typed_dict_fields(typed_dict);
                    if fields.is_empty() {
                        // Empty TypedDict - the `not in` check is always true
                        return ty.clone();
                    }
                    let key_types: Vec<Type> = fields
                        .keys()
                        .map(|name| Lit::Str(name.as_str().into()).to_implicit_type())
                        .collect();
                    return self.distribute_over_union(ty, |t| {
                        for key_type in &key_types {
                            if self.literal_equal(t, key_type) {
                                return self.heap.mk_never();
                            }
                        }
                        t.clone()
                    });
                }

                ty.clone()
            }
            AtomicNarrowOp::Is(v) => {
                let right = self
                    .final_sentinel_literal_for_expr(v, errors)
                    .unwrap_or_else(|| self.expr_infer(v, errors));
                // Get our best approximation of ty & right.
                self.intersect(ty, &right)
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self
                    .final_sentinel_literal_for_expr(v, errors)
                    .unwrap_or_else(|| self.expr_infer(v, errors));
                // Get our best approximation of ty - right.
                self.distribute_over_union(ty, |t| {
                    // Only certain literal types can be compared by identity.
                    match (t, &right) {
                        (
                            _,
                            Type::None
                            | Type::Literal(box Literal {
                                value: Lit::Bool(_) | Lit::Enum(_) | Lit::Sentinel(_),
                                ..
                            }),
                        ) if self.literal_equal(t, &right) => self.heap.mk_never(),
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
            AtomicNarrowOp::IsInstance(v, source) => {
                let right = self.expr_infer(v, errors);
                // For patterns, validation happens here since there's no call site.
                // For calls, validation already happened in special_calls.rs.
                if matches!(source, NarrowSource::Pattern) {
                    let mut contains_subscript = false;
                    v.visit(&mut |e| {
                        if matches!(e, Expr::Subscript(_)) {
                            contains_subscript = true;
                        }
                    });
                    self.check_type_is_class_object(
                        right.clone(),
                        Some(ty.clone()),
                        contains_subscript,
                        v.range(),
                        &FunctionKind::IsInstance,
                        errors,
                        ErrorKind::InvalidPattern,
                    );
                }
                self.narrow_isinstance(ty, &right)
            }
            AtomicNarrowOp::IsNotInstance(v, _source) => self.narrow_is_not_instance(ty, v, errors),
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
            AtomicNarrowOp::IsNotSubclass(v) => self.narrow_is_not_subclass(ty, v, errors),
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
                    // This error is raised elsewhere, swallow here to avoid duplicate errors
                    let swallowed_errors = self.error_swallower();
                    let ret = self.call_infer(
                        *call_target,
                        &args,
                        &kws,
                        range,
                        &swallowed_errors,
                        None,
                        None,
                        None,
                    );
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
                    // This error is raised elsewhere, swallow here to avoid duplicate errors
                    let swallowed_errors = self.error_swallower();
                    let ret = self.call_infer(
                        *call_target,
                        &args,
                        &kws,
                        range,
                        &swallowed_errors,
                        None,
                        None,
                        None,
                    );
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
                                    self.heap.mk_never()
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
                    // This error is raised elsewhere, swallow here to avoid duplicate errors
                    let swallowed_errors = self.error_swallower();
                    let ret = self.call_infer(
                        *call_target,
                        &args,
                        &kws,
                        range,
                        &swallowed_errors,
                        None,
                        None,
                        None,
                    );
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
                        return self.heap.mk_never();
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
                        (_, _) if self.literal_equal(t, &right) => self.heap.mk_never(),
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

    /// Narrow for pattern matching
    fn is_type_for_pattern(&self, ty: &Type, is_type: impl Fn(&Type) -> bool) -> Type {
        self.distribute_over_union(ty, |t| {
            if is_type(t) {
                t.clone()
            } else {
                self.heap.mk_never()
            }
        })
    }

    /// Narrow to exclude a type for pattern matching
    fn is_not_type_for_pattern(&self, ty: &Type, is_type: impl Fn(&Type) -> bool) -> Type {
        // Note: Any and classes that extend Any must be preserved (not narrowed to Never)
        // since we can't know at static analysis time whether they're the pattern type or not
        self.distribute_over_union(ty, |t| {
            if self.behaves_like_any(t) {
                t.clone()
            } else if is_type(t) {
                self.heap.mk_never()
            } else {
                t.clone()
            }
        })
    }

    pub(crate) fn get_facet_chain_type(
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
                if self.is_dict_like(&base_ty) {
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
                if self.is_dict_like(&base_ty) {
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
                    type_info.with_narrow(&facets, self.heap.mk_any_implicit())
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Atomic(subject, AtomicNarrowOp::GetAttr(attr, default)) => {
                let suppress_errors =
                    ErrorCollector::new(errors.module().clone(), ErrorStyle::Never);
                let default_ty = default.as_ref().map_or_else(
                    || self.heap.mk_none(),
                    |v| self.expr_infer(v, &suppress_errors),
                );
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
                if attr_ty.is_error() {
                    type_info.with_narrow(&facets, self.heap.mk_any_implicit())
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

    fn is_flag_enum(&self, cls: &ClassType) -> bool {
        self.get_metadata_for_class(cls.class_object())
            .enum_metadata()
            .is_some_and(|meta| meta.is_flag)
    }

    /// Determines if a type should be checked for match exhaustiveness.
    /// We check exhaustiveness when the type has a finite, known set of possible values.
    pub(crate) fn should_check_exhaustiveness(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) | Type::SelfType(cls) => {
                // Final classes can't have subclasses, so they are exhaustible, with the exception
                // of Flag enums, whose members can be combined into new members via bitwise ops
                !self.is_flag_enum(cls) && self.is_final(cls)
                    // bool is effectively Literal[True] | Literal[False]
                    || cls.is_builtin("bool")
            }

            // Literal types have explicit values
            Type::Literal(_) => true,

            // None is a singleton
            Type::None => true,

            // Unions are exhaustible if all members are exhaustible types
            Type::Union(union) => {
                !union.members.is_empty()
                    && union
                        .members
                        .iter()
                        .all(|m| self.should_check_exhaustiveness(m))
            }

            _ => false,
        }
    }

    /// Formats the missing cases for a non-exhaustive match error message.
    /// Returns None if the remaining type can't be formatted nicely.
    fn format_missing_cases(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Literal(lit) => Some(format!("{}", lit.value)),
            Type::None => Some("None".to_owned()),
            Type::ClassType(cls) => {
                let display = self.for_display(self.heap.mk_class_type(cls.clone()));
                Some(format!("{}", display))
            }
            Type::Union(union) => {
                let formatted: Option<Vec<String>> = union
                    .members
                    .iter()
                    .map(|m| self.format_missing_cases(m))
                    .collect();
                formatted.map(|cases| cases.join(", "))
            }
            _ => None,
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
        if !self.should_check_exhaustiveness(&subject_ty) {
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
                let type_info = TypeInfo::of_ty(self.heap.mk_any_implicit());
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
        if let Some(missing_cases) = self.format_missing_cases(&remaining_ty) {
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
