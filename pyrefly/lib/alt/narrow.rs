/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use num_traits::ToPrimitive;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_types::class::Class;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::facet::FacetChain;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::facet::UnresolvedFacetChain;
use pyrefly_types::facet::UnresolvedFacetKind;
use pyrefly_types::simplify::intersect;
use pyrefly_types::simplify::simplify_tuples;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_info::JoinStyle;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Arguments;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Int;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallTargetLookup;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::types::instance::Instance;
use crate::binding::binding::Key;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::FacetOrigin;
use crate::binding::narrow::FacetSubject;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowSource;
use crate::binding::narrow::NarrowingSubject;
use crate::error::collector::ErrorCollector;
use crate::error::style::ErrorStyle;
use crate::types::callable::FunctionKind;
use crate::types::class::ClassType;
use crate::types::lit_int::LitInt;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::type_var::Restriction;
use crate::types::types::CalleeKind;
use crate::types::types::TParams;
use crate::types::types::Type;

/// Synthesize an `Expr` for an integer index, producing a `NumberLiteral` for
/// non-negative values and `UnaryOp(USub, NumberLiteral)` for negative values.
fn synthesize_int_slice(idx: i64) -> Expr {
    let fake_range = TextRange::empty(TextSize::from(0));
    let node_index = AtomicNodeIndex::default();
    if idx >= 0 {
        Expr::NumberLiteral(ExprNumberLiteral {
            node_index,
            range: fake_range,
            value: Number::Int(Int::from(idx as u64)),
        })
    } else {
        Expr::UnaryOp(ExprUnaryOp {
            node_index,
            range: fake_range,
            op: UnaryOp::USub,
            operand: Box::new(Expr::NumberLiteral(ExprNumberLiteral {
                node_index: AtomicNodeIndex::default(),
                range: fake_range,
                value: Number::Int(Int::from(idx.unsigned_abs())),
            })),
        })
    }
}

/// Append `facet` to an existing facet chain, or start a new one.
fn extend_facet_chain(resolved_chain: Option<&FacetChain>, facet: FacetKind) -> Vec1<FacetKind> {
    match resolved_chain {
        Some(chain) => {
            let mut facets = chain.facets().clone();
            facets.push(facet);
            facets
        }
        None => Vec1::new(facet),
    }
}

/// Beyond this size, don't try and narrow an enum.
///
/// If we have over 100 fields, the odds of the negative-type being useful is vanishingly small.
/// But the cost to create such a type (and then probably knock individual elements out of it)
/// is very high.
const NARROW_ENUM_LIMIT: usize = 100;

#[derive(Clone, Copy)]
enum IntersectFallback {
    Never,
    Right,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // Get the union of all members of an enum, minus the specified member
    fn subtract_enum_member(&self, instance: Instance, name: &Name) -> Type {
        if self
            .get_class_fields(instance.class)
            .is_some_and(|f| f.len() > NARROW_ENUM_LIMIT)
        {
            return instance.to_type(self.heap);
        }
        self.get_enum_from_class(instance.class)
            .expect("enum subtraction requires an enum class");
        // Enums derived from enum.Flag cannot be treated as a union of their members
        if self.has_superclass(instance.class, self.stdlib.enum_flag().class_object()) {
            return instance.to_type(self.heap);
        }
        self.unions(
            self.get_enum_members(instance.class)
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

    /// Return the most specific disjoint base for a type per PEP 800.
    ///
    /// `Self` and bounded type variables inherit the representative of their
    /// class/bound, since every value they can denote is constrained by it.
    /// Falls back to `object` for anything not explicitly disjoint.
    pub fn disjoint_base(&self, t: &Type) -> Class {
        match t {
            Type::ClassType(cls) | Type::SelfType(cls) => {
                let class = cls.class_object();
                self.get_disjoint_base_for_class(class)
                    .representative()
                    .cloned()
                    .unwrap_or_else(|| self.stdlib.object().class_object().dupe())
            }
            Type::Quantified(q) if q.is_type_var() => match q.restriction() {
                Restriction::Bound(bound) => self.disjoint_base(bound),
                Restriction::Constraints(_) | Restriction::Unrestricted => {
                    self.stdlib.object().class_object().dupe()
                }
            },
            Type::Tuple(_) => self.stdlib.tuple_object().clone(),
            _ => self.stdlib.object().class_object().clone(),
        }
    }

    fn is_final(&self, class: &Class) -> bool {
        self.get_metadata_for_class(class).is_final()
            || (self.get_enum_from_class(class).is_some()
                && !self.get_enum_members(class).is_empty())
    }

    fn intersect_impl(&self, left: &Type, right: &Type, fallback: IntersectFallback) -> Type {
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
        } else if left.is_typed_dict() != right.is_typed_dict() {
            // Use runtime representation of TypedDict for intersections w/ non-TypedDicts
            // Normally, TypedDict is not assignable to dict to avoid unsafe aliasing
            let runtime_class = self.heap.mk_class_type(self.stdlib.dict(
                self.heap.mk_class_type(self.stdlib.str().clone()),
                self.heap.mk_class_type(self.stdlib.object().clone()),
            ));
            if left.is_typed_dict() && self.is_subset_eq(&runtime_class, right) {
                left.clone()
            } else if right.is_typed_dict() && self.is_subset_eq(&runtime_class, left) {
                right.clone()
            } else {
                self.heap.mk_never()
            }
        } else if self.is_subset_eq(left, right) {
            left.clone()
        } else if let (Type::Type(left), Type::Type(right)) = (left, right) {
            let inner = self.intersect_with_fallback(left, right, fallback);
            if inner.is_never() {
                inner
            } else {
                self.heap.mk_type_of(inner)
            }
        } else if let (Type::ClassType(cls), Type::SelfType(self_cls))
        | (Type::SelfType(self_cls), Type::ClassType(cls)) = (left, right)
            && self.as_superclass(cls, self_cls.class_object()).as_ref() == Some(self_cls)
        {
            // ClassType(C) & SelfType(Parent) simplifies to SelfType(C) when C
            // is a subclass of Parent with a matching inherited instantiation.
            // Self[Parent] represents "Parent or any subclass", so narrowing it
            // to the subclass C keeps it a self-type anchored at C: attribute and
            // constructor lookups resolve through C, while the value stays
            // assignable back to Self[Parent] (all self-types are mutually
            // assignable). Collapsing to a plain ClassType(C) instead would drop
            // the self-ness and spuriously reject `return self`/`return cls()`
            // against a declared `-> Self`.
            // Producing a SelfType (rather than an unsimplified Intersect) also
            // avoids leaking Intersect types to downstream consumers that don't
            // handle them.
            self.heap.mk_self_type(cls.clone())
        } else if left.is_scalar() || right.is_scalar() {
            // The only inhabited intersections of literals are things like
            // `Literal[0] & Literal[0]` or `Literal[0] & int` that would have already been
            // intercepted by the is_subset_eq checks above. type(None) cannot be subclassed.
            self.heap.mk_never()
        } else {
            let fallback = match fallback {
                IntersectFallback::Never => return self.heap.mk_never(),
                IntersectFallback::Right if right.is_never() => return right.clone(),
                IntersectFallback::Right => right,
            };
            if let Type::ClassType(left_cls) = left
                && let Type::ClassType(right_cls) = right
                && (self.is_final(left_cls.class_object())
                    || self.is_final(right_cls.class_object()))
            {
                // The only way for `left & right` to exist is if it is an instance of a class that
                // multiply inherits from both `left` and `right`'s classes. But at least one of
                // the classes is final, so such a class does not exist.
                self.heap.mk_never()
            } else {
                let left_base = self.disjoint_base(left);
                let right_base = self.disjoint_base(right);
                if self.has_superclass(&left_base, &right_base)
                    || self.has_superclass(&right_base, &left_base)
                {
                    intersect(
                        vec![left.clone(), right.clone()],
                        fallback.clone(),
                        self.heap,
                    )
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
        fallback: IntersectFallback,
    ) -> Type {
        self.distribute_over_union(left, |l| {
            self.distribute_over_union(right, |r| self.intersect_impl(l, r, fallback))
        })
    }

    fn intersect(&self, left: &Type, right: &Type) -> Type {
        self.intersect_with_fallback(left, right, IntersectFallback::Never)
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
            if !left.is_any() && !right.is_any() && left.is_typed_dict() && !right.is_typed_dict() {
                // Use runtime representation of TypedDict for subtraction w/ non-TypedDicts
                // Normally, TypedDict is not assignable to dict to avoid unsafe aliasing
                let runtime_class = self.heap.mk_class_type(self.stdlib.dict(
                    self.heap.mk_class_type(self.stdlib.str().clone()),
                    self.heap.mk_class_type(self.stdlib.object().clone()),
                ));
                if self.is_subset_eq(&runtime_class, right) {
                    self.heap.mk_never()
                } else {
                    left.clone()
                }
            } else if !left.is_any() && self.is_subset_eq(left, right) {
                // is_any check because `Any <: int` is always true, but `Any - int` shouldn't produce Never.
                self.heap.mk_never()
            } else {
                left.clone()
            }
        })
    }

    fn enum_instance<'b, 'c>(left: &'b Type, right: &'c Type) -> Option<(Instance<'b>, &'c Name)> {
        let left = match left {
            Type::ClassType(cls) => Instance::of_class(cls),
            Type::SelfType(cls) => Instance::of_self_type(cls),
            _ => return None,
        };
        if let Type::Literal(right) = right
            && let Lit::Enum(right) = &right.value
            && left.class == right.class.class_object()
            && left.targs == right.class.targs()
        {
            Some((left, &right.member))
        } else {
            None
        }
    }

    /// Narrow a type by removing values identity-equal to `right` (`is not` semantics).
    fn narrow_is_not(&self, ty: &Type, right: &Type) -> Type {
        self.distribute_over_union(ty, |t| match (t, right) {
            (_, Type::None | Type::Ellipsis) if self.literal_equal(t, right) => {
                self.heap.mk_never()
            }
            (_, Type::Literal(f))
                if matches!(f.value, Lit::Bool(_) | Lit::Enum(_))
                    && self.literal_equal(t, right) =>
            {
                self.heap.mk_never()
            }
            (Type::Sentinel(s1), Type::Sentinel(s2)) if s1 == s2 => self.heap.mk_never(),
            (Type::ClassType(cls), Type::Literal(lit))
                if cls.is_builtin("bool")
                    && let Lit::Bool(b) = &lit.value =>
            {
                Lit::Bool(!b).to_implicit_type()
            }
            (left, right) if let Some((instance, name)) = Self::enum_instance(left, right) => {
                self.subtract_enum_member(instance, name)
            }
            _ => t.clone(),
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

    /// Extract element expressions from a literal container (list, tuple, set)
    /// or a builtin container constructor call wrapping one (list/tuple/set/frozenset).
    /// Returns `None` if the expression is not a container whose elements can
    /// be statically enumerated.
    fn literal_membership_exprs(&self, expr: &Expr, errors: &ErrorCollector) -> Option<Vec<Expr>> {
        match expr {
            Expr::List(list) => Some(list.elts.clone()),
            Expr::Tuple(tuple) => Some(tuple.elts.clone()),
            Expr::Set(set) => Some(set.elts.clone()),
            Expr::Call(call) => {
                const CONTAINER_NAMES: &[&str] = &["list", "tuple", "set", "frozenset"];
                // Cheap syntactic pre-check: only proceed when the callee looks like
                // it *could* be a builtin container constructor (bare name or
                // `builtins.<name>`). This avoids an expensive `expr_infer` call for
                // the common case of arbitrary function calls.
                let callee_name = match &*call.func {
                    Expr::Name(name) => name.id.as_str(),
                    Expr::Attribute(attr) => attr.attr.as_str(),
                    _ => return None,
                };
                if !CONTAINER_NAMES.contains(&callee_name) {
                    return None;
                }
                if !call.arguments.keywords.is_empty() {
                    return None;
                }
                // Confirm via type inference that it's actually the builtin, to guard
                // against shadowing or unrelated attributes with the same name.
                let is_builtin_container = match self.expr_infer(&call.func, errors) {
                    Type::ClassDef(cls) => CONTAINER_NAMES.iter().any(|n| cls.is_builtin(n)),
                    _ => false,
                };
                if !is_builtin_container {
                    return None;
                }
                match &*call.arguments.args {
                    [] => Some(Vec::new()),
                    [expr] => self.literal_membership_exprs(expr, errors),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn tuple_membership_type(
        &self,
        tuple: &Tuple,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let elements = match tuple {
            Tuple::Concrete(elts) => elts.clone(),
            Tuple::Unbounded(elt) => vec![(**elt).clone()],
            Tuple::Unpacked(unpacked) => {
                let (prefix, middle, suffix) = &**unpacked;
                let mut elements = prefix.clone();
                let middle = if let Type::Var(_) = middle {
                    self.force_for_narrowing(middle, range, errors)
                } else {
                    middle.clone()
                };
                match middle {
                    Type::Tuple(tuple) => {
                        elements.push(self.tuple_membership_type(&tuple, range, errors)?)
                    }
                    Type::TypeVarTuple(_) | Type::Quantified(_) | Type::Unpack(_) => return None,
                    _ => elements.push(middle),
                }
                elements.extend(suffix.clone());
                elements
            }
        };
        if elements.iter().any(|ty| self.behaves_like_any(ty)) {
            None
        } else if elements.is_empty() {
            Some(self.heap.mk_never())
        } else {
            Some(self.unions(elements))
        }
    }

    /// Unwrap a class-info target to the instance type used for narrowing. When the right-hand
    /// side is `tuple` and the left-hand side is a heterogeneous tuple type, creates a TypeVarTuple
    /// for precise narrowing. Otherwise falls back to standard class object unwrapping.
    fn unwrap_class_info_target(&self, left: &Type, right: &Type) -> Option<(TParams, Type)> {
        let right_is_tuple = match right {
            Type::Type(f) if matches!(&**f, Type::Tuple(_)) => true,
            Type::ClassDef(cls) => cls.is_builtin("tuple"),
            _ => false,
        };
        let narrow_heterogeneous_tuple = right_is_tuple
            && match left {
                Type::Tuple(_) => true,
                Type::ClassType(cls) => self.as_tuple(cls).is_some(),
                _ => false,
            };
        if narrow_heterogeneous_tuple {
            Some(self.instantiate_type_var_tuple())
        } else if matches!(right, Type::ClassDef(c) if c == self.stdlib.builtins_type().class_object())
        {
            // `isinstance(x, type)` narrows `x` to its class-object part. When `x` is already a
            // type-expression value, that part is `type[inner]`: `type[int]` stays precise and
            // gradual `type[Any]` stays gradual.
            match left {
                Type::Type(_) => Some((TParams::empty(), left.clone())),
                Type::TypeForm(inner) => {
                    Some((TParams::empty(), self.heap.mk_type_of((**inner).clone())))
                }
                _ => self.unwrap_class_object_silently(right),
            }
        } else {
            self.unwrap_class_object_silently(right)
        }
    }

    /// Run `f` with the freshened instance type produced by unwrapping `right` as class info.
    fn with_fresh_class_info_target(
        &self,
        left: &Type,
        right: &Type,
        f: impl FnOnce(Type) -> Type,
    ) -> Option<Type> {
        let (tparams, right) = self.unwrap_class_info_target(left, right)?;
        let (vs, right) = self
            .solver()
            .fresh_quantified(&tparams, right, self.uniques);
        let result = f(right);
        // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
        // * If `left` is an invalid specialization, the error has already been reported at its definition site.
        // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
        let _specialization_errors = self.finish_quantified(vs, false);
        Some(result)
    }

    /// Strip top-level `Any` from `ty` for `isinstance` narrowing.
    /// Justified because `isinstance` gives definite runtime evidence that lets us
    /// eliminate the uncertainty of a top-level `Any`.
    /// (Note: this sacrifices the gradual guarantee, which Pyrefly treats as a
    /// guiding principle rather than a hard goal.)
    ///
    /// Returns `(non_any_part, had_any)`. `non_any_part` is `ty` with all top-level
    /// `Any`s stripped (or `None` if `ty` is purely `Any`). `had_any` is true when
    /// anything was stripped. When `false`, the caller can just do a naive intersection;
    /// when `true`, it must also union in the isinstance target type, since an
    /// `Any`-typed value may be the target at runtime.
    fn strip_any_for_narrowing(
        &self,
        ty: &Type,
        aliases: &mut SmallSet<TypeAliasData>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> (Option<Type>, bool) {
        match ty {
            Type::Any(_) => (None, true),
            Type::Union(union) => {
                let mut definites = Vec::new();
                let mut has_dynamic_alternative = false;
                for member in &union.members {
                    let (definite, has_dynamic) =
                        self.strip_any_for_narrowing(member, aliases, range, errors);
                    if let Some(definite) = definite {
                        definites.push(definite);
                    }
                    has_dynamic_alternative |= has_dynamic;
                }
                let definite = if definites.is_empty() {
                    None
                } else {
                    Some(self.unions(definites))
                };
                (definite, has_dynamic_alternative)
            }
            Type::Intersect(intersection) => {
                let (parts, fallback) = &**intersection;
                let parts = parts
                    .iter()
                    .filter_map(|part| self.strip_any_for_narrowing(part, aliases, range, errors).0)
                    .collect::<Vec<_>>();
                if parts.is_empty() {
                    (None, true)
                } else {
                    (Some(intersect(parts, fallback.clone(), self.heap)), false)
                }
            }
            Type::UntypedAlias(alias) => {
                if !aliases.insert((**alias).clone()) {
                    return (Some(ty.clone()), false);
                }
                let expanded = self.untype_alias(alias);
                let result = self.strip_any_for_narrowing(&expanded, aliases, range, errors);
                aliases.shift_remove(&**alias);
                result
            }
            Type::Var(_) => {
                let forced = self.force_for_narrowing(ty, range, errors);
                self.strip_any_for_narrowing(&forced, aliases, range, errors)
            }
            _ => (Some(ty.clone()), false),
        }
    }

    fn narrow_isinstance_from_definite(&self, left: &Type, right: &Type) -> Type {
        self.distribute_over_union(left, |l| {
            self.with_fresh_class_info_target(l, right, |right| {
                if right.is_any() {
                    // NOTE(grievejia): The most precise refinement would be `left`:
                    // `isinstance(x, Any)` provides no concrete evidence about the type
                    // of `x`, so keeping the original type is sound. In practice, that is
                    // currently too strict for some primer projects. Refining to `Any` is
                    // a gradual-typing compromise; we can revisit `left` in strict mode.
                    right.clone()
                } else {
                    // TODO: falling back to Never when the lhs is a union is a hack to get
                    // reasonable behavior in cases like this:
                    //     def f(x: int | list[int]):
                    //         if isinstance(x, Iterable):
                    //             reveal_type(x)
                    // We want to narrow x to just `list[int]`, rather than `(int & Iterable[Unknown]) | list[int]`
                    let fallback = if left.is_union() {
                        IntersectFallback::Never
                    } else {
                        IntersectFallback::Right
                    };
                    self.intersect_with_fallback(l, &right, fallback)
                }
            })
            .unwrap_or_else(|| l.clone())
        })
    }

    fn narrow_isinstance(
        &self,
        left: &Type,
        right: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let (definite, has_dynamic_alternative) =
            self.strip_any_for_narrowing(left, &mut SmallSet::new(), range, errors);
        let mut res = Vec::new();
        for right in self.as_class_info(right.clone()) {
            if let Some(definite) = &definite {
                res.push(self.narrow_isinstance_from_definite(definite, &right));
            }
            if definite.is_none() || has_dynamic_alternative {
                res.push(
                    self.with_fresh_class_info_target(left, &right, |right| right)
                        .unwrap_or_else(|| left.clone()),
                );
            }
        }
        self.unions(res)
    }

    fn narrow_typeis_target_from_definite(&self, left: &Type, right: &Type) -> Type {
        if right.is_any() {
            intersect(vec![left.clone(), right.clone()], left.clone(), self.heap)
        } else {
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
            let fallback = if left.is_union() {
                IntersectFallback::Never
            } else {
                IntersectFallback::Right
            };
            self.intersect_with_fallback(left, right, fallback)
        }
    }

    fn narrow_typeis(
        &self,
        left: &Type,
        right: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let (definite, has_dynamic_alternative) =
            self.strip_any_for_narrowing(left, &mut SmallSet::new(), range, errors);
        self.distribute_over_union(right, |right| {
            let mut res = Vec::new();
            if let Some(definite) = &definite {
                res.push(self.narrow_typeis_target_from_definite(definite, right));
            }
            if definite.is_none() || has_dynamic_alternative {
                res.push(right.clone());
            }
            self.unions(res)
        })
    }

    fn narrow_is_not_instance(
        &self,
        left: &Type,
        right_expr: &Expr,
        source: NarrowSource,
        errors: &ErrorCollector,
    ) -> Type {
        let force_allow_negative = matches!(source, NarrowSource::Pattern);
        let class_infos = self.expr_as_class_info(right_expr, errors);
        // Subtract each class info from each union member of `left` in turn,
        // rather than computing per-target results and intersecting them.
        // Intersecting `subtract(L, A)` with `subtract(L, B)` reintroduces
        // any union member that overlaps with both A and B (e.g. `Iterable[Any]`
        // overlaps with both `str` and `bytes`), so `isinstance(x, (str, bytes))`
        // would leave the negative branch unchanged. Sequential per-element
        // subtraction avoids that.
        self.distribute_over_union(left, |l| {
            let mut result = l.clone();
            for (right, allows_negative_narrow) in &class_infos {
                let allows_negative_narrow = *allows_negative_narrow || force_allow_negative;
                if !allows_negative_narrow {
                    continue;
                }
                if let Some((tparams, right)) = self.unwrap_class_info_target(&result, right) {
                    let (vs, right) = self
                        .solver()
                        .fresh_quantified(&tparams, right, self.uniques);
                    // For TypeVars, subtract from the concrete constraints
                    // so that e.g. isinstance(x, (int, float)) with T(int, str, float)
                    // narrows the else branch to str instead of leaving it as T.
                    result = if let Type::Quantified(q) = &result {
                        let concrete = q.upper_bound(self.stdlib, self.heap);
                        let subtraction = self.subtract(&concrete, &right);
                        self.intersect_with_fallback(
                            &result,
                            &subtraction,
                            IntersectFallback::Right,
                        )
                    } else {
                        self.subtract(&result, &right)
                    };
                    // These are safe to ignore, as the only possible specialization errors are handled elsewhere:
                    // * If `left` is an invalid specialization, the error has already been reported at its definition site.
                    // * Unsafe runtime protocol overlaps are separately checked for in special_calls.rs.
                    let _specialization_errors = self.finish_quantified(vs, false);
                }
            }
            result
        })
    }

    /// Narrow `type(X) != Y`. We can only do negative narrowing if Y is final,
    /// because otherwise X could still be a subclass of Y.
    fn narrow_type_not_eq(&self, left: &Type, right_expr: &Expr, errors: &ErrorCollector) -> Type {
        let right = self.expr_infer(right_expr, errors);
        // Only narrow if the RHS is a final class type (e.g., `type(x) != bool`)
        if let Type::ClassDef(cls) = &right
            && self.is_final(cls)
        {
            self.distribute_over_union(left, |l| {
                if let Some((tparams, unwrapped)) = self.unwrap_class_info_target(l, &right) {
                    let (vs, unwrapped) =
                        self.solver()
                            .fresh_quantified(&tparams, unwrapped, self.uniques);
                    let result = self.subtract(l, &unwrapped);
                    let _specialization_errors = self.finish_quantified(vs, false);
                    result
                } else {
                    l.clone()
                }
            })
        } else {
            left.clone()
        }
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
                    if let Type::Type(f) = &t
                        && let Type::ClassType(cls) = &**f
                    {
                        // If `C` is not final, `type[C]` may be a subclass of `C`,
                        // making negative narrowing unsafe.
                        let allows_negative_narrow = me.is_final(cls.class_object());
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
            self.heap.mk_type_of(instance_result)
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
                self.with_fresh_class_info_target(&left_untyped, &right, |right| {
                    self.issubclass_result(
                        self.intersect_with_fallback(
                            &left_untyped,
                            &right,
                            IntersectFallback::Right,
                        ),
                        left,
                    )
                })
                .unwrap_or_else(|| left.clone())
            } else {
                left.clone()
            }
        };

        for right in self.as_class_info(right.clone()) {
            if self.unwrap_class_object_silently(&right).is_some() {
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
                    let intersection =
                        narrow(&q.upper_bound(self.stdlib, self.heap), right.clone());
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
                    res.push(narrow(&self.unions(nonquantifieds), right.clone()));
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
                let _specialization_errors = self.finish_quantified(vs, false);
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
            Type::Tuple(Tuple::Unpacked(f)) if f.0.len() + f.2.len() >= len => self.heap.mk_never(),
            Type::ClassType(class) if let Some(tuple) = self.as_tuple(class) => match tuple {
                Tuple::Concrete(elts) if elts.len() >= len => self.heap.mk_never(),
                Tuple::Unpacked(f) if f.0.len() + f.2.len() >= len => self.heap.mk_never(),
                _ => ty.clone(),
            },
            _ => ty.clone(),
        })
    }

    /// Keep the union members consistent with `key` being present or absent.
    fn narrow_key_membership(&self, ty: &Type, key: &Name, present: bool) -> Type {
        self.distribute_over_union(ty, |member| {
            let Type::TypedDict(typed_dict) = member else {
                return member.clone();
            };
            if present {
                match self.typed_dict_field(typed_dict, key) {
                    Some(_) => member.clone(),
                    None => match self.typed_dict_extra_items(typed_dict) {
                        ExtraItems::Closed => self.heap.mk_never(),
                        ExtraItems::Default | ExtraItems::Extra(_) => member.clone(),
                    },
                }
            } else {
                match self.typed_dict_field(typed_dict, key) {
                    Some(field) if field.required => self.heap.mk_never(),
                    Some(_) | None => member.clone(),
                }
            }
        })
    }

    /// Unlike `is_dict_like`, this treats unions member-wise.
    fn has_dict_like_member(&self, ty: &Type) -> bool {
        match ty {
            Type::Union(union) => union
                .members
                .iter()
                .any(|member| self.has_dict_like_member(member)),
            _ => self.is_dict_like(ty),
        }
    }

    /// Narrow a union by keeping only members whose facet is identity-compatible with `right`.
    fn narrow_facet_is(
        &self,
        base: &Type,
        right: &Type,
        facet: &FacetKind,
        range: TextRange,
    ) -> Type {
        self.distribute_over_union(base, |t| {
            let base_info = TypeInfo::of_ty(t.clone());
            let facet_ty = self.get_facet_chain_type(
                &base_info,
                &FacetChain::new(Vec1::new(facet.clone())),
                range,
            );
            match right {
                Type::None | Type::Ellipsis => {
                    if self.is_subset_eq(right, &facet_ty) {
                        t.clone()
                    } else {
                        self.heap.mk_never()
                    }
                }
                Type::Literal(f) if matches!(f.value, Lit::Bool(_) | Lit::Enum(_)) => {
                    if self.is_subset_eq(right, &facet_ty) {
                        t.clone()
                    } else {
                        self.heap.mk_never()
                    }
                }
                _ => t.clone(),
            }
        })
    }

    /// Narrow a union by removing members whose facet is identity-equal to `right`.
    fn narrow_facet_is_not(
        &self,
        base: &Type,
        right: &Type,
        facet: &FacetKind,
        range: TextRange,
    ) -> Type {
        self.distribute_over_union(base, |t| {
            let base_info = TypeInfo::of_ty(t.clone());
            let facet_ty = self.get_facet_chain_type(
                &base_info,
                &FacetChain::new(Vec1::new(facet.clone())),
                range,
            );
            let is_identity_literal = |ty: &Type| {
                matches!(ty, Type::None | Type::Ellipsis)
                    || matches!(ty, Type::Literal(f) if matches!(f.value, Lit::Bool(_) | Lit::Enum(_)))
            };
            if is_identity_literal(&facet_ty)
                && is_identity_literal(right)
                && self.literal_equal(right, &facet_ty)
            {
                self.heap.mk_never()
            } else {
                t.clone()
            }
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
        allow_never_collapse: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        // We narrow `X.__class__ == Y` the same way as `type(X) == Y`
        if let FacetKind::Attribute(attr) = facet
            && *attr == dunder::CLASS
        {
            match op {
                AtomicNarrowOp::Is(v) | AtomicNarrowOp::Eq(v) => {
                    let right = self.expr_infer(v, errors);
                    return Some(self.narrow_isinstance(base, &right, range, errors));
                }
                AtomicNarrowOp::IsNot(v) | AtomicNarrowOp::NotEq(v) => {
                    return Some(self.narrow_type_not_eq(base, v, errors));
                }
                _ => {}
            }
        }
        match op {
            AtomicNarrowOp::Is(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.narrow_facet_is(base, &right, facet, range))
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self.expr_infer(v, errors);
                Some(self.narrow_facet_is_not(base, &right, facet, range))
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
                        Type::None | Type::Ellipsis | Type::Literal(_) | Type::Sentinel(_) => {
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
                        (
                            Type::None | Type::Ellipsis | Type::Literal(_) | Type::Sentinel(_),
                            Type::None | Type::Ellipsis | Type::Literal(_) | Type::Sentinel(_),
                        ) if self.literal_equal(&right, &facet_ty) => self.heap.mk_never(),
                        _ => t.clone(),
                    }
                }))
            }
            AtomicNarrowOp::In(v) | AtomicNarrowOp::NotIn(v)
                if self.literal_membership_exprs(v, errors).is_some() =>
            {
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    let narrowed_facet = self.atomic_narrow(&facet_ty, op, range, errors);
                    if narrowed_facet.is_never() {
                        self.heap.mk_never()
                    } else {
                        t.clone()
                    }
                }))
            }
            // If `allow_never_collapse` is not set, we only filter members of a union
            // to avoid inferring `Never` excessively
            AtomicNarrowOp::IsInstance(_, _) | AtomicNarrowOp::IsNotInstance(_, _)
                if base.is_union() || allow_never_collapse =>
            {
                let suppress_errors = self.error_swallower();
                Some(self.distribute_over_union(base, |t| {
                    let base_info = TypeInfo::of_ty(t.clone());
                    let facet_ty = self.get_facet_chain_type(
                        &base_info,
                        &FacetChain::new(Vec1::new(facet.clone())),
                        range,
                    );
                    let narrowed_facet = self.atomic_narrow(&facet_ty, op, range, &suppress_errors);
                    if narrowed_facet.is_never() {
                        self.heap.mk_never()
                    } else {
                        t.clone()
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
            Tuple::Unpacked(f) if f.0.len() + f.2.len() > len => self.heap.mk_never(),
            Tuple::Unpacked(f) if f.0.len() + f.2.len() == len => {
                let (prefix, _, suffix) = &**f;
                self.heap
                    .mk_concrete_tuple(prefix.iter().cloned().chain(suffix.clone()).collect())
            }
            Tuple::Unpacked(f)
                if let Type::Tuple(Tuple::Unbounded(middle)) = &f.1
                    && f.0.len() + f.2.len() < len =>
            {
                let (prefix, _, suffix) = &**f;
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
            Tuple::Unpacked(f) if let Type::Var(_) = &f.1 => {
                let (prefix, middle_var, suffix) = &**f;
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
            Tuple::Unpacked(f) if let Type::Var(_) = &f.1 => {
                let (prefix, middle_var, suffix) = &**f;
                let forced_middle = self.force_for_narrowing(middle_var, range, errors);
                let new_tuple =
                    Tuple::Unpacked(Box::new((prefix.clone(), forced_middle, suffix.clone())));
                self.tuple_len_not_eq(&simplify_tuples(new_tuple, self.heap), len, range, errors)
            }
            _ => self.heap.mk_tuple(tuple.clone()),
        }
    }

    pub(crate) fn atomic_narrow(
        &self,
        ty: &Type,
        op: &AtomicNarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match op {
            AtomicNarrowOp::Placeholder => ty.clone(),
            AtomicNarrowOp::ClassCoverageGate(_) => ty.clone(),
            AtomicNarrowOp::ClassCoverageGateNeg(keys) => {
                // Subtract the class only when every positional slot's sub-pattern exhausts its
                // matched slot, i.e. all slot-coverage keys resolved to `Never`.
                if !keys.is_empty() && keys.iter().all(|key| self.get_idx(*key).ty().is_never()) {
                    self.heap.mk_never()
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::LenEq(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
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
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
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
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.narrow_length_greater(ty, len)
            }
            AtomicNarrowOp::LenGte(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
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
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
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
                let Type::Literal(f) = &right else {
                    return ty.clone();
                };
                let Lit::Int(lit) = &f.value else {
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
                // First, check for literal containers. We also unwrap builtin
                // container constructor calls (list/tuple/set/frozenset) when
                // their argument is itself a literal container.
                if let Some(exprs) = self.literal_membership_exprs(v, errors) {
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
                                literal_types.push(Type::type_of(self.promote_silently(&cls)));
                            }
                            // Already-wrapped type[X] expressions pass through.
                            Type::Type(ref f) if matches!(&**f, Type::ClassType(_)) => {
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
                if let Type::Tuple(tuple) = &right_ty
                    && let Some(member_ty) = self.tuple_membership_type(tuple, range, errors)
                {
                    return self.intersect(ty, &member_ty);
                }
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

                // Check if the right operand is a mapping (e.g. dict[str, int]).
                // If so, we can narrow the left operand to the mapping's key type.
                if !self.behaves_like_any(&right_ty)
                    && let Some((key_ty, _)) = self.unwrap_mapping(&right_ty)
                {
                    self.intersect(ty, &key_ty)
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::NotIn(v) => {
                // First, check for literal containers. We also unwrap builtin
                // container constructor calls (list/tuple/set/frozenset) when
                // their argument is itself a literal container.
                if let Some(exprs) = self.literal_membership_exprs(v, errors) {
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
                                literal_types.push(Type::type_of(self.promote_silently(&cls)));
                            }
                            Type::Type(ref f) if matches!(&**f, Type::ClassType(_)) => {
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
                                (left, right)
                                    if let Some((instance, name)) =
                                        Self::enum_instance(left, right) =>
                                {
                                    result = self.subtract_enum_member(instance, name);
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
                let right = self.expr_infer(v, errors);
                // Get our best approximation of ty & right.
                self.intersect(ty, &right)
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_is_not(ty, &right)
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
                        Some(ty.clone()),
                        right.clone(),
                        contains_subscript,
                        v.range(),
                        &FunctionKind::IsInstance,
                        errors,
                        ErrorKind::InvalidPattern,
                    );
                }
                self.narrow_isinstance(ty, &right, v.range(), errors)
            }
            AtomicNarrowOp::IsNotInstance(v, source) => {
                self.narrow_is_not_instance(ty, v, *source, errors)
            }
            AtomicNarrowOp::TypeEq(v) => {
                // If type(X) == Y then X has to be exactly Y, not a subclass of Y
                // We can't model that, so we narrow it exactly like isinstance(X, Y)
                let right = self.expr_infer(v, errors);
                self.narrow_isinstance(ty, &right, v.range(), errors)
            }
            // If type(X) != Y, X can still be a subclass of Y so we can't do negative refinement
            // unless Y is final, in which case X cannot be a subclass of Y
            AtomicNarrowOp::TypeNotEq(v) => self.narrow_type_not_eq(ty, v, errors),
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
                        return self.narrow_typeis(ty, &t, range, errors);
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
                if matches!(
                    right,
                    Type::Literal(_) | Type::None | Type::Ellipsis | Type::Sentinel(_)
                ) {
                    self.intersect(ty, &right)
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::NotEq(v) => {
                let right = self.expr_infer(v, errors);
                if matches!(
                    right,
                    Type::Literal(_) | Type::None | Type::Ellipsis | Type::Sentinel(_)
                ) {
                    self.distribute_over_union(ty, |t| match (t, &right) {
                        (_, _) if self.literal_equal(t, &right) => self.heap.mk_never(),
                        (Type::ClassType(cls), Type::Literal(lit))
                            if cls.is_builtin("bool")
                                && let Lit::Bool(b) = &lit.value =>
                        {
                            Lit::Bool(!b).to_implicit_type()
                        }
                        (left, right)
                            if let Some((instance, name)) = Self::enum_instance(left, right) =>
                        {
                            self.subtract_enum_member(instance, name)
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
            FacetKind::Attribute(first_attr_name) => {
                // Use a synthesized fake range for the attribute lookup itself to avoid
                // overwriting typing traces (e.g. property getters, overload callees) keyed
                // on the real `range`, which typically points at unrelated code such as the
                // RHS of an equality narrow. The real `range` is still threaded through to
                // recursive facet-chain narrowing so that errors there remain locatable.
                let fake_range = TextRange::default();
                match remaining_facets.split_first() {
                    None => match base.type_at_facet(first_facet) {
                        Some(ty) => self.force_for_narrowing(ty, range, errors),
                        None => {
                            self.narrowable_for_attr(base.ty(), first_attr_name, fake_range, errors)
                        }
                    },
                    Some((next_name, remaining_facets)) => {
                        let base = self.attr_infer(base, first_attr_name, fake_range, errors, None);
                        self.narrowable_for_facet_chain(
                            &base,
                            next_name,
                            remaining_facets,
                            range,
                            errors,
                        )
                    }
                }
            }
            FacetKind::Index(idx) => {
                // We synthesize a slice expression for the subscript here.
                // For negative indices, we must produce `UnaryOp(USub, NumberLiteral(abs))`
                // to match what the parser generates for e.g. `xs[-1]`.
                // Use a synthesized fake range to avoid overwriting typing traces.
                let synthesized_slice = synthesize_int_slice(*idx);
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
                        None => self
                            .subscript_infer(base, &synthesized_slice, range, errors)
                            .into_ty(),
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
            NarrowOp::Atomic(
                subject,
                key_op @ (AtomicNarrowOp::HasKey(key) | AtomicNarrowOp::NotHasKey(key)),
            ) => {
                let key_present = matches!(key_op, AtomicNarrowOp::HasKey(_));
                let resolved_chain = subject
                    .as_ref()
                    .and_then(|s| self.resolve_facet_chain(s.chain.clone()));
                let base_ty = match (&subject, &resolved_chain) {
                    (Some(_), Some(chain)) => self.get_facet_chain_type(type_info, chain, range),
                    (Some(_), None) => return type_info.clone(),
                    (None, _) => self.force_for_narrowing(type_info.ty(), range, errors),
                };
                let narrowed_base = self.narrow_key_membership(&base_ty, key, key_present);
                let has_dict_member = self.has_dict_like_member(&narrowed_base);
                let mut narrowed = match &resolved_chain {
                    Some(chain) => type_info.with_narrow(chain.facets(), narrowed_base),
                    None => type_info.clone().with_ty(narrowed_base),
                };
                if has_dict_member {
                    let facet = FacetKind::Key(key.to_string());
                    let facets = extend_facet_chain(resolved_chain.as_ref(), facet);
                    if key_present {
                        // `key in x` records presence only; subscript inference computes the value type.
                        narrowed.record_present(&facets);
                    } else {
                        // `key not in x`: invalidate any narrow recorded for the key.
                        narrowed.update_for_assignment(&facets, None);
                    }
                }
                narrowed
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
                // Narrow the attribute facet only if it doesn't exist on all members.
                // The facet type is the union of attribute types from members that DO
                // have the attribute, plus `Any` for members that don't. Preserving
                // specific attribute types (rather than using a blanket `Any`) ensures
                // fixpoint convergence when `hasattr` is used in loops with reassignment.
                if let Some(narrow_ty) = self.hasattr_narrow_type(&base_ty, attr, range, errors) {
                    let facet = FacetKind::Attribute(attr.clone());
                    let facets = extend_facet_chain(resolved_chain.as_ref(), facet);
                    type_info.with_narrow(&facets, narrow_ty)
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
                let facet = FacetKind::Attribute(attr.clone());
                let facets = extend_facet_chain(resolved_chain.as_ref(), facet);
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
                                facet_subject.allow_never_collapse,
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
                                facet_subject.allow_never_collapse,
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
        self.get_enum_from_class(cls.class_object()).is_some()
            && self.has_superclass(cls.class_object(), self.stdlib.enum_flag().class_object())
    }

    pub(crate) fn with_type_for_exhaustiveness_check(&self, info: Arc<TypeInfo>) -> TypeInfo {
        info.arc_clone().map_ty(|mut ty| {
            self.expand_mut(&mut ty);
            match ty {
                Type::SelfType(cls) => Type::ClassType(cls),
                ty => ty,
            }
        })
    }

    /// Determines if a type should be checked for match exhaustiveness.
    /// We check exhaustiveness when the type has a finite, known set of possible values.
    fn should_check_exhaustiveness(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) => {
                // Final classes can't have subclasses, so they are exhaustible, with the exception
                // of Flag enums, whose members can be combined into new members via bitwise ops
                !self.is_flag_enum(cls) && self.is_final(cls.class_object())
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
        narrowing_subject: Option<&NarrowingSubject>,
        narrow_ops_for_fall_through: &(Box<NarrowOp>, TextRange),
        subject_range: &TextRange,
        show_subject_expr: bool,
        errors: &ErrorCollector,
    ) {
        let (op, narrow_range) = narrow_ops_for_fall_through;
        let subject_info = self.with_type_for_exhaustiveness_check(self.get_idx(*subject_idx));
        // We only check match exhaustiveness if the subject is an enum or a union of enum literals
        if !self.should_check_exhaustiveness(subject_info.ty()) {
            return;
        }
        let ignore_errors = self.error_swallower();
        // Get the narrowed type of the match subject when none of the cases match
        let mut remaining_ty = match narrowing_subject {
            None | Some(NarrowingSubject::Name(_)) => self
                .narrow(&subject_info, op.as_ref(), *narrow_range, &ignore_errors)
                .ty()
                .clone(),
            Some(NarrowingSubject::Facets(_, facets)) => {
                let Some(resolved_chain) = self.resolve_facet_chain(facets.chain.clone()) else {
                    return;
                };
                // If the narrowing subject is the facet of some variable like `x.foo`,
                // We need to make a `TypeInfo` rooted at `x` using the type of `x.foo`
                let type_info = TypeInfo::of_ty(self.heap.mk_any_implicit());
                let narrowing_subject_info =
                    type_info.with_narrow(resolved_chain.facets(), subject_info.ty().clone());
                let narrowed = self.narrow(
                    &narrowing_subject_info,
                    op.as_ref(),
                    *narrow_range,
                    &ignore_errors,
                );
                self.get_facet_chain_type(&narrowed, &resolved_chain, *subject_range)
            }
        };
        self.expand_mut(&mut remaining_ty);
        // If the result is `Never` then the cases were exhaustive
        if remaining_ty.is_never() || remaining_ty.is_any() {
            return;
        }
        let subject_display = self.for_display(subject_info.into_ty());
        let remaining_display = self.for_display(remaining_ty.clone());
        let ctx = TypeDisplayContext::new(&[&subject_display, &remaining_display]);
        let displayed_subject = if show_subject_expr {
            self.module().code_at(*subject_range).to_owned()
        } else {
            ctx.display(&subject_display).to_string()
        };
        let message = format!("Match on `{displayed_subject}` is not exhaustive");
        let mut builder =
            errors.error_builder(*subject_range, ErrorKind::NonExhaustiveMatch, message);
        if let Some(missing_cases) = self.format_missing_cases(&remaining_ty) {
            builder = builder.with_detail(format!("Missing cases: {}", missing_cases));
        }
        builder.emit();
    }

    pub fn check_match_case_reachability(
        &self,
        subject_idx: &Idx<Key>,
        narrowing_subject: Option<&NarrowingSubject>,
        narrow_ops_for_case: &(Box<NarrowOp>, TextRange),
        case_range: &TextRange,
        errors: &ErrorCollector,
    ) {
        let (op, narrow_range) = narrow_ops_for_case;
        if !Self::is_match_case_reachability_op(op) {
            return;
        }
        let subject_info = self.with_type_for_exhaustiveness_check(self.get_idx(*subject_idx));
        let subject_ty = subject_info.ty().clone();
        if subject_ty.is_any()
            || matches!(&subject_ty, Type::ClassType(cls) if cls.is_builtin("object"))
        {
            return;
        }
        let ignore_errors = self.error_swallower();
        let (mut narrowed_ty, has_never_trigger_facet) = match narrowing_subject {
            None | Some(NarrowingSubject::Name(_)) => {
                let narrowed =
                    self.narrow(&subject_info, op.as_ref(), *narrow_range, &ignore_errors);
                let has_never_trigger_facet =
                    self.has_never_match_trigger_facet(&narrowed, op, *case_range);
                (narrowed.ty().clone(), has_never_trigger_facet)
            }
            Some(NarrowingSubject::Facets(_, facets)) => {
                let Some(resolved_chain) = self.resolve_facet_chain(facets.chain.clone()) else {
                    return;
                };
                let type_info = TypeInfo::of_ty(self.heap.mk_any_implicit());
                let narrowing_subject_info =
                    type_info.with_narrow(resolved_chain.facets(), subject_ty.clone());
                let narrowed = self.narrow(
                    &narrowing_subject_info,
                    op.as_ref(),
                    *narrow_range,
                    &ignore_errors,
                );
                // No sub-facet check needed: the narrowed type already represents
                // the facet's type directly, so `is_never()` below is sufficient.
                (
                    self.get_facet_chain_type(&narrowed, &resolved_chain, *case_range),
                    false,
                )
            }
        };
        self.expand_mut(&mut narrowed_ty);
        if !narrowed_ty.is_never() && !has_never_trigger_facet {
            return;
        }
        let subject_display = self.for_display(subject_ty);
        self.error(
            errors,
            *case_range,
            ErrorKind::UnreachableMatchCase,
            format!("Case pattern can never match subject of type `{subject_display}`"),
        );
    }

    /// Check whether we can reliably determine reachability for this op.
    ///
    /// Returns `true` only when we understand all sub-ops AND at least one
    /// constrains a value or class tightly enough to produce `Never`
    /// (a "trigger", e.g. `Eq`, `Is`, or `IsInstance`).
    ///
    /// Structural ops like `IsSequence`/`LenEq` are checkable but aren't
    /// triggers on their own.
    fn is_match_case_reachability_op(op: &NarrowOp) -> bool {
        // Classifies as (can_check, has_trigger). The two dimensions can't be
        // collapsed into a single bool because And/Or need to distinguish
        // "structural but checkable" from "unknown and uncheckable" in sub-ops.
        fn classify(op: &NarrowOp) -> (bool, bool) {
            match op {
                NarrowOp::Atomic(_, AtomicNarrowOp::Eq(_) | AtomicNarrowOp::Is(_)) => (true, true),
                NarrowOp::Atomic(_, AtomicNarrowOp::IsInstance(_, NarrowSource::Pattern)) => {
                    (true, true)
                }
                NarrowOp::Atomic(
                    _,
                    AtomicNarrowOp::IsSequence
                    | AtomicNarrowOp::IsMapping
                    | AtomicNarrowOp::LenEq(_)
                    | AtomicNarrowOp::LenGte(_),
                ) => (true, false),
                NarrowOp::And(ops) | NarrowOp::Or(ops) => {
                    let mut has_trigger = false;
                    for op in ops {
                        let (can_check, op_has_trigger) = classify(op);
                        if !can_check {
                            return (false, false);
                        }
                        has_trigger |= op_has_trigger;
                    }
                    (true, has_trigger)
                }
                _ => (false, false),
            }
        }
        let (can_check, has_trigger) = classify(op);
        can_check && has_trigger
    }

    fn has_never_match_trigger_facet(
        &self,
        type_info: &TypeInfo,
        op: &NarrowOp,
        range: TextRange,
    ) -> bool {
        match op {
            NarrowOp::Atomic(
                Some(facet),
                AtomicNarrowOp::Eq(_)
                | AtomicNarrowOp::Is(_)
                | AtomicNarrowOp::IsInstance(_, NarrowSource::Pattern),
            ) => {
                let Some(resolved_chain) = self.resolve_facet_chain(facet.chain.clone()) else {
                    return false;
                };
                let mut facet_ty = self.get_facet_chain_type(type_info, &resolved_chain, range);
                self.expand_mut(&mut facet_ty);
                facet_ty.is_never()
            }
            NarrowOp::And(ops) => ops
                .iter()
                .any(|op| self.has_never_match_trigger_facet(type_info, op, range)),
            NarrowOp::Or(ops) => {
                !ops.is_empty()
                    && ops
                        .iter()
                        .all(|op| self.has_never_match_trigger_facet(type_info, op, range))
            }
            _ => false,
        }
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
                    Type::Literal(lit) if let Lit::Int(lit_int) = &lit.value => {
                        lit_int.as_i64().map(FacetKind::Index)
                    }
                    Type::Literal(lit) if let Lit::Str(s) = &lit.value => {
                        Some(FacetKind::Key(s.to_string()))
                    }
                    _ => None,
                }
            }
            UnresolvedFacetKind::MatchArg { class, index } => {
                // Resolve the positional slot to an attribute name via the class's `__match_args__`
                let suppress_errors = self.error_swallower();
                let class_range = class.range();
                let Type::ClassDef(cls) = self.expr_infer(&class, &suppress_errors) else {
                    return None;
                };
                let instance = self.promote_silently(&cls);
                let match_args = self.attr_infer_for_type(
                    &instance,
                    &dunder::MATCH_ARGS,
                    class_range,
                    &suppress_errors,
                    None,
                );
                if let Type::Tuple(Tuple::Concrete(ts)) = &match_args
                    && let Some(Type::Literal(lit)) = ts.get(index)
                    && let Lit::Str(attr_name) = &lit.value
                {
                    Some(FacetKind::Attribute(Name::new(attr_name)))
                } else {
                    None
                }
            }
        }
    }

    fn literal_equal(&self, left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::None, Type::None) => true,
            (Type::Ellipsis, Type::Ellipsis) => true,
            (Type::Sentinel(s1), Type::Sentinel(s2)) => s1 == s2,
            (Type::Literal(left), Type::Literal(right)) => left.value == right.value,
            _ => false,
        }
    }
}
