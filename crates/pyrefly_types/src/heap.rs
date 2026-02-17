/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type heap for allocating types.
//!
//! This module implements a factory pattern for type construction. In the future,
//! this will be backed by a per-module arena for efficient allocation and to enable
//! Copy type references.
//!
//! Currently this is a pass-through factory that returns boxed types, allowing
//! incremental migration of construction sites before switching to arena allocation.

use crate::callable::Callable;
use crate::callable::Function;
use crate::callable::Param;
use crate::callable::ParamList;
use crate::callable::Params;
use crate::callable::Required;
use crate::class::ClassType;
use crate::dimension::SizeExpr;
use crate::keywords::KwCall;
use crate::literal::LitStyle;
use crate::literal::Literal;
use crate::module::ModuleType;
use crate::param_spec::ParamSpec;
use crate::quantified::Quantified;
use crate::special_form::SpecialForm;
use crate::tensor::TensorType;
use crate::type_alias::TypeAliasData;
use crate::type_var::TypeVar;
use crate::type_var_tuple::TypeVarTuple;
use crate::typed_dict::TypedDict;
use crate::types::BoundMethod;
use crate::types::Forall;
use crate::types::Forallable;
use crate::types::Overload;
use crate::types::SuperObj;
use crate::types::Type;
use crate::types::Union;

/// A factory for constructing types.
///
/// Currently returns boxed types; will be backed by an arena in the future.
#[derive(Debug)]
pub struct TypeHeap(());

impl TypeHeap {
    /// Create a new type heap.
    pub fn new() -> Self {
        Self(())
    }

    /// Allocate a type in the heap.
    ///
    /// Currently this is a pass-through; in the future it will allocate in the arena.
    pub fn mk(&self, ty: Type) -> Type {
        ty
    }

    /// Create a `Type::None`.
    pub fn mk_none(&self) -> Type {
        Type::None
    }

    /// Create a `Type::Union` from members.
    pub fn mk_union(&self, members: Vec<Type>) -> Type {
        Type::Union(Box::new(Union {
            members,
            display_name: None,
        }))
    }

    /// Create a `Type::Union` with a display name.
    pub fn mk_union_with_name(&self, members: Vec<Type>, display_name: Box<str>) -> Type {
        Type::Union(Box::new(Union {
            members,
            display_name: Some(display_name),
        }))
    }

    /// Create a `Type::Callable` from params and return type.
    pub fn mk_callable(&self, params: Params, ret: Type) -> Type {
        Type::Callable(Box::new(Callable { params, ret }))
    }

    /// Create a `Type::Callable` from an existing Callable.
    pub fn mk_callable_from(&self, callable: Callable) -> Type {
        Type::Callable(Box::new(callable))
    }

    /// Create a `Type::Function` from a Function.
    pub fn mk_function(&self, func: Function) -> Type {
        Type::Function(Box::new(func))
    }

    /// Create a `Type::Type` wrapping an inner type.
    pub fn mk_type(&self, inner: Type) -> Type {
        Type::Type(Box::new(inner))
    }

    /// Create a `Type::Quantified` from a Quantified.
    pub fn mk_quantified(&self, quantified: Quantified) -> Type {
        Type::Quantified(Box::new(quantified))
    }

    /// Create a `Type::Forall` from a Forall.
    pub fn mk_forall(&self, forall: Forall<Forallable>) -> Type {
        Type::Forall(Box::new(forall))
    }

    /// Create a `Type::BoundMethod` from a BoundMethod.
    pub fn mk_bound_method(&self, bound_method: BoundMethod) -> Type {
        Type::BoundMethod(Box::new(bound_method))
    }

    /// Create a `Type::Unpack` wrapping an inner type.
    pub fn mk_unpack(&self, inner: Type) -> Type {
        Type::Unpack(Box::new(inner))
    }

    /// Create a `Type::TypeGuard` wrapping an inner type.
    pub fn mk_type_guard(&self, inner: Type) -> Type {
        Type::TypeGuard(Box::new(inner))
    }

    /// Create a `Type::TypeIs` wrapping an inner type.
    pub fn mk_type_is(&self, inner: Type) -> Type {
        Type::TypeIs(Box::new(inner))
    }

    /// Create a `Type::KwCall` from a KwCall.
    pub fn mk_kw_call(&self, kw_call: KwCall) -> Type {
        Type::KwCall(Box::new(kw_call))
    }

    /// Create a `Type::ElementOfTypeVarTuple` from a Quantified.
    pub fn mk_element_of_type_var_tuple(&self, quantified: Quantified) -> Type {
        Type::ElementOfTypeVarTuple(Box::new(quantified))
    }

    /// Create a `Type::SuperInstance` from lookup class and object.
    pub fn mk_super_instance(&self, lookup_cls: ClassType, obj: SuperObj) -> Type {
        Type::SuperInstance(Box::new((lookup_cls, obj)))
    }

    /// Create a `Type::ArgsValue` from a Quantified.
    pub fn mk_args_value(&self, quantified: Quantified) -> Type {
        Type::ArgsValue(Box::new(quantified))
    }

    /// Create a `Type::KwargsValue` from a Quantified.
    pub fn mk_kwargs_value(&self, quantified: Quantified) -> Type {
        Type::KwargsValue(Box::new(quantified))
    }

    /// Create a `Type::ClassType` from a ClassType.
    pub fn mk_class_type(&self, class_type: ClassType) -> Type {
        Type::ClassType(class_type)
    }

    /// Create a `Type::Any` with implicit style (unknown).
    pub fn mk_any_implicit(&self) -> Type {
        Type::any_implicit()
    }

    /// Create a `Type::Any` with explicit style.
    pub fn mk_any_explicit(&self) -> Type {
        Type::any_explicit()
    }

    /// Create an optional type (T | None).
    pub fn mk_optional(&self, inner: Type) -> Type {
        Type::optional(inner)
    }

    /// Create a `Type::SelfType` from a ClassType.
    pub fn mk_self_type(&self, class_type: ClassType) -> Type {
        Type::SelfType(class_type)
    }

    /// Create a `Type::ClassDef` from a Class.
    pub fn mk_class_def(&self, class: crate::class::Class) -> Type {
        Type::ClassDef(class)
    }

    /// Create a `Type::Tuple` with concrete elements.
    pub fn mk_concrete_tuple(&self, elts: Vec<Type>) -> Type {
        Type::concrete_tuple(elts)
    }

    /// Create a `Type::Never`.
    pub fn mk_never(&self) -> Type {
        Type::never()
    }

    /// Create a `Type::Tuple` from a Tuple.
    pub fn mk_tuple(&self, tuple: crate::tuple::Tuple) -> Type {
        Type::Tuple(tuple)
    }

    /// Create an unbounded tuple type.
    pub fn mk_unbounded_tuple(&self, elem: Type) -> Type {
        Type::unbounded_tuple(elem)
    }

    /// Create an unpacked tuple type.
    pub fn mk_unpacked_tuple(&self, before: Vec<Type>, middle: Type, after: Vec<Type>) -> Type {
        Type::unpacked_tuple(before, middle, after)
    }

    /// Create a `Type::Var` from a Var.
    pub fn mk_var(&self, var: crate::types::Var) -> Type {
        Type::Var(var)
    }

    /// Create a `Type::ParamSpecValue` from a ParamSpec.
    pub fn mk_param_spec_value(&self, params: ParamList) -> Type {
        Type::ParamSpecValue(params)
    }

    /// Create a `Type::Any` with error style.
    pub fn mk_any_error(&self) -> Type {
        Type::any_error()
    }

    /// Create a `Type::Concatenate` from types and a ParamSpec.
    pub fn mk_concatenate(&self, types: Box<[(Type, Required)]>, param_spec: Type) -> Type {
        Type::Concatenate(types, Box::new(param_spec))
    }
    /// Create a `Type::Callable` with ellipsis params.
    pub fn mk_callable_ellipsis(&self, ret: Type) -> Type {
        Type::callable_ellipsis(ret)
    }

    /// Create a `Type::Callable` from a vec of params and return type.
    pub fn mk_callable_from_vec(&self, params: Vec<Param>, ret: Type) -> Type {
        Type::callable(params, ret)
    }

    /// Create a `Type::Callable` with param spec.
    pub fn mk_callable_param_spec(&self, param_spec: Type, ret: Type) -> Type {
        Type::callable_param_spec(param_spec, ret)
    }

    /// Create a `Type::Callable` with concatenate.
    pub fn mk_callable_concatenate(
        &self,
        params: Box<[(Type, Required)]>,
        param_spec: Type,
        ret: Type,
    ) -> Type {
        Type::callable_concatenate(params, param_spec, ret)
    }

    /// Create a `Type::Type` wrapping an inner type (type form).
    ///
    /// This is an alias for `mk_type` matching the `Type::type_form` helper.
    pub fn mk_type_form(&self, inner: Type) -> Type {
        Type::type_form(inner)
    }

    /// Create a `Type::Literal` from a Literal.
    pub fn mk_literal(&self, literal: Literal) -> Type {
        Type::Literal(Box::new(literal))
    }

    /// Create a `Type::LiteralString` with the given style.
    pub fn mk_literal_string(&self, style: LitStyle) -> Type {
        Type::LiteralString(style)
    }

    /// Create a `Type::Overload` from an Overload.
    pub fn mk_overload(&self, overload: Overload) -> Type {
        Type::Overload(overload)
    }

    /// Create a `Type::Intersect` from members and a fallback type.
    pub fn mk_intersect(&self, members: Vec<Type>, fallback: Type) -> Type {
        Type::Intersect(Box::new((members, fallback)))
    }

    /// Create a `Type::TypedDict` from a TypedDict.
    pub fn mk_typed_dict(&self, typed_dict: TypedDict) -> Type {
        Type::TypedDict(typed_dict)
    }

    /// Create a `Type::PartialTypedDict` from a TypedDict.
    pub fn mk_partial_typed_dict(&self, typed_dict: TypedDict) -> Type {
        Type::PartialTypedDict(typed_dict)
    }

    /// Create a `Type::Module` from a ModuleType.
    pub fn mk_module(&self, module: ModuleType) -> Type {
        Type::Module(module)
    }

    /// Create a `Type::QuantifiedValue` from a Quantified.
    pub fn mk_quantified_value(&self, quantified: Quantified) -> Type {
        Type::QuantifiedValue(Box::new(quantified))
    }

    /// Create a `Type::TypeVar` from a TypeVar.
    pub fn mk_type_var(&self, type_var: TypeVar) -> Type {
        Type::TypeVar(type_var)
    }

    /// Create a `Type::ParamSpec` from a ParamSpec.
    pub fn mk_param_spec(&self, param_spec: ParamSpec) -> Type {
        Type::ParamSpec(param_spec)
    }

    /// Create a `Type::TypeVarTuple` from a TypeVarTuple.
    pub fn mk_type_var_tuple(&self, type_var_tuple: TypeVarTuple) -> Type {
        Type::TypeVarTuple(type_var_tuple)
    }

    /// Create a `Type::SpecialForm` from a SpecialForm.
    pub fn mk_special_form(&self, special_form: SpecialForm) -> Type {
        Type::SpecialForm(special_form)
    }

    /// Create a `Type::Args` from a Quantified.
    pub fn mk_args(&self, quantified: Quantified) -> Type {
        Type::Args(Box::new(quantified))
    }

    /// Create a `Type::Kwargs` from a Quantified.
    pub fn mk_kwargs(&self, quantified: Quantified) -> Type {
        Type::Kwargs(Box::new(quantified))
    }

    /// Create a `Type::Ellipsis`.
    pub fn mk_ellipsis(&self) -> Type {
        Type::Ellipsis
    }

    /// Create a `Type::TypeAlias` from a TypeAliasData.
    pub fn mk_type_alias(&self, type_alias: TypeAliasData) -> Type {
        Type::TypeAlias(Box::new(type_alias))
    }

    /// Create a `Type::Materialization`.
    pub fn mk_materialization(&self) -> Type {
        Type::Materialization
    }

    /// Create a `Type::Any` with the given style.
    pub fn mk_any(&self, style: crate::types::AnyStyle) -> Type {
        Type::Any(style)
    }

    /// Create a `Type::Never` with the given style.
    pub fn mk_never_style(&self, style: crate::types::NeverStyle) -> Type {
        Type::Never(style)
    }

    /// Create a `Type::Dim` wrapping an inner type.
    pub fn mk_dim(&self, inner: Type) -> Type {
        Type::Dim(Box::new(inner))
    }

    /// Create a `Type::Size` from a SizeExpr.
    pub fn mk_size(&self, size_expr: SizeExpr) -> Type {
        Type::Size(size_expr)
    }

    /// Create a `Type::Tensor` from a TensorType.
    pub fn mk_tensor(&self, tensor: TensorType) -> Type {
        Type::Tensor(Box::new(tensor))
    }
}
