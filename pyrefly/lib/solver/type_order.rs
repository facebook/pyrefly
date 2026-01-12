/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Clone_;
use dupe::Copy_;
use dupe::Dupe_;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::types::BoundMethod;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::variance_inference::VarianceMap;
use crate::binding::binding::KeyVariance;
use crate::solver::solver::QuantifiedHandle;
use crate::solver::solver::SubsetError;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::stdlib::Stdlib;
use crate::types::typed_dict::TypedDict;
use crate::types::typed_dict::TypedDictField;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Type;

/// `TypeOrder` provides a minimal API allowing `Subset` to request additional
/// information about types that may be required for solving bindings
///
/// This is needed for cases like the nominal type order and structural types where
/// the `Type` object itself does not contain enough information to determine
/// subset relations.
#[derive(Clone_, Copy_, Dupe_)]
pub struct TypeOrder<'a, Ans: LookupAnswer>(&'a AnswersSolver<'a, Ans>);

impl<'a, Ans: LookupAnswer> TypeOrder<'a, Ans> {
    pub fn new(solver: &'a AnswersSolver<'a, Ans>) -> Self {
        Self(solver)
    }

    #[allow(dead_code)]
    pub fn is_debug(self) -> bool {
        self.0.is_debug()
    }

    pub fn stdlib(self) -> &'a Stdlib {
        self.0.stdlib
    }

    pub fn has_superclass(self, got: &Class, want: &Class) -> bool {
        self.0.has_superclass(got, want)
    }

    pub fn as_superclass(self, class: &ClassType, want: &Class) -> Option<ClassType> {
        self.0.as_superclass(class, want)
    }

    pub fn as_class_type_unchecked(self, class: &Class) -> ClassType {
        self.0.as_class_type_unchecked(class)
    }

    pub fn has_metaclass(self, cls: &Class, metaclass: &ClassType) -> bool {
        let metadata = self.0.get_metadata_for_class(cls);
        self.0
            .as_superclass(metadata.metaclass(self.stdlib()), metaclass.class_object())
            .as_ref()
            == Some(metaclass)
    }

    pub fn is_protocol(self, cls: &Class) -> bool {
        self.0.get_metadata_for_class(cls).is_protocol()
    }

    pub fn is_new_type(self, cls: &Class) -> bool {
        self.0.get_metadata_for_class(cls).is_new_type()
    }

    pub fn get_protocol_member_names(self, cls: &Class) -> SmallSet<Name> {
        let meta = self.0.get_metadata_for_class(cls);
        if let Some(proto) = meta.protocol_metadata() {
            proto.members.clone()
        } else {
            SmallSet::new()
        }
    }

    pub fn get_enum_member_count(self, cls: &Class) -> Option<usize> {
        self.0.get_enum_member_count(cls)
    }

    pub fn instance_as_dunder_call(self, class_type: &ClassType) -> Option<Type> {
        self.0.instance_as_dunder_call(class_type)
    }

    pub fn is_protocol_subset_at_attr(
        self,
        got: &Type,
        protocol: &ClassType,
        name: &Name,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), SubsetError> {
        self.0
            .is_protocol_subset_at_attr(got, protocol, name, is_subset)
    }

    pub fn as_tuple_type(self, cls: &ClassType) -> Option<Type> {
        self.0.as_tuple(cls).map(Type::Tuple)
    }

    pub fn extends_any(self, cls: &Class) -> bool {
        self.0.extends_any(cls)
    }

    pub fn promote_silently(self, cls: &Class) -> Type {
        self.0.promote_silently(cls)
    }

    pub fn typed_dict_fields(self, typed_dict: &TypedDict) -> SmallMap<Name, TypedDictField> {
        self.0.typed_dict_fields(typed_dict)
    }

    pub fn typed_dict_kw_param_info(self, typed_dict: &TypedDict) -> Vec<(Name, Type, Required)> {
        self.0.typed_dict_kw_param_info(typed_dict)
    }

    pub fn typed_dict_extra_items(self, typed_dict: &TypedDict) -> ExtraItems {
        self.0.typed_dict_extra_items(typed_dict)
    }

    pub fn get_typed_dict_value_type(self, typed_dict: &TypedDict) -> Type {
        self.0.get_typed_dict_value_type(typed_dict)
    }

    pub fn get_typed_dict_value_type_as_builtins_dict(
        self,
        typed_dict: &TypedDict,
    ) -> Option<Type> {
        self.0
            .get_typed_dict_value_type_as_builtins_dict(typed_dict)
    }

    pub fn get_variance_from_class(self, cls: &Class) -> Arc<VarianceMap> {
        self.0
            .get_from_class(cls, &KeyVariance(cls.index()))
            .unwrap_or_default()
    }

    pub fn constructor_to_callable(self, cls: &ClassType) -> Type {
        self.0.constructor_to_callable(cls)
    }

    pub fn instantiate_fresh_forall(self, forall: Forall<Forallable>) -> (QuantifiedHandle, Type) {
        self.0.instantiate_fresh_forall(forall)
    }

    pub fn bind_boundmethod(
        self,
        m: &BoundMethod,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Option<Type> {
        self.0.bind_boundmethod(m, is_subset)
    }
}
