/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::shaped_array::ShapedArrayType;
use pyrefly_types::simplify::intersect;
use pyrefly_types::simplify::unions;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictInner;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Type;

use crate::alt::attr::ClassBase;
use crate::alt::class::class_field::DescriptorBase;

#[derive(Debug)]
pub enum InstanceKind {
    ClassType,
    TypedDict,
    TypeVar(Quantified),
    SelfType,
    Protocol(Type),
    Metaclass(ClassBase),
    LiteralString,
    /// Shaped-array instance: Self is substituted with the full shaped-array type.
    ShapedArray(ShapedArrayType),
}

/// Wrapper to hold a specialized instance of a class , unifying ClassType and TypedDict.
#[derive(Debug)]
pub struct Instance<'a> {
    pub kind: InstanceKind,
    pub class: &'a Class,
    pub targs: &'a TArgs,
}

impl<'a> Instance<'a> {
    pub fn literal_string(stdlib: &'a Stdlib) -> Self {
        Self {
            kind: InstanceKind::LiteralString,
            class: stdlib.str().class_object(),
            targs: stdlib.str().targs(),
        }
    }

    pub fn of_class(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::ClassType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_typed_dict(td: &'a TypedDictInner) -> Self {
        Self {
            kind: InstanceKind::TypedDict,
            class: td.class_object(),
            targs: td.targs(),
        }
    }

    pub fn of_type_var(q: Quantified, bound: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::TypeVar(q),
            class: bound.class_object(),
            targs: bound.targs(),
        }
    }

    pub fn of_self_type(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::SelfType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_protocol(cls: &'a ClassType, self_type: Type) -> Self {
        Self {
            kind: InstanceKind::Protocol(self_type),
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_metaclass(cls: ClassBase, metaclass: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::Metaclass(cls),
            class: metaclass.class_object(),
            targs: metaclass.targs(),
        }
    }

    pub fn of_shaped_array(shaped_array: &'a ShapedArrayType) -> Self {
        Self {
            kind: InstanceKind::ShapedArray(shaped_array.clone()),
            class: shaped_array.base_class.class_object(),
            targs: shaped_array.base_class.targs(),
        }
    }

    /// Instantiate a type that is relative to the class type parameters
    /// by substituting in the type arguments.
    pub fn instantiate_member(&self, raw_member: &mut Type) {
        self.targs.substitute_into_mut(raw_member)
    }

    pub fn to_type(&self, heap: &TypeHeap) -> Type {
        match &self.kind {
            InstanceKind::ClassType => {
                heap.mk_class_type(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::TypedDict => {
                heap.mk_typed_dict(TypedDict::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::TypeVar(q) => {
                let quantified_ty = q.clone().to_type(heap);
                let ub = match q.restriction() {
                    Restriction::Unrestricted => return quantified_ty,
                    Restriction::Bound(b) => b.clone(),
                    Restriction::Constraints(cs) => unions(cs.clone(), heap),
                };
                let instance_ty =
                    heap.mk_class_type(ClassType::new(self.class.dupe(), self.targs.clone()));
                if instance_ty == ub {
                    quantified_ty
                } else {
                    intersect(vec![quantified_ty, instance_ty.clone()], instance_ty, heap)
                }
            }
            InstanceKind::SelfType => {
                heap.mk_self_type(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => self_type.clone(),
            InstanceKind::Metaclass(cls) => cls.clone().to_type(heap),
            InstanceKind::LiteralString => heap.mk_literal_string(LitStyle::Implicit),
            InstanceKind::ShapedArray(shaped_array) => shaped_array.clone().to_type(),
        }
    }

    /// Looking up a classmethod/staticmethod from an instance base has class-like
    /// lookup behavior. When this happens, we convert from an instance base to a class base.
    pub fn to_class_base(&self) -> ClassBase {
        match &self.kind {
            InstanceKind::SelfType => {
                ClassBase::SelfType(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => ClassBase::Protocol(
                ClassType::new(self.class.dupe(), self.targs.clone()),
                self_type.clone(),
            ),
            InstanceKind::TypeVar(q) => ClassBase::Quantified(
                q.clone(),
                ClassType::new(self.class.dupe(), self.targs.clone()),
            ),
            _ => ClassBase::ClassType(ClassType::new(self.class.dupe(), self.targs.clone())),
        }
    }

    pub fn to_descriptor_base(&self) -> Option<DescriptorBase> {
        match self.kind {
            // There's no situation in which you can stick a usable descriptor in a TypedDict.
            // TODO(rechen): a descriptor in a TypedDict should be an error at class creation time.
            InstanceKind::TypedDict => None,
            InstanceKind::SelfType => Some(DescriptorBase::SelfInstance(ClassType::new(
                self.class.dupe(),
                self.targs.clone(),
            ))),
            InstanceKind::ClassType
            | InstanceKind::Protocol(..)
            | InstanceKind::Metaclass(..)
            | InstanceKind::TypeVar(..)
            | InstanceKind::LiteralString
            | InstanceKind::ShapedArray(..) => Some(DescriptorBase::Instance(ClassType::new(
                self.class.dupe(),
                self.targs.clone(),
            ))),
        }
    }
}
