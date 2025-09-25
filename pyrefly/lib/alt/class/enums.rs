/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_types::class::ClassType;
use pyrefly_types::read_only::ReadOnlyReason;
use pyrefly_types::tuple::Tuple;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassAttribute;
use crate::alt::class::class_field::ClassFieldInitialization;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::types::class::Class;
use crate::types::literal::Lit;
use crate::types::types::Type;

/// The `_value_` attribute in enums is reserved, and can be annotated to
/// indicate an explicit type restriction on enum members. Looking it up
/// on an enum member will give the raw value of that member.
pub const VALUE: Name = Name::new_static("_value_");
/// The `value` attribute of an enum is a property that returns `_value_`.
const VALUE_PROP: Name = Name::new_static("value");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_enum_member(&self, cls: &Class, name: &Name) -> Option<Lit> {
        self.get_field_from_current_class_only(cls, name)
            .and_then(|field| self.as_enum_member(Arc::unwrap_or_clone(field), cls))
    }

    pub fn get_enum_members(&self, cls: &Class) -> SmallSet<Lit> {
        cls.fields()
            .filter_map(|f| self.get_enum_member(cls, f))
            .collect()
    }

    pub fn is_valid_enum_member(
        &self,
        name: &Name,
        ty: &Type,
        initialization: &ClassFieldInitialization,
    ) -> bool {
        // Names starting but not ending with __ are private
        // Names starting and ending with _ are reserved by the enum
        if name.starts_with("__") && !name.ends_with("__")
            || name.starts_with("_") && name.ends_with("_")
        {
            return false;
        }
        // Enum members must be initialized on the class
        if !matches!(*initialization, ClassFieldInitialization::ClassBody(_)) {
            return false;
        }
        match ty {
            // Methods decorated with @member are members
            _ if ty.has_enum_member_decoration() => true,
            // Callables are not valid enum members
            Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_) => false,
            // Values initialized with nonmember() are not members
            Type::ClassType(cls)
                if cls.has_qname("enum", "nonmember")
                    || cls.is_builtin("staticmethod")
                    || cls.is_builtin("classmethod")
                    || cls.has_qname("types", "DynamicClassAttribute")
                    || cls.has_qname("enum", "property") =>
            {
                false
            }
            _ => true,
        }
    }

    /// Special-case enum attribute lookups:
    /// - if this is an enum and the attribute is `value`, we'll redirect it to
    ///   look up the type of `_value_` so that the `value` property understands
    ///   annotated `_value_`.
    /// - furthermore, if there is no annotation on `_value_` (meaning it inherits
    ///   the `Any` annotation from `enum.Enum` we will compute the type based
    ///   on the observed types of members).
    ///
    /// The resulting attribute is read-only if it is `value`, which is a property,
    /// and read-write if it is `_value_`. Whether `_value_` should be considered
    /// writable is unspecified, but we at least have to allow it in `__init__`.
    ///
    /// Return None if either this is not an enum or this is not a special-case
    /// attribute.
    pub fn special_case_enum_attr_lookup(
        &self,
        class: &ClassType,
        metadata: &ClassMetadata,
        name: &Name,
    ) -> Option<ClassAttribute> {
        if metadata.is_enum() && (name == &VALUE || name == &VALUE_PROP) {
            if self.field_is_inherited_from_enum(class.class_object(), &VALUE) {
                // The `_value_` annotation on `enum.Enum` is `Any`; we can infer a better type
                let enum_value_types: Vec<_> = self
                    .get_enum_members(class.class_object())
                    .into_iter()
                    .filter_map(|lit| {
                        if let Lit::Enum(lit_enum) = lit {
                            Some(lit_enum.ty)
                        } else {
                            None
                        }
                    })
                    .collect();
                let ty = if enum_value_types.is_empty() {
                    // Assume Any, rather than Never, if there are no members because they may
                    // be created dynamically and we don't want downstream analysis to be incorrect.
                    Type::any_implicit()
                } else {
                    self.unions(enum_value_types)
                };
                Some(if name == &VALUE_PROP {
                    ClassAttribute::read_only(ty, ReadOnlyReason::EnumMemberValue)
                } else {
                    ClassAttribute::read_write(ty)
                })
            } else {
                self.get_instance_attribute(class, &VALUE).map(|attr| {
                    // Do not allow writing `.value`, which is a property.
                    if name == &VALUE_PROP {
                        attr.read_only_equivalent(ReadOnlyReason::EnumMemberValue)
                    } else {
                        attr
                    }
                })
            }
        } else {
            None
        }
    }

    pub fn get_enum_member_count(&self, cls: &Class) -> Option<usize> {
        let meta = self.get_metadata_for_class(cls);
        if meta.is_enum() {
            Some(self.get_enum_members(cls).len())
        } else {
            None
        }
    }

    pub fn get_django_enum_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let enum_metadata = metadata.enum_metadata()?;
        if !enum_metadata.is_django {
            return None;
        }

        // Infer value type from actual enum members
        let enum_members = self.get_enum_members(cls);
        let value_types: Vec<_> = enum_members
            .iter()
            .filter_map(|lit| {
                if let Lit::Enum(lit_enum) = lit {
                    Some(lit_enum.ty.clone())
                } else {
                    None
                }
            })
            .collect();

        let base_value_type = if value_types.is_empty() {
            Type::any_implicit()
        } else {
            self.unions(value_types)
        };

        // Check if __empty__ makes values optional
        let empty_name = ruff_python_ast::name::Name::new_static("__empty__");
        let has_empty = self
            .get_field_from_current_class_only(cls, &empty_name)
            .is_some();
        let values_type = if has_empty {
            self.union(base_value_type.clone(), Type::None)
        } else {
            base_value_type
        };

        // Helper to create synthesized fields
        let mut fields = starlark_map::small_map::SmallMap::new();
        let str_type = self.stdlib.str().clone().to_type();

        // Synthesize class attributes: labels, values, choices
        let field_specs = [
            ("labels", self.stdlib.list(str_type.clone()).to_type()),
            ("values", self.stdlib.list(values_type.clone()).to_type()),
            (
                "choices",
                self.stdlib
                    .list(Type::Tuple(Tuple::Concrete(vec![values_type, str_type])))
                    .to_type(),
            ),
        ];

        for (name, ty) in field_specs {
            fields.insert(Name::new_static(name), ClassSynthesizedField::new(ty));
        }

        Some(ClassSynthesizedFields::new(fields))
    }
}
