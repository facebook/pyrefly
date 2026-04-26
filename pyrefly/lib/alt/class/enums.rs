/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_config::error_kind::ErrorKind;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::annotation::Annotation;
use pyrefly_types::class::ClassType;
use pyrefly_types::literal::LitEnum;
use pyrefly_types::read_only::ReadOnlyReason;
use ruff_python_ast::helpers::is_dunder;
use ruff_python_ast::helpers::is_sunder;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassAttribute;
use crate::alt::class::django::transform_django_enum_value;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::binding::binding::ClassFieldDefinition;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;
use crate::types::literal::Lit;
use crate::types::types::Type;

/// The `_value_` attribute in enums is reserved, and can be annotated to
/// indicate an explicit type restriction on enum members. Looking it up
/// on an enum member will give the raw value of that member.
pub const VALUE: Name = Name::new_static("_value_");
/// The `value` attribute of an enum is a property that returns `_value_`.
pub const VALUE_PROP: Name = Name::new_static("value");

pub const GENERATE_NEXT_VALUE: Name = Name::new_static("_generate_next_value_");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_enum_member(&self, cls: &Class, name: &Name) -> Option<Lit> {
        self.get_field_from_current_class_only(cls, name)
            .and_then(|field| self.as_enum_member(Arc::unwrap_or_clone(field), cls))
    }

    pub fn get_enum_members(&self, cls: &Class) -> SmallSet<Lit> {
        self.get_class_fields(cls)
            .map(|class_fields| {
                class_fields
                    .names()
                    .filter_map(|f| self.get_enum_member(cls, f))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn is_valid_enum_member(
        &self,
        name: &Name,
        ty: &Type,
        field_definition: &ClassFieldDefinition,
    ) -> bool {
        // Names starting but not ending with __ are private.
        // Names starting and ending with _ are reserved by the enum.
        if Ast::is_mangled_attr(name) || is_sunder(name.as_str()) || is_dunder(name.as_str()) {
            return false;
        }
        // Methods decorated with @enum.member are always enum members.
        if ty.has_enum_member_decoration() {
            return true;
        }
        // Only values assigned or defined in the class body can be enum members.
        // MethodLike definitions (def statements) are not enum member candidates
        // unless decorated with @member (handled above).
        match field_definition {
            ClassFieldDefinition::AssignedInBody { .. }
            | ClassFieldDefinition::DefinedWithoutAssign { .. } => {}
            _ => return false,
        }
        match ty {
            // Callables are not valid enum members.
            _ if ty.is_toplevel_callable() => false,
            // Values initialized with nonmember() or descriptor-like wrappers are not members.
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

    /// Checks for a special-cased enum attribute, falling back to a regular instance attribute lookup.
    pub fn get_enum_or_instance_attribute(
        &self,
        class: &ClassType,
        metadata: &ClassMetadata,
        attr_name: &Name,
    ) -> Option<ClassAttribute> {
        self.special_case_enum_attr_lookup(class, None, metadata, attr_name)
            .or_else(|| self.get_instance_attribute(class, attr_name))
    }

    /// Checks for a special-cased enum attribute on an enum literal, falling back to a regular instance attribute lookup.
    pub fn get_enum_literal_or_instance_attribute(
        &self,
        lit: &LitEnum,
        metadata: &ClassMetadata,
        attr_name: &Name,
    ) -> Option<ClassAttribute> {
        let class = &lit.class;
        self.special_case_enum_attr_lookup(class, Some(lit), metadata, attr_name)
            .or_else(|| self.get_instance_attribute(class, attr_name))
    }

    /// Special-case enum attribute lookups. Dispatches to the appropriate helper
    /// based on the attribute name and whether we have a known enum literal.
    ///
    /// `enum_literal` is set if we're looking this up on a known member, like `Literal[MyEnum.X]`
    ///
    /// Return None if either this is not an enum or this is not a special-case
    /// attribute.
    fn special_case_enum_attr_lookup(
        &self,
        class: &ClassType,
        enum_literal: Option<&LitEnum>,
        metadata: &ClassMetadata,
        name: &Name,
    ) -> Option<ClassAttribute> {
        let enum_metadata = metadata.enum_metadata()?;
        if name == &VALUE {
            if !self.field_defining_class_matches(class.class_object(), &VALUE, |c| {
                c.has_toplevel_qname(ModuleName::enum_().as_str(), "Enum")
                    || c.has_toplevel_qname(ModuleName::enum_().as_str(), "IntEnum")
                    || c.has_toplevel_qname(ModuleName::enum_().as_str(), "StrEnum")
                    || c.has_toplevel_qname(ModuleName::django_models_enums().as_str(), "Choices")
            }) {
                return None;
            }
            let ty = if let Some(lit_enum) = enum_literal {
                self.enum_value_lookup_on_member(class, lit_enum, enum_metadata)
            } else {
                self.enum_value_lookup_on_class(class, enum_metadata)
            };
            Some(ClassAttribute::read_write(ty))
        } else if name == &VALUE_PROP {
            if !self.field_defining_class_matches(class.class_object(), &VALUE_PROP, |c| {
                c.has_toplevel_qname(ModuleName::enum_().as_str(), "Enum")
                    || c.has_toplevel_qname(ModuleName::enum_().as_str(), "IntEnum")
                    || c.has_toplevel_qname(ModuleName::enum_().as_str(), "StrEnum")
                    || c.has_toplevel_qname(ModuleName::django_models_enums().as_str(), "Choices")
            }) {
                return None;
            }
            if let Some(lit_enum) = enum_literal {
                self.get_enum_literal_or_instance_attribute(lit_enum, metadata, &VALUE)
                    .map(|attr| attr.read_only_equivalent(ReadOnlyReason::EnumMemberValue))
            } else {
                self.get_enum_or_instance_attribute(class, metadata, &VALUE)
                    .map(|attr| attr.read_only_equivalent(ReadOnlyReason::EnumMemberValue))
            }
        } else {
            None
        }
    }

    /// Look up the `_value_` attribute for a specific enum member (e.g. `MyEnum.X._value_`).
    /// Whether `_value_` should be read-write is unspecified, but we need to allow assigning
    /// it in `__init__` so we make it read-write.
    fn enum_value_lookup_on_member(
        &self,
        class: &ClassType,
        lit_enum: &LitEnum,
        enum_metadata: &EnumMetadata,
    ) -> Type {
        let mixed_in = self.mixed_in_enum_data_type(class.class_object());
        let has_new = self
            .get_class_fields(class.class_object())
            .is_some_and(|f| f.contains(&dunder::NEW));
        // When `__new__` is defined, it can rewrite `_value_` at runtime, so the raw
        // RHS type is unreliable.
        // Fallbacks in order of priority: mixed-in type, type of `_value_`, `Any`
        if has_new {
            return if let Some(mixed_in) = mixed_in {
                mixed_in
            } else if let Some(value_ty) = self.type_of_enum_value(enum_metadata) {
                value_ty
            } else {
                self.heap.mk_any_implicit()
            };
        }
        let value_ty = self.enum_literal_to_value_type(lit_enum.clone(), enum_metadata.is_django);
        // Only preserve the literal type if its base class type matches the mixin exactly.
        if let Some(ref mixed_in) = mixed_in {
            let promoted = value_ty.clone().promote_implicit_literals(self.stdlib);
            if &promoted == mixed_in {
                value_ty
            } else {
                mixed_in.clone()
            }
        } else {
            value_ty
        }
    }

    /// Look up the `_value_` attribute for an enum type (not a specific member).
    /// Whether `_value_` should be read-write is unspecified, but we need to allow assigning
    /// it in `__init__` so we make it read-write.
    fn enum_value_lookup_on_class(&self, class: &ClassType, enum_metadata: &EnumMetadata) -> Type {
        let mixed_in = self.mixed_in_enum_data_type(class.class_object());
        let has_new = self
            .get_class_fields(class.class_object())
            .is_some_and(|f| f.contains(&dunder::NEW));
        // When `__new__` is defined, it can rewrite `_value_` at runtime. Fall back to the
        // mixed-in type, or `Any` if there is no mixin.
        if has_new {
            return if let Some(mixed_in) = mixed_in {
                mixed_in
            } else {
                self.heap.mk_any_implicit()
            };
        }
        if let Some(mixed_in) = mixed_in {
            return mixed_in;
        }
        // The `_value_` annotation on `enum.Enum` is `Any`; we can infer a better type.
        let enum_value_types: Vec<_> = self
            .get_enum_members(class.class_object())
            .into_iter()
            .filter_map(|lit| {
                if let Lit::Enum(lit_enum) = lit {
                    let value_ty =
                        self.enum_literal_to_value_type(*lit_enum, enum_metadata.is_django);
                    if value_ty.is_implicit_literal() {
                        Some(value_ty.promote_implicit_literals(self.stdlib))
                    } else {
                        Some(value_ty)
                    }
                } else {
                    None
                }
            })
            .collect();
        if enum_value_types.is_empty() {
            // Don't assume Never if there are no members, because they may
            // be created dynamically and we don't want false-positives downstream.
            self.heap.mk_any_implicit()
        } else {
            self.unions(enum_value_types)
        }
    }

    /// If this enum mixes in a data type by inheriting from it, return the mixed-in type.
    /// Searches all bases, not just the first, to handle cases like
    /// `IntegerChoices(Choices, IntEnum)` where the data type comes from `IntEnum`.
    /// A non-enum base is only treated as a data type if it has `__new__`
    /// inherited from a class other than `object`, which is how Python
    /// distinguishes data type mixins (`int`, `str`, `float`, and their
    /// subclasses like `MyStr(str)`) from regular method mixins.
    fn mixed_in_enum_data_type(&self, class: &Class) -> Option<Type> {
        let bases = self.get_base_types_for_class(class);
        let enum_class = self.stdlib.enum_class();
        for base in bases.iter() {
            if *base == *enum_class {
                continue;
            } else if self.has_superclass(base.class_object(), enum_class.class_object()) {
                if let Some(ty) = self.mixed_in_enum_data_type(base.class_object()) {
                    return Some(ty);
                }
            } else {
                let is_data_type = self
                    .get_class_member_with_defining_class(base.class_object(), &dunder::NEW)
                    .is_some_and(|field| {
                        !field
                            .defining_class
                            .has_toplevel_qname("builtins", "object")
                    });
                if is_data_type {
                    return Some(self.heap.mk_class_type(base.clone()));
                }
            }
        }
        None
    }

    /// Convert an enum literal's raw value type to its `.value` type.
    fn enum_literal_to_value_type(&self, lit_enum: LitEnum, is_django: bool) -> Type {
        let ty = if is_django {
            transform_django_enum_value(lit_enum.ty, self.heap)
        } else {
            lit_enum.ty
        };
        let auto_ty = self.auto_value_type(lit_enum.class.class_object());
        ty.transform(&mut |t| {
            if matches!(t, Type::ClassType(cls) if cls.has_qname(ModuleName::enum_().as_str(), "auto")) {
                *t = auto_ty.clone();
            }
        })
    }

    /// Determine the type that `auto()` produces for the given enum class.
    /// 1. If a data type is mixed in (e.g. `class E(str, Enum)`), `auto()` produces
    ///    that type because `__new__` converts the raw value.
    /// 2. Otherwise, look up `_generate_next_value_` on the class, defaulting to `int` for `enum.Enum`
    fn auto_value_type(&self, cls: &Class) -> Type {
        // A mixed-in data type takes priority: the enum's `__new__` converts values to that type.
        if let Some(mixed_in) = self.mixed_in_enum_data_type(cls) {
            return mixed_in;
        }
        self.get_class_member_with_defining_class(cls, &GENERATE_NEXT_VALUE)
            .and_then(|field| {
                // `enum.Enum` declares an `Any` return, but at runtime it generates `int`
                if field.defining_class.has_toplevel_qname("enum", "Enum") {
                    Some(self.heap.mk_class_type(self.stdlib.int().clone()))
                } else {
                    field.value.ty().callable_return_type(self.heap)
                }
            })
            .unwrap_or_else(|| self.heap.mk_any_implicit()) // Fall back to `Any` if `_generate_next_value_` is missing or not callable
    }

    pub fn get_enum_member_count(&self, cls: &Class) -> Option<usize> {
        let meta = self.get_metadata_for_class(cls);
        if meta.is_enum() {
            Some(self.get_enum_members(cls).len())
        } else {
            None
        }
    }

    /// Enum handling:
    /// - Check whether the field is a member (which depends only on its type and name)
    /// - Validate that a member should not have an annotation, and should respect any explicit annotation on `_value_`
    ///
    /// We currently skip the check for `_value_` if the class defines `__new__`, since that can
    /// change the value of the enum member. https://docs.python.org/3/howto/enum.html#when-to-use-new-vs-init
    pub fn get_enum_class_field_type(
        &self,
        class: &Class,
        name: &Name,
        direct_annotation: Option<&Annotation>,
        ty: &Type,
        field_definition: &ClassFieldDefinition,
        is_descriptor: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if is_descriptor {
            return None;
        }
        // Extract alias_of from field_definition for enum alias detection
        let alias_of = match field_definition {
            ClassFieldDefinition::AssignedInBody { alias_of, .. } => alias_of.as_ref(),
            _ => None,
        };
        let metadata = self.get_metadata_for_class(class);
        if let Some(enum_) = metadata.enum_metadata()
            && self.is_valid_enum_member(name, ty, field_definition)
        {
            if direct_annotation.is_some() {
                self.error(
                    errors, range,ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("Enum member `{name}` may not be annotated directly. Instead, annotate the `_value_` attribute."),
                );
            }
            if enum_.has_value
                && let Some(enum_value_ty) = self.type_of_enum_value(enum_)
                && !self
                    .get_class_fields(class)
                    .is_some_and(|f| f.contains(&dunder::NEW))
                && (!matches!(ty, Type::Ellipsis) || !self.module().path().is_interface())
            {
                self.check_enum_value_annotation(ty, &enum_value_ty, name, range, errors);
            }
            // If this field is an alias (value is a simple name referring to another field),
            // look up the aliased member and return its type instead of creating a new enum literal.
            if let Some(aliased_name) = alias_of
                && let Some(aliased_member_lit) = self.get_enum_member(class, aliased_name)
            {
                return Some(aliased_member_lit.to_implicit_type());
            }
            Some(
                Lit::Enum(Box::new(LitEnum {
                    class: enum_.cls.clone(),
                    member: name.clone(),
                    ty: self.solver().deep_force(ty.clone()),
                }))
                .to_implicit_type(),
            )
        } else if let Type::ClassType(cls) = &ty
            && cls.has_qname("enum", "nonmember")
            && let [targ] = cls.targs().as_slice()
        {
            Some(targ.clone())
        } else {
            None
        }
    }

    /// Look up the `_value_` attribute of an enum class. This field has to be a plain instance
    /// attribute annotated in the class body; it is used to validate enum member values, which are
    /// supposed to all share this type.
    ///
    /// TODO(stroxler): We don't currently enforce in this function that it is
    /// an instance attribute annotated in the class body. Should we? It is unclear; this helper
    /// is only used to validate enum members, not to produce errors on invalid `_value_`
    fn type_of_enum_value(&self, enum_: &EnumMetadata) -> Option<Type> {
        let field = self.get_class_member(enum_.cls.class_object(), &VALUE)?;
        if field.is_simple_instance_attribute() {
            Some(field.ty())
        } else {
            None
        }
    }

    fn check_enum_value_annotation(
        &self,
        mut value: &Type,
        annotation: &Type,
        member: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if matches!(value, Type::Tuple(_)) {
            // TODO: check tuple values against constructor signature
            // see https://typing.python.org/en/latest/spec/enums.html#member-values
            return;
        }
        if matches!(value, Type::ClassType(cls) if cls.has_qname("enum", "auto")) {
            return;
        }
        if let Type::ClassType(cls) = value
            && cls.has_qname("enum", "member")
            && let [member_targ] = cls.targs().as_slice()
        {
            value = member_targ;
        }
        if !self.is_subset_eq(value, annotation) {
            self.error(
                errors, range, ErrorInfo::Kind(ErrorKind::BadAssignment),
                format!(
                    "Enum member `{member}` has type `{}`, must match the `_value_` attribute annotation of `{}`",
                    self.for_display(value.clone()),
                    self.for_display(annotation.clone()),
                ),
            );
        }
    }
}
