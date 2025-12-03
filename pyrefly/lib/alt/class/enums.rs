/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use itertools::Itertools;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::annotation::Annotation;
use pyrefly_types::class::ClassType;
use pyrefly_types::literal::LitEnum;
use pyrefly_types::read_only::ReadOnlyReason;
use pyrefly_types::tuple::Tuple;
use ruff_python_ast::helpers::is_dunder;
use ruff_python_ast::helpers::is_sunder;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassAttribute;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
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

    fn is_valid_enum_member(
        &self,
        name: &Name,
        ty: &Type,
        is_initialized_on_class_body: bool,
    ) -> bool {
        // Names starting but not ending with __ are private
        // Names starting and ending with _ are reserved by the enum
        if Ast::is_mangled_attr(name) || (is_sunder(name.as_str()) || is_dunder(name.as_str())) {
            return false;
        }
        // Enum members must be initialized on the class
        if !is_initialized_on_class_body {
            return false;
        }
        match ty {
            // Methods decorated with @member are members
            _ if ty.has_enum_member_decoration() => true,
            // Callables are not valid enum members
            _ if ty.is_toplevel_callable() => false,
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
        if !((name == &VALUE || name == &VALUE_PROP)
            && (self.field_is_inherited_from(
                class.class_object(),
                name,
                (ModuleName::enum_().as_str(), "Enum"),
            ) || self.field_is_inherited_from(
                class.class_object(),
                name,
                (ModuleName::django_models_enums().as_str(), "Choices"),
            )))
        {
            return None;
        }
        if name == &VALUE {
            let ty = self
                .mixed_in_enum_data_type(class.class_object())
                .unwrap_or_else(|| {
                    if let Some(lit_enum) = enum_literal {
                        self.enum_literal_to_value_type(lit_enum.clone(), enum_metadata.is_django)
                    } else {
                        // The `_value_` annotation on `enum.Enum` is `Any`; we can infer a better type
                        let enum_value_types: Vec<_> = self
                            .get_enum_members(class.class_object())
                            .into_iter()
                            .filter_map(|lit| {
                                if let Lit::Enum(lit_enum) = lit {
                                    Some(self.enum_literal_to_value_type(
                                        *lit_enum,
                                        enum_metadata.is_django,
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if enum_value_types.is_empty() {
                            // Assume Any, rather than Never, if there are no members because they may
                            // be created dynamically and we don't want downstream analysis to be incorrect.
                            Type::any_implicit()
                        } else {
                            self.unions(enum_value_types)
                        }
                    }
                });
            Some(ClassAttribute::read_write(ty))
        } else if let Some(lit_enum) = enum_literal {
            self.get_enum_literal_or_instance_attribute(lit_enum, metadata, &VALUE)
                .map(|attr| {
                    // Do not allow writing `.value`, which is a property.
                    attr.read_only_equivalent(ReadOnlyReason::EnumMemberValue)
                })
        } else {
            self.get_enum_or_instance_attribute(class, metadata, &VALUE)
                .map(|attr| {
                    // Do not allow writing `.value`, which is a property.
                    attr.read_only_equivalent(ReadOnlyReason::EnumMemberValue)
                })
        }
    }

    /// If this enum mixes in a data type by inheriting from it, return the mixed-in type.
    fn mixed_in_enum_data_type(&self, class: &Class) -> Option<Type> {
        let bases = self.get_base_types_for_class(class);
        let first_base = bases.iter().next()?;
        let enum_class = self.stdlib.enum_class();
        if first_base == enum_class {
            None
        } else if self.has_superclass(first_base.class_object(), enum_class.class_object()) {
            self.mixed_in_enum_data_type(first_base.class_object())
        } else {
            Some(first_base.clone().to_type())
        }
    }

    fn enum_literal_to_value_type(&self, lit_enum: LitEnum, is_django: bool) -> Type {
        let ty = match lit_enum.ty {
            Type::Tuple(Tuple::Concrete(elements)) if is_django && elements.len() >= 2 => {
                // The last element is the label.
                let value_len = elements.len() - 1;
                Type::concrete_tuple(elements.into_iter().take(value_len).collect())
            }
            ty => ty,
        };
        let int_ty = self.stdlib.int();
        ty.transform(&mut |t| {
            if matches!(t, Type::ClassType(cls) if cls.has_qname(ModuleName::enum_().as_str(), "auto")) {
                *t = int_ty.clone().to_type();
            }
        })
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
    /// TODO(stroxler, yangdanny): We currently operate on promoted types, which means we do not infer `Literal[...]`
    /// types for the `.value` / `._value_` attributes of literals. This is permitted in the spec although not optimal
    /// for most cases; we are handling it this way in part because generic enum behavior is not yet well-specified.
    ///
    /// We currently skip the check for `_value_` if the class defines `__new__`, since that can
    /// change the value of the enum member. https://docs.python.org/3/howto/enum.html#when-to-use-new-vs-init
    pub fn get_enum_class_field_type(
        &self,
        class: &Class,
        name: &Name,
        direct_annotation: Option<&Annotation>,
        ty: &Type,
        is_initialized_on_class_body: bool,
        is_descriptor: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if is_descriptor {
            return None;
        }
        let metadata = self.get_metadata_for_class(class);
        if let Some(enum_) = metadata.enum_metadata()
            && self.is_valid_enum_member(name, ty, is_initialized_on_class_body)
        {
            if direct_annotation.is_some() {
                self.error(
                    errors, range,ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("Enum member `{name}` may not be annotated directly. Instead, annotate the `_value_` attribute."),
                );
            }
            if enum_.has_value
                && let Some(enum_value_ty) = self.type_of_enum_value(enum_)
                && !class.fields().contains(&dunder::NEW)
                && (!matches!(ty, Type::Ellipsis) || !self.module().path().is_interface())
            {
                self.check_enum_value_annotation(ty, &enum_value_ty, name, range, errors);
            }
            Some(Type::Literal(Lit::Enum(Box::new(LitEnum {
                class: enum_.cls.clone(),
                member: name.clone(),
                ty: ty.clone(),
            }))))
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
