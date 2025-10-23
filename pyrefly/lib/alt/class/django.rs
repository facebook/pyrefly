/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::ParamList;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::WithDefiningClass;
use crate::alt::class::enums::VALUE_PROP;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::types::simplify::unions;

/// Django stubs use this attribute to specify the Python type that a field should infer to
const DJANGO_PRIVATE_GET_TYPE: Name = Name::new_static("_pyi_private_get_type");

const CHOICES: Name = Name::new_static("choices");
const LABEL: Name = Name::new_static("label");
const LABELS: Name = Name::new_static("labels");
const VALUES: Name = Name::new_static("values");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_django_field_type(&self, ty: &Type, class: &Class) -> Option<Type> {
        match ty {
            Type::ClassType(cls)
                if cls.has_qname(ModuleName::django_utils_functional().as_str(), "_Getter") =>
            {
                cls.targs().as_slice().first().cloned()
            }
            Type::ClassType(cls)
                if self.get_metadata_for_class(class).is_django_model()
                    && self.inherits_from_django_field(cls.class_object()) =>
            {
                self.get_class_member(cls.class_object(), &DJANGO_PRIVATE_GET_TYPE)
                    .map(|member| member.value.ty())
            }
            Type::Union(union) => {
                let transformed: Vec<_> = union
                    .iter()
                    .map(|variant| {
                        self.get_django_field_type(variant, class)
                            .unwrap_or_else(|| variant.clone())
                    })
                    .collect();

                if transformed != union.to_vec() {
                    Some(unions(transformed))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if a class inherits from Django's Field class
    fn inherits_from_django_field(&self, cls: &crate::types::class::Class) -> bool {
        self.get_mro_for_class(cls)
            .ancestors(self.stdlib)
            .any(|ancestor| {
                ancestor.has_qname(ModuleName::django_models_fields().as_str(), "Field")
            })
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

        let enum_members = self.get_enum_members(cls);

        let mut label_types: Vec<Type> = enum_members
            .iter()
            .filter_map(|lit| {
                if let Lit::Enum(lit_enum) = lit
                    && let Type::Tuple(Tuple::Concrete(elements)) = &lit_enum.ty
                    && elements.len() >= 2
                {
                    Some(elements[elements.len() - 1].clone())
                } else {
                    None
                }
            })
            .collect();

        label_types.push(self.stdlib.str().clone().to_type());

        // Also include the type of __empty__ field if it exists, since it contributes to label types
        let empty_name = Name::new_static("__empty__");
        let has_empty = if let Some(WithDefiningClass { value, .. }) =
            self.get_class_member(cls, &empty_name)
        {
            label_types.push(value.ty());
            true
        } else {
            false
        };

        let label_type = self.unions(label_types);

        let base_value_attr = self.get_enum_or_instance_attribute(
            &self.as_class_type_unchecked(cls),
            &metadata,
            &VALUE_PROP,
        );
        let base_value_type = base_value_attr
            .and_then(|attr| {
                self.resolve_get_class_attr(
                    attr,
                    TextRange::default(),
                    &self.error_swallower(),
                    None,
                )
                .ok()
            })
            .unwrap_or_else(Type::any_implicit);

        // if value is optional, make the type optional
        let values_type = if has_empty {
            self.union(base_value_type.clone(), Type::None)
        } else {
            base_value_type
        };

        let mut fields = SmallMap::new();

        let field_specs = [
            (LABELS, self.stdlib.list(label_type.clone()).to_type()),
            (LABEL, self.property(cls, LABEL, label_type.clone())),
            (VALUES, self.stdlib.list(values_type.clone()).to_type()),
            (
                CHOICES,
                self.stdlib
                    .list(Type::Tuple(Tuple::Concrete(vec![values_type, label_type])))
                    .to_type(),
            ),
        ];

        for (name, ty) in field_specs {
            fields.insert(name, ClassSynthesizedField::new(ty));
        }

        Some(ClassSynthesizedFields::new(fields))
    }

    fn property(&self, cls: &Class, name: Name, ty: Type) -> Type {
        let signature = Callable::list(ParamList::new(vec![self.class_self_param(cls, false)]), ty);
        let mut metadata = FuncMetadata::def(self.module().name(), cls.name().clone(), name);
        metadata.flags.is_property_getter = true;
        Type::Function(Box::new(Function {
            signature,
            metadata,
        }))
    }
}
