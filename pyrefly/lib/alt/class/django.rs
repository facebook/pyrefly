/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::class::Class;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::simplify::unions;

/// Django stubs use this attribute to specify the Python type that a field should infer to
const DJANGO_PRIVATE_GET_TYPE: Name = Name::new_static("_pyi_private_get_type");

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
}
