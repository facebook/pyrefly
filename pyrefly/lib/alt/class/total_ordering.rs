/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::dunder;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::class::Class;

// https://github.com/python/cpython/blob/a8ec511900d0d84cffbb4ee6419c9a790d131129/Lib/functools.py#L173
// conversion order of rich comparison methods:
const LT_CONVERSION_ORDER: &[Name; 3] = &[dunder::GT, dunder::LE, dunder::GE];
const GT_CONVERSION_ORDER: &[Name; 3] = &[dunder::LT, dunder::GE, dunder::LE];
const LE_CONVERSION_ORDER: &[Name; 3] = &[dunder::GE, dunder::LT, dunder::GT];
const GE_CONVERSION_ORDER: &[Name; 3] = &[dunder::LE, dunder::GT, dunder::LT];

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn synthesize_rich_cmp(&self, cls: &Class, cmp: &Name) -> ClassSynthesizedField {
        let conversion_order = if cmp == &dunder::LT {
            LT_CONVERSION_ORDER
        } else if cmp == &dunder::GT {
            GT_CONVERSION_ORDER
        } else if cmp == &dunder::LE {
            LE_CONVERSION_ORDER
        } else if cmp == &dunder::GE {
            GE_CONVERSION_ORDER
        } else {
            unreachable!("Unexpected rich comparison method: {}", cmp);
        };
        // The first field in the conversion order is the one that we will use to synthesize the method.
        for other_cmp in conversion_order {
            if let Some(other_cmp_field) = self
                .get_non_synthesized_class_member_and_defining_class(cls, other_cmp)
                .filter(|field| !field.defining_class.is_builtin("object"))
            {
                let ty = other_cmp_field.value.as_named_tuple_type();
                return ClassSynthesizedField::new(ty);
            }
        }
        unreachable!("No rich comparison method found for {}", cmp);
    }

    pub fn get_total_ordering_synthesized_fields(
        &self,
        errors: &ErrorCollector,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        if !metadata.is_total_ordering() {
            return None;
        }
        // The class must have one of the rich comparison dunder methods defined
        if !dunder::RICH_CMPS_TOTAL_ORDERING.iter().any(|cmp| {
            self.get_non_synthesized_class_member_and_defining_class(cls, cmp)
                .is_some_and(|field| !field.defining_class.is_builtin("object"))
        }) {
            let total_ordering_metadata = metadata.total_ordering_metadata().unwrap();
            self.error(
                errors,
                total_ordering_metadata.location,
                ErrorKind::MissingAttribute,
                format!(
                    "Class `{}` must define at least one of the rich comparison methods.",
                    cls.name()
                ),
            );
            return None;
        }
        let rich_cmps_to_synthesize: Vec<_> = dunder::RICH_CMPS_TOTAL_ORDERING
            .iter()
            .filter(|cmp| {
                self.get_non_synthesized_class_member_and_defining_class(cls, cmp)
                    .is_none_or(|field| field.defining_class.is_builtin("object"))
            })
            .collect();
        let mut fields = SmallMap::with_capacity(rich_cmps_to_synthesize.len());
        for cmp in rich_cmps_to_synthesize {
            let synthesized_field = self.synthesize_rich_cmp(cls, cmp);
            fields.insert(cmp.clone(), synthesized_field);
        }
        Some(ClassSynthesizedFields::new(fields))
    }
}
