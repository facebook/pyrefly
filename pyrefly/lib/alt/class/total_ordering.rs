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
use crate::binding::binding::KeyClassField;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;

// https://github.com/python/cpython/blob/a8ec511900d0d84cffbb4ee6419c9a790d131129/Lib/functools.py#L173
// conversion order of rich comparison methods:
const LT_CONVERSION_ORDER: &[Name; 3] = &[dunder::GT, dunder::LE, dunder::GE];
const GT_CONVERSION_ORDER: &[Name; 3] = &[dunder::LT, dunder::GE, dunder::LE];
const LE_CONVERSION_ORDER: &[Name; 3] = &[dunder::GE, dunder::LT, dunder::GT];
const GE_CONVERSION_ORDER: &[Name; 3] = &[dunder::LE, dunder::GT, dunder::LT];

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
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
        let class_fields = self.get_class_fields(cls);
        for other_cmp in conversion_order {
            let has_field = class_fields.is_some_and(|f| f.contains(other_cmp));
            if has_field
                && let Some(other_cmp_field) =
                    self.get_from_class(cls, &KeyClassField(cls.index(), other_cmp.clone()))
            {
                let ty = other_cmp_field.as_named_tuple_type();
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
        let class_fields = self.get_class_fields(cls);
        // The class must have one of the rich comparison dunder methods defined
        if !class_fields.is_some_and(|f| {
            f.names().any(|f| {
                *f == dunder::LT || *f == dunder::LE || *f == dunder::GT || *f == dunder::GE
            })
        }) {
            let total_ordering_metadata = metadata.total_ordering_metadata().unwrap();
            self.error(
                errors,
                total_ordering_metadata.location,
                ErrorInfo::Kind(ErrorKind::MissingAttribute),
                format!(
                    "Class `{}` must define at least one of the rich comparison methods.",
                    cls.name()
                ),
            );
            return None;
        }
        let rich_cmps_to_synthesize: Vec<_> = dunder::RICH_CMPS_TOTAL_ORDERING
            .iter()
            .filter(|cmp| !class_fields.is_some_and(|f| f.contains(cmp)))
            .collect();
        let mut fields = SmallMap::with_capacity(rich_cmps_to_synthesize.len());
        for cmp in rich_cmps_to_synthesize {
            let synthesized_field = self.synthesize_rich_cmp(cls, cmp);
            fields.insert(cmp.clone(), synthesized_field);
        }
        Some(ClassSynthesizedFields::new(fields))
    }
}
