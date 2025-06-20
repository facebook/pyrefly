use core::panic;

use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::dunder;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::types::Type;

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
        for other_cmp in conversion_order {
            let other_cmp_field = cls.fields().find(|f| **f == *other_cmp);
            if other_cmp_field.is_some() {
                // FIXME: We should use the type from `other_cmp_field` instead of `cls_type`.
                // However, here we use the type of the class itself, which is not always correct.
                let cls_type = self.instantiate(cls);
                let self_param = self.class_self_param(cls, false);
                let other_param =
                    Param::Pos(Name::new_static("other"), cls_type, Required::Required);
                return ClassSynthesizedField::new(Type::Function(Box::new(Function {
                    signature: Callable {
                        params: Params::List(ParamList::new(vec![self_param, other_param])),
                        ret: self.stdlib.bool().clone().to_type(),
                    },
                    metadata: FuncMetadata::def(
                        self.module_info().name(),
                        cls.name().clone(),
                        cmp.clone(),
                    ),
                })));
            }
        }
        unreachable!("Rich comparison method not found in conversion order");
    }

    pub fn get_total_ordering_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        if !metadata.is_total_ordering() {
            return None;
        }
        // The class must have one of the rich comparison dunder methods defined
        if !cls
            .fields()
            .any(|f| *f == dunder::LT || *f == dunder::LE || *f == dunder::GT || *f == dunder::GE)
        {
            // TODO: raise an error properly.
            panic!("Class does not define any rich comparison methods");
        }
        let rich_cmps_to_synthesize: Vec<_> = dunder::RICH_CMPS_TOTAL_ORDERING
            .iter()
            .filter(|cmp| !cls.contains(cmp))
            .collect();
        let mut fields = SmallMap::new();
        for cmp in rich_cmps_to_synthesize {
            let synthesized_field = self.synthesize_rich_cmp(cls, cmp);
            fields.insert(cmp.clone(), synthesized_field);
        }
        Some(ClassSynthesizedFields::new(fields))
    }
}
