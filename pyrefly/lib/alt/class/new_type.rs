/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::dunder;
use ruff_python_ast::name::Name;
use starlark_map::smallmap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn base_type_for_newtype(&self, class: &Class, base_class: &Class) -> Type {
        if base_class.is_builtin("tuple")
            && let Some(tuple_ancestor) = self.get_base_types_for_class(class).tuple_ancestor()
        {
            // In order to make `__new__` and `__init__` accept only the exact right shape
            // of tuple, we have to special case the scenario where the NewType wraps a tuple -
            // we want to provide use the raw tuple type rather than the `tuple` class.
            Type::Tuple(tuple_ancestor.clone())
        } else {
            self.promote_nontypeddict_silently_to_classtype(base_class)
                .to_type()
        }
    }

    fn get_new_type_init(&self, cls: &Class, base_type: Type) -> ClassSynthesizedField {
        let params = vec![
            self.class_self_param(cls, false),
            Param::Pos(Name::new_static("_x"), base_type, Required::Required),
        ];
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), self.instantiate(cls)),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_new_type_new(&self, cls: &Class, base_type: Type) -> ClassSynthesizedField {
        let params = vec![
            Param::Pos(
                Name::new_static("cls"),
                Type::type_form(self.instantiate(cls)),
                Required::Required,
            ),
            Param::Pos(Name::new_static("_x"), base_type, Required::Required),
        ];
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), self.instantiate(cls)),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::NEW),
        }));
        ClassSynthesizedField::new(ty)
    }

    pub fn get_new_type_synthesized_fields(&self, cls: &Class) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);

        let base_classes = metadata.base_class_objects();
        let is_new_type = metadata.is_new_type();

        if is_new_type && base_classes.len() == 1 {
            let base_class = &base_classes[0];
            let base_type = self.base_type_for_newtype(cls, base_class);
            Some(ClassSynthesizedFields::new(smallmap! {
                dunder::NEW => self.get_new_type_new(cls, base_type.clone()),
                dunder::INIT => self.get_new_type_init(cls, base_type),
            }))
        } else {
            None
        }
    }
}
