/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;

const META: Name = Name::new_static("Meta");
const MODEL: Name = Name::new_static("model");
const CREATE: Name = Name::new_static("create");
const BUILD: Name = Name::new_static("build");
const CREATE_BATCH: Name = Name::new_static("create_batch");
const BUILD_BATCH: Name = Name::new_static("build_batch");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Synthesize `create`, `build`, `create_batch`, and `build_batch` on
    /// factory-boy `DjangoModelFactory` subclasses, returning the model type
    /// from `class Meta: model = X`.
    pub fn get_factory_boy_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        if !metadata.is_factory_boy_factory() {
            return None;
        }
        let model_type = self.get_factory_model_type(cls)?;
        let list_model_type = self
            .heap
            .mk_class_type(self.stdlib.list(model_type.clone()));

        let mut fields = SmallMap::new();
        fields.insert(
            CREATE,
            self.factory_classmethod(cls, &CREATE, vec![], model_type.clone()),
        );
        fields.insert(
            BUILD,
            self.factory_classmethod(cls, &BUILD, vec![], model_type),
        );
        let size_param = Param::Pos(
            Name::new_static("size"),
            self.heap.mk_class_type(self.stdlib.int().clone()),
            Required::Required,
        );
        fields.insert(
            CREATE_BATCH,
            self.factory_classmethod(
                cls,
                &CREATE_BATCH,
                vec![size_param.clone()],
                list_model_type.clone(),
            ),
        );
        fields.insert(
            BUILD_BATCH,
            self.factory_classmethod(cls, &BUILD_BATCH, vec![size_param], list_model_type),
        );
        Some(ClassSynthesizedFields::new(fields))
    }

    /// Extract the model type from `class Meta: model = X`.
    fn get_factory_model_type(&self, cls: &Class) -> Option<Type> {
        let meta_field = self.get_class_member(cls, &META)?;
        let meta_class = match meta_field.ty() {
            Type::ClassDef(cls) => cls,
            _ => return None,
        };
        let model_field = self.get_class_member(&meta_class, &MODEL)?;
        // `model = User` assigns the class object, so the type is ClassDef(User).
        // Convert to the instance type.
        match model_field.ty() {
            Type::ClassDef(class) => Some(self.instantiate(&class)),
            _ => None,
        }
    }

    /// Synthesize a classmethod with the given leading params, plus `**kwargs: Any`.
    fn factory_classmethod(
        &self,
        cls: &Class,
        name: &Name,
        mut params: Vec<Param>,
        ret_type: Type,
    ) -> ClassSynthesizedField {
        params.push(Param::Kwargs(None, self.heap.mk_any_implicit()));
        ClassSynthesizedField::new_classvar(self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), ret_type),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), name.clone(), None),
        }))
    }
}
