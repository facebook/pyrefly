/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::dunder;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;
use starlark_map::smallmap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_named_tuple_elements(&self, cls: &Class, errors: &ErrorCollector) -> SmallSet<Name> {
        let mut elements = Vec::new();
        for name in cls.fields() {
            if !cls.is_field_annotated(name) {
                continue;
            }
            if let Some(range) = cls.field_decl_range(name) {
                elements.push((name.clone(), range));
            }
        }
        elements.sort_by_key(|e: &(Name, ruff_text_size::TextRange)| e.1.start());
        let mut has_seen_default: bool = false;
        for (name, range) in &elements {
            let has_default = cls.is_field_initialized_on_class(name);
            if !has_default && has_seen_default {
                self.error(
                    errors,
                    *range,
                    ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                    format!(
                        "NamedTuple field '{name}' without a default may not follow NamedTuple field with a default"
                    ),
                );
            }
            if has_default {
                has_seen_default = true;
            }
        }
        elements.into_iter().map(|(name, _)| name).collect()
    }

    pub(crate) fn named_tuple_element_types(&self, cls: &ClassType) -> Option<Vec<Type>> {
        let class_metadata = self.get_metadata_for_class(cls.class_object());
        let named_tuple_metadata = class_metadata.named_tuple_metadata()?;
        Some(
            named_tuple_metadata
                .elements
                .iter()
                .map(|name| {
                    self.resolve_named_tuple_element(cls.clone(), name)
                        .unwrap_or_else(Type::any_implicit)
                })
                .collect(),
        )
    }

    fn get_named_tuple_field_params(&self, cls: &Class, elements: &SmallSet<Name>) -> Vec<Param> {
        elements
            .iter()
            .map(|name| {
                let (ty, required) = match self.get_non_synthesized_class_member(cls, name) {
                    None => (Type::any_implicit(), Required::Required),
                    Some(c) => (c.as_named_tuple_type(), c.as_named_tuple_requiredness()),
                };
                Param::Pos(name.clone(), ty, required)
            })
            .collect()
    }

    fn get_named_tuple_new(&self, cls: &Class, elements: &SmallSet<Name>) -> ClassSynthesizedField {
        let mut params = vec![Param::Pos(
            Name::new_static("cls"),
            Type::type_form(Type::SelfType(self.as_class_type_unchecked(cls))),
            Required::Required,
        )];
        params.extend(self.get_named_tuple_field_params(cls, elements));
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(
                ParamList::new(params),
                Type::SelfType(self.as_class_type_unchecked(cls)),
            ),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::NEW),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_init(
        &self,
        cls: &Class,
        elements: &SmallSet<Name>,
    ) -> ClassSynthesizedField {
        let mut params = vec![self.class_self_param(cls, false)];
        params.extend(self.get_named_tuple_field_params(cls, elements));
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), Type::None),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_iter(
        &self,
        cls: &Class,
        elements: &SmallSet<Name>,
    ) -> ClassSynthesizedField {
        let params = vec![self.class_self_param(cls, false)];
        let element_types: Vec<Type> = elements
            .iter()
            .map(
                |name| match self.get_non_synthesized_class_member(cls, name) {
                    None => Type::any_implicit(),
                    Some(c) => c.as_named_tuple_type(),
                },
            )
            .collect();
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(
                ParamList::new(params),
                Type::ClassType(self.stdlib.iterable(self.unions(element_types))),
            ),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::ITER),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_match_args(&self, elements: &SmallSet<Name>) -> ClassSynthesizedField {
        let ty = Type::Tuple(Tuple::Concrete(
            elements
                .iter()
                .map(|e| Type::Literal(Lit::Str(e.as_str().into())))
                .collect(),
        ));
        ClassSynthesizedField::new(ty)
    }

    pub fn get_named_tuple_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let named_tuple = metadata.named_tuple_metadata()?;
        Some(ClassSynthesizedFields::new(smallmap! {
            dunder::NEW => self.get_named_tuple_new(cls, &named_tuple.elements),
            dunder::INIT => self.get_named_tuple_init(cls, &named_tuple.elements),
            dunder::MATCH_ARGS => self.get_named_tuple_match_args(&named_tuple.elements),
            dunder::ITER => self.get_named_tuple_iter(cls, &named_tuple.elements)
        }))
    }
}
