/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
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
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_named_tuple_elements(&self, cls: &Class, errors: &ErrorCollector) -> SmallSet<Name> {
        let Some(class_fields) = self.get_class_fields(cls) else {
            return SmallSet::new();
        };
        let mut elements = Vec::with_capacity(class_fields.len());
        for name in class_fields.names() {
            if !class_fields.is_field_annotated(name) {
                continue;
            }
            if let Some(range) = class_fields.field_decl_range(name) {
                elements.push((name.clone(), range));
            }
        }
        elements.sort_by_key(|e: &(Name, ruff_text_size::TextRange)| e.1.start());
        let mut has_seen_default: bool = false;
        for (name, range) in &elements {
            let has_default = class_fields.is_field_initialized_on_class(name);
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
        // If the namedtuple has dynamic fields, we can't know the element types statically.
        // Return None so that tuple indexing falls back to regular tuple behavior.
        if named_tuple_metadata.has_dynamic_fields {
            return None;
        }
        Some(
            named_tuple_metadata
                .elements
                .iter()
                .map(|name| {
                    self.resolve_named_tuple_element(cls, name)
                        .unwrap_or_else(|| self.heap.mk_any_implicit())
                })
                .collect(),
        )
    }

    fn get_named_tuple_field_params(&self, cls: &Class, elements: &SmallSet<Name>) -> Vec<Param> {
        elements
            .iter()
            .map(|name| {
                let (ty, required) = match self.get_non_synthesized_class_member(cls, name) {
                    None => (self.heap.mk_any_implicit(), Required::Required),
                    Some(c) => (c.as_named_tuple_type(), c.as_named_tuple_requiredness()),
                };
                Param::Pos(name.clone(), ty, required)
            })
            .collect()
    }

    /// Synthesize `__new__` for a namedtuple. When `has_dynamic_fields` is true,
    /// the field list couldn't be fully resolved statically (e.g. starred
    /// unpacking), so we use `*args, **kwargs` instead of specific field params.
    /// We can't know the positions of the known fields relative to the dynamic
    /// ones, so requiring them as leading positional params would be incorrect.
    fn get_named_tuple_new(
        &self,
        cls: &Class,
        elements: &SmallSet<Name>,
        has_dynamic_fields: bool,
    ) -> ClassSynthesizedField {
        let mut params = vec![Param::Pos(
            Name::new_static("_cls"),
            self.heap
                .mk_type_form(self.heap.mk_self_type(self.as_class_type_unchecked(cls))),
            Required::Required,
        )];
        if has_dynamic_fields {
            // The known fields are interleaved with dynamically-resolved fields
            // at unknown positions, so we can't synthesize accurate positional
            // params. Accept everything and rely on runtime behavior.
            params.push(Param::VarArg(None, self.heap.mk_any_implicit()));
            params.push(Param::Kwargs(None, self.heap.mk_any_implicit()));
        } else {
            params.extend(self.get_named_tuple_field_params(cls, elements));
        }
        let ty = self.heap.mk_function(Function {
            signature: Callable::list(
                ParamList::new(params),
                self.heap.mk_self_type(self.as_class_type_unchecked(cls)),
            ),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::NEW, None),
        });
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_init(&self, cls: &Class) -> ClassSynthesizedField {
        let params = vec![
            self.class_self_param(cls, false),
            // NamedTuple.__init__ accepts any args at runtime; rely on __new__ for checking.
            Param::VarArg(None, self.heap.mk_any_implicit()),
            Param::Kwargs(None, self.heap.mk_any_implicit()),
        ];
        let ty = self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), self.heap.mk_none()),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::INIT, None),
        });
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
                    None => self.heap.mk_any_implicit(),
                    Some(c) => c.as_named_tuple_type(),
                },
            )
            .collect();
        let ty = self.heap.mk_function(Function {
            signature: Callable::list(
                ParamList::new(params),
                self.heap
                    .mk_class_type(self.stdlib.iterable(self.unions(element_types))),
            ),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::ITER, None),
        });
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_match_args(&self, elements: &SmallSet<Name>) -> ClassSynthesizedField {
        let ty = self.heap.mk_concrete_tuple(
            elements
                .iter()
                .map(|e| Lit::Str(e.as_str().into()).to_implicit_type())
                .collect(),
        );
        ClassSynthesizedField::new(ty)
    }

    pub fn get_named_tuple_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let named_tuple = metadata.named_tuple_metadata()?;
        Some(ClassSynthesizedFields::new(smallmap! {
            dunder::NEW => self.get_named_tuple_new(cls, &named_tuple.elements, named_tuple.has_dynamic_fields),
            dunder::INIT => self.get_named_tuple_init(cls),
            dunder::MATCH_ARGS => self.get_named_tuple_match_args(&named_tuple.elements),
            dunder::ITER => self.get_named_tuple_iter(cls, &named_tuple.elements)
        }))
    }
}
