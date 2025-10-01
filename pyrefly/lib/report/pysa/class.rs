/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::ops::Not;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::class::Class;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use serde::Serialize;
use starlark_map::Hashed;

use crate::alt::answers::Answers;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::bindings::Bindings;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::scope::get_scope_parent;
use crate::report::pysa::types::PysaType;

/// Represents a unique identifier for a class **within a module**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ClassId(u32);

impl ClassId {
    pub fn from_class(class: &Class) -> ClassId {
        ClassId(class.index().0)
    }

    #[cfg(test)]
    pub fn from_int(id: u32) -> ClassId {
        ClassId(id)
    }

    pub fn to_int(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClassRef {
    pub module_id: ModuleId,
    pub module_name: ModuleName, // For debugging purposes only. Reader should use the module id.
    pub class_id: ClassId,
    pub class_name: String, // For debugging purposes only. Reader should use the class id.
}

impl ClassRef {
    pub fn from_class(class: &Class, module_ids: &ModuleIds) -> ClassRef {
        ClassRef {
            module_id: module_ids
                .get(ModuleKey::from_module(class.module()))
                .unwrap(),
            module_name: class.module_name(),
            class_id: ClassId::from_class(class),
            class_name: class.qname().id().to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum PysaClassMro {
    Resolved(Vec<ClassRef>),
    Cyclic,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PysaClassField {
    #[serde(rename = "type")]
    pub type_: PysaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_annotation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<PysaLocation>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassDefinition {
    pub class_id: ClassId,
    pub name: String,
    pub bases: Vec<ClassRef>,
    pub mro: PysaClassMro,
    pub parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_synthesized: bool, // True if this class was synthesized (e.g., from namedtuple), false if from actual `class X:` statement
    pub fields: HashMap<String, PysaClassField>,
}

impl ClassDefinition {
    #[cfg(test)]
    pub fn with_bases(mut self, bases: Vec<ClassRef>) -> Self {
        self.bases = bases;
        self
    }

    #[cfg(test)]
    pub fn with_mro(mut self, mro: PysaClassMro) -> Self {
        self.mro = mro;
        self
    }

    #[cfg(test)]
    pub fn with_fields(mut self, fields: HashMap<String, PysaClassField>) -> Self {
        self.fields = fields;
        self
    }
}

pub fn get_all_classes(bindings: &Bindings, answers: &Answers) -> impl Iterator<Item = Class> {
    bindings
        .keys::<KeyClass>()
        .map(|idx| answers.get_idx(idx).unwrap().0.dupe().unwrap())
}

pub fn get_class_field(
    class: &Class,
    field: &Name,
    context: &ModuleContext,
) -> Option<Arc<ClassField>> {
    context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
            solver.get_field_from_current_class_only(class, field)
        })
        .unwrap()
}

pub fn get_class_field_declaration<'a>(
    class: &'a Class,
    field: &'a Name,
    context: &'a ModuleContext,
) -> Option<&'a BindingClassField> {
    let key_class_field = KeyClassField(class.index(), field.clone());
    // We use `key_to_idx_hashed_opt` below because the key might not be valid (could be a synthesized field).
    context
        .bindings
        .key_to_idx_hashed_opt(Hashed::new(&key_class_field))
        .map(|idx| context.bindings.get(idx))
}

pub fn get_class_mro(class: &Class, bindings: &Bindings, answers: &Answers) -> Arc<ClassMro> {
    answers
        .get_idx(bindings.key_to_idx(&KeyClassMro(class.index())))
        .unwrap()
}

pub fn export_class_fields(
    class: &Class,
    context: &ModuleContext,
) -> HashMap<String, PysaClassField> {
    class
        .fields()
        .filter_map(|name| get_class_field(class, name, context).map(|field| (name, field)))
        .filter_map(|(name, field)| {
            let field_binding = get_class_field_declaration(class, name, context);

            let explicit_annotation = match field_binding {
                Some(BindingClassField {
                    definition: ClassFieldDefinition::DeclaredByAnnotation { annotation },
                    ..
                }) => Some(*annotation),
                Some(BindingClassField {
                    definition: ClassFieldDefinition::AssignedInBody { annotation, .. },
                    ..
                }) => *annotation,
                Some(BindingClassField {
                    definition: ClassFieldDefinition::DefinedInMethod { annotation, .. },
                    ..
                }) => *annotation,
                _ => None,
            }
            .map(|idx| context.bindings.idx_to_key(idx))
            .and_then(|key_annotation| match key_annotation {
                // We want to export the annotation as it is in the source code.
                // We cannot use the answer for `key_annotation` (which wraps a `Type`),
                // because it contains a normalized type where some elements have
                // been stripped out (most notably, `typing.Annotated`).
                KeyAnnotation::Annotation(identifier) => {
                    // `Ast::locate_node` returns all covering AST nodes, from innermost to outermost.
                    // The innermost will be the Name node, so we need the second node.
                    match Ast::locate_node(&context.ast, identifier.range().start()).get(1) {
                        Some(AnyNodeRef::StmtAnnAssign(assign)) => Some(
                            context
                                .module_info
                                .code_at(assign.annotation.range())
                                .to_owned(),
                        ),
                        _ => None,
                    }
                }
                KeyAnnotation::AttrAnnotation(range) => {
                    Some(context.module_info.code_at(*range).to_owned())
                }
                _ => None,
            });

            match field_binding {
                Some(BindingClassField {
                    definition: ClassFieldDefinition::MethodLike { .. },
                    ..
                }) => {
                    // Exclude fields that are functions definitions, because they are already exported in `function_definitions`.
                    None
                }
                Some(BindingClassField { range, .. }) => Some((
                    name.to_string(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: Some(PysaLocation::new(
                            context.module_info.display_range(*range),
                        )),
                    },
                )),
                _ => Some((
                    name.to_string(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: None,
                    },
                )),
            }
        })
        .collect()
}

pub fn export_all_classes(context: &ModuleContext) -> HashMap<PysaLocation, ClassDefinition> {
    let mut class_definitions = HashMap::new();

    for class_idx in context.bindings.keys::<KeyClass>() {
        let class = context
            .answers
            .get_idx(class_idx)
            .unwrap()
            .0
            .dupe()
            .unwrap();
        let display_range = context.module_info.display_range(class.qname().range());
        let class_index = class.index();
        let parent = get_scope_parent(&context.ast, &context.module_info, class.qname().range());
        let metadata = context
            .answers
            .get_idx(context.bindings.key_to_idx(&KeyClassMetadata(class_index)))
            .unwrap();

        let is_synthesized = match context.bindings.get(class_idx) {
            BindingClass::FunctionalClassDef(_, _, _, _) => true,
            BindingClass::ClassDef(_) => false,
        };

        let fields = export_class_fields(&class, context);

        let bases = metadata
            .base_class_objects()
            .iter()
            .map(|base_class| ClassRef::from_class(base_class, context.module_ids))
            .collect::<Vec<_>>();

        let mro = match &*get_class_mro(&class, &context.bindings, &context.answers) {
            ClassMro::Resolved(mro) => PysaClassMro::Resolved(
                mro.iter()
                    .map(|class_type| {
                        ClassRef::from_class(class_type.class_object(), context.module_ids)
                    })
                    .collect(),
            ),
            ClassMro::Cyclic => PysaClassMro::Cyclic,
        };

        let class_definition = ClassDefinition {
            class_id: ClassId::from_class(&class),
            name: class.qname().id().to_string(),
            parent,
            bases,
            mro,
            is_synthesized,
            fields,
        };

        assert!(
            class_definitions
                .insert(PysaLocation::new(display_range), class_definition)
                .is_none(),
            "Found class definitions with the same location"
        );
    }

    class_definitions
}
