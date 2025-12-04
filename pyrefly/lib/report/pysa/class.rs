/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Not;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use serde::Serialize;
use serde::ser::SerializeStruct;
use starlark_map::Hashed;

use crate::alt::class::class_field::ClassField;
use crate::alt::class::class_field::WithDefiningClass;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::resolve_decorator_callees;
use crate::report::pysa::collect::CollectNoDuplicateKeys;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::scope::get_scope_parent;
use crate::report::pysa::types::PysaType;
use crate::report::pysa::types::is_callable_like;

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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassRef {
    pub module_id: ModuleId,
    pub class_id: ClassId,
    pub class: Class,
}

impl ClassRef {
    pub fn from_class(class: &Class, module_ids: &ModuleIds) -> ClassRef {
        ClassRef {
            module_id: module_ids
                .get(ModuleKey::from_module(class.module()))
                .unwrap(),
            class_id: ClassId::from_class(class),
            class: class.clone(),
        }
    }
}

impl std::fmt::Debug for ClassRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassRef")
            .field("module_id", &self.module_id)
            .field("class_id", &self.class_id)
            .field("module_name", &self.class.module_name().as_str())
            .field("class_name", &self.class.name().as_str())
            // We don't print `class` because it's way too large.
            .finish_non_exhaustive()
    }
}

impl Serialize for ClassRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ClassRef", 4)?;
        state.serialize_field("module_id", &self.module_id)?;
        // Exported for debugging purposes only
        state.serialize_field("module_name", &self.class.module_name())?;
        state.serialize_field("class_id", &self.class_id)?;
        // Exported for debugging purposes only
        state.serialize_field("class_name", &self.class.name())?;
        state.end()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum PysaClassMro {
    Resolved(Vec<ClassRef>),
    Cyclic,
}

/// See `pyrefly::binding::ClassFieldDeclaration`
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum PysaClassFieldDeclaration {
    DeclaredByAnnotation,
    DeclaredWithoutAnnotation,
    AssignedInBody,
    DefinedWithoutAssign,
    DefinedInMethod,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PysaClassField {
    #[serde(rename = "type")]
    pub type_: PysaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_annotation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<PysaLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declaration_kind: Option<PysaClassFieldDeclaration>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassDefinition {
    pub class_id: ClassId,
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bases: Vec<ClassRef>,
    pub mro: PysaClassMro,
    pub parent: ScopeParent,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_synthesized: bool, // True if this class was synthesized (e.g., from namedtuple), false if from actual `class X:` statement
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_dataclass: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_named_tuple: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_typed_dict: bool,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub fields: HashMap<Name, PysaClassField>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub decorator_callees: HashMap<PysaLocation, Vec<Target<FunctionRef>>>,
}

impl PysaClassFieldDeclaration {
    fn from(definition: &ClassFieldDefinition) -> Option<Self> {
        match definition {
            ClassFieldDefinition::DeclaredByAnnotation { .. } => {
                Some(PysaClassFieldDeclaration::DeclaredByAnnotation)
            }
            ClassFieldDefinition::DeclaredWithoutAnnotation => {
                Some(PysaClassFieldDeclaration::DeclaredWithoutAnnotation)
            }
            ClassFieldDefinition::AssignedInBody { .. } => {
                Some(PysaClassFieldDeclaration::AssignedInBody)
            }
            ClassFieldDefinition::NestedClass { .. }
            | ClassFieldDefinition::DefinedWithoutAssign { .. } => {
                // TODO: we handled NestedClass this way for backward
                // compatibility after a Pyrefly refactor. Should it be modeled
                // differently?
                Some(PysaClassFieldDeclaration::DefinedWithoutAssign)
            }
            ClassFieldDefinition::DefinedInMethod { .. } => {
                Some(PysaClassFieldDeclaration::DefinedInMethod)
            }
            ClassFieldDefinition::MethodLike { .. } => None,
        }
    }
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
    pub fn with_fields(mut self, fields: HashMap<Name, PysaClassField>) -> Self {
        self.fields = fields;
        self
    }

    #[cfg(test)]
    pub fn with_decorator_callees(
        mut self,
        decorator_callees: HashMap<PysaLocation, Vec<Target<FunctionRef>>>,
    ) -> Self {
        self.decorator_callees = decorator_callees;
        self
    }

    #[cfg(test)]
    pub fn with_is_dataclass(mut self, is_dataclass: bool) -> Self {
        self.is_dataclass = is_dataclass;
        self
    }
}

pub fn get_all_classes(context: &ModuleContext) -> impl Iterator<Item = Class> {
    context
        .bindings
        .keys::<KeyClass>()
        .map(|idx| context.answers.get_idx(idx).unwrap().0.dupe().unwrap())
}

pub fn get_class_field_from_current_class_only(
    class: &Class,
    field_name: &Name,
    context: &ModuleContext,
) -> Option<Arc<ClassField>> {
    context
        .transaction
        .ad_hoc_solve(&context.handle, |solver| {
            solver.get_field_from_current_class_only(class, field_name)
        })
        .unwrap()
}

pub fn get_super_class_member(
    class: &Class,
    field_name: &Name,
    start_lookup_cls: Option<&ClassType>,
    context: &ModuleContext,
) -> Option<WithDefiningClass<Arc<ClassField>>> {
    context
        .transaction
        .ad_hoc_solve(&context.handle, |solver| {
            solver.get_super_class_member(class, start_lookup_cls, field_name)
        })
        .flatten()
}

pub fn get_context_from_class<'a>(
    class: &'a Class,
    context: &'a ModuleContext<'a>,
) -> ModuleContext<'a> {
    let handle = Handle::new(
        class.module_name(),
        class.module_path().clone(),
        context.handle.sys_info().clone(),
    );
    ModuleContext::create(handle, context.transaction, context.module_ids).unwrap()
}

pub fn get_class_field_declaration<'a>(
    class: &Class,
    field_name: &Name,
    context: &'a ModuleContext,
) -> Option<&'a BindingClassField> {
    assert_eq!(class.module(), &context.module_info);
    let key_class_field = KeyClassField(class.index(), field_name.clone());
    // We use `key_to_idx_hashed_opt` below because the key might not be valid (could be a synthesized field).
    context
        .bindings
        .key_to_idx_hashed_opt(Hashed::new(&key_class_field))
        .map(|idx| context.bindings.get(idx))
}

pub fn get_class_mro(class: &Class, context: &ModuleContext) -> Arc<ClassMro> {
    assert_eq!(class.module(), &context.module_info);
    context
        .answers
        .get_idx(context.bindings.key_to_idx(&KeyClassMro(class.index())))
        .unwrap()
}

pub fn get_class_fields<'a>(
    class: &'a Class,
    context: &'a ModuleContext<'a>,
) -> impl Iterator<Item = (Cow<'a, Name>, Arc<ClassField>)> {
    let regular_fields = class.fields().filter_map(|name| {
        get_class_field_from_current_class_only(class, name, context)
            .map(|field| (Cow::Borrowed(name), field))
    });

    let synthesized_fields_idx = context
        .bindings
        .key_to_idx(&KeyClassSynthesizedFields(class.index()));
    let synthesized_fields = context.answers.get_idx(synthesized_fields_idx).unwrap();
    let synthesized_fields = synthesized_fields
        .fields()
        .filter(|(name, _)| !class.contains(name))
        .map(|(name, field)| (Cow::Owned(name.clone()), field.inner.dupe()))
        // Required by the borrow checker.
        // This is fine since the amount of synthesized fields should be small.
        .collect::<Vec<_>>()
        .into_iter();

    regular_fields.chain(synthesized_fields)
}

pub fn export_class_fields(
    class: &Class,
    context: &ModuleContext,
) -> HashMap<Name, PysaClassField> {
    assert_eq!(class.module(), &context.module_info);
    get_class_fields(class, context)
        .filter(|(_, field)| !is_callable_like(&field.ty()))
        .filter_map(|(name, field)| {
            let field_binding = get_class_field_declaration(class, &name, context);

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
                Some(BindingClassField {
                    range, definition, ..
                }) => Some((
                    name.into_owned(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: Some(PysaLocation::new(
                            context.module_info.display_range(*range),
                        )),
                        declaration_kind: PysaClassFieldDeclaration::from(definition),
                    },
                )),
                _ => Some((
                    name.into_owned(),
                    PysaClassField {
                        type_: PysaType::from_type(&field.ty(), context),
                        explicit_annotation,
                        location: None,
                        declaration_kind: None,
                    },
                )),
            }
        })
        .collect_no_duplicate_keys()
        .expect("Found duplicate class fields")
}

fn find_definition_ast<'a>(
    class: &Class,
    context: &'a ModuleContext<'a>,
) -> Option<&'a StmtClassDef> {
    assert_eq!(class.module(), &context.module_info);
    Ast::locate_node(&context.ast, class.qname().range().start())
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::StmtClassDef(stmt) if stmt.name.range == class.qname().range() => {
                Some(*stmt)
            }
            _ => None,
        })
}

fn get_decorator_callees(
    class: &Class,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    context: &ModuleContext,
) -> HashMap<PysaLocation, Vec<Target<FunctionRef>>> {
    assert_eq!(class.module(), &context.module_info);
    if let Some(class_def) = find_definition_ast(class, context) {
        resolve_decorator_callees(
            &class_def.decorator_list,
            function_base_definitions,
            context,
        )
    } else {
        HashMap::new()
    }
}

pub fn export_all_classes(
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    context: &ModuleContext,
) -> HashMap<PysaLocation, ClassDefinition> {
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

        let mro = match &*get_class_mro(&class, context) {
            ClassMro::Resolved(mro) => PysaClassMro::Resolved(
                mro.iter()
                    .map(|class_type| {
                        ClassRef::from_class(class_type.class_object(), context.module_ids)
                    })
                    .collect::<Vec<_>>(),
            ),
            ClassMro::Cyclic => PysaClassMro::Cyclic,
        };

        let decorator_callees = get_decorator_callees(&class, function_base_definitions, context);

        let class_definition = ClassDefinition {
            class_id: ClassId::from_class(&class),
            name: class.qname().id().to_string(),
            parent,
            bases,
            mro,
            is_synthesized,
            is_dataclass: metadata.dataclass_metadata().is_some(),
            is_named_tuple: metadata.named_tuple_metadata().is_some(),
            is_typed_dict: metadata.typed_dict_metadata().is_some(),
            fields,
            decorator_callees,
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
