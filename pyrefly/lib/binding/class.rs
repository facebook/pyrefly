/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;
use std::sync::LazyLock;

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::prelude::SliceExt;
use regex::Regex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::class::base_class::BaseClass;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::BindingClassMetadata;
use crate::binding::binding::BindingClassMro;
use crate::binding::binding::BindingClassSynthesizedFields;
use crate::binding::binding::BindingTParams;
use crate::binding::binding::BindingVariance;
use crate::binding::binding::ClassBinding;
use crate::binding::binding::ClassFieldInitialValue;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyTParams;
use crate::binding::binding::KeyVariance;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::CurrentIdx;
use crate::binding::bindings::LegacyTParamBuilder;
use crate::binding::scope::ClassIndices;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::InstanceAttribute;
use crate::binding::scope::MethodThatSetsAttr;
use crate::binding::scope::Scope;
use crate::binding::scope::ScopeKind;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::class::ClassDefIndex;
use crate::types::class::ClassFieldProperties;
use crate::types::special_form::SpecialForm;
use crate::types::types::Type;

enum IllegalIdentifierHandling {
    Error,
    Allow,
    Rename,
}

#[derive(Eq, PartialEq)]
enum SynthesizedClassKind {
    Enum,
    TypedDict,
    NamedTuple,
    NewType,
}

impl<'a> BindingsBuilder<'a> {
    fn def_index(&mut self) -> ClassDefIndex {
        let res = ClassDefIndex(self.class_count);
        self.class_count += 1;
        res
    }

    fn class_object_and_indices(&mut self, class_name: &Identifier) -> (CurrentIdx, ClassIndices) {
        let def_index = self.def_index();
        let class_indices = ClassIndices {
            def_index,
            class_idx: self.idx_for_promise(KeyClass(ShortIdentifier::new(class_name))),
            metadata_idx: self.idx_for_promise(KeyClassMetadata(def_index)),
            mro_idx: self.idx_for_promise(KeyClassMro(def_index)),
            synthesized_fields_idx: self.idx_for_promise(KeyClassSynthesizedFields(def_index)),
            variance_idx: self.idx_for_promise(KeyVariance(def_index)),
        };
        // The user - used for first-usage tracking of any expressions we analyze in a class definition -
        // is the `Idx<Key>` of the class object bound to the class name.
        let class_object =
            self.declare_current_idx(Key::Definition(ShortIdentifier::new(class_name)));
        (class_object, class_indices)
    }

    pub fn class_def(&mut self, mut x: StmtClassDef) {
        if self.module_info.name() == ModuleName::typing() && x.name.as_str() == "Any" {
            // We special case the definition of `Any`, because it isn't a `SpecialForm`,
            // but an ordinary `class`.
            self.bind_definition(
                &x.name,
                Binding::Type(Type::type_form(Type::any_explicit())),
                FlowStyle::Other,
            );
            return;
        }

        let (mut class_object, class_indices) = self.class_object_and_indices(&x.name);
        let mut key_class_fields: SmallSet<Idx<KeyClassField>> = SmallSet::new();

        let body = mem::take(&mut x.body);
        let decorators_with_ranges = self.ensure_and_bind_decorators_with_ranges(
            mem::take(&mut x.decorator_list),
            class_object.usage(),
        );

        self.scopes.push(Scope::annotation(x.range));

        x.type_params.iter_mut().for_each(|x| {
            self.type_params(x);
        });

        let mut legacy = Some(LegacyTParamBuilder::new(x.type_params.is_some()));
        let bases = x.bases().map(|base| {
            let mut base = base.clone();
            // Forward refs are fine *inside* of a base expression in the type arguments,
            // but outermost class cannot be a forward ref.
            match &base {
                Expr::StringLiteral(v) => {
                    self.error(
                        base.range(),
                        ErrorKind::InvalidInheritance,
                        None,
                        format!(
                            "Cannot use string annotation `{}` as a base class",
                            v.value.to_str()
                        ),
                    );
                }
                _ => {}
            }
            // If it's really obvious this can't be a legacy type var (since they can't be raw names under bases)
            // then don't even record it.
            let mut none = None;
            let legacy = if matches!(base, Expr::Name(_) | Expr::Attribute(_)) {
                &mut none
            } else {
                &mut legacy
            };
            self.ensure_type(&mut base, legacy);
            base
        });

        let mut keywords = Vec::new();
        if let Some(args) = &mut x.arguments {
            args.keywords.iter_mut().for_each(|keyword| {
                if let Some(name) = &keyword.arg {
                    self.ensure_expr(&mut keyword.value, class_object.usage());
                    keywords.push((name.id.clone(), keyword.value.clone()));
                } else {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidInheritance,
                        None,
                        format!(
                            "The use of unpacking in class header of `{}` is not supported",
                            x.name
                        ),
                    )
                }
            });
        }

        self.insert_binding_idx(
            class_indices.metadata_idx,
            BindingClassMetadata {
                class_idx: class_indices.class_idx,
                bases: bases.clone().into_boxed_slice(),
                keywords: keywords.into_boxed_slice(),
                decorators: decorators_with_ranges.clone().into_boxed_slice(),
                is_new_type: false,
                special_base: None,
            },
        );
        self.insert_binding_idx(
            class_indices.mro_idx,
            BindingClassMro {
                class_idx: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.synthesized_fields_idx,
            BindingClassSynthesizedFields(class_indices.class_idx),
        );

        let legacy_tparam_builder = legacy.unwrap();
        legacy_tparam_builder.add_name_definitions(self);

        self.scopes.push(Scope::class_body(
            x.range,
            class_indices.clone(),
            x.name.clone(),
        ));
        self.init_static_scope(&body, false);
        self.stmts(body);

        let last_scope = self.scopes.pop();
        self.scopes.pop(); // annotation scope
        let mut fields_possibly_defined_by_this_class =
            SmallMap::with_capacity(last_scope.stat.0.len());
        for (name, info) in last_scope.flow.info.iter_hashed() {
            let is_function_without_return_annotation =
                if let FlowStyle::FunctionDef(_, has_return_annotation) = info.style {
                    !has_return_annotation
                } else {
                    false
                };
            // A name with flow in the last_scope, but whose static is in a parent scope, is a reference to something that isn't a class field.
            // Can occur when we narrow a parent scopes variable, thus producing a fresh flow for it, but no static.
            if let Some(stat_info) = last_scope.stat.0.get_hashed(name) {
                let initial_value = info.as_initial_value();
                let value = match &initial_value {
                    ClassFieldInitialValue::Class(Some(e)) => ExprOrBinding::Expr(e.clone()),
                    _ => ExprOrBinding::Binding(Binding::Forward(info.key)),
                };
                let is_initialized_on_class =
                    matches!(initial_value, ClassFieldInitialValue::Class(_));
                let binding = BindingClassField {
                    class_idx: class_indices.class_idx,
                    name: name.into_key().clone(),
                    value,
                    annotation: stat_info.annot,
                    range: stat_info.loc,
                    initial_value,
                    is_function_without_return_annotation,
                    implicit_def_method: None,
                };
                fields_possibly_defined_by_this_class.insert_hashed(
                    name.cloned(),
                    ClassFieldProperties::new(
                        stat_info.annot.is_some(),
                        is_initialized_on_class,
                        stat_info.loc,
                    ),
                );
                let key_field = KeyClassField(class_indices.def_index, name.into_key().clone());
                key_class_fields.insert(self.idx_for_promise(key_field.clone()));
                self.insert_binding(key_field, binding);
            }
        }
        if let ScopeKind::Class(class_scope) = last_scope.kind {
            for (
                name,
                MethodThatSetsAttr {
                    method_name,
                    recognized_attribute_defining_method,
                },
                InstanceAttribute(value, annotation, range),
            ) in class_scope.method_defined_attributes()
            {
                if !fields_possibly_defined_by_this_class.contains_key_hashed(name.as_ref()) {
                    let implicit_def_method = if !recognized_attribute_defining_method {
                        Some(method_name.clone())
                    } else {
                        None
                    };

                    fields_possibly_defined_by_this_class.insert_hashed(
                        name.clone(),
                        ClassFieldProperties::new(annotation.is_some(), false, range),
                    );

                    let key_field = KeyClassField(class_indices.def_index, name.key().clone());
                    key_class_fields.insert(self.idx_for_promise(key_field.clone()));

                    self.insert_binding(
                        key_field,
                        BindingClassField {
                            class_idx: class_indices.class_idx,
                            name: name.into_key(),
                            value,
                            annotation,
                            range,
                            initial_value: ClassFieldInitialValue::Instance(Some(
                                method_name.clone(),
                            )),
                            is_function_without_return_annotation: false,
                            implicit_def_method,
                        },
                    );
                } else if annotation.is_some() {
                    self.error(range, ErrorKind::InvalidAnnotation, None, format!("Cannot annotate attribute `{}`, which is already annotated in the class body", &name), );
                }
            }
        } else {
            unreachable!("Expected class body scope, got {:?}", last_scope.kind);
        }

        let decorator_keys = decorators_with_ranges
            .map(|(idx, _)| *idx)
            .into_boxed_slice();
        self.bind_definition_current(
            &x.name,
            class_object,
            Binding::ClassDef(class_indices.class_idx, decorator_keys),
            FlowStyle::Other,
        );

        // Insert a `KeyTParams` / `BindingTParams` pair, but only if there is at least
        // one generic base class - otherwise, it is not possible that legacy tparams are used.
        let legacy_tparams = legacy_tparam_builder.lookup_keys();
        let tparams_require_binding = !legacy_tparams.is_empty();
        if tparams_require_binding {
            let scoped_type_params = mem::take(&mut x.type_params);
            self.insert_binding(
                KeyTParams(class_indices.def_index),
                BindingTParams {
                    name: x.name.clone(),
                    scoped_type_params,
                    bases: bases.clone().into_boxed_slice(),
                    legacy_tparams: legacy_tparams.into_boxed_slice(),
                },
            );
        }

        fields_possibly_defined_by_this_class.reserve(0); // Attempt to shrink to capacity
        self.insert_binding_idx(
            class_indices.class_idx,
            BindingClass::ClassDef(ClassBinding {
                def_index: class_indices.def_index,
                def: x,
                fields: fields_possibly_defined_by_this_class,
                tparams_require_binding,
            }),
        );

        self.insert_binding_idx(
            class_indices.variance_idx,
            BindingVariance {
                class_key: class_indices.class_idx,
            },
        );
    }

    fn extract_string_literals(
        &mut self,
        items: &[Expr],
    ) -> Vec<(String, TextRange, Option<Expr>)> {
        items
            .iter()
            .filter_map(|item| match item {
                Expr::StringLiteral(x) => Some((x.value.to_string(), x.range(), None)),
                _ => {
                    self.error(
                        item.range(),
                        ErrorKind::InvalidLiteral,
                        None,
                        "Expected a string literal".to_owned(),
                    );
                    None
                }
            })
            .collect()
    }

    fn decompose_key_value_pairs(
        &mut self,
        items: &[Expr],
    ) -> Vec<(String, TextRange, Option<Expr>)> {
        items
            .iter()
            .filter_map(|item| match item {
                Expr::Tuple(ExprTuple { elts, .. }) => match elts.as_slice() {
                    [Expr::StringLiteral(k), v] => {
                        Some((k.value.to_string(), k.range(), Some(v.clone())))
                    }
                    [k, _] => {
                        self.error(
                            k.range(),
                            ErrorKind::InvalidArgument,
                            None,
                            "Expected first item to be a string literal".to_owned(),
                        );
                        None
                    }
                    _ => {
                        self.error(
                            item.range(),
                            ErrorKind::InvalidArgument,
                            None,
                            "Expected a pair".to_owned(),
                        );
                        None
                    }
                },
                _ => {
                    self.error(
                        item.range(),
                        ErrorKind::InvalidArgument,
                        None,
                        "Expected a tuple".to_owned(),
                    );
                    None
                }
            })
            .collect()
    }

    fn synthesize_class_def(
        &mut self,
        class_name: Identifier,
        class_object: CurrentIdx,
        class_indices: ClassIndices,
        base: Option<Expr>,
        keywords: Box<[(Name, Expr)]>,
        // name, position, annotation, value
        member_definitions: Vec<(String, TextRange, Option<Expr>, Option<Expr>)>,
        illegal_identifier_handling: IllegalIdentifierHandling,
        force_class_initialization: bool,
        class_kind: SynthesizedClassKind,
        special_base: Option<Box<BaseClass>>,
    ) {
        let mut key_class_fields: SmallSet<Idx<KeyClassField>> = SmallSet::new();

        self.insert_binding_idx(
            class_indices.metadata_idx,
            BindingClassMetadata {
                class_idx: class_indices.class_idx,
                bases: base
                    .clone()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                keywords,
                decorators: Box::new([]),
                is_new_type: class_kind == SynthesizedClassKind::NewType,
                special_base,
            },
        );
        self.insert_binding_idx(
            class_indices.mro_idx,
            BindingClassMro {
                class_idx: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.synthesized_fields_idx,
            BindingClassSynthesizedFields(class_indices.class_idx),
        );

        let mut fields = SmallMap::new();
        for (idx, (member_name, range, member_annotation, member_value)) in
            member_definitions.into_iter().enumerate()
        {
            let mut member_name = member_name;
            if !is_valid_identifier(member_name.as_str()) {
                match illegal_identifier_handling {
                    IllegalIdentifierHandling::Allow => {}
                    IllegalIdentifierHandling::Error => {
                        self.error(
                            range,
                            ErrorKind::BadClassDefinition,
                            None,
                            format!("`{member_name}` is not a valid identifier"),
                        );
                        continue;
                    }
                    IllegalIdentifierHandling::Rename => member_name = format!("_{idx}"),
                }
            }
            if class_kind == SynthesizedClassKind::NamedTuple && member_name.starts_with("_") {
                match illegal_identifier_handling {
                    IllegalIdentifierHandling::Allow => {}
                    IllegalIdentifierHandling::Error => {
                        self.error(
                             range,
                             ErrorKind::BadClassDefinition,
                             None,
                             format!(
                                 "NamedTuple field name may not start with an underscore: `{member_name}`"
                             ),

                         );
                        continue;
                    }
                    IllegalIdentifierHandling::Rename => member_name = format!("_{idx}"),
                }
            }
            let member_name = Name::new(member_name);
            if fields.contains_key(&member_name) {
                self.error(
                    range,
                    ErrorKind::BadClassDefinition,
                    None,
                    format!("Duplicate field `{member_name}`"),
                );
                continue;
            }
            // Synthesized fields for named tuples are always considered annotated
            fields.insert(
                member_name.clone(),
                ClassFieldProperties::new(
                    member_annotation.is_some() || class_kind == SynthesizedClassKind::NamedTuple,
                    member_value.is_some(),
                    range,
                ),
            );
            let initial_value = if force_class_initialization || member_value.is_some() {
                ClassFieldInitialValue::Class(member_value.clone())
            } else {
                ClassFieldInitialValue::Instance(None)
            };
            let value = match member_value {
                Some(value) => ExprOrBinding::Expr(value),
                None => ExprOrBinding::Binding(Binding::Type(Type::any_implicit())),
            };
            let annotation_binding = if let Some(annotation) = member_annotation {
                let ann_key = KeyAnnotation::Annotation(ShortIdentifier::new(&Identifier::new(
                    member_name.clone(),
                    range,
                )));
                let ann_val = if let Some(special) = SpecialForm::new(&member_name, &annotation) {
                    BindingAnnotation::Type(
                        AnnotationTarget::ClassMember(member_name.clone()),
                        special.to_type(),
                    )
                } else {
                    BindingAnnotation::AnnotateExpr(
                        AnnotationTarget::ClassMember(member_name.clone()),
                        annotation,
                        None,
                    )
                };
                Some(self.insert_binding(ann_key, ann_val))
            } else {
                None
            };

            let key_field = KeyClassField(class_indices.def_index, member_name.clone());
            key_class_fields.insert(self.idx_for_promise(key_field.clone()));

            self.insert_binding(
                key_field,
                BindingClassField {
                    class_idx: class_indices.class_idx,
                    name: member_name,
                    value,
                    annotation: annotation_binding,
                    range,
                    initial_value,
                    is_function_without_return_annotation: false,
                    implicit_def_method: None,
                },
            );
        }
        self.bind_definition_current(
            &class_name,
            class_object,
            Binding::ClassDef(class_indices.class_idx, Box::new([])),
            FlowStyle::Other,
        );
        self.insert_binding_idx(
            class_indices.class_idx,
            BindingClass::FunctionalClassDef(class_indices.def_index, class_name, fields),
        );

        self.insert_binding_idx(
            class_indices.variance_idx,
            BindingVariance {
                class_key: class_indices.class_idx,
            },
        );
    }

    pub fn synthesize_enum_def(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &mut Expr,
        members: &mut [Expr],
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.check_functional_definition_name(&name.id, arg_name);
        self.ensure_expr(func, class_object.usage());
        self.ensure_expr(arg_name, class_object.usage());
        for arg in &mut *members {
            self.ensure_expr(arg, class_object.usage());
        }
        let member_definitions: Vec<(String, TextRange, Option<Expr>, Option<Expr>)> =
            match members {
                // Enum('Color', 'RED, GREEN, BLUE')
                // Enum('Color', 'RED GREEN BLUE')
                [Expr::StringLiteral(x)] => {
                    let s = x.value.to_str();
                    if s.contains(',') {
                        s.split(',')
                            .map(str::trim)
                            .map(|s| (s.to_owned(), x.range(), None))
                            .collect()
                    } else {
                        s.split_whitespace()
                            .map(|s| (s.to_owned(), x.range(), None))
                            .collect()
                    }
                }
                // Enum('Color', 'RED', 'GREEN', 'BLUE')
                [Expr::StringLiteral(_), ..] => self.extract_string_literals(members),
                // Enum('Color', ['RED', 'GREEN', 'BLUE'])
                [Expr::List(ExprList { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::StringLiteral(_), ..]) =>
                {
                    self.extract_string_literals(elts)
                }
                // Enum('Color', ('RED', 'GREEN', 'BLUE'))
                [Expr::Tuple(ExprTuple { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::StringLiteral(_), ..]) =>
                {
                    self.extract_string_literals(elts)
                }
                // Enum('Color', [('RED', 1), ('GREEN', 2), ('BLUE', 3)])
                [Expr::List(ExprList { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::Tuple(_), ..]) =>
                {
                    self.decompose_key_value_pairs(elts)
                }
                // Enum('Color', (('RED', 1), ('GREEN', 2), ('BLUE', 3)))
                [Expr::Tuple(ExprTuple { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::Tuple(_), ..]) =>
                {
                    self.decompose_key_value_pairs(elts)
                }
                // Enum('Color', {'RED': 1, 'GREEN': 2, 'BLUE': 3})
                [Expr::Dict(ExprDict { items, .. })] => items
                    .iter()
                    .filter_map(|item| match (&item.key, &item.value) {
                        (Some(Expr::StringLiteral(k)), v) => {
                            Some((k.value.to_string(), k.range(), Some(v.clone())))
                        }
                        (Some(k), _) => {
                            self.error(
                                k.range(),
                                ErrorKind::InvalidArgument,
                                None,
                                "Expected first item to be a string literal".to_owned(),
                            );
                            None
                        }
                        _ => {
                            self.error(
                                item.range(),
                                ErrorKind::InvalidArgument,
                                None,
                                "Expected a key-value pair".to_owned(),
                            );
                            None
                        }
                    })
                    .collect(),
                _ => {
                    self.error(
                        class_name.range,
                        ErrorKind::InvalidArgument,
                        None,
                        "Expected valid functional enum definition".to_owned(),
                    );
                    Vec::new()
                }
            }
            .into_iter()
            .map(|(name, range, value)| (name, range, None, value))
            .collect();
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            Some(func.clone()),
            Box::new([]),
            member_definitions,
            IllegalIdentifierHandling::Error,
            true,
            SynthesizedClassKind::Enum,
            None,
        );
    }

    // This functional form supports renaming illegal identifiers and specifying defaults
    // but cannot specify the type of each element
    pub fn synthesize_collections_named_tuple_def(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        members: &mut [Expr],
        keywords: &mut [Keyword],
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.ensure_expr(func, class_object.usage());
        self.check_functional_definition_name(&name.id, arg_name);
        let member_definitions: Vec<(String, TextRange, Option<Expr>)> = match members {
            // namedtuple('Point', 'x y')
            // namedtuple('Point', 'x, y')
            [Expr::StringLiteral(x)] => {
                let s = x.value.to_str();
                if s.contains(',') {
                    s.split(',')
                        .map(str::trim)
                        .map(|s| (s.to_owned(), x.range(), None))
                        .collect()
                } else {
                    s.split_whitespace()
                        .map(|s| (s.to_owned(), x.range(), None))
                        .collect()
                }
            }
            // namedtuple('Point', ['x', 'y'])
            [Expr::List(ExprList { elts, .. })]
                if matches!(elts.as_slice(), [Expr::StringLiteral(_), ..]) =>
            {
                self.extract_string_literals(elts)
            }
            // namedtuple('Point', ('x', 'y'))
            [Expr::Tuple(ExprTuple { elts, .. })]
                if matches!(elts.as_slice(), [Expr::StringLiteral(_), ..]) =>
            {
                self.extract_string_literals(elts)
            }
            _ => {
                self.error(
                    class_name.range,
                    ErrorKind::InvalidArgument,
                    None,
                    "Expected valid functional named tuple definition".to_owned(),
                );
                Vec::new()
            }
        };
        let n_members = member_definitions.len();
        let mut illegal_identifier_handling = IllegalIdentifierHandling::Error;
        let mut defaults: Vec<Option<Expr>> = vec![None; n_members];
        for kw in keywords {
            self.ensure_expr(&mut kw.value, class_object.usage());
            if let Some(name) = &kw.arg
                && name.id == "rename"
                && let Expr::BooleanLiteral(lit) = &kw.value
            {
                if lit.value {
                    illegal_identifier_handling = IllegalIdentifierHandling::Rename;
                }
            } else if let Some(name) = &kw.arg
                && name.id == "defaults"
                && let Expr::Tuple(ExprTuple { elts, .. }) = &kw.value
            {
                let n_defaults = elts.len();
                if n_defaults > n_members {
                    self.error(
                        kw.value.range(),
                        ErrorKind::InvalidArgument,
                        None,
                        format!(
                            "Too many defaults values: expected up to {}, got {}",
                            n_members, n_defaults
                        ),
                    );
                    let n_to_drop = n_defaults - n_members;
                    defaults = elts[n_to_drop..].map(|x| Some(x.clone()));
                } else {
                    defaults.splice(n_members - n_defaults.., elts.map(|x| Some(x.clone())));
                }
            } else {
                self.error(
                    kw.value.range(),
                    ErrorKind::InvalidArgument,
                    None,
                    "Unrecognized argument for typed dictionary definition".to_owned(),
                );
            }
        }
        let member_definitions_with_defaults: Vec<(String, TextRange, Option<Expr>, Option<Expr>)> =
            member_definitions
                .into_iter()
                .zip(defaults)
                .map(|((name, range, annotation), default)| (name, range, annotation, default))
                .collect();
        let range = class_name.range();
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            None,
            Box::new([]),
            member_definitions_with_defaults,
            illegal_identifier_handling,
            false,
            SynthesizedClassKind::NamedTuple,
            Some(Box::new(BaseClass::NamedTuple(range))),
        );
    }

    // This functional form allows specifying types for each element, but not default values
    pub fn synthesize_typing_named_tuple_def(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        members: &[Expr],
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.ensure_expr(func, class_object.usage());
        self.check_functional_definition_name(&name.id, arg_name);
        let member_definitions: Vec<(String, TextRange, Option<Expr>, Option<Expr>)> =
            match members {
                // NamedTuple('Point', [('x', int), ('y', int)])
                [Expr::List(ExprList { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::Tuple(_), ..]) =>
                {
                    self.decompose_key_value_pairs(elts)
                }
                // NamedTuple('Point', (('x', int), ('y', int)))
                [Expr::Tuple(ExprTuple { elts, .. })]
                    if matches!(elts.as_slice(), [Expr::Tuple(_), ..]) =>
                {
                    self.decompose_key_value_pairs(elts)
                }
                _ => {
                    self.error(
                        class_name.range,
                        ErrorKind::InvalidArgument,
                        None,
                        "Expected valid functional named tuple definition".to_owned(),
                    );
                    Vec::new()
                }
            }
            .into_iter()
            .map(|(name, range, annotation)| {
                if let Some(mut ann) = annotation {
                    self.ensure_type(&mut ann, &mut None);
                    (name, range, Some(ann), None)
                } else {
                    (name, range, None, None)
                }
            })
            .collect();
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            Some(func.clone()),
            Box::new([]),
            member_definitions,
            IllegalIdentifierHandling::Error,
            false,
            SynthesizedClassKind::NamedTuple,
            None,
        );
    }

    // Synthesize a class definition for NewType
    pub fn synthesize_typing_new_type(
        &mut self,
        name: &ExprName,
        new_type_name: &mut Expr,
        base: &mut Expr,
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.ensure_expr(new_type_name, class_object.usage());
        self.check_functional_definition_name(&name.id, new_type_name);
        self.ensure_type(base, &mut None);
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            Some(base.clone()),
            Box::new([]),
            Vec::new(),
            IllegalIdentifierHandling::Error,
            false,
            SynthesizedClassKind::NewType,
            None,
        );
    }

    pub fn synthesize_typed_dict_def(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        args: &mut [Expr],
        keywords: &mut [Keyword],
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.ensure_expr(func, class_object.usage());
        self.check_functional_definition_name(&name.id, arg_name);
        let mut base_class_keywords: Box<[(Name, Expr)]> = Box::new([]);
        for kw in keywords {
            self.ensure_expr(&mut kw.value, class_object.usage());
            if let Some(name) = &kw.arg
                && name.id == "total"
                && matches!(kw.value, Expr::BooleanLiteral(_))
            {
                base_class_keywords = Box::new([(name.id.clone(), kw.value.clone())])
            } else {
                self.error(
                    kw.value.range(),
                    ErrorKind::InvalidArgument,
                    None,
                    "Unrecognized argument for typed dictionary definition".to_owned(),
                );
            }
        }
        let member_definitions: Vec<(String, TextRange, Option<Expr>, Option<Expr>)> = match args {
            // Movie = TypedDict('Movie', {'name': str, 'year': int})
            [Expr::Dict(ExprDict { items, .. })] => items
                .iter_mut()
                .filter_map(|item| {
                    if let Some(key) = &mut item.key {
                        self.ensure_expr(key, class_object.usage());
                    }
                    self.ensure_type(&mut item.value.clone(), &mut None);
                    match (&item.key, &item.value) {
                        (Some(Expr::StringLiteral(k)), v) => {
                            Some((k.value.to_string(), k.range(), Some(v.clone()), None))
                        }
                        (Some(k), _) => {
                            self.error(
                                k.range(),
                                ErrorKind::InvalidArgument,
                                None,
                                "Expected first item to be a string literal".to_owned(),
                            );
                            None
                        }
                        _ => {
                            self.error(
                                item.range(),
                                ErrorKind::InvalidArgument,
                                None,
                                "Expected a key-value pair".to_owned(),
                            );
                            None
                        }
                    }
                })
                .collect(),
            _ => {
                self.error(
                    class_name.range,
                    ErrorKind::InvalidArgument,
                    None,
                    "Expected valid functional typed dictionary definition".to_owned(),
                );
                Vec::new()
            }
        };
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            Some(func.clone()),
            base_class_keywords,
            member_definitions,
            IllegalIdentifierHandling::Allow,
            false,
            SynthesizedClassKind::TypedDict,
            None,
        );
    }

    // Check that the variable name in a functional class definition matches the first argument string
    fn check_functional_definition_name(&mut self, name: &Name, arg: &Expr) {
        if let Expr::StringLiteral(x) = arg {
            if x.value.to_str() != name.as_str() {
                self.error(
                    arg.range(),
                    ErrorKind::InvalidArgument,
                    None,
                    format!("Expected string literal \"{}\"", name),
                );
            }
        } else {
            self.error(
                arg.range(),
                ErrorKind::InvalidArgument,
                None,
                format!("Expected string literal \"{}\"", name),
            );
        }
    }
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield",
    )
}

fn is_valid_identifier(name: &str) -> bool {
    static IDENTIFIER_REGEX: LazyLock<Regex> =
        LazyLock::new(|| Regex::new("^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap());
    !is_keyword(name) && IDENTIFIER_REGEX.is_match(name)
}
