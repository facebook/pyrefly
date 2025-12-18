/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;
use std::sync::LazyLock;

use dupe::Dupe as _;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::visit::Visit;
use regex::Regex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::binding::base_class::BaseClass;
use crate::binding::base_class::BaseClassGeneric;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAbstractClassCheck;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassBaseType;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::BindingClassMetadata;
use crate::binding::binding::BindingClassMro;
use crate::binding::binding::BindingClassSynthesizedFields;
use crate::binding::binding::BindingConsistentOverrideCheck;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::BindingTParams;
use crate::binding::binding::BindingVariance;
use crate::binding::binding::ClassBinding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAbstractClassCheck;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassBaseType;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyConsistentOverrideCheck;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::KeyTParams;
use crate::binding::binding::KeyVariance;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::CurrentIdx;
use crate::binding::bindings::LegacyTParamCollector;
use crate::binding::pydantic::PydanticConfigDict;
use crate::binding::scope::ClassIndices;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::Scope;
use crate::config::error_kind::ErrorKind;
use crate::error::context::ErrorInfo;
use crate::export::special::SpecialExport;
use crate::types::class::ClassDefIndex;
use crate::types::class::ClassFieldProperties;
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
            base_type_idx: self.idx_for_promise(KeyClassBaseType(def_index)),
            metadata_idx: self.idx_for_promise(KeyClassMetadata(def_index)),
            mro_idx: self.idx_for_promise(KeyClassMro(def_index)),
            synthesized_fields_idx: self.idx_for_promise(KeyClassSynthesizedFields(def_index)),
            variance_idx: self.idx_for_promise(KeyVariance(def_index)),
            consistent_override_check_idx: self
                .idx_for_promise(KeyConsistentOverrideCheck(def_index)),
            abstract_class_check_idx: self.idx_for_promise(KeyAbstractClassCheck(def_index)),
        };
        // The user - used for first-usage tracking of any expressions we analyze in a class definition -
        // is the `Idx<Key>` of the class object bound to the class name.
        let class_object =
            self.declare_current_idx(Key::Definition(ShortIdentifier::new(class_name)));
        (class_object, class_indices)
    }

    pub fn class_def(&mut self, mut x: StmtClassDef, parent: &NestingContext) {
        let (mut class_object, class_indices) = self.class_object_and_indices(&x.name);
        let mut pydantic_config_dict = PydanticConfigDict::default();
        let docstring_range = Docstring::range_from_stmts(x.body.as_slice());
        let body = mem::take(&mut x.body);
        let field_docstrings = self.extract_field_docstrings(&body);
        let decorators =
            self.ensure_and_bind_decorators(mem::take(&mut x.decorator_list), class_object.usage());

        self.scopes.push(Scope::annotation(x.range));

        let scoped_type_param_names = x
            .type_params
            .as_mut()
            .map(|x| self.type_params(x))
            .unwrap_or_default();

        let mut legacy = Some(LegacyTParamCollector::new(x.type_params.is_some()));
        let bases = x.bases().map(|base| {
            let mut base = base.clone();
            // Forward refs are fine *inside* of a base expression in the type arguments,
            // but outermost class cannot be a forward ref.
            match &base {
                Expr::StringLiteral(v) => {
                    self.error(
                        base.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        format!(
                            "Cannot use string annotation `{}` as a base class",
                            v.value.to_str()
                        ),
                    );
                }
                _ => {}
            }
            // If it's really obvious this can't be a legacy type var then don't even record it.
            let mut none = None;
            let legacy = match &base {
                Expr::Subscript(ExprSubscript { value, slice, .. }) => {
                    // Syntactically, this may be a legacy type var.
                    if matches!(&**slice, Expr::Name(x) if scoped_type_param_names.contains(&x.id))
                        && !matches!(
                            self.as_special_export(value),
                            Some(SpecialExport::Generic | SpecialExport::Protocol)
                        )
                    {
                        // This definitely isn't a legacy type var: it's a reference to a scoped
                        // type var. Note that even if there exists a legacy type var with the same
                        // name, the scoped type var shadows it.
                        &mut none
                    } else {
                        &mut legacy
                    }
                }
                _ => &mut none,
            };
            self.ensure_type(&mut base, legacy);

            let base_class = self.base_class_of(base.clone());
            // NOTE(grievejia): If any of the class base is a specialized generic class (e.g. `Foo[Bar]`), and if the tparam of the
            // generic class has a bound or constraint, we won't be validating the type `Bar` against that bound/constraint of the
            // tparam eagerly in order to avoid dependency cycle. But the validation needs to happen somewhere.
            //
            // Since we can't create "delayed" computations on-the-fly in the answer phase, we'll create a static computation here
            // trying to type check all valid base class expressions. We are duplicating a lot of work with class base calculation,
            // which is sad. Hence we were able to figure out a better places to insert those checks we should migrate.
            //
            // Also note that there's no risk of first-usage tracking issues here because `ensure_type` does not participate in first
            // usage tracking.
            if matches!(base_class, BaseClass::BaseClassExpr(..)) {
                self.insert_binding(
                    KeyExpect(base.range()),
                    BindingExpect::TypeCheckBaseClassExpr(base),
                );
            }
            base_class
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
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        format!(
                            "The use of unpacking in class header of `{}` is not supported",
                            x.name
                        ),
                    )
                }
            });
        }

        self.insert_binding_idx(
            class_indices.base_type_idx,
            BindingClassBaseType {
                class_idx: class_indices.class_idx,
                is_new_type: false,
                bases: bases.clone().into_boxed_slice(),
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

        let legacy_tparam_collector = legacy.unwrap();
        self.add_name_definitions(&legacy_tparam_collector);

        self.scopes.push(Scope::class_body(
            x.range,
            class_indices.clone(),
            x.name.clone(),
        ));
        self.init_static_scope(&body, false);
        self.stmts(
            body,
            &NestingContext::class(ShortIdentifier::new(&x.name), parent.dupe()),
        );
        let field_definitions = self.scopes.finish_class_and_get_field_definitions();

        let mut django_primary_key_field: Option<Name> = None;
        let mut fields = SmallMap::with_capacity(field_definitions.len());
        for (name, (definition, range)) in field_definitions.into_iter_hashed() {
            if let ClassFieldDefinition::AssignedInBody {
                value: ExprOrBinding::Expr(e),
                ..
            } = &definition
            {
                self.extract_pydantic_config_dict(e, &name, &mut pydantic_config_dict);

                if self.extract_django_primary_key(e) {
                    django_primary_key_field = Some(name.clone().into_key());
                }
            }
            let (is_initialized_on_class, is_annotated) = match &definition {
                ClassFieldDefinition::DefinedInMethod { annotation, .. } => {
                    (false, annotation.is_some())
                }
                ClassFieldDefinition::DeclaredByAnnotation { .. } => (false, true),
                ClassFieldDefinition::AssignedInBody { annotation, .. } => {
                    (true, annotation.is_some())
                }
                _ => (true, false),
            };

            let docstring_range = field_docstrings.get(&range).copied();

            fields.insert_hashed(
                name.clone(),
                ClassFieldProperties::new(
                    is_annotated,
                    is_initialized_on_class,
                    range,
                    docstring_range,
                ),
            );
            let key_field = KeyClassField(class_indices.def_index, name.clone().into_key());
            let binding = BindingClassField {
                class_idx: class_indices.class_idx,
                name: name.into_key(),
                range,
                definition,
            };
            self.insert_binding(key_field, binding);
        }

        self.bind_current_as(
            &x.name,
            class_object,
            Binding::ClassDef(
                class_indices.class_idx,
                decorators.clone().into_boxed_slice(),
            ),
            FlowStyle::ClassDef,
        );

        // Insert a `KeyTParams` / `BindingTParams` pair, but only if there is at least
        // one generic base class - otherwise, it is not possible that legacy tparams are used.
        let legacy_tparams = legacy_tparam_collector.lookup_keys();
        let tparams_require_binding = !legacy_tparams.is_empty();
        if tparams_require_binding {
            let scoped_type_params = mem::take(&mut x.type_params);
            self.insert_binding(
                KeyTParams(class_indices.def_index),
                BindingTParams {
                    name: x.name.clone(),
                    scoped_type_params,
                    generic_bases: bases
                        .iter()
                        .filter_map(|base| match base {
                            BaseClass::Generic(x) => Some(x),
                            _ => None,
                        })
                        .cloned()
                        .collect::<Box<[BaseClassGeneric]>>(),
                    legacy_tparams: legacy_tparams.into_boxed_slice(),
                },
            );
        }

        fields.reserve(0); // Attempt to shrink to capacity
        self.insert_binding_idx(
            class_indices.class_idx,
            BindingClass::ClassDef(ClassBinding {
                def_index: class_indices.def_index,
                def: x,
                parent: parent.dupe(),
                fields,
                tparams_require_binding,
                docstring_range,
            }),
        );

        self.insert_binding_idx(
            class_indices.variance_idx,
            BindingVariance {
                class_key: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.consistent_override_check_idx,
            BindingConsistentOverrideCheck {
                class_key: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.metadata_idx,
            BindingClassMetadata {
                class_idx: class_indices.class_idx,
                bases: bases.clone().into_boxed_slice(),
                keywords: keywords.into_boxed_slice(),
                decorators: decorators.into_boxed_slice(),
                is_new_type: false,
                pydantic_config_dict,
                django_primary_key_field,
            },
        );
        self.insert_binding_idx(
            class_indices.abstract_class_check_idx,
            BindingAbstractClassCheck {
                class_idx: class_indices.class_idx,
            },
        );
    }

    /// Extracts docstrings for each field, mapping the field's range to the docstring's range.
    fn extract_field_docstrings(
        &self,
        body: &[ruff_python_ast::Stmt],
    ) -> SmallMap<TextRange, TextRange> {
        use ruff_python_ast::Expr;
        use ruff_python_ast::Stmt;

        let mut field_docstrings = SmallMap::new();
        let mut i = 0;

        while i < body.len() {
            let stmt = &body[i];

            let is_field = matches!(stmt, Stmt::AnnAssign(_) | Stmt::Assign(_));

            if let Stmt::FunctionDef(func_def) = stmt {
                if let Some(docstring_range) = Docstring::range_from_stmts(&func_def.body) {
                    field_docstrings.insert(func_def.name.range, docstring_range);
                }
            } else if let Stmt::ClassDef(class_def) = stmt {
                if let Some(docstring_range) = Docstring::range_from_stmts(&class_def.body) {
                    field_docstrings.insert(class_def.name.range, docstring_range);
                }
            } else if is_field
                && let Some(next_stmt) = body.get(i + 1)
                && let Stmt::Expr(expr_stmt) = next_stmt
                && matches!(&*expr_stmt.value, Expr::StringLiteral(_))
            {
                let docstring_range = next_stmt.range();
                let mut target_ranges = Vec::new();
                Self::collect_field_docstring_target_ranges(stmt, &mut target_ranges);
                for range in target_ranges {
                    field_docstrings.insert(range, docstring_range);
                }
            }

            i += 1;
        }

        field_docstrings
    }

    fn collect_field_docstring_target_ranges(stmt: &Stmt, ranges: &mut Vec<TextRange>) {
        match stmt {
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    Self::collect_ranges_from_expr(target, ranges);
                }
            }
            Stmt::AnnAssign(ann_assign) => {
                Self::collect_ranges_from_expr(&ann_assign.target, ranges);
            }
            _ => {}
        }
    }

    fn collect_ranges_from_expr(expr: &Expr, ranges: &mut Vec<TextRange>) {
        if let Expr::Name(name) = expr {
            ranges.push(name.range);
        }
        expr.recurse(&mut |e| Self::collect_ranges_from_expr(e, ranges));
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
                        ErrorInfo::Kind(ErrorKind::InvalidLiteral),
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
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Expected first item to be a string literal".to_owned(),
                        );
                        None
                    }
                    _ => {
                        self.error(
                            item.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Expected a pair".to_owned(),
                        );
                        None
                    }
                },
                _ => {
                    self.error(
                        item.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
        parent: &NestingContext,
        base: Option<Expr>,
        keywords: Box<[(Name, Expr)]>,
        // name, position, annotation, value
        member_definitions: Vec<(String, TextRange, Option<Expr>, Option<Expr>)>,
        illegal_identifier_handling: IllegalIdentifierHandling,
        force_class_initialization: bool,
        class_kind: SynthesizedClassKind,
        special_base: Option<BaseClass>,
    ) {
        let base_classes = base
            .into_iter()
            .map(|base| self.base_class_of(base))
            .chain(special_base)
            .collect::<Vec<_>>();
        let is_new_type = class_kind == SynthesizedClassKind::NewType;
        self.insert_binding_idx(
            class_indices.base_type_idx,
            BindingClassBaseType {
                class_idx: class_indices.class_idx,
                is_new_type,
                bases: base_classes.clone().into_boxed_slice(),
            },
        );
        self.insert_binding_idx(
            class_indices.metadata_idx,
            BindingClassMetadata {
                class_idx: class_indices.class_idx,
                bases: base_classes.into_boxed_slice(),
                keywords,
                decorators: Box::new([]),
                is_new_type,
                pydantic_config_dict: PydanticConfigDict::default(),
                django_primary_key_field: None,
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
                            ErrorInfo::Kind(ErrorKind::BadClassDefinition),
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
                             ErrorInfo::Kind(ErrorKind::BadClassDefinition),
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
                    ErrorInfo::Kind(ErrorKind::BadClassDefinition),
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
                    None, // Synthesized fields don't have docstrings
                ),
            );
            let annotation = member_annotation.map(|annotation_expr| {
                self.insert_binding(
                    KeyAnnotation::Annotation(ShortIdentifier::new(&Identifier::new(
                        member_name.clone(),
                        range,
                    ))),
                    BindingAnnotation::AnnotateExpr(
                        AnnotationTarget::ClassMember(member_name.clone()),
                        annotation_expr,
                        None,
                    ),
                )
            });
            let definition = match (member_value, force_class_initialization) {
                (Some(value), _) => ClassFieldDefinition::AssignedInBody {
                    value: ExprOrBinding::Expr(value),
                    annotation,
                },
                (None, true) => ClassFieldDefinition::AssignedInBody {
                    value: ExprOrBinding::Binding(Binding::Type(Type::any_implicit())),
                    annotation,
                },
                (None, false) => match annotation {
                    Some(annotation) => ClassFieldDefinition::DeclaredByAnnotation { annotation },
                    None => ClassFieldDefinition::DeclaredWithoutAnnotation,
                },
            };
            self.insert_binding(
                KeyClassField(class_indices.def_index, member_name.clone()),
                BindingClassField {
                    class_idx: class_indices.class_idx,
                    name: member_name,
                    range,
                    definition,
                },
            );
        }
        self.bind_current_as(
            &class_name,
            class_object,
            Binding::ClassDef(class_indices.class_idx, Box::new([])),
            FlowStyle::ClassDef,
        );
        self.insert_binding_idx(
            class_indices.class_idx,
            BindingClass::FunctionalClassDef(
                class_indices.def_index,
                class_name,
                parent.dupe(),
                fields,
            ),
        );

        self.insert_binding_idx(
            class_indices.variance_idx,
            BindingVariance {
                class_key: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.consistent_override_check_idx,
            BindingConsistentOverrideCheck {
                class_key: class_indices.class_idx,
            },
        );
        self.insert_binding_idx(
            class_indices.abstract_class_check_idx,
            BindingAbstractClassCheck {
                class_idx: class_indices.class_idx,
            },
        );
    }

    pub fn synthesize_enum_def(
        &mut self,
        name: &ExprName,
        parent: &NestingContext,
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
                // Enum('Color', []), Enum('Color', ())
                [
                    Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. }),
                    ..,
                ] if elts.is_empty() => Vec::new(),
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
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Expected first item to be a string literal".to_owned(),
                            );
                            None
                        }
                        _ => {
                            self.error(
                                item.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Expected a key-value pair".to_owned(),
                            );
                            None
                        }
                    })
                    .collect(),
                _ => {
                    self.error(
                        class_name.range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
            parent,
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
        parent: &NestingContext,
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
            // namedtuple('Point', []), namedtuple('Point', ())
            [Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. })]
                if elts.is_empty() =>
            {
                Vec::new()
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
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
                && let Expr::Tuple(ExprTuple { elts, .. }) | Expr::List(ExprList { elts, .. }) =
                    &kw.value
            {
                let n_defaults = elts.len();
                if n_defaults > n_members {
                    self.error(
                        kw.value.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        format!(
                            "Too many defaults values: expected up to {n_members}, got {n_defaults}",
                        ),
                    );
                    let n_to_drop = n_defaults - n_members;
                    defaults = elts[n_to_drop..].map(|x| Some(x.clone()));
                } else {
                    defaults.splice(n_members - n_defaults.., elts.map(|x| Some(x.clone())));
                }
            } else {
                let maybe_name = if let Some(name) = &kw.arg {
                    format!(" `{name}`")
                } else {
                    "".to_owned()
                };
                self.error(
                    kw.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!("Unrecognized argument{maybe_name} for named tuple definition"),
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
            parent,
            None,
            Box::new([]),
            member_definitions_with_defaults,
            illegal_identifier_handling,
            false,
            SynthesizedClassKind::NamedTuple,
            Some(BaseClass::NamedTuple(range)),
        );
    }

    // This functional form allows specifying types for each element, but not default values
    pub fn synthesize_typing_named_tuple_def(
        &mut self,
        name: &ExprName,
        parent: &NestingContext,
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
                // NamedTuple('Point', []), NamedTuple('Point', ())
                [Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. })]
                    if elts.is_empty() =>
                {
                    Vec::new()
                }
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
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
            parent,
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
        parent: &NestingContext,
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
            parent,
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
        parent: &NestingContext,
        func: &mut Expr,
        arg_name: &Expr,
        args: &mut [Expr],
        keywords: &mut [Keyword],
    ) {
        let class_name = Ast::expr_name_identifier(name.clone());
        let (mut class_object, class_indices) = self.class_object_and_indices(&class_name);
        self.ensure_expr(func, class_object.usage());
        self.check_functional_definition_name(&name.id, arg_name);
        let mut base_class_keywords = Vec::new();
        for kw in keywords {
            self.ensure_expr(&mut kw.value, class_object.usage());
            let recognized_kw = match (kw.arg.as_ref().map(|id| &id.id), &kw.value) {
                (Some(name), Expr::BooleanLiteral(_)) if name == "total" || name == "closed" => {
                    Some(name)
                }
                (Some(name), _) if name == "extra_items" => Some(name),
                _ => None,
            };
            if let Some(kw_name) = recognized_kw {
                base_class_keywords.push((kw_name.clone(), kw.value.clone()));
            } else {
                let maybe_name = if let Some(name) = &kw.arg {
                    format!(" `{name}`")
                } else {
                    "".to_owned()
                };
                self.error(
                    kw.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!("Unrecognized argument{maybe_name} for typed dictionary definition"),
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
                    self.ensure_type(&mut item.value, &mut None);
                    match (&item.key, &item.value) {
                        (Some(Expr::StringLiteral(k)), v) => {
                            Some((k.value.to_string(), k.range(), Some(v.clone()), None))
                        }
                        (Some(k), _) => {
                            self.error(
                                k.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Expected first item to be a string literal".to_owned(),
                            );
                            None
                        }
                        _ => {
                            self.error(
                                item.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    "Expected valid functional typed dictionary definition".to_owned(),
                );
                Vec::new()
            }
        };
        self.synthesize_class_def(
            class_name,
            class_object,
            class_indices,
            parent,
            Some(func.clone()),
            base_class_keywords.into_boxed_slice(),
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
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!("Expected string literal \"{name}\""),
                );
            }
        } else {
            self.error(
                arg.range(),
                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                format!("Expected string literal \"{name}\""),
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
