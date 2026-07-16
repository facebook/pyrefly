/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::dunder;
use pyrefly_types::typed_dict::AnonymousTypedDictInner;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictField;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr::EllipsisLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::TargetWithTParams;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::class::attrs::AttrsInitName;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::class_field::DataclassMember;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::class_metadata::DataclassKind;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::pydantic::PydanticModelKind;
use crate::alt::unwrap::HintRef;
use crate::binding::pydantic::GE;
use crate::binding::pydantic::GT;
use crate::binding::pydantic::LE;
use crate::binding::pydantic::LT;
use crate::binding::pydantic::STRICT;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::display::ClassDisplayContext;
use crate::types::keywords::ConverterMap;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::types::Forallable;
use crate::types::types::Type;

/// Which constructor-copy builtin `call_dataclasses_replace` is serving. Chosen by the caller so the
/// method itself doesn't re-derive dispatch from boolean flags.
#[derive(Clone, Copy)]
pub enum ReplaceKind {
    /// `dataclasses.replace` / `copy.replace`: accepts any dataclass.
    Replace,
    /// `attrs.evolve`: requires an attrs class; keys by constructor alias, init-only.
    Evolve,
    /// `attrs.assoc`: requires an attrs class; keys by attribute name, includes `init=False`.
    Assoc,
}

impl ReplaceKind {
    fn requires_attrs(self) -> bool {
        matches!(self, ReplaceKind::Evolve | ReplaceKind::Assoc)
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Gets dataclass fields for an `@dataclass`-decorated class. attrs with
    /// `auto_attribs=False` collects only `attr.ib()`/`field()` assignments;
    /// every other kind is annotation-driven.
    pub fn get_dataclass_fields(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        kind: &DataclassKind,
    ) -> SmallSet<Name> {
        let attrs_initializer_only = matches!(
            kind,
            DataclassKind::Attrs {
                auto_attribs: Some(false),
                ..
            }
        );
        // attrs relocates a redefined field to its newest declaration site (it deletes the earlier
        // occurrence and re-appends), whereas stdlib `@dataclass` keeps the original position (it
        // reassigns a dict entry in place). `SmallSet::insert` keeps the existing position, so for
        // attrs we `shift_remove` first to move the name to the end.
        let relocate_redefined = matches!(kind, DataclassKind::Attrs { .. });
        let mut all_fields = SmallSet::new();
        for (_, metadata) in bases_with_metadata.iter().rev() {
            if let Some(dataclass) = metadata.dataclass_metadata() {
                for name in dataclass.fields.iter() {
                    if relocate_redefined {
                        all_fields.shift_remove(name);
                    }
                    all_fields.insert(name.clone());
                }
            }
        }
        if let Some(class_fields) = self.get_class_fields(cls) {
            for name in class_fields.class_body_fields() {
                let is_field = if attrs_initializer_only {
                    class_fields.is_attrs_field_specifier(name)
                } else {
                    class_fields.is_field_annotated(name)
                };
                if is_field {
                    if relocate_redefined {
                        all_fields.shift_remove(name);
                    }
                    all_fields.insert(name.clone());
                }
            }
        }
        all_fields
    }

    pub fn get_dataclass_synthesized_fields(
        &self,
        cls: &Class,
        errors: &ErrorCollector,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let dataclass = metadata.dataclass_metadata()?;
        let mut fields = SmallMap::new();

        // Compute kw_only fields once for all methods that need it
        let kw_only_by_class = self.compute_kw_only_fields_by_class(cls);

        self.check_dataclass_non_data_descriptors(cls, dataclass, errors);
        self.check_dataclass_data_descriptor_defaults(cls, dataclass, errors);
        self.check_attrs_default_decorator_return_types(cls, dataclass, errors);
        if dataclass.kws.init {
            let init_method = if let Some((root_model_type, has_strict)) =
                self.get_pydantic_root_model_type_via_mro(cls, &metadata)
            {
                self.get_pydantic_root_model_init(cls, root_model_type, has_strict)
            } else if metadata.is_pydantic_model() {
                // Pydantic models with RootModel fields need type expansion
                let transform_type: &dyn Fn(Type) -> Type = &|ty: Type| {
                    if let Some(root_type) = self.extract_root_model_inner_type(&ty) {
                        self.union(ty, root_type)
                    } else {
                        ty
                    }
                };

                // For BaseSettings, all fields are treated as having defaults
                // since they can be populated from environment variables
                let force_optional = matches!(
                    metadata.pydantic_model_kind(),
                    Some(PydanticModelKind::BaseSettings)
                );

                let field_types: Vec<Type> = self
                    .iter_fields(cls, dataclass, false, &kw_only_by_class)
                    .into_iter()
                    .map(|(_, field, _)| field.ty())
                    .collect();
                let converter_table = self.build_pydantic_lax_conversion_table(&field_types);

                self.get_dataclass_init(
                    cls,
                    dataclass,
                    dataclass.kws.strict,
                    transform_type,
                    force_optional,
                    converter_table,
                    &kw_only_by_class,
                    errors,
                )
            } else {
                // Regular dataclasses: no type transformation, no conversion table
                self.get_dataclass_init(
                    cls,
                    dataclass,
                    dataclass.kws.strict,
                    &|ty| ty,
                    false,
                    ConverterMap::new(),
                    &kw_only_by_class,
                    errors,
                )
            };
            fields.insert(dunder::INIT, init_method);
        } else if matches!(dataclass.kind, DataclassKind::Attrs { .. }) {
            // With `init=False`, attrs still synthesizes the field initializer but names it
            // `__attrs_init__` so a hand-written `__init__` can delegate to it.
            let init_method = self.get_dataclass_init(
                cls,
                dataclass,
                dataclass.kws.strict,
                &|ty| ty,
                false,
                ConverterMap::new(),
                &kw_only_by_class,
                errors,
            );
            fields.insert(Name::new_static("__attrs_init__"), init_method);
        }
        let dataclass_fields_type = self.stdlib.dict(
            self.heap.mk_class_type(self.stdlib.str().clone()),
            self.heap.mk_any_implicit(),
        );
        fields.insert(
            dunder::DATACLASS_FIELDS,
            ClassSynthesizedField::new_classvar(self.heap.mk_class_type(dataclass_fields_type)),
        );

        if dataclass.kws.order {
            fields.extend(self.get_dataclass_rich_comparison_methods(cls));
        }
        if dataclass.kws.match_args {
            fields.insert(
                dunder::MATCH_ARGS,
                self.get_dataclass_match_args(cls, dataclass, &kw_only_by_class),
            );
        }
        if dataclass.kws.slots {
            // It's a runtime error to set slots=True on a class that already defines __slots__.
            // Note that inheriting __slots__ from a base class is fine.
            if self
                .get_class_fields(cls)
                .is_some_and(|f| f.contains(&dunder::SLOTS))
            {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::BadClassDefinition,
                    "Cannot specify both `slots=True` and `__slots__`".to_owned(),
                );
            } else {
                fields.insert(
                    dunder::SLOTS,
                    self.get_dataclass_slots(cls, dataclass, &kw_only_by_class),
                );
            }
        }
        if dataclass.kws.frozen {
            // Only emit BadClassDefinition when no ancestor is already frozen.
            // If an ancestor is frozen its synthesized `@final __setattr__`/`__delattr__`
            // will trigger BadOverride, which gives a richer message naming the parent.
            let has_frozen_ancestor = self
                .get_mro_for_class(cls)
                .ancestors_no_object()
                .iter()
                .any(|ancestor| {
                    self.get_metadata_for_class(ancestor.class_object())
                        .dataclass_metadata()
                        .is_some_and(|dm| dm.kws.frozen)
                });
            if !has_frozen_ancestor && let Some(class_fields) = self.get_class_fields(cls) {
                if let Some(range) = class_fields.field_decl_range(&dunder::SETATTR) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::BadClassDefinition,
                        "Cannot override `__setattr__` in a frozen dataclass".to_owned(),
                    );
                }
                if let Some(range) = class_fields.field_decl_range(&dunder::DELATTR) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::BadClassDefinition,
                        "Cannot override `__delattr__` in a frozen dataclass".to_owned(),
                    );
                }
            }
            fields.insert(dunder::SETATTR, self.get_frozen_setattr(cls));
            fields.insert(dunder::DELATTR, self.get_frozen_delattr(cls));
        }
        // `__hash__` synthesis: stdlib dataclass follows CPython's rules; attrs follows its
        // `hash=`/`unsafe_hash=` arguments. https://docs.python.org/3/library/dataclasses.html#module-contents
        enum HashAction {
            Synthesize,
            SetNone,
            Inherit,
        }
        let hash_action = match &dataclass.kind {
            DataclassKind::Attrs { hash, .. } => match hash {
                Some(true) => HashAction::Synthesize,
                Some(false) => HashAction::Inherit,
                None if !dataclass.kws.eq => HashAction::Inherit,
                None if dataclass.kws.frozen => HashAction::Synthesize,
                None => HashAction::SetNone,
            },
            DataclassKind::Dataclass { .. } => {
                if dataclass.kws.unsafe_hash || (dataclass.kws.eq && dataclass.kws.frozen) {
                    HashAction::Synthesize
                } else if dataclass.kws.eq {
                    HashAction::SetNone
                } else {
                    HashAction::Inherit
                }
            }
        };
        match hash_action {
            HashAction::Synthesize => {
                fields.insert(dunder::HASH, self.get_dataclass_hash(cls));
            }
            HashAction::SetNone => {
                fields.insert(
                    dunder::HASH,
                    ClassSynthesizedField::new(self.heap.mk_none()),
                );
            }
            HashAction::Inherit => {}
        }
        fields.insert(
            dunder::REPLACE,
            self.get_dataclass_replace(cls, dataclass, &kw_only_by_class, errors),
        );
        Some(ClassSynthesizedFields::new(fields))
    }

    /// Check for non-data descriptors in dataclass fields and emit errors.
    ///
    /// Non-data descriptors (having __get__ but no __set__) are unsound in dataclasses
    /// because the dataclass __init__ writes to the instance dict, shadowing the
    /// class-level descriptor.
    ///
    /// Exception: a __get__ returning Self or the descriptor's own class is sound.
    fn check_dataclass_non_data_descriptors(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        errors: &ErrorCollector,
    ) {
        for name in dataclass.fields.iter() {
            if let DataclassMember::Field(field, _) = self.get_dataclass_member(cls, name)
                && let Some((range, descriptor_cls)) = field.value.non_data_descriptor_info()
            {
                let get_return_ty = self
                    .get_class_member(descriptor_cls.class_object(), &dunder::GET)
                    .and_then(|get_field| get_field.ty().callable_return_type(self.heap));

                match &get_return_ty {
                    Some(Type::SelfType(_)) => continue,
                    Some(Type::ClassType(ret)) if *ret == descriptor_cls => continue,
                    _ => {}
                }

                let cls = descriptor_cls.name();
                errors
                    .error_builder(
                        range,
                        ErrorKind::BadClassDefinition,
                        format!("Cannot set field `{name}` to non-data descriptor `{cls}`"),
                    )
                    .with_detail(format!(
                        "Hint: add a `__set__` method to make `{cls}` a data descriptor"
                    ))
                    .emit();
            }
        }
    }

    /// Check that data descriptor defaults are type-safe in dataclass fields.
    ///
    /// For a data descriptor (having both __get__ and __set__), the "default" value
    /// when the field is not provided to __init__ is the class-level descriptor.
    /// Reading the field returns the `__get__` return type, but setting the field
    /// expects the `__set__` value parameter type. For the default to be type-safe,
    /// the `__get__` return type must be assignable to the `__set__` value type.
    fn check_dataclass_data_descriptor_defaults(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        errors: &ErrorCollector,
    ) {
        for name in dataclass.fields.iter() {
            if let DataclassMember::Field(field, _) = self.get_dataclass_member(cls, name)
                && let Some((range, descriptor_cls)) = field.value.data_descriptor_info()
            {
                // Get the __get__ method's return type from the descriptor class.
                let get_return_ty = self
                    .get_class_member(descriptor_cls.class_object(), &dunder::GET)
                    .and_then(|get_field| get_field.ty().callable_return_type(self.heap));

                // Get the __set__ method and extract the value parameter type (3rd param).
                let set_value_ty = self
                    .get_class_member(descriptor_cls.class_object(), &dunder::SET)
                    .and_then(|set_field| {
                        set_field
                            .ty()
                            .callable_signatures()
                            .first()
                            .and_then(|sig| {
                                if let Params::List(params) = &sig.params {
                                    match params.items().get(2) {
                                        Some(Param::Pos(_, t, _) | Param::PosOnly(_, t, _)) => {
                                            Some(t.clone())
                                        }
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            })
                    });

                if let (Some(get_ty), Some(set_ty)) = (get_return_ty, set_value_ty) {
                    // Check if the __get__ return type is assignable to the __set__ value type.
                    if !self.is_subset_eq(&get_ty, &set_ty) {
                        let cls = descriptor_cls.name();
                        errors
                            .error_builder(
                                range,
                                ErrorKind::BadClassDefinition,
                                format!("Cannot set field `{name}` to data descriptor `{cls}` with inconsistent types"),
                            )
                            .with_detail(format!(
                                "Return type `{get_ty}` of `{cls}.__get__` is not assignable to value type `{set_ty}` of `{cls}.__set__`"
                            ))
                            .emit();
                    }
                }
            }
        }
    }

    pub fn call_dataclasses_replace(
        &self,
        kind: ReplaceKind,
        replace_ty: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let Some(CallArg::Arg(obj_arg)) = args.first() else {
            return self.freeform_call_infer(
                replace_ty.clone(),
                args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            );
        };
        let obj_ty = obj_arg.infer(self, errors);

        // `evolve`/`assoc` require an attrs class; `replace` accepts any dataclass.
        let is_valid_target = |cls: &ClassType| {
            let metadata = self.get_metadata_for_class(cls.class_object());
            if kind.requires_attrs() {
                metadata
                    .dataclass_metadata()
                    .is_some_and(|dm| matches!(dm.kind, DataclassKind::Attrs { .. }))
            } else {
                metadata.dataclass_metadata().is_some()
            }
        };

        let mut dataclasses = Vec::new();
        let mut non_dataclasses = Vec::new();
        // Best-effort: reject only a concrete non-attrs `ClassType`. Gradual and non-class types are
        // left to the stub, avoiding false positives and staying robust to new `Type` variants.
        let mut has_non_attrs_class = false;
        self.map_over_union(&obj_ty, |ty| match ty {
            Type::ClassType(cls) if is_valid_target(cls) => dataclasses.push(ty.clone()),
            _ => {
                has_non_attrs_class = has_non_attrs_class
                    || (kind.requires_attrs() && matches!(ty, Type::ClassType(_)));
                non_dataclasses.push(ty.clone());
            }
        });

        if has_non_attrs_class {
            self.error(
                errors,
                obj_arg.range(),
                ErrorKind::BadArgumentType,
                "First argument is not an attrs class".to_owned(),
            );
        }

        // For unions, typecheck each valid target individually. We treat the first argument
        // as the member type to avoid rejecting `A | B` as not assignable to `A`.
        let rest_args = args.iter().skip(1).cloned().collect::<Vec<_>>();
        let mut rets = dataclasses.map(|ty| {
            if matches!(kind, ReplaceKind::Assoc) {
                let Type::ClassType(cls) = ty else {
                    unreachable!("assoc targets are validated attrs ClassTypes")
                };
                return self.call_attrs_assoc(
                    cls,
                    &rest_args,
                    kws,
                    callee_range,
                    arg_range,
                    hint,
                    errors,
                );
            }
            let ret = self.call_magic_dunder_method(
                ty,
                &dunder::REPLACE,
                arg_range,
                &rest_args,
                kws,
                errors,
                None,
            );
            ret.unwrap_or_else(|| ty.clone())
        });
        if !non_dataclasses.is_empty() {
            let mut new_args = Vec::with_capacity(args.len());
            let new_first_arg = self.unions(non_dataclasses);
            new_args.push(CallArg::ty(&new_first_arg, obj_arg.range()));
            new_args.extend(args.iter().skip(1).cloned());
            rets.push(self.freeform_call_infer(
                replace_ty.clone(),
                &new_args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            ));
        }
        self.unions(rets)
    }

    pub fn call_dataclasses_asdict(
        &self,
        asdict_ty: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        const MAX_ASDICT_FIELDS: usize = 20;

        if let [CallArg::Arg(obj_arg)] = args
            && kws.is_empty()
            && let Type::ClassType(cls) = &obj_arg.infer(self, errors)
            && let Some(dataclass) = self
                .get_metadata_for_class(cls.class_object())
                .dataclass_metadata()
        {
            let kw_only = self.compute_kw_only_fields_by_class(cls.class_object());
            // excludes InitVar (and ClassVar) members but keeps `init=False` ones.
            let raw_fields = self.iter_fields(cls.class_object(), dataclass, false, &kw_only);
            // Decide on field count before doing the per-field work: bail to the freeform
            // path for wide dataclasses to bound the size of the synthesized TypedDict.
            if raw_fields.len() <= MAX_ASDICT_FIELDS {
                let sub = cls.targs().substitution();
                // At runtime, asdict traverses the object recursively and converts any contained dataclasses into a dict.
                // Here, we traverse the type & transform any dataclasses into dict[str, Any]
                let dict_str_any = self.heap.mk_class_type(self.stdlib.dict(
                    self.heap.mk_class_type(self.stdlib.str().clone()),
                    self.heap.mk_any_explicit(),
                ));
                let fields: Vec<(Name, TypedDictField)> = raw_fields
                    .into_iter()
                    .map(|(name, field, _)| {
                        let mut ty = sub.substitute_into(field.ty());
                        ty.transform_mut(&mut |t| {
                            if let Type::ClassType(c) = t
                                && self
                                    .get_metadata_for_class(c.class_object())
                                    .dataclass_metadata()
                                    .is_some()
                            {
                                *t = dict_str_any.clone();
                            }
                        });
                        (
                            name,
                            TypedDictField {
                                ty,
                                required: true,
                                read_only_reason: None,
                            },
                        )
                    })
                    .collect();
                return self.heap.mk_typed_dict(TypedDict::Anonymous(Box::new(
                    AnonymousTypedDictInner { fields },
                )));
            }
        }
        self.freeform_call_infer(
            asdict_ty.clone(),
            args,
            kws,
            callee_range,
            arg_range,
            hint,
            errors,
        )
    }

    fn get_dataclass_replace(
        &self,
        cls: &Class,
        dataclass_metadata: &DataclassMetadata,
        kw_only_by_class: &SmallMap<Class, SmallSet<Name>>,
        errors: &ErrorCollector,
    ) -> ClassSynthesizedField {
        let mut params = vec![self.class_self_param(cls, true)];

        let strict_default = dataclass_metadata.kws.strict;
        for (name, field, field_flags) in
            self.iter_fields(cls, dataclass_metadata, true, kw_only_by_class)
        {
            if !field_flags.init {
                continue;
            }

            let strict = field_flags.strict.unwrap_or(strict_default);
            let has_default = !field.is_init_var() || field_flags.default.is_some();
            if field_flags.init_by_name {
                let param_name = match self.attrs_init_param_name(cls, &name, dataclass_metadata) {
                    AttrsInitName::Renamed(stripped) => stripped,
                    AttrsInitName::Collision(_) | AttrsInitName::Unchanged => name.clone(),
                };
                params.push(self.as_param(
                    &field,
                    &param_name,
                    has_default,
                    true,
                    strict,
                    field_flags.converter_param.clone(),
                    &|t| t,
                    errors,
                ));
            }
            if let Some(alias) = &field_flags.init_by_alias {
                params.push(self.as_param(
                    &field,
                    alias,
                    has_default,
                    true,
                    strict,
                    field_flags.converter_param.clone(),
                    &|t| t,
                    errors,
                ));
            }
        }
        if dataclass_metadata.kws.extra {
            params.push(Param::Kwargs(None, self.heap.mk_any_implicit()));
        }

        let ty = self.synthesized_method(cls, dunder::REPLACE, params, self.instantiate(cls));
        ClassSynthesizedField::new(ty)
    }

    /// Validate that frozen and non-frozen dataclasses are not mixed in an inheritance chain.
    /// `@dataclass` and attrs reject both directions (frozen-from-non-frozen and
    /// non-frozen-from-frozen). Other `@dataclass_transform` classes only reject non-frozen
    /// inheriting from frozen, since the transform lets each class independently opt into frozen.
    pub fn validate_frozen_dataclass_inheritance(
        &self,
        cls: &Class,
        dataclass_metadata: &DataclassMetadata,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        is_from_dataclass_transform: bool,
        errors: &ErrorCollector,
    ) {
        for (base, base_metadata) in bases_with_metadata {
            if let Some(base_dataclass_metadata) = base_metadata.dataclass_metadata() {
                let is_base_frozen = base_dataclass_metadata.kws.frozen;
                let is_current_frozen = dataclass_metadata.kws.frozen;

                if is_current_frozen != is_base_frozen {
                    // For non-attrs dataclass_transform classes, a frozen subclass of a non-frozen
                    // base is allowed; the restriction only applies when a non-frozen subclass
                    // inherits from a frozen base, which would violate the parent's immutability
                    // guarantee.
                    let is_attrs = matches!(dataclass_metadata.kind, DataclassKind::Attrs { .. });
                    if is_from_dataclass_transform
                        && !is_attrs
                        && is_current_frozen
                        && !is_base_frozen
                    {
                        continue;
                    }

                    let current_status = if is_current_frozen {
                        "frozen"
                    } else {
                        "non-frozen"
                    };
                    let base_status = if is_base_frozen {
                        "frozen"
                    } else {
                        "non-frozen"
                    };

                    let ctx = ClassDisplayContext::new(&[cls, base]);
                    self.error(
                        errors,
                        cls.range(),
                        ErrorKind::InvalidInheritance,
                        format!(
                            "Cannot inherit {} dataclass `{}` from {} dataclass `{}`",
                            current_status,
                            ctx.display(cls),
                            base_status,
                            ctx.display(base),
                        ),
                    );
                }
            }
        }
    }

    pub fn validate_post_init(
        &self,
        cls: &Class,
        dataclass_metadata: &DataclassMetadata,
        post_init: Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        // `__post_init__` is called with a dataclass's `InitVar`s, so we use the `InitVar` types
        // to generate a callable signature to check `__post_init__` against.
        let kw_only_by_class = self.compute_kw_only_fields_by_class(cls);
        let mut params = Vec::new();
        for (name, field, _) in self.iter_fields(cls, dataclass_metadata, true, &kw_only_by_class) {
            if field.is_init_var() {
                params.push(self.as_param(
                    &field,
                    &name,
                    false,
                    false,
                    true,
                    None,
                    &|ty| ty,
                    errors,
                ));
            }
        }
        let want = self.heap.mk_callable_from(Callable::list(
            ParamList::new(params),
            self.heap.mk_class_type(self.stdlib.object().clone()),
        ));
        self.check_type(&post_init, &want, range, errors, &|| {
            TypeCheckContext::of_kind(TypeCheckKind::PostInit)
        });
    }

    pub fn dataclass_field_keywords(
        &self,
        func: &Type,
        args: &Arguments,
        annotated_field_ty: Option<&Type>,
        dataclass_metadata: &DataclassMetadata,
        errors: &ErrorCollector,
    ) -> DataclassFieldKeywords {
        let mut map = TypeMap::new();
        let alias_keyword = &dataclass_metadata.alias_keyword;
        for kw in args.keywords.iter() {
            if let Some(name) = &kw.arg {
                map.0
                    .insert(name.id.clone(), self.expr_infer(&kw.value, errors));
            }
        }
        let mut init = map.get_bool(&DataclassFieldKeywords::INIT);
        let mut default = self.get_default(&map);

        if default.is_none()
            && dataclass_metadata.default_can_be_positional
            && let Some(default_expr) = args.args.first()
            && !matches!(default_expr, EllipsisLiteral(_))
        {
            // Check whether a default was passed positionally. This is needed for `pydantic.Field`.
            default = Some(self.expr_infer(default_expr, errors));
        }

        let mut kw_only = map.get_bool(&DataclassFieldKeywords::KW_ONLY);

        let mut alias = if dataclass_metadata.init_defaults.init_by_alias {
            map.get_string(alias_keyword)
                .or_else(|| map.get_string(&DataclassFieldKeywords::ALIAS))
                .map(Name::new)
        } else {
            None
        };

        let gt = map.0.get(&GT).cloned();
        let lt = map.0.get(&LT).cloned();
        let ge = map.0.get(&GE).cloned();
        let le = map.0.get(&LE).cloned();

        let strict: Option<bool> = map.0.get(&STRICT).and_then(|v| v.as_bool());

        // The raw expression of the argument for `on_setattr`.
        let attrs_on_setattr_expr = args
            .keywords
            .iter()
            .find(|kw| {
                kw.arg
                    .as_ref()
                    .is_some_and(|n| n.id == DataclassFieldKeywords::ON_SETATTR)
            })
            .map(|kw| &kw.value);
        let attrs_setattr_frozen = attrs_on_setattr_expr.map(|e| self.on_setattr_is_frozen(e));

        // Read the converter from an explicit `converter=` argument only, not the specifier
        // signature (which always declares one) — else every plain field's param goes Unknown.
        let converter_param = self
            .attrs_converters_combinator_param(args, annotated_field_ty, errors)
            .or_else(|| {
                map.0
                    .get(&DataclassFieldKeywords::CONVERTER)
                    .map(|converter| self.get_converter_param(converter, annotated_field_ty))
            });
        // Note that we intentionally don't try to fill in `default`, since we can't distinguish
        // between a real default and something like `dataclasses.MISSING`.
        if init.is_none() || kw_only.is_none() || alias.is_none() {
            self.fill_in_field_keywords_from_function_signature(
                func,
                args,
                errors,
                if dataclass_metadata.init_defaults.init_by_alias {
                    Some(alias_keyword)
                } else {
                    None
                },
                &mut init,
                &mut kw_only,
                &mut alias,
            );
        }
        DataclassFieldKeywords {
            init: init.unwrap_or(true),
            default,
            kw_only,
            init_by_name: dataclass_metadata.init_defaults.init_by_name || alias.is_none(),
            init_by_alias: alias,
            lt,
            gt,
            ge,
            le,
            strict,
            converter_param,
            attrs_setattr_frozen,
        }
    }

    /// Fill in keyword values from the function signature of a dataclass field specifier.
    fn fill_in_field_keywords_from_function_signature(
        &self,
        func: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
        // The name of the function parameter from which to fill in an alias keyword value
        alias_keyword: Option<&Name>,
        init: &mut Option<bool>,
        kw_only: &mut Option<bool>,
        alias: &mut Option<Name>,
    ) {
        // Class-based field specifiers (e.g. `field_specifiers=(CustomField,)`) need to be
        // resolved to their constructor callable so that we can read keyword defaults from
        // `__init__`. This mirrors how Pyright handles `field_specifiers` per PEP 681.
        let constructor_callable = self.constructor_to_callable_distributed(func);
        let func = constructor_callable.as_ref().unwrap_or(func);
        let sigs = func.callable_signatures();
        let sig = if sigs.len() == 1 {
            sigs[0].clone()
        } else if sigs.len() > 1
            && let Type::Overload(overload) = func
        {
            // Overloaded function. Call it to see which signature is actually used.
            // TODO: sigs could contain unbound type parameters, because `callable_signatures`
            // looks through foralls. Overload selection might fail spuriously.
            self.call_overloads(
                Vec1::try_from_vec(sigs.map(|x| {
                    TargetWithTParams(
                        None,
                        Function {
                            signature: ((*x).clone()),
                            metadata: *overload.metadata.clone(),
                        },
                    )
                }))
                .unwrap(),
                &overload.metadata,
                None, // no shape_transform for dataclass constructors
                None,
                &args.args.map(CallArg::expr_maybe_starred),
                &args.keywords.map(CallKeyword::new),
                args.range,
                errors,
                None,
                None,
                None,
            )
            .1
        } else {
            return;
        };
        if let Params::List(params) = &sig.params {
            for param in params.items() {
                // Look for a parameter that can be called by name, to attempt to read a default value for a keyword argument.
                let (name, ty, default_ty) = match param {
                    Param::Pos(name, ty, Required::Required)
                    | Param::KwOnly(name, ty, Required::Required) => (name, ty, None),
                    Param::Pos(name, ty, Required::Optional(default))
                    | Param::KwOnly(name, ty, Required::Optional(default)) => {
                        (name, ty, default.as_ref().map(|d| &d.ty))
                    }
                    _ => continue,
                };
                if name == &DataclassFieldKeywords::INIT {
                    self.fill_in_literal(init, ty, default_ty, |ty| ty.as_bool());
                }
                if name == &DataclassFieldKeywords::KW_ONLY {
                    self.fill_in_literal(kw_only, ty, default_ty, |ty| ty.as_bool());
                }
                if alias.is_none() && Some(name) == alias_keyword {
                    self.fill_in_literal(alias, ty, default_ty, |ty| match ty {
                        Type::Literal(lit) if let Lit::Str(s) = &lit.value => Some(Name::new(s)),
                        _ => None,
                    });
                }
            }
        }
    }

    /// Fills in a keyword with a literal value from a parameter type and default, if possible.
    fn fill_in_literal<T>(
        &self,
        keyword: &mut Option<T>,
        ty: &Type,
        default: Option<&Type>,
        type_to_literal: impl Fn(&Type) -> Option<T>,
    ) {
        if keyword.is_none() {
            if let Some(lit) = type_to_literal(ty) {
                *keyword = Some(lit);
            } else if let Some(default) = default
                && let Some(lit) = type_to_literal(default)
            {
                *keyword = Some(lit);
            }
        }
    }

    pub(crate) fn constructor_to_callable_distributed(&self, ty: &Type) -> Option<Type> {
        let instance = match ty {
            Type::ClassDef(cls) => self.promote_silently(cls),
            Type::Type(inner) => (**inner).clone(),
            _ => return None,
        };
        let Type::ClassType(instance) = instance else {
            return None;
        };
        let callable = self.constructor_to_callable(&instance);
        Some(self.distribute_over_union(&callable, |ty| {
            if let Type::BoundMethod(m) = ty {
                self.bind_boundmethod(m, &mut |a, b| self.is_subset_eq(a, b))
                    .unwrap_or_else(|| ty.clone())
            } else {
                ty.clone()
            }
        }))
    }

    pub(crate) fn get_converter_param(
        &self,
        converter: &Type,
        annotated_field_ty: Option<&Type>,
    ) -> Type {
        let solved = match (converter, annotated_field_ty) {
            (Type::ClassDef(cls), Some(hint)) => self.solve_generic_class_converter(cls, hint),
            (Type::Forall(_), Some(hint)) => self.solve_generic_function_converter(converter, hint),
            _ => None,
        };
        let converter = solved.as_ref().unwrap_or(converter);
        let constructor_callable = self.constructor_to_callable_distributed(converter);
        let converter = constructor_callable.as_ref().unwrap_or(converter);
        self.distribute_over_union(converter, |ty| {
            // Only overloads callable with a single positional argument contribute an input type;
            // an overload requiring a second positional arg can't be the converter.
            let inputs: Vec<Type> = ty
                .callable_signatures()
                .iter()
                .filter(|sig| sig.accepts_single_positional_arg())
                .filter_map(|sig| sig.get_first_param())
                .collect();
            if inputs.is_empty() {
                self.heap.mk_any_implicit()
            } else {
                self.unions(inputs)
            }
        })
    }

    /// A generic class converter is type-preserving, so solve its type parameters against the
    /// declared field `hint` (e.g. `tuple` against `Sequence[int]` becomes `tuple[int, ...]`).
    /// `None` for a non-generic class or an output incompatible with `hint`.
    fn solve_generic_class_converter(&self, cls: &Class, hint: &Type) -> Option<Type> {
        let mut class_type = self.as_class_type_unchecked(cls);
        if class_type.targs().is_empty() {
            return None;
        }
        let vs = self
            .solver()
            .freshen_class_targs(class_type.targs_mut(), self.uniques);
        let matched = self.is_subset_eq(&self.heap.mk_class_type(class_type.clone()), hint);
        self.solver()
            .generalize_class_targs(class_type.targs_mut(), &SmallSet::new());
        self.solver()
            .finish_class_targs(class_type.targs_mut(), self.uniques);
        // Finalizing the fresh vars is required; its specialization errors are dropped because this
        // is best-effort param inference with no call site to report them against.
        let _ = self.finish_quantified(vs, self.solver().infer_with_first_use);
        matched.then(|| self.heap.mk_type(self.heap.mk_class_type(class_type)))
    }

    /// A generic function converter is solved by constraining its return type against the declared
    /// field `hint`, so the `__init__` param is the solved input, not a leaked type variable.
    fn solve_generic_function_converter(&self, converter: &Type, hint: &Type) -> Option<Type> {
        let Type::Forall(forall) = converter else {
            return None;
        };
        if !matches!(forall.body, Forallable::Function(_)) {
            return None;
        }
        let (vs, instantiated) = self.solver().fresh_quantified(
            &forall.tparams,
            forall.body.clone().as_type(),
            self.uniques,
        );
        if let Some(ret) = instantiated.callable_return_type(self.heap) {
            self.is_subset_eq(&ret, hint);
        }
        let _ = self.finish_quantified(vs, self.solver().infer_with_first_use);
        Some(self.solver().expand(instantiated))
    }

    fn get_default(&self, map: &TypeMap) -> Option<Type> {
        if let Some(default) = map.0.get(&DataclassFieldKeywords::DEFAULT) {
            return Some(default.clone());
        }
        let factory = map
            .0
            .get(&DataclassFieldKeywords::DEFAULT_FACTORY)
            .or_else(|| map.0.get(&DataclassFieldKeywords::FACTORY))?;
        let constructor_callable = self.constructor_to_callable_distributed(factory);
        Some(
            constructor_callable
                .as_ref()
                .unwrap_or(factory)
                .callable_return_type(self.heap)
                .unwrap_or_else(|| self.heap.mk_any_implicit()),
        )
    }

    pub fn compute_kw_only_fields_by_class(&self, cls: &Class) -> SmallMap<Class, SmallSet<Name>> {
        let is_kw_only_marker = |ty: &Type| matches!(ty, Type::ClassType(cls) if cls.has_qname("dataclasses", "KW_ONLY"));

        let compute_for_class = |target_cls: &Class| -> SmallSet<Name> {
            let mut kw_only_fields = SmallSet::new();
            let mut seen_kw_only_marker = false;
            let Some(class_fields) = self.get_class_fields(target_cls) else {
                return kw_only_fields;
            };
            for name in class_fields.names() {
                if !class_fields.is_field_annotated(name) {
                    continue;
                }
                let Some(field) =
                    self.get_non_synthesized_field_from_current_class_only(target_cls, name)
                else {
                    continue;
                };
                if is_kw_only_marker(&field.ty()) {
                    seen_kw_only_marker = true;
                } else if seen_kw_only_marker {
                    kw_only_fields.insert(name.clone());
                }
            }
            kw_only_fields
        };

        let mut result: SmallMap<Class, SmallSet<Name>> = SmallMap::new();
        result.insert(cls.clone(), compute_for_class(cls));

        for ancestor in self.get_mro_for_class(cls).ancestors_no_object() {
            let ancestor_cls = ancestor.class_object();
            if ancestor_cls == cls {
                continue;
            }
            result.insert(ancestor_cls.clone(), compute_for_class(ancestor_cls));
        }
        result
    }

    pub fn iter_fields(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        include_initvar: bool,
        kw_only_fields_by_class: &SmallMap<Class, SmallSet<Name>>,
    ) -> Vec<(Name, ClassField, DataclassFieldKeywords)> {
        let mut positional_fields = Vec::new();
        let mut kwonly_fields = Vec::new();
        let cls_is_kw_only = dataclass.kws.kw_only;
        for name in dataclass.fields.iter() {
            match (self.get_dataclass_member(cls, name), include_initvar) {
                (DataclassMember::KwOnlyMarker, _) => {
                    // KW_ONLY markers are not fields, skip them
                }
                (DataclassMember::NotAField, _) => {}
                (DataclassMember::Field(field, mut keywords), _)
                | (DataclassMember::InitVar(field, mut keywords), true) => {
                    if keywords.kw_only.is_none() {
                        // kw_only hasn't been explicitly set on the field.
                        // A field is kw_only if:
                        // 1. It appears after a KW_ONLY marker in its defining class, OR
                        // 2. Its defining class has kw_only=True in the decorator
                        let after_kw_only_marker = kw_only_fields_by_class
                            .get(&field.defining_class)
                            .is_some_and(|fields| fields.contains(name));
                        let defining_class_is_kw_only = if field.defining_class == *cls {
                            cls_is_kw_only
                        } else {
                            self.get_metadata_for_class(&field.defining_class)
                                .dataclass_metadata()
                                .is_some_and(|m| m.kws.kw_only)
                        };
                        keywords.kw_only = Some(after_kw_only_marker || defining_class_is_kw_only);
                    };
                    if keywords.is_kw_only() {
                        kwonly_fields.push((name.clone(), (*field.value).clone(), keywords))
                    } else {
                        positional_fields.push((name.clone(), (*field.value).clone(), keywords))
                    }
                }
                (DataclassMember::InitVar(..), false) => {}
            }
        }
        positional_fields.extend(kwonly_fields);
        positional_fields
    }

    /// Whether class `c`'s own init layout orders the non-default, non-kw-only field `name` after a
    /// defaulted field. `kw_only_by_class` must cover `c`'s fields (a subclass's map suffices, since
    /// `c`'s fields are all defined in the subclass's ancestors).
    fn init_field_after_default(
        &self,
        c: &Class,
        name: &Name,
        kw_only_by_class: &SmallMap<Class, SmallSet<Name>>,
    ) -> bool {
        let metadata = self.get_metadata_for_class(c);
        let Some(dataclass) = metadata.dataclass_metadata() else {
            return false;
        };
        let mut has_seen_default = false;
        for (n, _field, flags) in self.iter_fields(c, dataclass, true, kw_only_by_class) {
            if !flags.init || flags.is_kw_only() {
                continue;
            }
            let has_default =
                flags.default.is_some() || (flags.init_by_name && flags.init_by_alias.is_some());
            if &n == name {
                return !has_default && has_seen_default;
            }
            if has_default {
                has_seen_default = true;
            }
        }
        false
    }

    /// Gets __init__ method for an `@dataclass`-decorated class.
    fn get_dataclass_init(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        strict_default: bool,
        param_type_transform: &dyn Fn(Type) -> Type,
        force_optional: bool,
        converter_table: ConverterMap,
        kw_only_by_class: &SmallMap<Class, SmallSet<Name>>,
        errors: &ErrorCollector,
    ) -> ClassSynthesizedField {
        // CPython renames the receiver to `__dataclass_self__` when a field is named `self`; mirror that.
        let self_param = if dataclass.fields.contains(&Name::new_static("self")) {
            Param::Pos(
                Name::new_static("__dataclass_self__"),
                self.instantiate(cls),
                Required::Required,
            )
        } else {
            self.class_self_param(cls, false)
        };
        let mut params = vec![self_param];
        let mut has_seen_default = false;
        for (name, field, field_flags) in self.iter_fields(cls, dataclass, true, kw_only_by_class) {
            let strict = field_flags.strict.unwrap_or(strict_default);
            if field_flags.init {
                let has_default = force_optional
                    || field_flags.default.is_some()
                    || (field_flags.init_by_name && field_flags.init_by_alias.is_some());
                let is_kw_only = field_flags.is_kw_only();
                if !is_kw_only {
                    if !has_default && has_seen_default {
                        // `Some` only when the field is declared in this class's own body;
                        // `None` when it is inherited from a base.
                        let own_decl_range = self
                            .get_class_fields(cls)
                            .and_then(|f| f.field_decl_range(&name));
                        // Report at the class where the conflict first appears: if the field is
                        // declared in this class, report here; if it is inherited, report only when
                        // no base already reports it (else every subclass would repeat the error).
                        let reported_by_base = own_decl_range.is_none()
                            && self
                                .get_mro_for_class(cls)
                                .ancestors_no_object()
                                .iter()
                                .any(|base| {
                                    base.class_object() != cls
                                        && self.init_field_after_default(
                                            base.class_object(),
                                            &name,
                                            kw_only_by_class,
                                        )
                                });
                        if !reported_by_base {
                            self.error(
                                errors,
                                own_decl_range.unwrap_or_else(|| cls.range()),
                                ErrorKind::BadClassDefinition,
                                format!(
                                    "Dataclass field `{name}` without a default may not follow dataclass field with a default"
                                ),
                            );
                        }
                    }
                    if has_default {
                        has_seen_default = true;
                    }
                }

                // If this field has a `@field_validator(..., mode='before'|'plain')`, the init
                // parameter accepts `Any` because the validator transforms arbitrary input.
                let converter_param = if dataclass.pydantic_before_validator_fields.contains(&name)
                {
                    Some(self.heap.mk_any_explicit())
                } else {
                    field_flags.converter_param.clone().or_else(|| {
                        if !strict {
                            converter_table.get(&field.ty()).cloned()
                        } else {
                            None
                        }
                    })
                };

                if field_flags.init_by_name {
                    let param_name = match self.attrs_init_param_name(cls, &name, dataclass) {
                        AttrsInitName::Renamed(stripped) => stripped,
                        AttrsInitName::Collision(stripped) => {
                            if let Some(range) = self
                                .get_class_fields(cls)
                                .and_then(|f| f.field_decl_range(&name))
                            {
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::BadClassDefinition,
                                    format!(
                                        "Field `{name}` collides with `{stripped}` after stripping leading underscores"
                                    ),
                                );
                            }
                            name.clone()
                        }
                        AttrsInitName::Unchanged => name.clone(),
                    };
                    params.push(self.as_param(
                        &field,
                        &param_name,
                        has_default,
                        is_kw_only,
                        strict,
                        converter_param.clone(),
                        param_type_transform,
                        errors,
                    ));
                }
                if let Some(alias) = &field_flags.init_by_alias {
                    params.push(self.as_param(
                        &field,
                        alias,
                        has_default,
                        is_kw_only,
                        strict,
                        converter_param,
                        param_type_transform,
                        errors,
                    ));
                }
            }
        }
        if dataclass.kws.extra {
            params.push(Param::Kwargs(None, self.heap.mk_any_implicit()));
        }

        let ty = self.synthesized_method(cls, dunder::INIT, params, self.heap.mk_none());
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_match_args(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        kw_only_by_class: &SmallMap<Class, SmallSet<Name>>,
    ) -> ClassSynthesizedField {
        // Keyword-only fields do not appear in __match_args__.
        let kw_only = dataclass.kws.kw_only;
        let ts = if kw_only {
            Vec::new()
        } else {
            let filtered_fields = self.iter_fields(cls, dataclass, true, kw_only_by_class);
            filtered_fields
                .iter()
                .filter_map(|(name, _, field_flags)| {
                    if field_flags.is_kw_only() || !field_flags.init {
                        None
                    } else {
                        Some(Lit::Str(name.as_str().into()).to_implicit_type())
                    }
                })
                .collect()
        };
        let ty = self.heap.mk_concrete_tuple(ts);
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_slots(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        kw_only_by_class: &SmallMap<Class, SmallSet<Name>>,
    ) -> ClassSynthesizedField {
        let filtered_fields = self.iter_fields(cls, dataclass, false, kw_only_by_class);
        let ts = filtered_fields
            .iter()
            .map(|(name, _, _)| Lit::Str(name.as_str().into()).to_implicit_type())
            .collect();
        let ty = self.heap.mk_concrete_tuple(ts);
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_rich_comparison_methods(
        &self,
        cls: &Class,
    ) -> SmallMap<Name, ClassSynthesizedField> {
        let bool_ty = self.heap.mk_class_type(self.stdlib.bool().clone());
        let make_signature = |other_type| {
            let other = Param::Pos(Name::new_static("other"), other_type, Required::Required);
            Callable::list(
                ParamList::new(vec![self.class_self_param(cls, false), other]),
                bool_ty.clone(),
            )
        };
        let callable = make_signature(self.instantiate(cls));
        let callable_eq = make_signature(self.heap.mk_class_type(self.stdlib.object().clone()));
        dunder::RICH_CMPS
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    ClassSynthesizedField::new(self.heap.mk_function(Function {
                        signature: if *name == dunder::EQ || *name == dunder::NE {
                            callable_eq.clone()
                        } else {
                            callable.clone()
                        },
                        metadata: FuncMetadata::method(cls, name.clone()),
                    })),
                )
            })
            .collect()
    }

    fn get_dataclass_hash(&self, cls: &Class) -> ClassSynthesizedField {
        let params = vec![self.class_self_param(cls, false)];
        let ret = self.heap.mk_class_type(self.stdlib.int().clone());
        ClassSynthesizedField::new(self.synthesized_method(cls, dunder::HASH, params, ret))
    }

    fn get_frozen_setattr(&self, cls: &Class) -> ClassSynthesizedField {
        // Match typeshed's `object.__setattr__(self, name: str, value: Any, /) -> None`
        // — all parameters are positional-only.
        let params = vec![
            self.class_self_param(cls, true),
            Param::PosOnly(
                Some(Name::new_static("name")),
                self.heap.mk_class_type(self.stdlib.str().clone()),
                Required::Required,
            ),
            Param::PosOnly(
                Some(Name::new_static("value")),
                self.heap.mk_any_implicit(),
                Required::Required,
            ),
        ];
        let mut metadata = FuncMetadata::def(self.module(), Some(cls), dunder::SETATTR);
        metadata.flags.has_final_decoration = true;
        ClassSynthesizedField::new(self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), self.heap.mk_none()),
            metadata,
        }))
    }

    fn get_frozen_delattr(&self, cls: &Class) -> ClassSynthesizedField {
        // Match typeshed's `object.__delattr__(self, name: str, /) -> None`
        // — all parameters are positional-only.
        let params = vec![
            self.class_self_param(cls, true),
            Param::PosOnly(
                Some(Name::new_static("name")),
                self.heap.mk_class_type(self.stdlib.str().clone()),
                Required::Required,
            ),
        ];
        let mut metadata = FuncMetadata::def(self.module(), Some(cls), dunder::DELATTR);
        metadata.flags.has_final_decoration = true;
        ClassSynthesizedField::new(self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), self.heap.mk_none()),
            metadata,
        }))
    }
}
