/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::dunder;
use pyrefly_types::simplify::unions_with_literals;
use pyrefly_types::typed_dict::ExtraItems;
use ruff_python_ast::DictItem;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallmap;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::solver::solver::SubsetError;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::literal::Lit;
use crate::types::quantified::Quantified;
use crate::types::tuple::Tuple;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Restriction;
use crate::types::typed_dict::TypedDict;
use crate::types::typed_dict::TypedDictField;
use crate::types::types::Forall;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::Substitution;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::Type;

const CLEAR_METHOD: Name = Name::new_static("clear");
const GET_METHOD: Name = Name::new_static("get");
const POP_METHOD: Name = Name::new_static("pop");
const POPITEM_METHOD: Name = Name::new_static("popitem");
const SETDEFAULT_METHOD: Name = Name::new_static("setdefault");
const KEY_PARAM: Name = Name::new_static("key");
const DEFAULT_PARAM: Name = Name::new_static("default");
const UPDATE_METHOD: Name = Name::new_static("update");
const ITEMS_METHOD: Name = Name::new_static("items");
const VALUES_METHOD: Name = Name::new_static("values");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn check_dict_items_against_typed_dict(
        &self,
        dict_items: &Vec<&DictItem>,
        typed_dict: &TypedDict,
        // Check whether `typed_dict` can be updated with `dict_items`
        is_update: bool,
        range: TextRange,
        check_errors: &ErrorCollector,
        item_errors: &ErrorCollector,
    ) {
        let fields = self.typed_dict_fields(typed_dict);
        let extra_items = self.typed_dict_extra_items(typed_dict.class_object());
        let mut has_expansion = false;
        let mut keys: SmallSet<Name> = SmallSet::new();
        dict_items.iter().for_each(|x| match &x.key {
            Some(key) => {
                let key_type = self.expr_infer(key, item_errors);
                if let Type::Literal(Lit::Str(name)) = key_type {
                    let key_name = Name::new(name);
                    match fields.get(&key_name) {
                        Some(field) if is_update && field.is_read_only() => {
                            self.error(
                                check_errors,
                                key.range(),
                                ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                                format!("Cannot update read-only field `{key_name}`"),
                            );
                        }
                        Some(field) => {
                            self.expr_with_separate_check_errors(
                                &x.value,
                                Some((&field.ty, check_errors, &|| {
                                    TypeCheckContext::of_kind(TypeCheckKind::TypedDictKey(Some(
                                        key_name.clone(),
                                    )))
                                })),
                                item_errors,
                            );
                        }
                        None if let ExtraItems::Extra(extra) = &extra_items => {
                            self.expr_with_separate_check_errors(
                                &x.value,
                                Some((&extra.ty, check_errors, &|| {
                                    TypeCheckContext::of_kind(TypeCheckKind::TypedDictKey(None))
                                })),
                                item_errors,
                            );
                        }
                        None => {
                            self.error(
                                check_errors,
                                key.range(),
                                ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                                format!(
                                    "Key `{}` is not defined in TypedDict `{}`",
                                    key_name,
                                    typed_dict.name()
                                ),
                            );
                        }
                    }
                    keys.insert(key_name);
                } else {
                    self.error(
                        check_errors,
                        key.range(),
                        ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                        format!(
                            "Expected string literal key, got `{}`",
                            self.for_display(key_type)
                        ),
                    );
                }
            }
            None => {
                // This is an unpacked item (`**some_dict`).
                has_expansion = true;
                let partial_td_ty = Type::PartialTypedDict(typed_dict.clone());
                let item_ty = self.expr_infer_with_hint(
                    &x.value,
                    Some(HintRef::soft(&partial_td_ty)),
                    item_errors,
                );
                let subset_result = self.is_subset_eq_with_reason(&item_ty, &partial_td_ty);
                if let Some(subset_error) = subset_result.err() {
                    let tcc: &dyn Fn() -> TypeCheckContext =
                        if matches!(subset_error, SubsetError::OpenTypedDict(_)) {
                            // This SubsetError variant is used to report cases in which the unpacked
                            // item is an open TypedDict that, via inheritance, may contain a key from
                            // `partial_td_ty` with an incompatible type. We report this error via a
                            // dedicated error kind that is off by default.
                            &|| TypeCheckContext::of_kind(TypeCheckKind::TypedDictOpenUnpacking)
                        } else {
                            &|| TypeCheckContext::of_kind(TypeCheckKind::TypedDictUnpacking)
                        };
                    self.solver().error(
                        &item_ty,
                        &partial_td_ty,
                        check_errors,
                        range,
                        tcc,
                        subset_error,
                    );
                }
            }
        });
        // You can update a TypedDict with a subset of its items. Otherwise, all required fields must be present.
        if !has_expansion && !is_update {
            for (key, field) in &fields {
                if field.required && !keys.contains(key) {
                    self.error(
                        check_errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                        format!(
                            "Missing required key `{}` for TypedDict `{}`",
                            key,
                            typed_dict.name()
                        ),
                    );
                }
            }
        }
    }

    pub fn typed_dict_extra_items(&self, cls: &Class) -> ExtraItems {
        self.get_metadata_for_class(cls)
            .typed_dict_metadata()
            .map_or(ExtraItems::Default, |m| m.extra_items.clone())
    }

    fn class_field_to_typed_dict_field(
        &self,
        class: &Class,
        substitution: &Substitution,
        name: &Name,
        is_total: bool,
    ) -> Option<TypedDictField> {
        self.get_class_member(class, name).and_then(|member| {
            Arc::unwrap_or_clone(member.value)
                .as_typed_dict_field_info(is_total)
                .map(|field| field.substitute_with(substitution))
        })
    }

    pub fn typed_dict_fields(&self, typed_dict: &TypedDict) -> SmallMap<Name, TypedDictField> {
        let class = typed_dict.class_object();
        let metadata = self.get_metadata_for_class(class);
        let substitution = typed_dict.targs().substitution();

        match metadata.typed_dict_metadata() {
            None => {
                // This may happen during incremental update where `class` is stale/outdated
                SmallMap::new()
            }
            Some(typed_dict_metadata) => typed_dict_metadata
                .fields
                .iter()
                .filter_map(|(name, is_total)| {
                    self.class_field_to_typed_dict_field(class, &substitution, name, *is_total)
                        .map(|field| (name.clone(), field))
                })
                .collect(),
        }
    }

    pub fn typed_dict_field(&self, typed_dict: &TypedDict, name: &Name) -> Option<TypedDictField> {
        let class = typed_dict.class_object();
        let metadata = self.get_metadata_for_class(class);
        let substitution = typed_dict.targs().substitution();

        metadata
            .typed_dict_metadata()
            .and_then(|typed_dict_metadata| {
                typed_dict_metadata.fields.get(name).and_then(|is_total| {
                    self.class_field_to_typed_dict_field(class, &substitution, name, *is_total)
                })
            })
    }

    fn names_to_fields<'b>(
        &'b self,
        cls: &'b Class,
        fields: &'b SmallMap<Name, bool>,
    ) -> impl Iterator<Item = (&'b Name, TypedDictField)> + 'b {
        // TODO(stroxler): Look into whether we can re-wire the code so that it is not possible to
        // have the typed dict think a field exists that cannot be converted to a `TypedDictField`
        // (this can happen for any unannotated field - e.g. a classmethod or staticmethod).
        fields.iter().filter_map(|(name, is_total)| {
            self.get_non_synthesized_class_member(cls, name)
                .and_then(|member| {
                    Arc::unwrap_or_clone(member)
                        .as_typed_dict_field_info(*is_total)
                        .map(|field| (name, field))
                })
        })
    }

    fn get_typed_dict_init(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let mut params = vec![self.class_self_param(cls, true)];
        for (name, field) in self.names_to_fields(cls, fields) {
            params.push(Param::KwOnly(
                name.clone(),
                field.ty,
                if field.required {
                    Required::Required
                } else {
                    Required::Optional(None)
                },
            ));
        }
        if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(cls) {
            params.push(Param::Kwargs(None, extra.ty));
        }
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), Type::None),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_typed_dict_update(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), UPDATE_METHOD);
        let self_param = self.class_self_param(cls, true);
        let extra = if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(cls) {
            Some(extra.ty)
        } else {
            None
        };

        // ---- Overload: def update(m: Partial[C], /)
        let full_typed_dict = self.as_typed_dict_unchecked(cls);
        let partial_typed_dict_ty = Type::PartialTypedDict(full_typed_dict);

        let partial_overload = OverloadType::Function(Function {
            signature: Callable::list(
                ParamList::new(vec![
                    self_param.clone(),
                    Param::PosOnly(
                        Some(Name::new_static("m")),
                        partial_typed_dict_ty,
                        Required::Required,
                    ),
                ]),
                Type::None,
            ),
            metadata: metadata.clone(),
        });

        // ---- Overload: update(m: Iterable[tuple[Literal["key"], value]], /)
        let get_tuple = |name, ty| Type::Tuple(Tuple::Concrete(vec![self.name_or_str(name), ty]));
        let mut tuple_types: Vec<Type> = self
            .names_to_fields(cls, fields)
            .filter(|(_, field)| !field.is_read_only()) // filter read-only fields
            .map(|(name, field)| get_tuple(Some(name), field.ty))
            .collect();
        if let Some(extra) = &extra {
            tuple_types.push(get_tuple(None, extra.clone()));
        }

        let iterable_ty = self.stdlib.iterable(self.unions(tuple_types)).to_type();

        let tuple_overload = OverloadType::Function(Function {
            signature: Callable::list(
                ParamList::new(vec![
                    self_param.clone(),
                    Param::PosOnly(Some(Name::new_static("m")), iterable_ty, Required::Required),
                ]),
                Type::None,
            ),
            metadata: metadata.clone(),
        });

        // ---- Overload: update(*, x=..., y=...)
        let mut keyword_params = self
            .names_to_fields(cls, fields)
            .filter(|(_, field)| !field.is_read_only()) // filter read-only fields
            .map(|(name, field)| {
                Param::KwOnly(name.clone(), field.ty.clone(), Required::Optional(None))
            })
            .collect::<Vec<_>>();
        if let Some(extra) = extra {
            keyword_params.push(Param::Kwargs(None, extra));
        }

        let overload_kwargs = OverloadType::Function(Function {
            signature: Callable::list(
                ParamList::new(
                    std::iter::once(self_param.clone())
                        .chain(keyword_params)
                        .collect(),
                ),
                Type::None,
            ),
            metadata: metadata.clone(),
        });

        let signatures = vec1![partial_overload, tuple_overload, overload_kwargs];

        ClassSynthesizedField::new(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    /// Get a (key, default: T) -> ValueType | T overload.
    fn get_overload_with_default(
        &self,
        metadata: &FuncMetadata,
        self_param: &Param,
        name: Option<&Name>,
        ty: Type,
    ) -> OverloadType {
        let q = Quantified::type_var(
            Name::new("_T"),
            self.uniques,
            None,
            Restriction::Unrestricted,
        );
        let tparams = vec![TParam {
            quantified: q.clone(),
            variance: PreInferenceVariance::PInvariant,
        }];
        OverloadType::Forall(Forall {
            tparams: Arc::new(TParams::new(tparams)),
            body: Function {
                signature: Callable::list(
                    ParamList::new(vec![
                        self_param.clone(),
                        self.key_param(name),
                        Param::PosOnly(
                            Some(DEFAULT_PARAM.clone()),
                            q.clone().to_type(),
                            Required::Required,
                        ),
                    ]),
                    self.union(ty, q.to_type()),
                ),
                metadata: metadata.clone(),
            },
        })
    }

    fn get_typed_dict_get(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), GET_METHOD);
        // Synthesizes signatures for each field and a fallback `(self, key: str, default: object = ...) -> object` signature.
        let self_param = self.class_self_param(cls, true);
        let object_ty = self.stdlib.object().clone().to_type();
        let mut literal_signatures = Vec::new();
        for (name, field) in self.names_to_fields(cls, fields) {
            let key_param = Param::PosOnly(
                Some(KEY_PARAM.clone()),
                name_to_literal_type(name),
                Required::Required,
            );
            if field.required {
                // (self, key: Literal["key"], default: object = ...) -> ValueType
                literal_signatures.push(OverloadType::Function(Function {
                    signature: Callable::list(
                        ParamList::new(vec![
                            self_param.clone(),
                            key_param,
                            Param::PosOnly(
                                Some(DEFAULT_PARAM.clone()),
                                object_ty.clone(),
                                Required::Optional(None),
                            ),
                        ]),
                        field.ty.clone(),
                    ),
                    metadata: metadata.clone(),
                }));
            } else {
                // (self, key: Literal["key"]) -> ValueType | None
                literal_signatures.push(OverloadType::Function(Function {
                    signature: Callable::list(
                        ParamList::new(vec![self_param.clone(), key_param.clone()]),
                        Type::optional(field.ty.clone()),
                    ),
                    metadata: metadata.clone(),
                }));
                // (self, key: Literal["key"], default: T) -> ValueType | T
                literal_signatures.push(self.get_overload_with_default(
                    &metadata,
                    &self_param,
                    Some(name),
                    field.ty,
                ));
            }
        }
        let value_ty = self.get_typed_dict_value_type_from_fields(cls, fields);
        let mut signatures = Vec1::from_vec_push(
            literal_signatures,
            OverloadType::Function(Function {
                signature: Callable::list(
                    ParamList::new(vec![
                        self_param.clone(),
                        Param::PosOnly(
                            Some(KEY_PARAM.clone()),
                            self.stdlib.str().clone().to_type(),
                            Required::Required,
                        ),
                    ]),
                    Type::optional(value_ty.clone()),
                ),
                metadata: metadata.clone(),
            }),
        );
        signatures.push(self.get_overload_with_default(&metadata, &self_param, None, value_ty));
        ClassSynthesizedField::new(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    fn name_or_str(&self, name: Option<&Name>) -> Type {
        if let Some(name) = name {
            name_to_literal_type(name)
        } else {
            self.stdlib.str().clone().to_type()
        }
    }

    fn key_param(&self, name: Option<&Name>) -> Param {
        Param::PosOnly(
            Some(KEY_PARAM.clone()),
            self.name_or_str(name),
            Required::Required,
        )
    }

    fn gen_pop_overloads_for_field(
        &self,
        metadata: &FuncMetadata,
        self_param: &Param,
        name: Option<&Name>,
        read_only: bool,
        required: bool,
        ty: Type,
        overloads: &mut Vec<OverloadType>,
    ) {
        if required || read_only {
            // do not pop required or read-only keys
            return;
        }

        // 1) no default: (self, key: Literal["field_name"]) -> FieldType
        overloads.push(OverloadType::Function(Function {
            signature: Callable::list(
                ParamList::new(vec![self_param.clone(), self.key_param(name)]),
                ty.clone(),
            ),
            metadata: metadata.clone(),
        }));

        // 2) default: (self, key: Literal["field_name"], default: _T) -> FieldType | _T
        overloads.push(self.get_overload_with_default(metadata, self_param, name, ty));
    }

    /// Synthesize a method for every non-required field. Thus, this method returns None if all fields are required since no methods are synthesized
    fn get_typed_dict_pop(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), POP_METHOD);
        let self_param = self.class_self_param(cls, true);
        let mut literal_signatures: Vec<OverloadType> = Vec::new();
        for (name, field) in self.names_to_fields(cls, fields) {
            self.gen_pop_overloads_for_field(
                &metadata,
                &self_param,
                Some(name),
                field.is_read_only(),
                field.required,
                field.ty,
                &mut literal_signatures,
            );
        }
        if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(cls) {
            self.gen_pop_overloads_for_field(
                &metadata,
                &self_param,
                None,
                extra.read_only,
                false,
                self.get_typed_dict_value_type_from_fields(cls, fields),
                &mut literal_signatures,
            );
        }
        let signatures = Vec1::try_from_vec(literal_signatures).ok()?;
        Some(ClassSynthesizedField::new(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        })))
    }

    fn get_typed_dict_setdefault(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        // Synthesizes a `(self, k: Literal["key"], default: ValueType) -> ValueType` signature for each field.
        let fields_iter = self.names_to_fields(cls, fields);
        let self_param = self.class_self_param(cls, true);
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), SETDEFAULT_METHOD);
        let make_overload = |name: Option<&Name>, read_only: bool, field_ty: Type| {
            if read_only {
                None
            } else {
                Some(OverloadType::Function(Function {
                    signature: Callable::list(
                        ParamList::new(vec![
                            self_param.clone(),
                            self.key_param(name),
                            Param::PosOnly(
                                Some(DEFAULT_PARAM.clone()),
                                field_ty.clone(),
                                Required::Required,
                            ),
                        ]),
                        field_ty,
                    ),
                    metadata: metadata.clone(),
                }))
            }
        };
        let mut overloads = fields_iter
            .filter_map(|(name, field)| make_overload(Some(name), field.is_read_only(), field.ty))
            .collect::<Vec<_>>();
        if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(cls)
            && let Some(overload) = make_overload(
                None,
                extra.read_only,
                self.get_typed_dict_value_type_from_fields(cls, fields),
            )
        {
            overloads.push(overload);
        }
        Some(ClassSynthesizedField::new(Type::Overload(Overload {
            signatures: Vec1::try_from_vec(overloads).ok()?,
            metadata: Box::new(metadata),
        })))
    }

    pub fn get_typed_dict_value_type(&self, typed_dict: &TypedDict) -> Type {
        let cls = typed_dict.class_object();
        if let Some(metadata) = self.get_metadata_for_class(cls).typed_dict_metadata() {
            self.get_typed_dict_value_type_from_fields(cls, &metadata.fields)
        } else {
            self.stdlib.object().clone().to_type()
        }
    }

    /// If the TypedDict is assignable to `dict[str, VT]`, return `VT`.
    pub fn get_typed_dict_value_type_as_builtins_dict(
        &self,
        typed_dict: &TypedDict,
    ) -> Option<Type> {
        if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(typed_dict.class_object())
            && !extra.read_only
            && self.typed_dict_fields(typed_dict).values().all(|field| {
                !field.is_read_only() && !field.required && self.is_equal(&field.ty, &extra.ty)
            })
        {
            Some(extra.ty)
        } else {
            None
        }
    }

    /// Get the type of a value in the TypedDict.
    fn get_typed_dict_value_type_from_fields(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Type {
        let extra_items = self.typed_dict_extra_items(cls);
        if matches!(extra_items, ExtraItems::Default) {
            // extra_items defaults to `ReadOnly[object]`, and `object | ...` simplifies to `object`.
            return self.stdlib.object().clone().to_type();
        }
        let extra = extra_items.extra_item(self.stdlib).ty;
        let mut values = self
            .names_to_fields(cls, fields)
            .map(|(_, field)| field.ty)
            .collect::<Vec<_>>();
        // We can't use self.unions(...) here because it calls is_subset_eq, which may need to get
        // a TypedDict's value type, which would lead to infinite recursion if the TypedDict
        // contains itself.
        if values.is_empty() {
            extra
        } else {
            values.push(extra);
            unions_with_literals(values, self.stdlib, &|cls| self.get_enum_member_count(cls))
        }
    }

    /// Synthesize an `items()` method.
    fn get_typed_dict_items(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let dict_items = self.stdlib.dict_items(
            self.stdlib.str().clone().to_type(),
            self.get_typed_dict_value_type_from_fields(cls, fields),
        );
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), ITEMS_METHOD);
        ClassSynthesizedField::new(Type::Function(Box::new(Function {
            signature: Callable::list(
                ParamList::new(vec![self.class_self_param(cls, false)]),
                dict_items.to_type(),
            ),
            metadata,
        })))
    }

    /// Synthesize a `values()` method.
    fn get_typed_dict_values(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let dict_values = self.stdlib.dict_values(
            self.stdlib.str().clone().to_type(),
            self.get_typed_dict_value_type_from_fields(cls, fields),
        );
        let metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), VALUES_METHOD);
        ClassSynthesizedField::new(Type::Function(Box::new(Function {
            signature: Callable::list(
                ParamList::new(vec![self.class_self_param(cls, false)]),
                dict_values.to_type(),
            ),
            metadata,
        })))
    }

    fn all_items_are_removable(&self, cls: &Class, fields: &SmallMap<Name, bool>) -> bool {
        !self
            .typed_dict_extra_items(cls)
            .extra_item(self.stdlib)
            .read_only
            && self
                .names_to_fields(cls, fields)
                .all(|(_, field)| !field.is_read_only() && !field.required)
    }

    fn get_typed_dict_clear(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        if !self.all_items_are_removable(cls, fields) {
            return None;
        }
        Some(ClassSynthesizedField::new(Type::Function(Box::new(
            Function {
                signature: Callable::list(
                    ParamList::new(vec![self.class_self_param(cls, true)]),
                    Type::None,
                ),
                metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), CLEAR_METHOD),
            },
        ))))
    }

    fn get_typed_dict_popitem(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        if !self.all_items_are_removable(cls, fields) {
            return None;
        }
        Some(ClassSynthesizedField::new(Type::Function(Box::new(
            Function {
                signature: Callable::list(
                    ParamList::new(vec![self.class_self_param(cls, true)]),
                    Type::Tuple(Tuple::Concrete(vec![
                        self.stdlib.str().clone().to_type(),
                        self.get_typed_dict_value_type_from_fields(cls, fields),
                    ])),
                ),
                metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), POPITEM_METHOD),
            },
        ))))
    }

    pub fn get_typed_dict_synthesized_fields(&self, cls: &Class) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let td = metadata.typed_dict_metadata()?;
        let mut fields = smallmap! {
            dunder::INIT => self.get_typed_dict_init(cls, &td.fields),
            ITEMS_METHOD => self.get_typed_dict_items(cls, &td.fields),
            GET_METHOD => self.get_typed_dict_get(cls, &td.fields),
            UPDATE_METHOD => self.get_typed_dict_update(cls, &td.fields),
            VALUES_METHOD => self.get_typed_dict_values(cls, &td.fields),
        };
        if let Some(m) = self.get_typed_dict_clear(cls, &td.fields) {
            fields.insert(CLEAR_METHOD, m);
        }
        if let Some(m) = self.get_typed_dict_pop(cls, &td.fields) {
            fields.insert(POP_METHOD, m);
        }
        if let Some(m) = self.get_typed_dict_popitem(cls, &td.fields) {
            fields.insert(POPITEM_METHOD, m);
        }
        if let Some(m) = self.get_typed_dict_setdefault(cls, &td.fields) {
            fields.insert(SETDEFAULT_METHOD, m);
        }
        Some(ClassSynthesizedFields::new(fields))
    }

    pub fn typed_dict_kw_param_info(&self, typed_dict: &TypedDict) -> Vec<(Name, Type, Required)> {
        self.typed_dict_fields(typed_dict)
            .iter()
            .map(|(name, field)| {
                (
                    name.clone(),
                    field.ty.clone(),
                    if field.required {
                        Required::Required
                    } else {
                        Required::Optional(None)
                    },
                )
            })
            .collect()
    }
}

fn name_to_literal_type(name: &Name) -> Type {
    Type::Literal(Lit::Str(name.as_str().into()))
}
