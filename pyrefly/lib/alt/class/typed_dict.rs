/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::dunder;
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
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
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

const GET_METHOD: Name = Name::new_static("get");
const POP_METHOD: Name = Name::new_static("pop");
const SETDEFAULT_METHOD: Name = Name::new_static("setdefault");
const KEY_PARAM: Name = Name::new_static("key");
const DEFAULT_PARAM: Name = Name::new_static("default");
const UPDATE_METHOD: Name = Name::new_static("update");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn check_dict_items_against_typed_dict(
        &self,
        dict_items: Vec<&DictItem>,
        typed_dict: &TypedDict,
        // Check whether `typed_dict` can be updated with `dict_items`
        is_update: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let fields = self.typed_dict_fields(typed_dict);
        let mut has_expansion = false;
        let mut keys: SmallSet<Name> = SmallSet::new();
        dict_items.iter().for_each(|x| match &x.key {
            Some(key) => {
                let key_type = self.expr_infer(key, errors);
                if let Type::Literal(Lit::Str(name)) = key_type {
                    let key_name = Name::new(name);
                    match fields.get(&key_name) {
                        Some(field) if is_update && field.is_read_only() => {
                            self.error(
                                errors,
                                key.range(),
                                ErrorInfo::Kind(ErrorKind::TypedDictKeyError),
                                format!("Cannot update read-only field `{key_name}`"),
                            );
                        }
                        Some(field) => {
                            self.expr(
                                &x.value,
                                Some((&field.ty, &|| {
                                    TypeCheckContext::of_kind(TypeCheckKind::TypedDictKey(
                                        key_name.clone(),
                                    ))
                                })),
                                errors,
                            );
                        }
                        None => {
                            self.error(
                                errors,
                                key.range(),
                                ErrorInfo::Kind(ErrorKind::TypedDictKeyError),
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
                        errors,
                        key.range(),
                        ErrorInfo::Kind(ErrorKind::TypedDictKeyError),
                        format!(
                            "Expected string literal key, got `{}`",
                            self.for_display(key_type)
                        ),
                    );
                }
            }
            None => {
                has_expansion = true;
                self.expr(
                    &x.value,
                    Some((&Type::TypedDict(typed_dict.clone()), &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::TypedDictUnpacking)
                    })),
                    errors,
                );
            }
        });
        // You can update a TypedDict with a subset of its items. Otherwise, all required fields must be present.
        if !has_expansion && !is_update {
            for (key, field) in &fields {
                if field.required && !keys.contains(key) {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::TypedDictKeyError),
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
                .map(|field| field.substitute(substitution))
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
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), Type::None),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_typed_dict_update(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let metadata = FuncMetadata::def(self.module().name(), cls.name().clone(), UPDATE_METHOD);

        let self_param = self.class_self_param(cls, true);

        // ---- Overload: def update(__m: Partial[C], /)
        let full_typed_dict = self.as_typed_dict_unchecked(cls);
        let partial_typed_dict_ty = Type::PartialTypedDict(full_typed_dict);

        let partial_overload = OverloadType::Callable(Function {
            signature: Callable::list(
                ParamList::new(vec![
                    self_param.clone(),
                    Param::PosOnly(
                        Some(Name::new_static("__m")),
                        partial_typed_dict_ty,
                        Required::Required,
                    ),
                ]),
                Type::None,
            ),
            metadata: metadata.clone(),
        });

        // ---- Overload: update(__m: Iterable[tuple[Literal["key"], value]])
        let tuple_types: Vec<Type> = self
            .names_to_fields(cls, fields)
            .filter(|(_, field)| !field.is_read_only()) // filter read-only fields
            .map(|(name, field)| {
                Type::Tuple(Tuple::Concrete(vec![
                    name_to_literal_type(name),
                    field.ty.clone(),
                ]))
            })
            .collect();

        let iterable_ty = self.stdlib.iterable(self.unions(tuple_types)).to_type();

        let tuple_overload = OverloadType::Callable(Function {
            signature: Callable::list(
                ParamList::new(vec![
                    self_param.clone(),
                    Param::PosOnly(
                        Some(Name::new_static("__m")),
                        iterable_ty,
                        Required::Required,
                    ),
                ]),
                Type::None,
            ),
            metadata: metadata.clone(),
        });

        // ---- Overload: update(*, x=..., y=...)
        let keyword_params: Vec<_> = self
            .names_to_fields(cls, fields)
            .filter(|(_, field)| !field.is_read_only()) // filter read-only fields
            .map(|(name, field)| {
                Param::KwOnly(name.clone(), field.ty.clone(), Required::Optional(None))
            })
            .collect();

        let overload_kwargs = OverloadType::Callable(Function {
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

    fn get_typed_dict_get(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> ClassSynthesizedField {
        let metadata = FuncMetadata::def(self.module().name(), cls.name().clone(), GET_METHOD);
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
                literal_signatures.push(OverloadType::Callable(Function {
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
                literal_signatures.push(OverloadType::Callable(Function {
                    signature: Callable::list(
                        ParamList::new(vec![self_param.clone(), key_param.clone()]),
                        Type::optional(field.ty.clone()),
                    ),
                    metadata: metadata.clone(),
                }));
                // (self, key: Literal["key"], default: T) -> ValueType | T
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

                literal_signatures.push(OverloadType::Forall(Forall {
                    tparams: Arc::new(TParams::new(tparams)),
                    body: Function {
                        signature: Callable::list(
                            ParamList::new(vec![
                                self_param.clone(),
                                key_param.clone(),
                                Param::PosOnly(
                                    Some(DEFAULT_PARAM.clone()),
                                    q.clone().to_type(),
                                    Required::Required,
                                ),
                            ]),
                            self.union(field.ty.clone(), q.to_type()),
                        ),
                        metadata: metadata.clone(),
                    },
                }));
            }
        }
        let signatures = Vec1::from_vec_push(
            literal_signatures,
            OverloadType::Callable(Function {
                signature: Callable::list(
                    ParamList::new(vec![
                        self_param.clone(),
                        Param::PosOnly(
                            Some(KEY_PARAM.clone()),
                            self.stdlib.str().clone().to_type(),
                            Required::Required,
                        ),
                        Param::PosOnly(
                            Some(DEFAULT_PARAM.clone()),
                            object_ty.clone(),
                            Required::Optional(None),
                        ),
                    ]),
                    object_ty.clone(),
                ),
                metadata: metadata.clone(),
            }),
        );
        ClassSynthesizedField::new(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    /// Synthesize a method for every non-required field. Thus, this method returns None if all fields are required since no methods are synthesized
    fn get_typed_dict_pop(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        let metadata = FuncMetadata::def(self.module().name(), cls.name().clone(), POP_METHOD);
        let self_param = self.class_self_param(cls, true);

        let mut literal_signatures: Vec<OverloadType> = Vec::new();
        for (name, field) in self.names_to_fields(cls, fields) {
            if field.required {
                // do not pop required keys
                continue;
            } else {
                let key_param = Param::PosOnly(
                    Some(KEY_PARAM.clone()),
                    name_to_literal_type(name),
                    Required::Required,
                );

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

                // 1) no default: (self, key: Literal["field_name"]) -> Optional[FieldType]
                literal_signatures.push(OverloadType::Callable(Function {
                    signature: Callable::list(
                        ParamList::new(vec![self_param.clone(), key_param.clone()]),
                        Type::optional(field.ty.clone()),
                    ),
                    metadata: metadata.clone(),
                }));

                // 2) default: (self, key: Literal["field_name"], default: _T) -> Union[FieldType, _T]
                literal_signatures.push(OverloadType::Forall(Forall {
                    tparams: Arc::new(TParams::new(tparams.clone())),
                    body: Function {
                        signature: Callable::list(
                            ParamList::new(vec![
                                self_param.clone(),
                                key_param.clone(),
                                Param::PosOnly(
                                    Some(DEFAULT_PARAM.clone()),
                                    q.clone().to_type(),
                                    Required::Required,
                                ),
                            ]),
                            self.union(field.ty.clone(), q.clone().to_type()),
                        ),
                        metadata: metadata.clone(),
                    },
                }));
            }
        }

        let signatures = Vec1::try_from_vec(literal_signatures).ok()?;

        Some(ClassSynthesizedField::new(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        })))
    }

    fn get_typed_dict_delitem(
        &self,
        cls: &Class,
        fields: &SmallMap<Name, bool>,
    ) -> Option<ClassSynthesizedField> {
        let metadata = FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::DELITEM);

        let self_param = self.class_self_param(cls, true);

        let mut literal_signatures: Vec<OverloadType> = Vec::new();
        for (name, field) in self.names_to_fields(cls, fields) {
            if field.required {
                // Do not allow deletion of required keys
                // Call resolution will fall back to typeshed
                continue;
            }

            let key_param = Param::PosOnly(
                Some(KEY_PARAM.clone()),
                name_to_literal_type(name),
                Required::Required,
            );

            // Construct an an overload of the form: (self, key: Literal["field_name"]) -> None
            literal_signatures.push(OverloadType::Callable(Function {
                signature: Callable::list(
                    ParamList::new(vec![self_param.clone(), key_param]),
                    Type::None,
                ),
                metadata: metadata.clone(),
            }));
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
        let metadata =
            FuncMetadata::def(self.module().name(), cls.name().clone(), SETDEFAULT_METHOD);
        let make_overload = |(name, field): (&Name, TypedDictField)| {
            if field.is_read_only() {
                None
            } else {
                Some(OverloadType::Callable(Function {
                    signature: Callable::list(
                        ParamList::new(vec![
                            self_param.clone(),
                            Param::PosOnly(
                                Some(KEY_PARAM.clone()),
                                name_to_literal_type(name),
                                Required::Required,
                            ),
                            Param::PosOnly(
                                Some(DEFAULT_PARAM.clone()),
                                field.ty.clone(),
                                Required::Required,
                            ),
                        ]),
                        field.ty.clone(),
                    ),
                    metadata: metadata.clone(),
                }))
            }
        };
        let overloads = fields_iter.filter_map(make_overload).collect::<Vec<_>>();
        Some(ClassSynthesizedField::new(Type::Overload(Overload {
            signatures: Vec1::try_from_vec(overloads).ok()?,
            metadata: Box::new(metadata),
        })))
    }

    pub fn get_typed_dict_synthesized_fields(&self, cls: &Class) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let td = metadata.typed_dict_metadata()?;
        let mut fields = smallmap! {
            dunder::INIT => self.get_typed_dict_init(cls, &td.fields),
            GET_METHOD => self.get_typed_dict_get(cls, &td.fields),
            UPDATE_METHOD => self.get_typed_dict_update(cls, &td.fields),
        };
        if let Some(m) = self.get_typed_dict_pop(cls, &td.fields) {
            fields.insert(POP_METHOD, m);
        }
        if let Some(m) = self.get_typed_dict_delitem(cls, &td.fields) {
            fields.insert(dunder::DELITEM, m);
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
