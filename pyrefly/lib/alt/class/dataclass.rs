/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::dunder;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr::EllipsisLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::TargetWithTParams;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::class_field::DataclassMember;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::binding::pydantic::GE;
use crate::binding::pydantic::GT;
use crate::binding::pydantic::LT;
use crate::binding::pydantic::STRICT;
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
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::display::ClassDisplayContext;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::AnyStyle;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Gets dataclass fields for an `@dataclass`-decorated class.
    pub fn get_dataclass_fields(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
    ) -> SmallSet<Name> {
        let mut all_fields = SmallSet::new();
        for (_, metadata) in bases_with_metadata.iter().rev() {
            if let Some(dataclass) = metadata.dataclass_metadata() {
                all_fields.extend(dataclass.fields.clone());
            }
        }
        for name in cls.fields() {
            if cls.is_field_annotated(name) {
                all_fields.insert(name.clone());
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

        if dataclass.kws.init {
            let init_method = if let Some(root_model_type) =
                self.get_pydantic_root_model_type_via_mro(cls, &metadata)
            {
                self.get_pydantic_root_model_init(cls, root_model_type)
            } else {
                self.get_dataclass_init(cls, dataclass, dataclass.kws.strict, errors)
            };
            fields.insert(dunder::INIT, init_method);
        }
        let dataclass_fields_type = self.stdlib.dict(
            self.stdlib.str().clone().to_type(),
            Type::Any(AnyStyle::Implicit),
        );
        fields.insert(
            dunder::DATACLASS_FIELDS,
            ClassSynthesizedField::new(dataclass_fields_type.to_type()),
        );

        if dataclass.kws.order {
            fields.extend(self.get_dataclass_rich_comparison_methods(cls));
        }
        if dataclass.kws.match_args {
            fields.insert(
                dunder::MATCH_ARGS,
                self.get_dataclass_match_args(cls, dataclass),
            );
        }
        if dataclass.kws.slots {
            // It's a runtime error to set slots=True on a class that already defines __slots__.
            // Note that inheriting __slots__ from a base class is fine.
            if cls.contains(&dunder::SLOTS) {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                    "Cannot specify both `slots=True` and `__slots__`".to_owned(),
                );
            } else {
                fields.insert(dunder::SLOTS, self.get_dataclass_slots(cls, dataclass));
            }
        }
        // See rules for `__hash__` creation under "unsafe_hash":
        // https://docs.python.org/3/library/dataclasses.html#module-contents
        if dataclass.kws.unsafe_hash || (dataclass.kws.eq && dataclass.kws.frozen) {
            fields.insert(dunder::HASH, self.get_dataclass_hash(cls));
        } else if dataclass.kws.eq {
            fields.insert(dunder::HASH, ClassSynthesizedField::new(Type::None));
        }
        Some(ClassSynthesizedFields::new(fields))
    }

    pub fn validate_frozen_dataclass_inheritance(
        &self,
        cls: &Class,
        dataclass_metadata: &DataclassMetadata,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        errors: &ErrorCollector,
    ) {
        for (base, base_metadata) in bases_with_metadata {
            if let Some(base_dataclass_metadata) = base_metadata.dataclass_metadata() {
                let is_base_frozen = base_dataclass_metadata.kws.frozen;
                let is_current_frozen = dataclass_metadata.kws.frozen;

                if is_current_frozen != is_base_frozen {
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
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
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
        let mut params = Vec::new();
        for (name, field, _) in self.iter_fields(cls, dataclass_metadata, true) {
            if field.is_init_var() {
                params.push(self.as_param(&field, &name, false, false, true, None, errors));
            }
        }
        let want = Type::Callable(Box::new(Callable::list(
            ParamList::new(params),
            self.stdlib.object().clone().to_type(),
        )));
        self.check_type(&post_init, &want, range, errors, &|| {
            TypeCheckContext::of_kind(TypeCheckKind::PostInit)
        });
    }

    pub fn dataclass_field_keywords(
        &self,
        func: &Type,
        args: &Arguments,
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
        let mut default = [
            &DataclassFieldKeywords::DEFAULT,
            &DataclassFieldKeywords::DEFAULT_FACTORY,
            &DataclassFieldKeywords::FACTORY,
        ]
        .iter()
        .any(|k| map.0.contains_key(*k));

        if !default && dataclass_metadata.default_can_be_positional {
            // Check whether a default was passed positionally. This is needed for `pydantic.Field`.
            default = !matches!(args.args.first(), Some(EllipsisLiteral(_)) | None);
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

        let strict: Option<bool> = map.0.get(&STRICT).and_then(|v| v.as_bool());

        let mut converter_param = map
            .0
            .get(&DataclassFieldKeywords::CONVERTER)
            .map(|converter| self.get_converter_param(converter));
        // Note that we intentionally don't try to fill in `default`, since we can't distinguish
        // between a real default and something like `dataclasses.MISSING`.
        if init.is_none() || kw_only.is_none() || alias.is_none() || converter_param.is_none() {
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
                &mut converter_param,
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
            strict,
            converter_param,
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
        converter_param: &mut Option<Type>,
    ) {
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
                (*overload.metadata).clone(),
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
                        (name, ty, default.as_ref())
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
                        Type::Literal(Lit::Str(s)) => Some(Name::new(s)),
                        _ => None,
                    });
                }
                if converter_param.is_none() && name == &DataclassFieldKeywords::CONVERTER {
                    *converter_param = Some(self.get_converter_param(ty));
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

    fn get_converter_param(&self, converter: &Type) -> Type {
        let converter = {
            if let Type::ClassDef(cls) = converter
                && let Type::ClassType(instance) = self.promote_silently(cls)
            {
                let callable = self.constructor_to_callable(&instance);
                &self.distribute_over_union(&callable, |ty| {
                    if let Type::BoundMethod(m) = ty {
                        self.bind_boundmethod(m, &mut |a, b| self.is_subset_eq(a, b))
                            .unwrap_or_else(|| ty.clone())
                    } else {
                        ty.clone()
                    }
                })
            } else {
                converter
            }
        };
        self.distribute_over_union(converter, |ty| {
            ty.callable_first_param().unwrap_or_else(Type::any_implicit)
        })
    }

    fn iter_fields(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        include_initvar: bool,
    ) -> Vec<(Name, ClassField, DataclassFieldKeywords)> {
        let mut seen_kw_only_marker = false;
        let mut positional_fields = Vec::new();
        let mut kwonly_fields = Vec::new();
        let cls_is_kw_only = dataclass.kws.kw_only;
        for name in dataclass.fields.iter() {
            match (self.get_dataclass_member(cls, name), include_initvar) {
                (DataclassMember::KwOnlyMarker, _) => {
                    seen_kw_only_marker = true;
                }
                (DataclassMember::NotAField, _) => {}
                (DataclassMember::Field(field, mut keywords), _)
                | (DataclassMember::InitVar(field, mut keywords), true) => {
                    if keywords.kw_only.is_none() {
                        // kw_only hasn't been explicitly set on the field
                        keywords.kw_only = Some(
                            seen_kw_only_marker
                                || if field.defining_class == *cls {
                                    cls_is_kw_only
                                } else {
                                    self.get_metadata_for_class(&field.defining_class)
                                        .dataclass_metadata()
                                        .is_some_and(|m| m.kws.kw_only)
                                },
                        );
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

    /// Gets __init__ method for an `@dataclass`-decorated class.
    fn get_dataclass_init(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        strict_default: bool,
        errors: &ErrorCollector,
    ) -> ClassSynthesizedField {
        let mut params = vec![self.class_self_param(cls, false)];
        let mut has_seen_default = false;
        for (name, field, field_flags) in self.iter_fields(cls, dataclass, true) {
            let strict = field_flags.strict.unwrap_or(strict_default);
            if field_flags.init {
                let has_default = field_flags.default
                    || (field_flags.init_by_name && field_flags.init_by_alias.is_some());
                let is_kw_only = field_flags.is_kw_only();
                if !is_kw_only {
                    if !has_default
                        && has_seen_default
                        && let Some(range) = cls.field_decl_range(&name)
                    {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                            format!(
                                "Dataclass field `{name}` without a default may not follow dataclass field with a default"
                            ),
                        );
                    }
                    if has_default {
                        has_seen_default = true;
                    }
                }
                if field_flags.init_by_name {
                    params.push(self.as_param(
                        &field,
                        &name,
                        has_default,
                        is_kw_only,
                        strict,
                        field_flags.converter_param.clone(),
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
                        field_flags.converter_param.clone(),
                        errors,
                    ));
                }
            }
        }
        if dataclass.kws.extra {
            params.push(Param::Kwargs(None, Type::Any(AnyStyle::Implicit)));
        }

        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), Type::None),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_match_args(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
    ) -> ClassSynthesizedField {
        // Keyword-only fields do not appear in __match_args__.
        let kw_only = dataclass.kws.kw_only;
        let ts = if kw_only {
            Vec::new()
        } else {
            let filtered_fields = self.iter_fields(cls, dataclass, true);
            filtered_fields
                .iter()
                .filter_map(|(name, _, field_flags)| {
                    if field_flags.is_kw_only() || !field_flags.init {
                        None
                    } else {
                        Some(Type::Literal(Lit::Str(name.as_str().into())))
                    }
                })
                .collect()
        };
        let ty = Type::Tuple(Tuple::Concrete(ts));
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_slots(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
    ) -> ClassSynthesizedField {
        let filtered_fields = self.iter_fields(cls, dataclass, false);
        let ts = filtered_fields
            .iter()
            .map(|(name, _, _)| Type::Literal(Lit::Str(name.as_str().into())))
            .collect();
        let ty = Type::Tuple(Tuple::Concrete(ts));
        ClassSynthesizedField::new(ty)
    }

    fn get_dataclass_rich_comparison_methods(
        &self,
        cls: &Class,
    ) -> SmallMap<Name, ClassSynthesizedField> {
        let make_signature = |other_type| {
            let other = Param::Pos(Name::new_static("other"), other_type, Required::Required);
            Callable::list(
                ParamList::new(vec![self.class_self_param(cls, false), other]),
                self.stdlib.bool().clone().to_type(),
            )
        };
        let callable = make_signature(self.instantiate(cls));
        let callable_eq = make_signature(self.stdlib.object().clone().to_type());
        dunder::RICH_CMPS
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    ClassSynthesizedField::new(Type::Function(Box::new(Function {
                        signature: if *name == dunder::EQ || *name == dunder::NE {
                            callable_eq.clone()
                        } else {
                            callable.clone()
                        },
                        metadata: FuncMetadata::def(
                            self.module().name(),
                            cls.name().clone(),
                            name.clone(),
                        ),
                    }))),
                )
            })
            .collect()
    }

    fn get_dataclass_hash(&self, cls: &Class) -> ClassSynthesizedField {
        let params = vec![self.class_self_param(cls, false)];
        let ret = self.stdlib.int().clone().to_type();
        ClassSynthesizedField::new(Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), ret),
            metadata: FuncMetadata::def(self.module().name(), cls.name().clone(), dunder::HASH),
        })))
    }
}
