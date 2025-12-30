/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use itertools::Itertools;
use pyrefly_graph::index::Idx;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::annotation::Annotation;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::ExtraItem;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::types::Forallable;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::abstract_class::AbstractClassMembers;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::class_metadata::DjangoModelMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::alt::types::class_metadata::InitDefaults;
use crate::alt::types::class_metadata::Metaclass;
use crate::alt::types::class_metadata::NamedTupleMetadata;
use crate::alt::types::class_metadata::ProtocolMetadata;
use crate::alt::types::class_metadata::TotalOrderingMetadata;
use crate::alt::types::class_metadata::TypedDictMetadata;
use crate::alt::types::decorated_function::Decorator;
use crate::alt::types::pydantic::PydanticConfig;
use crate::binding::base_class::BaseClass;
use crate::binding::base_class::BaseClassExpr;
use crate::binding::base_class::BaseClassGeneric;
use crate::binding::base_class::BaseClassGenericKind;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecorator;
use crate::binding::pydantic::PydanticConfigDict;
use crate::binding::pydantic::VALIDATION_ALIAS;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::keywords::DataclassKeywords;
use crate::types::keywords::DataclassTransformMetadata;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

#[derive(Debug, Clone)]
struct ParsedBaseClass {
    class_object: Class,
    range: TextRange,
    metadata: Arc<ClassMetadata>,
}

#[derive(Debug, Clone)]
enum BaseClassParseResult {
    /// We can successfully extract the class object and metadata from the base class
    Parsed(ParsedBaseClass),
    /// We can't parse the base class because its corresponding `BaseClass` is not valid (e.g. base is a `TypedDict`
    /// when is_new_type is true)
    InvalidBase(TextRange),
    /// We can't parse the base class because its expression is not recognized to be a valid base class expression
    InvalidExpr(Expr),
    /// We can't parse the base class because its type is not valid to be put in the base class list
    InvalidType(Type, TextRange),
    /// We can't parse the base class but we also don't want to error on it for some reason (e.g. the error
    /// will be reported elsewhere, or the base class literally just has the `Any` type)
    AnyType,
    /// This base class does not participate in inheritance related computation (e.g. `Generic`, `Protocol`, etc.)
    Ignored,
}

impl BaseClassParseResult {
    fn is_any(&self) -> bool {
        match self {
            BaseClassParseResult::InvalidBase(..)
            | BaseClassParseResult::InvalidExpr(..)
            | BaseClassParseResult::InvalidType(..)
            | BaseClassParseResult::AnyType => true,
            _ => false,
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn class_metadata_of(
        &self,
        cls: &Class,
        bases: &[BaseClass],
        keywords: &[(Name, Expr)],
        decorators: &[Idx<KeyDecorator>],
        is_new_type: bool,
        pydantic_config_dict: &PydanticConfigDict,
        django_primary_key_field: Option<&Name>,
        errors: &ErrorCollector,
    ) -> ClassMetadata {
        // Get class decorators.
        let decorators = decorators.map(|decorator_key| {
            (
                self.get_idx(*decorator_key),
                self.bindings().idx_to_key(*decorator_key).range(),
            )
        });

        // Compute data that depends on the `BaseClass` representation of base classes.
        let initial_protocol_metadata = Self::initial_protocol_metadata(cls, bases);
        let has_generic_base_class = bases.iter().any(|x| x.is_generic());
        let has_typed_dict_base_class = bases.iter().any(|x| x.is_typed_dict());

        // Parse base classes and compute data that depends on the `BaseClassParseResult`
        // representation of base classes.
        let parsed_results = bases
            .iter()
            .map(|x| self.parse_base_class(x, is_new_type))
            .collect::<Vec<_>>();
        let contains_base_class_any = parsed_results.iter().any(|x| x.is_any());
        let protocol_metadata = self.final_protocol_metadata(
            initial_protocol_metadata,
            &decorators,
            &parsed_results,
            errors,
        );

        // Compute base classes with metadata.
        let bases_with_metadata = self.bases_with_metadata(parsed_results, is_new_type, errors);

        // Compute class keywords, including the metaclass.
        let (metaclasses, keywords): (Vec<_>, Vec<(_, _)>) =
            keywords.iter().partition_map(|(n, x)| match n.as_str() {
                "metaclass" => Either::Left(x),
                _ => Either::Right((n.clone(), self.expr_class_keyword(x, errors))),
            });

        let base_metaclasses = bases_with_metadata
            .iter()
            .filter_map(|(b, metadata)| metadata.custom_metaclass().map(|m| (b.name(), m)))
            .collect::<Vec<_>>();
        let calculated_metaclass = self.calculate_metaclass(
            cls,
            metaclasses.into_iter().next(),
            &base_metaclasses,
            errors,
        );
        let metaclass = calculated_metaclass.get();
        if let Some(metaclass) = &metaclass {
            self.check_base_class_metaclasses(cls, metaclass, &base_metaclasses, errors);
            if metaclass.targs().as_slice().iter().any(|targ| {
                targ.any(|ty| {
                    matches!(
                        ty,
                        Type::TypeVar(_) | Type::TypeVarTuple(_) | Type::ParamSpec(_)
                    )
                })
            }) {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                    "Metaclass may not be an unbound generic".to_owned(),
                );
            }
        }

        let mut directly_inherits_model = false;
        let mut inherited_django_metadata: Option<&DjangoModelMetadata> = None;

        // TODO Zeina: This pattern is repeated a lot in this file. See if we can refactor it (BE).
        for (base_class_object, metadata) in &bases_with_metadata {
            if base_class_object.has_toplevel_qname(ModuleName::django_models().as_str(), "Model") {
                directly_inherits_model = true;
            }

            if let Some(dm) = metadata.django_model_metadata() {
                inherited_django_metadata = Some(dm);
            }
        }

        let django_model_metadata =
            if directly_inherits_model || inherited_django_metadata.is_some() {
                Some(DjangoModelMetadata {
                    custom_primary_key_field: django_primary_key_field.cloned().or_else(|| {
                        inherited_django_metadata.and_then(|dm| dm.custom_primary_key_field.clone())
                    }),
                })
            } else {
                None
            };

        // Compute various pieces of special metadata.
        let has_base_any = contains_base_class_any
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.has_base_any());

        let named_tuple_metadata = self.named_tuple_metadata(cls, &bases_with_metadata, errors);
        if named_tuple_metadata.is_some() && bases_with_metadata.len() > 1 {
            self.error(
                errors,
                cls.range(),
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                "Named tuples do not support multiple inheritance".to_owned(),
            );
        }

        let pydantic_config = self.pydantic_config(
            &bases_with_metadata,
            pydantic_config_dict,
            &keywords,
            errors,
            cls.range(),
        );

        let is_typed_dict = has_typed_dict_base_class
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.is_typed_dict());
        if is_typed_dict
            && let Some(bad) = bases_with_metadata.iter().find(|x| !x.1.is_typed_dict())
        {
            self.error(errors,
                cls.range(),
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                format!("`{}` is not a typed dictionary. Typed dictionary definitions may only extend other typed dictionaries.", bad.0.name()),
            );
        }
        let typed_dict_metadata =
            self.typed_dict_metadata(cls, &bases_with_metadata, &keywords, is_typed_dict, errors);
        if metaclass.is_some() && is_typed_dict {
            self.error(
                errors,
                cls.range(),
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                "Typed dictionary definitions may not specify a metaclass".to_owned(),
            );
        }

        let enum_metadata = self.enum_metadata(cls, metaclass, &bases_with_metadata, errors);

        let is_final = decorators.iter().any(|(decorator, _)| {
            decorator.ty.callee_kind() == Some(CalleeKind::Function(FunctionKind::Final))
        });
        let deprecation = decorators
            .iter()
            .find_map(|(decorator, _)| decorator.deprecation.clone());
        let is_disjoint_base = decorators.iter().any(|(decorator, _)| {
            decorator.ty.callee_kind() == Some(CalleeKind::Function(FunctionKind::DisjointBase))
        });

        let total_ordering_metadata = decorators.iter().find_map(|(decorator, decorator_range)| {
            decorator.ty.callee_kind().and_then(|kind| {
                if kind == CalleeKind::Function(FunctionKind::TotalOrdering) {
                    Some(TotalOrderingMetadata {
                        location: *decorator_range,
                    })
                } else {
                    None
                }
            })
        });

        // If this class inherits from a dataclass_transform-ed class, record the defaults that we
        // should use for dataclass parameters.
        let dataclass_defaults_from_base_class = bases_with_metadata
            .iter()
            .find_map(|(_, metadata)| metadata.dataclass_transform_metadata().cloned());
        let dataclass_transform_metadata = self.dataclass_transform_metadata(
            &decorators,
            metaclass,
            dataclass_defaults_from_base_class.clone(),
        );
        let dataclass_from_dataclass_transform = self.dataclass_from_dataclass_transform(
            &keywords,
            &decorators,
            dataclass_defaults_from_base_class,
            pydantic_config.as_ref(),
        );
        let dataclass_metadata = self.dataclass_metadata(
            cls,
            &decorators,
            &bases_with_metadata,
            dataclass_from_dataclass_transform,
            pydantic_config.as_ref(),
        );
        if let Some(dm) = dataclass_metadata.as_ref()
            && pydantic_config.is_none()
        {
            self.validate_frozen_dataclass_inheritance(cls, dm, &bases_with_metadata, errors);
        }
        let extends_abc = self.extends_abc(&bases_with_metadata, metaclass);

        // Compute final base class list.
        let bases = if is_typed_dict && bases_with_metadata.is_empty() {
            // This is a "fallback" class that contains attributes that are available on all TypedDict subclasses.
            // Note that this also makes those attributes available on *instances* of said subclasses; this is
            // desirable for methods but problematic for fields like `__total__` that should be available on the class
            // but not the instance. For now, we make all fields available on both classes and instances.
            let td_fallback = self.stdlib.typed_dict_fallback();
            vec![td_fallback.class_object().clone()]
        } else {
            bases_with_metadata
                .into_iter()
                .map(|(base, _)| base)
                .collect::<Vec<_>>()
        };

        // Get types of class keywords.
        let keywords =
            keywords.into_map(|(name, annot)| (name, annot.ty.unwrap_or_else(Type::any_implicit)));

        // get pydantic model info. A root model is by default also a base model, while not every base model is a root model
        let pydantic_model_kind = pydantic_config
            .as_ref()
            .map(|m| m.pydantic_model_kind.clone());

        ClassMetadata::new(
            bases,
            calculated_metaclass,
            keywords,
            typed_dict_metadata,
            named_tuple_metadata,
            enum_metadata,
            protocol_metadata,
            dataclass_metadata,
            extends_abc,
            has_generic_base_class,
            has_base_any,
            is_new_type,
            is_final,
            deprecation,
            is_disjoint_base,
            total_ordering_metadata,
            dataclass_transform_metadata,
            pydantic_model_kind,
            django_model_metadata,
        )
    }

    fn initial_protocol_metadata(cls: &Class, bases: &[BaseClass]) -> Option<ProtocolMetadata> {
        if bases.iter().any(|x| {
            matches!(
                x,
                BaseClass::Generic(BaseClassGeneric {
                    kind: BaseClassGenericKind::Protocol,
                    ..
                })
            )
        }) {
            Some(ProtocolMetadata {
                members: cls.fields().cloned().collect(),
                is_runtime_checkable: false,
            })
        } else {
            None
        }
    }

    fn final_protocol_metadata(
        &self,
        mut protocol_metadata: Option<ProtocolMetadata>,
        decorators: &[(Arc<Decorator>, TextRange)],
        parsed_results: &[BaseClassParseResult],
        errors: &ErrorCollector,
    ) -> Option<ProtocolMetadata> {
        if let Some(proto) = &mut protocol_metadata {
            for base in parsed_results.iter() {
                if let BaseClassParseResult::Parsed(ParsedBaseClass {
                    class_object: _,
                    range,
                    metadata,
                }) = base
                {
                    if let Some(base_proto) = metadata.protocol_metadata() {
                        proto.members.extend(base_proto.members.iter().cloned());
                        if base_proto.is_runtime_checkable {
                            proto.is_runtime_checkable = true;
                        }
                    } else {
                        self.error(errors,
                            *range,
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            "If `Protocol` is included as a base class, all other bases must be protocols".to_owned(),
                        );
                    }
                }
            }
        }
        for (decorator, range) in decorators {
            match decorator.ty.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::RuntimeCheckable)) => {
                    if let Some(proto) = &mut protocol_metadata {
                        proto.is_runtime_checkable = true;
                    } else {
                        self.error(
                            errors,
                            *range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "@runtime_checkable can only be applied to Protocol classes".to_owned(),
                        );
                    }
                }
                _ => {}
            }
        }
        protocol_metadata
    }

    fn named_tuple_metadata(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        errors: &ErrorCollector,
    ) -> Option<NamedTupleMetadata> {
        bases_with_metadata
            .iter()
            .find_map(|(base_class_object, metadata)| {
                if base_class_object.has_toplevel_qname(
                    ModuleName::type_checker_internals().as_str(),
                    "NamedTupleFallback",
                ) {
                    Some(NamedTupleMetadata {
                        elements: self.get_named_tuple_elements(cls, errors),
                    })
                } else {
                    metadata.named_tuple_metadata().cloned()
                }
            })
    }

    fn typed_dict_metadata(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        keywords: &[(Name, Annotation)],
        is_typed_dict: bool,
        errors: &ErrorCollector,
    ) -> Option<TypedDictMetadata> {
        if is_typed_dict {
            // Validate and extract the values of class keywords.
            let mut is_total = true;
            let mut extra_items = None;
            for (name, value) in keywords {
                match (name.as_str(), value.get_type()) {
                    ("total", Type::Literal(Lit::Bool(false))) => {
                        is_total = false;
                    }
                    ("closed" | "extra_items", _) if extra_items.is_some() => {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorInfo::Kind(ErrorKind::BadTypedDict),
                            "TypedDict keywords `closed` and `extra_items` cannot be used together"
                                .to_owned(),
                        );
                    }
                    ("closed", Type::Literal(Lit::Bool(true))) => {
                        extra_items = Some(ExtraItems::Closed);
                    }
                    ("closed", Type::Literal(Lit::Bool(false))) => {
                        // Note that we need to distinguish between explicitly setting and
                        // implicitly defaulting to `closed=False` in order to catch illegal
                        // attempts to open a closed TypedDict.
                        extra_items = Some(ExtraItems::Default);
                    }
                    ("extra_items", value_ty) => {
                        let ty = self.untype_opt(value_ty.clone(), cls.range(), errors).unwrap_or_else(|| {
                            self.error(
                                errors,
                                cls.range(),
                                ErrorInfo::Kind(ErrorKind::BadTypedDict),
                                format!("Expected `extra_items` to be a type form, got instance of `{}`", self.for_display(value_ty.clone())),
                            )
                        });
                        extra_items = Some(ExtraItems::extra(ty, &value.qualifiers));
                    }
                    ("total", Type::Literal(Lit::Bool(_))) => {}
                    ("total" | "closed", value_ty) => {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorInfo::Kind(ErrorKind::BadTypedDict),
                            format!(
                                "Expected literal True or False for keyword `{}`, got instance of `{}`",
                                name,
                                self.for_display(value_ty.clone())
                            ),
                        );
                    }
                    _ => {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorInfo::Kind(ErrorKind::BadTypedDict),
                            format!(
                                "TypedDict does not support keyword argument `{}`",
                                name.as_str()
                            ),
                        );
                    }
                }
            }
            let fields =
                self.calculate_typed_dict_metadata_fields(cls, bases_with_metadata, is_total);
            let extra_items = self.calculate_typed_dict_extra_items(
                extra_items,
                bases_with_metadata,
                cls.range(),
                errors,
            );
            Some(TypedDictMetadata {
                fields,
                extra_items,
            })
        } else {
            None
        }
    }

    fn calculate_typed_dict_extra_items(
        &self,
        cur_extra_items: Option<ExtraItems>,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> ExtraItems {
        let inherited_extra_items = bases_with_metadata.iter().find_map(|(base, metadata)| {
            metadata
                .typed_dict_metadata()
                .map(|td| (base, &td.extra_items))
        });
        if cur_extra_items.is_none() || inherited_extra_items.is_none() {
            return cur_extra_items.unwrap_or_else(|| {
                inherited_extra_items.map_or(ExtraItems::Default, |(_, extra)| extra.clone())
            });
        }
        let cur_extra_items = cur_extra_items.unwrap();
        let (base_typed_dict, inherited_extra_items) = inherited_extra_items.unwrap();
        match (&cur_extra_items, inherited_extra_items) {
            (ExtraItems::Default, ExtraItems::Closed | ExtraItems::Extra(_)) => {
                let base = if *inherited_extra_items == ExtraItems::Closed {
                    format!("closed TypedDict `{}`", base_typed_dict.name())
                } else {
                    format!("TypedDict `{}` with extra items", base_typed_dict.name())
                };
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDict),
                    format!("Non-closed TypedDict cannot inherit from {base}"),
                );
            }
            (
                ExtraItems::Closed,
                ExtraItems::Extra(ExtraItem {
                    read_only: false, ..
                }),
            ) => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDict),
                    format!("Closed TypedDict cannot inherit from TypedDict `{}` with non-read-only extra items", base_typed_dict.name()),
                );
            }
            (
                ExtraItems::Extra(ExtraItem { ty: cur_ty, .. }),
                ExtraItems::Extra(ExtraItem {
                    ty: inherited_ty,
                    read_only: false,
                }),
            ) if cur_ty != inherited_ty => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDict),
                    format!(
                        "Cannot change the non-read-only extra items type of TypedDict `{}`",
                        base_typed_dict.name()
                    ),
                );
            }
            _ => {}
        }
        cur_extra_items
    }

    fn enum_metadata(
        &self,
        cls: &Class,
        metaclass: Option<&ClassType>,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        errors: &ErrorCollector,
    ) -> Option<EnumMetadata> {
        let is_django = bases_with_metadata.iter().any(|(base, base_meta)| {
            base.has_toplevel_qname(ModuleName::django_models_enums().as_str(), "Choices")
                || base_meta
                    .enum_metadata()
                    .as_ref()
                    .is_some_and(|meta| meta.is_django)
        });

        if let Some(metaclass) = metaclass
            && self
                .as_superclass(metaclass, self.stdlib.enum_meta().class_object())
                .is_some()
        {
            // NOTE(grievejia): This may create potential cycle if metaclass is generic. Need to look into
            // whether it can be removed or not.
            if !self.get_class_tparams(cls).is_empty() {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                    "Enums may not be generic".to_owned(),
                );
            }
            Some(EnumMetadata {
                // A generic enum is an error, but we create Any type args anyway to handle it gracefully.
                cls: self.promote_nontypeddict_silently_to_classtype(cls),
                has_value: bases_with_metadata
                    .iter()
                    .any(|(base, _)| base.contains(&Name::new_static("_value_"))),
                is_flag: bases_with_metadata.iter().any(|(base, _)| {
                    self.is_subset_eq(
                        &Type::ClassType(self.promote_nontypeddict_silently_to_classtype(base)),
                        &Type::ClassType(self.stdlib.enum_flag().clone()),
                    )
                }),
                is_django,
            })
        } else {
            None
        }
    }

    fn dataclass_transform_metadata(
        &self,
        decorators: &[(Arc<Decorator>, TextRange)],
        metaclass: Option<&ClassType>,
        dataclass_defaults_from_base_class: Option<DataclassTransformMetadata>,
    ) -> Option<DataclassTransformMetadata> {
        // This is set when a class is decorated with `@typing.dataclass_transform(...)`. Note that
        // this does not turn the class into a dataclass! Instead, it becomes a special base class
        // (or metaclass) that turns child classes into dataclasses.
        let mut dataclass_transform_metadata = dataclass_defaults_from_base_class;
        if let Some(c) = metaclass
            && let Some(m) = self
                .get_metadata_for_class(c.class_object())
                .dataclass_transform_metadata()
        {
            dataclass_transform_metadata = Some(m.clone());
        }
        for (decorator, _) in decorators {
            // `@dataclass_transform(...)`
            if let Type::KwCall(call) = &decorator.ty
                && call.has_function_kind(FunctionKind::DataclassTransform)
            {
                dataclass_transform_metadata =
                    Some(DataclassTransformMetadata::from_type_map(&call.keywords));
            }
        }
        dataclass_transform_metadata
    }

    fn dataclass_from_dataclass_transform(
        &self,
        keywords: &[(Name, Annotation)],
        decorators: &[(Arc<Decorator>, TextRange)],
        dataclass_defaults_from_base_class: Option<DataclassTransformMetadata>,
        pydantic_config: Option<&PydanticConfig>,
    ) -> Option<(DataclassKeywords, Vec<CalleeKind>)> {
        // This is set when we should apply dataclass-like transformations to the class. The class
        // should be transformed if:
        // - it inherits from a base class decorated with `dataclass_transform(...)`, or
        // - it inherits from a base class whose metaclass is decorated with `dataclass_transform(...)`, or
        // - it is decorated with a decorator that is decorated with `dataclass_transform(...)`.
        let mut dataclass_from_dataclass_transform = None;
        if let Some(defaults) = dataclass_defaults_from_base_class {
            // This class inherits from a dataclass_transform-ed base class, so its keywords are
            // interpreted as dataclass keywords.
            let map = keywords
                .iter()
                .map(|(name, annot)| (name.clone(), annot.get_type().clone()))
                .collect::<OrderedMap<_, _>>();
            let mut kws = DataclassKeywords::from_type_map(&TypeMap(map), &defaults);

            // Inject pydantic model configuration
            if let Some(pydantic) = pydantic_config {
                kws.frozen = pydantic.frozen;
                kws.extra = pydantic.extra;
                kws.strict = pydantic.strict;
            }

            dataclass_from_dataclass_transform = Some((kws, defaults.field_specifiers));
        }
        for (decorator, _) in decorators {
            // `@foo` where `foo` is decorated with `@dataclass_transform(...)`
            if let Some(defaults) = decorator.ty.dataclass_transform_metadata() {
                dataclass_from_dataclass_transform = Some((
                    DataclassKeywords::from_type_map(&TypeMap::new(), &defaults),
                    defaults.field_specifiers,
                ));
            }
            // `@foo(...)` where `foo` is decorated with `@dataclass_transform(...)`
            else if let Type::KwCall(call) = &decorator.ty
                && let Some(defaults) = &call.func_metadata.flags.dataclass_transform_metadata
            {
                dataclass_from_dataclass_transform = Some((
                    DataclassKeywords::from_type_map(&call.keywords, defaults),
                    defaults.field_specifiers.clone(),
                ));
            }
        }
        dataclass_from_dataclass_transform
    }

    fn dataclass_metadata(
        &self,
        cls: &Class,
        decorators: &[(Arc<Decorator>, TextRange)],
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        dataclass_from_dataclass_transform: Option<(DataclassKeywords, Vec<CalleeKind>)>,
        pydantic_config: Option<&PydanticConfig>,
    ) -> Option<DataclassMetadata> {
        // If we inherit from a dataclass, inherit its metadata. Note that if this class is
        // itself decorated with @dataclass, we'll compute new metadata and overwrite this.
        let mut dataclass_metadata = bases_with_metadata.iter().find_map(|(_, metadata)| {
            let mut m = metadata.dataclass_metadata().cloned()?;
            // Avoid accidentally overwriting a non-synthesized `__init__`.
            m.kws.init = false;
            Some(m)
        });

        let init_defaults = pydantic_config
            .map(|pyd| InitDefaults {
                init_by_name: pyd.validation_flags.validate_by_name,
                init_by_alias: pyd.validation_flags.validate_by_alias,
            })
            .unwrap_or_default();
        let default_can_be_positional = pydantic_config.is_some();

        let mut alias_keyword = DataclassFieldKeywords::ALIAS;
        if pydantic_config.is_some() {
            alias_keyword = VALIDATION_ALIAS;
        }
        for (decorator, _) in decorators {
            match decorator.ty.callee_kind() {
                // `@dataclass`
                Some(CalleeKind::Function(FunctionKind::Dataclass)) => {
                    let dataclass_fields = self.get_dataclass_fields(cls, bases_with_metadata);
                    dataclass_metadata = Some(DataclassMetadata {
                        fields: dataclass_fields,
                        kws: DataclassKeywords::new(),
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                        alias_keyword: alias_keyword.clone(),
                        init_defaults: init_defaults.clone(),
                        default_can_be_positional,
                    });
                }
                // `@dataclass(...)`
                _ if let Type::KwCall(call) = &decorator.ty
                    && call.has_function_kind(FunctionKind::Dataclass) =>
                {
                    let dataclass_fields = self.get_dataclass_fields(cls, bases_with_metadata);
                    dataclass_metadata = Some(DataclassMetadata {
                        fields: dataclass_fields,
                        kws: DataclassKeywords::from_type_map(
                            &call.keywords,
                            &DataclassTransformMetadata::new(),
                        ),
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                        alias_keyword: alias_keyword.clone(),
                        init_defaults: init_defaults.clone(),
                        default_can_be_positional,
                    });
                }
                _ => {}
            }
        }
        if let Some((kws, field_specifiers)) = dataclass_from_dataclass_transform {
            dataclass_metadata = Some(DataclassMetadata {
                fields: self.get_dataclass_fields(cls, bases_with_metadata),
                kws,
                field_specifiers,
                alias_keyword,
                init_defaults,
                default_can_be_positional,
            });
        }
        dataclass_metadata
    }

    // To avoid circular computation on targs, we have a special version of `expr_infer` that does not look into any subscript of any expr
    fn base_class_expr_infer_for_metadata(
        &self,
        expr: &BaseClassExpr,
        errors: &ErrorCollector,
    ) -> Type {
        match expr {
            BaseClassExpr::Name(x) => self
                .get(&Key::BoundName(ShortIdentifier::expr_name(x)))
                .arc_clone_ty(),
            BaseClassExpr::Attribute { value, attr, range } => {
                let base = self.base_class_expr_infer_for_metadata(value, errors);
                self.attr_infer_for_type(&base, &attr.id, *range, errors, None)
            }
            BaseClassExpr::Subscript { value, slice, .. } => {
                let ty = self.base_class_expr_infer_for_metadata(value, errors);

                // One niche special-case: the base expr has type `Forall T. type[T]`. This usually happens for snippets like this:
                // ```
                // T = TypeVar("T")
                // Foo: TypeAlias = T  # or `Foo: TypeAlias = Annotated[T, ...]`
                // class A(Foo[B]): ...
                // ```
                // In this case, we can be sure that `Foo[B]` would be the same as `B`, so we inspect the subscript.
                // This does create a potential cycle (e.g. `class A(Foo["A"])`), but not supporting it ended up breaking some important use cases.
                match &ty {
                    Type::Forall(forall)
                        if forall.tparams.len() == 1
                            && let Forallable::TypeAlias(type_alias) = &forall.body
                            && let Type::Type(box Type::Quantified(quantified)) =
                                type_alias.as_type()
                            && quantified.kind() == QuantifiedKind::TypeVar
                            && matches!(quantified.restriction(), Restriction::Unrestricted)
                            && let Some(tparam) = forall.tparams.as_vec().first()
                            && *quantified == tparam.quantified
                            && let Some(subscript_base_expr) = BaseClassExpr::from_expr(slice) =>
                    {
                        self.base_class_expr_infer_for_metadata(&subscript_base_expr, errors)
                    }
                    _ => ty,
                }
            }
        }
    }

    fn parse_base_class(&self, base: &BaseClass, is_new_type: bool) -> BaseClassParseResult {
        let range = base.range();
        let parse_base_class_type = |ty| match ty {
            Type::ClassType(c) => {
                let base_cls = c.class_object();
                let base_class_metadata = self.get_metadata_for_class(base_cls);
                BaseClassParseResult::Parsed({
                    ParsedBaseClass {
                        class_object: base_cls.dupe(),
                        range,
                        metadata: base_class_metadata,
                    }
                })
            }
            Type::Tuple(_) => {
                let tuple_obj = self.stdlib.tuple_object();
                let metadata = self.get_metadata_for_class(tuple_obj);
                BaseClassParseResult::Parsed({
                    ParsedBaseClass {
                        class_object: tuple_obj.dupe(),
                        range,
                        metadata,
                    }
                })
            }
            Type::TypedDict(typed_dict) => {
                if is_new_type {
                    BaseClassParseResult::InvalidType(typed_dict.to_type(), range)
                } else {
                    match typed_dict {
                        TypedDict::TypedDict(inner) => {
                            let class_object = inner.class_object();
                            let class_metadata = self.get_metadata_for_class(class_object);
                            BaseClassParseResult::Parsed({
                                ParsedBaseClass {
                                    class_object: class_object.dupe(),
                                    range,
                                    metadata: class_metadata,
                                }
                            })
                        }
                        TypedDict::Anonymous(_) => {
                            BaseClassParseResult::InvalidType(typed_dict.to_type(), range)
                        }
                    }
                }
            }
            _ => {
                if is_new_type || !ty.is_any() {
                    BaseClassParseResult::InvalidType(ty, range)
                } else {
                    BaseClassParseResult::AnyType
                }
            }
        };

        match base {
            BaseClass::InvalidExpr(x) => BaseClassParseResult::InvalidExpr(x.clone()),
            BaseClass::BaseClassExpr(x) => {
                // Ignore all type errors here since they'll be reported in `class_bases_of` anyway
                let errors = ErrorCollector::new(self.module().dupe(), ErrorStyle::Never);
                let ty = self.base_class_expr_infer_for_metadata(x, &errors);
                match self.untype_opt(ty.clone(), x.range(), &errors) {
                    None => BaseClassParseResult::InvalidType(ty, x.range()),
                    Some(ty) => parse_base_class_type(ty),
                }
            }
            BaseClass::NamedTuple(..) => {
                parse_base_class_type(self.stdlib.named_tuple_fallback().clone().to_type())
            }
            BaseClass::TypedDict(..) | BaseClass::Generic(..) => {
                if is_new_type {
                    BaseClassParseResult::InvalidBase(base.range())
                } else {
                    BaseClassParseResult::Ignored
                }
            }
        }
    }

    fn bases_with_metadata(
        &self,
        parsed_results: Vec<BaseClassParseResult>,
        is_new_type: bool,
        errors: &ErrorCollector,
    ) -> Vec<(Class, Arc<ClassMetadata>)> {
        parsed_results
            .into_iter()
            .filter_map(|x| match x {
                BaseClassParseResult::Ignored | BaseClassParseResult::AnyType => None,
                BaseClassParseResult::InvalidBase(range) => {
                    if is_new_type {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Second argument to NewType is invalid".to_owned(),
                        );
                    }
                    None
                }
                BaseClassParseResult::InvalidExpr(expr) => {
                    if is_new_type {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Second argument to NewType is invalid".to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            format!(
                                "Invalid expression form for base class: `{}`",
                                expr.display_with(self.module())
                            ),
                        );
                    }
                    None
                }
                BaseClassParseResult::InvalidType(ty, range) => {
                    if is_new_type {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Second argument to NewType is invalid".to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            format!("Invalid base class: `{}`", self.for_display(ty)),
                        );
                    }
                    None
                }
                BaseClassParseResult::Parsed(ParsedBaseClass {
                    class_object,
                    range,
                    metadata,
                }) => {
                    if metadata.is_final()
                        || (metadata.is_enum() && !self.get_enum_members(&class_object).is_empty())
                    {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            format!("Cannot extend final class `{}`", class_object.name()),
                        );
                    }
                    if is_new_type {
                        // TODO: raise an error for generic classes and other forbidden types such as hashable
                        if metadata.is_protocol() {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Second argument to NewType cannot be a protocol".to_owned(),
                            );
                            return None;
                        } else {
                            return Some((class_object, metadata));
                        }
                    } else if metadata.is_new_type() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            "Subclassing a NewType not allowed".to_owned(),
                        );
                    }
                    Some((class_object, metadata))
                }
            })
            .collect::<Vec<_>>()
    }

    fn calculate_typed_dict_metadata_fields(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        is_total: bool,
    ) -> SmallMap<Name, bool> {
        let mut all_fields = SmallMap::new();
        for (_, metadata) in bases_with_metadata.iter().rev() {
            if let Some(td) = metadata.typed_dict_metadata() {
                all_fields.extend(td.fields.clone());
            }
        }
        for name in cls.fields() {
            if cls.is_field_annotated(name) {
                all_fields.insert(name.clone(), is_total);
            }
        }
        all_fields
    }

    fn calculate_metaclass(
        &self,
        cls: &Class,
        raw_metaclass: Option<&Expr>,
        base_metaclasses: &[(&Name, &ClassType)],
        errors: &ErrorCollector,
    ) -> Metaclass {
        let direct_meta = raw_metaclass.and_then(|x| self.direct_metaclass(cls, x, errors));

        if let Some(metaclass) = direct_meta {
            Metaclass::Direct(metaclass)
        } else {
            let mut inherited_meta: Option<ClassType> = None;
            for (_, m) in base_metaclasses {
                let m = (*m).clone();
                let accept_m = match &inherited_meta {
                    None => true,
                    Some(inherited) => self.is_subset_eq(
                        &Type::ClassType(m.clone()),
                        &Type::ClassType(inherited.clone()),
                    ),
                };
                if accept_m {
                    inherited_meta = Some(m);
                }
            }
            inherited_meta
                .map(Metaclass::Inherited)
                .unwrap_or(Metaclass::None)
        }
    }

    fn check_base_class_metaclasses(
        &self,
        cls: &Class,
        metaclass: &ClassType,
        base_metaclasses: &[(&Name, &ClassType)],
        errors: &ErrorCollector,
    ) {
        // It is a runtime error to define a class whose metaclass (whether
        // specified directly or through inheritance) is not a subtype of all
        // base class metaclasses.
        let metaclass_type = Type::ClassType(metaclass.clone());
        for (base_name, m) in base_metaclasses {
            let base_metaclass_type = Type::ClassType((*m).clone());
            if !self.is_subset_eq(&metaclass_type, &base_metaclass_type) {
                self.error(errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                    format!(
                        "Class `{}` has metaclass `{}` which is not a subclass of metaclass `{}` from base class `{}`",
                        cls.name(),
                        self.for_display(metaclass_type.clone()),
                        self.for_display(base_metaclass_type),
                        base_name,
                    ),
                );
            }
        }
    }

    fn direct_metaclass(
        &self,
        cls: &Class,
        raw_metaclass: &Expr,
        errors: &ErrorCollector,
    ) -> Option<ClassType> {
        match self.expr_untype(raw_metaclass, TypeFormContext::BaseClassList, errors) {
            Type::ClassType(meta) => {
                if self.is_subset_eq(
                    &Type::ClassType(meta.clone()),
                    &Type::ClassType(self.stdlib.builtins_type().clone()),
                ) {
                    Some(meta)
                } else {
                    self.error(
                        errors,
                        raw_metaclass.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        format!(
                            "Metaclass of `{}` has type `{}` which is not a subclass of `type`",
                            cls.name(),
                            self.for_display(Type::ClassType(meta)),
                        ),
                    );
                    None
                }
            }
            ty => {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                    format!(
                        "Metaclass of `{}` has type `{}` that is not a simple class type",
                        cls.name(),
                        self.for_display(ty),
                    ),
                );
                None
            }
        }
    }

    pub fn calculate_class_mro(&self, cls: &Class, errors: &ErrorCollector) -> ClassMro {
        let bases = self.get_base_types_for_class(cls);
        let bases_with_mros = bases
            .iter()
            .map(|base| {
                let mro = self.get_mro_for_class(base.class_object());
                (base, mro)
            })
            .collect();
        ClassMro::new(cls, bases_with_mros, errors)
    }

    pub fn calculate_abstract_members(&self, cls: &Class) -> AbstractClassMembers {
        let metadata = self.get_metadata_for_class(cls);
        let mut fields_to_check: SmallSet<Name>;
        if metadata.extends_abc() || metadata.is_protocol() {
            fields_to_check = SmallSet::from_iter(cls.fields().cloned());
        } else {
            fields_to_check = SmallSet::new();
        }
        // Check inherited abstract methods + all fields defined in the current class
        for base_class in metadata.base_class_objects() {
            let base_class_metadata = self.get_metadata_for_class(base_class);
            // For now, skip any non-protocols base classes that don't extend `ABC` or have metaclass `ABCMeta`
            // Consider adding a stricter check in the future
            if !base_class_metadata.extends_abc() && !base_class_metadata.is_protocol() {
                continue;
            }
            let base_class_abstract_members = self.get_abstract_members_for_class(base_class);
            fields_to_check.extend(
                base_class_abstract_members
                    .unimplemented_abstract_methods()
                    .iter()
                    .cloned(),
            );
        }

        let mut abstract_members = SmallSet::new();
        for field_name in fields_to_check {
            if let Some(field) =
                self.get_non_synthesized_class_member_and_defining_class(cls, &field_name)
                && (field.value.is_abstract() ||
                // Uninitialized class vars in protocols are considered absract, unless it is in a stub file
                (!cls.module().path().is_interface() && field.value.is_uninit_class_var() &&
                self.get_metadata_for_class(&field.defining_class).is_protocol()))
            {
                abstract_members.insert(field_name.clone());
            }
        }
        AbstractClassMembers::new(abstract_members)
    }

    fn extends_abc(
        &self,
        bases_with_metadata: &Vec<(Class, Arc<ClassMetadata>)>,
        metaclass: Option<&ClassType>,
    ) -> bool {
        for (base, base_metadata) in bases_with_metadata {
            if base.has_toplevel_qname("abc", "ABC") {
                return true;
            }
            if let Some(metaclass) = base_metadata.custom_metaclass()
                && metaclass
                    .class_object()
                    .has_toplevel_qname("abc", "ABCMeta")
            {
                return true;
            }
            if base_metadata.extends_abc() {
                return true;
            }
        }
        if let Some(metaclass) = metaclass
            && metaclass
                .class_object()
                .has_toplevel_qname("abc", "ABCMeta")
        {
            return true;
        }
        false
    }
}
