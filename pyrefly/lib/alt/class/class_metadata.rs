/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use itertools::Itertools;
use pyrefly_graph::index::Idx;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::annotation::Annotation;
use pyrefly_types::callable::Params;
use pyrefly_types::quantified::Quantified;
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
use starlark_map::Hashed;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::attrs::is_attrs_setters_frozen;
use crate::alt::class::django::is_django_choices_subclass;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::abstract_class::AbstractClassMembers;
use crate::alt::types::class_metadata::ClassDisjointBase;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::DataclassKind;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::class_metadata::DjangoModelMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::alt::types::class_metadata::ExplicitSlots;
use crate::alt::types::class_metadata::InitDefaults;
use crate::alt::types::class_metadata::Metaclass;
use crate::alt::types::class_metadata::NamedTupleMetadata;
use crate::alt::types::class_metadata::ProtocolMetadata;
use crate::alt::types::class_metadata::SlotsInfo;
use crate::alt::types::class_metadata::TotalOrderingMetadata;
use crate::alt::types::class_metadata::TypedDictMetadata;
use crate::alt::types::decorated_function::Decorator;
use crate::alt::types::pydantic::PydanticConfig;
use crate::alt::types::pydantic::PydanticModelKind;
use crate::binding::base_class::BaseClass;
use crate::binding::base_class::BaseClassExpr;
use crate::binding::base_class::BaseClassGeneric;
use crate::binding::base_class::BaseClassGenericKind;
use crate::binding::binding::BindingShapedArrayMetadata;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyDecorator;
use crate::binding::django::DjangoFieldInfo;
use crate::binding::pydantic::PydanticConfigDict;
use crate::binding::pydantic::VALIDATION_ALIAS;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::style::ErrorStyle;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::display::ClassDisplayContext;
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

/// The dataclass configuration derived from a `@dataclass_transform` decorator or an inherited
/// transform base.
pub(crate) struct TransformDataclass {
    keywords: DataclassKeywords,
    /// Callees recognized as field specifiers (PEP 681), e.g. `attrs.field`.
    pub(crate) field_specifiers: Vec<CalleeKind>,
    /// attrs' `hash=`/`unsafe_hash=` argument; `None` for non-attrs classes or when unset.
    attrs_hash: Option<bool>,
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
        pydantic_before_validator_fields: &[Name],
        django_field_info: &DjangoFieldInfo,
        capture_init: Option<&[Name]>,
        shaped_array_metadata: Option<&BindingShapedArrayMetadata>,
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
        let initial_protocol_metadata = self.initial_protocol_metadata(cls, bases);
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
        let mut calculated_metaclass = self.calculate_metaclass(
            cls,
            metaclasses.into_iter().next(),
            &base_metaclasses,
            errors,
        );
        if let Some(metaclass) = calculated_metaclass.get() {
            self.check_base_class_metaclasses(cls, metaclass, &base_metaclasses, errors);
            if metaclass
                .targs()
                .as_slice()
                .iter()
                .any(|targ| targ.contains_type_variable())
            {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
                    "Metaclass may not be an unbound generic".to_owned(),
                );
            }
        }
        // If the metaclass has unresolved type variables, replace them with their
        // gradual types (e.g. Any) to avoid cascading errors from bare TypeVars.
        // We do a targeted substitution inside each targ so that e.g. Meta[list[T]]
        // becomes Meta[list[Any]] rather than Meta[Any].
        if let Some(metaclass) = calculated_metaclass.get_mut() {
            for targ in metaclass.targs_mut().as_mut().iter_mut() {
                if targ.contains_type_variable() {
                    targ.transform_mut(&mut |ty| match ty {
                        Type::Quantified(q) => *ty = q.as_gradual_type(),
                        Type::TypeVar(t) => {
                            *ty = Quantified::as_gradual_type_helper(
                                QuantifiedKind::TypeVar,
                                t.default(),
                            )
                        }
                        Type::TypeVarTuple(t) => {
                            *ty = Quantified::as_gradual_type_helper(
                                QuantifiedKind::TypeVarTuple,
                                t.default(),
                            )
                        }
                        Type::ParamSpec(p) => {
                            *ty = Quantified::as_gradual_type_helper(
                                QuantifiedKind::ParamSpec,
                                p.default(),
                            )
                        }
                        _ => {}
                    });
                }
            }
        }
        let metaclass = calculated_metaclass.get();

        let mut directly_inherits_model = false;
        let mut inherited_django_metadata: Option<&DjangoModelMetadata> = None;

        // TODO Zeina: This pattern is repeated a lot in this file. See if we can refactor it (BE).
        for (base_class_object, metadata) in &bases_with_metadata {
            if base_class_object.has_toplevel_qname(ModuleName::django_models().as_str(), "Model") {
                directly_inherits_model = true;
            }

            if let Some(dm) = metadata.django_model_metadata() {
                // Prefer the base that has a custom primary key field,
                // so a later base without one doesn't overwrite it.
                if inherited_django_metadata
                    .is_none_or(|prev| prev.custom_primary_key_field.is_none())
                {
                    inherited_django_metadata = Some(dm);
                }
            }
        }

        let django_model_metadata = if directly_inherits_model
            || inherited_django_metadata.is_some()
        {
            Some(DjangoModelMetadata {
                custom_primary_key_field: django_field_info.primary_key_field.clone().or_else(
                    || inherited_django_metadata.and_then(|dm| dm.custom_primary_key_field.clone()),
                ),
                foreign_key_like_fields: django_field_info.foreign_key_like_fields.clone(),
                fields_with_choices: django_field_info.fields_with_choices.clone(),
            })
        } else {
            None
        };

        // Check if this class inherits from marshmallow.Schema
        let is_marshmallow_schema =
            bases_with_metadata
                .iter()
                .any(|(base_class_object, metadata)| {
                    base_class_object
                        .has_toplevel_qname(ModuleName::marshmallow_schema().as_str(), "Schema")
                        || metadata.is_marshmallow_schema()
                });

        let is_factory_boy_factory =
            bases_with_metadata
                .iter()
                .any(|(base_class_object, metadata)| {
                    base_class_object
                        .has_toplevel_qname(ModuleName::factory_base().as_str(), "Factory")
                        || metadata.is_factory_boy_factory()
                });

        let is_metaclass = bases_with_metadata
            .iter()
            .any(|(base_class_object, metadata)| {
                base_class_object.is_builtin("type") || metadata.is_metaclass()
            });

        // Compute various pieces of special metadata.
        let has_base_any = contains_base_class_any
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.has_base_any());

        let named_tuple_metadata =
            self.named_tuple_metadata(cls, bases, &bases_with_metadata, errors);
        // Only `class X(NamedTuple, ...)` is rejected at runtime. Subclassing
        // a concrete NamedTuple alongside other bases is allowed.
        let directly_defines_named_tuple = bases
            .iter()
            .any(|base| matches!(base, BaseClass::NamedTuple(..)));
        if directly_defines_named_tuple
            && bases_with_metadata.len() > 1
            && !cls.module().path().is_interface()
        {
            // Typeshed models some stdlib namedtuple result objects, such as urllib.parse.ParseResult,
            // by mixing methods into a NamedTuple subclass inside a `.pyi`. Keep rejecting this in
            // user code, but allow it in stubs so we can type-check those result objects precisely.
            self.error(
                errors,
                cls.range(),
                ErrorKind::InvalidInheritance,
                "Named tuples do not support multiple inheritance".to_owned(),
            );
        }

        let pydantic_config = self.pydantic_config(
            &bases_with_metadata,
            pydantic_config_dict,
            &keywords,
            &decorators,
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
                ErrorKind::InvalidInheritance,
                format!("`{}` is not a typed dictionary. Typed dictionary definitions may only extend other typed dictionaries.", bad.0.name()),
            );
        }
        let typed_dict_metadata =
            self.typed_dict_metadata(cls, &bases_with_metadata, &keywords, is_typed_dict, errors);
        if metaclass.is_some() && is_typed_dict {
            self.error(
                errors,
                cls.range(),
                ErrorKind::InvalidInheritance,
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

        let explicit_slots = self.explicit_slots(cls);
        let has_nonempty_explicit_slots = explicit_slots
            .slots_info()
            .is_some_and(|slots| !slots.names.is_empty());
        // PEP 800: `@disjoint_base` is only valid on nominal classes. Marking
        // an invalid TypedDict/Protocol target as disjoint would let narrowing
        // incorrectly intersect it to Never.
        let has_valid_disjoint_base_decorator = decorators.iter().any(|(decorator, range)| {
            if decorator.ty.callee_kind() != Some(CalleeKind::Function(FunctionKind::DisjointBase))
            {
                return false;
            }
            if is_typed_dict {
                self.error(
                    errors,
                    *range,
                    ErrorKind::BadClassDefinition,
                    "`@disjoint_base` cannot be applied to a TypedDict".to_owned(),
                );
                false
            } else if protocol_metadata.is_some() {
                self.error(
                    errors,
                    *range,
                    ErrorKind::BadClassDefinition,
                    "`@disjoint_base` cannot be applied to a Protocol".to_owned(),
                );
                false
            } else {
                true
            }
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

        // If this class inherits from a dataclass_transform-ed class or uses a metaclass decorated
        // with @dataclass_transform, record the defaults that we should use for dataclass parameters.
        let dataclass_defaults_from_base_class = bases_with_metadata
            .iter()
            .find_map(|(_, metadata)| metadata.dataclass_transform_metadata().cloned())
            .or_else(|| {
                metaclass.and_then(|c| {
                    self.get_metadata_for_class(c.class_object())
                        .dataclass_transform_metadata()
                        .cloned()
                })
            });
        let dataclass_transform_metadata = self.dataclass_transform_metadata(
            &keywords,
            &decorators,
            metaclass,
            dataclass_defaults_from_base_class.clone(),
        );
        let dataclass_from_dataclass_transform = self.dataclass_from_dataclass_transform(
            cls,
            &keywords,
            &decorators,
            dataclass_defaults_from_base_class,
            pydantic_config.as_ref(),
            errors,
        );
        let is_attrs_class =
            self.is_attrs_class(&dataclass_from_dataclass_transform, &bases_with_metadata);
        let is_from_dataclass_transform = dataclass_from_dataclass_transform.is_some();
        let (dataclass_metadata, has_fresh_local_slots_decorator) = self.dataclass_metadata(
            cls,
            &decorators,
            &bases_with_metadata,
            dataclass_from_dataclass_transform,
            pydantic_config.as_ref(),
            pydantic_before_validator_fields,
            is_attrs_class,
            protocol_metadata.is_some(),
            enum_metadata.is_some(),
            is_typed_dict,
            named_tuple_metadata.is_some(),
            errors,
        );
        // Store only local class-body disjointness here; generated dataclass
        // slots and inherited representatives need the MRO and are resolved by
        // `KeyClassDisjointBase`.
        let is_local_disjoint_base = has_valid_disjoint_base_decorator
            || (!is_typed_dict && protocol_metadata.is_none() && has_nonempty_explicit_slots);
        if let Some(dm) = dataclass_metadata.as_ref()
            && pydantic_config.is_none()
        {
            self.validate_frozen_dataclass_inheritance(
                cls,
                dm,
                &bases_with_metadata,
                is_from_dataclass_transform,
                errors,
            );
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
        let keywords = keywords.into_map(|(name, annot)| {
            (
                name,
                annot.ty.unwrap_or_else(|| self.heap.mk_any_implicit()),
            )
        });

        // get pydantic model info. A root model is by default also a base model, while not every base model is a root model
        let pydantic_model_kind = pydantic_config
            .as_ref()
            .map(|m| m.pydantic_model_kind.clone());

        let shaped_array_shape = self.shaped_array_shape(cls, shaped_array_metadata, errors);

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
            is_local_disjoint_base,
            has_fresh_local_slots_decorator,
            total_ordering_metadata,
            dataclass_transform_metadata,
            pydantic_model_kind,
            is_attrs_class,
            django_model_metadata,
            is_marshmallow_schema,
            is_factory_boy_factory,
            is_metaclass,
            explicit_slots,
            capture_init.map(|names| names.to_vec()),
            shaped_array_shape,
        )
    }

    fn shaped_array_shape(
        &self,
        cls: &Class,
        metadata: Option<&BindingShapedArrayMetadata>,
        errors: &ErrorCollector,
    ) -> Option<Quantified> {
        let BindingShapedArrayMetadata { shape_name, range } = metadata?;
        let tparams = self.get_class_tparams(cls);
        match tparams.iter().find(|param| param.name() == shape_name) {
            // A shape parameter may be either a `TypeVarTuple` (variadic shape,
            // e.g. `Array[*Shape]`) or a regular `TypeVar` that carries the shape
            // as a single tuple type (e.g. NumPy's `ndarray[Shape, DType]`).
            Some(param)
                if matches!(
                    param.kind,
                    QuantifiedKind::TypeVarTuple | QuantifiedKind::TypeVar
                ) =>
            {
                Some(param.clone())
            }
            Some(param) => {
                self.error(
                    errors,
                    *range,
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Shape parameter `{}` must be a `TypeVar` or `TypeVarTuple`, got `{}`",
                        shape_name, param.kind
                    ),
                );
                None
            }
            None => {
                self.error(
                    errors,
                    *range,
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Shape parameter `{}` is not a type parameter of class `{}`",
                        shape_name,
                        cls.name()
                    ),
                );
                None
            }
        }
    }

    fn explicit_slots(&self, cls: &Class) -> ExplicitSlots {
        let key = KeyClassField(cls.index(), dunder::SLOTS.clone());
        let Some(idx) = self.bindings().key_to_idx_hashed_opt(Hashed::new(&key)) else {
            return ExplicitSlots::Absent;
        };
        let binding = self.bindings().get::<KeyClassField>(idx);
        let ClassFieldDefinition::AssignedInBody { value, .. } = &binding.definition else {
            return ExplicitSlots::Unknown;
        };
        let ExprOrBinding::Expr(expr) = value.as_ref() else {
            return ExplicitSlots::Unknown;
        };

        fn name_of(expr: &Expr) -> Option<Name> {
            match expr {
                Expr::StringLiteral(s) => Some(Name::new(s.value.to_str())),
                _ => None,
            }
        }
        let names: Option<SmallSet<Name>> = match expr {
            Expr::Tuple(t) => t.elts.iter().map(name_of).collect(),
            Expr::List(l) => l.elts.iter().map(name_of).collect(),
            Expr::Dict(d) => d
                .items
                .iter()
                .map(|item| item.key.as_ref().and_then(name_of))
                .collect(),
            Expr::StringLiteral(s) => Some(iter::once(Name::new(s.value.to_str())).collect()),
            _ => return ExplicitSlots::Unknown,
        };
        match names {
            Some(names) => ExplicitSlots::Known(SlotsInfo { names }),
            None => ExplicitSlots::Unknown,
        }
    }

    fn initial_protocol_metadata(
        &self,
        cls: &Class,
        bases: &[BaseClass],
    ) -> Option<ProtocolMetadata> {
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
                members: self
                    .get_class_fields(cls)
                    .map(|f| f.class_body_fields().cloned().collect())
                    .unwrap_or_default(),
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
                            ErrorKind::InvalidInheritance,
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
                            ErrorKind::BadClassDefinition,
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
        bases: &[BaseClass],
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        errors: &ErrorCollector,
    ) -> Option<NamedTupleMetadata> {
        // Check if any base is a NamedTuple with dynamic fields
        let has_dynamic_fields = bases
            .iter()
            .any(|b| matches!(b, BaseClass::NamedTuple(_, true)));

        bases_with_metadata
            .iter()
            .find_map(|(base_class_object, metadata)| {
                if base_class_object.has_toplevel_qname(
                    ModuleName::type_checker_internals().as_str(),
                    "NamedTupleFallback",
                ) {
                    Some(NamedTupleMetadata {
                        elements: self.get_named_tuple_elements(cls, errors),
                        has_dynamic_fields,
                        directly_extends_named_tuple: true,
                    })
                } else {
                    metadata
                        .named_tuple_metadata()
                        .map(|nt| NamedTupleMetadata {
                            elements: nt.elements.clone(),
                            has_dynamic_fields: nt.has_dynamic_fields,
                            directly_extends_named_tuple: false,
                        })
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
                    ("total", Type::Literal(lit)) if matches!(lit.value, Lit::Bool(false)) => {
                        is_total = false;
                    }
                    ("closed" | "extra_items", _) if extra_items.is_some() => {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorKind::BadTypedDict,
                            "TypedDict keywords `closed` and `extra_items` cannot be used together"
                                .to_owned(),
                        );
                    }
                    ("closed", Type::Literal(lit)) if matches!(lit.value, Lit::Bool(true)) => {
                        extra_items = Some(ExtraItems::Closed);
                    }
                    ("closed", Type::Literal(lit)) if matches!(lit.value, Lit::Bool(false)) => {
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
                                ErrorKind::BadTypedDict,
                                format!("Expected `extra_items` to be a type form, got instance of `{}`", self.for_display(value_ty.clone())),
                            )
                        });
                        extra_items = Some(ExtraItems::extra(ty, &value.qualifiers));
                    }
                    ("total", Type::Literal(lit)) if matches!(lit.value, Lit::Bool(_)) => {}
                    ("total" | "closed", value_ty) => {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorKind::BadTypedDict,
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
                            ErrorKind::BadTypedDict,
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
                    ErrorKind::BadTypedDict,
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
                    ErrorKind::BadTypedDict,
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
                    ErrorKind::BadTypedDict,
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
        let is_django = is_django_choices_subclass(bases_with_metadata);

        let metaclass_is_enum = metaclass.is_some_and(|m| {
            self.as_superclass(m, self.stdlib.enum_meta().class_object())
                .is_some()
        });
        let base_is_enum = bases_with_metadata.iter().any(|(_, meta)| meta.is_enum());

        if metaclass_is_enum || base_is_enum {
            // NOTE(grievejia): This may create potential cycle if metaclass is generic. Need to look into
            // whether it can be removed or not.
            if !self.get_class_tparams(cls).is_empty() {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
                    "Enums may not be generic".to_owned(),
                );
            }
            Some(EnumMetadata {
                // A generic enum is an error, but we create Any type args anyway to handle it gracefully.
                cls: self.promote_nontypeddict_silently_to_classtype(cls),
                has_value: bases_with_metadata.iter().any(|(base, _)| {
                    self.get_class_fields(base)
                        .is_some_and(|f| f.contains(&Name::new_static("_value_")))
                }),
                is_flag: bases_with_metadata.iter().any(|(base, _)| {
                    self.is_subset_eq(
                        &self
                            .heap
                            .mk_class_type(self.promote_nontypeddict_silently_to_classtype(base)),
                        &self.heap.mk_class_type(self.stdlib.enum_flag().clone()),
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
        keywords: &[(Name, Annotation)],
        decorators: &[(Arc<Decorator>, TextRange)],
        metaclass: Option<&ClassType>,
        dataclass_defaults_from_base_class: Option<DataclassTransformMetadata>,
    ) -> Option<DataclassTransformMetadata> {
        // This is set when a class is decorated with `@typing.dataclass_transform(...)`. Note that
        // this does not turn the class into a dataclass! Instead, it becomes a special base class
        // (or metaclass) that turns child classes into dataclasses.
        // `dataclass_defaults_from_base_class` already falls back to the metaclass's transform
        // metadata, so prefer it here: a base's accumulated keyword defaults (folded below) must
        // propagate down the whole subtree rather than being reset to the metaclass's raw defaults.
        let mut dataclass_transform_metadata = dataclass_defaults_from_base_class;
        for (decorator, _) in decorators {
            // `@dataclass_transform(...)`
            if let Type::KwCall(call) = &decorator.ty
                && call.has_function_kind(FunctionKind::DataclassTransform)
            {
                dataclass_transform_metadata =
                    Some(DataclassTransformMetadata::from_type_map(&call.keywords));
            }
        }
        // A metaclass-based dataclass_transform (e.g. SQLAlchemy's `DCTransformDeclarative`)
        // re-applies its dataclass keywords to every subclass
        let metaclass_is_transform = metaclass.is_some_and(|c| {
            self.get_metadata_for_class(c.class_object())
                .dataclass_transform_metadata()
                .is_some()
        });
        if metaclass_is_transform && let Some(metadata) = &mut dataclass_transform_metadata {
            for (name, annot) in keywords {
                let Some(value) = annot.get_type().as_bool() else {
                    continue;
                };
                match name.as_str() {
                    "kw_only" => metadata.kw_only_default = value,
                    "eq" => metadata.eq_default = value,
                    "order" => metadata.order_default = value,
                    _ => {}
                }
            }
        }
        dataclass_transform_metadata
    }

    fn dataclass_from_dataclass_transform(
        &self,
        cls: &Class,
        keywords: &[(Name, Annotation)],
        decorators: &[(Arc<Decorator>, TextRange)],
        dataclass_defaults_from_base_class: Option<DataclassTransformMetadata>,
        pydantic_config: Option<&PydanticConfig>,
        errors: &ErrorCollector,
    ) -> Option<TransformDataclass> {
        // This is set when we should apply dataclass-like transformations to the class. The class
        // should be transformed if:
        // - it inherits from a base class decorated with `dataclass_transform(...)`, or
        // - it inherits from a base class whose metaclass is decorated with `dataclass_transform(...)`, or
        // - it is decorated with a decorator that is decorated with `dataclass_transform(...)`.

        // Pydantic models and dataclasses default to strict=false (lax mode).
        // Regular dataclasses default to strict=true.
        let strict_default = pydantic_config.is_none();

        let mut dataclass_from_dataclass_transform = None;
        if let Some(defaults) = dataclass_defaults_from_base_class {
            // This class inherits from a dataclass_transform-ed base class, so its keywords are
            // interpreted as dataclass keywords.
            let map = TypeMap(
                keywords
                    .iter()
                    .map(|(name, annot)| (name.clone(), annot.get_type().clone()))
                    .collect::<OrderedMap<_, _>>(),
            );
            let mut kws = DataclassKeywords::from_type_map(&map, &defaults, strict_default);

            // Inject pydantic model configuration from ConfigDict.
            // This path is for pydantic models (BaseModel, etc.), not pydantic dataclasses.
            if let Some(pydantic) = pydantic_config {
                if let Some(frozen) = pydantic.frozen {
                    kws.frozen = frozen;
                }
                if let Some(extra) = pydantic.extra {
                    kws.extra = extra;
                }
                if let Some(strict) = pydantic.strict {
                    kws.strict = strict;
                }
            }

            kws.attrs_setattr_frozen = map
                .0
                .get(&DataclassFieldKeywords::ON_SETATTR)
                .is_some_and(is_attrs_setters_frozen);
            dataclass_from_dataclass_transform = Some(TransformDataclass {
                keywords: kws,
                field_specifiers: defaults.field_specifiers,
                attrs_hash: None,
            });
        }
        for (decorator, decorator_range) in decorators {
            // `@foo` where `foo` is decorated with `@dataclass_transform(...)`
            if let Some(defaults) = decorator.ty.dataclass_transform_metadata() {
                let mut kws =
                    DataclassKeywords::from_type_map(&TypeMap::new(), defaults, strict_default);
                if kws.auto_attribs.is_none() {
                    kws.auto_attribs = Some(self.attrs_default_auto_attribs(
                        cls,
                        *decorator_range,
                        defaults.order_default,
                    ));
                }
                dataclass_from_dataclass_transform = Some(TransformDataclass {
                    keywords: kws,
                    field_specifiers: defaults.field_specifiers.clone(),
                    attrs_hash: None,
                });
            }
            // `@foo(...)` where `foo` is decorated with `@dataclass_transform(...)`
            else if let Type::KwCall(call) = &decorator.ty
                && let Some(defaults) = &call.func_metadata.flags.dataclass_transform_metadata
            {
                let mut kws =
                    DataclassKeywords::from_type_map(&call.keywords, defaults, strict_default);
                if kws.auto_attribs.is_none() {
                    kws.auto_attribs = Some(self.attrs_default_auto_attribs(
                        cls,
                        *decorator_range,
                        defaults.order_default,
                    ));
                }
                kws.attrs_setattr_frozen = call
                    .keywords
                    .0
                    .get(&DataclassFieldKeywords::ON_SETATTR)
                    .is_some_and(is_attrs_setters_frozen);
                let attrs_hash =
                    if Self::field_specifiers_reference_attrs(&defaults.field_specifiers) {
                        self.validate_attrs_eq_order_cmp(&call.keywords, *decorator_range, errors);
                        DataclassKeywords::attrs_hash_from_map(&call.keywords)
                    } else {
                        None
                    };
                dataclass_from_dataclass_transform = Some(TransformDataclass {
                    keywords: kws,
                    field_specifiers: defaults.field_specifiers.clone(),
                    attrs_hash,
                });
            }
        }
        dataclass_from_dataclass_transform
    }

    /// Single annotation walk returning local
    /// `(instance, pseudo_overrides, pydantic_privates)` — pairwise disjoint;
    /// together they cover every annotated body field. `pseudo_overrides`
    /// (`ClassVar`/`InitVar`/`KW_ONLY`) replace the matching inherited
    /// dataclass entry per CPython `_process_class`; pydantic privates are
    /// non-overriding non-instance fields that still belong in
    /// `DataclassMetadata.pseudo_fields`. Cycle-safe: must not call helpers
    /// that consume solved class fields.
    fn local_dataclass_field_classification(
        &self,
        cls: &Class,
        pydantic_config: Option<&PydanticConfig>,
    ) -> (SmallSet<Name>, SmallSet<Name>, SmallSet<Name>) {
        let mut instance = SmallSet::new();
        let mut pseudo_overrides = SmallSet::new();
        let mut pydantic_privates = SmallSet::new();
        let Some(class_fields) = self.get_class_fields(cls) else {
            return (instance, pseudo_overrides, pydantic_privates);
        };
        let pydantic_drops_private = pydantic_config.is_some_and(|pydantic| {
            matches!(
                pydantic.pydantic_model_kind,
                PydanticModelKind::BaseModel
                    | PydanticModelKind::RootModel
                    | PydanticModelKind::BaseSettings
            )
        });
        for name in class_fields.class_body_fields() {
            if !class_fields.is_field_annotated(name) {
                continue;
            }
            if pydantic_drops_private && name.as_str().starts_with('_') {
                pydantic_privates.insert(name.clone());
                continue;
            }
            let key = KeyClassField(cls.index(), name.clone());
            let Some(field_idx) = self.bindings().key_to_idx_hashed_opt(Hashed::new(&key)) else {
                continue;
            };
            let binding = self.bindings().get::<KeyClassField>(field_idx);
            let annotation_key: Idx<KeyAnnotation> = match &binding.definition {
                ClassFieldDefinition::DeclaredByAnnotation { annotation, .. } => *annotation,
                ClassFieldDefinition::AssignedInBody {
                    annotation: Some(annotation),
                    ..
                } => *annotation,
                _ => continue,
            };
            let annotation = &self.get_idx(annotation_key).annotation;
            if annotation.is_class_var()
                || annotation.is_init_var()
                || matches!(annotation.get_type(), Type::ClassType(c) if c.has_qname("dataclasses", "KW_ONLY"))
            {
                pseudo_overrides.insert(name.clone());
                continue;
            }
            instance.insert(name.clone());
        }
        (instance, pseudo_overrides, pydantic_privates)
    }

    /// Populates `DataclassMetadata.pseudo_field_names`.
    fn get_dataclass_pseudo_field_names(
        &self,
        cls: &Class,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        pydantic_config: Option<&PydanticConfig>,
    ) -> SmallSet<Name> {
        let (local_instance, local_pseudo_overrides, local_pydantic_privates) =
            self.local_dataclass_field_classification(cls, pydantic_config);
        let mut pseudo_field_names = SmallSet::new();
        for (_, metadata) in bases_with_metadata.iter().rev() {
            if let Some(dm) = metadata.dataclass_metadata() {
                for name in &dm.pseudo_field_names {
                    // A local instance annotation overrides an inherited pseudo entry.
                    if !local_instance.contains(name) {
                        pseudo_field_names.insert(name.clone());
                    }
                }
            }
        }
        pseudo_field_names.extend(local_pseudo_overrides);
        pseudo_field_names.extend(local_pydantic_privates);
        pseudo_field_names
    }

    fn dataclass_metadata(
        &self,
        cls: &Class,
        decorators: &[(Arc<Decorator>, TextRange)],
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        dataclass_from_dataclass_transform: Option<TransformDataclass>,
        pydantic_config: Option<&PydanticConfig>,
        pydantic_before_validator_fields: &[Name],
        is_attrs_class: bool,
        is_protocol: bool,
        is_enum: bool,
        is_typed_dict: bool,
        is_named_tuple: bool,
        errors: &ErrorCollector,
    ) -> (Option<DataclassMetadata>, bool) {
        // If we inherit from a dataclass, inherit its metadata. Note that if this class is
        // itself decorated with @dataclass, we'll compute new metadata and overwrite this.
        let mut dataclass_metadata = bases_with_metadata.iter().find_map(|(_, metadata)| {
            let mut m = metadata.dataclass_metadata().cloned()?;
            // Avoid accidentally overwriting a non-synthesized `__init__`.
            m.kws.init = false;
            Some(m)
        });
        let mut has_fresh_local_slots_decorator = false;

        let init_defaults = pydantic_config
            .map(|pyd| InitDefaults {
                init_by_name: pyd.validation_flags.validate_by_name,
                init_by_alias: pyd.validation_flags.validate_by_alias,
            })
            .unwrap_or_default();
        let default_can_be_positional = pydantic_config.is_some() || is_attrs_class;

        let mut alias_keyword = DataclassFieldKeywords::ALIAS;
        if pydantic_config.is_some() {
            alias_keyword = VALIDATION_ALIAS;
        }
        let mut has_dataclass_decorator = false;
        for (decorator, _) in decorators {
            match decorator.ty.callee_kind() {
                // `@dataclass`
                Some(CalleeKind::Function(FunctionKind::Dataclass)) => {
                    has_dataclass_decorator = true;
                    let kind = DataclassKind::Dataclass {
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                    };
                    let fields = self.get_dataclass_fields(cls, bases_with_metadata, &kind);
                    let pseudo_field_names = self.get_dataclass_pseudo_field_names(
                        cls,
                        bases_with_metadata,
                        pydantic_config,
                    );
                    let kws = DataclassKeywords::new();
                    // Bare `@dataclass` never sets slots.
                    has_fresh_local_slots_decorator = false;
                    dataclass_metadata = Some(DataclassMetadata {
                        fields,
                        pseudo_field_names,
                        kws,
                        alias_keyword: alias_keyword.clone(),
                        init_defaults: init_defaults.clone(),
                        default_can_be_positional,
                        pydantic_before_validator_fields: SmallSet::new(),
                        kind,
                    });
                }
                // `@dataclass(...)`
                _ if let Type::KwCall(call) = &decorator.ty
                    && call.has_function_kind(FunctionKind::Dataclass) =>
                {
                    has_dataclass_decorator = true;
                    let kind = DataclassKind::Dataclass {
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                    };
                    let fields = self.get_dataclass_fields(cls, bases_with_metadata, &kind);
                    let pseudo_field_names = self.get_dataclass_pseudo_field_names(
                        cls,
                        bases_with_metadata,
                        pydantic_config,
                    );
                    let kws = DataclassKeywords::from_type_map(
                        &call.keywords,
                        &DataclassTransformMetadata::new(),
                        true, // Regular dataclasses are always strict
                    );
                    has_fresh_local_slots_decorator = kws.slots;
                    dataclass_metadata = Some(DataclassMetadata {
                        fields,
                        pseudo_field_names,
                        kws,
                        alias_keyword: alias_keyword.clone(),
                        init_defaults: init_defaults.clone(),
                        default_can_be_positional,
                        pydantic_before_validator_fields: SmallSet::new(),
                        kind,
                    });
                }
                _ => {}
            }
        }
        // @dataclass cannot be applied to Protocol, Enum, TypedDict, or NamedTuple classes.
        // Emit the error and return no metadata so the class is not treated as a dataclass.
        if has_dataclass_decorator {
            if is_protocol {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::BadClassDefinition,
                    format!(
                        "`@dataclass` cannot be applied to Protocol `{}`",
                        cls.name()
                    ),
                );
                return (None, false);
            }
            if is_enum {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::BadClassDefinition,
                    format!("Cannot apply `@dataclass` to Enum `{}`", cls.name()),
                );
                return (None, false);
            }
            if is_typed_dict {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::BadClassDefinition,
                    format!("Cannot apply `@dataclass` to TypedDict `{}`", cls.name()),
                );
                return (None, false);
            }
            if is_named_tuple {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::BadClassDefinition,
                    format!("Cannot apply `@dataclass` to NamedTuple `{}`", cls.name()),
                );
                return (None, false);
            }
        }
        if let Some(TransformDataclass {
            keywords: kws,
            field_specifiers,
            attrs_hash,
        }) = dataclass_from_dataclass_transform
        {
            // Inherit before-validator fields from base pydantic models, then add our own.
            let mut inherited_before_validator_fields: SmallSet<Name> = bases_with_metadata
                .iter()
                .filter(|(_, metadata)| metadata.is_pydantic_model())
                .filter_map(|(_, metadata)| metadata.dataclass_metadata())
                .flat_map(|dm| dm.pydantic_before_validator_fields.iter().cloned())
                .collect();
            inherited_before_validator_fields
                .extend(pydantic_before_validator_fields.iter().cloned());
            // TODO: a transform-derived spec silently drops the explicit
            // `@dataclass(...)` kws from the loop above. Needs a merge policy.
            has_fresh_local_slots_decorator = kws.slots;
            let kind = if is_attrs_class {
                DataclassKind::Attrs {
                    auto_attribs: kws.auto_attribs,
                    hash: attrs_hash,
                    field_specifiers,
                }
            } else {
                DataclassKind::Dataclass { field_specifiers }
            };
            let fields = self.get_dataclass_fields(cls, bases_with_metadata, &kind);
            let pseudo_field_names =
                self.get_dataclass_pseudo_field_names(cls, bases_with_metadata, pydantic_config);
            dataclass_metadata = Some(DataclassMetadata {
                fields,
                pseudo_field_names,
                kws,
                alias_keyword,
                init_defaults,
                default_can_be_positional,
                pydantic_before_validator_fields: inherited_before_validator_fields,
                kind,
            });
        }
        (dataclass_metadata, has_fresh_local_slots_decorator)
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
                // The alias body is `type[T]` for bare aliases and `Annotated[T]` for Annotated aliases;
                // we must handle both.
                match &ty {
                    Type::Forall(forall)
                        if forall.tparams.len() == 1
                            && let Forallable::TypeAlias(type_alias) = &forall.body
                            && let quantified = match self.get_type_alias(type_alias).as_type() {
                                Type::Type(f) if matches!(&*f, Type::Quantified(_)) => {
                                    let Type::Quantified(q) = *f else {
                                        unreachable!("guarded by matches! above")
                                    };
                                    Some(q)
                                }
                                Type::Annotated(f, _) if matches!(&*f, Type::Quantified(_)) => {
                                    let Type::Quantified(q) = *f else {
                                        unreachable!("guarded by matches! above")
                                    };
                                    Some(q)
                                }
                                _ => None,
                            }
                            && let Some(quantified) = quantified
                            && quantified.kind() == QuantifiedKind::TypeVar
                            && matches!(quantified.restriction(), Restriction::Unrestricted)
                            && let Some(tparam) = forall.tparams.as_vec().first()
                            && *quantified == *tparam
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
                    BaseClassParseResult::InvalidType(typed_dict.to_type(self.heap), range)
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
                            BaseClassParseResult::InvalidType(typed_dict.to_type(self.heap), range)
                        }
                    }
                }
            }
            Type::None if is_new_type => {
                let base_cls = self.stdlib.none_type().class_object();
                let metadata = self.get_metadata_for_class(base_cls);
                BaseClassParseResult::Parsed({
                    ParsedBaseClass {
                        class_object: base_cls.dupe(),
                        range,
                        metadata,
                    }
                })
            }
            Type::Type(f) if f.is_any() => {
                // `type[Any]` is equivalent to `type` or `Type`
                let type_obj = self.stdlib.builtins_type().class_object();
                let metadata = self.get_metadata_for_class(type_obj);
                BaseClassParseResult::Parsed(ParsedBaseClass {
                    class_object: type_obj.dupe(),
                    range,
                    metadata,
                })
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
                // The value `None` has type `Type::None`, which `untype_opt` passes
                // through because `None` is both a value and a type. But as a NewType
                // base, the value `None` is not valid — only the class `NoneType` is.
                // Reject it here before `untype_opt` erases the distinction.
                if is_new_type && matches!(&ty, Type::None) {
                    return BaseClassParseResult::InvalidType(ty, x.range());
                }
                match self.untype_opt(ty.clone(), x.range(), &errors) {
                    None => BaseClassParseResult::InvalidType(ty, x.range()),
                    Some(ty) => parse_base_class_type(ty),
                }
            }
            BaseClass::NamedTuple(..) => parse_base_class_type(
                self.heap
                    .mk_class_type(self.stdlib.named_tuple_fallback().clone()),
            ),
            BaseClass::SynthesizedBase(class_idx, _) => {
                match &self.get_idx(*class_idx).as_ref().0 {
                    Some(cls) => {
                        // At the moment we never synthesize a typed dict, so this is safe
                        let ct = self.promote_nontypeddict_silently_to_classtype(cls);
                        parse_base_class_type(self.heap.mk_class_type(ct))
                    }
                    None => BaseClassParseResult::InvalidBase(base.range()),
                }
            }
            BaseClass::TypeOf(inner_expr, _) => {
                // Ignore all type errors here since they'll be reported in `class_bases_of` anyway
                let errors = ErrorCollector::new(self.module().dupe(), ErrorStyle::Never);
                let ty = self.base_class_expr_infer_for_metadata(inner_expr, &errors);
                // Determine what class `type(X)` resolves to:
                // - If X is a class (type is type[C]), type(X) = C's metaclass
                // - If X is an instance (type is C), type(X) = C
                match self.untype_opt(ty.clone(), inner_expr.range(), &errors) {
                    Some(Type::ClassType(c)) => {
                        // X is a class C. type(C) = metaclass of C.
                        let class_obj = c.class_object();
                        let inner_metadata = self.get_metadata_for_class(class_obj);
                        let metaclass_ct = inner_metadata.metaclass(self.stdlib);
                        let metaclass_class = metaclass_ct.class_object();
                        let metaclass_metadata = self.get_metadata_for_class(metaclass_class);
                        BaseClassParseResult::Parsed(ParsedBaseClass {
                            class_object: metaclass_class.dupe(),
                            range,
                            metadata: metaclass_metadata,
                        })
                    }
                    Some(_) => BaseClassParseResult::InvalidType(ty, range),
                    None => {
                        // X is an instance of class C. type(X) = C.
                        match &ty {
                            Type::ClassType(c) => {
                                let class_obj = c.class_object();
                                let metadata = self.get_metadata_for_class(class_obj);
                                BaseClassParseResult::Parsed(ParsedBaseClass {
                                    class_object: class_obj.dupe(),
                                    range,
                                    metadata,
                                })
                            }
                            _ => BaseClassParseResult::InvalidType(ty, range),
                        }
                    }
                }
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
                            ErrorKind::InvalidArgument,
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
                            ErrorKind::InvalidArgument,
                            "Second argument to NewType is invalid".to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorKind::InvalidInheritance,
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
                            ErrorKind::InvalidArgument,
                            "Second argument to NewType is invalid".to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            range,
                            ErrorKind::InvalidInheritance,
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
                    if !is_new_type
                        && (metadata.is_final()
                            || (metadata.is_enum()
                                && !self.get_enum_members(&class_object).is_empty()))
                    {
                        self.error(
                            errors,
                            range,
                            ErrorKind::InvalidInheritance,
                            format!("Cannot extend final class `{}`", class_object.name()),
                        );
                    }
                    if is_new_type {
                        // TODO: raise an error for generic classes and other forbidden types such as hashable
                        if metadata.is_protocol() {
                            self.error(
                                errors,
                                range,
                                ErrorKind::InvalidArgument,
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
                            ErrorKind::InvalidInheritance,
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
        if let Some(class_fields) = self.get_class_fields(cls) {
            for name in class_fields.names() {
                if class_fields.is_field_annotated(name) {
                    all_fields.insert(name.clone(), is_total);
                }
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
                        &self.heap.mk_class_type(m.clone()),
                        &self.heap.mk_class_type(inherited.clone()),
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
        let metaclass_type = self.heap.mk_class_type(metaclass.clone());
        for (base_name, m) in base_metaclasses {
            let base_metaclass_type = self.heap.mk_class_type((*m).clone());
            if !self.is_subset_eq(&metaclass_type, &base_metaclass_type) {
                self.error(errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
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
                    &self.heap.mk_class_type(meta.clone()),
                    &self.heap.mk_class_type(self.stdlib.builtins_type().clone()),
                ) {
                    Some(meta)
                } else {
                    self.error(
                        errors,
                        raw_metaclass.range(),
                        ErrorKind::InvalidInheritance,
                        format!(
                            "Metaclass of `{}` has type `{}` which is not a subclass of `type`",
                            cls.name(),
                            self.for_display(self.heap.mk_class_type(meta)),
                        ),
                    );
                    None
                }
            }
            ty => {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
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
        let bases_with_mros: Vec<_> = bases
            .iter()
            .map(|base| {
                let mro = self.get_mro_for_class(base.class_object());
                (base, mro)
            })
            .collect();
        ClassMro::new(cls, bases_with_mros, errors)
    }

    /// Resolve `cls`'s disjoint-base representative.
    ///
    /// This is the first point where both metadata and MRO are available, so
    /// generated dataclass slots are handled here rather than in `ClassMetadata`.
    pub fn calculate_class_disjoint_base(
        &self,
        cls: &Class,
        errors: &ErrorCollector,
    ) -> ClassDisjointBase {
        let bases = self.get_base_types_for_class(cls);
        // A cyclic direct base has no representative, but other bases can
        // still be checked against each other.
        let mut survivors: Vec<Class> = Vec::new();
        for base in bases.iter() {
            let mro = self.get_mro_for_class(base.class_object());
            if matches!(&*mro, ClassMro::Cyclic) {
                continue;
            }
            let base_disjoint = self.get_disjoint_base_for_class(base.class_object());
            let Some(candidate) = base_disjoint.representative() else {
                continue;
            };
            // A more-specific (or equal) base already survives, drop this one.
            if survivors
                .iter()
                .any(|survivor| self.has_superclass(survivor, candidate))
            {
                continue;
            }
            // Otherwise drop any less-specific survivors and keep `candidate`.
            survivors.retain(|survivor| !self.has_superclass(candidate, survivor));
            survivors.push(candidate.dupe());
        }

        if survivors.len() > 1 {
            let ctx_classes: Vec<&Class> = std::iter::once(cls).chain(survivors.iter()).collect();
            let ctx = ClassDisplayContext::new(&ctx_classes);
            let listed = survivors
                .iter()
                .map(|s| format!("`{}`", ctx.display(s)))
                .join(", ");
            self.error(
                errors,
                cls.range(),
                ErrorKind::InvalidInheritance,
                format!(
                    "Class `{}` inherits from incompatible disjoint bases {}",
                    ctx.display(cls),
                    listed,
                ),
            );
        }

        let metadata = self.get_metadata_for_class(cls);
        let mro = self.get_mro_for_class(cls);
        let has_nonempty_generated_slots =
            self.has_nonempty_generated_slots_from_complete_mro(&metadata, &mro);

        // Skip `object` so narrowing's fallback to `object` stays meaningful.
        let local = !cls.is_builtin("object")
            && (metadata.is_local_disjoint_base() || has_nonempty_generated_slots);
        let local_representative = if local { Some(cls.dupe()) } else { None };
        let inherited_representative = survivors.into_iter().next();
        ClassDisjointBase::from_representative(local_representative.or(inherited_representative))
    }

    /// Returns whether the current class would synthesize non-empty dataclass slots.
    ///
    /// CPython dedups generated slots against inherited slot names, so this
    /// requires a complete MRO. A class-body `__slots__` suppresses dataclass
    /// slot synthesis even when the explicit slot names are dynamic.
    fn has_nonempty_generated_slots_from_complete_mro(
        &self,
        metadata: &ClassMetadata,
        mro: &ClassMro,
    ) -> bool {
        if metadata.is_typed_dict() || metadata.is_protocol() {
            return false;
        }
        if !metadata.has_local_dataclass_slots_request() {
            return false;
        }
        if metadata.has_explicit_slots() {
            return false;
        }
        if !mro.linearization_complete() {
            return false;
        }
        let Some(dataclass) = metadata.dataclass_metadata() else {
            return false;
        };

        let mut inherited_slot_names: SmallSet<Name> = SmallSet::new();
        for ancestor in mro.ancestors_no_object() {
            let ancestor_cls = ancestor.class_object();
            let ancestor_metadata = self.get_metadata_for_class(ancestor_cls);
            if let Some(s) = ancestor_metadata.slots_info() {
                inherited_slot_names.extend(s.names.iter().cloned());
            }
            // Only ancestors that synthesized slots materialize dataclass fields
            // as inherited slot names; inherited `kws.slots` is not enough.
            let ancestor_is_nominal =
                !ancestor_metadata.is_typed_dict() && !ancestor_metadata.is_protocol();
            if ancestor_is_nominal
                && ancestor_metadata.has_local_dataclass_slots_request()
                && !ancestor_metadata.has_explicit_slots()
                && let Some(ancestor_dataclass) = ancestor_metadata.dataclass_metadata()
            {
                inherited_slot_names.extend(ancestor_dataclass.instance_fields().cloned());
            }
        }
        dataclass
            .instance_fields()
            .any(|name| !inherited_slot_names.contains(name))
    }

    pub fn calculate_abstract_members(&self, cls: &Class) -> AbstractClassMembers {
        let metadata = self.get_metadata_for_class(cls);
        let mut fields_to_check: SmallSet<Name>;
        if metadata.extends_abc() || metadata.is_protocol() {
            fields_to_check = self
                .get_class_fields(cls)
                .map(|f| SmallSet::from_iter(f.names().cloned()))
                .unwrap_or_default();
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
            // If the class has a synthesized concrete implementation (e.g., `__dataclass_fields__`
            // from @dataclass), that satisfies any protocol requirement for this field.
            if self
                .get_class_member(cls, &field_name)
                .is_some_and(|f| !f.is_abstract() && !f.is_uninit_class_var())
            {
                continue;
            }
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

    pub fn calculate_subscript_symmetry(&self, cls: &Class) -> bool {
        // Built-in mutable containers are hardcoded as symmetric: their
        // typeshed slice overloads would otherwise classify them as asymmetric.
        if cls.is_builtin("list")
            || cls.is_builtin("dict")
            || cls.is_builtin("bytearray")
            || cls.is_builtin("memoryview")
        {
            return true;
        }

        let base = self.promote_silently(cls);
        let swallower = self.error_swallower();

        let Some(setitem_ty) = self.type_of_magic_dunder_attr(
            &base,
            &dunder::SETITEM,
            cls.range(),
            &swallower,
            None,
            "calculate_subscript_symmetry",
            false,
        ) else {
            return false;
        };
        let Some(getitem_ty) = self.type_of_magic_dunder_attr(
            &base,
            &dunder::GETITEM,
            cls.range(),
            &swallower,
            None,
            "calculate_subscript_symmetry",
            false,
        ) else {
            return false;
        };

        let setitem_sigs = setitem_ty.callable_signatures();
        let getitem_sigs = getitem_ty.callable_signatures();
        if setitem_sigs.is_empty() || getitem_sigs.is_empty() {
            return false;
        }

        let candidate = &getitem_sigs[0].ret;
        let all_getters_match = getitem_sigs
            .iter()
            .all(|sig| self.is_equivalent(&sig.ret, candidate));
        // Bound `__setitem__` has params `[key, value]`; we want `value` at index 1.
        let all_setters_match = setitem_sigs.iter().all(|sig| {
            matches!(&sig.params, Params::List(params)
                if params.items().get(1)
                    .is_some_and(|p| self.is_equivalent(p.as_type(), candidate)))
        });

        all_getters_match && all_setters_match
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
                && self.metaclass_extends_abcmeta(metaclass.class_object())
            {
                return true;
            }
            if base_metadata.extends_abc() {
                return true;
            }
        }
        if let Some(metaclass) = metaclass
            && self.metaclass_extends_abcmeta(metaclass.class_object())
        {
            return true;
        }
        false
    }

    /// Check if `metaclass_cls` is `abc.ABCMeta` or has `abc.ABCMeta` anywhere in its
    /// inheritance chain.
    fn metaclass_extends_abcmeta(&self, metaclass_cls: &Class) -> bool {
        if metaclass_cls.has_toplevel_qname("abc", "ABCMeta") {
            return true;
        }
        let metadata = self.get_metadata_for_class(metaclass_cls);
        metadata
            .base_class_objects()
            .iter()
            .any(|base| self.metaclass_extends_abcmeta(base))
    }
}
