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
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::base_class::BaseClass;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::alt::types::class_metadata::NamedTupleMetadata;
use crate::alt::types::class_metadata::ProtocolMetadata;
use crate::alt::types::class_metadata::TotalOrderingMetadata;
use crate::alt::types::class_metadata::TypedDictMetadata;
use crate::binding::binding::Key;
use crate::error::collector::ErrorCollector;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassKeywords;
use crate::types::keywords::DataclassTransformKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn new_type_base(
        &self,
        base_type_and_range: Option<(Type, TextRange)>,
        fallback_range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<(ClassType, Arc<ClassMetadata>)> {
        match base_type_and_range {
            // TODO: raise an error for generic classes and other forbidden types such as hashable
            Some((Type::ClassType(c), range)) => {
                let base_cls = c.class_object();
                let base_class_metadata = self.get_metadata_for_class(base_cls);
                if base_class_metadata.is_protocol() {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidArgument,
                        None,
                        "Second argument to NewType cannot be a protocol".to_owned(),
                    );
                }
                if c.targs().as_slice().iter().any(|ty| {
                    ty.any(|ty| {
                        matches!(
                            ty,
                            Type::TypeVar(_) | Type::TypeVarTuple(_) | Type::ParamSpec(_)
                        )
                    })
                }) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidArgument,
                        None,
                        "Second argument to NewType cannot be an unbound generic".to_owned(),
                    );
                }
                let metadata = self.get_metadata_for_class(c.class_object());
                Some((c, metadata))
            }
            Some((Type::Tuple(tuple), _)) => {
                let class_ty = self.erase_tuple_type(tuple);
                let metadata = self.get_metadata_for_class(class_ty.class_object());
                Some((class_ty, metadata))
            }
            Some((_, range)) => {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidArgument,
                    None,
                    "Second argument to NewType is invalid".to_owned(),
                );
                None
            }
            None => {
                self.error(
                    errors,
                    fallback_range,
                    ErrorKind::InvalidArgument,
                    None,
                    "Second argument to NewType is invalid".to_owned(),
                );
                None
            }
        }
    }

    fn protocol_metadata(cls: &Class, bases: &[BaseClass]) -> Option<ProtocolMetadata> {
        if bases.iter().any(|x| matches!(x, BaseClass::Protocol(_))) {
            Some(ProtocolMetadata {
                members: cls.fields().cloned().collect(),
                is_runtime_checkable: false,
            })
        } else {
            None
        }
    }

    pub fn class_metadata_of(
        &self,
        cls: &Class,
        bases: &[Expr],
        keywords: &[(Name, Expr)],
        decorators: &[(Idx<Key>, TextRange)],
        is_new_type: bool,
        special_base: &Option<Box<BaseClass>>,
        errors: &ErrorCollector,
    ) -> ClassMetadata {
        let mut is_typed_dict = false;
        let mut named_tuple_metadata = None;
        let mut enum_metadata = None;
        let mut dataclass_metadata = None;
        let mut bases: Vec<BaseClass> = bases.map(|x| self.base_class_of(x, errors));
        if let Some(special_base) = special_base {
            bases.push((**special_base).clone());
        }
        let mut protocol_metadata = Self::protocol_metadata(cls, bases.as_slice());

        let mut has_base_any = false;
        let mut has_generic_base_class = false;
        // If this class inherits from a dataclass_transform-ed class, record the defaults that we
        // should use for dataclass parameters.
        let mut dataclass_defaults_from_base_class = None;
        // This is set when a class is decorated with `@typing.dataclass_transform(...)`. Note that
        // this does not turn the class into a dataclass! Instead, it becomes a special base class
        // (or metaclass) that turns child classes into dataclasses.
        let mut dataclass_transform_metadata = None;
        let bases_with_metadata = bases
            .iter()
            .filter_map(|x| {
                let base_type_and_range = match x {
                    BaseClass::Expr(x) => Some((self.expr_untype(x, TypeFormContext::BaseClassList, errors), x.range())),
                    BaseClass::TypedDict => {
                        is_typed_dict = true;
                        None
                    }
                    BaseClass::NamedTuple(range) => {
                        Some((self.stdlib.named_tuple_fallback().clone().to_type(), *range))
                    }
                    BaseClass::Generic(ts) | BaseClass::Protocol(ts) if !ts.is_empty() => {
                        has_generic_base_class = true;
                        None
                    }
                    // Skip over empty generic. Empty protocol is only relevant for `protocol_metadata`, defined
                    // above so we can skip it here.
                    BaseClass::Generic(_) | BaseClass::Protocol(_) => None
                };
                if is_new_type {
                    self.new_type_base(base_type_and_range, cls.range(), errors)
                } else {
                    match base_type_and_range {
                        Some((Type::ClassType(c), range)) => {
                            let base_cls = c.class_object();
                            let base_class_metadata = self.get_metadata_for_class(base_cls);
                            if base_class_metadata.has_base_any() {
                                has_base_any = true;
                            }
                            if base_class_metadata.is_typed_dict() {
                                is_typed_dict = true;
                            }
                            if base_class_metadata.is_final() {
                                self.error(errors,
                                    range,
                                    ErrorKind::InvalidInheritance,
                                    None,
                                    format!("Cannot extend final class `{}`", base_cls.name()),
                                );
                            }
                           if base_class_metadata.is_new_type() {
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::InvalidInheritance,
                                    None,
                                    "Subclassing a NewType not allowed".to_owned(),
                                );
                            }
                            if base_cls.has_qname(ModuleName::type_checker_internals().as_str(), "NamedTupleFallback")
                            {
                                if named_tuple_metadata.is_none() {
                                    named_tuple_metadata = Some(NamedTupleMetadata {
                                        elements: self.get_named_tuple_elements(cls, errors)
                                    })
                                }
                            } else if let Some(base_named_tuple) = base_class_metadata.named_tuple_metadata() {
                                if named_tuple_metadata.is_none() {
                                    named_tuple_metadata = Some(base_named_tuple.clone());
                                }
                            }
                            if let Some(proto) = &mut protocol_metadata {
                                if let Some(base_proto) = base_class_metadata.protocol_metadata() {
                                    proto.members.extend(base_proto.members.iter().cloned());
                                    if base_proto.is_runtime_checkable {
                                        proto.is_runtime_checkable = true;
                                    }
                                } else {
                                    self.error(errors,
                                        range,
                                        ErrorKind::InvalidInheritance,
                                        None,
                                        "If `Protocol` is included as a base class, all other bases must be protocols".to_owned(),
                                    );
                                }
                            }
                            if dataclass_metadata.is_none() && let Some(base_dataclass) = base_class_metadata.dataclass_metadata() {
                                // If we inherit from a dataclass, inherit its metadata. Note that if this class is
                                // itself decorated with @dataclass, we'll compute new metadata and overwrite this.
                                dataclass_metadata = Some(base_dataclass.clone());
                            }
                            if let Some(m) = base_class_metadata.dataclass_transform_metadata() {
                                dataclass_defaults_from_base_class = Some(m.clone());
                                // When a class C is transformed into a dataclass via inheriting from a class decorated
                                // with `@dataclass_transform(...)`, then C in turn causes classes inheriting from it
                                // to be transformed (and so on). Note that this differs from dataclass transformation
                                // via a decorator in that if you inherit from a class transformed via decorator, you
                                // inherit its dataclass-ness but your own fields are *not* transformed.
                                dataclass_transform_metadata = Some(m.clone());
                            }
                            Some((c, base_class_metadata))
                        }
                        Some((Type::Tuple(tuple), _)) => {
                            let class_ty = self.erase_tuple_type(tuple);
                            let metadata = self.get_metadata_for_class(class_ty.class_object());
                            Some((class_ty, metadata))
                        }
                        Some((Type::TypedDict(typed_dict), _)) => {
                            is_typed_dict = true;
                            let class_object = typed_dict.class_object();
                            let class_metadata = self.get_metadata_for_class(class_object);
                            // HACK HACK HACK - TypedDict instances behave very differently from instances of other
                            // classes, so we don't represent them as ClassType in normal typechecking logic. However,
                            // class ancestors are represented as ClassType all over the code base, and changing this
                            // would be quite painful. So we convert TypedDict to ClassType in this one spot. Please do
                            // not do this anywhere else.
                            Some((
                                ClassType::new(typed_dict.class_object().dupe(), typed_dict.targs().clone()),
                                class_metadata,
                            ))
                        }
                        // todo zeina: Ideally, we can directly add this class to the list of base classes. Revisit this when fixing the "Any" representation.
                        Some((Type::Any(_), _)) => {
                            has_base_any = true;
                            None
                        }
                        Some((t, range)) => {
                            self.error(
                                errors, range, ErrorKind::InvalidInheritance, None,
                                format!("Invalid base class: `{}`", self.for_display(t)));
                            has_base_any = true;
                            None
                        }
                        None => None,
                    }
                }
            })
            .collect::<Vec<_>>();
        if named_tuple_metadata.is_some() && bases_with_metadata.len() > 1 {
            self.error(
                errors,
                cls.range(),
                ErrorKind::InvalidInheritance,
                None,
                "Named tuples do not support multiple inheritance".to_owned(),
            );
        }
        let (metaclasses, keywords): (Vec<_>, Vec<(_, _)>) =
            keywords.iter().partition_map(|(n, x)| match n.as_str() {
                "metaclass" => Either::Left(x),
                _ => Either::Right((n.clone(), self.expr_infer(x, errors))),
            });
        // This is set when we should apply dataclass-like transformations to the class. The class
        // should be transformed if:
        // - it inherits from a base class decorated with `dataclass_transform(...)`, or
        // - it inherits from a base class whose metaclass is decorated with `dataclass_transform(...)`, or
        // - it is decorated with a decorator that is decorated with `dataclass_transform(...)`.
        let mut dataclass_from_dataclass_transform = None;
        if let Some(defaults) = dataclass_defaults_from_base_class {
            // This class inherits from a dataclass_transform-ed base class, so its keywords are
            // interpreted as dataclass keywords.
            let map = keywords.clone().into_iter().collect::<OrderedMap<_, _>>();
            dataclass_from_dataclass_transform = Some((
                DataclassKeywords::from_type_map(&TypeMap(map), &defaults),
                defaults.field_specifiers,
            ));
        }
        let typed_dict_metadata = if is_typed_dict {
            // Validate that only 'total' keyword is allowed for TypedDict and determine is_total
            let mut is_total = true;
            for (name, value) in &keywords {
                if name.as_str() != "total" {
                    self.error(
                        errors,
                        cls.range(),
                        ErrorKind::BadTypedDict,
                        None,
                        format!(
                            "TypedDict does not support keyword argument `{}`",
                            name.as_str()
                        ),
                    );
                } else if matches!(value, Type::Literal(Lit::Bool(false))) {
                    is_total = false;
                }
            }
            let fields =
                self.calculate_typed_dict_metadata_fields(cls, &bases_with_metadata, is_total);
            Some(TypedDictMetadata { fields })
        } else {
            None
        };
        let base_metaclasses = bases_with_metadata
            .iter()
            .filter_map(|(b, metadata)| metadata.metaclass().map(|m| (b.name(), m)))
            .collect::<Vec<_>>();
        let metaclass = self.calculate_metaclass(
            cls,
            metaclasses.into_iter().next(),
            &base_metaclasses,
            errors,
        );
        if let Some(c) = &metaclass
            && let Some(m) = self
                .get_metadata_for_class(c.class_object())
                .dataclass_transform_metadata()
        {
            dataclass_transform_metadata = Some(m.clone());
        }
        let empty_tparams = self.get_class_tparams(cls).is_empty();
        if let Some(metaclass) = &metaclass {
            self.check_base_class_metaclasses(cls, metaclass, &base_metaclasses, errors);
            if self
                .as_superclass(metaclass, self.stdlib.enum_meta().class_object())
                .is_some()
            {
                if !empty_tparams {
                    self.error(
                        errors,
                        cls.range(),
                        ErrorKind::InvalidInheritance,
                        None,
                        "Enums may not be generic".to_owned(),
                    );
                }
                enum_metadata = Some(EnumMetadata {
                    // A generic enum is an error, but we create Any type args anyway to handle it gracefully.
                    cls: self.promote_nontypeddict_silently_to_classtype(cls),
                    has_value: bases_with_metadata.iter().any(|(base, _)| {
                        base.class_object().contains(&Name::new_static("_value_"))
                    }),
                    is_flag: bases_with_metadata.iter().any(|(base, _)| {
                        self.is_subset_eq(
                            &Type::ClassType(base.clone()),
                            &Type::ClassType(self.stdlib.enum_flag().clone()),
                        )
                    }),
                })
            }
            if is_typed_dict {
                self.error(
                    errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
                    None,
                    "Typed dictionary definitions may not specify a metaclass".to_owned(),
                );
            }
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
                    ErrorKind::InvalidInheritance,
                    None,
                    "Metaclass may not be an unbound generic".to_owned(),
                );
            }
        }
        let mut is_final = false;
        let mut total_ordering_metadata = None;
        for (decorator_key, decorator_range) in decorators {
            let decorator = self.get_idx(*decorator_key);
            let decorator_ty = decorator.ty();
            match decorator_ty.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::Final)) => {
                    is_final = true;
                }
                Some(CalleeKind::Function(FunctionKind::RuntimeCheckable)) => {
                    if let Some(proto) = &mut protocol_metadata {
                        proto.is_runtime_checkable = true;
                    } else {
                        self.error(
                            errors,
                            cls.range(),
                            ErrorKind::InvalidArgument,
                            None,
                            "@runtime_checkable can only be applied to Protocol classes".to_owned(),
                        );
                    }
                }
                Some(CalleeKind::Function(FunctionKind::TotalOrdering)) => {
                    total_ordering_metadata = Some(TotalOrderingMetadata {
                        location: *decorator_range,
                    });
                }
                // `@dataclass`
                Some(CalleeKind::Function(FunctionKind::Dataclass)) => {
                    let dataclass_fields = self.get_dataclass_fields(cls, &bases_with_metadata);
                    dataclass_metadata = Some(DataclassMetadata {
                        fields: dataclass_fields,
                        kws: DataclassKeywords::new(),
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                    });
                }
                // `@dataclass(...)`
                _ if let Type::KwCall(call) = decorator_ty
                    && call.has_function_kind(FunctionKind::Dataclass) =>
                {
                    let dataclass_fields = self.get_dataclass_fields(cls, &bases_with_metadata);
                    dataclass_metadata = Some(DataclassMetadata {
                        fields: dataclass_fields,
                        kws: DataclassKeywords::from_type_map(
                            &call.keywords,
                            &DataclassTransformKeywords::new(),
                        ),
                        field_specifiers: vec![
                            CalleeKind::Function(FunctionKind::DataclassField),
                            CalleeKind::Class(ClassKind::DataclassField),
                        ],
                    });
                }
                // `@dataclass_transform(...)`
                _ if let Type::KwCall(call) = decorator_ty
                    && call.has_function_kind(FunctionKind::DataclassTransform) =>
                {
                    dataclass_transform_metadata =
                        Some(DataclassTransformKeywords::from_type_map(&call.keywords));
                }
                // `@foo` where `foo` is decorated with `@dataclass_transform(...)`
                _ if let Some(defaults) = decorator_ty.dataclass_transform_metadata() => {
                    dataclass_from_dataclass_transform = Some((
                        DataclassKeywords::from_type_map(&TypeMap::new(), &defaults),
                        defaults.field_specifiers,
                    ));
                }
                // `@foo(...)` where `foo` is decorated with `@dataclass_transform(...)`
                _ if let Type::KwCall(call) = decorator_ty
                    && let Some(defaults) =
                        &call.func_metadata.flags.dataclass_transform_metadata =>
                {
                    dataclass_from_dataclass_transform = Some((
                        DataclassKeywords::from_type_map(&call.keywords, defaults),
                        defaults.field_specifiers.clone(),
                    ));
                }
                _ => {}
            }
        }
        if let Some((kws, field_specifiers)) = dataclass_from_dataclass_transform {
            dataclass_metadata = Some(DataclassMetadata {
                fields: self.get_dataclass_fields(cls, &bases_with_metadata),
                kws,
                field_specifiers,
            });
        }
        if is_typed_dict
            && let Some(bad) = bases_with_metadata.iter().find(|x| !x.1.is_typed_dict())
        {
            self.error(errors,
                cls.range(),
                ErrorKind::InvalidInheritance,
                None,
                format!("`{}` is not a typed dictionary. Typed dictionary definitions may only extend other typed dictionaries.", bad.0),
            );
        }
        let bases_with_metadata = if is_typed_dict && bases_with_metadata.is_empty() {
            // This is a "fallback" class that contains attributes that are available on all TypedDict subclasses.
            // Note that this also makes those attributes available on *instances* of said subclasses; this is
            // desirable for methods but problematic for fields like `__total__` that should be available on the class
            // but not the instance. For now, we make all fields available on both classes and instances.
            let td_fallback = self.stdlib.typed_dict_fallback();
            vec![(
                td_fallback.clone(),
                self.get_metadata_for_class(td_fallback.class_object()),
            )]
        } else {
            bases_with_metadata
        };
        // We didn't find any type parameters for this class, but it may have ones we don't know about if:
        // - the class inherits from Any, or
        // - the class inherits from Generic[...] or Protocol [...]. We probably dropped the type
        //   arguments because we found an error in them.
        let has_unknown_tparams = empty_tparams && (has_base_any || has_generic_base_class);
        ClassMetadata::new(
            cls,
            bases_with_metadata,
            metaclass,
            keywords,
            typed_dict_metadata,
            named_tuple_metadata,
            enum_metadata,
            protocol_metadata,
            dataclass_metadata,
            has_base_any,
            is_new_type,
            is_final,
            has_unknown_tparams,
            total_ordering_metadata,
            dataclass_transform_metadata,
            errors,
        )
    }

    fn calculate_typed_dict_metadata_fields(
        &self,
        cls: &Class,
        bases_with_metadata: &[(ClassType, Arc<ClassMetadata>)],
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
    ) -> Option<ClassType> {
        let direct_meta = raw_metaclass.and_then(|x| self.direct_metaclass(cls, x, errors));

        if let Some(metaclass) = direct_meta {
            Some(metaclass)
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
            if !self
                .solver()
                .is_subset_eq(&metaclass_type, &base_metaclass_type, self.type_order())
            {
                self.error(errors,
                    cls.range(),
                    ErrorKind::InvalidInheritance,
                    None,
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
                        ErrorKind::InvalidInheritance,
                        None,
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
                    ErrorKind::InvalidInheritance,
                    None,
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
        let metadata = self.get_metadata_for_class(cls);
        let bases_with_mros = metadata
            .bases_with_metadata()
            .iter()
            .map(|(base, _)| {
                let mro = self.get_mro_for_class(base.class_object());
                (base, mro)
            })
            .collect();
        ClassMro::new(cls, bases_with_mros, errors)
    }
}
