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
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
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
    fn protocol_metadata(cls: &Class, bases: &[BaseClass]) -> Option<ProtocolMetadata> {
        if bases.iter().any(|x| matches!(x, BaseClass::Protocol(..))) {
            Some(ProtocolMetadata {
                members: cls.fields().cloned().collect(),
                is_runtime_checkable: false,
            })
        } else {
            None
        }
    }

    fn find_has_generic_base_class(bases: &[BaseClass]) -> bool {
        bases.iter().any(|x| match x {
            BaseClass::Generic(ts, ..) | BaseClass::Protocol(ts, ..) if !ts.is_empty() => true,
            _ => false,
        })
    }

    fn find_has_typed_dict_base_class(bases: &[BaseClass]) -> bool {
        bases.iter().any(|x| match x {
            BaseClass::TypedDict(..) => true,
            _ => false,
        })
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
        let mut enum_metadata = None;
        let mut bases: Vec<BaseClass> = bases.map(|x| self.base_class_of(x, errors));
        if let Some(special_base) = special_base {
            bases.push((**special_base).clone());
        }
        let mut protocol_metadata = Self::protocol_metadata(cls, bases.as_slice());
        let has_generic_base_class = Self::find_has_generic_base_class(bases.as_slice());
        let has_typed_dict_base_class = Self::find_has_typed_dict_base_class(bases.as_slice());

        let bases_with_range = bases
            .into_iter()
            .filter_map(|x| {
                let range = x.range();
                match x {
                    BaseClass::Expr(x) => Some((
                        self.expr_untype(&x, TypeFormContext::BaseClassList, errors),
                        range,
                    )),
                    BaseClass::NamedTuple(..) => {
                        Some((self.stdlib.named_tuple_fallback().clone().to_type(), range))
                    }
                    BaseClass::TypedDict(..) => {
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
                    BaseClass::Generic(args, _) | BaseClass::Protocol(args, _) => {
                        if is_new_type {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Second argument to NewType is invalid".to_owned(),
                            );
                        } else {
                            let mut type_var_tuple_count = 0;
                            args.iter().for_each(|x| {
                                let ty = self.expr_untype(x, TypeFormContext::GenericBase, errors);
                                if let Type::Unpack(unpacked) = &ty
                                    && unpacked.is_kind_type_var_tuple()
                                {
                                    if type_var_tuple_count == 1 {
                                        self.error(
                                            errors,
                                            x.range(),
                                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                            "There cannot be more than one TypeVarTuple type parameter".to_owned(),
                                        );
                                    }
                                    type_var_tuple_count += 1;
                                }
                            });
                        }
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        let mut tuple_base = if is_new_type {
            None
        } else {
            bases_with_range.iter().find_map(|(ty, _)| {
                if let Type::Tuple(tuple) = ty {
                    Some(tuple.clone())
                } else {
                    None
                }
            })
        };

        let (bases_with_range_and_metadata, invalid_bases): (
            Vec<(ClassType, TextRange, Arc<ClassMetadata>)>,
            Vec<()>,
        ) = bases_with_range
            .into_iter()
            .map(|base_type_and_range| {
                // Return Ok() if the base class is valid, or Err() if it is not.
                match base_type_and_range {
                    (Type::ClassType(c), range) => {
                        let base_cls = c.class_object();
                        let base_class_metadata = self.get_metadata_for_class(base_cls);
                        Ok((c, range, base_class_metadata))
                    }
                    (Type::Tuple(tuple), range) => {
                        let class_ty = self.erase_tuple_type(tuple);
                        let metadata = self.get_metadata_for_class(class_ty.class_object());
                        Ok((class_ty, range, metadata))
                    }
                    (Type::TypedDict(typed_dict), range) => {
                        if is_new_type {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Second argument to NewType is invalid".to_owned(),
                            );
                            Err(())
                        } else {
                            let class_object = typed_dict.class_object();
                            let class_metadata = self.get_metadata_for_class(class_object);
                            // HACK HACK HACK - TypedDict instances behave very differently from instances of other
                            // classes, so we don't represent them as ClassType in normal typechecking logic. However,
                            // class ancestors are represented as ClassType all over the code base, and changing this
                            // would be quite painful. So we convert TypedDict to ClassType in this one spot. Please do
                            // not do this anywhere else.
                            Ok((
                                ClassType::new(
                                    typed_dict.class_object().dupe(),
                                    typed_dict.targs().clone(),
                                ),
                                range,
                                class_metadata,
                            ))
                        }
                    }
                    (t, range) => {
                        if is_new_type {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                "Second argument to NewType is invalid".to_owned(),
                            );
                        } else if !t.is_any() {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                format!("Invalid base class: `{}`", self.for_display(t)),
                            );
                        }
                        Err(())
                    }
                }
            })
            .partition_result();

        let bases_with_metadata = bases_with_range_and_metadata
            .into_iter()
            .map(|(cls, range, metadata)| {
                if metadata.is_final() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        format!("Cannot extend final class `{}`", cls.name()),
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
                    }
                    if cls.targs().as_slice().iter().any(|ty| {
                        ty.any(|ty| {
                            matches!(
                                ty,
                                Type::TypeVar(_)
                                    | Type::TypeVarTuple(_)
                                    | Type::ParamSpec(_)
                            )
                        })
                    }) {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "Second argument to NewType cannot be an unbound generic"
                                .to_owned(),
                        );
                    }
                } else if metadata.is_new_type() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        "Subclassing a NewType not allowed".to_owned(),
                    );
                }
                if let Some(base_class_tuple_base) = metadata.tuple_base() {
                    if let Some(existing_tuple_base) = &tuple_base {
                        if existing_tuple_base.is_any_tuple() {
                            tuple_base = Some(base_class_tuple_base.clone());
                        } else if !base_class_tuple_base.is_any_tuple()
                            && base_class_tuple_base != existing_tuple_base {
                                self.error(errors,
                                    range,
                                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                    format!("Cannot extend multiple incompatible tuples: `{}` and `{}`",
                                        self.for_display(Type::Tuple(existing_tuple_base.clone())),
                                        self.for_display(Type::Tuple(base_class_tuple_base.clone())),
                                ),
                                );
                            }
                    } else {
                        tuple_base = Some(base_class_tuple_base.clone());
                    }
                }
                if let Some(proto) = &mut protocol_metadata {
                    if let Some(base_proto) = metadata.protocol_metadata() {
                        proto.members.extend(base_proto.members.iter().cloned());
                        if base_proto.is_runtime_checkable {
                            proto.is_runtime_checkable = true;
                        }
                    } else {
                        self.error(errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            "If `Protocol` is included as a base class, all other bases must be protocols".to_owned(),
                        );
                    }
                }
                (cls, metadata)
            })
            .collect::<Vec<_>>();

        let has_base_any = !invalid_bases.is_empty()
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.has_base_any());

        let named_tuple_metadata = bases_with_metadata.iter().find_map(|(base_cls, metadata)| {
            let base_class_object = base_cls.class_object();
            if base_class_object.has_qname(
                ModuleName::type_checker_internals().as_str(),
                "NamedTupleFallback",
            ) {
                Some(NamedTupleMetadata {
                    elements: self.get_named_tuple_elements(cls, errors),
                })
            } else {
                metadata.named_tuple_metadata().cloned()
            }
        });
        if named_tuple_metadata.is_some() && bases_with_metadata.len() > 1 {
            self.error(
                errors,
                cls.range(),
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                "Named tuples do not support multiple inheritance".to_owned(),
            );
        }
        let (metaclasses, keywords): (Vec<_>, Vec<(_, _)>) =
            keywords.iter().partition_map(|(n, x)| match n.as_str() {
                "metaclass" => Either::Left(x),
                _ => Either::Right((n.clone(), self.expr_infer(x, errors))),
            });

        // If this class inherits from a dataclass_transform-ed class, record the defaults that we
        // should use for dataclass parameters.
        let dataclass_defaults_from_base_class = bases_with_metadata
            .iter()
            .find_map(|(_, metadata)| metadata.dataclass_transform_metadata().cloned());
        // This is set when a class is decorated with `@typing.dataclass_transform(...)`. Note that
        // this does not turn the class into a dataclass! Instead, it becomes a special base class
        // (or metaclass) that turns child classes into dataclasses.
        let mut dataclass_transform_metadata = dataclass_defaults_from_base_class.clone();
        let mut dataclass_metadata = bases_with_metadata
            .iter()
            .find_map(|(_, metadata)| metadata.dataclass_metadata().cloned());
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
        let is_typed_dict = has_typed_dict_base_class
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.is_typed_dict());
        let typed_dict_metadata = if is_typed_dict {
            // Validate that only 'total' keyword is allowed for TypedDict and determine is_total
            let mut is_total = true;
            for (name, value) in &keywords {
                if name.as_str() != "total" {
                    self.error(
                        errors,
                        cls.range(),
                        ErrorInfo::Kind(ErrorKind::BadTypedDict),
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
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
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
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
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
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
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
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
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
        if let Some(dm) = dataclass_metadata.as_ref() {
            self.validate_frozen_dataclass_inheritance(cls, dm, &bases_with_metadata, errors);
        }
        ClassMetadata::new(
            bases_with_metadata,
            metaclass,
            keywords,
            typed_dict_metadata,
            named_tuple_metadata,
            enum_metadata,
            protocol_metadata,
            dataclass_metadata,
            tuple_base,
            has_base_any,
            is_new_type,
            is_final,
            has_unknown_tparams,
            total_ordering_metadata,
            dataclass_transform_metadata,
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
