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
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_util::display::DisplayWithCtx;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::alt::types::class_metadata::NamedTupleMetadata;
use crate::alt::types::class_metadata::ProtocolMetadata;
use crate::alt::types::class_metadata::TotalOrderingMetadata;
use crate::alt::types::class_metadata::TypedDictMetadata;
use crate::alt::types::pydantic::PydanticMetadata;
use crate::binding::base_class::BaseClass;
use crate::binding::binding::Key;
use crate::binding::pydantic::PydanticMetadataBinding;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
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

    // To avoid circular computation on targs, we have a special version of `expr_infer` that only recognize a small
    // subset of syntactical forms, and does not look into any subscript of any expr
    fn base_class_expr_infer(&self, expr: &Expr, errors: &ErrorCollector) -> Option<Type> {
        match expr {
            Expr::Name(x) => Some(
                self.get(&Key::BoundName(ShortIdentifier::expr_name(x)))
                    .arc_clone_ty(),
            ),
            Expr::Attribute(x) => {
                let base = self.base_class_expr_infer(&x.value, errors)?;
                Some(self.attr_infer_for_type(&base, &x.attr.id, x.range, errors, None))
            }
            Expr::Subscript(x) => self.base_class_expr_infer(&x.value, errors),
            _ => None,
        }
    }

    fn parse_base_class(&self, base: BaseClass, is_new_type: bool) -> BaseClassParseResult {
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
                    let class_object = typed_dict.class_object();
                    let class_metadata = self.get_metadata_for_class(class_object);
                    BaseClassParseResult::Parsed({
                        ParsedBaseClass {
                            class_object: class_object.dupe(),
                            range,
                            metadata: class_metadata,
                        }
                    })
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
            BaseClass::Expr(x) => {
                // Ignore all type errors here since they'll be reported in `class_bases_of` anyway
                let errors = ErrorCollector::new(self.module().dupe(), ErrorStyle::Never);
                match self.base_class_expr_infer(&x, &errors) {
                    None => BaseClassParseResult::InvalidExpr(x),
                    Some(ty) => match self.untype_opt(ty.clone(), x.range()) {
                        None => BaseClassParseResult::InvalidType(ty, x.range()),
                        Some(ty) => parse_base_class_type(ty),
                    },
                }
            }
            BaseClass::NamedTuple(..) => {
                parse_base_class_type(self.stdlib.named_tuple_fallback().clone().to_type())
            }
            BaseClass::TypedDict(..) | BaseClass::Generic(..) | BaseClass::Protocol(..) => {
                if is_new_type {
                    BaseClassParseResult::InvalidBase(base.range())
                } else {
                    BaseClassParseResult::Ignored
                }
            }
        }
    }

    pub fn class_metadata_of(
        &self,
        cls: &Class,
        bases: &[BaseClass],
        keywords: &[(Name, Expr)],
        decorators: &[(Idx<Key>, TextRange)],
        is_new_type: bool,
        special_base: &Option<Box<BaseClass>>,
        pydantic_metadata_binding: &PydanticMetadataBinding,
        errors: &ErrorCollector,
    ) -> ClassMetadata {
        let mut enum_metadata = None;
        let mut bases: Vec<BaseClass> = bases.to_vec();
        if let Some(special_base) = special_base {
            bases.push((**special_base).clone());
        }
        let mut protocol_metadata = Self::protocol_metadata(cls, bases.as_slice());
        let has_generic_base_class = bases.iter().any(|x| x.is_generic());
        let has_typed_dict_base_class = bases.iter().any(|x| x.is_typed_dict());

        let parsed_results = bases
            .into_iter()
            .map(|x| self.parse_base_class(x, is_new_type))
            .collect::<Vec<_>>();
        let contains_base_class_any = parsed_results.iter().any(|x| x.is_any());
        let bases_with_metadata = parsed_results.into_iter().filter_map(|x| match x {
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
            BaseClassParseResult::Parsed(ParsedBaseClass { class_object, range, metadata }) => {
                if metadata.is_final() {
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
                    }
                } else if metadata.is_new_type() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        "Subclassing a NewType not allowed".to_owned(),
                    );
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
                Some((class_object, metadata))
            }
        }).collect::<Vec<_>>();

        let has_base_any = contains_base_class_any
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.has_base_any());

        let named_tuple_metadata =
            bases_with_metadata
                .iter()
                .find_map(|(base_class_object, metadata)| {
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
        // If we inherit from a dataclass, inherit its metadata. Note that if this class is
        // itself decorated with @dataclass, we'll compute new metadata and overwrite this.
        let mut dataclass_metadata = bases_with_metadata.iter().find_map(|(_, metadata)| {
            let mut m = metadata.dataclass_metadata().cloned()?;
            // Avoid accidentally overwriting a non-synthesized `__init__`.
            m.kws.init = false;
            Some(m)
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
        if let Some(metaclass) = &metaclass {
            self.check_base_class_metaclasses(cls, metaclass, &base_metaclasses, errors);
            if self
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
                enum_metadata = Some(EnumMetadata {
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
        if let Some(dm) = dataclass_metadata.as_ref() {
            self.validate_frozen_dataclass_inheritance(cls, dm, &bases_with_metadata, errors);
        }

        let has_pydantic_base_model_base_class =
            bases_with_metadata.iter().any(|(base_class_object, _)| {
                base_class_object.has_qname(ModuleName::pydantic().as_str(), "BaseModel")
            });

        let is_pydantic_model = has_pydantic_base_model_base_class
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.is_pydantic_model());

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

        // Determine final PydanticMetadata only if the class inherits from BaseModel in the MRO
        let pydantic_metadata = match pydantic_metadata_binding {
            PydanticMetadataBinding { frozen } if is_pydantic_model => {
                Some(PydanticMetadata { frozen: *frozen })
            }
            _ => None,
        };

        ClassMetadata::new(
            bases,
            metaclass,
            keywords,
            typed_dict_metadata,
            named_tuple_metadata,
            enum_metadata,
            protocol_metadata,
            dataclass_metadata,
            has_generic_base_class,
            has_base_any,
            is_new_type,
            is_final,
            total_ordering_metadata,
            dataclass_transform_metadata,
            pydantic_metadata,
        )
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
}
