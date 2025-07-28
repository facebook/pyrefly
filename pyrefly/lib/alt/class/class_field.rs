/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_types::callable::Params;
use pyrefly_types::simplify::unions;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::ResultExt;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::Attribute;
use crate::alt::attr::DescriptorBase;
use crate::alt::attr::NoAccessReason;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::binding::binding::Binding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::MethodThatSetsAttr;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::annotation::Annotation;
use crate::types::annotation::Qualifier;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::literal::Lit;
use crate::types::literal::LitEnum;
use crate::types::quantified::Quantified;
use crate::types::read_only::ReadOnlyReason;
use crate::types::typed_dict::TypedDict;
use crate::types::typed_dict::TypedDictField;
use crate::types::types::BoundMethod;
use crate::types::types::BoundMethodType;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::SuperObj;
use crate::types::types::TArgs;
use crate::types::types::Type;

/// Helper type for going from binding information to the calculated class field.
///
/// TODO(stroxler): This type is mostly an artifact of a refactor, it used to be
/// used in `BindingClassField`. We probably can eliminate it.
enum RawClassFieldInitialization {
    /// At the point where the field is declared, it does not have an initial value. This includes
    /// fields declared but not initialized in the class body, and instance-only fields of
    /// synthesized classes.
    Uninitialized,
    /// The field is set in a method *and declared nowhere else*. Consider:
    ///   class A:
    ///     x: int
    ///     def __init__(self):
    ///         self.x = 42
    ///         self.y = 42
    /// `x`'s initialization type is `Uninitialized`, whereas y's is `Method('__init__')`.
    Method(MethodThatSetsAttr),
    /// The field is declared and initialized to a value in the class body.
    ///
    /// If the value is from an assignment, stores the expression that the field is assigned to,
    /// which is needed for some cases like dataclass fields. The `None` case is for fields that
    /// have values which don't come from assignment (e.g. function defs, imports in a class body)
    ClassBody(Option<Expr>),
}

/// Correctly analyzing which attributes are visible on class objects, as well
/// as handling method binding correctly, requires distinguishing which fields
/// are assigned values in the class body.
#[derive(Clone, Debug, TypeEq, VisitMut, PartialEq, Eq)]
pub enum ClassFieldInitialization {
    /// If this is a dataclass field, DataclassFieldKeywords stores the field's
    /// dataclass flags (which are options that control how fields behave).
    ClassBody(Option<DataclassFieldKeywords>),
    /// This field is initialized in a method. Note that this applies only if the field is not
    /// declared anywhere else.
    Method,
    /// The field is not initialized at the point where it is declared. This usually means that the
    /// field is instance-only and is declared but not initialized in the class body.
    Uninitialized,
    /// The field is not initialized in the class body or any method in the class,
    /// but we treat it as if it was initialized.
    /// For example, any field defined in a stub file.
    Magic,
}

impl Display for ClassFieldInitialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClassBody(_) => write!(f, "initialized on class body"),
            Self::Method => write!(f, "initialized in method"),
            Self::Uninitialized => write!(f, "initialized on instances"),
            Self::Magic => {
                write!(f, "not initialized on class body/method")
            }
        }
    }
}

impl ClassFieldInitialization {
    fn recursive() -> Self {
        ClassFieldInitialization::ClassBody(None)
    }
}

/// Raw information about an attribute declared somewhere in a class. We need to
/// know whether it is initialized in the class body in order to determine
/// both visibility rules and whether method binding should be performed.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
pub struct ClassField(ClassFieldInner);

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
enum ClassFieldInner {
    // TODO(stroxler): We should refactor `ClassFieldInner` into enum cases; currently
    // the semantics are encoded ad-hoc into the fields of a large product which
    // has made hacking features relatively easy, but makes the code hard to read.
    Simple {
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        /// The reason this field is read-only. `None` indicates it is read-write.
        read_only_reason: Option<ReadOnlyReason>,
        // Descriptor getter method, if there is one. `None` indicates no getter.
        descriptor_getter: Option<Type>,
        // Descriptor setter method, if there is one. `None` indicates no setter.
        descriptor_setter: Option<Type>,
        is_function_without_return_annotation: bool,
    },
}

impl Display for ClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ClassFieldInner::Simple {
                ty, initialization, ..
            } => write!(f, "{ty} ({initialization})"),
        }
    }
}

impl ClassField {
    fn new(
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        read_only_reason: Option<ReadOnlyReason>,
        descriptor_getter: Option<Type>,
        descriptor_setter: Option<Type>,
        is_function_without_return_annotation: bool,
    ) -> Self {
        Self(ClassFieldInner::Simple {
            ty,
            annotation,
            initialization,
            read_only_reason,
            descriptor_getter,
            descriptor_setter,
            is_function_without_return_annotation,
        })
    }

    pub fn for_variance_inference(
        &self,
    ) -> Option<(
        &Type,
        Option<&Annotation>,
        bool,
        &Option<Type>,
        &Option<Type>,
    )> {
        match &self.0 {
            ClassFieldInner::Simple {
                ty,
                annotation,
                descriptor_getter,
                descriptor_setter,
                ..
            } => Some((
                ty,
                annotation.as_ref(),
                self.is_read_only(),
                descriptor_getter,
                descriptor_setter,
            )),
        }
    }

    /// Get the raw type. Only suitable for use in this module, this type may
    /// not correspond to the type of any actual operations on the attribute.
    fn raw_type(&self) -> &Type {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty,
        }
    }

    pub fn new_synthesized(ty: Type) -> Self {
        ClassField(ClassFieldInner::Simple {
            ty,
            annotation: None,
            initialization: ClassFieldInitialization::ClassBody(None),
            read_only_reason: None,
            descriptor_getter: None,
            descriptor_setter: None,
            is_function_without_return_annotation: false,
        })
    }

    pub fn recursive() -> Self {
        Self(ClassFieldInner::Simple {
            ty: Type::any_implicit(),
            annotation: None,
            initialization: ClassFieldInitialization::recursive(),
            read_only_reason: None,
            descriptor_getter: None,
            descriptor_setter: None,
            is_function_without_return_annotation: false,
        })
    }

    fn initialization(&self) -> ClassFieldInitialization {
        match &self.0 {
            ClassFieldInner::Simple { initialization, .. } => initialization.clone(),
        }
    }

    fn instantiate_for(&self, instance: &Instance) -> Self {
        match &self.0 {
            ClassFieldInner::Simple {
                ty,
                annotation,
                initialization,
                read_only_reason,
                descriptor_getter,
                descriptor_setter,
                is_function_without_return_annotation,
            } => Self(ClassFieldInner::Simple {
                ty: instance.instantiate_member(ty.clone()),
                annotation: annotation.clone(),
                initialization: initialization.clone(),
                read_only_reason: read_only_reason.clone(),
                descriptor_getter: descriptor_getter
                    .as_ref()
                    .map(|ty| instance.instantiate_member(ty.clone())),
                descriptor_setter: descriptor_setter
                    .as_ref()
                    .map(|ty| instance.instantiate_member(ty.clone())),
                is_function_without_return_annotation: *is_function_without_return_annotation,
            }),
        }
    }

    /// Given a `__set__(self, instance, value)` function, gets the type of `value`.
    fn get_descriptor_setter_value(setter: &Type) -> Type {
        let mut values = Vec::new();
        setter.visit_toplevel_callable(|callable| match &callable.params {
            Params::List(params) => match params.items().get(2) {
                Some(Param::Pos(_, t, _) | Param::PosOnly(_, t, _)) => values.push(t.clone()),
                _ => {}
            },
            _ => {}
        });
        if values.is_empty() {
            Type::any_implicit()
        } else {
            unions(values)
        }
    }

    pub fn as_param(
        self,
        name: &Name,
        default: bool,
        kw_only: bool,
        converter_param: Option<Type>,
    ) -> Param {
        let ClassField(ClassFieldInner::Simple {
            ty,
            descriptor_setter,
            ..
        }) = self;
        let param_ty = if let Some(converter_param) = converter_param {
            converter_param
        } else if let Some(descriptor_setter) = descriptor_setter {
            Self::get_descriptor_setter_value(&descriptor_setter)
        } else {
            ty
        };
        let required = match default {
            true => Required::Optional(None),
            false => Required::Required,
        };
        if kw_only {
            Param::KwOnly(name.clone(), param_ty, required)
        } else {
            Param::Pos(name.clone(), param_ty, required)
        }
    }

    fn as_raw_special_method_type(&self, instance: &Instance) -> Option<Type> {
        match self.instantiate_for(instance).0 {
            ClassFieldInner::Simple { ty, .. } => match self.initialization() {
                ClassFieldInitialization::ClassBody(_) => Some(ty),
                ClassFieldInitialization::Method
                | ClassFieldInitialization::Uninitialized
                | ClassFieldInitialization::Magic => None,
            },
        }
    }

    fn as_special_method_type(&self, instance: &Instance) -> Option<Type> {
        self.as_raw_special_method_type(instance)
            .and_then(|ty| make_bound_method(instance.to_type(), ty).ok())
    }

    pub fn as_named_tuple_type(&self) -> Type {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty.clone(),
        }
    }

    pub fn as_named_tuple_requiredness(&self) -> Required {
        match &self.0 {
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::ClassBody(_),
                ..
            } => Required::Optional(None),
            ClassFieldInner::Simple {
                initialization:
                    ClassFieldInitialization::Method
                    | ClassFieldInitialization::Uninitialized
                    | ClassFieldInitialization::Magic,
                ..
            } => Required::Required,
        }
    }

    pub fn as_typed_dict_field_info(self, required_by_default: bool) -> Option<TypedDictField> {
        match &self.0 {
            ClassFieldInner::Simple {
                annotation:
                    Some(Annotation {
                        ty: Some(ty),
                        qualifiers,
                    }),
                ..
            } => Some(TypedDictField {
                ty: ty.clone(),
                read_only_reason: if qualifiers.contains(&Qualifier::ReadOnly) {
                    Some(ReadOnlyReason::ReadOnlyQualifier)
                } else {
                    None
                },
                required: if qualifiers.contains(&Qualifier::Required) {
                    true
                } else if qualifiers.contains(&Qualifier::NotRequired) {
                    false
                } else {
                    required_by_default
                },
            }),
            _ => None,
        }
    }

    pub fn as_enum_member(self, enum_cls: &Class) -> Option<Lit> {
        match self.0 {
            ClassFieldInner::Simple {
                ty: Type::Literal(lit),
                ..
            } if matches!(&lit, Lit::Enum(lit_enum) if lit_enum.class.class_object() == enum_cls) => {
                Some(lit)
            }
            _ => None,
        }
    }

    fn is_dataclass_kwonly_marker(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => {
                matches!(ty, Type::ClassType(cls) if cls.has_qname("dataclasses", "KW_ONLY"))
            }
        }
    }

    pub fn is_class_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_class_var())
            }
        }
    }

    pub fn is_init_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_init_var())
            }
        }
    }

    pub fn is_final(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, ty, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_final()) || ty.has_final_decoration()
            }
        }
    }

    pub fn read_only_reason(&self) -> &Option<ReadOnlyReason> {
        match &self.0 {
            ClassFieldInner::Simple {
                read_only_reason, ..
            } => read_only_reason,
        }
    }

    /// Check if this field is read-only for any reason.
    pub fn is_read_only(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple {
                read_only_reason, ..
            } => read_only_reason.is_some(),
        }
    }

    fn has_explicit_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, .. } => annotation.is_some(),
        }
    }

    pub fn is_function_without_return_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple {
                is_function_without_return_annotation,
                ..
            } => *is_function_without_return_annotation,
        }
    }

    fn dataclass_flags_of(&self) -> DataclassFieldKeywords {
        match &self.0 {
            ClassFieldInner::Simple { initialization, .. } => match initialization {
                ClassFieldInitialization::ClassBody(Some(field_flags)) => field_flags.clone(),
                ClassFieldInitialization::ClassBody(None) => {
                    let mut kws = DataclassFieldKeywords::new();
                    kws.default = true;
                    kws
                }
                ClassFieldInitialization::Method
                | ClassFieldInitialization::Uninitialized
                | ClassFieldInitialization::Magic => DataclassFieldKeywords::new(),
            },
        }
    }

    fn is_initialized_in_method(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { initialization, .. } => {
                matches!(initialization, ClassFieldInitialization::Method)
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum InstanceKind {
    ClassType,
    TypedDict,
    TypeVar(Quantified),
}

/// Wrapper to hold a specialized instance of a class , unifying ClassType and TypedDict.
struct Instance<'a> {
    kind: InstanceKind,
    class: &'a Class,
    targs: &'a TArgs,
}

impl<'a> Instance<'a> {
    fn of_class(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::ClassType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    fn of_typed_dict(td: &'a TypedDict) -> Self {
        Self {
            kind: InstanceKind::TypedDict,
            class: td.class_object(),
            targs: td.targs(),
        }
    }

    fn of_type_var(q: Quantified, bound: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::TypeVar(q),
            class: bound.class_object(),
            targs: bound.targs(),
        }
    }

    /// Instantiate a type that is relative to the class type parameters
    /// by substituting in the type arguments.
    fn instantiate_member(&self, raw_member: Type) -> Type {
        self.targs.substitute(raw_member)
    }

    fn to_type(&self) -> Type {
        match &self.kind {
            InstanceKind::ClassType => {
                ClassType::new(self.class.dupe(), self.targs.clone()).to_type()
            }
            InstanceKind::TypedDict => {
                Type::TypedDict(TypedDict::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::TypeVar(q) => Type::Quantified(q.clone()),
        }
    }
}

fn bind_class_attribute(
    cls: &Class,
    attr: Type,
    read_only_reason: &Option<ReadOnlyReason>,
) -> Attribute {
    let ty = make_bound_classmethod(cls, attr).into_inner();
    if let Some(reason) = read_only_reason {
        Attribute::read_only(ty, reason.clone())
    } else {
        Attribute::read_write(ty)
    }
}

/// Return the type of making it bound, or if not,
fn make_bound_method_helper(
    obj: Type,
    attr: Type,
    should_bind: &dyn Fn(&FuncMetadata) -> bool,
) -> Result<Type, Type> {
    let func = match attr {
        Type::Forall(box Forall {
            tparams,
            body: Forallable::Function(func),
        }) if should_bind(&func.metadata) => BoundMethodType::Forall(Forall {
            tparams,
            body: func,
        }),
        Type::Function(func) if should_bind(&func.metadata) => BoundMethodType::Function(*func),
        Type::Overload(overload) if should_bind(&overload.metadata) => {
            BoundMethodType::Overload(overload)
        }
        _ => return Err(attr),
    };
    Ok(Type::BoundMethod(Box::new(BoundMethod { obj, func })))
}

fn make_bound_classmethod(cls: &Class, attr: Type) -> Result<Type, Type> {
    let should_bind = |meta: &FuncMetadata| meta.flags.is_classmethod;
    make_bound_method_helper(Type::ClassDef(cls.dupe()), attr, &should_bind)
}

fn make_bound_method(obj: Type, attr: Type) -> Result<Type, Type> {
    let should_bind =
        |meta: &FuncMetadata| !meta.flags.is_staticmethod && !meta.flags.is_classmethod;
    make_bound_method_helper(obj, attr, &should_bind)
}

fn bind_instance_attribute(
    instance: &Instance,
    attr: Type,
    is_class_var: bool,
    read_only: Option<ReadOnlyReason>,
) -> Attribute {
    // Decorated objects are methods, so they can't be ClassVars
    if attr.is_property_getter() {
        Attribute::property(
            make_bound_method(instance.to_type(), attr).into_inner(),
            None,
            instance.class.dupe(),
        )
    } else if let Some(getter) = attr.is_property_setter_with_getter() {
        // The attribute lookup code and function decorator logic together ensure that
        // a property with a setter winds up being bound to the raw setter function
        // type, with function metadata that includes the raw getter function type.
        //
        // See the `attr.rs` and `function.rs` code for more details on how this works.
        Attribute::property(
            make_bound_method(instance.to_type(), getter).into_inner(),
            Some(make_bound_method(instance.to_type(), attr).into_inner()),
            instance.class.dupe(),
        )
    } else if is_class_var {
        Attribute::read_only(
            make_bound_method(instance.to_type(), attr).into_inner(),
            ReadOnlyReason::ClassVar,
        )
    } else if let Some(reason) = read_only {
        Attribute::read_only(
            make_bound_method(instance.to_type(), attr).into_inner(),
            reason,
        )
    } else {
        Attribute::read_write(
            make_bound_method(instance.to_type(), attr)
                .unwrap_or_else(|attr| make_bound_classmethod(instance.class, attr).into_inner()),
        )
    }
}

/// Result of looking up a member of a class in the MRO, including a handle to the defining
/// class which may be some ancestor.
///
/// For example, given `class A: x: int; class B(A): pass`, the defining class
/// for attribute `x` is `A` even when `x` is looked up on `B`.
#[derive(Debug)]
pub struct WithDefiningClass<T> {
    pub value: T,
    pub defining_class: Class,
}

impl<T> WithDefiningClass<T> {
    pub(in crate::alt::class) fn defined_on(&self, module: &str, cls: &str) -> bool {
        self.defining_class.has_qname(module, cls)
    }
}

/// The result of processing a raw dataclass member (any annotated assignment in its body).
pub enum DataclassMember {
    /// A dataclass field
    Field(WithDefiningClass<Arc<ClassField>>, DataclassFieldKeywords),
    /// A pseudo-field that only appears as a constructor argument
    InitVar(WithDefiningClass<Arc<ClassField>>, DataclassFieldKeywords),
    /// A pseudo-field annotated with KW_ONLY
    KwOnlyMarker,
    /// Anything else
    NotAField,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn check_enum_value_annotation(
        &self,
        mut value: &Type,
        annotation: &Type,
        member: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if matches!(value, Type::Tuple(_)) {
            // TODO: check tuple values against constructor signature
            // see https://typing.python.org/en/latest/spec/enums.html#member-values
            return;
        }
        if matches!(value, Type::ClassType(cls) if cls.has_qname("enum", "auto")) {
            return;
        }
        if let Type::ClassType(cls) = value
            && cls.has_qname("enum", "member")
            && let [member_targ] = cls.targs().as_slice()
        {
            value = member_targ;
        }
        if !self
            .solver()
            .is_subset_eq(value, annotation, self.type_order())
        {
            self.error(
                errors, range, ErrorInfo::Kind(ErrorKind::BadAssignment),
                format!(
                    "Enum member `{member}` has type `{}`, must match the `_value_` attribute annotation of `{}`",
                    self.for_display(value.clone()),
                    self.for_display(annotation.clone()),
                ),
            );
        }
    }

    pub fn calculate_class_field(
        &self,
        class: &Class,
        name: &Name,
        range: TextRange,
        field_definition: &ClassFieldDefinition,
        errors: &ErrorCollector,
    ) -> ClassField {
        // TODO(stroxler): Clean this up, as we convert more of the class field logic to using enums.
        //
        // It's a mess becasue we are relying on refs to fields that don't make sense for some cases,
        // which requires us having a place to store synthesized dummy values until we've refactored more.
        let value_storage = Owner::new();
        let initial_value_storage = Owner::new();
        let (value, direct_annotation, initial_value, is_function_without_return_annotation) =
            match field_definition {
                ClassFieldDefinition::DeclaredByAnnotation { annotation } => {
                    let annotation = self.get_idx(*annotation).as_ref().annotation.clone();
                    (
                        value_storage
                            .push(ExprOrBinding::Binding(Binding::Type(Type::any_implicit()))),
                        Some(annotation),
                        initial_value_storage.push(RawClassFieldInitialization::Uninitialized),
                        false,
                    )
                }
                ClassFieldDefinition::DeclaredWithoutAnnotation => (
                    value_storage.push(ExprOrBinding::Binding(Binding::Type(Type::any_implicit()))),
                    None,
                    initial_value_storage.push(RawClassFieldInitialization::Uninitialized),
                    false,
                ),
                ClassFieldDefinition::AssignedInBody { value, annotation } => {
                    let annotation = annotation
                        .map(|a| self.get_idx(a))
                        .as_deref()
                        .map(|annot| annot.annotation.clone());
                    (
                        value,
                        annotation,
                        initial_value_storage.push(RawClassFieldInitialization::ClassBody(
                            match value {
                                ExprOrBinding::Expr(e) => Some(e.clone()),
                                ExprOrBinding::Binding(_) => None,
                            },
                        )),
                        false,
                    )
                }
                ClassFieldDefinition::MethodLike {
                    definition,
                    has_return_annotation,
                } => (
                    value_storage.push(ExprOrBinding::Binding(Binding::Forward(*definition))),
                    None,
                    initial_value_storage.push(RawClassFieldInitialization::ClassBody(None)),
                    !has_return_annotation,
                ),
                ClassFieldDefinition::DefinedWithoutAssign { definition } => (
                    value_storage.push(ExprOrBinding::Binding(Binding::Forward(*definition))),
                    None,
                    initial_value_storage.push(RawClassFieldInitialization::ClassBody(None)),
                    false,
                ),
                ClassFieldDefinition::DefinedInMethod {
                    value,
                    annotation,
                    method,
                } => {
                    let annotation = annotation
                        .map(|a| self.get_idx(a))
                        .as_deref()
                        .map(|annot| annot.annotation.clone());
                    (
                        value,
                        annotation,
                        initial_value_storage
                            .push(RawClassFieldInitialization::Method(method.clone())),
                        false,
                    )
                }
            };

        // Optimisation. If we can determine that the name definitely doesn't exist in the inheritance
        // then we can avoid a bunch of work with checking for override errors.
        let mut name_might_exist_in_inherited = true;

        let (value_ty, inherited_annotation) = match value {
            ExprOrBinding::Expr(e) => {
                let inherited_annot = if direct_annotation.is_some() {
                    None
                } else {
                    let (found_field, annotation) = self.get_inherited_annotation(class, name);
                    if !found_field {
                        name_might_exist_in_inherited = false;
                    }
                    annotation
                };
                let mut ty = if let Some(annot) = &inherited_annot {
                    let ctx: &dyn Fn() -> TypeCheckContext =
                        &|| TypeCheckContext::of_kind(TypeCheckKind::Attribute(name.clone()));
                    let hint = Some((annot.get_type(), ctx));
                    self.expr(e, hint, errors)
                } else {
                    self.expr_infer(e, errors)
                };
                self.expand_type_mut(&mut ty);
                (ty, inherited_annot)
            }
            ExprOrBinding::Binding(b) => (
                Arc::unwrap_or_clone(self.solve_binding(b, errors)).into_ty(),
                None,
            ),
        };
        let metadata = self.get_metadata_for_class(class);
        let magically_initialized = {
            // We consider fields to be always-initialized if it's defined within stub files.
            // See https://github.com/python/typeshed/pull/13875 for reasoning.
            class.module_path().is_interface()
            // We consider fields to be always-initialized if it's annotated explicitly with `ClassVar`.
            || direct_annotation
                .as_ref()
                .is_some_and(|annot| annot.has_qualifier(&Qualifier::ClassVar))
        };
        let initialization =
            self.get_class_field_initialization(&metadata, initial_value, magically_initialized);

        // Ban typed dict from containing values; fields should be annotation-only.
        // TODO(stroxler): we ought to look into this more: class-level attributes make sense on a `TypedDict` class;
        // the typing spec does not explicitly define whether this is permitted.
        if metadata.is_typed_dict()
            && matches!(initialization, ClassFieldInitialization::ClassBody(_))
        {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                format!("TypedDict item `{name}` may not be initialized"),
            );
        }
        if metadata.is_typed_dict()
            || metadata
                .named_tuple_metadata()
                .is_some_and(|m| m.elements.contains(name))
        {
            for q in &[Qualifier::Final, Qualifier::ClassVar] {
                if direct_annotation
                    .as_ref()
                    .is_some_and(|ann| ann.has_qualifier(q))
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("`{q}` may not be used for TypedDict or NamedTuple members",),
                    );
                }
            }
        }
        if !metadata.is_typed_dict() {
            for q in &[
                Qualifier::Required,
                Qualifier::NotRequired,
                Qualifier::ReadOnly,
            ] {
                if direct_annotation
                    .as_ref()
                    .is_some_and(|ann| ann.has_qualifier(q))
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("`{q}` may only be used for TypedDict members"),
                    );
                }
            }
        }

        // Determine whether this is an explicit `@override`.
        let is_override = value_ty.is_override();

        let annotation = direct_annotation.as_ref().or(inherited_annotation.as_ref());
        let read_only_reason =
            self.determine_read_only_reason(class, name, annotation, &value_ty, &initialization);
        let is_namedtuple_member = metadata
            .named_tuple_metadata()
            .is_some_and(|nt| nt.elements.contains(name));

        // Promote literals. The check on `annotation` is an optimization, it does not (currently) affect semantics.
        let value_ty = if (read_only_reason.is_none() || is_namedtuple_member)
            && annotation.is_none_or(|a| a.ty.is_none())
            && value_ty.is_literal()
        {
            value_ty.promote_literals(self.stdlib)
        } else {
            value_ty
        };

        // Types provided in annotations shadow inferred types
        let ty = if let Some(ann) = annotation {
            match &ann.ty {
                Some(ty) => ty.clone(),
                None => value_ty,
            }
        } else {
            value_ty
        };

        let ty = match initial_value {
            RawClassFieldInitialization::ClassBody(_)
            | RawClassFieldInitialization::Uninitialized => ty,
            RawClassFieldInitialization::Method(method) => self
                .check_and_sanitize_method_scope_type_parameters(
                    class,
                    &method.method_name,
                    ty,
                    name,
                    range,
                    errors,
                ),
        };

        // Enum handling:
        // - Check whether the field is a member (which depends only on its type and name)
        // - Validate that a member should not have an annotation, and should respect any explicit annotation on `_value_`
        //
        // TODO(stroxler, yangdanny): We currently operate on promoted types, which means we do not infer `Literal[...]`
        // types for the `.value` / `._value_` attributes of literals. This is permitted in the spec although not optimal
        // for most cases; we are handling it this way in part because generic enum behavior is not yet well-specified.
        //
        // We currently skip the check for `_value_` if the class defines `__new__`, since that can
        // change the value of the enum member. https://docs.python.org/3/howto/enum.html#when-to-use-new-vs-init
        let ty = if let Some(enum_) = metadata.enum_metadata()
            && self.is_valid_enum_member(name, &ty, &initialization)
        {
            if direct_annotation.is_some() {
                self.error(
                    errors, range,ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("Enum member `{name}` may not be annotated directly. Instead, annotate the `_value_` attribute."),
                );
            }
            if enum_.has_value
                && let Some(enum_value_ty) = self.type_of_enum_value(enum_)
                && !class.fields().contains(&dunder::NEW)
                && (!matches!(value, ExprOrBinding::Expr(Expr::EllipsisLiteral(_)))
                    || !self.module().path().is_interface())
            {
                self.check_enum_value_annotation(&ty, &enum_value_ty, name, range, errors);
            }
            Type::Literal(Lit::Enum(Box::new(LitEnum {
                class: enum_.cls.clone(),
                member: name.clone(),
                ty: ty.clone(),
            })))
        } else {
            ty
        };

        // Identify whether this is a descriptor
        let (mut descriptor_getter, mut descriptor_setter) = (None, None);
        match &ty {
            // TODO(stroxler): This works for simple descriptors. There three known gaps, there may be others:
            // - If the field is instance-only, descriptor dispatching won't occur, an instance-only attribute
            //   that happens to be a descriptor just behaves like a normal instance-only attribute.
            // - Gracefully handle instance-only `__get__`/`__set__`. Descriptors only seem to be detected
            //   when the descriptor attribute is initialized on the class body of the descriptor.
            // - Do we care about distributing descriptor behavior over unions? If so, what about the case when
            //   the raw class field is a union of a descriptor and a non-descriptor? Do we want to allow this?
            Type::ClassType(c) => {
                if c.class_object().contains(&dunder::GET) {
                    descriptor_getter =
                        Some(self.attr_infer_for_type(&ty, &dunder::GET, range, errors, None));
                }
                if c.class_object().contains(&dunder::SET) {
                    descriptor_setter =
                        Some(self.attr_infer_for_type(&ty, &dunder::SET, range, errors, None));
                }
            }
            _ => {}
        };

        // Pin any vars in the type: leaking a var in a class field is particularly
        // likely to lead to data races where downstream uses can pin inconsistently.
        //
        // TODO(stroxler): Ideally we would implement some simple heuristics, similar to
        // first-use based inference we use with assignments, to get more useful types here.
        let ty = self.solver().deep_force(ty);

        // Create the resulting field and check for override inconsistencies before returning
        let class_field = ClassField::new(
            ty,
            direct_annotation,
            initialization,
            read_only_reason,
            descriptor_getter,
            descriptor_setter,
            is_function_without_return_annotation,
        );
        if name_might_exist_in_inherited || is_override {
            self.check_class_field_for_override_mismatch(
                name,
                &class_field,
                class,
                is_override,
                range,
                errors,
            );
        }
        if let RawClassFieldInitialization::Method(MethodThatSetsAttr {
            method_name,
            recognized_attribute_defining_method: false,
        }) = initial_value
        {
            let mut defined_in_parent = false;
            let parents = metadata.bases_with_metadata();
            for (parent, _) in parents {
                if self.get_class_member(parent.class_object(), name).is_some() {
                    defined_in_parent = true;
                    break;
                };
            }
            if !defined_in_parent {
                self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::ImplicitlyDefinedAttribute,
                ),
                format!("Attribute `{}` is implicitly defined by assignment in method `{method_name}`, which is not a constructor", &name),
            );
            }
        }
        if let Some(dm) = metadata.dataclass_metadata()
            && name == &dunder::POST_INIT
            && let Some(post_init) = class_field
                .as_special_method_type(&Instance::of_class(&self.as_class_type_unchecked(class)))
        {
            self.validate_post_init(class, dm, post_init, range, errors);
        }
        class_field
    }

    fn determine_read_only_reason(
        &self,
        cls: &Class,
        name: &Name,
        annotation: Option<&Annotation>,
        ty: &Type,
        initialization: &ClassFieldInitialization,
    ) -> Option<ReadOnlyReason> {
        if let Some(ann) = annotation {
            // TODO: enable this for Final attrs that aren't initialized on the class
            if ann.is_final() && matches!(initialization, ClassFieldInitialization::ClassBody(_)) {
                return Some(ReadOnlyReason::Final);
            }
            if ann.has_qualifier(&Qualifier::ReadOnly) {
                return Some(ReadOnlyReason::ReadOnlyQualifier);
            }
        }
        let metadata = self.get_metadata_for_class(cls);
        // NamedTuple members are read-only
        if metadata
            .named_tuple_metadata()
            .is_some_and(|nt| nt.elements.contains(name))
        {
            return Some(ReadOnlyReason::NamedTuple);
        }
        // Frozen dataclass fields (not methods) are read-only
        if let Some(dm) = metadata.dataclass_metadata()
            && dm.kws.frozen
            && dm.fields.contains(name)
        {
            return Some(ReadOnlyReason::FrozenDataclass);
        }
        // A `type[X]` that is initialized on the class is assumed to be ReadOnly. This
        // is partly because that's what nested class defs look like to Pyrefly. But even
        // for assignments of class objects, this implies covariance and is probably more
        // useful than allowing reassignment but forcing invariance.
        //
        // TODO(stroxler): We may need to revisit this if we find projects that require
        // read-write behavior. Covariance is known to be strictly necessary for many
        // projects, so the obvious alternative to read-only semantics is likely to
        // allow covariance unsoundly.
        let is_class_object_type = match ty {
            Type::ClassDef(..) => true,
            Type::Type(c) if matches!(**c, Type::ClassType(..)) => true,
            _ => false,
        };
        if is_class_object_type && matches!(initialization, ClassFieldInitialization::ClassBody(..))
        {
            return Some(ReadOnlyReason::ClassObjectInitializedOnBody);
        }
        // Default: the field is read-write
        None
    }

    /// Return (did you find any fields, first one with an annotation)
    fn get_inherited_annotation(&self, class: &Class, name: &Name) -> (bool, Option<Annotation>) {
        let mut found_field = false;
        let annotation = self
            .get_mro_for_class(class)
            .ancestors(self.stdlib)
            .find_map(|parent| {
                let parent_field =
                    self.get_field_from_current_class_only(parent.class_object(), name)?;
                found_field = true;
                let ClassField(ClassFieldInner::Simple { annotation, .. }) = &*parent_field;
                annotation.clone()
            });
        (found_field, annotation)
    }

    fn get_class_field_initialization(
        &self,
        metadata: &ClassMetadata,
        initial_value: &RawClassFieldInitialization,
        magically_initialized: bool,
    ) -> ClassFieldInitialization {
        match initial_value {
            RawClassFieldInitialization::ClassBody(None) => {
                ClassFieldInitialization::ClassBody(None)
            }
            RawClassFieldInitialization::ClassBody(Some(e)) => {
                // If this field was created via a call to a dataclass field specifier, extract field flags from the call.
                if let Some(dm) = metadata.dataclass_metadata()
                    && let Expr::Call(ExprCall {
                        node_index: _,
                        range: _,
                        func,
                        arguments,
                    }) = e
                {
                    // We already type-checked this expression as part of computing the type for the ClassField,
                    // so we can ignore any errors encountered here.
                    let ignore_errors = self.error_swallower();
                    let func_ty = self.expr_infer(func, &ignore_errors);
                    let func_kind = func_ty.callee_kind();
                    if let Some(func_kind) = func_kind
                        && dm.field_specifiers.contains(&func_kind)
                    {
                        let flags =
                            self.dataclass_field_keywords(&func_ty, arguments, &ignore_errors);
                        ClassFieldInitialization::ClassBody(Some(flags))
                    } else {
                        ClassFieldInitialization::ClassBody(None)
                    }
                } else {
                    ClassFieldInitialization::ClassBody(None)
                }
            }
            RawClassFieldInitialization::Method(_) | RawClassFieldInitialization::Uninitialized
                if magically_initialized =>
            {
                ClassFieldInitialization::Magic
            }
            RawClassFieldInitialization::Method(_) => ClassFieldInitialization::Method,
            RawClassFieldInitialization::Uninitialized => ClassFieldInitialization::Uninitialized,
        }
    }

    /// This is used for dataclass field synthesis; when accessing attributes on dataclass instances,
    /// use `get_instance_attribute` or `get_class_attribute`
    pub fn get_dataclass_member(&self, cls: &Class, name: &Name) -> DataclassMember {
        // Even though we check that the class member exists before calling this function,
        // it can be None if the class has an invalid MRO.
        let Some(member) = self.get_class_member_impl(cls, name) else {
            return DataclassMember::NotAField;
        };
        let field = &*member.value;
        // A field with type KW_ONLY is a sentinel value that indicates that the remaining
        // fields should be keyword-only params in the generated `__init__`.
        if field.is_dataclass_kwonly_marker() {
            DataclassMember::KwOnlyMarker
        } else if field.is_initialized_in_method() // This member is defined in a method without being declared on the class
            || field.is_class_var() // Class variables are not dataclass fields
            || (!field.has_explicit_annotation()
                && self
                    .get_inherited_annotation(cls, name)
                    .1
                    .is_some_and(|annot| annot.has_qualifier(&Qualifier::ClassVar)))
        {
            DataclassMember::NotAField
        } else {
            let flags = field.dataclass_flags_of();
            if field.is_init_var() {
                DataclassMember::InitVar(member, flags)
            } else {
                DataclassMember::Field(member, flags)
            }
        }
    }

    fn check_and_sanitize_method_scope_type_parameters(
        &self,
        class: &Class,
        method_name: &Name,
        ty: Type,
        name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if let Some(method_field) =
            self.get_non_synthesized_field_from_current_class_only(class, method_name)
            && !method_field.is_init_var()
        {
            match &method_field.raw_type() {
                Type::Forall(box Forall { tparams, .. }) => {
                    let mut qs = SmallSet::new();
                    ty.collect_quantifieds(&mut qs);
                    let ts = Owner::new();
                    let gradual_fallbacks: SmallMap<_, _> = tparams
                        .iter()
                        .filter_map(|param| {
                            let q = &param.quantified;
                            if qs.contains(q) {
                                self.error(
                                    errors,
                                    range,
                                    ErrorInfo::Kind(ErrorKind::InvalidTypeVar,
                            ),
                                format!(
                                        "Cannot initialize attribute `{}` to a value that depends on method-scoped type variable `{}`",
                                        name,
                                        param.name(),
                                    ),
                                );
                                Some((q , ts.push(q.as_gradual_type())))
                            } else {
                                None
                            }
                        })
                        .collect();
                    drop(qs);
                    ty.subst(&gradual_fallbacks)
                }
                _ => ty,
            }
        } else {
            ty
        }
    }

    fn as_instance_attribute(&self, field: &ClassField, instance: &Instance) -> Attribute {
        match field.instantiate_for(instance).0 {
            // TODO(stroxler): Clean up this match by making `ClassFieldInner` an
            // enum; the match is messy
            ClassFieldInner::Simple {
                ty,
                descriptor_getter,
                descriptor_setter,
                ..
            } if (descriptor_getter.is_some() || descriptor_setter.is_some())
                // There's no situation in which you can stick a usable descriptor in a TypedDict.
                // TODO(rechen): a descriptor in a TypedDict should be an error at class creation time.
                && instance.kind == InstanceKind::ClassType =>
            {
                Attribute::descriptor(
                    ty,
                    DescriptorBase::Instance(ClassType::new(
                        instance.class.dupe(),
                        instance.targs.clone(),
                    )),
                    descriptor_getter,
                    descriptor_setter,
                )
            }
            ClassFieldInner::Simple {
                mut ty,
                read_only_reason,
                annotation,
                ..
            } => {
                let is_class_var = annotation.is_some_and(|ann| ann.is_class_var());
                match field.initialization() {
                    ClassFieldInitialization::ClassBody(_) => {
                        self.expand_type_mut(&mut ty); // bind_instance matches on the type, so resolve it if we can
                        bind_instance_attribute(instance, ty, is_class_var, read_only_reason)
                    }
                    ClassFieldInitialization::Method
                    | ClassFieldInitialization::Uninitialized
                    | ClassFieldInitialization::Magic
                        if let Some(read_only_reason) = read_only_reason =>
                    {
                        Attribute::read_only(ty, read_only_reason)
                    }
                    ClassFieldInitialization::Method
                    | ClassFieldInitialization::Uninitialized
                    | ClassFieldInitialization::Magic
                        if is_class_var =>
                    {
                        Attribute::read_only(ty, ReadOnlyReason::ClassVar)
                    }
                    ClassFieldInitialization::Method
                    | ClassFieldInitialization::Uninitialized
                    | ClassFieldInitialization::Magic => Attribute::read_write(ty),
                }
            }
        }
    }

    fn as_class_attribute(&self, field: ClassField, cls: &Class) -> Attribute {
        match &field.0 {
            ClassFieldInner::Simple {
                ty,
                descriptor_getter,
                descriptor_setter,
                ..
            } if descriptor_getter.is_some() || descriptor_setter.is_some() => {
                Attribute::descriptor(
                    ty.clone(),
                    DescriptorBase::ClassDef(cls.dupe()),
                    descriptor_getter.clone(),
                    descriptor_setter.clone(),
                )
            }
            ClassFieldInner::Simple {
                initialization:
                    ClassFieldInitialization::Method | ClassFieldInitialization::Uninitialized,
                ..
            } => Attribute::no_access(NoAccessReason::ClassUseOfInstanceAttribute(cls.dupe())),
            ClassFieldInner::Simple { ty, .. } => {
                if self.depends_on_class_type_parameter(&field, cls) {
                    self.get_function_depending_on_class_type_parameter(cls, ty)
                        .unwrap_or_else(|| {
                            Attribute::no_access(NoAccessReason::ClassAttributeIsGeneric(
                                cls.dupe(),
                            ))
                        })
                } else {
                    bind_class_attribute(cls, ty.clone(), field.read_only_reason())
                }
            }
        }
    }

    fn depends_on_class_type_parameter(&self, field: &ClassField, cls: &Class) -> bool {
        let tparams = self.get_class_tparams(cls);
        let mut qs = SmallSet::new();
        match &field.0 {
            ClassFieldInner::Simple { ty, .. } => ty.collect_quantifieds(&mut qs),
        };
        tparams.quantifieds().any(|q| qs.contains(q))
    }

    fn get_function_depending_on_class_type_parameter(
        &self,
        cls: &Class,
        ty: &Type,
    ) -> Option<Attribute> {
        let mut foralled = match ty {
            Type::Function(func) => Type::Forall(Box::new(Forall {
                tparams: self.get_class_tparams(cls),
                body: Forallable::Function((**func).clone()),
            })),
            Type::Forall(box Forall {
                tparams,
                body: body @ Forallable::Function(_),
            }) => {
                let mut new_tparams = tparams.as_ref().clone();
                new_tparams.extend(&self.get_class_tparams(cls));
                Type::Forall(Box::new(Forall {
                    tparams: Arc::new(new_tparams),
                    body: body.clone(),
                }))
            }
            Type::Overload(Overload {
                signatures,
                metadata,
            }) => {
                let new_signatures = signatures.clone().mapped(|sig| match sig {
                    OverloadType::Callable(function) => OverloadType::Forall(Forall {
                        tparams: self.get_class_tparams(cls),
                        body: Function {
                            signature: function.signature,
                            metadata: (**metadata).clone(),
                        },
                    }),
                    OverloadType::Forall(Forall { tparams, body }) => {
                        let mut new_tparams = tparams.as_ref().clone();
                        new_tparams.extend(&self.get_class_tparams(cls));
                        OverloadType::Forall(Forall {
                            tparams: Arc::new(new_tparams),
                            body,
                        })
                    }
                });
                Type::Overload(Overload {
                    signatures: new_signatures,
                    metadata: metadata.clone(),
                })
            }
            _ => {
                return None;
            }
        };
        foralled.subst_self_type_mut(&self.instantiate(cls), &|a, b| self.is_subset_eq(a, b));
        Some(bind_class_attribute(cls, foralled, &None))
    }

    fn check_class_field_for_override_mismatch(
        &self,
        name: &Name,
        class_field: &ClassField,
        class: &Class,
        is_override: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let mut got_attr = None;
        let metadata = self.get_metadata_for_class(class);
        let parents = metadata.bases_with_metadata();
        let mut parent_attr_found = false;
        let mut parent_has_any = false;

        // TODO(zeina): skip private properties and dunder methods for now. This will need some special casing.
        if (name.starts_with('_') && name.ends_with('_'))
            || (name.starts_with("__") && !name.ends_with("__"))
        {
            return;
        }

        for (parent, parent_metadata) in parents {
            parent_has_any = parent_has_any || parent_metadata.has_base_any();
            // Don't allow overriding a namedtuple element
            if let Some(named_tuple_metadata) = parent_metadata.named_tuple_metadata()
                && named_tuple_metadata.elements.contains(name)
            {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!("Cannot override named tuple element `{name}`"),
                );
            }
            let Some(want_member) = self.get_class_member(parent.class_object(), name) else {
                continue;
            };
            parent_attr_found = true;
            let want_class_field = Arc::unwrap_or_clone(want_member.value);
            if want_class_field.is_final() {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!(
                        "`{}` is declared as final in parent class `{}`",
                        name,
                        parent.name()
                    ),
                );
                continue;
            }
            if want_class_field.has_explicit_annotation() && class_field.has_explicit_annotation() {
                let want_is_class_var = want_class_field.is_class_var();
                let got_is_class_var = class_field.is_class_var();
                if want_is_class_var && !got_is_class_var {
                    self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadOverride),
                            format!(
                                "Instance variable `{}.{}` overrides ClassVar of the same name in parent class `{}`",
                                class.name(),
                                name,
                                parent.name()
                            ),
                        );
                    continue;
                } else if !want_is_class_var && got_is_class_var {
                    self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadOverride),
                            format!(
                                "ClassVar `{}.{}` overrides instance variable of the same name in parent class `{}`",
                                class.name(),
                                name,
                                parent.name()
                            ),
                        );
                    continue;
                }
            }
            let want_attr =
                self.as_instance_attribute(&want_class_field, &Instance::of_class(parent));
            if got_attr.is_none() {
                // Optimisation: Only compute the `got_attr` once, and only if we actually need it.
                got_attr = Some(self.as_instance_attribute(
                    class_field,
                    &Instance::of_class(&self.as_class_type_unchecked(class)),
                ));
            }
            let attr_check = self.is_attribute_subset(
                got_attr.as_ref().unwrap(),
                &want_attr,
                &mut |got, want| self.is_subset_eq(got, want),
            );
            if let Err(error) = attr_check {
                let msg = vec1![
                    format!(
                        "Class member `{}.{}` overrides parent class `{}` in an inconsistent manner",
                        class.name(),
                        name,
                        parent.name()
                    ),
                    error.to_error_msg(class.name(), parent.name(), name)
                ];
                errors.add(range, ErrorInfo::Kind(ErrorKind::BadOverride), msg);
            }
        }
        if is_override && !parent_attr_found && !parent_has_any {
            self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!(
                        "Class member `{}.{}` is marked as an override, but no parent class has a matching attribute",
                        class.name(),
                        name,
                    ),
                );
        }
    }

    fn get_non_synthesized_field_from_current_class_only(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        if cls.contains(name)
            && let Some(field) = self.get_from_class(cls, &KeyClassField(cls.index(), name.clone()))
        {
            Some(field)
        } else {
            None
        }
    }

    /// Only look up fields that are not synthesized. This is useful when synthesizing method signatures
    /// for typeddict, named tuple, etc.
    pub fn get_non_synthesized_class_member(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        self.get_non_synthesized_field_from_current_class_only(cls, name)
            .filter(|field| !field.is_init_var())
            .or_else(|| {
                self.get_mro_for_class(cls)
                    .ancestors(self.stdlib)
                    .find_map(|ancestor| {
                        self.get_non_synthesized_field_from_current_class_only(
                            ancestor.class_object(),
                            name,
                        )
                        .filter(|field| !field.is_init_var())
                    })
            })
    }

    fn get_synthesized_field_from_current_class_only(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        Some(
            self.get_from_class(cls, &KeyClassSynthesizedFields(cls.index()))?
                .get(name)?
                .inner
                .dupe(),
        )
    }

    /// This function does not return fields defined in parent classes
    pub fn get_field_from_current_class_only(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        self.get_non_synthesized_field_from_current_class_only(cls, name)
            .or_else(|| self.get_synthesized_field_from_current_class_only(cls, name))
    }

    fn get_class_member_impl(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        if let Some(field) = self.get_field_from_current_class_only(cls, name) {
            Some(WithDefiningClass {
                value: field,
                defining_class: cls.dupe(),
            })
        } else {
            self.get_mro_for_class(cls)
                .ancestors(self.stdlib)
                .find_map(|ancestor| {
                    self.get_field_from_current_class_only(ancestor.class_object(), name)
                        .map(|field| WithDefiningClass {
                            value: Arc::new(field.instantiate_for(&Instance::of_class(ancestor))),
                            defining_class: ancestor.class_object().dupe(),
                        })
                })
        }
    }

    pub(in crate::alt::class) fn get_class_member(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        if let Some(member) = self.get_class_member_impl(cls, name)
            && !member.value.is_init_var()
        {
            Some(member)
        } else {
            None
        }
    }

    pub fn get_instance_attribute(&self, cls: &ClassType, name: &Name) -> Option<Attribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|member| self.as_instance_attribute(&member.value, &Instance::of_class(cls)))
    }

    pub fn get_bounded_type_var_attribute(
        &self,
        type_var: Quantified,
        upper_bound: &ClassType,
        name: &Name,
    ) -> Option<Attribute> {
        self.get_class_member(upper_bound.class_object(), name)
            .map(|member| {
                self.as_instance_attribute(
                    &member.value,
                    &Instance::of_type_var(type_var, upper_bound),
                )
            })
    }

    pub fn get_typed_dict_attribute(&self, td: &TypedDict, name: &Name) -> Option<Attribute> {
        if let Some(meta) = self
            .get_metadata_for_class(td.class_object())
            .typed_dict_metadata()
            && meta.fields.contains_key(name)
        {
            // TypedDict fields are dictionary key declarations, not real fields.
            return None;
        }
        self.get_class_member(td.class_object(), name)
            .map(|member| self.as_instance_attribute(&member.value, &Instance::of_typed_dict(td)))
    }

    fn get_super_class_member(
        &self,
        cls: &Class,
        start_lookup_cls: &ClassType,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        // Skip ancestors in the MRO until we find the class we want to start at
        let metadata = self.get_mro_for_class(cls);
        let ancestors = metadata
            .ancestors(self.stdlib)
            .skip_while(|ancestor| *ancestor != start_lookup_cls);
        for ancestor in ancestors {
            if let Some(field) =
                self.get_field_from_current_class_only(ancestor.class_object(), name)
                && !field.is_init_var()
            {
                return Some(WithDefiningClass {
                    value: Arc::new(field.instantiate_for(&Instance::of_class(ancestor))),
                    defining_class: ancestor.class_object().dupe(),
                });
            }
        }
        None
    }

    /// Looks up an attribute on a super instance.
    pub fn get_super_attribute(
        &self,
        start_lookup_cls: &ClassType,
        super_obj: &SuperObj,
        name: &Name,
    ) -> Option<Attribute> {
        match super_obj {
            SuperObj::Instance(obj) => self
                .get_super_class_member(obj.class_object(), start_lookup_cls, name)
                .map(|member| self.as_instance_attribute(&member.value, &Instance::of_class(obj))),
            SuperObj::Class(obj) => self
                .get_super_class_member(obj, start_lookup_cls, name)
                .map(|member| self.as_class_attribute(Arc::unwrap_or_clone(member.value), obj)),
        }
    }

    /// Gets an attribute from a class definition.
    ///
    /// Returns `None` if there is no such attribute, otherwise an `Attribute` object
    /// that describes whether access is allowed and the type if so.
    ///
    /// Access is disallowed for instance-only attributes and for attributes whose
    /// type contains a class-scoped type parameter - e.g., `class A[T]: x: T`.
    pub fn get_class_attribute(&self, cls: &Class, name: &Name) -> Option<Attribute> {
        self.get_class_member(cls, name)
            .map(|member| self.as_class_attribute(Arc::unwrap_or_clone(member.value), cls))
    }

    pub fn field_is_inherited_from_object(&self, cls: &Class, name: &Name) -> bool {
        let member = self.get_class_member(cls, name);
        match member {
            Some(member) => member.defined_on("builtins", "object"),
            None => false,
        }
    }

    pub fn field_is_inherited_from_enum(&self, cls: &Class, name: &Name) -> bool {
        let member = self.get_class_member(cls, name);
        match member {
            Some(member) => member.defined_on("enum", "Enum"),
            None => false,
        }
    }

    /// Get the class's `__new__` method.
    ///
    /// This lookup skips normal method binding logic (it behaves like a cross
    /// between a classmethod and a constructor; downstream code handles this
    /// using the raw callable type).
    pub fn get_dunder_new(&self, cls: &ClassType) -> Option<Type> {
        let new_member = self.get_class_member(cls.class_object(), &dunder::NEW)?;
        if new_member.defined_on("builtins", "object") {
            // The default behavior of `object.__new__` is already baked into our implementation of
            // class construction; we only care about `__new__` if it is overridden.
            None
        } else {
            Arc::unwrap_or_clone(new_member.value)
                .as_raw_special_method_type(&Instance::of_class(cls))
        }
    }

    fn get_dunder_init_helper(&self, instance: &Instance, get_object_init: bool) -> Option<Type> {
        let init_method = self.get_class_member(instance.class, &dunder::INIT)?;
        if get_object_init || !init_method.defined_on("builtins", "object") {
            Arc::unwrap_or_clone(init_method.value).as_special_method_type(instance)
        } else {
            None
        }
    }

    /// Get the class's `__init__` method. The second argument controls whether we return an inherited `object.__init__`.
    pub fn get_dunder_init(&self, cls: &ClassType, get_object_init: bool) -> Option<Type> {
        self.get_dunder_init_helper(&Instance::of_class(cls), get_object_init)
    }

    pub fn get_typed_dict_dunder_init(&self, td: &TypedDict) -> Option<Type> {
        self.get_dunder_init_helper(&Instance::of_typed_dict(td), true)
    }

    /// Get the metaclass `__call__` method
    pub fn get_metaclass_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        let metadata = self.get_metadata_for_class(cls.class_object());
        let metaclass = metadata.metaclass()?;
        let attr = self.get_class_member(metaclass.class_object(), &dunder::CALL)?;
        if attr.defined_on("builtins", "type") {
            // The behavior of `type.__call__` is already baked into our implementation of constructors,
            // so we can skip analyzing it at the type level.
            None
        } else if attr.value.is_function_without_return_annotation() {
            // According to the typing spec:
            // If a custom metaclass __call__ method is present but does not have an annotated return type,
            // type checkers may assume that the method acts like type.__call__.
            // https://typing.python.org/en/latest/spec/constructors.html#converting-a-constructor-to-callable
            None
        } else {
            Arc::unwrap_or_clone(attr.value)
                .as_raw_special_method_type(&Instance::of_class(metaclass))
                .and_then(|ty| make_bound_method(Type::type_form(cls.clone().to_type()), ty).ok())
        }
    }
}
