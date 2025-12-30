/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::iter;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::FuncFlags;
use pyrefly_types::callable::FuncId;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::read_only::IsFinalVariableInitialized;
use pyrefly_types::simplify::unions;
use pyrefly_types::type_var::PreInferenceVariance;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDictInner;
use pyrefly_types::types::TParam;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Union;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::ResultExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::helpers::is_dunder;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::AttrSubsetError;
use crate::alt::attr::ClassBase;
use crate::alt::attr::NoAccessReason;
use crate::alt::callable::CallArg;
use crate::alt::expr::TypeOrExpr;
use crate::alt::types::class_bases::ClassBases;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::binding::binding::Binding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::MethodSelfKind;
use crate::binding::binding::MethodThatSetsAttr;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::solver::solver::SubsetError;
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
use crate::types::quantified::Quantified;
use crate::types::read_only::ReadOnlyReason;
use crate::types::stdlib::Stdlib;
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

/// The result of looking up an attribute access on a class (either as an instance or a
/// class access, and possibly through a special case lookup such as a type var with a bound).
#[derive(Debug, Clone)]
pub enum ClassAttribute {
    /// A read-write attribute with a closed form type for both get and set actions.
    ReadWrite(Type),
    /// A read-only attribute with a closed form type for get actions.
    ReadOnly(Type, ReadOnlyReason),
    /// A `NoAccess` attribute indicates that the attribute is well-defined, but does
    /// not allow the access pattern (for example class access on an instance-only attribute)
    NoAccess(NoAccessReason),
    /// A property is a special attribute were regular access invokes a getter.
    /// It optionally might have a setter method; if not, trying to set it is an access error
    Property(Type, Option<Type>, Class),
    /// A descriptor is a user-defined type whose actions may dispatch to special method calls
    /// for the get and set actions.
    Descriptor(Descriptor, DescriptorBase),
}

impl ClassAttribute {
    pub fn read_write(ty: Type) -> Self {
        Self::ReadWrite(ty)
    }

    pub fn read_only(ty: Type, reason: ReadOnlyReason) -> Self {
        Self::ReadOnly(ty, reason)
    }

    pub fn no_access(reason: NoAccessReason) -> Self {
        Self::NoAccess(reason)
    }

    pub fn property(getter: Type, setter: Option<Type>, cls: Class) -> Self {
        Self::Property(getter, setter, cls)
    }

    pub fn descriptor(descriptor: Descriptor, base: DescriptorBase) -> Self {
        Self::Descriptor(descriptor, base)
    }

    pub fn read_only_equivalent(self, reason: ReadOnlyReason) -> Self {
        match self {
            Self::ReadWrite(ty) => Self::ReadOnly(ty, reason),
            Self::Property(getter, _, cls) => Self::Property(getter, None, cls),
            Self::Descriptor(
                Descriptor {
                    range, cls, getter, ..
                },
                base,
            ) => Self::Descriptor(
                Descriptor {
                    range,
                    cls,
                    getter,
                    setter: false,
                },
                base,
            ),
            attr @ (Self::NoAccess(..) | Self::ReadOnly(..)) => attr,
        }
    }

    /// Given a `ClassAttribute`, try to unwrap it as a method type, assuming
    /// that methods are always simple read-only or read-write attributes.
    ///
    /// If we encounter any other case, return `None`.
    pub fn as_instance_method(self) -> Option<Type> {
        match self {
            // TODO(stroxler): ReadWrite attributes are not actually methods but limiting access to
            // ReadOnly breaks unit tests; we should investigate callsites to understand this better.
            ClassAttribute::ReadWrite(ty) | ClassAttribute::ReadOnly(ty, _) => Some(ty),
            ClassAttribute::NoAccess(..)
            | ClassAttribute::Property(..)
            | ClassAttribute::Descriptor(..) => None,
        }
    }

    pub fn is_read_only(&self) -> bool {
        match self {
            ClassAttribute::ReadOnly(_, _)
            | ClassAttribute::Property(_, None, _)
            | ClassAttribute::Descriptor(Descriptor { setter: false, .. }, _) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
pub struct Descriptor {
    /// The location of the property where the descriptor is bound, where we should raise
    /// errors attempting to access the getter/setter.
    range: TextRange,
    /// This is the descriptor class, which is needed both for attribute subtyping
    /// checks in structural types and in the case where there is no getter method.
    cls: ClassType,
    /// Does `__get__` exists on the descriptor?  It is typically a `BoundMethod` although
    /// it is possible for a user to erroneously define a `__get__` with any type, including a
    /// non-callable one.
    getter: bool,
    /// Does `__set__` exists on the descriptor? Similar considerations to `getter` apply.
    setter: bool,
}

#[derive(Clone, Debug)]
pub enum DescriptorBase {
    Instance(ClassType),
    ClassDef(Class),
}

/// Correctly analyzing which attributes are visible on class objects, as well
/// as handling method binding correctly, requires distinguishing which fields
/// are assigned values in the class body.
#[derive(Clone, Debug, TypeEq, VisitMut, PartialEq, Eq)]
pub enum ClassFieldInitialization {
    /// If this is a dataclass field, DataclassFieldKeywords stores the field's
    /// dataclass flags (which are options that control how fields behave).
    ClassBody(Option<Box<DataclassFieldKeywords>>),
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
pub struct ClassField(ClassFieldInner, IsInherited);

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
enum ClassFieldInner {
    /// Properties discovered via @property decorator.
    /// Read-onlyness is handled by presence of and type of setter.
    #[allow(dead_code)]
    Property { ty: Type, is_abstract: bool },
    /// Descriptors: attributes initialized in the class body whose type has __get__/__set__ methods.
    /// Read-onlyness is handled by descriptor protocol calls.
    #[allow(dead_code)]
    Descriptor {
        ty: Type,
        annotation: Option<Annotation>,
        descriptor: Descriptor,
    },
    /// Methods (including abstract methods, functions without return annotations). We always
    /// treat them as read only.
    ///
    /// Callable types are only methods if some form of method binding applies; staticmethods
    /// or Callables that we decide not to model as descriptors become ClassAttributes.
    #[allow(dead_code)]
    Method {
        ty: Type,
        is_abstract: bool,
        is_function_without_return_annotation: bool,
    },
    /// Nested class definitions (class statements inside class body).
    /// These are always of type `Type::ClassDef`, and we treat them as read-only.
    #[allow(dead_code)]
    NestedClass { ty: Type },
    /// Class attributes (includes staticmethods, Django fields, regular attrs).
    /// These may also be shadowed on instances, unless they are marked as ClassVar.
    ///
    /// To minimize false positives, we treat attributes annotated but not initialized on the
    /// class body as class attributes even though in many cases they will not be defined on
    /// the class; `initialization` tracks information about whether we are sure that access
    /// should succeed.
    #[allow(dead_code)]
    ClassAttribute {
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        read_only_reason: Option<ReadOnlyReason>,
        /// ClassVar: can read from instance, but cannot write/shadow from instance
        is_classvar: bool,
        is_staticmethod: bool,
        /// Django ForeignKey - triggers synthesis of _id field
        is_foreign_key: bool,
        /// Django field with choices - triggers synthesis of get_FOO_display method
        has_choices: bool,
    },
    /// Instance-only attributes (defined in methods, not in class body).
    #[allow(dead_code)]
    InstanceAttribute {
        ty: Type,
        annotation: Option<Annotation>,
        read_only_reason: Option<ReadOnlyReason>,
    },
}

/// For efficiency, keep track of whether we know from `calculate_class_field`
/// that this is not an inherited field so that we can skip override consistency
/// checks. This information is not needed to understand the class field, it is
/// only used for efficiency.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
pub enum IsInherited {
    No,
    Maybe,
}

impl Display for ClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => write!(f, "{ty} (property)"),
            ClassFieldInner::Descriptor { ty, .. } => write!(f, "{ty} (descriptor)"),
            ClassFieldInner::Method { ty, .. } => write!(f, "{ty} (method)"),
            ClassFieldInner::NestedClass { ty, .. } => write!(f, "{ty} (nested class)"),
            ClassFieldInner::ClassAttribute {
                ty, initialization, ..
            } => write!(f, "{ty} ({initialization})"),
            ClassFieldInner::InstanceAttribute { ty, .. } => {
                write!(f, "{ty} (instance attribute)")
            }
        }
    }
}

impl ClassField {
    fn new(
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        read_only_reason: Option<ReadOnlyReason>,
        is_foreign_key: bool,
        has_choices: bool,
        is_inherited: IsInherited,
    ) -> Self {
        Self(
            ClassFieldInner::ClassAttribute {
                ty,
                annotation,
                initialization,
                read_only_reason,
                is_classvar: false,
                is_staticmethod: false,
                is_foreign_key,
                has_choices,
            },
            is_inherited,
        )
    }

    pub fn invalid_typed_dict_field() -> Self {
        ClassField::new(
            Type::any_error(),
            None,
            ClassFieldInitialization::Magic,
            None,
            false,
            false,
            IsInherited::Maybe,
        )
    }

    pub fn typed_dict_field(
        ty: Type,
        annotation: Annotation,
        read_only_reason: Option<ReadOnlyReason>,
    ) -> Self {
        Self::new(
            ty,
            Some(annotation),
            ClassFieldInitialization::Uninitialized,
            read_only_reason,
            false,
            false,
            IsInherited::Maybe,
        )
    }

    pub fn for_variance_inference(&self) -> Option<(&Type, Option<&Annotation>, bool)> {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => {
                // Properties don't have annotations (defined by decorator)
                Some((ty, None, self.is_read_only()))
            }
            ClassFieldInner::Descriptor { ty, annotation, .. } => {
                // Descriptors may have annotations
                Some((ty, annotation.as_ref(), self.is_read_only()))
            }
            ClassFieldInner::Method { ty, .. } => {
                // Methods don't have annotations and are always read-only
                Some((ty, None, self.is_read_only()))
            }
            ClassFieldInner::NestedClass { ty, .. } => Some((ty, None, self.is_read_only())),
            ClassFieldInner::ClassAttribute { ty, annotation, .. } => {
                Some((ty, annotation.as_ref(), self.is_read_only()))
            }
            ClassFieldInner::InstanceAttribute { ty, annotation, .. } => {
                Some((ty, annotation.as_ref(), self.is_read_only()))
            }
        }
    }

    pub fn new_synthesized(ty: Type) -> Self {
        // Detect if this is a property and construct the appropriate variant
        if ty.is_property_getter() || ty.is_property_setter_with_getter().is_some() {
            ClassField(
                ClassFieldInner::Property {
                    ty,
                    is_abstract: false,
                },
                IsInherited::Maybe,
            )
        } else if is_method_type(&ty) {
            // Synthesized methods (like __init__, __eq__, __iter__, etc.)
            ClassField(
                ClassFieldInner::Method {
                    ty,
                    is_abstract: false,
                    is_function_without_return_annotation: false,
                },
                IsInherited::Maybe,
            )
        } else {
            ClassField(
                ClassFieldInner::ClassAttribute {
                    ty,
                    annotation: None,
                    initialization: ClassFieldInitialization::ClassBody(None),
                    read_only_reason: None,
                    is_classvar: false,
                    is_staticmethod: false,
                    is_foreign_key: false,
                    has_choices: false,
                },
                IsInherited::Maybe,
            )
        }
    }

    pub fn recursive() -> Self {
        Self(
            ClassFieldInner::ClassAttribute {
                ty: Type::any_implicit(),
                annotation: None,
                initialization: ClassFieldInitialization::recursive(),
                read_only_reason: None,
                is_classvar: false,
                is_staticmethod: false,
                is_foreign_key: false,
                has_choices: false,
            },
            IsInherited::Maybe,
        )
    }

    fn initialization(&self) -> ClassFieldInitialization {
        match &self.0 {
            ClassFieldInner::Property { .. } => ClassFieldInitialization::ClassBody(None),
            ClassFieldInner::Descriptor { .. } => ClassFieldInitialization::ClassBody(None),
            ClassFieldInner::Method { .. } => ClassFieldInitialization::ClassBody(None),
            ClassFieldInner::NestedClass { .. } => ClassFieldInitialization::ClassBody(None),
            ClassFieldInner::ClassAttribute { initialization, .. } => initialization.clone(),
            ClassFieldInner::InstanceAttribute { .. } => ClassFieldInitialization::Method,
        }
    }

    fn instantiate_helper(&self, f: &mut dyn FnMut(&mut Type)) -> Self {
        match &self.0 {
            ClassFieldInner::Property { ty, is_abstract } => {
                let mut ty = ty.clone();
                f(&mut ty);
                Self(
                    ClassFieldInner::Property {
                        ty,
                        is_abstract: *is_abstract,
                    },
                    self.1.clone(),
                )
            }
            ClassFieldInner::Descriptor {
                ty,
                annotation,
                descriptor,
            } => {
                let mut ty = ty.clone();
                f(&mut ty);
                let mut descriptor = descriptor.clone();
                descriptor.cls.visit_mut(f);
                Self(
                    ClassFieldInner::Descriptor {
                        ty,
                        annotation: annotation.clone(),
                        descriptor,
                    },
                    self.1.clone(),
                )
            }
            ClassFieldInner::Method {
                ty,
                is_abstract,
                is_function_without_return_annotation,
            } => {
                let mut ty = ty.clone();
                f(&mut ty);
                Self(
                    ClassFieldInner::Method {
                        ty,
                        is_abstract: *is_abstract,
                        is_function_without_return_annotation:
                            *is_function_without_return_annotation,
                    },
                    self.1.clone(),
                )
            }
            ClassFieldInner::NestedClass { ty } => {
                let mut ty = ty.clone();
                f(&mut ty);
                Self(ClassFieldInner::NestedClass { ty }, self.1.clone())
            }
            ClassFieldInner::ClassAttribute {
                ty,
                annotation,
                initialization,
                read_only_reason,
                is_classvar,
                is_staticmethod,
                is_foreign_key,
                has_choices,
            } => {
                let mut ty = ty.clone();
                f(&mut ty);
                Self(
                    ClassFieldInner::ClassAttribute {
                        ty,
                        annotation: annotation.clone(),
                        initialization: initialization.clone(),
                        read_only_reason: read_only_reason.clone(),
                        is_classvar: *is_classvar,
                        is_staticmethod: *is_staticmethod,
                        is_foreign_key: *is_foreign_key,
                        has_choices: *has_choices,
                    },
                    self.1.clone(),
                )
            }
            ClassFieldInner::InstanceAttribute {
                ty,
                annotation,
                read_only_reason,
            } => {
                let mut ty = ty.clone();
                f(&mut ty);
                Self(
                    ClassFieldInner::InstanceAttribute {
                        ty,
                        annotation: annotation.clone(),
                        read_only_reason: read_only_reason.clone(),
                    },
                    self.1.clone(),
                )
            }
        }
    }

    fn instantiate_for(&self, instance: &Instance) -> Self {
        self.instantiate_helper(&mut |ty| {
            ty.subst_self_type_mut(&instance.to_type());
            instance.instantiate_member(ty)
        })
    }

    fn instantiate_for_class_targs(
        &self,
        targs: &TArgs,
        self_type: Type,
        ambiguous: &mut bool,
    ) -> Self {
        let mp = targs.substitution_map();
        self.instantiate_helper(&mut |ty| {
            ty.subst_self_type_mut(&self_type);
            match ty {
                Type::Function(_)
                | Type::Overload(_)
                | Type::Forall(box Forall {
                    body: Forallable::Function(_),
                    ..
                }) => ty.subst_mut_fn(&mut |q| mp.get(q).map(|ty| (*ty).clone())),
                _ => {
                    let mut qs: SmallSet<&Quantified> = SmallSet::new();
                    ty.collect_quantifieds(&mut qs);
                    *ambiguous = targs.tparams().iter().any(|x| qs.contains(&x.quantified));
                }
            }
        })
    }

    fn instantiate_for_class_tparams(
        &self,
        cls_tparams: Arc<TParams>,
        self_type: Type,
        ambiguous: &mut bool,
    ) -> Self {
        let prepend_class_tparams_if_used = |f: &Function, tparams_opt: Option<&TParams>| {
            if cls_tparams.is_empty() {
                return None;
            }
            let mut qs = SmallSet::new();
            f.visit(&mut |ty| ty.collect_quantifieds(&mut qs));
            if cls_tparams.iter().any(|tp| qs.contains(&tp.quantified)) {
                match tparams_opt {
                    None => Some(cls_tparams.dupe()),
                    Some(tparams) => {
                        let mut new_tparams = (*cls_tparams).clone();
                        new_tparams.extend(tparams);
                        Some(Arc::new(new_tparams))
                    }
                }
            } else {
                None
            }
        };
        self.instantiate_helper(&mut |ty| {
            ty.subst_self_type_mut(&self_type);
            match ty {
                Type::Function(func) => {
                    if let Some(tparams) = prepend_class_tparams_if_used(func, None) {
                        *ty = Type::Forall(Box::new(Forall {
                            tparams,
                            body: Forallable::Function((**func).clone()),
                        }));
                    }
                }
                Type::Forall(forall) => {
                    let Forall { tparams, body } = &mut **forall;
                    if let Forallable::Function(func) = body
                        && let Some(new_tparams) =
                            prepend_class_tparams_if_used(func, Some(tparams))
                    {
                        *tparams = new_tparams;
                    }
                }
                Type::Overload(Overload { signatures, .. }) => {
                    signatures.iter_mut().for_each(|sig| match sig {
                        OverloadType::Function(body)
                            if let Some(tparams) = prepend_class_tparams_if_used(body, None) =>
                        {
                            *sig = OverloadType::Forall(Forall {
                                tparams,
                                body: body.clone(),
                            })
                        }
                        OverloadType::Forall(Forall { tparams, body })
                            if let Some(new_tparams) =
                                prepend_class_tparams_if_used(body, Some(tparams)) =>
                        {
                            *tparams = new_tparams;
                        }
                        _ => {}
                    });
                }
                ty => {
                    if !cls_tparams.is_empty() {
                        let mut qs: SmallSet<&Quantified> = SmallSet::new();
                        ty.collect_quantifieds(&mut qs);
                        *ambiguous = cls_tparams.iter().any(|x| qs.contains(&x.quantified));
                    }
                }
            }
        })
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

    fn as_raw_special_method_type(&self, instance: &Instance) -> Option<Type> {
        match self.instantiate_for(instance).0 {
            ClassFieldInner::Descriptor { ty, .. } => Some(ty),
            ClassFieldInner::Method { ty, .. } => Some(ty),
            ClassFieldInner::NestedClass { ty, .. } => Some(ty),
            ClassFieldInner::ClassAttribute { ty, .. } => match self.initialization() {
                ClassFieldInitialization::ClassBody(_) => Some(ty),
                ClassFieldInitialization::Method
                | ClassFieldInitialization::Uninitialized
                | ClassFieldInitialization::Magic => None,
            },
            ClassFieldInner::InstanceAttribute { .. } => None, // Instance attrs not in class body
            ClassFieldInner::Property { ty, .. } => Some(ty),
        }
    }

    fn as_special_method_type(&self, instance: &Instance) -> Option<Type> {
        self.as_raw_special_method_type(instance)
            .and_then(|ty| make_bound_method(instance.to_type(), ty).ok())
    }

    pub fn is_simple_instance_attribute(&self) -> bool {
        matches!(&self.0, ClassFieldInner::InstanceAttribute { .. })
    }

    pub fn ty(&self) -> Type {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => ty.clone(),
            ClassFieldInner::Descriptor { ty, .. } => ty.clone(),
            ClassFieldInner::Method { ty, .. } => ty.clone(),
            ClassFieldInner::NestedClass { ty, .. } => ty.clone(),
            ClassFieldInner::ClassAttribute { ty, .. } => ty.clone(),
            ClassFieldInner::InstanceAttribute { ty, .. } => ty.clone(),
        }
    }

    pub fn is_abstract(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { is_abstract, .. } => *is_abstract,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { is_abstract, .. } => *is_abstract,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { .. } => false,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    fn is_non_callable_protocol_method(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { ty, .. } => ty.is_non_callable_protocol_method(),
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { ty, .. } => ty.is_non_callable_protocol_method(),
            ClassFieldInner::InstanceAttribute { ty, .. } => ty.is_non_callable_protocol_method(),
        }
    }

    pub fn is_foreign_key(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { is_foreign_key, .. } => *is_foreign_key,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    pub fn has_choices(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { has_choices, .. } => *has_choices,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    pub fn as_named_tuple_type(&self) -> Type {
        self.ty()
    }

    pub fn as_named_tuple_requiredness(&self) -> Required {
        match &self.0 {
            ClassFieldInner::Property { .. } => Required::Optional(None),
            ClassFieldInner::Descriptor { .. } => Required::Optional(None),
            ClassFieldInner::Method { .. } => Required::Optional(None),
            ClassFieldInner::NestedClass { .. } => Required::Optional(None),
            ClassFieldInner::ClassAttribute {
                initialization: ClassFieldInitialization::ClassBody(_),
                ..
            } => Required::Optional(None),
            ClassFieldInner::ClassAttribute {
                initialization:
                    ClassFieldInitialization::Method
                    | ClassFieldInitialization::Uninitialized
                    | ClassFieldInitialization::Magic,
                ..
            } => Required::Required,
            ClassFieldInner::InstanceAttribute { .. } => Required::Required,
        }
    }

    pub fn as_typed_dict_field_info(self, required_by_default: bool) -> Option<TypedDictField> {
        match &self.0 {
            ClassFieldInner::ClassAttribute {
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
            ClassFieldInner::InstanceAttribute {
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
            ClassFieldInner::Property { .. } => None,
            _ => None,
        }
    }

    fn is_dataclass_kwonly_marker(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { ty, .. } => {
                matches!(ty, Type::ClassType(cls) if cls.has_qname("dataclasses", "KW_ONLY"))
            }
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    fn is_class_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_class_var())
            }
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { is_classvar, .. } => *is_classvar,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    pub fn is_uninit_class_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute {
                is_classvar,
                initialization,
                ..
            } => {
                *is_classvar
                    && matches!(
                        initialization,
                        ClassFieldInitialization::Uninitialized | ClassFieldInitialization::Magic
                    )
            }
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    pub fn is_init_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_init_var())
            }
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_init_var())
            }
            ClassFieldInner::InstanceAttribute { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_init_var())
            }
        }
    }

    pub fn is_final(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => ty.has_final_decoration(),
            ClassFieldInner::Descriptor { annotation, ty, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_final()) || ty.has_final_decoration()
            }
            ClassFieldInner::Method { ty, .. } => ty.has_final_decoration(),
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { annotation, ty, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_final()) || ty.has_final_decoration()
            }
            ClassFieldInner::InstanceAttribute { annotation, ty, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_final()) || ty.has_final_decoration()
            }
        }
    }

    fn is_override(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => ty.is_override(),
            ClassFieldInner::Descriptor { ty, .. } => ty.is_override(),
            ClassFieldInner::Method { ty, .. } => ty.is_override(),
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { ty, .. } => ty.is_override(),
            ClassFieldInner::InstanceAttribute { ty, .. } => ty.is_override(),
        }
    }

    /// Check if this field is read-only for any reason.
    fn is_read_only(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => ty.is_property_setter_with_getter().is_none(),
            ClassFieldInner::Descriptor { descriptor, .. } => !descriptor.setter,
            ClassFieldInner::Method { .. } => true,
            ClassFieldInner::NestedClass { .. } => true,
            ClassFieldInner::ClassAttribute {
                read_only_reason: Some(_),
                ..
            } => true,
            ClassFieldInner::ClassAttribute { is_classvar, .. } => *is_classvar,
            ClassFieldInner::InstanceAttribute {
                read_only_reason: Some(_),
                ..
            } => true,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    pub fn has_explicit_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { annotation, .. } => annotation.is_some(),
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { annotation, .. } => annotation.is_some(),
            ClassFieldInner::InstanceAttribute { annotation, .. } => annotation.is_some(),
        }
    }

    fn is_function_without_return_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method {
                is_function_without_return_annotation,
                ..
            } => *is_function_without_return_annotation,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { .. } => false,
            ClassFieldInner::InstanceAttribute { .. } => false,
        }
    }

    fn dataclass_flags_of(&self) -> DataclassFieldKeywords {
        match &self.0 {
            ClassFieldInner::Property { .. } => DataclassFieldKeywords::new(),
            ClassFieldInner::Descriptor { .. } => DataclassFieldKeywords::new(),
            ClassFieldInner::Method { .. } => DataclassFieldKeywords::new(),
            ClassFieldInner::NestedClass { .. } => DataclassFieldKeywords::new(),
            ClassFieldInner::ClassAttribute { initialization, .. } => match initialization {
                ClassFieldInitialization::ClassBody(Some(field_flags)) => (**field_flags).clone(),
                ClassFieldInitialization::ClassBody(None) => {
                    let mut kws = DataclassFieldKeywords::new();
                    kws.default = Some(Type::any_implicit());
                    kws
                }
                ClassFieldInitialization::Method
                | ClassFieldInitialization::Uninitialized
                | ClassFieldInitialization::Magic => DataclassFieldKeywords::new(),
            },
            ClassFieldInner::InstanceAttribute { .. } => DataclassFieldKeywords::new(),
        }
    }

    fn is_initialized_in_method(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Property { .. } => false,
            ClassFieldInner::Descriptor { .. } => false,
            ClassFieldInner::Method { .. } => false,
            ClassFieldInner::NestedClass { .. } => false,
            ClassFieldInner::ClassAttribute { initialization, .. } => {
                matches!(initialization, ClassFieldInitialization::Method)
            }
            ClassFieldInner::InstanceAttribute { .. } => true, // By definition, always assigned in methods
        }
    }
}

#[derive(Debug)]
enum InstanceKind {
    ClassType,
    TypedDict,
    TypeVar(Quantified),
    SelfType,
    Protocol(Type),
    Metaclass(ClassBase),
    LiteralString,
}

/// Wrapper to hold a specialized instance of a class , unifying ClassType and TypedDict.
#[derive(Debug)]
struct Instance<'a> {
    kind: InstanceKind,
    class: &'a Class,
    targs: &'a TArgs,
}

impl<'a> Instance<'a> {
    fn literal_string(stdlib: &'a Stdlib) -> Self {
        Self {
            kind: InstanceKind::LiteralString,
            class: stdlib.str().class_object(),
            targs: stdlib.str().targs(),
        }
    }

    fn of_class(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::ClassType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    fn of_typed_dict(td: &'a TypedDictInner) -> Self {
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

    fn of_self_type(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::SelfType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    fn of_protocol(cls: &'a ClassType, self_type: Type) -> Self {
        Self {
            kind: InstanceKind::Protocol(self_type),
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    fn of_metaclass(cls: ClassBase, metaclass: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::Metaclass(cls),
            class: metaclass.class_object(),
            targs: metaclass.targs(),
        }
    }

    /// Instantiate a type that is relative to the class type parameters
    /// by substituting in the type arguments.
    fn instantiate_member(&self, raw_member: &mut Type) {
        self.targs.substitute_into_mut(raw_member)
    }

    fn to_type(&self) -> Type {
        match &self.kind {
            InstanceKind::ClassType => {
                ClassType::new(self.class.dupe(), self.targs.clone()).to_type()
            }
            InstanceKind::TypedDict => {
                Type::TypedDict(TypedDict::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::TypeVar(q) => q.clone().to_type(),
            InstanceKind::SelfType => {
                Type::SelfType(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => self_type.clone(),
            InstanceKind::Metaclass(cls) => cls.clone().to_type(),
            InstanceKind::LiteralString => Type::LiteralString,
        }
    }

    /// Looking up a classmethod/staticmethod from an instance base has class-like
    /// lookup behavior. When this happens, we convert from an instance base to a class base.
    fn to_class_base(&self) -> ClassBase {
        match &self.kind {
            InstanceKind::SelfType => {
                ClassBase::SelfType(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => ClassBase::Protocol(
                ClassType::new(self.class.dupe(), self.targs.clone()),
                self_type.clone(),
            ),
            InstanceKind::TypeVar(q) => ClassBase::Quantified(
                q.clone(),
                ClassType::new(self.class.dupe(), self.targs.clone()),
            ),
            _ => ClassBase::ClassType(ClassType::new(self.class.dupe(), self.targs.clone())),
        }
    }

    fn to_descriptor_base(&self) -> Option<DescriptorBase> {
        match self.kind {
            // There's no situation in which you can stick a usable descriptor in a TypedDict.
            // TODO(rechen): a descriptor in a TypedDict should be an error at class creation time.
            InstanceKind::TypedDict => None,
            InstanceKind::ClassType
            | InstanceKind::SelfType
            | InstanceKind::Protocol(..)
            | InstanceKind::Metaclass(..)
            | InstanceKind::TypeVar(..)
            | InstanceKind::LiteralString => Some(DescriptorBase::Instance(ClassType::new(
                self.class.dupe(),
                self.targs.clone(),
            ))),
        }
    }
}

fn bind_class_attribute(
    cls: &ClassBase,
    attr: Type,
    read_only_reason: Option<ReadOnlyReason>,
) -> ClassAttribute {
    let ty = make_bound_classmethod(cls, attr).into_inner();
    if let Some(reason) = read_only_reason {
        ClassAttribute::read_only(ty, reason)
    } else {
        ClassAttribute::read_write(ty)
    }
}

/// Return the type of making it bound, or if not, the unbound type.
fn make_bound_method_helper(
    obj: Type,
    attr: Type,
    should_bind: &dyn Fn(&FuncMetadata) -> bool,
) -> Result<Type, Type> {
    // Don't bind functions originating from callback protocols, because the self param
    // has already been removed.
    let should_bind2 = |metadata: &FuncMetadata| {
        !matches!(metadata.kind, FunctionKind::CallbackProtocol(_)) && should_bind(metadata)
    };
    let func = match attr {
        Type::Forall(box Forall {
            tparams,
            body: Forallable::Function(func),
        }) if should_bind2(&func.metadata) => BoundMethodType::Forall(Forall {
            tparams,
            body: func,
        }),
        Type::Function(func) if should_bind2(&func.metadata) => BoundMethodType::Function(*func),
        Type::Overload(overload) if should_bind2(&overload.metadata) => {
            BoundMethodType::Overload(overload)
        }
        Type::Union(box Union {
            members: ref ts, ..
        }) => {
            let mut bound_methods = Vec::with_capacity(ts.len());
            for t in ts {
                match make_bound_method_helper(obj.clone(), t.clone(), should_bind) {
                    Ok(x) => bound_methods.push(x),
                    Err(_) => return Err(attr),
                }
            }
            return Ok(unions(bound_methods));
        }
        _ => return Err(attr),
    };
    Ok(Type::BoundMethod(Box::new(BoundMethod { obj, func })))
}

fn make_bound_classmethod(cls: &ClassBase, attr: Type) -> Result<Type, Type> {
    let should_bind = |meta: &FuncMetadata| meta.flags.is_classmethod;
    make_bound_method_helper(cls.clone().to_type(), attr, &should_bind)
}

fn make_bound_method(obj: Type, attr: Type) -> Result<Type, Type> {
    let should_bind =
        |meta: &FuncMetadata| !meta.flags.is_staticmethod && !meta.flags.is_classmethod;
    make_bound_method_helper(obj, attr, &should_bind)
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
    fn is_defined_on(&self, module: &str, cls: &str) -> bool {
        self.defining_class.has_toplevel_qname(module, cls)
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

fn is_method_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Function(_)
            | Type::Overload(_)
            | Type::Forall(box Forall {
                body: Forallable::Function(_),
                ..
            })
    )
}

fn has_any_method_type(ty: &Type) -> bool {
    match ty {
        Type::Union(union) => union.members.iter().any(is_method_type),
        _ => is_method_type(ty),
    }
}

fn has_any_abstract(ty: &Type) -> bool {
    match ty {
        Type::Union(union) => union.members.iter().any(|item| item.is_abstract_method()),
        _ => ty.is_abstract_method(),
    }
}

/// Determine if a class field should be treated as a method (getting method binding behavior). It is if:
/// - It's a function type (including staticmethods), initialized on the class body
/// - or, it's a Callable initialized on the class body and satisfying some special case:
///   - it's marked as a ClassVar
///   - it's assigned to a dunder name like `__add__`
/// - or, it's a union where ANY element satisfies the above rules
///
/// Note: staticmethods and union types that have at least one method type are included.
/// We rely on make_bound_method_helper to handle defining the binding logic for those cases.
fn is_method(
    ty: &Type,
    initialization: &ClassFieldInitialization,
    name: &Name,
    annotation: Option<&Annotation>,
) -> bool {
    let initialized_in_class_body =
        matches!(initialization, ClassFieldInitialization::ClassBody(_));

    if !initialized_in_class_body {
        return false;
    }

    // Check if it's a bindable function or union of functions
    if has_any_method_type(ty) {
        return true;
    }

    // Special cases where Callable is assumed to be a method
    if matches!(ty, Type::Callable(_)) {
        if is_dunder(name.as_str()) {
            return true;
        }
        if annotation
            .is_some_and(|ann| ann.is_class_var() && matches!(ann.get_type(), Type::Callable(_)))
        {
            return true;
        }
    }

    false
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn calculate_class_field(
        &self,
        class: &Class,
        name: &Name,
        range: TextRange,
        field_definition: &ClassFieldDefinition,
        functional_class_def: bool,
        errors: &ErrorCollector,
    ) -> ClassField {
        let metadata = self.get_metadata_for_class(class);
        if metadata.is_typed_dict() {
            return self.calculate_typed_dict_field(
                &metadata,
                name,
                range,
                field_definition,
                errors,
            );
        }

        // TODO(stroxler): Clean this up, as we convert more of the class field logic to using enums.
        //
        // It's a mess because we are relying on refs to fields that don't make sense for some cases,
        // which requires us having a place to store synthesized dummy values until we've refactored more.
        let value_storage = Owner::new();

        let (
            initialization,
            is_function_without_return_annotation,
            value_ty,
            annotation,
            is_inherited,
            direct_annotation,
        ) = match field_definition {
            ClassFieldDefinition::DeclaredByAnnotation { annotation: annot } => {
                let direct_annotation = Some(self.get_idx(*annot).annotation.clone());
                let initialization = if class.module_path().is_interface()
                    || direct_annotation
                        .as_ref()
                        .is_some_and(|annot| annot.has_qualifier(&Qualifier::ClassVar))
                {
                    ClassFieldInitialization::Magic
                } else if let Some(flags) =
                    self.extract_pydantic_field_from_annotation(*annot, &metadata)
                {
                    ClassFieldInitialization::ClassBody(Some(Box::new(flags)))
                } else {
                    ClassFieldInitialization::Uninitialized
                };
                let value =
                    value_storage.push(ExprOrBinding::Binding(Binding::Type(Type::any_implicit())));
                let (value_ty, annotation, is_inherited) = self.analyze_class_field_value(
                    value,
                    class,
                    name,
                    direct_annotation.as_ref(),
                    false,
                    errors,
                );
                (
                    initialization,
                    false,
                    value_ty,
                    annotation,
                    is_inherited,
                    direct_annotation,
                )
            }
            ClassFieldDefinition::AssignedInBody {
                value,
                annotation: annot,
                ..
            } => {
                let direct_annotation = annot.map(|a| self.get_idx(a).annotation.clone());
                let initialization = if let ExprOrBinding::Expr(e) = value
                    && let Some(dm) = metadata.dataclass_metadata()
                    && let Expr::Call(call) = e
                {
                    let flags = self.compute_dataclass_field_initialization(call, dm);
                    ClassFieldInitialization::ClassBody(flags.map(Box::new))
                } else {
                    ClassFieldInitialization::ClassBody(None)
                };
                let (value_ty, annotation, is_inherited) = self.analyze_class_field_value(
                    value,
                    class,
                    name,
                    direct_annotation.as_ref(),
                    false,
                    errors,
                );
                (
                    initialization,
                    false,
                    value_ty,
                    annotation,
                    is_inherited,
                    direct_annotation,
                )
            }
            ClassFieldDefinition::DefinedInMethod {
                value,
                method,
                annotation: annot,
                ..
            } => {
                let direct_annotation = annot.map(|a| self.get_idx(a).annotation.clone());
                // Check if there's an inherited property field from a parent class
                // If so, we should just use the parent's property instead of creating a new field
                if !Ast::is_mangled_attr(name) {
                    // Use get_field_from_ancestors to only look at parent classes, not the current class
                    if let Some(parent_field) = self.get_field_from_ancestors(
                        class,
                        self.get_mro_for_class(class).ancestors(self.stdlib),
                        name,
                        &|cls, name| self.get_field_from_current_class_only(cls, name),
                    ) {
                        match &*parent_field.value {
                            ClassField(ClassFieldInner::Property { .. }, ..) => {
                                // If we found a property in the parent, just return the parent's field
                                // This ensures the property with its setter is properly inherited
                                return Arc::unwrap_or_clone(parent_field.value);
                            }
                            _ => {
                                // For non-property fields, continue with normal processing
                            }
                        }
                    }
                }

                let initialization = match method.instance_or_class {
                    MethodSelfKind::Class => ClassFieldInitialization::ClassBody(None),
                    MethodSelfKind::Instance => ClassFieldInitialization::Method,
                };
                let (mut value_ty, annotation, is_inherited) = self.analyze_class_field_value(
                    value,
                    class,
                    name,
                    direct_annotation.as_ref(),
                    true,
                    errors,
                );
                if matches!(method.instance_or_class, MethodSelfKind::Instance) {
                    value_ty = self
                        .check_and_sanitize_type_parameters(class, value_ty, name, range, errors);
                }
                (
                    initialization,
                    false,
                    value_ty,
                    annotation,
                    is_inherited,
                    direct_annotation,
                )
            }
            ClassFieldDefinition::MethodLike {
                definition,
                has_return_annotation,
            } => {
                let initialization = ClassFieldInitialization::ClassBody(None);
                // Evaluate the binding directly without analyzing inherited annotations
                let binding = Binding::Forward(*definition);
                let value_ty = Arc::unwrap_or_clone(self.solve_binding(&binding, errors)).into_ty();
                (
                    initialization,
                    !has_return_annotation,
                    value_ty,
                    None, // No annotation for methods
                    IsInherited::Maybe,
                    None,
                )
            }
            ClassFieldDefinition::NestedClass { definition } => {
                // Evaluate the binding directly without analyzing inherited annotations
                let initialization = ClassFieldInitialization::ClassBody(None);
                let binding = Binding::Forward(*definition);
                let value_ty = Arc::unwrap_or_clone(self.solve_binding(&binding, errors)).into_ty();
                (
                    initialization,
                    false,
                    value_ty,
                    None, // No annotation for nested classes
                    IsInherited::Maybe,
                    None,
                )
            }
            ClassFieldDefinition::DefinedWithoutAssign { definition } => {
                let initialization = ClassFieldInitialization::ClassBody(None);
                let value =
                    value_storage.push(ExprOrBinding::Binding(Binding::Forward(*definition)));
                let (value_ty, annotation, is_inherited) =
                    self.analyze_class_field_value(value, class, name, None, false, errors);
                (
                    initialization,
                    false,
                    value_ty,
                    annotation,
                    is_inherited,
                    None,
                )
            }
            ClassFieldDefinition::DeclaredWithoutAnnotation => {
                // This is a field in a synthesized class with no information at all, treat it as Any.
                let initialization = if class.module_path().is_interface() {
                    ClassFieldInitialization::Magic
                } else {
                    ClassFieldInitialization::Uninitialized
                };
                let value =
                    value_storage.push(ExprOrBinding::Binding(Binding::Type(Type::any_implicit())));
                let (value_ty, annotation, is_inherited) =
                    self.analyze_class_field_value(value, class, name, None, false, errors);
                (
                    initialization,
                    false,
                    value_ty,
                    annotation,
                    is_inherited,
                    None,
                )
            }
        };

        if let Some(annotation) = direct_annotation.as_ref() {
            self.validate_direct_annotation(
                annotation,
                &metadata,
                &initialization,
                name,
                range,
                errors,
            );
        }

        let read_only_reason =
            self.determine_read_only_reason(name, annotation.as_ref(), &metadata, field_definition);

        // Determine the final type, promoting literals when appropriate.
        let ty = if annotation
            .as_ref()
            .and_then(|ann| ann.ty.as_ref())
            .is_none()
            && matches!(read_only_reason, None | Some(ReadOnlyReason::NamedTuple))
            && value_ty.is_literal()
        {
            value_ty.promote_literals(self.stdlib)
        } else {
            value_ty
        };

        // Identify whether this is a descriptor
        let mut descriptor = None;
        match &ty {
            // TODO(stroxler): This works for simple descriptors. There three known gaps, there may be others:
            // - If the field is instance-only, descriptor dispatching won't occur, an instance-only attribute
            //   that happens to be a descriptor just behaves like a normal instance-only attribute.
            // - Gracefully handle instance-only `__get__`/`__set__`. Descriptors only seem to be detected
            //   when the descriptor attribute is initialized on the class body of the descriptor.
            // - Do we care about distributing descriptor behavior over unions? If so, what about the case when
            //   the raw class field is a union of a descriptor and a non-descriptor? Do we want to allow this?
            Type::ClassType(cls) => {
                let getter = self
                    .get_class_member(cls.class_object(), &dunder::GET)
                    .is_some();
                let setter = self
                    .get_class_member(cls.class_object(), &dunder::SET)
                    .is_some();
                if getter || setter {
                    descriptor = Some(Descriptor {
                        range,
                        cls: cls.clone(),
                        getter,
                        setter,
                    })
                }
            }
            _ => {}
        };
        // Check if this is a Django ForeignKey field
        let is_foreign_key = metadata.is_django_model()
            && matches!(&ty, Type::ClassType(cls) if self.is_foreign_key_field(cls.class_object()));

        // Check if this is a Django field with choices
        let has_choices = if let ClassFieldDefinition::AssignedInBody {
            value: ExprOrBinding::Expr(expr),
            ..
        } = field_definition
            && let Some(call_expr) = expr.as_call_expr()
        {
            metadata.is_django_model() && self.has_django_field_choices(call_expr)
        } else {
            false
        };

        let ty = if let Some(special_ty) = self.get_special_class_field_type(
            class,
            name,
            direct_annotation.as_ref(),
            &ty,
            field_definition,
            descriptor.is_some(),
            range,
            errors,
        ) {
            // Don't use the descriptor, since we've set a custom type instead.
            descriptor = None;
            special_ty
        } else {
            ty
        };

        // Pin any vars in the type: leaking a var in a class field is particularly
        // likely to lead to data races where downstream uses can pin inconsistently.
        //
        // TODO(stroxler): Ideally we would implement some simple heuristics, similar to
        // first-use based inference we use with assignments, to get more useful types here.
        let ty = self.solver().deep_force(ty);

        // Create the resulting field and check for override inconsistencies before returning
        let is_abstract = ty.is_abstract_method();

        // Detect if this is a property or descriptor and construct the appropriate variant
        let class_field = if matches!(field_definition, ClassFieldDefinition::NestedClass { .. }) {
            // Nested classes have their own variant
            ClassField(ClassFieldInner::NestedClass { ty }, is_inherited)
        } else if ty.is_property_getter() || ty.is_property_setter_with_getter().is_some() {
            ClassField(ClassFieldInner::Property { ty, is_abstract }, is_inherited)
        } else if let Some(descriptor) = descriptor {
            // Descriptors are always initialized in class body (or wouldn't trigger descriptor protocol)
            ClassField(
                ClassFieldInner::Descriptor {
                    ty,
                    annotation,
                    descriptor,
                },
                is_inherited,
            )
        } else if is_method(&ty, &initialization, name, annotation.as_ref()) {
            // Use helper functions to compute flags for unions
            let is_abstract_flag = has_any_abstract(&ty);
            ClassField(
                ClassFieldInner::Method {
                    ty,
                    is_abstract: is_abstract_flag,
                    is_function_without_return_annotation,
                },
                is_inherited,
            )
        } else {
            // Decide between ClassAttribute and InstanceAttribute
            let is_class_var = annotation.as_ref().is_some_and(|ann| ann.is_class_var());

            // Special case: In enums, the `_value_` field should be an InstanceAttribute
            // even when it's just an annotation (Uninitialized), because it's used for
            // enum member validation
            let is_enum_value_field = metadata.is_enum() && name.as_str() == "_value_";

            match (&initialization, is_class_var, is_enum_value_field) {
                // Enum `_value_` field: always InstanceAttribute
                (_, _, true) => ClassField(
                    ClassFieldInner::InstanceAttribute {
                        ty,
                        annotation,
                        read_only_reason,
                    },
                    is_inherited,
                ),
                // Instance attributes: only those assigned in methods
                // Note: annotation-only (Uninitialized) fields are treated as ClassAttribute
                // to avoid false positives, even though semantically they're instance-only
                (ClassFieldInitialization::Method, false, false) => ClassField(
                    ClassFieldInner::InstanceAttribute {
                        ty,
                        annotation,
                        read_only_reason,
                    },
                    is_inherited,
                ),
                // Everything else is a class attribute
                _ => {
                    let is_staticmethod =
                        matches!(&ty, Type::ClassType(cls) if cls.is_builtin("staticmethod"));
                    ClassField(
                        ClassFieldInner::ClassAttribute {
                            ty,
                            annotation,
                            initialization,
                            read_only_reason,
                            is_classvar: is_class_var,
                            is_staticmethod,
                            is_foreign_key,
                            has_choices,
                        },
                        is_inherited,
                    )
                }
            }
        };

        // *** Everything below here is for validation only and has no impact on downstream analysis ***

        if let ClassFieldDefinition::DefinedInMethod {
            method:
                MethodThatSetsAttr {
                    method_name,
                    recognized_attribute_defining_method,
                    instance_or_class: _,
                },
            ..
        } = field_definition
        {
            let mut defined_in_parent = false;
            let parents = metadata.base_class_objects();
            for parent in parents {
                if self.get_class_member(parent, name).is_some() {
                    defined_in_parent = true;
                    break;
                };
            }
            if !defined_in_parent {
                if metadata.is_protocol() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::ProtocolImplicitlyDefinedAttribute),
                        "Instance or class variables within a Protocol class must be explicitly declared within the class body".to_owned(),
                    );
                } else if !recognized_attribute_defining_method {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::ImplicitlyDefinedAttribute,),
                        format!("Attribute `{}` is implicitly defined by assignment in method `{method_name}`, which is not a constructor", &name),
                    );
                }
            }
        }

        if let Some(dm) = metadata.dataclass_metadata()
            && name == &dunder::POST_INIT
            && let Some(post_init) = class_field
                .as_special_method_type(&Instance::of_class(&self.as_class_type_unchecked(class)))
        {
            self.validate_post_init(class, dm, post_init, range, errors);
        }

        if let Some(named_tuple_metadata) = metadata.named_tuple_metadata()
            && !functional_class_def
            && named_tuple_metadata.elements.contains(name)
            && name.as_str().starts_with('_')
        {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                format!("NamedTuple field name may not start with an underscore: `{name}`"),
            );
        }

        class_field
    }

    /// Helper to infer with an optional annotation as a hint and then expand
    pub fn attribute_expr_infer(
        &self,
        x: &Expr,
        annotation: Option<&Annotation>,
        name: &Name,
        errors: &ErrorCollector,
    ) -> Type {
        let mut ty = match (annotation, x) {
            (Some(annotation), _) => {
                let ctx: &dyn Fn() -> TypeCheckContext =
                    &|| TypeCheckContext::of_kind(TypeCheckKind::Attribute(name.clone()));
                let hint = Some((annotation.get_type(), ctx));
                self.expr(x, hint, errors)
            }
            // We interpret `self.foo = None` to mean the type of foo is None or some unknown type.
            (None, Expr::NoneLiteral(_)) => {
                self.error(errors, x.range(), ErrorInfo::Kind(ErrorKind::UnannotatedAttribute), "This expression is implicitly inferred to be `Any | None`. Please provide an explicit type annotation.".to_owned());
                self.union(Type::None, Type::any_implicit())
            }
            (None, _) => self.expr_infer(x, errors),
        };
        self.expand_vars_mut(&mut ty);
        ty
    }

    /// Apply any class-specific logic for postprocessing the type of a class field. Hook into this
    /// function if you need to modify the type of an existing field. If you need to add a new
    /// field, see ClassSynthesizedField instead.
    fn get_special_class_field_type(
        &self,
        class: &Class,
        name: &Name,
        direct_annotation: Option<&Annotation>,
        ty: &Type,
        field_definition: &ClassFieldDefinition,
        is_descriptor: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let is_initialized_on_class_body = matches!(
            field_definition,
            ClassFieldDefinition::AssignedInBody { .. }
                | ClassFieldDefinition::MethodLike { .. }
                | ClassFieldDefinition::DefinedWithoutAssign { .. }
        );
        self.get_enum_class_field_type(
            class,
            name,
            direct_annotation,
            ty,
            is_initialized_on_class_body,
            is_descriptor,
            range,
            errors,
        )
        .or_else(|| self.get_pydantic_root_model_class_field_type(class, name))
        .or_else(|| {
            let initial_value_expr = match field_definition {
                ClassFieldDefinition::AssignedInBody {
                    value: ExprOrBinding::Expr(expr),
                    ..
                } => Some(expr),
                _ => None,
            };
            self.get_django_field_type(ty, class, Some(name), initial_value_expr)
        })
    }

    fn determine_read_only_reason(
        &self,
        name: &Name,
        annotation: Option<&Annotation>,
        metadata: &ClassMetadata,
        field_definition: &ClassFieldDefinition,
    ) -> Option<ReadOnlyReason> {
        if let Some(ann) = annotation {
            if ann.is_final() {
                let is_initialized = matches!(
                    field_definition,
                    ClassFieldDefinition::AssignedInBody { .. }
                );
                return Some(ReadOnlyReason::Final(if is_initialized {
                    IsFinalVariableInitialized::Yes
                } else {
                    IsFinalVariableInitialized::No
                }));
            }
            if ann.has_qualifier(&Qualifier::ReadOnly) {
                return Some(ReadOnlyReason::ReadOnlyQualifier);
            }
        }
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
            let reason = if metadata.is_pydantic_base_model() {
                ReadOnlyReason::PydanticFrozen
            } else {
                ReadOnlyReason::FrozenDataclass
            };
            return Some(reason);
        }

        // Nested class definitions are read-only
        if matches!(field_definition, ClassFieldDefinition::NestedClass { .. }) {
            return Some(ReadOnlyReason::ClassObjectInitializedOnBody);
        }
        // Default: the field is read-write
        None
    }

    /// Return (type of first inherited field, first inherited annotation). May not be from the same class!
    /// For example, in:
    ///   class A:
    ///     x: int
    ///   class B(A):
    ///     x = 0
    ///   class C(B):
    ///     x = 1
    /// `get_inherited_type_and_annotation(C, 'x')` will get the type from `B` and the annotation from `A`.
    fn get_inherited_type_and_annotation(
        &self,
        class: &Class,
        name: &Name,
    ) -> (Option<Type>, Option<Annotation>) {
        // Private (double-underscore) attributes are name-mangled at runtime and should not
        // inherit types or annotations from parent classes.
        if Ast::is_mangled_attr(name) {
            return (None, None);
        }
        let mut found_field = None;
        let annotation = self
            .get_mro_for_class(class)
            .ancestors(self.stdlib)
            .find_map(|parent| {
                let parent_field =
                    self.get_field_from_current_class_only(parent.class_object(), name)?;
                match &*parent_field {
                    ClassField(ClassFieldInner::Property { ty, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        None
                    }
                    ClassField(ClassFieldInner::Descriptor { ty, annotation, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        annotation
                            .clone()
                            .map(|ann| ann.substitute_with(parent.targs().substitution()))
                    }
                    ClassField(ClassFieldInner::Method { ty, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        None
                    }
                    ClassField(ClassFieldInner::NestedClass { ty, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        // Nested classes don't have annotations
                        None
                    }
                    ClassField(ClassFieldInner::ClassAttribute { ty, annotation, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        annotation
                            .clone()
                            .map(|ann| ann.substitute_with(parent.targs().substitution()))
                    }
                    ClassField(ClassFieldInner::InstanceAttribute { ty, annotation, .. }, ..) => {
                        if found_field.is_none() {
                            found_field =
                                Some(parent.targs().substitution().substitute_into(ty.clone()));
                        }
                        annotation
                            .clone()
                            .map(|ann| ann.substitute_with(parent.targs().substitution()))
                    }
                }
            });
        (found_field, annotation)
    }

    /// Compute the type, final annotation (direct or inherited), and inheritance status for a field
    /// that has a `value: ExprOrBinding` and possibly a directly annotated type.
    fn analyze_class_field_value(
        &self,
        value: &ExprOrBinding,
        class: &Class,
        name: &Name,
        direct_annotation: Option<&Annotation>,
        inferred_from_method: bool,
        errors: &ErrorCollector,
    ) -> (Type, Option<Annotation>, IsInherited) {
        // If we have a direct annotation with a type, use it and skip analyzing the value
        let direct_qualifiers = if let Some(ann) = direct_annotation {
            if let Some(ty) = ann.ty.clone() {
                let is_inherited = if Ast::is_mangled_attr(name) {
                    IsInherited::No
                } else {
                    IsInherited::Maybe
                };
                return (ty, direct_annotation.cloned(), is_inherited);
            } else {
                Some(&ann.qualifiers)
            }
        } else {
            None
        };
        // Otherwise, analyze the value to determine the type
        let (inherited_ty, inherited_annotation) =
            self.get_inherited_type_and_annotation(class, name);
        let is_inherited = if inherited_ty.is_none() {
            IsInherited::No
        } else {
            IsInherited::Maybe
        };
        let final_annotation = Self::merge_direct_qualifiers_with_inherited_annotation(
            inherited_annotation.clone(),
            direct_qualifiers,
        );
        let inferred_ty = match value {
            ExprOrBinding::Expr(e) => {
                match inherited_ty {
                    Some(inherited_ty) if inferred_from_method => {
                        // Inherit the previous type of the attribute if the only declaration-like
                        // thing the current class does is assign to the attribute in a method.
                        inherited_ty
                    }
                    Some(inherited_ty)
                        if inherited_annotation.is_none() && direct_annotation.is_none() =>
                    {
                        // If there are no explicit annotations, use the inherited type as a contextual hint.
                        let errors2 = self.error_collector();
                        self.attribute_expr_infer(
                            e,
                            Some(&Annotation::new_type(inherited_ty.clone())),
                            name,
                            &errors2,
                        );
                        if errors2.is_empty() {
                            // The new type is compatible with the inherited one; use the inherited type to
                            // avoid spurious errors about changing the type of a read-write attribute.
                            inherited_ty
                        } else {
                            // The hint was no good; infer the type without it.
                            self.attribute_expr_infer(e, None, name, errors)
                        }
                    }
                    _ => self.attribute_expr_infer(e, inherited_annotation.as_ref(), name, errors),
                }
            }
            ExprOrBinding::Binding(b) => {
                Arc::unwrap_or_clone(self.solve_binding(b, errors)).into_ty()
            }
        };
        // Note that we use `final_annotation`'s `ty` rather than `inherited_ty`
        // because we only want to override the `inferred_ty` when there's an inherited
        // *annotation*, and in some cases `inherited_ty` is inferred (which means we only
        // use it for contextual typing, not as an explicit type declaration)
        let ty = final_annotation
            .as_ref()
            .and_then(|ann| ann.ty.clone())
            .unwrap_or(inferred_ty);
        (ty, final_annotation, is_inherited)
    }

    /// Given an inherited annotation and possible qualifiers from a direct annotation,
    /// combine them:
    /// - If neither is present, return `None`
    /// - If only one is non-None, create an annotation from it
    /// - If both are non-None, combine the type from the inherited annotation from
    ///   the direct qualifiers.
    ///
    /// Note that qualifiers from the inherited annotation are always discarded
    /// if the direct annotation is present. This is arguable (the typing rules
    /// don't specify behavior) but simpler than combining qualifiers.
    fn merge_direct_qualifiers_with_inherited_annotation(
        inherited: Option<Annotation>,
        direct_qualifiers: Option<&Vec<Qualifier>>,
    ) -> Option<Annotation> {
        match (inherited, direct_qualifiers) {
            (inherited, Some(qualifiers)) => Some(Annotation {
                ty: inherited.and_then(|ann| ann.ty),
                qualifiers: qualifiers.clone(),
            }),
            (ann, None) => ann,
        }
    }

    /// Extract dataclass field keywords from a call expression if it's a dataclass field specifier.
    pub fn compute_dataclass_field_initialization(
        &self,
        call: &ExprCall,
        dm: &crate::alt::types::class_metadata::DataclassMetadata,
    ) -> Option<DataclassFieldKeywords> {
        let ExprCall {
            node_index: _,
            range: _,
            func,
            arguments,
        } = call;
        // We already type-checked this expression as part of computing the type for the ClassField,
        // so we can ignore any errors encountered here.
        let ignore_errors = self.error_swallower();
        let func_ty = self.expr_infer(func, &ignore_errors);
        let func_kind = func_ty.callee_kind();
        if let Some(func_kind) = func_kind
            && dm.field_specifiers.contains(&func_kind)
        {
            let flags = self.dataclass_field_keywords(&func_ty, arguments, dm, &ignore_errors);
            Some(flags)
        } else {
            None
        }
    }

    fn validate_direct_annotation(
        &self,
        annotation: &Annotation,
        metadata: &ClassMetadata,
        initialization: &ClassFieldInitialization,
        name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if metadata.is_pydantic_base_model()
            && let ClassFieldInitialization::ClassBody(Some(kws)) = initialization
        {
            let field_ty = annotation.get_type();
            self.check_pydantic_range_constraints(name, field_ty, kws, range, errors);
        }

        // Check for qualifiers that are used in improper contexts.
        if metadata
            .named_tuple_metadata()
            .is_some_and(|m| m.elements.contains(name))
        {
            for q in &[Qualifier::Final, Qualifier::ClassVar] {
                if annotation.has_qualifier(q) {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("`{q}` may not be used for NamedTuple members",),
                    );
                }
            }
        }
        for q in &[
            Qualifier::Required,
            Qualifier::NotRequired,
            Qualifier::ReadOnly,
        ] {
            if annotation.has_qualifier(q) {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("`{q}` may only be used for TypedDict members"),
                );
            }
        }
    }

    /// This is used for dataclass field synthesis; when accessing attributes on dataclass instances,
    /// use `get_instance_attribute` or `get_class_attribute`
    pub fn get_dataclass_member(&self, cls: &Class, name: &Name) -> DataclassMember {
        // Even though we check that the class member exists before calling this function,
        // it can be None if the class has an invalid MRO.
        let Some(member) = self.get_non_synthesized_dataclass_member_impl(cls, name) else {
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
                    .get_inherited_type_and_annotation(cls, name)
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

    fn check_and_sanitize_type_parameters(
        &self,
        class: &Class,
        ty: Type,
        name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut qs = SmallSet::new();
        ty.collect_quantifieds(&mut qs);
        if qs.is_empty() {
            drop(qs);
            return ty;
        }
        let class_tparams = self.get_class_tparams(class);
        let qs_owner = Owner::new();
        let ts = Owner::new();
        let gradual_fallbacks = qs
            .difference(&class_tparams.quantifieds().collect())
            .map(|q| {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                    format!(
                        "Attribute `{}` cannot depend on type variable `{}`, which is not in the scope of class `{}`",
                        name,
                        q.name(),
                        class.name(),
                    ),
                );
                (qs_owner.push((*q).clone()), ts.push(q.as_gradual_type()))
            })
            .collect::<SmallMap<_, _>>();
        drop(qs);
        ty.subst(&gradual_fallbacks)
    }

    /// Creates a Quantified for a class's "self" type.
    fn get_self_quantified(&self, class_name: &Name, instance_type: Type) -> Quantified {
        Quantified::new(
            self.uniques.fresh(),
            Name::new(format!("Self@{class_name}")),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Bound(instance_type),
        )
    }

    /// Makes `ty` generic in `quantified` by wrapping in a `Forall`.
    fn wrap_with_quantified(&self, ty: Type, quantified: Quantified) -> Type {
        fn forall<T: Visit<Type>>(
            body: T,
            quantified: &Quantified,
            existing_tparams: Option<&TParams>,
        ) -> Option<Forall<T>> {
            let mut quantifieds = SmallSet::new();
            body.visit(&mut |t| t.collect_quantifieds(&mut quantifieds));
            let uses_quantified = quantifieds.contains(quantified);
            drop(quantifieds);
            if !uses_quantified {
                return None;
            }
            let new_tparams = TParams::new(vec![TParam {
                quantified: quantified.clone(),
                variance: PreInferenceVariance::Undefined,
            }]);
            let tparams = if let Some(mut tparams) = existing_tparams.cloned() {
                tparams.extend(&new_tparams);
                tparams
            } else {
                new_tparams
            };
            Some(Forall {
                tparams: Arc::new(tparams),
                body,
            })
        }
        match &ty {
            Type::Function(func) => {
                forall(Forallable::Function((**func).clone()), &quantified, None)
                    .map_or(ty, |forall| Type::Forall(Box::new(forall)))
            }
            Type::Forall(box Forall { tparams, body }) => {
                forall(body.clone(), &quantified, Some(tparams))
                    .map_or(ty, |forall| Type::Forall(Box::new(forall)))
            }
            Type::Overload(overload) => Type::Overload(Overload {
                signatures: overload.signatures.clone().mapped(|sig| match &sig {
                    OverloadType::Function(func) => {
                        forall(func.clone(), &quantified, None).map_or(sig, OverloadType::Forall)
                    }
                    OverloadType::Forall(Forall { tparams, body }) => {
                        forall(body.clone(), &quantified, Some(tparams))
                            .map_or(sig, OverloadType::Forall)
                    }
                }),
                metadata: overload.metadata.clone(),
            }),
            _ => ty,
        }
    }

    fn as_instance_attribute(
        &self,
        field_name: &Name,
        field: &ClassField,
        instance: &Instance,
    ) -> ClassAttribute {
        // Special handling for `__new__`: because `__new__` is a static method, it can be called
        // with a `cls` argument that differs from the class on which it is accessed, so we use a
        // quantified to capture `cls`. Note that `__new__` is the only method that needs this
        // special case because it's not legal to use `Self` in other static methods.
        let self_quantified =
            if field_name == &dunder::NEW && matches!(instance.kind, InstanceKind::ClassType) {
                Some(self.get_self_quantified(instance.class.name(), instance.to_type()))
            } else {
                None
            };
        let instance = match &self_quantified {
            Some(quantified) => &Instance {
                kind: InstanceKind::TypeVar(quantified.clone()),
                class: instance.class,
                targs: instance.targs,
            },
            None => instance,
        };
        match field.instantiate_for(instance).0 {
            ClassFieldInner::Property { ty, .. } => {
                // Properties on instances bind to the getter/setter
                if let Some(getter) = ty.is_property_setter_with_getter() {
                    // Property with a setter: bind both getter and setter
                    ClassAttribute::property(
                        make_bound_method(instance.to_type(), getter).into_inner(),
                        Some(make_bound_method(instance.to_type(), ty.clone()).into_inner()),
                        instance.class.dupe(),
                    )
                } else {
                    // Property getter only (no setter)
                    ClassAttribute::property(
                        make_bound_method(instance.to_type(), ty.clone()).into_inner(),
                        None,
                        instance.class.dupe(),
                    )
                }
            }
            ClassFieldInner::Descriptor { descriptor, .. } => {
                if let Some(base) = instance.to_descriptor_base() {
                    ClassAttribute::descriptor(descriptor, base)
                } else {
                    // Unreachable because only TypedDicts can hit this, and we never
                    // construct Descriptors for typed dicts.
                    unreachable!("A descriptor attribute should always have a valid base")
                }
            }
            ClassFieldInner::Method { mut ty, .. } => {
                // bind_instance matches on the type, so resolve it if we can
                self.expand_vars_mut(&mut ty);
                // If the field is a dunder or ClassVar[Callable] & the assigned value is a callable, we replace it with a named function
                // so that it gets treated as a bound method.
                //
                // Both of these are heuristics that aren't guaranteed to be correct, but the dunder heuristic has useability benefits
                // and the ClassVar heuristic aligns us with existing type checkers.
                if let Type::Callable(box callable) = ty {
                    let module = self.module();
                    let func_id = FuncId {
                        module: module.clone(),
                        cls: None,
                        name: field_name.clone(),
                    };
                    ty = Type::Function(Box::new(Function {
                        signature: callable,
                        metadata: FuncMetadata {
                            kind: FunctionKind::Def(Box::new(func_id)),
                            flags: FuncFlags::default(),
                        },
                    }))
                }
                if let Some(quantified) = self_quantified {
                    ty = self.wrap_with_quantified(ty, quantified);
                }
                ClassAttribute::read_write(
                    make_bound_method(instance.to_type(), ty).unwrap_or_else(|ty| {
                        make_bound_classmethod(&instance.to_class_base(), ty).into_inner()
                    }),
                )
            }
            ClassFieldInner::NestedClass { ty, .. } => {
                // Nested classes are always read-only (ClassObjectInitializedOnBody)
                ClassAttribute::read_only(ty, ReadOnlyReason::ClassObjectInitializedOnBody)
            }
            ClassFieldInner::ClassAttribute {
                mut ty,
                is_classvar,
                read_only_reason,
                ..
            } => {
                self.expand_vars_mut(&mut ty);
                if is_classvar {
                    ClassAttribute::read_only(ty, ReadOnlyReason::ClassVar)
                } else if let Some(reason) = read_only_reason {
                    ClassAttribute::read_only(ty, reason)
                } else {
                    ClassAttribute::read_write(ty)
                }
            }
            ClassFieldInner::InstanceAttribute {
                ty,
                read_only_reason,
                ..
            } => {
                if let Some(reason) = read_only_reason {
                    ClassAttribute::read_only(ty, reason)
                } else {
                    ClassAttribute::read_write(ty)
                }
            }
        }
    }

    fn as_class_attribute(
        &self,
        field_name: &Name,
        field: &ClassField,
        cls: &ClassBase,
    ) -> ClassAttribute {
        // Special handling for `__new__`: because `__new__` is a static method, it can be called
        // with a `cls` argument that differs from the class on which it is accessed, so we use a
        // quantified to capture `cls`. Note that `__new__` is the only method that needs this
        // special case because it's not legal to use `Self` in other static methods.
        let self_quantified = if field_name == &dunder::NEW
            && let ClassBase::ClassDef(cls) = cls
        {
            Some(self.get_self_quantified(cls.class_object().name(), cls.clone().to_type()))
        } else {
            None
        };
        let self_type = match &self_quantified {
            Some(quantified) => quantified.clone().to_type(),
            None => cls.clone().to_self_type(),
        };
        let mut ambiguous = false;
        let field = match cls.targs() {
            Some(targs) => field.instantiate_for_class_targs(targs, self_type, &mut ambiguous),
            None => {
                let tparams = self.get_class_tparams(cls.class_object());
                field.instantiate_for_class_tparams(tparams, self_type, &mut ambiguous)
            }
        };
        match field.0 {
            ClassFieldInner::Property { ty, .. } => {
                // When accessing a property on a class (not instance), you get the property object itself
                bind_class_attribute(cls, ty, None)
            }
            ClassFieldInner::Descriptor { descriptor, .. } => ClassAttribute::descriptor(
                descriptor,
                DescriptorBase::ClassDef(cls.class_object().dupe()),
            ),
            ClassFieldInner::Method { mut ty, .. } => {
                // When accessing a method on a class (not instance), you get the unbound function
                if let Some(quantified) = self_quantified {
                    ty = self.wrap_with_quantified(ty, quantified);
                }
                bind_class_attribute(cls, ty, None)
            }
            ClassFieldInner::NestedClass { ty, .. } => {
                // Nested classes are always read-only (ClassObjectInitializedOnBody)
                bind_class_attribute(cls, ty, Some(ReadOnlyReason::ClassObjectInitializedOnBody))
            }
            ClassFieldInner::ClassAttribute {
                initialization: ClassFieldInitialization::Method,
                ..
            } => ClassAttribute::no_access(NoAccessReason::ClassUseOfInstanceAttribute(
                cls.class_object().dupe(),
            )),
            ClassFieldInner::ClassAttribute {
                ty,
                read_only_reason,
                ..
            } => {
                if ambiguous {
                    ClassAttribute::no_access(NoAccessReason::ClassAttributeIsGeneric(
                        cls.class_object().dupe(),
                    ))
                } else {
                    bind_class_attribute(cls, ty, read_only_reason)
                }
            }
            ClassFieldInner::InstanceAttribute { .. } => {
                // Instance attributes are always initialized in methods
                ClassAttribute::no_access(NoAccessReason::ClassUseOfInstanceAttribute(
                    cls.class_object().dupe(),
                ))
            }
        }
    }

    pub fn as_param(
        &self,
        field: &ClassField,
        name: &Name,
        default: bool,
        kw_only: bool,
        strict: bool,
        converter_param: Option<Type>,
        param_type_transform: &dyn Fn(Type) -> Type,
        errors: &ErrorCollector,
    ) -> Param {
        let (ty, descriptor) = match &field.0 {
            ClassFieldInner::Property { ty, .. } => (ty, None),
            ClassFieldInner::Descriptor { ty, descriptor, .. } => (ty, Some(descriptor)),
            ClassFieldInner::Method { ty, .. } => (ty, None),
            ClassFieldInner::NestedClass { ty, .. } => (ty, None),
            ClassFieldInner::ClassAttribute { ty, .. } => (ty, None),
            ClassFieldInner::InstanceAttribute { ty, .. } => (ty, None),
        };
        let param_ty = if let Some(converter_param) = converter_param {
            // If a converter is specified (e.g., from pydantic lax mode or explicit field converter),
            // use it regardless of strict mode
            converter_param
        } else if !strict {
            Type::any_explicit()
        } else if let Some(x) = descriptor
            && let Some(setter) = self.resolve_descriptor_setter(name, x, errors)
        {
            ClassField::get_descriptor_setter_value(&setter)
        } else {
            ty.clone()
        };
        let param_ty = param_type_transform(param_ty);
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

    pub fn as_enum_member(&self, field: ClassField, enum_cls: &Class) -> Option<Lit> {
        match field.0 {
            ClassFieldInner::ClassAttribute {
                ty: Type::Literal(mut lit),
                ..
            } if matches!(&lit, Lit::Enum(lit_enum) if lit_enum.class.class_object() == enum_cls) =>
            {
                let replacement = self.instantiate(enum_cls);
                lit.visit_mut(&mut |ty| ty.subst_self_type_mut(&replacement));
                Some(lit)
            }
            _ => None,
        }
    }

    fn is_typed_dict_field(&self, metadata: &ClassMetadata, field_name: &Name) -> bool {
        metadata
            .typed_dict_metadata()
            .is_some_and(|metadata| metadata.fields.contains_key(field_name))
    }

    fn typed_dict_field_info(
        &self,
        metadata: &ClassMetadata,
        field_name: &Name,
        field: &ClassField,
    ) -> Option<TypedDictField> {
        metadata
            .typed_dict_metadata()
            .and_then(|typed_dict| typed_dict.fields.get(field_name))
            .and_then(|is_total| field.clone().as_typed_dict_field_info(*is_total))
    }

    fn validate_typed_dict_field_override(
        &self,
        child_cls: &Class,
        child_metadata: &ClassMetadata,
        parent_cls: &Class,
        parent_metadata: &ClassMetadata,
        field_name: &Name,
        child_field: &ClassField,
        parent_field: &ClassField,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> bool {
        let Some(child_info) = self.typed_dict_field_info(child_metadata, field_name, child_field)
        else {
            return true;
        };
        let Some(parent_info) =
            self.typed_dict_field_info(parent_metadata, field_name, parent_field)
        else {
            return true;
        };

        let parent_mutable = !parent_info.is_read_only();
        let child_mutable = !child_info.is_read_only();

        if parent_mutable {
            if !child_mutable {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                    format!(
                        "TypedDict field `{field_name}` in `{}` cannot be marked read-only; parent TypedDict `{}` defines it as mutable",
                        child_cls.name(),
                        parent_cls.name()
                    ),
                );
                return false;
            }
            if parent_info.required && !child_info.required {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                    format!(
                        "TypedDict field `{field_name}` in `{}` must remain required because parent TypedDict `{}` defines it as required",
                        child_cls.name(),
                        parent_cls.name()
                    ),
                );
                return false;
            }
            if !parent_info.required && child_info.required {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                    format!(
                        "TypedDict field `{field_name}` in `{}` cannot be made required; parent TypedDict `{}` defines it as non-required",
                        child_cls.name(),
                        parent_cls.name()
                    ),
                );
                return false;
            }
        } else if parent_info.required && !child_info.required {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                format!(
                    "TypedDict field `{field_name}` in `{}` cannot be made non-required; parent TypedDict `{}` defines it as required",
                    child_cls.name(),
                    parent_cls.name()
                ),
            );
            return false;
        }

        true
    }

    fn should_check_field_for_override_consistency(
        &self,
        field_name: &Name,
        class_metadata: &Arc<ClassMetadata>,
    ) -> bool {
        // Object construction (`__new__`, `__init__`, `__init_subclass__`) should not participate in override checks
        if field_name == &dunder::NEW
            || field_name == &dunder::INIT
            || field_name == &dunder::INIT_SUBCLASS
        {
            return false;
        }

        // `__hash__` is often overridden to `None` to signal hashability
        if field_name == &dunder::HASH {
            return false;
        }

        // TODO(grievejia): In principle we should not really skip `__call__`. But the reality is that
        // there are too many classes on typeshed whose `__call__` are marked as follows:
        // ```
        // def __call__(self, *args: Any, **kwds: Any) -> Any: ...
        // ```
        // If we follow our pre-existing subtyping rule, this kind of signature would be non-overridable
        // -- any overrider must be able to take ANY arguments which can't be practical. We need to either
        // special-case typeshed or special-case callable subtyping to make `__call__` override check more usable.
        if field_name == &dunder::CALL {
            return false;
        }

        // Private attributes should not participate in override checks
        if Ast::is_mangled_attr(field_name) {
            return false;
        }

        // TODO: This should only be ignored when `cls` is a dataclass
        if field_name == &dunder::POST_INIT {
            return false;
        }

        // TODO: This should only be ignored when `_ignore_` is defined on enums
        if field_name.as_str() == "_ignore_" {
            return false;
        }

        // TODO: skipping slots for now to unblock typeshed upgrade
        if field_name == &dunder::SLOTS {
            return false;
        }

        // Django models: skip override check for `Meta` class
        if class_metadata.is_django_model() && field_name.as_str() == "Meta" {
            return false;
        }

        true
    }

    pub fn check_consistent_override_for_field(
        &self,
        cls: &Class,
        field_name: &Name,
        class_field: &ClassField,
        bases: &ClassBases,
        errors: &ErrorCollector,
    ) {
        let is_override = class_field.is_override();
        if matches!(class_field.1, IsInherited::No) && !is_override {
            return;
        }
        let metadata = self.get_metadata_for_class(cls);
        if !self.should_check_field_for_override_consistency(field_name, &metadata) {
            return;
        }

        let range = if let Some(range) = cls.field_decl_range(field_name) {
            range
        } else {
            return;
        };

        let mut got_attribute = None;
        let mut parent_attr_found = false;
        let mut parent_has_any = false;
        let is_typed_dict_field = self.is_typed_dict_field(metadata.as_ref(), field_name);

        let bases_to_check: Box<dyn Iterator<Item = &ClassType>> = if bases.is_empty() {
            // If the class doesn't have any base type, we should just use `object` as base to ensure
            // the inconsistent override check is not skipped
            Box::new(iter::once(self.stdlib.object()))
        } else {
            Box::new(bases.iter())
        };

        for parent in bases_to_check {
            let parent_cls = parent.class_object();
            let parent_metadata = self.get_metadata_for_class(parent_cls);
            parent_has_any = parent_has_any || parent_metadata.has_base_any();
            // Don't allow overriding a namedtuple element
            if let Some(named_tuple_metadata) = parent_metadata.named_tuple_metadata()
                && named_tuple_metadata.elements.contains(field_name)
            {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!("Cannot override named tuple element `{field_name}`"),
                );
            }
            let Some(want_field) = self.get_class_member(parent_cls, field_name) else {
                continue;
            };
            parent_attr_found = true;
            let want_class_field = Arc::unwrap_or_clone(want_field);
            if want_class_field.is_final() {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!(
                        "`{}` is declared as final in parent class `{}`",
                        field_name,
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
                                cls.name(),
                                field_name,
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
                                cls.name(),
                                field_name,
                                parent.name()
                            ),
                        );
                    continue;
                }
            }
            if is_typed_dict_field != self.is_typed_dict_field(&parent_metadata, field_name) {
                // TypedDict fields are actually dict keys, so we want to check them against other
                // keys but not regular fields.
                continue;
            }
            if is_typed_dict_field
                && !self.validate_typed_dict_field_override(
                    cls,
                    metadata.as_ref(),
                    parent_cls,
                    parent_metadata.as_ref(),
                    field_name,
                    class_field,
                    &want_class_field,
                    range,
                    errors,
                )
            {
                continue;
            }
            // Special case: if parent field is an unannotated `x = None`, allow child to override
            // with any type (effectively treating it as Optional[T])
            if !want_class_field.has_explicit_annotation()
                && matches!(want_class_field.ty(), Type::None)
            {
                continue;
            }
            // Substitute `Self` with derived class to support contravariant occurrences of `Self`
            let want_attribute = self.as_instance_attribute(
                field_name,
                &want_class_field,
                &Instance::of_protocol(parent, self.instantiate(cls)),
            );
            if got_attribute.is_none() {
                // Optimisation: Only compute the `got_attr` once, and only if we actually need it.
                got_attribute = Some(self.as_instance_attribute(
                    field_name,
                    class_field,
                    &Instance::of_class(&self.as_class_type_unchecked(cls)),
                ));
            }
            let attr_check = self.is_class_attribute_subset(
                got_attribute.as_ref().unwrap(),
                &want_attribute,
                &mut |got, want| self.is_subset_eq_with_reason(got, want),
            );
            let error = match attr_check {
                Err(
                    AttrSubsetError::Covariant {
                        subset_error: SubsetError::PosParamName(child, parent),
                        ..
                    }
                    | AttrSubsetError::Invariant {
                        subset_error: SubsetError::PosParamName(child, parent),
                        ..
                    }
                    | AttrSubsetError::Contravariant {
                        subset_error: SubsetError::PosParamName(child, parent),
                        ..
                    },
                ) => Some((
                    ErrorKind::BadParamNameOverride,
                    format!("Got parameter name `{child}`, expected `{parent}`"),
                )),
                Err(error) => Some((
                    ErrorKind::BadOverride,
                    error.to_error_msg(cls.name(), parent.name(), field_name),
                )),
                Ok(()) => None,
            };
            if let Some((kind, error)) = error {
                let msg = vec1![
                    format!(
                        "Class member `{}.{}` overrides parent class `{}` in an inconsistent manner",
                        cls.name(),
                        field_name,
                        parent.name()
                    ),
                    error,
                ];
                errors.add(range, ErrorInfo::Kind(kind), msg);
            }
        }
        if is_override && !parent_attr_found && !parent_has_any {
            self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadOverride),
                    format!(
                        "Class member `{}.{}` is marked as an override, but no parent class has a matching attribute",
                        cls.name(),
                        field_name,
                    ),
                );
        }
        // Check for missing @override decorator when overriding a parent attribute.
        // This error is emitted when a method overrides a parent but doesn't have @override.
        // Since this error has Severity::Ignore by default, it won't be shown unless enabled.
        if !is_override && parent_attr_found && !parent_has_any {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::MissingOverrideDecorator),
                format!(
                    "Class member `{}.{}` overrides a member in a parent class but is missing an `@override` decorator",
                    cls.name(),
                    field_name,
                ),
            );
        }
    }

    /// For classes with multiple inheritance, check that fields inherited from multiple base classes are consistent.
    pub fn check_consistent_multiple_inheritance(&self, cls: &Class, errors: &ErrorCollector) {
        struct InheritedFieldInfo {
            class: Class,
            metadata: Arc<ClassMetadata>,
            field: ClassField,
            ty: Type,
            read_only: bool,
        }

        let current_class_fields: SmallSet<_> = cls.fields().collect();
        let current_class_metadata = self.get_metadata_for_class(cls);
        let current_class_bases = self.get_base_types_for_class(cls);
        let swallow_access_errors = self.error_swallower();
        let mut inherited_fields: SmallMap<&Name, Vec<InheritedFieldInfo>> = SmallMap::new();

        for parent_class in current_class_bases.iter() {
            let parent_class_object = parent_class.class_object();
            let parent_class_fields = parent_class_object.fields();
            let parent_metadata = self.get_metadata_for_class(parent_class_object);
            for parent_field_name in parent_class_fields {
                if !self.should_check_field_for_override_consistency(
                    parent_field_name,
                    &current_class_metadata,
                ) {
                    continue;
                }
                if current_class_fields.contains(parent_field_name) {
                    continue;
                }
                if let Some(parent_field_arc) =
                    self.get_class_member(parent_class_object, parent_field_name)
                {
                    let parent_field = Arc::unwrap_or_clone(parent_field_arc.clone());
                    let class_attr = self.as_instance_attribute(
                        parent_field_name,
                        &parent_field,
                        &Instance::of_protocol(parent_class, self.instantiate(cls)),
                    );
                    let read_only = class_attr.is_read_only();
                    // For simple fields and methods, use their types directly
                    // For properties and descriptors, call their getter and use the result
                    let Ok(ty) = self.resolve_get_class_attr(
                        parent_field_name,
                        class_attr,
                        TextRange::default(),
                        &swallow_access_errors,
                        None,
                    ) else {
                        continue;
                    };
                    // If there is an access error or the result is an overload, skip it
                    if ty.is_error() {
                        continue;
                    }
                    inherited_fields.entry(parent_field_name).or_default().push(
                        InheritedFieldInfo {
                            class: parent_class_object.dupe(),
                            metadata: parent_metadata.clone(),
                            field: parent_field,
                            ty,
                            read_only,
                        },
                    );
                }
            }
        }

        for (field_name, inherited_field_infos_by_ancestor) in inherited_fields.iter() {
            if inherited_field_infos_by_ancestor.len() > 1 {
                let types: Vec<Type> = inherited_field_infos_by_ancestor
                    .iter()
                    .map(|info| info.ty.clone())
                    .collect();
                let intersect = self.intersects(&types);
                // Before intersecting `types`, `intersects` attempts to simplify to a single
                // element of `types` that is a common base type for all elements. If an
                // intersection is produced (or `Never` if the types are disjoint), then there was
                // no common base type, so the inheritance is inconsistent.
                if matches!(intersect, Type::Intersect(_) | Type::Never(_)) {
                    let mut error_msg = vec1![
                        format!(
                            "Field `{field_name}` has inconsistent types inherited from multiple base classes"
                        ),
                        "Inherited types include:".to_owned()
                    ];
                    for info in inherited_field_infos_by_ancestor.iter() {
                        error_msg.push(format!(
                            "  `{}` from `{}`",
                            self.for_display(info.ty.clone()),
                            info.class.name()
                        ));
                    }
                    errors.add(
                        cls.range(),
                        ErrorInfo::Kind(ErrorKind::InconsistentInheritance),
                        error_msg,
                    );
                } else {
                    for info in inherited_field_infos_by_ancestor {
                        // Read-write fields should check that parent field's type
                        // is assignable to the intersection.
                        // Skip function types for this check for now.
                        if !info.read_only
                            && !info.ty.is_toplevel_callable()
                            && !self.is_subset_eq(&info.ty, &intersect)
                        {
                            self.error(
                                errors,
                                cls.range(),
                                ErrorInfo::Kind(ErrorKind::InconsistentInheritance),
                                format!(
                                    "Field `{field_name}` is declared `{}` in ancestor `{}`, which is not assignable to the type `{}` implied by multiple inheritance",
                                    info.ty,
                                    info.class,
                                    intersect,
                                ),
                            );
                        }
                    }
                }

                if let Some(child_field_arc) = self.get_class_member(cls, field_name) {
                    let child_field = Arc::unwrap_or_clone(child_field_arc.clone());
                    if self
                        .typed_dict_field_info(
                            current_class_metadata.as_ref(),
                            field_name,
                            &child_field,
                        )
                        .is_some()
                    {
                        for info in inherited_field_infos_by_ancestor {
                            if self
                                .typed_dict_field_info(
                                    info.metadata.as_ref(),
                                    field_name,
                                    &info.field,
                                )
                                .is_some()
                                && !self.validate_typed_dict_field_override(
                                    cls,
                                    current_class_metadata.as_ref(),
                                    &info.class,
                                    info.metadata.as_ref(),
                                    field_name,
                                    &child_field,
                                    &info.field,
                                    cls.range(),
                                    errors,
                                )
                            {
                                continue;
                            }
                        }
                    }
                }
            }
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

    fn get_field_from_ancestors(
        &self,
        cls: &Class,
        mut ancestors: impl Iterator<Item = &'a ClassType>,
        name: &Name,
        get_field: &impl Fn(&Class, &Name) -> Option<Arc<ClassField>>,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        ancestors.find_map(|ancestor| {
            if ancestor.is_builtin("object")
                && [dunder::NEW, dunder::INIT].contains(name)
                && self.extends_any(cls)
            {
                // If this class has an `Any` ancestor, then we assume that `__new__` and `__init__`
                // can be called with any arguments. These attributes are special because they are
                // commonly overridden with incompatible type signatures. For most attributes, it's
                // more helpful to return the attribute type from `object` because it's unlikely to
                // have been changed by the unknown `Any` ancestor. Note that we put this check
                // right before `object` in the MRO because we know that `object` is always last.
                // While it would be safer to assume that the `Any` ancestor could appear first in
                // the MRO, we choose to instead return a more precise attribute type if we can find
                // one on a non-`Any` ancestor.
                Some(Arc::new(ClassField::new_synthesized(Type::any_implicit())))
            } else {
                get_field(ancestor.class_object(), name)
            }
            .map(|field| WithDefiningClass {
                value: Arc::new(
                    field.instantiate_helper(&mut |ty| ancestor.targs().substitute_into_mut(ty)),
                ),
                defining_class: ancestor.class_object().dupe(),
            })
        })
    }

    fn get_field_from_mro(
        &self,
        cls: &Class,
        name: &Name,
        get_field: &impl Fn(&Class, &Name) -> Option<Arc<ClassField>>,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        get_field(cls, name)
            .map(|field| WithDefiningClass {
                value: field,
                defining_class: cls.dupe(),
            })
            .or_else(|| {
                self.get_field_from_ancestors(
                    cls,
                    self.get_mro_for_class(cls).ancestors(self.stdlib),
                    name,
                    get_field,
                )
            })
    }

    /// Only look up fields that are not synthesized. This is useful when synthesizing method signatures
    /// for typeddict, named tuple, etc.
    pub fn get_non_synthesized_class_member(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        self.get_non_synthesized_class_member_and_defining_class(cls, name)
            .map(|x| x.value)
    }

    pub fn get_non_synthesized_class_member_and_defining_class(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        self.get_field_from_mro(cls, name, &|cls, name| {
            self.get_non_synthesized_field_from_current_class_only(cls, name)
                .filter(|field| !field.is_init_var())
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
        self.get_field_from_mro(cls, name, &|cls, name| {
            self.get_field_from_current_class_only(cls, name)
        })
    }

    fn get_non_synthesized_dataclass_member_impl(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        self.get_field_from_mro(cls, name, &|cls, name| {
            let field = self.get_non_synthesized_field_from_current_class_only(cls, name)?;
            if field.initialization() == ClassFieldInitialization::Method {
                // This parent happens to assign to the field in a method but doesn't define it.
                None
            } else {
                Some(field)
            }
        })
    }

    fn get_class_member_with_defining_class(
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

    pub(in crate::alt::class) fn get_class_member(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        self.get_class_member_with_defining_class(cls, name)
            .map(|member| member.value)
    }

    pub fn get_instance_attribute(&self, cls: &ClassType, name: &Name) -> Option<ClassAttribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|field| self.as_instance_attribute(name, &field, &Instance::of_class(cls)))
    }

    pub fn get_self_attribute(&self, cls: &ClassType, name: &Name) -> Option<ClassAttribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|field| self.as_instance_attribute(name, &field, &Instance::of_self_type(cls)))
    }

    pub fn get_protocol_attribute(
        &self,
        cls: &ClassType,
        self_type: Type,
        name: &Name,
    ) -> Option<ClassAttribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|field| {
                self.as_instance_attribute(name, &field, &Instance::of_protocol(cls, self_type))
            })
    }

    /// Return the first inherited method signature (parameters and flags) for `name`.
    pub(crate) fn inherited_method_signature(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<(ParamList, FuncFlags)> {
        let derived_instance = self.instantiate(cls);
        for ancestor in self.get_mro_for_class(cls).ancestors(self.stdlib) {
            let parent_cls = ancestor.class_object();
            let Some(member) = self.get_class_member(parent_cls, name) else {
                continue;
            };
            let instance = Instance::of_protocol(ancestor, derived_instance.clone());
            let instantiated = member.instantiate_for(&instance);
            if let Some(sig) = Self::callable_params_and_flags(instantiated.ty()) {
                return Some(sig);
            }
        }
        None
    }

    pub fn get_metaclass_attribute(
        &self,
        cls: &ClassBase,
        metaclass: &ClassType,
        name: &Name,
    ) -> Option<ClassAttribute> {
        let attr = self.get_class_member(metaclass.class_object(), name)?;
        let attr = self.as_instance_attribute(
            name,
            &attr,
            &Instance::of_metaclass(cls.clone(), metaclass),
        );
        if name == &dunder::CALL
            && metaclass
                .class_object()
                .has_toplevel_qname(ModuleName::builtins().as_str(), "type")
        {
            return Some(ClassAttribute::read_write(
                self.constructor_to_callable(cls.class_type()),
            ));
        }
        Some(attr)
    }

    // When we're accessing the attribute of a string literal, we bind methods to
    // `LiteralString` instead of `str`, so that overload selection works correctly
    // for `LiteralString`-specific overloads defined in `str`.
    pub fn get_literal_string_attribute(&self, name: &Name) -> Option<ClassAttribute> {
        self.get_class_member(self.stdlib.str().class_object(), name)
            .map(|field| {
                self.as_instance_attribute(name, &field, &Instance::literal_string(self.stdlib))
            })
    }

    pub fn get_bounded_quantified_attribute(
        &self,
        quantified: Quantified,
        upper_bound: &ClassType,
        name: &Name,
    ) -> Option<ClassAttribute> {
        let quantified_with_specific_upper_bound = match quantified.restriction() {
            Restriction::Constraints(_) => {
                quantified.with_restriction(Restriction::Constraints(vec![
                    upper_bound.clone().to_type(),
                ]))
            }
            Restriction::Bound(_) => {
                quantified.with_restriction(Restriction::Bound(upper_bound.clone().to_type()))
            }
            Restriction::Unrestricted => quantified,
        };
        self.get_class_member(upper_bound.class_object(), name)
            .map(|field| {
                self.as_instance_attribute(
                    name,
                    &field,
                    &Instance::of_type_var(quantified_with_specific_upper_bound, upper_bound),
                )
            })
    }

    pub fn get_typed_dict_attribute(
        &self,
        td: &TypedDictInner,
        name: &Name,
    ) -> Option<ClassAttribute> {
        if let Some(meta) = self
            .get_metadata_for_class(td.class_object())
            .typed_dict_metadata()
            && meta.fields.contains_key(name)
        {
            // TypedDict fields are dictionary key declarations, not real fields.
            return None;
        }
        self.get_class_member(td.class_object(), name)
            .map(|field| self.as_instance_attribute(name, &field, &Instance::of_typed_dict(td)))
    }

    pub fn get_super_class_member(
        &self,
        cls: &Class,
        start_lookup_cls: Option<&ClassType>,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        // Skip ancestors in the MRO until we find the class we want to start at
        let metadata = self.get_mro_for_class(cls);
        let ancestors = metadata.ancestors(self.stdlib).skip_while(|ancestor| {
            if let Some(start_lookup_cls) = start_lookup_cls {
                *ancestor != start_lookup_cls
            } else {
                false
            }
        });
        self.get_field_from_ancestors(cls, ancestors, name, &|cls, name| {
            self.get_field_from_current_class_only(cls, name)
                .filter(|field| !field.is_init_var())
        })
    }

    /// Looks up an attribute on a super instance.
    pub fn get_super_attribute(
        &self,
        start_lookup_cls: &ClassType,
        super_obj: &SuperObj,
        name: &Name,
    ) -> Option<ClassAttribute> {
        match super_obj {
            SuperObj::Instance(obj) => self
                .get_super_class_member(obj.class_object(), Some(start_lookup_cls), name)
                .map(|member| {
                    if let Some(reason) = self.super_method_needs_impl_reason(&member) {
                        ClassAttribute::no_access(reason)
                    } else {
                        self.as_instance_attribute(
                            name,
                            &member.value,
                            &Instance::of_self_type(obj),
                        )
                    }
                }),
            SuperObj::Class(obj) => self
                .get_super_class_member(obj.class_object(), Some(start_lookup_cls), name)
                .map(|member| {
                    if let Some(reason) = self.super_method_needs_impl_reason(&member) {
                        ClassAttribute::no_access(reason)
                    } else {
                        self.as_class_attribute(
                            name,
                            &member.value,
                            &ClassBase::SelfType(obj.clone()),
                        )
                    }
                }),
        }
    }

    fn super_method_needs_impl_reason(
        &self,
        member: &WithDefiningClass<Arc<ClassField>>,
    ) -> Option<NoAccessReason> {
        if member.value.is_abstract() {
            return Some(NoAccessReason::SuperMethodNeedsImplementation(
                member.defining_class.dupe(),
            ));
        }
        let metadata = self.get_metadata_for_class(&member.defining_class);
        if metadata.is_protocol() && member.value.is_non_callable_protocol_method() {
            Some(NoAccessReason::SuperMethodNeedsImplementation(
                member.defining_class.dupe(),
            ))
        } else {
            None
        }
    }

    /// Gets an attribute from a class definition.
    ///
    /// Returns `None` if there is no such attribute, otherwise an `Attribute` object
    /// that describes whether access is allowed and the type if so.
    ///
    /// Access is disallowed for instance-only attributes and for attributes whose
    /// type contains a class-scoped type parameter - e.g., `class A[T]: x: T`.
    pub fn get_class_attribute(&self, cls: &ClassBase, name: &Name) -> Option<ClassAttribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|field| self.as_class_attribute(name, &field, cls))
    }

    pub fn get_bounded_quantified_class_attribute(
        &self,
        quantified: Quantified,
        class: &ClassType,
        name: &Name,
    ) -> Option<ClassAttribute> {
        self.get_class_member(class.class_object(), name)
            .map(|field| {
                self.as_class_attribute(
                    name,
                    &field,
                    &ClassBase::Quantified(quantified, class.clone()),
                )
            })
    }

    pub fn field_is_inherited_from(
        &self,
        cls: &Class,
        field: &Name,
        ancestor: (&str, &str),
    ) -> bool {
        let member = self.get_class_member_with_defining_class(cls, field);
        match member {
            Some(member) => member.is_defined_on(ancestor.0, ancestor.1),
            None => false,
        }
    }

    /// Get the class's `__new__` method.
    ///
    /// This lookup skips normal method binding logic (it behaves like a cross
    /// between a classmethod and a constructor; downstream code handles this
    /// using the raw callable type).
    pub fn get_dunder_new(&self, cls: &ClassType) -> Option<Type> {
        let new_member =
            self.get_class_member_with_defining_class(cls.class_object(), &dunder::NEW)?;
        if new_member.is_defined_on("builtins", "object") {
            // The default behavior of `object.__new__` is already baked into our implementation of
            // class construction; we only care about `__new__` if it is overridden.
            None
        } else {
            Arc::unwrap_or_clone(new_member.value)
                .as_raw_special_method_type(&Instance::of_class(cls))
        }
    }

    fn get_dunder_init_helper(&self, instance: &Instance, get_object_init: bool) -> Option<Type> {
        let init_method =
            self.get_class_member_with_defining_class(instance.class, &dunder::INIT)?;
        if get_object_init || !init_method.is_defined_on("builtins", "object") {
            Arc::unwrap_or_clone(init_method.value).as_special_method_type(instance)
        } else {
            None
        }
    }

    /// Get the class's `__init__` method. The second argument controls whether we return an inherited `object.__init__`.
    pub fn get_dunder_init(&self, cls: &ClassType, get_object_init: bool) -> Option<Type> {
        self.get_dunder_init_helper(&Instance::of_class(cls), get_object_init)
    }

    pub fn get_typed_dict_dunder_init(&self, td: &TypedDictInner) -> Type {
        // We synthesize `__init__`, so the lookup will never entirely fail.
        //
        // But if the user defined an `__init__` directly on the typed dict - which is always
        // a type error in the user code but can still occur - then it's possible the lookup
        // will fail because we find a class field but cannot resolve it to a special method type.
        //
        // TODO(stroxler): We probably should rethink this, for many reasons:
        // - There's no such thing as `__init__`, typed dicts use `dict.__new__` so we're
        //   modeling the runtime poorly
        // - There's no reason to use `get_dunder_init_helper` here if we know
        //   it's a synthesized field - too many layers of things that can go wrong.
        // - Even if the user overrides `__new__` in a typed dict (e.g. with a field named
        //   `__new__`), under at least some circumstances constructor calls *still* work
        //   because `__new__` is likely only part of `__annotations__`
        self.get_dunder_init_helper(&Instance::of_typed_dict(td), true)
            .unwrap_or(Type::any_error())
    }

    /// Get the metaclass `__call__` method
    pub fn get_metaclass_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        let metadata = self.get_metadata_for_class(cls.class_object());
        let metaclass = metadata.custom_metaclass()?;
        let attr =
            self.get_class_member_with_defining_class(metaclass.class_object(), &dunder::CALL)?;
        if attr.is_defined_on("builtins", "type") {
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
                .as_raw_special_method_type(&Instance::of_metaclass(
                    ClassBase::ClassType(cls.clone()),
                    metaclass,
                ))
                .and_then(|ty| make_bound_method(Type::type_form(cls.clone().to_type()), ty).ok())
        }
    }

    pub fn resolve_named_tuple_element(&self, cls: ClassType, name: &Name) -> Option<Type> {
        let field = self.get_class_member(cls.class_object(), name)?;
        match field.instantiate_for(&Instance::of_class(&cls)).0 {
            ClassFieldInner::ClassAttribute {
                ty,
                read_only_reason: Some(_),
                ..
            } => Some(ty),
            ClassFieldInner::InstanceAttribute {
                ty,
                read_only_reason: Some(_),
                ..
            } => Some(ty),
            _ => None,
        }
    }

    fn check_dataclass_field_final(
        &self,
        cls: &Class,
        attr_name: &Name,
        errors: &ErrorCollector,
        range: TextRange,
    ) {
        if let DataclassMember::Field(field, _) = self.get_dataclass_member(cls, attr_name) {
            let field = field.value;
            if field.is_final() {
                let msg = vec1![
                    format!("Cannot set field `{attr_name}`"),
                    ReadOnlyReason::Final(IsFinalVariableInitialized::No).error_message()
                ];
                errors.add(range, ErrorInfo::Kind(ErrorKind::ReadOnly), msg);
            }
        }
    }

    pub fn check_class_attr_set_and_infer_narrow(
        &self,
        class_attr: ClassAttribute,
        instance_class: Option<&ClassType>,
        class_base: Option<&ClassBase>,
        attr_name: &Name,
        got: TypeOrExpr,
        allow_assign_to_final: bool,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        should_narrow: &mut bool,
        narrowed_types: &mut Vec<Type>,
    ) {
        match class_attr {
            ClassAttribute::NoAccess(e) => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::NoAccess, context),
                    e.to_error_msg(attr_name),
                );
                *should_narrow = false;
            }
            ClassAttribute::ReadOnly(attr_ty, reason) => {
                // In pydantic, if a non-frozen model inherits from a frozen model,
                // attributes of the frozen model are no longer readonly.
                let should_raise_error = if let Some(instance_class) = instance_class {
                    let class = instance_class.class_object();
                    let metadata = self.get_metadata_for_class(class);
                    !(metadata.is_pydantic_base_model()
                        && metadata
                            .dataclass_metadata()
                            .is_some_and(|dm| !dm.kws.frozen))
                } else {
                    true
                };

                if allow_assign_to_final
                    && matches!(
                        reason,
                        ReadOnlyReason::Final(IsFinalVariableInitialized::No)
                    )
                {
                    self.check_set_read_write_and_infer_narrow(
                        attr_ty,
                        attr_name,
                        got,
                        range,
                        errors,
                        context,
                        *should_narrow,
                        narrowed_types,
                    );
                } else if should_raise_error {
                    let msg = vec1![
                        format!("Cannot set field `{attr_name}`"),
                        reason.error_message()
                    ];
                    errors.add(range, ErrorInfo::Kind(ErrorKind::ReadOnly), msg);
                    *should_narrow = false;
                }
            }
            ClassAttribute::ReadWrite(attr_ty) => {
                // If the attribute has a converter, then `want` should be the type expected by the converter.
                let attr_ty = match instance_class {
                    Some(cls) => match self.get_dataclass_member(cls.class_object(), attr_name) {
                        DataclassMember::Field(_, kws) => kws.converter_param.unwrap_or(attr_ty),
                        _ => attr_ty,
                    },
                    _ => attr_ty,
                };
                // Obtain a class object for checking whether we're writing to a final field in dataclass
                let class_object = instance_class
                    .map(|cls| cls.class_object())
                    .or(class_base.map(|cb| cb.class_object()));
                if let Some(class_object) = class_object {
                    self.check_dataclass_field_final(class_object, attr_name, errors, range);
                }
                self.check_set_read_write_and_infer_narrow(
                    attr_ty,
                    attr_name,
                    got,
                    range,
                    errors,
                    context,
                    *should_narrow,
                    narrowed_types,
                );
            }
            ClassAttribute::Property(getter, None, cls) => {
                let is_cached_property = getter.is_cached_property();
                if is_cached_property {
                    let attr_ty = self.call_property_getter(getter, range, errors, context);
                    self.check_set_read_write_and_infer_narrow(
                        attr_ty,
                        attr_name,
                        got,
                        range,
                        errors,
                        context,
                        *should_narrow,
                        narrowed_types,
                    );
                    *should_narrow = false;
                } else {
                    let e = NoAccessReason::SettingReadOnlyProperty(cls);
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::ReadOnly, context),
                        e.to_error_msg(attr_name),
                    );
                    *should_narrow = false;
                }
            }
            ClassAttribute::Property(_, Some(setter), _) => {
                let got = CallArg::arg(got);
                self.call_property_setter(setter, got, range, errors, context);
                *should_narrow = false;
            }
            ClassAttribute::Descriptor(x, base) => {
                match base {
                    DescriptorBase::Instance(class_type)
                        if let Some(setter) =
                            self.resolve_descriptor_setter(attr_name, &x, errors) =>
                    {
                        let got = CallArg::arg(got);
                        self.call_descriptor_setter(
                            setter, class_type, got, range, errors, context,
                        );
                    }
                    DescriptorBase::Instance(class_type) => {
                        let e = NoAccessReason::SettingReadOnlyDescriptor(
                            class_type.class_object().dupe(),
                        );
                        self.error(
                            errors,
                            range,
                            ErrorInfo::new(ErrorKind::ReadOnly, context),
                            e.to_error_msg(attr_name),
                        );
                    }
                    DescriptorBase::ClassDef(class) => {
                        let e = NoAccessReason::SettingDescriptorOnClass(class.dupe());
                        self.error(
                            errors,
                            range,
                            ErrorInfo::new(ErrorKind::NoAccess, context),
                            e.to_error_msg(attr_name),
                        );
                    }
                };
                *should_narrow = false;
            }
        }
    }

    pub fn check_class_attr_delete(
        &self,
        class_attr: ClassAttribute,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        match class_attr {
            ClassAttribute::NoAccess(reason) => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::NoAccess, context),
                    reason.to_error_msg(attr_name),
                );
            }
            ClassAttribute::ReadOnly(_, reason) => {
                let msg = vec1![
                    format!("Cannot delete field `{attr_name}`"),
                    reason.error_message()
                ];
                errors.add(range, ErrorInfo::Kind(ErrorKind::ReadOnly), msg);
            }
            ClassAttribute::ReadWrite(..)
            | ClassAttribute::Property(..)
            | ClassAttribute::Descriptor(..) => {
                // Allow deleting most attributes for now, for compatibility with mypy.
            }
        }
    }

    pub fn is_class_attribute_subset(
        &self,
        got: &ClassAttribute,
        want: &ClassAttribute,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), AttrSubsetError> {
        match (got, want) {
            (_, ClassAttribute::NoAccess(_)) => Ok(()),
            (ClassAttribute::NoAccess(_), _) => Err(AttrSubsetError::NoAccess),
            (
                ClassAttribute::Property(_, _, _),
                ClassAttribute::ReadOnly(..) | ClassAttribute::ReadWrite(..),
            ) => Err(AttrSubsetError::Property),
            (
                ClassAttribute::ReadOnly(..),
                ClassAttribute::Property(_, Some(_), _) | ClassAttribute::ReadWrite(_),
            ) => Err(AttrSubsetError::ReadOnly),
            (
                // TODO(stroxler): Investigate this case more: methods should be ReadOnly, but
                // in some cases for unknown reasons they wind up being ReadWrite.
                ClassAttribute::ReadWrite(got),
                ClassAttribute::ReadWrite(want),
            ) if got.has_toplevel_func_metadata() && want.has_toplevel_func_metadata() => {
                is_subset(got, want).map_err(|subset_error| AttrSubsetError::Covariant {
                    got: got.clone(),
                    want: want.clone(),
                    got_is_property: false,
                    want_is_property: false,
                    subset_error,
                })
            }
            (ClassAttribute::ReadWrite(got), ClassAttribute::ReadWrite(want)) => {
                let subset_error = is_subset(got, want)
                    .map_or_else(Some, |_| is_subset(want, got).map_or_else(Some, |_| None));
                if let Some(subset_error) = subset_error {
                    Err(AttrSubsetError::Invariant {
                        got: got.clone(),
                        want: want.clone(),
                        subset_error,
                    })
                } else {
                    Ok(())
                }
            }
            (
                ClassAttribute::ReadWrite(got) | ClassAttribute::ReadOnly(got, ..),
                ClassAttribute::ReadOnly(want, _),
            ) => is_subset(got, want).map_err(|subset_error| AttrSubsetError::Covariant {
                got: got.clone(),
                want: want.clone(),
                got_is_property: false,
                want_is_property: false,
                subset_error,
            }),
            (ClassAttribute::ReadOnly(got, _), ClassAttribute::Property(want, _, _)) => {
                is_subset(
                    // Synthesize a getter method
                    &Type::callable_ellipsis(got.clone()),
                    want,
                )
                .map_err(|subset_error| AttrSubsetError::Covariant {
                    got: got.clone(),
                    want: want.clone(),
                    got_is_property: false,
                    want_is_property: true,
                    subset_error,
                })
            }
            (ClassAttribute::ReadWrite(got), ClassAttribute::Property(want, want_setter, _)) => {
                is_subset(
                    // Synthesize a getter method
                    &Type::callable_ellipsis(got.clone()),
                    want,
                )
                .map_err(|subset_error| AttrSubsetError::Covariant {
                    got: got.clone(),
                    want: want.clone(),
                    got_is_property: false,
                    want_is_property: true,
                    subset_error,
                })?;
                if let Some(want_setter) = want_setter {
                    // Synthesize a setter method
                    is_subset(
                        want_setter,
                        &Type::callable(
                            vec![Param::PosOnly(None, got.clone(), Required::Required)],
                            Type::None,
                        ),
                    )
                    .map_err(|subset_error| AttrSubsetError::Contravariant {
                        want: want_setter.clone(),
                        got: got.clone(),
                        got_is_property: true,
                        subset_error,
                    })
                } else {
                    Ok(())
                }
            }
            (
                ClassAttribute::Property(got_getter, got_setter, _),
                ClassAttribute::Property(want_getter, want_setter, _),
            ) => {
                is_subset(got_getter, want_getter).map_err(|subset_error| {
                    AttrSubsetError::Covariant {
                        got: got_getter.clone(),
                        want: want_getter.clone(),
                        got_is_property: true,
                        want_is_property: true,
                        subset_error,
                    }
                })?;
                match (got_setter, want_setter) {
                    (Some(got_setter), Some(want_setter)) => is_subset(got_setter, want_setter)
                        .map_err(|subset_error| AttrSubsetError::Contravariant {
                            want: want_setter.clone(),
                            got: got_setter.clone(),
                            got_is_property: true,
                            subset_error,
                        }),
                    (None, Some(_)) => Err(AttrSubsetError::ReadOnly),
                    (_, None) => Ok(()),
                }
            }
            (
                ClassAttribute::Descriptor(Descriptor { cls: got_cls, .. }, ..),
                ClassAttribute::Descriptor(Descriptor { cls: want_cls, .. }, ..),
            ) => {
                let got_ty = got_cls.clone().to_type();
                let want_ty = want_cls.clone().to_type();
                is_subset(&got_ty, &want_ty).map_err(|subset_error| AttrSubsetError::Covariant {
                    got: got_ty,
                    want: want_ty,
                    got_is_property: false,
                    want_is_property: false,
                    subset_error,
                })
            }
            (ClassAttribute::Descriptor(..), _) | (_, ClassAttribute::Descriptor(..)) => {
                Err(AttrSubsetError::Descriptor)
            }
        }
    }

    pub fn resolve_get_class_attr(
        &self,
        attr_name: &Name,
        class_attr: ClassAttribute,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Result<Type, NoAccessReason> {
        match class_attr {
            ClassAttribute::NoAccess(reason) => Err(reason),
            ClassAttribute::ReadWrite(ty) | ClassAttribute::ReadOnly(ty, _) => Ok(ty),
            ClassAttribute::Property(getter, ..) => {
                self.record_property_getter(range, &getter);
                Ok(self.call_property_getter(getter, range, errors, context))
            }
            ClassAttribute::Descriptor(x, base) => {
                if let Some(getter) = self.resolve_descriptor_getter(attr_name, &x, errors) {
                    // Reading a descriptor with a getter resolves to a method call
                    //
                    // TODO(stroxler): Once we have more complex error traces, it would be good to pass
                    // context down so that errors inside the call can mention that it was a descriptor read.
                    Ok(self.call_descriptor_getter(getter, base, range, errors, context))
                } else {
                    // Reading descriptor with no getter resolves to the descriptor itself
                    Ok(x.cls.to_type())
                }
            }
        }
    }

    fn resolve_descriptor_getter(
        &self,
        attr_name: &Name,
        x: &Descriptor,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if x.getter
            && let Some(getter) = self.get_class_member(x.cls.class_object(), &dunder::GET)
        {
            let attr =
                self.as_instance_attribute(&dunder::GET, &getter, &Instance::of_class(&x.cls));
            Some(
                self.resolve_get_class_attr(attr_name, attr, x.range, errors, None)
                    .unwrap_or_else(|e| {
                        self.error(
                            errors,
                            x.range,
                            ErrorInfo::new(ErrorKind::NoAccess, None),
                            e.to_error_msg(&dunder::GET),
                        )
                    }),
            )
        } else {
            None
        }
    }

    fn resolve_descriptor_setter(
        &self,
        attr_name: &Name,
        x: &Descriptor,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if x.setter
            && let Some(setter) = self.get_class_member(x.cls.class_object(), &dunder::SET)
        {
            let attr =
                self.as_instance_attribute(&dunder::SET, &setter, &Instance::of_class(&x.cls));
            Some(
                self.resolve_get_class_attr(attr_name, attr, x.range, errors, None)
                    .unwrap_or_else(|e| {
                        self.error(
                            errors,
                            x.range,
                            ErrorInfo::new(ErrorKind::NoAccess, None),
                            e.to_error_msg(&dunder::SET),
                        )
                    }),
            )
        } else {
            None
        }
    }

    /// Return `__call__` as a bound method if instances of `cls` have `__call__`.
    /// This is what the runtime automatically does when we try to call an instance.
    pub fn instance_as_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        self.get_instance_attribute(cls, &dunder::CALL)
            .and_then(|attr| attr.as_instance_method())
    }

    /// Return `__call__` as bound method when called on `Self`.
    pub fn self_as_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        self.get_self_attribute(cls, &dunder::CALL)
            .and_then(|attr| attr.as_instance_method())
    }

    /// Return `__call__` as a bound method if instances of `type_var` have `__call__`.
    /// We look up `__call__` from the upper bound of `type_var`, but `Self` is substituted with
    /// the `type_var` instead of the upper bound class.
    pub fn quantified_instance_as_dunder_call(
        &self,
        quantified: Quantified,
        upper_bound: &ClassType,
    ) -> Option<Type> {
        self.get_bounded_quantified_attribute(quantified, upper_bound, &dunder::CALL)
            .and_then(|attr| attr.as_instance_method())
    }

    fn callable_params_and_flags(mut ty: Type) -> Option<(ParamList, FuncFlags)> {
        let mut flags = None;
        ty.transform_toplevel_func_metadata(|meta| {
            if flags.is_none() {
                flags = Some(meta.flags.clone());
            }
        });
        let flags = flags?;
        let params = match ty.callable_signatures().as_slice() {
            [sig] if let Params::List(list) = &sig.params => Some(list.clone()),
            _ => None,
        }?;
        Some((params, flags))
    }
}
