/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;

use dupe::Dupe;
use itertools::Either;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::callable::CallArg;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyExport;
use crate::dunder;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::kind::ErrorKind;
use crate::export::exports::Exports;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::module::Module;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::tuple::Tuple;
use crate::types::type_var::Restriction;
use crate::types::typed_dict::TypedDict;
use crate::types::types::AnyStyle;
use crate::types::types::Overload;
use crate::types::types::SuperObj;
use crate::types::types::Type;

#[derive(Debug)]
enum LookupResult {
    /// The lookup succeeded, resulting in a type.
    Found(Attribute),
    /// The attribute was not found. Callers can use fallback behavior, for
    /// example looking up a different attribute.
    NotFound(NotFound),
    /// There was a Pyre-internal error
    InternalError(InternalError),
}

/// The result of a read for narrowing purposes. We track whether we are narrowing
/// a property or descriptor that might not be idempotent, and if so we also
/// indicate when this came through a union (so that error messages can be clearer).
#[derive(Debug)]
pub enum Narrowable {
    Simple(Type),
    PropertyOrDescriptor(Type),
    UnionPropertyOrDescriptor(Type),
}

/// The result of looking up an attribute. We can analyze get and set actions
/// on an attribute, each of which can be allowed with some type or disallowed.
#[derive(Debug)]
pub struct Attribute {
    inner: AttributeInner,
}

#[derive(Debug)]
enum Visibility {
    ReadOnly,
    ReadWrite,
}

/// The result of an attempt to access an attribute (with a get or set operation).
///
/// The operation is either permitted with an attribute `Type`, or is not allowed
/// and has a reason.
#[derive(Debug)]
enum AttributeInner {
    /// A `NoAccess` attribute indicates that the attribute is well-defined, but does
    /// not allow the access pattern (for example class access on an instance-only attribute)
    NoAccess(NoAccessReason),
    /// A read-write attribute with a closed form type for both get and set actions.
    Simple(Type, Visibility),
    /// A property is a special attribute were regular access invokes a getter.
    /// It optionally might have a setter method; if not, trying to set it is an access error
    Property(Type, Option<Type>, Class),
    /// A descriptor is a user-defined type whose actions may dispatch to special method calls
    /// for the get and set actions.
    Descriptor(Descriptor),
    /// The attribute being looked up is not defined explicitly, but it may be defined via a
    /// __getattr__ fallback.
    /// The `NotFound` field stores the (failed) lookup result on the original attribute for
    /// better error reporting downstream. The `AttributeInner` field stores the (successful)
    /// lookup result of the `__getattr__` function or method. The `Name` field stores the name
    /// of the original attribute being looked up.
    GetAttr(NotFound, Box<AttributeInner>, Name),
}

#[derive(Clone, Debug)]
struct Descriptor {
    /// This is the raw type of the descriptor, which is needed both for attribute subtyping
    /// checks in structural types and in the case where there is no getter method.
    descriptor_ty: Type,
    /// Descriptor behavior depends on the base against which the attribute is resolved, so
    /// we have to preserve information about whether it is a class instance or class def.
    base: DescriptorBase,
    /// If `__get__` exists on the descriptor, this is the type of `__get__`
    /// method type (as resolved by accessing it on an instance of the
    /// descriptor). It is typically a `BoundMethod` although it is possible for
    /// a user to erroneously define a `__get__` with any type, including a
    /// non-callable one.
    getter: Option<Type>,
    /// If `__set__` exists on the descriptor, this is the type of `__set__`. Similar considerations
    /// to `getter` apply.
    setter: Option<Type>,
}

#[derive(Clone, Debug)]
pub enum DescriptorBase {
    Instance(ClassType),
    ClassDef(Class),
}

#[derive(Debug)]
pub enum NotFound {
    Attribute(Class),
    ClassAttribute(Class),
    ModuleExport(Module),
}

#[derive(Clone, Debug)]
pub enum NoAccessReason {
    /// The attribute is only initialized on instances, but we saw an attempt
    /// to use it as a class attribute.
    ClassUseOfInstanceAttribute(Class),
    /// A generic class attribute exists, but has an invalid definition.
    /// Callers should treat the attribute as `Any`.
    ClassAttributeIsGeneric(Class),
    /// A set operation on a read-only property is an access error.
    SettingReadOnlyProperty(Class),
    /// A descriptor that only has `__get__` should be treated as read-only on instances.
    SettingReadOnlyDescriptor(Class),
    /// We do not allow class-level mutation of descriptors (this is conservative,
    /// it is unspecified whether monkey-patching descriptors should be permitted).
    SettingDescriptorOnClass(Class),
}

#[derive(Debug)]
enum InternalError {
    /// An internal error caused by `as_attribute_base` being partial.
    AttributeBaseUndefined(Type),
}

impl Attribute {
    fn new(inner: AttributeInner) -> Self {
        Attribute { inner }
    }

    pub fn no_access(reason: NoAccessReason) -> Self {
        Attribute {
            inner: AttributeInner::NoAccess(reason),
        }
    }

    pub fn read_write(ty: Type) -> Self {
        Attribute {
            inner: AttributeInner::Simple(ty, Visibility::ReadWrite),
        }
    }

    pub fn read_only(ty: Type) -> Self {
        Attribute {
            inner: AttributeInner::Simple(ty, Visibility::ReadOnly),
        }
    }

    pub fn property(getter: Type, setter: Option<Type>, cls: Class) -> Self {
        Attribute {
            inner: AttributeInner::Property(getter, setter, cls),
        }
    }

    pub fn descriptor(
        ty: Type,
        base: DescriptorBase,
        getter: Option<Type>,
        setter: Option<Type>,
    ) -> Self {
        Attribute {
            inner: AttributeInner::Descriptor(Descriptor {
                descriptor_ty: ty,
                base,
                getter,
                setter,
            }),
        }
    }

    pub fn getattr(not_found: NotFound, getattr: Attribute, name: Name) -> Self {
        Attribute {
            inner: AttributeInner::GetAttr(not_found, Box::new(getattr.inner), name),
        }
    }
}

impl NoAccessReason {
    pub fn to_error_msg(&self, attr_name: &Name) -> String {
        match self {
            NoAccessReason::ClassUseOfInstanceAttribute(class) => {
                let class_name = class.name();
                format!(
                    "Instance-only attribute `{attr_name}` of class `{class_name}` is not visible on the class"
                )
            }
            NoAccessReason::ClassAttributeIsGeneric(class) => {
                let class_name = class.name();
                format!(
                    "Generic attribute `{attr_name}` of class `{class_name}` is not visible on the class"
                )
            }
            NoAccessReason::SettingReadOnlyProperty(class) => {
                let class_name = class.name();
                format!(
                    "Attribute `{attr_name}` of class `{class_name}` is a read-only property and cannot be set"
                )
            }
            NoAccessReason::SettingDescriptorOnClass(class) => {
                let class_name = class.name();
                format!(
                    "Attribute `{attr_name}` of class `{class_name}` is a descriptor, which may not be overwritten"
                )
            }
            NoAccessReason::SettingReadOnlyDescriptor(class) => {
                let class_name = class.name();
                format!(
                    "Attribute `{attr_name}` of class `{class_name}` is a read-only descriptor with no `__set__` and cannot be set"
                )
            }
        }
    }
}

impl LookupResult {
    /// We found a simple attribute type.
    ///
    /// This means we assume it is both readable and writable with that type.
    ///
    /// TODO(stroxler) The uses of this eventually need to be audited, but we
    /// need to prioiritize the class logic first.
    fn found_type(ty: Type) -> Self {
        Self::Found(Attribute::read_write(ty))
    }
}

impl NotFound {
    pub fn to_error_msg(self, attr_name: &Name) -> String {
        match self {
            NotFound::Attribute(class) => {
                let class_name = class.name();
                format!("Object of class `{class_name}` has no attribute `{attr_name}`",)
            }
            NotFound::ClassAttribute(class) => {
                let class_name = class.name();
                format!("Class `{class_name}` has no class attribute `{attr_name}`")
            }
            NotFound::ModuleExport(module) => {
                format!("No attribute `{attr_name}` in module `{module}`")
            }
        }
    }
}

impl InternalError {
    pub fn to_error_msg(self, attr_name: &Name, todo_ctx: &str) -> String {
        match self {
            InternalError::AttributeBaseUndefined(ty) => format!(
                "TODO: {todo_ctx} attribute base undefined for type: {} (trying to access {})",
                ty.deterministic_printing(),
                attr_name
            ),
        }
    }
}

/// A normalized type representation which is convenient for attribute lookup,
/// since many cases are collapsed. For example, Type::Literal is converted to
/// it's corresponding class type.
#[derive(Clone, Debug)]
enum AttributeBase {
    EnumLiteral(ClassType, Name, Type),
    ClassInstance(ClassType),
    ClassObject(Class),
    Module(Module),
    /// The attribute access is on a quantified type form (as in `args: P.args` - this
    /// is only used when the base *is* a quantified type, not when the base is
    /// a term that *has* a quantified type.
    /// The second element is a bound or constraint for the type variable.
    TypeVar(Quantified, Option<ClassType>),
    Any(AnyStyle),
    Never,
    /// type[Any] is a special case where attribute lookups first check the
    /// builtin `type` class before falling back to `Any`.
    TypeAny(AnyStyle),
    /// Properties are handled via a special case so that we can understand
    /// setter decorators.
    Property(Type),
    /// Result of a super() call. See Type::SuperInstance for details on what these fields are.
    SuperInstance(ClassType, SuperObj),
    /// Typed dictionaries have similar properties to dict and Mapping, with some exceptions
    TypedDict(TypedDict),
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Gets the possible attribute bases for a type:
    /// If the type is a union, we will attempt to generate bases for each member of the union
    /// If the type is a bounded type var w/ a union upper bound, we will attempt to generate 1 base for
    /// each member of the union
    /// If the type is a constrained type var, we will attempt to generate 1 base for each constraint
    fn get_possible_attribute_bases(&self, base: &Type) -> Vec<Option<AttributeBase>> {
        let mut bases = Vec::new();
        self.map_over_union(base, |base| {
            match base {
                Type::Quantified(quantified) => match quantified.restriction() {
                    Restriction::Bound(upper_bound) => {
                        let mut use_fallback = false;
                        self.map_over_union(upper_bound, |bound| {
                            let bound_attr_base = self.as_attribute_base_no_union(bound.clone());
                            if let Some(AttributeBase::ClassInstance(cls)) = bound_attr_base {
                                bases.push(Some(AttributeBase::TypeVar(
                                    quantified.clone(),
                                    Some(cls),
                                )));
                            } else {
                                use_fallback = true;
                            }
                        });
                        if use_fallback {
                            bases.push(Some(AttributeBase::TypeVar(quantified.clone(), None)));
                        }
                    }
                    Restriction::Constraints(constraints) => {
                        let mut use_fallback = false;
                        for constraint in constraints {
                            let constraint_attr_base =
                                self.as_attribute_base_no_union(constraint.clone());
                            if let Some(AttributeBase::ClassInstance(cls)) = constraint_attr_base {
                                bases.push(Some(AttributeBase::TypeVar(
                                    quantified.clone(),
                                    Some(cls),
                                )));
                            } else {
                                use_fallback = true;
                            }
                        }
                        if use_fallback {
                            bases.push(Some(AttributeBase::TypeVar(quantified.clone(), None)));
                        }
                    }
                    Restriction::Unrestricted => bases.push(Some(AttributeBase::ClassInstance(
                        self.stdlib.object().clone(),
                    ))),
                },
                _ => bases.push(self.as_attribute_base_no_union(base.clone())),
            };
        });
        bases
    }

    /// Compute the get (i.e. read) type of an attribute. If the attribute cannot be found or read,
    /// error and return `Any`. Use this to infer the type of a direct attribute fetch.
    pub fn type_of_attr_get(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) -> Type {
        let bases = self.get_possible_attribute_bases(base);
        let mut results = Vec::new();
        for attr_base in bases {
            let lookup_result = attr_base.map_or_else(
                || LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone())),
                |attr_base| self.lookup_attr_from_base_no_union(attr_base, attr_name),
            );
            match self.get_type_or_conflated_error_msg(
                lookup_result,
                attr_name,
                range,
                errors,
                context,
                todo_ctx,
            ) {
                Ok(ty) => results.push(ty),
                Err(msg) => results.push(self.error(
                    errors,
                    range,
                    ErrorKind::MissingAttribute,
                    context,
                    msg,
                )),
            }
        }
        self.unions(results)
    }

    /// Compute the get (i.e., read) type of a magic dunder attribute, if it can be found. If reading is not
    /// permitted, return an error and `Some(Any)`. If no attribute is found, return `None`.
    ///
    /// Note that this method is only expected to be used for magic attr lookups and is not expected to
    /// produce correct results for arbitrary kinds of attributes. If you don't know whether an attribute lookup
    /// is magic or not, it's highly likely that this method isn't the right thing to do for you.
    ///
    /// Magic attrs are almost always dunder names, e.g. `__getattr__`, `__eq__`, `__contains__`, etc.
    pub fn type_of_magic_dunder_attr(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) -> Option<Type> {
        let mut not_found = false;
        let mut attr_tys = Vec::new();
        self.map_over_union(base, |base| {
            match self.lookup_magic_dunder_attr_no_union(base, attr_name) {
                LookupResult::Found(attr) => attr_tys.push(
                    self.resolve_get_access(attr, range, errors, context)
                        .unwrap_or_else(|e| {
                            self.error(
                                errors,
                                range,
                                ErrorKind::MissingAttribute,
                                context,
                                e.to_error_msg(attr_name),
                            )
                        }),
                ),
                LookupResult::InternalError(e) => attr_tys.push(self.error(
                    errors,
                    range,
                    ErrorKind::InternalError,
                    context,
                    e.to_error_msg(attr_name, todo_ctx),
                )),
                LookupResult::NotFound(_) => {
                    not_found = true;
                }
            }
        });
        if not_found {
            return None;
        }
        Some(self.unions(attr_tys))
    }

    /// Look up the `_value_` attribute of an enum class. This field has to be a plain instance
    /// attribute annotated in the class body; it is used to validate enum member values, which are
    /// supposed to all share this type.
    pub fn type_of_enum_value(&self, enum_: &EnumMetadata) -> Option<Type> {
        let base = Type::ClassType(enum_.cls.clone());
        match self.lookup_attr_no_union(&base, &Name::new_static("_value_")) {
            LookupResult::Found(attr) => match attr.inner {
                AttributeInner::Simple(ty, ..) => Some(ty),
                // NOTE: We currently do not expect to use `__getattr__` for `_value_` annotation lookup.
                AttributeInner::NoAccess(_)
                | AttributeInner::Property(..)
                | AttributeInner::Descriptor(..)
                | AttributeInner::GetAttr(..) => None,
            },
            _ => None,
        }
    }

    /// Check whether a type or expression is assignable to an attribute, using contextual
    /// typing in the expression case.
    ///
    /// If (and only if) an attribute is a simple read-write attribute, returns the
    /// type of the term to which we set it which may be used for narrowing.
    pub fn check_assign_to_attribute_and_infer_narrow(
        &self,
        base: &Type,
        name: &Name,
        got: &ExprOrBinding,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let got = match got {
            ExprOrBinding::Expr(value) => Either::Left(value),
            ExprOrBinding::Binding(got) => {
                Either::Right(self.solve_binding(got, errors).arc_clone_ty())
            }
        };
        self.check_attr_set_and_infer_narrow(
            base,
            name,
            got,
            range,
            errors,
            None,
            "attr::check_assign_to_attribute_and_infer_narrow",
        )
    }

    fn check_attr_set_and_infer_narrow(
        &self,
        base: &Type,
        attr_name: &Name,
        got: Either<&Expr, Type>,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) -> Option<Type> {
        let mut narrowed_types = Some(Vec::new());
        let bases = self.get_possible_attribute_bases(base);
        for attr_base in bases {
            let lookup_result = attr_base.map_or_else(
                || LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone())),
                |attr_base| self.lookup_attr_from_base_no_union(attr_base, attr_name),
            );
            match lookup_result {
                LookupResult::Found(attr) => match attr.inner {
                    AttributeInner::Simple(want, Visibility::ReadWrite) => {
                        let ty = match &got {
                            Either::Left(got) => self.expr(
                                got,
                                Some((&want, &|| TypeCheckContext {
                                    kind: TypeCheckKind::Attribute(attr_name.clone()),
                                    context: context.map(|ctx| ctx()),
                                })),
                                errors,
                            ),
                            Either::Right(got) => {
                                self.check_type(&want, got, range, errors, &|| TypeCheckContext {
                                    kind: TypeCheckKind::Attribute(attr_name.clone()),
                                    context: context.map(|ctx| ctx()),
                                });
                                got.clone()
                            }
                        };
                        if let Some(narrowed_types) = &mut narrowed_types {
                            narrowed_types.push(ty)
                        }
                        // Avoid the hook where we wipe `narrowed_types` in all other cases.
                        continue;
                    }
                    AttributeInner::NoAccess(e) => {
                        self.error(
                            errors,
                            range,
                            ErrorKind::NoAccess,
                            context,
                            e.to_error_msg(attr_name),
                        );
                    }
                    AttributeInner::Simple(_, Visibility::ReadOnly) => {
                        self.error(
                            errors,
                            range,
                            ErrorKind::ReadOnly,
                            context,
                            format!("Cannot assign to read-only attribute `{attr_name}`"),
                        );
                    }
                    AttributeInner::Property(_, None, cls) => {
                        let e = NoAccessReason::SettingReadOnlyProperty(cls);
                        self.error(
                            errors,
                            range,
                            ErrorKind::ReadOnly,
                            context,
                            e.to_error_msg(attr_name),
                        );
                    }
                    AttributeInner::Property(_, Some(setter), _) => {
                        let got = match &got {
                            Either::Left(got) => CallArg::Expr(got),
                            Either::Right(got) => CallArg::Type(got, range),
                        };
                        self.call_property_setter(setter, got, range, errors, context);
                    }
                    AttributeInner::Descriptor(d) => {
                        match (d.base, d.setter) {
                            (DescriptorBase::Instance(class_type), Some(setter)) => {
                                let got = match &got {
                                    Either::Left(got) => CallArg::Expr(got),
                                    Either::Right(got) => CallArg::Type(got, range),
                                };
                                self.call_descriptor_setter(
                                    setter, class_type, got, range, errors, context,
                                );
                            }
                            (DescriptorBase::Instance(class_type), None) => {
                                let e = NoAccessReason::SettingReadOnlyDescriptor(
                                    class_type.class_object().dupe(),
                                );
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::ReadOnly,
                                    context,
                                    e.to_error_msg(attr_name),
                                );
                            }
                            (DescriptorBase::ClassDef(class), _) => {
                                let e = NoAccessReason::SettingDescriptorOnClass(class.dupe());
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::NoAccess,
                                    context,
                                    e.to_error_msg(attr_name),
                                );
                            }
                        };
                    }
                    AttributeInner::GetAttr(not_found, _, name) => {
                        // Attribute setting bypasses `__getattr__` lookup and behaves the same
                        // as if the `__getattr__` lookup did not happen.
                        self.error(
                            errors,
                            range,
                            ErrorKind::MissingAttribute,
                            context,
                            not_found.to_error_msg(&name),
                        );
                    }
                },
                LookupResult::InternalError(e) => {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InternalError,
                        context,
                        e.to_error_msg(attr_name, todo_ctx),
                    );
                }
                LookupResult::NotFound(e) => {
                    self.error(
                        errors,
                        range,
                        ErrorKind::MissingAttribute,
                        context,
                        e.to_error_msg(attr_name),
                    );
                }
            }
            // If we hit anything other than a simple, read-write attribute then we will not infer
            // a type for narrowing.
            narrowed_types = None;
        }
        narrowed_types.map(|ts| self.unions(ts))
    }

    pub fn check_attr_delete(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) {
        let bases = self.get_possible_attribute_bases(base);
        for attr_base in bases {
            let lookup_result = attr_base.map_or_else(
                || LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone())),
                |attr_base| self.lookup_attr_from_base_no_union(attr_base, attr_name),
            );
            match lookup_result {
                LookupResult::Found(attr) => match attr.inner {
                    // TODO: deleting attributes is allowed at runtime, but is not type-safe
                    // except for descriptors that implement `__delete__`
                    AttributeInner::Simple(_, Visibility::ReadWrite)
                    | AttributeInner::Property(_, _, _)
                    | AttributeInner::Descriptor(_) => {}
                    AttributeInner::NoAccess(e) => {
                        self.error(
                            errors,
                            range,
                            ErrorKind::NoAccess,
                            context,
                            e.to_error_msg(attr_name),
                        );
                    }
                    AttributeInner::Simple(_, Visibility::ReadOnly) => {
                        self.error(
                            errors,
                            range,
                            ErrorKind::ReadOnly,
                            context,
                            format!("Cannot delete read-only attribute `{attr_name}`"),
                        );
                    }
                    AttributeInner::GetAttr(not_found, _, name) => {
                        // Attribute deleting bypasses `__getattr__` lookup and behaves the same
                        // as if the `__getattr__` lookup did not happen.
                        self.error(
                            errors,
                            range,
                            ErrorKind::MissingAttribute,
                            context,
                            not_found.to_error_msg(&name),
                        );
                    }
                },
                LookupResult::InternalError(e) => {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InternalError,
                        context,
                        e.to_error_msg(attr_name, todo_ctx),
                    );
                }
                LookupResult::NotFound(e) => {
                    self.error(
                        errors,
                        range,
                        ErrorKind::MissingAttribute,
                        context,
                        e.to_error_msg(attr_name),
                    );
                }
            }
        }
    }

    pub fn is_attr_subset(
        &self,
        got: &Attribute,
        want: &Attribute,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> bool {
        match (&got.inner, &want.inner) {
            (_, AttributeInner::NoAccess(_)) => true,
            (AttributeInner::NoAccess(_), _) => false,
            (AttributeInner::Property(_, _, _), AttributeInner::Simple(..)) => false,
            (
                AttributeInner::Simple(_, Visibility::ReadOnly),
                AttributeInner::Property(_, Some(_), _)
                | AttributeInner::Simple(_, Visibility::ReadWrite),
            ) => false,
            (
                // TODO(stroxler): Investigate this case more: methods should be ReadOnly, but
                // in some cases for unknown reasons they wind up being ReadWrite.
                AttributeInner::Simple(got @ Type::BoundMethod(_), Visibility::ReadWrite),
                AttributeInner::Simple(want @ Type::BoundMethod(_), Visibility::ReadWrite),
            ) => is_subset(got, want),
            (
                AttributeInner::Simple(got, Visibility::ReadWrite),
                AttributeInner::Simple(want, Visibility::ReadWrite),
            ) => is_subset(got, want) && is_subset(want, got),
            (
                AttributeInner::Simple(got, ..),
                AttributeInner::Simple(want, Visibility::ReadOnly),
            ) => is_subset(got, want),
            (
                AttributeInner::Simple(got, Visibility::ReadOnly),
                AttributeInner::Property(want, _, _),
            ) => {
                is_subset(
                    // Synthesize a getter method
                    &Type::callable_ellipsis(got.clone()),
                    want,
                )
            }
            (
                AttributeInner::Simple(got, Visibility::ReadWrite),
                AttributeInner::Property(want, want_setter, _),
            ) => {
                if !is_subset(
                    // Synthesize a getter method
                    &Type::callable_ellipsis(got.clone()),
                    want,
                ) {
                    return false;
                }
                if let Some(want_setter) = want_setter {
                    // Synthesize a setter method
                    is_subset(
                        want_setter,
                        &Type::callable(
                            vec![Param::PosOnly(None, got.clone(), Required::Required)],
                            Type::None,
                        ),
                    )
                } else {
                    true
                }
            }
            (
                AttributeInner::Property(got_getter, got_setter, _),
                AttributeInner::Property(want_getter, want_setter, _),
            ) => {
                if !is_subset(got_getter, want_getter) {
                    false
                } else {
                    match (got_setter, want_setter) {
                        (Some(got_setter), Some(want_setter)) => is_subset(got_setter, want_setter),
                        (None, Some(_)) => false,
                        (_, None) => true,
                    }
                }
            }
            (
                AttributeInner::Descriptor(
                    Descriptor {
                        descriptor_ty: got_ty,
                        ..
                    },
                    ..,
                ),
                AttributeInner::Descriptor(
                    Descriptor {
                        descriptor_ty: want_ty,
                        ..
                    },
                    ..,
                ),
            ) => is_subset(got_ty, want_ty),
            (AttributeInner::Descriptor(..), _) | (_, AttributeInner::Descriptor(..)) => false,
            (AttributeInner::GetAttr(..), _) | (_, AttributeInner::GetAttr(..)) => {
                // NOTE(grievejia): `__getattr__` does not participate in structural subtyping
                // check for now. We may revisit this in the future if the need comes.
                false
            }
        }
    }

    fn resolve_get_access(
        &self,
        attr: Attribute,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Result<Type, NoAccessReason> {
        match attr.inner {
            AttributeInner::NoAccess(reason) => Err(reason),
            AttributeInner::Simple(ty, Visibility::ReadWrite)
            | AttributeInner::Simple(ty, Visibility::ReadOnly) => Ok(ty),
            AttributeInner::Property(getter, ..) => {
                Ok(self.call_property_getter(getter, range, errors, context))
            }
            AttributeInner::Descriptor(d, ..) => {
                match d {
                    // Reading a descriptor with a getter resolves to a method call
                    //
                    // TODO(stroxler): Once we have more complex error traces, it would be good to pass
                    // context down so that errors inside the call can mention that it was a descriptor read.
                    Descriptor {
                        base,
                        getter: Some(getter),
                        ..
                    } => Ok(self.call_descriptor_getter(getter, base, range, errors, context)),
                    // Reading descriptor with no getter resolves to the descriptor itself
                    Descriptor {
                        descriptor_ty,
                        getter: None,
                        ..
                    } => Ok(descriptor_ty),
                }
            }
            AttributeInner::GetAttr(_, getattr_attr, name) => self
                .resolve_get_access(Attribute::new(*getattr_attr), range, errors, context)
                .map(|getattr_ty| self.call_getattr(getattr_ty, name, range, errors, context)),
        }
    }

    /// A convenience function for callers who don't care about reasons a lookup failed and are
    /// only interested in simple, read-write attributes (in particular, this covers instance access to
    /// regular methods, and is useful for edge cases where we handle cases like `__call__` and `__new__`).
    pub fn resolve_as_instance_method(&self, attr: Attribute) -> Option<Type> {
        self.resolve_as_instance_method_with_attribute_inner(attr.inner)
    }

    fn resolve_as_instance_method_with_attribute_inner(
        &self,
        inner: AttributeInner,
    ) -> Option<Type> {
        match inner {
            // TODO(stroxler): ReadWrite attributes are not actually methods but limiting access to
            // ReadOnly breaks unit tests; we should investigate callsites to understand this better.
            // NOTE(grievejia): We currently do not expect to use `__getattr__` for this lookup.
            AttributeInner::Simple(ty, Visibility::ReadOnly)
            | AttributeInner::Simple(ty, Visibility::ReadWrite) => Some(ty),
            AttributeInner::NoAccess(_)
            | AttributeInner::Property(..)
            | AttributeInner::Descriptor(..)
            | AttributeInner::GetAttr(..) => None,
        }
    }

    pub fn resolve_named_tuple_element(&self, attr: Attribute) -> Option<Type> {
        // NamedTuples are immutable, so their attributes are always read-only
        // NOTE(grievejia): We do not use `__getattr__` here because this lookup is expected to be inovked
        // on NamedTuple attributes with known names.
        match attr.inner {
            AttributeInner::Simple(ty, Visibility::ReadOnly) => Some(ty),
            AttributeInner::Simple(_, Visibility::ReadWrite)
            | AttributeInner::NoAccess(_)
            | AttributeInner::Property(..)
            | AttributeInner::Descriptor(..)
            | AttributeInner::GetAttr(..) => None,
        }
    }

    /// A convenience function for callers which want an error but do not need to distinguish
    /// between NotFound and Error results.
    fn get_type_or_conflated_error_msg(
        &self,
        lookup: LookupResult,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) -> Result<Type, String> {
        match lookup {
            LookupResult::Found(attr) => {
                match self.resolve_get_access(attr, range, errors, context) {
                    Ok(ty) => Ok(ty.clone()),
                    Err(err) => Err(err.to_error_msg(attr_name)),
                }
            }
            LookupResult::NotFound(err) => Err(err.to_error_msg(attr_name)),
            LookupResult::InternalError(err) => Err(err.to_error_msg(attr_name, todo_ctx)),
        }
    }

    fn lookup_attr_from_attribute_base(
        &self,
        base: AttributeBase,
        attr_name: &Name,
    ) -> LookupResult {
        match base {
            AttributeBase::EnumLiteral(_, member, _)
                if matches!(attr_name.as_str(), "name" | "_name_") =>
            {
                LookupResult::found_type(Type::Literal(Lit::Str(member.as_str().into())))
            }
            AttributeBase::EnumLiteral(_, _, raw_type)
                if matches!(attr_name.as_str(), "value" | "_value_") =>
            {
                LookupResult::found_type(raw_type.clone())
            }
            AttributeBase::ClassInstance(class) | AttributeBase::EnumLiteral(class, _, _) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let mut attr_name = attr_name.clone();
                // Special case magic enum properties
                if metadata.is_enum() && attr_name.as_str() == "value" {
                    attr_name = Name::new("_value_")
                }
                if metadata.is_enum() && attr_name.as_str() == "name" {
                    attr_name = Name::new("_name_")
                }
                match self.get_instance_attribute(&class, &attr_name) {
                    Some(attr) => LookupResult::Found(attr),
                    None if metadata.has_base_any() => {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None => {
                        LookupResult::NotFound(NotFound::Attribute(class.class_object().dupe()))
                    }
                }
            }
            AttributeBase::SuperInstance(cls, obj) => {
                match self.get_super_attribute(&cls, &obj, attr_name) {
                    Some(attr) => LookupResult::Found(attr),
                    None if let SuperObj::Instance(cls) = &obj
                        && self.extends_any(cls.class_object()) =>
                    {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None if let SuperObj::Class(cls) = &obj
                        && self.extends_any(cls) =>
                    {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None => LookupResult::NotFound(NotFound::Attribute(cls.class_object().dupe())),
                }
            }
            AttributeBase::ClassObject(class) => {
                match self.get_class_attribute(&class, attr_name) {
                    Some(attr) => LookupResult::Found(attr),
                    None => {
                        // Classes are instances of their metaclass, which defaults to `builtins.type`.
                        // NOTE(grievejia): This lookup serves as fallback for normal class attribute lookup for regular
                        // attributes, but for magic dunder methods it needs to supersede normal class attribute lookup.
                        // See `lookup_magic_dunder_attr()`.
                        let metadata = self.get_metadata_for_class(&class);
                        let instance_attr = match metadata.metaclass() {
                            Some(meta) => self.get_instance_attribute(meta, attr_name),
                            None => {
                                self.get_instance_attribute(self.stdlib.builtins_type(), attr_name)
                            }
                        };
                        match instance_attr {
                            Some(attr) => LookupResult::Found(attr),
                            None if metadata.has_base_any() => {
                                // We can't immediately fall back to Any in this case -- `type[Any]` is actually a special
                                // AttributeBase which requires additional lookup on `type` itself before the Any fallback.
                                self.lookup_attr_from_attribute_base(
                                    AttributeBase::TypeAny(AnyStyle::Implicit),
                                    attr_name,
                                )
                            }
                            None => LookupResult::NotFound(NotFound::ClassAttribute(class)),
                        }
                    }
                }
            }
            AttributeBase::Module(module) => match self.get_module_attr(&module, attr_name) {
                // TODO(samzhou19815): Support module attribute go-to-definition
                Some(attr) => LookupResult::found_type(attr),
                None => LookupResult::NotFound(NotFound::ModuleExport(module)),
            },
            AttributeBase::TypeVar(q, bound) => match (q.kind(), attr_name.as_str()) {
                // Note that is is for cases like `P.args` where `P` is a param spec, or `T.x` where
                // `T` is a type variable (the latter is illegal, but a user could write it). It is
                // not for cases where `base` is a term with a quantified type.
                (QuantifiedKind::ParamSpec, "args") => {
                    LookupResult::found_type(Type::type_form(Type::Args(q)))
                }
                (QuantifiedKind::ParamSpec, "kwargs") => {
                    LookupResult::found_type(Type::type_form(Type::Kwargs(q)))
                }
                (QuantifiedKind::TypeVar, _) if let Some(upper_bound) = bound => {
                    match self.get_bounded_type_var_attribute(q.clone(), &upper_bound, attr_name) {
                        Some(attr) => LookupResult::Found(attr),
                        None => LookupResult::NotFound(NotFound::Attribute(
                            upper_bound.class_object().dupe(),
                        )),
                    }
                }
                _ => {
                    let class = q.as_value(self.stdlib);
                    match self.get_instance_attribute(class, attr_name) {
                        Some(attr) => LookupResult::Found(attr),
                        None => {
                            LookupResult::NotFound(NotFound::Attribute(class.class_object().dupe()))
                        }
                    }
                }
            },
            AttributeBase::TypeAny(style) => {
                let builtins_type_classtype = self.stdlib.builtins_type();
                self.get_instance_attribute(builtins_type_classtype, attr_name)
                    .and_then(|Attribute { inner }| {
                        self.resolve_as_instance_method_with_attribute_inner(inner)
                            .map(LookupResult::found_type)
                    })
                    .map_or_else(
                        || LookupResult::found_type(style.propagate()),
                        |result| result,
                    )
            }
            AttributeBase::Any(style) => LookupResult::found_type(style.propagate()),
            AttributeBase::Never => LookupResult::found_type(Type::never()),
            AttributeBase::Property(mut getter) => {
                if attr_name == "setter" {
                    // Get the property's `setter` method, which, when called with a function, returns
                    // a copy of the property with the passed-in function as its setter. We hack this
                    // by updating the getter's metadata to mark it as a setter method.
                    // TODO(stroxler): it is probably possible to synthesize a forall type here
                    // that uses a type var to propagate the setter. Investigate this option later.
                    getter.transform_func_metadata(|meta: &mut FuncMetadata| {
                        meta.kind = FunctionKind::PropertySetter(Box::new(meta.kind.as_func_id()));
                    });
                    LookupResult::found_type(
                        // TODO(samzhou19815): Support go-to-definition for @property applied symbols
                        getter,
                    )
                } else {
                    let class = self.stdlib.property();
                    match self.get_instance_attribute(class, attr_name) {
                        Some(attr) => LookupResult::Found(attr),
                        None => {
                            LookupResult::NotFound(NotFound::Attribute(class.class_object().dupe()))
                        }
                    }
                }
            }
            AttributeBase::TypedDict(typed_dict) => {
                match self.get_typed_dict_attribute(&typed_dict, attr_name) {
                    Some(attr) => LookupResult::Found(attr),
                    None if self.extends_any(typed_dict.class_object()) => {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None => LookupResult::NotFound(NotFound::Attribute(
                        typed_dict.class_object().dupe(),
                    )),
                }
            }
        }
    }

    /// A magic dunder attribute differs from a normal attribute in one crucial aspect:
    /// if looked up from a base of `type[A]` directly, the attribute needs to be defined
    /// on the metaclass instead of class `A` (i.e. we are looking for `type.__magic_dunder_attr__`
    /// instead of `A.__magic_dunder_attr__`).
    fn lookup_magic_dunder_attr(&self, base: AttributeBase, dunder_name: &Name) -> LookupResult {
        match base {
            AttributeBase::ClassObject(class) => {
                let metadata = self.get_metadata_for_class(&class);
                let metaclass = metadata.metaclass().unwrap_or(self.stdlib.builtins_type());
                match self.get_instance_attribute(metaclass, dunder_name) {
                    Some(attr) => LookupResult::Found(attr),
                    None => LookupResult::NotFound(NotFound::Attribute(
                        metaclass.class_object().clone(),
                    )),
                }
            }
            base => self.lookup_attr_from_attribute_base(base, dunder_name),
        }
    }

    fn lookup_attr_from_base_getattr_fallback(
        &self,
        base: AttributeBase,
        attr_name: &Name,
        direct_lookup_result: LookupResult,
    ) -> LookupResult {
        match direct_lookup_result {
            LookupResult::Found(_) | LookupResult::InternalError(_) => direct_lookup_result,
            LookupResult::NotFound(not_found) => {
                let getattr_lookup_result = self.lookup_magic_dunder_attr(base, &dunder::GETATTR);
                match getattr_lookup_result {
                    LookupResult::NotFound(_) | LookupResult::InternalError(_) => {
                        LookupResult::NotFound(not_found)
                    }
                    LookupResult::Found(attr) => {
                        LookupResult::Found(Attribute::getattr(not_found, attr, attr_name.clone()))
                    }
                }
            }
        }
    }

    fn lookup_attr_from_base_no_union(
        &self,
        base: AttributeBase,
        attr_name: &Name,
    ) -> LookupResult {
        let direct_lookup_result = self.lookup_attr_from_attribute_base(base.clone(), attr_name);
        self.lookup_attr_from_base_getattr_fallback(base, attr_name, direct_lookup_result)
    }

    // This function is intended as a low-level building block
    // Unions or intersections should be handled by callers
    fn lookup_magic_dunder_attr_no_union(&self, base: &Type, attr_name: &Name) -> LookupResult {
        match self.as_attribute_base_no_union(base.clone()) {
            None => {
                LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone()))
            }
            Some(base) => {
                let direct_lookup_result = self.lookup_magic_dunder_attr(base.clone(), attr_name);
                self.lookup_attr_from_base_getattr_fallback(base, attr_name, direct_lookup_result)
            }
        }
    }

    // This function is intended as a low-level building block
    // Unions or intersections should be handled by callers
    fn lookup_attr_no_union(&self, base: &Type, attr_name: &Name) -> LookupResult {
        if let Some(base) = self.as_attribute_base_no_union(base.clone()) {
            self.lookup_attr_from_base_no_union(base, attr_name)
        } else {
            LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone()))
        }
    }

    pub fn try_lookup_attr_from_class_type(
        &self,
        cls: ClassType,
        attr_name: &Name,
    ) -> Option<Attribute> {
        match self.lookup_attr_from_base_no_union(AttributeBase::ClassInstance(cls), attr_name) {
            LookupResult::Found(attr) => Some(attr),
            _ => None,
        }
    }

    pub fn try_lookup_attr(&self, base: &Type, attr_name: &Name) -> Vec<Attribute> {
        let mut result = Vec::new();
        let bases = self.get_possible_attribute_bases(base);
        for attr_base in bases {
            let lookup_result = attr_base.map_or_else(
                || LookupResult::InternalError(InternalError::AttributeBaseUndefined(base.clone())),
                |attr_base| self.lookup_attr_from_base_no_union(attr_base, attr_name),
            );
            match lookup_result {
                LookupResult::Found(attr) => result.push(attr),
                _ => {}
            }
        }
        result
    }

    fn get_module_exports(&self, module_name: ModuleName) -> Option<Exports> {
        self.exports.get(module_name).ok()
    }

    fn get_exported_type(&self, exports: &Exports, from: ModuleName, name: &Name) -> Option<Type> {
        if exports.exports(self.exports).contains_key(name) {
            Some(
                self.get_from_module(from, None, &KeyExport(name.clone()))
                    .arc_clone(),
            )
        } else {
            None
        }
    }

    fn get_module_attr(&self, module: &Module, attr_name: &Name) -> Option<Type> {
        let module_name = ModuleName::from_parts(module.path());
        let module_exports = self.get_module_exports(module_name);

        // `module_name` could refer to a package, in which case we need to check if
        // `module_name.attr_name`:
        // - Has been imported. This can happen in two ways:
        //   Either there's an explicit import statement earlier than import directly from `module_name.attr_name`,
        //   or `module_name` is imported, and `module_name` marked itself as implicitly importing `attr_name`.
        //   In other cases, just importing `module_name` shouldn't automatically make the submodule name
        //   `module_name.attr_name` accessible.
        // - Actually exists as a submodule on the filesystem.
        //
        // This check always takes precedence over the result of the module export lookup, because the import system
        // would always bind the submodule name `attr_name` to the namespace of `module_name` *after* the module
        // toplevel of `module_name` has been executed.
        let submodule = module.push_path(attr_name.clone());
        let submodule_name = module_name.append(attr_name);
        let is_imported = submodule.is_submodules_imported_directly()
            || module_exports
                .as_ref()
                .is_some_and(|exports| exports.is_submodule_imported_implicitly(attr_name));
        if is_imported && self.get_module_exports(submodule_name).is_some() {
            Some(submodule.to_type())
        } else {
            module_exports.map_or(Some(Type::any_error()), |exports| {
                self.get_exported_type(&exports, module_name, attr_name)
            })
        }
    }

    // This function is intended as a low-level building block
    // Unions or intersections should be handled by callers
    fn as_attribute_base_no_union(&self, ty: Type) -> Option<AttributeBase> {
        match ty {
            Type::ClassType(class_type) => Some(AttributeBase::ClassInstance(class_type)),
            Type::ClassDef(cls) => Some(AttributeBase::ClassObject(cls)),
            Type::SelfType(class_type) => Some(AttributeBase::ClassInstance(class_type)),
            Type::Type(box Type::SelfType(class_type)) => {
                Some(AttributeBase::ClassObject(class_type.class_object().dupe()))
            }
            Type::TypedDict(typed_dict) => Some(AttributeBase::TypedDict(typed_dict.clone())),
            Type::Tuple(Tuple::Unbounded(element)) => {
                Some(AttributeBase::ClassInstance(self.stdlib.tuple(*element)))
            }
            Type::Tuple(Tuple::Concrete(elements)) => {
                Some(AttributeBase::ClassInstance(if elements.is_empty() {
                    self.stdlib.tuple(Type::Any(AnyStyle::Implicit))
                } else {
                    self.stdlib.tuple(self.unions(elements))
                }))
            }
            Type::Tuple(Tuple::Unpacked(box (
                prefix,
                Type::Tuple(Tuple::Unbounded(middle)),
                suffix,
            ))) => {
                let mut elements = prefix;
                elements.push(*middle);
                elements.extend(suffix);
                Some(AttributeBase::ClassInstance(
                    self.stdlib.tuple(self.unions(elements)),
                ))
            }
            // TODO(yangdanny): Can we do better here? There might be some information
            // in the unpacked bit that would be useful.
            Type::Tuple(Tuple::Unpacked(_)) => Some(AttributeBase::ClassInstance(
                self.stdlib.tuple(Type::any_implicit()),
            )),
            Type::LiteralString => Some(AttributeBase::ClassInstance(self.stdlib.str().clone())),
            Type::Literal(Lit::Enum(box (class, member, raw_ty))) => {
                Some(AttributeBase::EnumLiteral(class, member, raw_ty))
            }
            Type::Literal(lit) => Some(AttributeBase::ClassInstance(
                lit.general_class_type(self.stdlib).clone(),
            )),
            Type::TypeGuard(_) | Type::TypeIs(_) => {
                Some(AttributeBase::ClassInstance(self.stdlib.bool().clone()))
            }
            Type::Any(style) => Some(AttributeBase::Any(style)),
            Type::TypeAlias(ta) => self.as_attribute_base_no_union(ta.as_value(self.stdlib)),
            Type::Type(box Type::ClassType(class)) => {
                Some(AttributeBase::ClassObject(class.class_object().dupe()))
            }
            Type::Type(box Type::Quantified(q)) if q.is_type_var() => match q.restriction() {
                // TODO(#119): this is wrong, because we lose the information that this is a type var
                Restriction::Bound(bound) => {
                    self.as_attribute_base_no_union(Type::type_form(bound.clone()))
                }
                _ => Some(AttributeBase::TypeVar(q, None)),
            },
            Type::Type(box Type::Quantified(q)) => Some(AttributeBase::TypeVar(q, None)),
            Type::Type(box Type::Any(style)) => Some(AttributeBase::TypeAny(style)),
            Type::Module(module) => Some(AttributeBase::Module(module)),
            Type::TypeVar(_) => Some(AttributeBase::ClassInstance(self.stdlib.type_var().clone())),
            Type::ParamSpec(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.param_spec().clone(),
            )),
            Type::TypeVarTuple(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.type_var_tuple().clone(),
            )),
            Type::Args(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.param_spec_args().clone(),
            )),
            Type::Kwargs(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.param_spec_kwargs().clone(),
            )),
            Type::None => Some(AttributeBase::ClassInstance(
                self.stdlib.none_type().clone(),
            )),
            Type::Never(_) => Some(AttributeBase::Never),
            _ if ty.is_property_getter() => Some(AttributeBase::Property(ty)),
            Type::Callable(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.function_type().clone(),
            )),
            Type::Function(box Function {
                signature: _,
                metadata,
            })
            | Type::Overload(Overload {
                signatures: _,
                metadata: box metadata,
            }) => Some(AttributeBase::ClassInstance(
                if let FunctionKind::CallbackProtocol(cls) = metadata.kind {
                    *cls
                } else {
                    self.stdlib.function_type().clone()
                },
            )),
            Type::BoundMethod(_) => Some(AttributeBase::ClassInstance(
                self.stdlib.method_type().clone(),
            )),
            Type::Ellipsis => Some(AttributeBase::ClassInstance(
                self.stdlib.ellipsis_type()?.clone(),
            )),
            Type::Forall(forall) => self.as_attribute_base_no_union(forall.body.as_type()),
            Type::Var(v) => {
                if let Some(_guard) = self.recurser.recurse(v) {
                    self.as_attribute_base_no_union(self.solver().force_var(v))
                } else {
                    None
                }
            }
            Type::SuperInstance(box (cls, obj)) => Some(AttributeBase::SuperInstance(cls, obj)),
            Type::Quantified(q) if q.is_type_var() => match q.restriction() {
                // TODO(#119): this is wrong, because we lose the information that this is a type var
                Restriction::Bound(bound) => self.as_attribute_base_no_union(bound.clone()),
                // TODO: handle constraints
                Restriction::Constraints(_) => None,
                Restriction::Unrestricted => {
                    Some(AttributeBase::ClassInstance(self.stdlib.object().clone()))
                }
            },
            // TODO: check to see which ones should have class representations
            Type::Union(_)
            | Type::SpecialForm(_)
            | Type::Type(_)
            | Type::Intersect(_)
            | Type::Unpack(_)
            | Type::Concatenate(_, _)
            | Type::ParamSpecValue(_)
            | Type::Quantified(_) => None,
        }
    }

    /// Compute the get (i.e. read) type information of an attribute for narrowing.
    /// - If the attribute is a descriptor that cannot be narrowed, return `PropertyOrDescriptor({read_type})`,
    ///   where the `read_type` is the type of a fetch (which can be narrowed under the unchecked
    ///   assumption that the descriptor return type is consistent).
    /// - If the base type is a union and at least one case has a descriptor that cannot be narrowed,
    ///   return UnionPropertyOrDescriptor({read_type}), which will allow us to make error messages
    ///   clearer for this case.
    /// - If the attribute is safely narrowable (up to data races, which we do not currently attempt
    ///   to model) - which is true for normal attributes and may eventually for known-to-be-sound
    ///   built-in descriptors, return `Simple({read_type})`
    /// - If the attribute comes from `__getattr__`, treat it as safely narrowable. This is unsound but
    ///   pragmatic, because `__getattr__` stubs are often used to indicate gradual typing.
    /// - If the attribute cannot be found or read return `Simple(ClassType({object}))`. There will
    ///   still be a type error on the narrow, but we should treat it as a valid narrow starting
    ///   from `object` in downstream code.
    pub fn narrowable_for_attr(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Narrowable {
        match base {
            Type::Union(base_tys) => {
                let mut has_property_or_descriptor = false;
                let ty = self.unions(
                    base_tys
                        .iter()
                        .map(|base_ty| {
                            match self
                                .narrowable_for_attr_no_union(base_ty, attr_name, range, errors)
                            {
                                Narrowable::Simple(ty) => ty,
                                // UnionPropertyOrDescriptor shouldn't happen in practice
                                Narrowable::PropertyOrDescriptor(ty)
                                | Narrowable::UnionPropertyOrDescriptor(ty) => {
                                    has_property_or_descriptor = true;
                                    ty
                                }
                            }
                        })
                        .collect(),
                );
                if has_property_or_descriptor {
                    Narrowable::UnionPropertyOrDescriptor(ty)
                } else {
                    Narrowable::Simple(ty)
                }
            }
            _ => self.narrowable_for_attr_no_union(base, attr_name, range, errors),
        }
    }

    fn narrowable_for_attr_no_union(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Narrowable {
        let fall_back_to_object_narrowable =
            || Narrowable::Simple(Type::ClassType(self.stdlib.object().clone()));
        match self.lookup_attr_no_union(base, attr_name) {
            LookupResult::InternalError(..) | LookupResult::NotFound(..) => {
                fall_back_to_object_narrowable()
            }
            LookupResult::Found(attr) => {
                let is_property_or_descriptor = match &attr.inner {
                    AttributeInner::Simple(..)
                    | AttributeInner::NoAccess(..)
                    | AttributeInner::GetAttr(..) => false,
                    AttributeInner::Property(..) | AttributeInner::Descriptor(..) => true,
                };
                match self.resolve_get_access(attr, range, errors, None) {
                    Err(..) => fall_back_to_object_narrowable(),
                    Ok(ty) => {
                        if is_property_or_descriptor {
                            Narrowable::PropertyOrDescriptor(ty)
                        } else {
                            Narrowable::Simple(ty)
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum AttrDefinition {
    FullyResolved(TextRangeWithModuleInfo),
    PartiallyResolvedImportedModuleAttribute { module_name: ModuleName },
}

#[derive(Debug)]
pub struct AttrInfo {
    pub name: Name,
    pub ty: Option<Type>,
    pub definition: Option<AttrDefinition>,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn completions_class(
        &self,
        cls: &Class,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        let mut seen = SmallSet::new();
        for c in iter::once(cls).chain(
            self.get_metadata_for_class(cls)
                .ancestors(self.stdlib)
                .map(|x| x.class_object()),
        ) {
            if c == self.stdlib.object().class_object() {
                // Don't want to suggest `__hash__`
                break;
            }
            match expected_attribute_name {
                None => {
                    for fld in c.fields() {
                        if seen.insert(fld.clone())
                            && let Some(range) = c.field_decl_range(fld)
                        {
                            res.push(AttrInfo {
                                name: fld.clone(),
                                ty: None,
                                definition: Some(AttrDefinition::FullyResolved(
                                    TextRangeWithModuleInfo::new(c.module_info().dupe(), range),
                                )),
                            });
                        }
                    }
                }
                Some(expected_attribute_name) => {
                    if let Some(range) = c.field_decl_range(expected_attribute_name) {
                        res.push(AttrInfo {
                            name: expected_attribute_name.clone(),
                            ty: None,
                            definition: Some(AttrDefinition::FullyResolved(
                                TextRangeWithModuleInfo::new(c.module_info().dupe(), range),
                            )),
                        });
                    }
                }
            }
        }
    }

    fn completions_class_type(
        &self,
        cls: &ClassType,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        self.completions_class(cls.class_object(), expected_attribute_name, res);
    }

    fn completions_module(
        &self,
        module: &Module,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        let module_name = ModuleName::from_parts(module.path());
        if let Some(exports) = self.get_module_exports(module_name) {
            match expected_attribute_name {
                None => {
                    res.extend(exports.wildcard(self.exports).iter().map(|x| AttrInfo {
                        name: x.clone(),
                        ty: None,
                        definition: Some(
                            AttrDefinition::PartiallyResolvedImportedModuleAttribute {
                                module_name,
                            },
                        ),
                    }));
                }
                Some(expected_attribute_name) => {
                    if exports
                        .wildcard(self.exports)
                        .contains(expected_attribute_name)
                    {
                        res.push(AttrInfo {
                            name: expected_attribute_name.clone(),
                            ty: None,
                            definition: Some(
                                AttrDefinition::PartiallyResolvedImportedModuleAttribute {
                                    module_name,
                                },
                            ),
                        });
                    }
                }
            }
        }
    }

    /// List all the attributes available from a type. Used to power completion.
    /// Not all usages need types, so we can skip type computation with `include_types=false`.
    pub fn completions(
        &self,
        base: Type,
        expected_attribute_name: Option<&Name>,
        include_types: bool,
    ) -> Vec<AttrInfo> {
        let mut res = Vec::new();
        // TODO: expose attributes shared across all union members
        if let Some(base) = self.as_attribute_base_no_union(base) {
            match &base {
                AttributeBase::ClassInstance(class) | AttributeBase::EnumLiteral(class, _, _) => {
                    self.completions_class_type(class, expected_attribute_name, &mut res)
                }
                AttributeBase::TypedDict(_) => self.completions_class_type(
                    self.stdlib.typed_dict_fallback(),
                    expected_attribute_name,
                    &mut res,
                ),
                AttributeBase::SuperInstance(class, _) => {
                    self.completions_class_type(class, expected_attribute_name, &mut res)
                }
                AttributeBase::ClassObject(class) => {
                    self.completions_class(class, expected_attribute_name, &mut res)
                }
                AttributeBase::TypeVar(q, _) => self.completions_class_type(
                    q.as_value(self.stdlib),
                    expected_attribute_name,
                    &mut res,
                ),
                AttributeBase::TypeAny(_) => self.completions_class_type(
                    self.stdlib.builtins_type(),
                    expected_attribute_name,
                    &mut res,
                ),
                AttributeBase::Module(module) => {
                    self.completions_module(module, expected_attribute_name, &mut res);
                }
                AttributeBase::Any(_) => {}
                AttributeBase::Never => {}
                AttributeBase::Property(_) => {
                    // TODO(samzhou19815): Support autocomplete for properties
                    {}
                }
            }
            if include_types {
                for info in &mut res {
                    if let Some(definition) = &info.definition
                        && matches!(definition, AttrDefinition::FullyResolved(..))
                        && let LookupResult::Found(attr) =
                            self.lookup_attr_from_attribute_base(base.clone(), &info.name)
                        && let Ok(ty) = self.resolve_get_access(
                            attr,
                            // Important we do not use the resolved TextRange, as it might be in a different module.
                            // Whereas the empty TextRange is valid for all modules.
                            TextRange::default(),
                            &self.error_swallower(),
                            None,
                        )
                        && !ty.is_error()
                    {
                        info.ty = Some(ty);
                    }
                }
            }
        }
        res
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // When coercing an instance of condition_type to bool, check that either it does not override
    // __bool__, or that condition_type.__bool__ is callable.
    pub fn check_dunder_bool_is_callable(
        &self,
        condition_type: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let cond_bool_ty = self.type_of_magic_dunder_attr(
            condition_type,
            &dunder::BOOL,
            range,
            errors,
            None,
            "__bool__",
        );

        // test::narrow::test_walrus_value is an example of a valid union type that
        // as_call_target() does not handle.
        if let Some(ty) = cond_bool_ty
            && !matches!(ty, Type::Union(_) | Type::Never(_))
            && self.as_call_target(ty.clone()).is_none()
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidArgument,
                None,
                format!(
                    "`{}.__bool__` has type `{}`, which is not callable",
                    self.for_display(condition_type.clone()),
                    self.for_display(ty.clone()),
                ),
            );
        }
    }
}
