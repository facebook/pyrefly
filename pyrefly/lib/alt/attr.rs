/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;

use dupe::Dupe;
use pyrefly_python::dunder;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::literal::LitEnum;
use pyrefly_types::special_form::SpecialForm;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Var;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::class::class_field::DataclassMember;
use crate::alt::expr::TypeOrExpr;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyExport;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::export::exports::Exports;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::module::ModuleType;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::read_only::ReadOnlyReason;
use crate::types::type_var::Restriction;
use crate::types::typed_dict::TypedDict;
use crate::types::types::AnyStyle;
use crate::types::types::Overload;
use crate::types::types::SuperObj;
use crate::types::types::Type;

/// The result of looking up an attribute from a particular base.
/// If the base is an `AttributeBase::Union`, multiple results can be
/// returned since each union member is looked up separately.
#[derive(Debug)]
struct LookupResult {
    /// The lookup was successful and an attribute was found.
    pub found: Vec<Attribute>,
    /// The attribute was not found. Callers can use fallback behavior, for
    /// example looking up a different attribute.
    pub not_found: Vec<NotFoundOn>,
    /// There was a Pyrefly-internal error
    pub internal_error: Vec<InternalError>,
}

#[derive(Debug)]
pub enum AttrSubsetError {
    // `got` is not accessible, but `want` is
    NoAccess,
    // `got` is a property, but `want` is not
    Property,
    // `got` is read-only, but `want` is read-write
    ReadOnly,
    // one of `got` or `want` is a descriptor, the other is not
    Descriptor,
    // either `got` or `want` is a call to `__getattr__`
    Getattr,
    // either `got` or `want` is a module fallback
    ModuleFallback,
    // `got` is not a subtype of `want`
    // applies to methods, read-only attributes, and property getters
    Covariant {
        got: Type,
        want: Type,
        got_is_property: bool,
        want_is_property: bool,
    },
    // `got` and `want` are not subtypes of each other
    // applies to read-write attributes
    Invariant {
        got: Type,
        want: Type,
    },
    // `want` is not a subtype of `got`
    // applies to property setters
    Contravariant {
        got: Type,
        want: Type,
        got_is_property: bool,
    },
}

impl AttrSubsetError {
    pub fn to_error_msg(self, child_class: &Name, parent_class: &Name, attr_name: &Name) -> String {
        match self {
            AttrSubsetError::NoAccess => {
                format!("`{child_class}.{attr_name}` is not accessible from the instance")
            }
            AttrSubsetError::Property => {
                format!(
                    "`{child_class}.{attr_name}` is a property, but `{parent_class}.{attr_name}` is not"
                )
            }
            AttrSubsetError::ReadOnly => {
                format!(
                    "`{child_class}.{attr_name}` is read-only, but `{parent_class}.{attr_name}` is read-write"
                )
            }
            AttrSubsetError::Descriptor => {
                format!(
                    "`{child_class}.{attr_name}` and `{parent_class}.{attr_name}` must both be descriptors"
                )
            }
            AttrSubsetError::Getattr => {
                format!(
                    "`{child_class}.{attr_name}` or `{parent_class}.{attr_name}` uses `__getattr__`, which cannot be checked for override compatibility"
                )
            }
            AttrSubsetError::ModuleFallback => {
                format!(
                    "`{child_class}.{attr_name}` or `{parent_class}.{attr_name}` are module fallbacks, which cannot be checked for override compatibility"
                )
            }
            AttrSubsetError::Covariant {
                got,
                want,
                got_is_property,
                want_is_property,
            } => {
                let got_desc = if got_is_property {
                    "Property getter for "
                } else {
                    ""
                };
                let want_desc = if want_is_property {
                    ", the property getter for "
                } else {
                    ", the type of "
                };
                format!(
                    "{got_desc}`{child_class}.{attr_name}` has type `{}`, which is not assignable to `{}`{want_desc}`{parent_class}.{attr_name}`",
                    got.deterministic_printing(),
                    want.deterministic_printing()
                )
            }
            AttrSubsetError::Invariant { got, want } => {
                format!(
                    "`{child_class}.{attr_name}` has type `{}`, which is not consistent with `{}` in `{parent_class}.{attr_name}` (the type of read-write attributes cannot be changed)",
                    got.deterministic_printing(),
                    want.deterministic_printing()
                )
            }
            AttrSubsetError::Contravariant {
                got,
                want,
                got_is_property,
            } => {
                let desc = if got_is_property {
                    "The property setter for "
                } else {
                    ""
                };
                format!(
                    "{desc}`{child_class}.{attr_name}` has type `{}`, which is not assignable from `{}`, the property getter for `{parent_class}.{attr_name}`",
                    got.deterministic_printing(),
                    want.deterministic_printing()
                )
            }
        }
    }
}

/// The result a successful attribute lookup, which can be used for structural
/// subtyping checks or performing get / set / delete actions.
#[derive(Debug)]
pub struct Attribute {
    inner: AttributeInner,
}

#[derive(Debug)]
enum Visibility {
    ReadOnly(ReadOnlyReason),
    ReadWrite,
}

/// The result of an attempt to access an attribute (which will eventually be
/// used either for an action like get / set / delete, or in a structural subtype
/// check).
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
    /// `__getattr__` or `__getattribute__` fallback.
    /// The `NotFound` field stores the (failed) lookup result on the original attribute for
    /// better error reporting downstream. The `AttributeInner` field stores the (successful)
    /// lookup result of the `__getattr__`/`__getattribute__` function or method.
    /// The `Name` field stores the name of the original attribute being looked up.
    GetAttr(NotFoundOn, Box<AttributeInner>, Name),
    /// We did `a.b`, which is a real module on the file system, but not one the user explicitly
    /// or implicitly imported. In some cases, treat this as NotFound. In others, emit an error
    /// but continue on with type.
    ModuleFallback(NotFoundOn, ModuleName, Type),
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

#[derive(Clone, Debug)]
enum NotFoundOn {
    ClassInstance(Class, AttributeBase),
    ClassObject(Class, AttributeBase),
    Module(ModuleType),
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
        Self { inner }
    }

    pub fn no_access(reason: NoAccessReason) -> Self {
        Self {
            inner: AttributeInner::NoAccess(reason),
        }
    }

    pub fn read_write(ty: Type) -> Self {
        Self {
            inner: AttributeInner::Simple(ty, Visibility::ReadWrite),
        }
    }

    pub fn read_only(ty: Type, reason: ReadOnlyReason) -> Self {
        Self {
            inner: AttributeInner::Simple(ty, Visibility::ReadOnly(reason)),
        }
    }

    pub fn read_only_equivalent(self, reason: ReadOnlyReason) -> Self {
        match self.inner {
            AttributeInner::Simple(ty, Visibility::ReadWrite) => Attribute::read_only(ty, reason),
            AttributeInner::Property(getter, _, cls) => Attribute::property(getter, None, cls),
            AttributeInner::Descriptor(descriptor) => Attribute::descriptor(
                descriptor.descriptor_ty,
                descriptor.base,
                descriptor.getter,
                None,
            ),
            inner => Attribute { inner },
        }
    }

    pub fn property(getter: Type, setter: Option<Type>, cls: Class) -> Self {
        Self {
            inner: AttributeInner::Property(getter, setter, cls),
        }
    }

    pub fn descriptor(
        ty: Type,
        base: DescriptorBase,
        getter: Option<Type>,
        setter: Option<Type>,
    ) -> Self {
        Self {
            inner: AttributeInner::Descriptor(Descriptor {
                descriptor_ty: ty,
                base,
                getter,
                setter,
            }),
        }
    }

    fn getattr(not_found: NotFoundOn, getattr: Self, name: Name) -> Self {
        Self {
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
    /// need to prioritize the class logic first.
    fn found_type(ty: Type) -> Self {
        Self {
            found: vec![Attribute::read_write(ty)],
            not_found: Vec::new(),
            internal_error: Vec::new(),
        }
    }

    fn found_type_read_only(ty: Type, reason: ReadOnlyReason) -> Self {
        Self {
            found: vec![Attribute::read_only(ty, reason)],
            not_found: Vec::new(),
            internal_error: Vec::new(),
        }
    }

    fn found(attr: Attribute) -> Self {
        Self {
            found: vec![attr],
            not_found: Vec::new(),
            internal_error: Vec::new(),
        }
    }

    fn not_found(not_found: NotFoundOn) -> Self {
        Self {
            found: Vec::new(),
            not_found: vec![not_found],
            internal_error: Vec::new(),
        }
    }

    fn internal_error(internal_error: InternalError) -> Self {
        Self {
            found: Vec::new(),
            not_found: Vec::new(),
            internal_error: vec![internal_error],
        }
    }

    fn decompose(self) -> (Vec<Attribute>, Vec<NotFoundOn>, Vec<InternalError>) {
        (self.found, self.not_found, self.internal_error)
    }

    fn merge(&mut self, other: LookupResult) {
        self.found.extend(other.found);
        self.not_found.extend(other.not_found);
        self.internal_error.extend(other.internal_error);
    }
}

impl NotFoundOn {
    pub fn to_error_msg(self, attr_name: &Name) -> String {
        match self {
            NotFoundOn::ClassInstance(class, _) => {
                let class_name = class.name();
                format!("Object of class `{class_name}` has no attribute `{attr_name}`",)
            }
            NotFoundOn::ClassObject(class, _) => {
                let class_name = class.name();
                format!("Class `{class_name}` has no class attribute `{attr_name}`")
            }
            NotFoundOn::Module(module) => {
                format!("No attribute `{attr_name}` in module `{module}`")
            }
        }
    }

    fn attr_base(&self) -> AttributeBase {
        match self {
            NotFoundOn::ClassInstance(_, base) | NotFoundOn::ClassObject(_, base) => base.clone(),
            NotFoundOn::Module(module) => AttributeBase::Module(module.clone()),
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
    EnumLiteral(LitEnum),
    ClassInstance(ClassType),
    ClassObject(ClassBase),
    Module(ModuleType),
    /// The attribute access is on a quantified type form (as in `args: P.args` - this
    /// is only used when the base *is* a quantified type, not when the base is
    /// a term that *has* a quantified type.
    /// The second element is a bound or constraint for the type variable.
    TypeVar(Quantified, Option<ClassType>),
    /// Attribute access on a `type[T]`.
    /// The second element is a bound or constraint for the type variable.
    TypeVarType(Quantified, ClassType),
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
    Union(Vec<AttributeBase>),
}

/// A normalized type for attribute lookup which has "class-like" lookup behavior. For example,
/// when we look up an instance method from a class base, we get the unbound function type.

#[derive(Clone, Debug)]
pub enum ClassBase {
    ClassDef(Class),
    ClassType(ClassType),
    Quantified(Quantified, ClassType),
}

impl ClassBase {
    pub fn class_object(&self) -> &Class {
        match self {
            ClassBase::ClassDef(c) => c,
            ClassBase::ClassType(c) => c.class_object(),
            ClassBase::Quantified(_, c) => c.class_object(),
        }
    }

    pub fn targs(&self) -> Option<&TArgs> {
        match self {
            ClassBase::ClassDef(..) => None,
            ClassBase::ClassType(c) => Some(c.targs()),
            ClassBase::Quantified(_, c) => Some(c.targs()),
        }
    }

    pub fn to_type(self) -> Type {
        match self {
            ClassBase::ClassDef(c) => Type::ClassDef(c),
            ClassBase::ClassType(c) => Type::Type(Box::new(c.to_type())),
            ClassBase::Quantified(q, _) => Type::type_form(q.to_type()),
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
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
        let attr_base = self.as_attribute_base(base.clone());
        let lookup_result = attr_base.map_or_else(
            || LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone())),
            |attr_base| self.lookup_attr_from_base(attr_base, attr_name),
        );
        match self.get_type_or_conflated_error_msg(
            lookup_result,
            attr_name,
            range,
            errors,
            context,
            todo_ctx,
        ) {
            Ok(ty) => ty,
            Err(msg) => self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::MissingAttribute, context),
                msg,
            ),
        }
    }

    /// Compute the get (i.e., read) type of a magic dunder attribute, if it can
    /// be found. Handles distributing over unions.
    /// - If we find it, return `Some(dunder_type)`
    /// - If no attribute is found, return `None`.
    /// - If we hit an internal error, record it in a type error so we can debug, and assume `Any`
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
        let lookup_result = match self.as_attribute_base(base.clone()) {
            None => {
                LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone()))
            }
            Some(base) => {
                let direct_lookup_result = self.lookup_magic_dunder_attr(base.clone(), attr_name);
                self.lookup_attr_from_base_getattr_fallback(attr_name, direct_lookup_result)
            }
        };
        for attr in lookup_result.found {
            attr_tys.push(
                self.resolve_get_access(attr, range, errors, context)
                    .unwrap_or_else(|e| {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::new(ErrorKind::MissingAttribute, context),
                            e.to_error_msg(attr_name),
                        )
                    }),
            );
        }
        if !lookup_result.not_found.is_empty() {
            not_found = true;
        }
        for internal_error in lookup_result.internal_error {
            attr_tys.push(self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::InternalError, context),
                internal_error.to_error_msg(attr_name, todo_ctx),
            ))
        }
        if not_found {
            return None;
        }
        Some(self.unions(attr_tys))
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
        let ty;
        let got = match got {
            ExprOrBinding::Expr(value) => TypeOrExpr::Expr(value),
            ExprOrBinding::Binding(got) => {
                ty = self.solve_binding(got, errors);
                TypeOrExpr::Type(ty.ty(), range)
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

    fn check_setattr(
        &self,
        attr_base: AttributeBase,
        attr_name: &Name,
        got: TypeOrExpr,
        not_found: NotFoundOn,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        let (setattr_found, setattr_not_found, setattr_error) = self
            .lookup_magic_dunder_attr(attr_base, &dunder::SETATTR)
            .decompose();
        for setattr_attr in setattr_found {
            let result = self
                .resolve_get_access(setattr_attr, range, errors, context)
                .map(|setattr_ty| {
                    self.call_setattr(
                        setattr_ty,
                        CallArg::Arg(got),
                        attr_name.clone(),
                        range,
                        errors,
                        context,
                    )
                });
            match result {
                Ok(_) => {}
                Err(no_access) => {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::MissingAttribute, context),
                        no_access.to_error_msg(attr_name),
                    );
                }
            }
        }
        if !(setattr_not_found.is_empty() && setattr_error.is_empty()) {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::MissingAttribute, context),
                not_found.to_error_msg(attr_name),
            );
        }
    }

    fn check_delattr(
        &self,
        attr_base: AttributeBase,
        attr_name: &Name,
        not_found: NotFoundOn,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        let (delattr_found, delattr_not_found, delattr_error) = self
            .lookup_magic_dunder_attr(attr_base, &dunder::DELATTR)
            .decompose();
        for delattr_attr in delattr_found {
            let result = self
                .resolve_get_access(delattr_attr, range, errors, context)
                .map(|delattr_ty| {
                    self.call_getattr_or_delattr(
                        delattr_ty,
                        attr_name.clone(),
                        range,
                        errors,
                        context,
                    )
                });
            match result {
                Ok(_) => {}
                Err(no_access) => {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::MissingAttribute, context),
                        no_access.to_error_msg(attr_name),
                    );
                }
            }
        }
        if !(delattr_not_found.is_empty() && delattr_error.is_empty()) {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::MissingAttribute, context),
                not_found.to_error_msg(attr_name),
            );
        }
    }

    fn check_attr_set_and_infer_narrow(
        &self,
        base: &Type,
        attr_name: &Name,
        got: TypeOrExpr,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        todo_ctx: &str,
    ) -> Option<Type> {
        // If we hit anything other than a simple, read-write attribute then we will not infer
        // a type for narrowing.
        let mut should_narrow = true;
        let mut narrowed_types = Vec::new();
        let Some(attr_base) = self.as_attribute_base(base.clone()) else {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::InternalError, context),
                InternalError::AttributeBaseUndefined(base.clone())
                    .to_error_msg(attr_name, todo_ctx),
            );
            return None;
        };
        let (lookup_found, lookup_not_found, lookup_error) = self
            .lookup_attr_from_base(attr_base.clone(), attr_name)
            .decompose();
        for e in lookup_error {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::InternalError, context),
                e.to_error_msg(attr_name, todo_ctx),
            );
            should_narrow = false;
        }
        for not_found in lookup_not_found {
            self.check_setattr(
                attr_base.clone(),
                attr_name,
                got,
                not_found,
                range,
                errors,
                context,
            );
            should_narrow = false;
        }
        for attr in lookup_found {
            match attr {
                // Attribute setting bypasses `__getattr__` lookup and checks `__setattr__`
                // If the attribute is not found, we fall back to `__setattr__`
                Attribute {
                    inner:
                        AttributeInner::GetAttr(not_found, _, _)
                        | AttributeInner::ModuleFallback(not_found, _, _),
                } => {
                    self.check_setattr(
                        attr_base.clone(),
                        attr_name,
                        got,
                        not_found,
                        range,
                        errors,
                        context,
                    );
                    should_narrow = false;
                }
                Attribute {
                    inner: AttributeInner::Simple(want, Visibility::ReadWrite),
                } => {
                    // If the attribute has a converter, then `want` should be the type expected by the converter.
                    let want = match attr_base.clone() {
                        AttributeBase::ClassInstance(cls) => {
                            match self.get_dataclass_member(cls.class_object(), attr_name) {
                                DataclassMember::Field(_, kws) => {
                                    kws.converter_param.unwrap_or(want)
                                }
                                _ => want,
                            }
                        }
                        _ => want,
                    };
                    let ty = match &got {
                        TypeOrExpr::Expr(got) => self.expr(
                            got,
                            Some((&want, &|| TypeCheckContext {
                                kind: TypeCheckKind::Attribute(attr_name.clone()),
                                context: context.map(|ctx| ctx()),
                            })),
                            errors,
                        ),
                        TypeOrExpr::Type(got, _) => {
                            self.check_type(got, &want, range, errors, &|| TypeCheckContext {
                                kind: TypeCheckKind::Attribute(attr_name.clone()),
                                context: context.map(|ctx| ctx()),
                            });
                            (*got).clone()
                        }
                    };
                    if should_narrow {
                        narrowed_types.push(ty);
                    }
                }
                Attribute {
                    inner: AttributeInner::NoAccess(e),
                } => {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::NoAccess, context),
                        e.to_error_msg(attr_name),
                    );
                    should_narrow = false;
                }
                Attribute {
                    inner: AttributeInner::Simple(_, Visibility::ReadOnly(reason)),
                } => {
                    let msg = vec1![
                        format!("Cannot set field `{attr_name}`"),
                        reason.error_message()
                    ];
                    errors.add(range, ErrorInfo::Kind(ErrorKind::ReadOnly), msg);
                    should_narrow = false;
                }
                Attribute {
                    inner: AttributeInner::Property(_, None, cls),
                } => {
                    let e = NoAccessReason::SettingReadOnlyProperty(cls);
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::ReadOnly, context),
                        e.to_error_msg(attr_name),
                    );
                    should_narrow = false;
                }
                Attribute {
                    inner: AttributeInner::Property(_, Some(setter), _),
                } => {
                    let got = CallArg::arg(got);
                    self.call_property_setter(setter, got, range, errors, context);
                    should_narrow = false;
                }
                Attribute {
                    inner: AttributeInner::Descriptor(d),
                } => {
                    should_narrow = false;
                    match (d.base, d.setter) {
                        (DescriptorBase::Instance(class_type), Some(setter)) => {
                            let got = CallArg::arg(got);
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
                                ErrorInfo::new(ErrorKind::ReadOnly, context),
                                e.to_error_msg(attr_name),
                            );
                        }
                        (DescriptorBase::ClassDef(class), _) => {
                            let e = NoAccessReason::SettingDescriptorOnClass(class.dupe());
                            self.error(
                                errors,
                                range,
                                ErrorInfo::new(ErrorKind::NoAccess, context),
                                e.to_error_msg(attr_name),
                            );
                        }
                    };
                }
            }
        }
        if should_narrow {
            Some(self.unions(narrowed_types))
        } else {
            None
        }
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
        let Some(attr_base) = self.as_attribute_base(base.clone()) else {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::InternalError, context),
                InternalError::AttributeBaseUndefined(base.clone())
                    .to_error_msg(attr_name, todo_ctx),
            );
            return;
        };
        let (lookup_found, lookup_not_found, lookup_error) = self
            .lookup_attr_from_base(attr_base.clone(), attr_name)
            .decompose();
        for not_found in lookup_not_found {
            self.check_delattr(
                attr_base.clone(),
                attr_name,
                not_found,
                range,
                errors,
                context,
            );
        }
        for error in lookup_error {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::InternalError, context),
                error.to_error_msg(attr_name, todo_ctx),
            );
        }
        for attr in lookup_found {
            match attr {
                // Attribute deletion bypasses `__getattr__` lookup and checks `__delattr__`
                // If the attribute is not found, we fall back to `__delattr__`
                Attribute {
                    inner:
                        AttributeInner::GetAttr(not_found, _, _)
                        | AttributeInner::ModuleFallback(not_found, _, _),
                } => {
                    self.check_delattr(
                        attr_base.clone(),
                        attr_name,
                        not_found,
                        range,
                        errors,
                        context,
                    );
                }
                // TODO: deleting attributes is allowed at runtime, but is not type-safe
                // except for descriptors that implement `__delete__`
                Attribute {
                    inner:
                        AttributeInner::Simple(_, Visibility::ReadWrite)
                        | AttributeInner::Property(_, _, _)
                        | AttributeInner::Descriptor(_),
                } => {}
                Attribute {
                    inner: AttributeInner::NoAccess(e),
                } => {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::NoAccess, context),
                        e.to_error_msg(attr_name),
                    );
                }
                Attribute {
                    inner: AttributeInner::Simple(_, Visibility::ReadOnly(reason)),
                } => {
                    let msg = vec1![
                        format!("Cannot delete field `{attr_name}`"),
                        reason.error_message()
                    ];
                    errors.add(range, ErrorInfo::Kind(ErrorKind::ReadOnly), msg);
                }
            }
        }
    }

    /// Predicate for whether a specific attribute name matches a protocol during structural
    /// subtyping checks.
    ///
    /// The `is_subset` function (which in most cases will just behave as the
    /// usual subset function) is provided as a callback because we need a way
    /// to track the recursive hypothesis.
    pub fn is_protocol_subset_at_attr(
        &self,
        got: &Type,
        protocol: &ClassType,
        name: &Name,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> bool {
        let got_attrs = self.try_lookup_attr(got, name);
        if (!got_attrs.is_empty())
            && let Some(want) = self.try_lookup_attr_from_class_type(protocol.clone(), name)
        {
            got_attrs.iter().all(|got_attr| {
                self.is_attribute_subset(got_attr, &want, &mut |got, want| is_subset(got, want))
                    .is_ok()
            })
        } else {
            false
        }
    }

    pub fn is_attribute_subset(
        &self,
        got: &Attribute,
        want: &Attribute,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Result<(), AttrSubsetError> {
        match (&got.inner, &want.inner) {
            (_, AttributeInner::NoAccess(_)) => Ok(()),
            (AttributeInner::NoAccess(_), _) => Err(AttrSubsetError::NoAccess),
            (AttributeInner::Property(_, _, _), AttributeInner::Simple(..)) => {
                Err(AttrSubsetError::Property)
            }
            (
                AttributeInner::Simple(_, Visibility::ReadOnly(_)),
                AttributeInner::Property(_, Some(_), _)
                | AttributeInner::Simple(_, Visibility::ReadWrite),
            ) => Err(AttrSubsetError::ReadOnly),
            (
                // TODO(stroxler): Investigate this case more: methods should be ReadOnly, but
                // in some cases for unknown reasons they wind up being ReadWrite.
                AttributeInner::Simple(got @ Type::BoundMethod(_), Visibility::ReadWrite),
                AttributeInner::Simple(want @ Type::BoundMethod(_), Visibility::ReadWrite),
            ) => {
                if is_subset(got, want) {
                    Ok(())
                } else {
                    Err(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: false,
                    })
                }
            }
            (
                AttributeInner::Simple(got, Visibility::ReadWrite),
                AttributeInner::Simple(want, Visibility::ReadWrite),
            ) => {
                if is_subset(got, want) && is_subset(want, got) {
                    Ok(())
                } else {
                    Err(AttrSubsetError::Invariant {
                        got: got.clone(),
                        want: want.clone(),
                    })
                }
            }
            (
                AttributeInner::Simple(got, ..),
                AttributeInner::Simple(want, Visibility::ReadOnly(_)),
            ) => {
                if is_subset(got, want) {
                    Ok(())
                } else {
                    Err(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: false,
                    })
                }
            }
            (
                AttributeInner::Simple(got, Visibility::ReadOnly(_)),
                AttributeInner::Property(want, _, _),
            ) => {
                if is_subset(
                    // Synthesize a getter method
                    &Type::callable_ellipsis(got.clone()),
                    want,
                ) {
                    Ok(())
                } else {
                    Err(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: true,
                    })
                }
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
                    return Err(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: true,
                    });
                }
                if let Some(want_setter) = want_setter {
                    // Synthesize a setter method
                    if is_subset(
                        want_setter,
                        &Type::callable(
                            vec![Param::PosOnly(None, got.clone(), Required::Required)],
                            Type::None,
                        ),
                    ) {
                        Ok(())
                    } else {
                        Err(AttrSubsetError::Contravariant {
                            want: want_setter.clone(),
                            got: got.clone(),
                            got_is_property: true,
                        })
                    }
                } else {
                    Ok(())
                }
            }
            (
                AttributeInner::Property(got_getter, got_setter, _),
                AttributeInner::Property(want_getter, want_setter, _),
            ) => {
                if !is_subset(got_getter, want_getter) {
                    Err(AttrSubsetError::Covariant {
                        got: got_getter.clone(),
                        want: want_getter.clone(),
                        got_is_property: true,
                        want_is_property: true,
                    })
                } else {
                    match (got_setter, want_setter) {
                        (Some(got_setter), Some(want_setter)) => {
                            if is_subset(got_setter, want_setter) {
                                Ok(())
                            } else {
                                Err(AttrSubsetError::Contravariant {
                                    want: want_setter.clone(),
                                    got: got_setter.clone(),
                                    got_is_property: true,
                                })
                            }
                        }
                        (None, Some(_)) => Err(AttrSubsetError::ReadOnly),
                        (_, None) => Ok(()),
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
            ) => {
                if is_subset(got_ty, want_ty) {
                    Ok(())
                } else {
                    Err(AttrSubsetError::Covariant {
                        got: got_ty.clone(),
                        want: want_ty.clone(),
                        got_is_property: false,
                        want_is_property: false,
                    })
                }
            }
            (AttributeInner::Descriptor(..), _) | (_, AttributeInner::Descriptor(..)) => {
                Err(AttrSubsetError::Descriptor)
            }
            (AttributeInner::GetAttr(..), _) | (_, AttributeInner::GetAttr(..)) => {
                // NOTE(grievejia): `__getattr__` does not participate in structural subtyping
                // check for now. We may revisit this in the future if the need comes.
                Err(AttrSubsetError::Getattr)
            }
            (AttributeInner::ModuleFallback(..), _) | (_, AttributeInner::ModuleFallback(..)) => {
                Err(AttrSubsetError::ModuleFallback)
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
            | AttributeInner::Simple(ty, Visibility::ReadOnly(_)) => Ok(ty),
            AttributeInner::Property(getter, ..) => {
                self.record_property_getter(range, &getter);
                Ok(self.call_property_getter(getter, range, errors, context))
            }
            AttributeInner::ModuleFallback(_, name, ty) => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::ImplicitImport, context),
                    format!("Module `{name}` exists, but was not imported explicitly. You are relying on other modules to load it."),
                );
                Ok(ty)
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
                .map(|getattr_ty| {
                    self.call_getattr_or_delattr(getattr_ty, name, range, errors, context)
                }),
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
        let mut types = Vec::new();
        let mut error_messages = Vec::new();
        let (found, not_found, error) = lookup.decompose();
        for attr in found {
            match self.resolve_get_access(attr, range, errors, context) {
                Ok(ty) => types.push(ty),
                Err(err) => error_messages.push(err.to_error_msg(attr_name)),
            }
        }
        for err in not_found {
            error_messages.push(err.to_error_msg(attr_name))
        }
        for err in error {
            error_messages.push(err.to_error_msg(attr_name, todo_ctx))
        }
        if error_messages.is_empty() {
            Ok(self.unions(types))
        } else {
            Err(error_messages.join("\n"))
        }
    }

    fn lookup_attr_from_attribute_base(
        &self,
        base: AttributeBase,
        attr_name: &Name,
    ) -> LookupResult {
        let base_copy = base.clone();
        match base {
            AttributeBase::Union(bases) => {
                let mut bases_iter = bases.into_iter();
                let first = bases_iter.next();
                if let Some(first) = first {
                    let mut result = self.lookup_attr_from_attribute_base(first, attr_name);
                    for base in bases_iter {
                        result.merge(self.lookup_attr_from_attribute_base(base, attr_name));
                    }
                    result
                } else {
                    LookupResult::found_type(Type::never())
                }
            }
            AttributeBase::Any(style) => LookupResult::found_type(style.propagate()),
            AttributeBase::TypeAny(style) => {
                let builtins_type_classtype = self.stdlib.builtins_type();
                self.resolve_instance_method(builtins_type_classtype, attr_name)
                    .map(LookupResult::found_type)
                    .map_or_else(
                        || LookupResult::found_type(style.propagate()),
                        |result| result,
                    )
            }
            AttributeBase::Never => LookupResult::found_type(Type::never()),
            AttributeBase::EnumLiteral(e) if matches!(attr_name.as_str(), "name" | "_name_") => {
                LookupResult::found_type(Type::Literal(Lit::Str(e.member.as_str().into())))
            }
            AttributeBase::EnumLiteral(e) if matches!(attr_name.as_str(), "value" | "_value_") => {
                LookupResult::found_type(e.ty)
            }
            AttributeBase::ClassInstance(class)
            | AttributeBase::EnumLiteral(LitEnum { class, .. }) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let attr_lookup_result = self
                    .special_case_enum_attr_lookup(&class, &metadata, attr_name)
                    .or_else(|| self.get_instance_attribute(&class, attr_name));
                match attr_lookup_result {
                    Some(attr) => LookupResult::found(attr),
                    None if metadata.has_base_any() => {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None => LookupResult::not_found(NotFoundOn::ClassInstance(
                        class.class_object().dupe(),
                        base_copy,
                    )),
                }
            }
            AttributeBase::SuperInstance(cls, obj) => {
                match self.get_super_attribute(&cls, &obj, attr_name) {
                    Some(attr) => {
                        LookupResult::found(attr.read_only_equivalent(ReadOnlyReason::Super))
                    }
                    None if let SuperObj::Instance(cls) = &obj
                        && self.extends_any(cls.class_object()) =>
                    {
                        LookupResult::found_type_read_only(
                            Type::Any(AnyStyle::Implicit),
                            ReadOnlyReason::Super,
                        )
                    }
                    None if let SuperObj::Class(cls) = &obj
                        && self.extends_any(cls.class_object()) =>
                    {
                        LookupResult::found_type_read_only(
                            Type::Any(AnyStyle::Implicit),
                            ReadOnlyReason::Super,
                        )
                    }
                    None => LookupResult::not_found(NotFoundOn::ClassInstance(
                        cls.class_object().dupe(),
                        base_copy,
                    )),
                }
            }
            AttributeBase::TypeVarType(quantified, class) => {
                match (quantified.kind(), attr_name.as_str()) {
                    (QuantifiedKind::ParamSpec, "args") => {
                        LookupResult::found_type(Type::type_form(Type::Args(Box::new(quantified))))
                    }
                    (QuantifiedKind::ParamSpec, "kwargs") => LookupResult::found_type(
                        Type::type_form(Type::Kwargs(Box::new(quantified))),
                    ),
                    _ => match self
                        .get_bounded_quantified_class_attribute(quantified, &class, attr_name)
                    {
                        Some(attr) => LookupResult::found(attr),
                        None => LookupResult::not_found(NotFoundOn::ClassObject(
                            class.class_object().dupe(),
                            base_copy,
                        )),
                    },
                }
            }
            AttributeBase::ClassObject(class) => {
                match self.get_class_attribute(&class, attr_name) {
                    Some(attr) => LookupResult::found(attr),
                    None => {
                        // Classes are instances of their metaclass, which defaults to `builtins.type`.
                        // NOTE(grievejia): This lookup serves as fallback for normal class attribute lookup for regular
                        // attributes, but for magic dunder methods it needs to supersede normal class attribute lookup.
                        // See `lookup_magic_dunder_attr()`.
                        let metadata = self.get_metadata_for_class(class.class_object());
                        let instance_attr = match metadata.metaclass() {
                            Some(meta) => self.get_instance_attribute(meta, attr_name),
                            None => {
                                self.get_instance_attribute(self.stdlib.builtins_type(), attr_name)
                            }
                        };
                        match instance_attr {
                            Some(attr) => LookupResult::found(attr),
                            None if metadata.has_base_any() => {
                                // We can't immediately fall back to Any in this case -- `type[Any]` is actually a special
                                // AttributeBase which requires additional lookup on `type` itself before the Any fallback.
                                self.lookup_attr_from_attribute_base(
                                    AttributeBase::TypeAny(AnyStyle::Implicit),
                                    attr_name,
                                )
                            }
                            None => LookupResult::not_found(NotFoundOn::ClassObject(
                                class.class_object().dupe(),
                                base_copy,
                            )),
                        }
                    }
                }
            }
            AttributeBase::Module(module) => match self.get_module_attr(&module, attr_name) {
                // TODO(samzhou19815): Support module attribute go-to-definition
                Some(attr) => LookupResult::found(attr),
                None => LookupResult::not_found(NotFoundOn::Module(module)),
            },
            AttributeBase::TypeVar(q, bound) => {
                if let Some(upper_bound) = bound {
                    let specific_q = match q.restriction() {
                        Restriction::Constraints(_) => {
                            q.with_restriction(Restriction::Constraints(vec![
                                upper_bound.clone().to_type(),
                            ]))
                        }
                        Restriction::Bound(_) => {
                            q.with_restriction(Restriction::Bound(upper_bound.clone().to_type()))
                        }
                        Restriction::Unrestricted => q,
                    };
                    match self.get_bounded_quantified_attribute(specific_q, &upper_bound, attr_name)
                    {
                        Some(attr) => LookupResult::found(attr),
                        None => LookupResult::not_found(NotFoundOn::ClassInstance(
                            upper_bound.class_object().dupe(),
                            base_copy,
                        )),
                    }
                } else {
                    let class = q.as_value(self.stdlib);
                    match self.get_instance_attribute(class, attr_name) {
                        Some(attr) => LookupResult::found(attr),
                        None => LookupResult::not_found(NotFoundOn::ClassInstance(
                            class.class_object().dupe(),
                            base_copy,
                        )),
                    }
                }
            }
            AttributeBase::Property(mut getter) => {
                if attr_name == "setter" {
                    // When given a decorator `@some_property.setter`, instead of modeling the setter
                    // directly at the type level we just return the getter (the raw `some_property`)
                    // but with the function metadata marked to indicate this is a setter invocation.
                    //
                    // This doesn't accurately model the runtime semantics (the setter function is
                    // not at all the same type), but makes it easy for us to use function metadata
                    // to track both the getter and setter in a way that class field can use.
                    //
                    // See also the function decorator and class field code to
                    // understand how all this works end-to-end.
                    //
                    // TODO(stroxler): it is probably possible to synthesize a forall type here
                    // that uses a type var to propagate the setter. Investigate this option later.
                    getter.transform_toplevel_func_metadata(|meta: &mut FuncMetadata| {
                        meta.flags.is_property_setter_decorator = true;
                    });
                    LookupResult::found_type(
                        // TODO(samzhou19815): Support go-to-definition for @property applied symbols
                        getter,
                    )
                } else {
                    let class = self.stdlib.property();
                    match self.get_instance_attribute(class, attr_name) {
                        Some(attr) => LookupResult::found(attr),
                        None => LookupResult::not_found(NotFoundOn::ClassInstance(
                            class.class_object().dupe(),
                            base_copy,
                        )),
                    }
                }
            }
            AttributeBase::TypedDict(typed_dict) => {
                match self.get_typed_dict_attribute(&typed_dict, attr_name) {
                    Some(attr) => LookupResult::found(attr),
                    None if self.extends_any(typed_dict.class_object()) => {
                        LookupResult::found_type(Type::Any(AnyStyle::Implicit))
                    }
                    None => LookupResult::not_found(NotFoundOn::ClassInstance(
                        typed_dict.class_object().dupe(),
                        base_copy,
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
        match &base {
            AttributeBase::ClassObject(class) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let metaclass = metadata.metaclass().unwrap_or(self.stdlib.builtins_type());
                if *dunder_name == dunder::GETATTRIBUTE
                    && self.field_is_inherited_from_object(metaclass.class_object(), dunder_name)
                {
                    return LookupResult::not_found(NotFoundOn::ClassInstance(
                        metaclass.class_object().clone(),
                        base,
                    ));
                }
                match self.get_instance_attribute(metaclass, dunder_name) {
                    Some(attr) => LookupResult::found(attr),
                    None => LookupResult::not_found(NotFoundOn::ClassInstance(
                        metaclass.class_object().clone(),
                        base,
                    )),
                }
            }
            AttributeBase::ClassInstance(cls)
            | AttributeBase::EnumLiteral(LitEnum { class: cls, .. })
            | AttributeBase::TypeVar(_, Some(cls))
            | AttributeBase::SuperInstance(cls, _)
                if (*dunder_name == dunder::SETATTR
                    || *dunder_name == dunder::DELATTR
                    || *dunder_name == dunder::GETATTRIBUTE)
                    && self.field_is_inherited_from_object(cls.class_object(), dunder_name) =>
            {
                LookupResult::not_found(NotFoundOn::ClassInstance(cls.class_object().clone(), base))
            }
            AttributeBase::TypedDict(typed_dict) if *dunder_name == dunder::GETATTRIBUTE => {
                LookupResult::not_found(NotFoundOn::ClassInstance(
                    typed_dict.class_object().clone(),
                    base,
                ))
            }
            _ => self.lookup_attr_from_attribute_base(base, dunder_name),
        }
    }

    fn lookup_attr_from_base_getattr_fallback(
        &self,
        attr_name: &Name,
        direct_lookup_result: LookupResult,
    ) -> LookupResult {
        let LookupResult {
            found,
            not_found: direct_lookup_not_found,
            internal_error,
        } = direct_lookup_result;
        let mut result = LookupResult {
            found,
            internal_error,
            not_found: Vec::new(),
        };
        for not_found in direct_lookup_not_found {
            let (getattr_found, getattr_not_found, getattr_internal_error) = self
                .lookup_magic_dunder_attr(not_found.attr_base(), &dunder::GETATTR)
                .decompose();
            if !(getattr_not_found.is_empty() && getattr_internal_error.is_empty()) {
                // If the `__getattr__` lookup fails, we fall back to `__getattribute__`
                // Note: at runtime, `__getattribute__` is checked BEFORE looking up the attribute by name,
                // but because the declaration is on `object` and returns `Any`, all attribute accesses
                // would return `Any`.
                let (getattribute_found, getattribute_not_found, getattribute_internal_error) =
                    self.lookup_magic_dunder_attr(not_found.attr_base(), &dunder::GETATTRIBUTE)
                        .decompose();
                if !(getattribute_not_found.is_empty() && getattribute_internal_error.is_empty()) {
                    result.not_found.push(not_found.clone())
                } else {
                    for attr in getattribute_found {
                        result.found.push(Attribute::getattr(
                            not_found.clone(),
                            attr,
                            attr_name.clone(),
                        ));
                    }
                }
            } else {
                for attr in getattr_found {
                    result.found.push(Attribute::getattr(
                        not_found.clone(),
                        attr,
                        attr_name.clone(),
                    ));
                }
            }
        }
        result
    }

    fn lookup_attr_from_base(&self, base: AttributeBase, attr_name: &Name) -> LookupResult {
        let direct_lookup_result = self.lookup_attr_from_attribute_base(base.clone(), attr_name);
        self.lookup_attr_from_base_getattr_fallback(attr_name, direct_lookup_result)
    }

    // This function is intended as a low-level building block
    // Unions or intersections should be handled by callers
    fn lookup_attr(&self, base: &Type, attr_name: &Name) -> LookupResult {
        if let Some(base) = self.as_attribute_base(base.clone()) {
            self.lookup_attr_from_base(base, attr_name)
        } else {
            LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone()))
        }
    }

    fn try_lookup_attr_from_class_type(
        &self,
        cls: ClassType,
        attr_name: &Name,
    ) -> Option<Attribute> {
        // Looking something up from a `ClassInstance` should not yield multiple `Attribute`
        self.lookup_attr_from_base(AttributeBase::ClassInstance(cls), attr_name)
            .found
            .into_iter()
            .next()
    }

    fn try_lookup_attr(&self, base: &Type, attr_name: &Name) -> Vec<Attribute> {
        let mut result = Vec::new();
        let attr_base = self.as_attribute_base(base.clone());
        let lookup_result = attr_base.map_or_else(
            || LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone())),
            |attr_base| self.lookup_attr_from_base(attr_base, attr_name),
        );
        result.extend(lookup_result.found);
        result
    }

    fn get_module_exports(&self, module_name: ModuleName) -> Option<Exports> {
        self.exports.get(module_name).ok()
    }

    fn get_module_attr(&self, module: &ModuleType, attr_name: &Name) -> Option<Attribute> {
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
        let submodule = module.push_part(attr_name.clone());
        if submodule.is_submodules_imported_directly() {
            return Some(Attribute::read_write(submodule.to_type()));
        }

        let module_name = ModuleName::from_parts(module.parts());
        let module_exports = match self.get_module_exports(module_name) {
            Some(x) => x,
            None => return Some(Attribute::read_write(Type::any_error())), // This module doesn't exist, we must have already errored
        };

        if module_exports.is_submodule_imported_implicitly(attr_name)
            && self
                .get_module_exports(module_name.append(attr_name))
                .is_some()
        {
            Some(Attribute::read_write(submodule.to_type()))
        } else if module_exports.exports(self.exports).contains_key(attr_name) {
            Some(Attribute::read_write(
                self.get_from_export(module_name, None, &KeyExport(attr_name.clone()))
                    .arc_clone(),
            ))
        } else if self
            .get_module_exports(module_name.append(attr_name))
            .is_some()
        {
            // The module isn't imported, but does exist on disk, so user must
            // be observing someone else's import.
            Some(Attribute::new(AttributeInner::ModuleFallback(
                NotFoundOn::Module(module.clone()),
                module_name.append(attr_name),
                submodule.to_type(),
            )))
        } else {
            None
        }
    }

    fn force_var_for_attribute_base(&self, var: Var) -> Type {
        if let Some(_guard) = self.recurser.recurse(var) {
            self.solver().force_var(var)
        } else {
            Type::any_implicit()
        }
    }

    fn as_attribute_base(&self, ty: Type) -> Option<AttributeBase> {
        match ty {
            Type::ClassType(class_type) => Some(AttributeBase::ClassInstance(class_type)),
            Type::ClassDef(cls) => Some(AttributeBase::ClassObject(ClassBase::ClassDef(cls))),
            Type::SelfType(class_type) => Some(AttributeBase::ClassInstance(class_type)),
            Type::Type(box Type::SelfType(class_type)) => Some(AttributeBase::ClassObject(
                ClassBase::ClassType(class_type.clone()),
            )),
            Type::TypedDict(td) | Type::PartialTypedDict(td) => {
                Some(AttributeBase::TypedDict(td.clone()))
            }
            Type::Tuple(tuple) => Some(AttributeBase::ClassInstance(self.erase_tuple_type(tuple))),
            Type::LiteralString => Some(AttributeBase::ClassInstance(self.stdlib.str().clone())),
            Type::Literal(Lit::Enum(lit_enum)) => Some(AttributeBase::EnumLiteral(*lit_enum)),
            Type::Literal(lit) => Some(AttributeBase::ClassInstance(
                lit.general_class_type(self.stdlib).clone(),
            )),
            Type::TypeGuard(_) | Type::TypeIs(_) => {
                Some(AttributeBase::ClassInstance(self.stdlib.bool().clone()))
            }
            Type::Any(style) => Some(AttributeBase::Any(style)),
            Type::TypeAlias(ta) => self.as_attribute_base(ta.as_value(self.stdlib)),
            Type::Type(box Type::Tuple(tuple)) => {
                self.as_attribute_base(Type::type_form(self.erase_tuple_type(tuple).to_type()))
            }
            Type::Type(box Type::ClassType(class)) => Some(AttributeBase::ClassObject(
                ClassBase::ClassType(class.clone()),
            )),
            Type::Type(box Type::Quantified(quantified)) => match quantified.restriction() {
                Restriction::Bound(upper_bound) => {
                    let mut res = Vec::new();
                    let mut use_fallback = false;
                    self.map_over_union(upper_bound, |bound| {
                        let bound_attr_base = self.as_attribute_base(bound.clone());
                        if let Some(AttributeBase::ClassInstance(cls)) = bound_attr_base {
                            res.push(AttributeBase::TypeVarType((*quantified).clone(), cls));
                        } else {
                            use_fallback = true;
                        }
                    });
                    if use_fallback {
                        res.push(AttributeBase::TypeVarType(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        ));
                    }
                    Some(AttributeBase::Union(res))
                }
                Restriction::Constraints(constraints) => {
                    let mut res = Vec::new();
                    let mut use_fallback = false;
                    for constraint in constraints {
                        let constraint_attr_base = self.as_attribute_base(constraint.clone());
                        if let Some(AttributeBase::ClassInstance(cls)) = constraint_attr_base {
                            res.push(AttributeBase::TypeVarType((*quantified).clone(), cls));
                        } else {
                            use_fallback = true;
                        }
                    }
                    if use_fallback {
                        res.push(AttributeBase::TypeVarType(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        ));
                    }
                    Some(AttributeBase::Union(res))
                }
                Restriction::Unrestricted => Some(AttributeBase::TypeVarType(
                    (*quantified).clone(),
                    self.stdlib.object().clone(),
                )),
            },
            Type::Type(box Type::Any(style)) => Some(AttributeBase::TypeAny(style)),
            // At runtime, these special forms are classes. This has been tested with Python
            // versions 3.11-3.13. Note that other special forms are classes in some versions, but
            // their representations aren't stable across versions.
            //
            // We don't have access to the class definitions, so the best we can do is model these
            // as type[Any].
            Type::Type(box Type::SpecialForm(
                SpecialForm::Callable
                | SpecialForm::Generic
                | SpecialForm::Protocol
                | SpecialForm::Tuple,
            )) => Some(AttributeBase::TypeAny(AnyStyle::Implicit)),
            Type::Type(box Type::SpecialForm(SpecialForm::Type)) => {
                Some(AttributeBase::ClassObject(ClassBase::ClassDef(
                    self.stdlib.builtins_type().class_object().dupe(),
                )))
            }
            Type::Module(module) => Some(AttributeBase::Module(module)),
            Type::TypeVar(_) | Type::Type(box Type::TypeVar(_)) => {
                Some(AttributeBase::ClassInstance(self.stdlib.type_var().clone()))
            }
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
            Type::KwCall(call) => self.as_attribute_base(call.return_ty),
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
            Type::Forall(forall) => self.as_attribute_base(forall.body.as_type()),
            Type::Var(v) => self.as_attribute_base(self.force_var_for_attribute_base(v)),
            Type::Type(box Type::Var(v)) => {
                self.as_attribute_base(Type::type_form(self.force_var_for_attribute_base(v)))
            }
            Type::SuperInstance(box (cls, obj)) => Some(AttributeBase::SuperInstance(cls, obj)),
            Type::Union(members) => {
                let mut res = Vec::new();
                for member in members {
                    if let Some(attr_base) = self.as_attribute_base(member) {
                        res.push(attr_base)
                    } else {
                        return None;
                    }
                }
                Some(AttributeBase::Union(res))
            }
            Type::Type(box Type::Union(members)) => {
                let mut res = Vec::new();
                for member in members {
                    if let Some(attr_base) = self.as_attribute_base(Type::type_form(member)) {
                        res.push(attr_base)
                    } else {
                        return None;
                    }
                }
                Some(AttributeBase::Union(res))
            }
            Type::Quantified(quantified) => match quantified.restriction() {
                Restriction::Bound(upper_bound) => {
                    let mut res = Vec::new();
                    let mut use_fallback = false;
                    self.map_over_union(upper_bound, |bound| {
                        let bound_attr_base = self.as_attribute_base(bound.clone());
                        if let Some(AttributeBase::ClassInstance(cls)) = bound_attr_base {
                            res.push(AttributeBase::TypeVar((*quantified).clone(), Some(cls)));
                        } else {
                            use_fallback = true;
                        }
                    });
                    if use_fallback {
                        res.push(AttributeBase::TypeVar((*quantified).clone(), None));
                    }
                    Some(AttributeBase::Union(res))
                }
                Restriction::Constraints(constraints) => {
                    let mut res = Vec::new();
                    let mut use_fallback = false;
                    for constraint in constraints {
                        let constraint_attr_base = self.as_attribute_base(constraint.clone());
                        if let Some(AttributeBase::ClassInstance(cls)) = constraint_attr_base {
                            res.push(AttributeBase::TypeVar((*quantified).clone(), Some(cls)));
                        } else {
                            use_fallback = true;
                        }
                    }
                    if use_fallback {
                        res.push(AttributeBase::TypeVar((*quantified).clone(), None));
                    }
                    Some(AttributeBase::Union(res))
                }
                Restriction::Unrestricted => Some(AttributeBase::TypeVar(
                    (*quantified).clone(),
                    Some(self.stdlib.object().clone()),
                )),
            },
            // TODO: check to see which ones should have class representations
            Type::SpecialForm(_)
            | Type::Type(_)
            | Type::Intersect(_)
            | Type::Unpack(_)
            | Type::Concatenate(_, _)
            | Type::ParamSpecValue(_) => None,
        }
    }

    /// Compute the get (i.e. read) type information of an attribute for narrowing.
    ///
    /// We assume that any attribute read coming from a method call (be it a descriptor
    /// of some sort, including property, or `__getattr__` / `__getattribute__`)
    /// is idempotent, and allow narrowing that will be unsound if it is not.
    pub fn narrowable_for_attr(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let fall_back_to_object = || Type::ClassType(self.stdlib.object().clone());
        let (found, not_found, internal_errors) = self.lookup_attr(base, attr_name).decompose();
        let mut results = Vec::new();
        for attr in found {
            let found_ty = match self.resolve_get_access(attr, range, errors, None) {
                Err(..) => fall_back_to_object(),
                Ok(ty) => ty,
            };
            results.push(found_ty);
        }
        if !(not_found.is_empty() && internal_errors.is_empty()) {
            results.push(fall_back_to_object());
        }
        self.unions(results)
    }

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
                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                format!(
                    "`{}.__bool__` has type `{}`, which is not callable",
                    self.for_display(condition_type.clone()),
                    self.for_display(ty.clone()),
                ),
            );
        }
    }

    fn resolve_as_instance_method(&self, attr: Attribute) -> Option<Type> {
        match attr.inner {
            // TODO(stroxler): ReadWrite attributes are not actually methods but limiting access to
            // ReadOnly breaks unit tests; we should investigate callsites to understand this better.
            // NOTE(grievejia): We currently do not expect to use `__getattr__` for this lookup.
            AttributeInner::Simple(ty, Visibility::ReadOnly(_))
            | AttributeInner::Simple(ty, Visibility::ReadWrite) => Some(ty),
            AttributeInner::NoAccess(_)
            | AttributeInner::Property(..)
            | AttributeInner::Descriptor(..)
            | AttributeInner::GetAttr(..)
            | AttributeInner::ModuleFallback(..) => None,
        }
    }

    /// A convenience helper for parts of the code where we have a class and we want to look up
    /// a method if it exists (typically when handling special semantics around dunder methods
    /// and metaclass behavior), without producing type errors on lookup failures.
    pub fn resolve_instance_method(&self, cls: &ClassType, name: &Name) -> Option<Type> {
        self.get_instance_attribute(cls, name)
            .and_then(|attr| self.resolve_as_instance_method(attr))
    }

    /// Return `__call__` as a bound method if instances of `cls` have `__call__`.
    /// This is what the runtime automatically does when we try to call an instance.
    pub fn instance_as_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        self.resolve_instance_method(cls, &dunder::CALL)
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
            .and_then(|attr| self.resolve_as_instance_method(attr))
    }
}

#[derive(Debug)]
pub enum AttrDefinition {
    FullyResolved(TextRangeWithModule),
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
        let mro = self.get_mro_for_class(cls);
        let mut seen = SmallSet::new();
        for c in iter::once(cls).chain(mro.ancestors(self.stdlib).map(|x| x.class_object())) {
            if c == self.stdlib.object().class_object() {
                // Don't want to suggest `__hash__`
                break;
            }
            match expected_attribute_name {
                None => {
                    for fld in c.fields() {
                        if seen.insert(fld)
                            && let Some(range) = c.field_decl_range(fld)
                        {
                            res.push(AttrInfo {
                                name: fld.clone(),
                                ty: None,
                                definition: Some(AttrDefinition::FullyResolved(
                                    TextRangeWithModule::new(c.module().dupe(), range),
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
                                TextRangeWithModule::new(c.module().dupe(), range),
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
        module: &ModuleType,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        let module_name = ModuleName::from_parts(module.parts());
        if let Some(exports) = self.get_module_exports(module_name) {
            match expected_attribute_name {
                None => {
                    res.extend(exports.exports(self.exports).iter().map(|(x, _)| AttrInfo {
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
                        .exports(self.exports)
                        .get(expected_attribute_name)
                        .is_some()
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

    // `base` is expected to be neither a union nor an intersection type
    // if this precondition doesn't hold, the function won't crash but it
    // will also not find an answer.
    pub fn completions_no_union_intersection(
        &self,
        base: Type,
        expected_attribute_name: Option<&Name>,
        include_types: bool,
    ) -> Vec<AttrInfo> {
        let mut res = Vec::new();
        if let Some(base) = self.as_attribute_base(base) {
            match &base {
                AttributeBase::ClassInstance(class)
                | AttributeBase::EnumLiteral(LitEnum { class, .. })
                | AttributeBase::TypeVar(_, Some(class)) => {
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
                    self.completions_class(class.class_object(), expected_attribute_name, &mut res)
                }
                AttributeBase::TypeVarType(_, class) => {
                    self.completions_class(class.class_object(), expected_attribute_name, &mut res)
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
                AttributeBase::Union(_) => {
                    // TODO: handle unions
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
                    {
                        let found_attrs = self
                            .lookup_attr_from_attribute_base(base.clone(), &info.name)
                            .found;
                        let found_types: Vec<_> = found_attrs
                            .into_iter()
                            .filter_map(|attr| {
                                let result = self
                                    .resolve_get_access(
                                        attr,
                                        // Important we do not use the resolved TextRange, as it might be in a different module.
                                        // Whereas the empty TextRange is valid for all modules.
                                        TextRange::default(),
                                        &self.error_swallower(),
                                        None,
                                    )
                                    .ok();
                                if matches!(&result, Some(Type::Any(_))) {
                                    None
                                } else {
                                    result
                                }
                            })
                            .collect();
                        if !found_types.is_empty() {
                            info.ty = Some(self.unions(found_types));
                        }
                    }
                }
            }
        }
        res
    }

    /// List all the attributes available from a type. Used to power completion.
    /// Not all usages need types, so we can skip type computation with `include_types=false`.
    pub fn completions(
        &self,
        base: Type,
        expected_attribute_name: Option<&Name>,
        include_types: bool,
    ) -> Vec<AttrInfo> {
        // TODO:
        // - If `base` is a union, expose only attributes shared by all members
        // - If `base` is an intersection, expose all possible attributes for any members
        self.completions_no_union_intersection(base, expected_attribute_name, include_types)
    }
}
