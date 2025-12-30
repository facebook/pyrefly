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
use pyrefly_types::typed_dict::TypedDictInner;
use pyrefly_types::types::Forall;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Union;
use pyrefly_types::types::Var;
use pyrefly_util::suggest::best_suggestion;
use ruff_python_ast::helpers::is_dunder;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::class::class_field::ClassAttribute;
use crate::alt::expr::TypeOrExpr;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyExport;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::export::exports::Exports;
use crate::solver::solver::SubsetError;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::PropertyMetadata;
use crate::types::callable::PropertyRole;
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
use crate::types::types::BoundMethodType;
use crate::types::types::Overload;
use crate::types::types::SuperObj;
use crate::types::types::Type;

/// The result of looking up an attribute from a particular base.
/// If the base type is a union, multiple results can be returned
/// since each union member is looked up separately.
#[derive(Debug)]
struct LookupResult {
    /// The lookup was successful and an attribute was found.
    pub found: Vec<(Attribute, AttributeBase1)>,
    /// The attribute was not found. Callers can use fallback behavior, for
    /// example looking up a different attribute.
    pub not_found: Vec<NotFoundOn>,
    /// There was a Pyrefly-internal error
    pub internal_error: Vec<InternalError>,
}

#[derive(Debug, Clone)]
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
        subset_error: SubsetError,
    },
    // `got` and `want` are not subtypes of each other
    // applies to read-write attributes
    Invariant {
        got: Type,
        want: Type,
        subset_error: SubsetError,
    },
    // `want` is not a subtype of `got`
    // applies to property setters
    Contravariant {
        got: Type,
        want: Type,
        got_is_property: bool,
        subset_error: SubsetError,
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
                subset_error: _,
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
            AttrSubsetError::Invariant {
                got,
                want,
                subset_error: _,
            } => {
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
                subset_error: _,
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

/// The result of an attempt to access an attribute (which will eventually be
/// used either for an action like get / set / delete, or in a structural subtype
/// check).
#[derive(Debug)]
enum Attribute {
    /// An attribute resolved through a class field lookup.
    ClassAttribute(ClassAttribute),
    /// A read-write attribute with a closed form type for both get and set actions. Used
    /// for non-class-attribute cases (for example, reads against Any, Never, or module objects)
    Simple(Type),
    /// The attribute being looked up is not defined explicitly, but it may be defined via a
    /// `__getattr__` or `__getattribute__` fallback.
    /// The `NotFound` field stores the (failed) lookup result on the original attribute for
    /// better error reporting downstream. The `Attribute` field stores the (successful)
    /// lookup result of the `__getattr__`/`__getattribute__` function or method.
    /// The `Name` field stores the name of the original attribute being looked up.
    GetAttr(NotFoundOn, Box<Attribute>, Name),
    /// We did `a.b`, which is a real module on the file system, but not one the user explicitly
    /// or implicitly imported. In some cases, treat this as NotFound. In others, emit an error
    /// but continue on with type.
    ModuleFallback(NotFoundOn, ModuleName, Type),
}

#[derive(Clone, Debug)]
enum NotFoundOn {
    ClassInstance(Class, AttributeBase1),
    ClassObject(Class, AttributeBase1),
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
    /// Calling a method via `super()` when no implementation is available (e.g. abstract protocol or abstract base method).
    SuperMethodNeedsImplementation(Class),
}

#[derive(Debug)]
enum InternalError {
    /// An internal error caused by `as_attribute_base` being partial.
    AttributeBaseUndefined(Type),
}

impl Attribute {
    pub fn simple(ty: Type) -> Self {
        Self::Simple(ty)
    }

    pub fn class_attribute(class_attr: ClassAttribute) -> Self {
        Self::ClassAttribute(class_attr)
    }

    fn getattr(not_found: NotFoundOn, getattr: Self, name: Name) -> Self {
        Self::GetAttr(not_found, Box::new(getattr), name)
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
            NoAccessReason::SuperMethodNeedsImplementation(class) => {
                let class_name = class.name();
                format!(
                    "Method `{attr_name}` inherited from class `{class_name}` has no implementation and cannot be accessed via `super()`"
                )
            }
        }
    }
}

impl LookupResult {
    fn empty() -> Self {
        Self {
            found: Vec::new(),
            not_found: Vec::new(),
            internal_error: Vec::new(),
        }
    }

    /// We found a simple attribute type.
    ///
    /// This means we assume it is both readable and writable with that type.
    ///
    /// TODO(stroxler) The uses of this eventually need to be audited, but we
    /// need to prioritize the class logic first.
    fn found_type(&mut self, ty: Type, on: AttributeBase1) {
        self.found(Attribute::simple(ty), on)
    }

    fn found(&mut self, attr: Attribute, on: AttributeBase1) {
        self.found.push((attr, on))
    }

    fn found_class_attribute(&mut self, class_attr: ClassAttribute, on: AttributeBase1) {
        self.found
            .push((Attribute::class_attribute(class_attr), on))
    }

    fn not_found(&mut self, not_found: NotFoundOn) {
        self.not_found.push(not_found);
    }

    fn internal_error(internal_error: InternalError) -> Self {
        Self {
            found: Vec::new(),
            not_found: Vec::new(),
            internal_error: vec![internal_error],
        }
    }

    fn decompose(
        self,
    ) -> (
        Vec<(Attribute, AttributeBase1)>,
        Vec<NotFoundOn>,
        Vec<InternalError>,
    ) {
        (self.found, self.not_found, self.internal_error)
    }
}

impl NotFoundOn {
    pub fn to_error_msg(self, attr_name: &Name) -> String {
        match self {
            NotFoundOn::ClassInstance(class, _) => {
                let class_name = class.name();
                format!("Object of class `{class_name}` has no attribute `{attr_name}`")
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

    fn attr_base1(&self) -> AttributeBase1 {
        match self {
            NotFoundOn::ClassInstance(_, base) | NotFoundOn::ClassObject(_, base) => base.clone(),
            NotFoundOn::Module(module) => AttributeBase1::Module(module.clone()),
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
/// Note that a single type can correspond to multiple "atomic" bases, so all
/// attribute lookups should be prepared to handle unions.
#[derive(Clone, Debug)]
struct AttributeBase(Vec1<AttributeBase1>);

#[derive(Clone, Debug)]
struct PropertyAttr {
    getter: Type,
    setter: Option<Type>,
    deleter: bool,
}

/// A single, "atomic" attribute base, not coming from a union or an intersection.
/// An attribute is either found or not found by a search on this.
#[derive(Clone, Debug)]
enum AttributeBase1 {
    EnumLiteral(LitEnum),
    LiteralString,
    ClassInstance(ClassType),
    ClassObject(ClassBase),
    Module(ModuleType),
    /// Attribute access on a type parameter in a value position.
    /// This is almost always `P.args` or `P.kwargs`.
    QuantifiedValue(Quantified),
    /// Attribute access on a quantified value, i.e. a value with type `T` for
    /// some in-scope type variable `T`. The optional `ClassType` is an upper
    /// bound, which may be the original bound on `T` or a decomposition of
    /// it (e.g. if the original bound is a union).
    Quantified(Quantified, ClassType),
    Any(AnyStyle),
    Never,
    /// type[Any] is a special case where attribute lookups first check the
    /// builtin `type` class before falling back to `Any`.
    TypeAny(AnyStyle),
    /// type[Never] is a special case where attribute lookups first check the builtin `type` class
    /// before falling back to `Never`.
    TypeNever,
    /// Properties are handled via a special case so that we can understand
    /// setter/deleter decorators.
    Property(PropertyAttr),
    /// Attribute access on `Self` from inside a class
    SelfType(ClassType),
    /// Result of a super() call. See Type::SuperInstance for details on what these fields are.
    SuperInstance(ClassType, SuperObj),
    /// Typed dictionaries have similar properties to dict and Mapping, with some exceptions
    TypedDict(TypedDictInner),
    /// Attribute lookup on a base as part of a subset check against a protocol.
    ProtocolSubset(Box<AttributeBase1>),
    Intersect(Vec<AttributeBase1>, Vec<AttributeBase1>),
    /// Bound methods prefer exposing builtin `types.MethodType` attributes but fall back to the
    /// underlying function's attributes when the builtin ones are missing.
    BoundMethod(BoundMethodType),
}

impl AttributeBase1 {
    fn to_attr_base(self) -> AttributeBase {
        AttributeBase(Vec1::new(self))
    }
}

/// A normalized type for attribute lookup which has "class-like" lookup behavior. For example,
/// when we look up an instance method from a class base, we get the unbound function type.

#[derive(Clone, Debug)]
pub enum ClassBase {
    ClassDef(ClassType),
    ClassType(ClassType),
    Quantified(Quantified, ClassType),
    SelfType(ClassType),
    Protocol(ClassType, Type),
}

impl ClassBase {
    pub fn class_type(&self) -> &ClassType {
        match self {
            ClassBase::ClassDef(c)
            | ClassBase::ClassType(c)
            | ClassBase::Quantified(_, c)
            | ClassBase::SelfType(c)
            | ClassBase::Protocol(c, _) => c,
        }
    }

    pub fn class_object(&self) -> &Class {
        self.class_type().class_object()
    }

    pub fn targs(&self) -> Option<&TArgs> {
        match self {
            ClassBase::ClassDef(..) => None,
            ClassBase::ClassType(c)
            | ClassBase::Quantified(_, c)
            | ClassBase::SelfType(c)
            | ClassBase::Protocol(c, _) => Some(c.targs()),
        }
    }

    pub fn to_type(self) -> Type {
        match self {
            ClassBase::ClassDef(c) => Type::ClassDef(c.into_class_object()),
            ClassBase::ClassType(c) => Type::Type(Box::new(c.to_type())),
            ClassBase::Quantified(q, _) => Type::type_form(q.to_type()),
            ClassBase::SelfType(c) => Type::type_form(Type::SelfType(c)),
            ClassBase::Protocol(_, self_type) => Type::type_form(self_type),
        }
    }

    pub fn to_self_type(self) -> Type {
        match self {
            ClassBase::ClassDef(c) => c.to_type(),
            ClassBase::ClassType(c) => c.to_type(),
            ClassBase::Quantified(q, _) => q.to_type(),
            ClassBase::SelfType(c) => Type::SelfType(c),
            ClassBase::Protocol(_, self_type) => self_type,
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
        let lookup_result = attr_base.clone().map_or_else(
            || LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone())),
            |attr_base| self.lookup_attr_from_base(attr_base, attr_name),
        );
        let mut types = Vec::new();
        let mut error_messages = Vec::new();
        let (found, not_found, error) = lookup_result.decompose();
        for (attr, _) in found {
            match self.resolve_get_access(attr_name, attr, range, errors, context) {
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

        // Both types and error messages can be duplicated if elements in `attr_base` gets duplicated (can happen with
        // if base type contain vars). Make sure that dedup logic applies to both branches.
        if error_messages.is_empty() {
            self.unions(types)
        } else {
            error_messages.sort();
            error_messages.dedup();
            let mut msg = vec1![error_messages.join("\n")];
            if let Some(suggestion) = attr_base
                .as_ref()
                .and_then(|attr_base| self.suggest_attribute_name(attr_name, attr_base))
            {
                msg.push(format!("Did you mean `{suggestion}`?"));
            }
            errors.add(
                range,
                ErrorInfo::new(ErrorKind::MissingAttribute, context),
                msg,
            );
            Type::any_error()
        }
    }

    fn add_class_fields(&self, class: &Class, candidates: &mut SmallSet<Name>) {
        let mut add_fields_for = |cls: &Class| {
            for name in cls.fields() {
                candidates.insert(name.clone());
            }
        };
        add_fields_for(class);
        let mro = self.get_mro_for_class(class);
        for ancestor in mro.ancestors_no_object() {
            add_fields_for(ancestor.class_object());
        }
    }

    fn collect_attribute_candidates_from_base(
        &self,
        base: &AttributeBase1,
        candidates: &mut SmallSet<Name>,
    ) {
        match base {
            AttributeBase1::ClassInstance(class) => {
                self.add_class_fields(class.class_object(), candidates);
            }
            AttributeBase1::ClassObject(class_base) => {
                self.add_class_fields(class_base.class_object(), candidates);
            }
            AttributeBase1::LiteralString => {
                self.add_class_fields(self.stdlib.str().class_object(), candidates);
            }
            AttributeBase1::TypedDict(td) => {
                self.add_class_fields(td.class_object(), candidates);
            }
            AttributeBase1::SuperInstance(class, ..) => {
                self.add_class_fields(class.class_object(), candidates);
            }
            AttributeBase1::Quantified(_, class_type) => {
                self.add_class_fields(class_type.class_object(), candidates);
            }
            AttributeBase1::SelfType(class_type) => {
                self.add_class_fields(class_type.class_object(), candidates);
            }
            AttributeBase1::ProtocolSubset(inner) => {
                self.collect_attribute_candidates_from_base(inner, candidates);
            }
            AttributeBase1::Intersect(options, fallback) => {
                for b in options {
                    self.collect_attribute_candidates_from_base(b, candidates);
                }
                for b in fallback {
                    self.collect_attribute_candidates_from_base(b, candidates);
                }
            }
            _ => {}
        }
    }

    fn suggest_attribute_name(&self, missing: &Name, attr_base: &AttributeBase) -> Option<Name> {
        let mut candidates = SmallSet::new();
        for base in attr_base.0.iter() {
            self.collect_attribute_candidates_from_base(base, &mut candidates);
        }
        best_suggestion(missing, candidates.iter().map(|candidate| (candidate, 0)))
    }

    /// Can the attribute be successfully looked up in all cases?
    pub fn has_attr(&self, base: &Type, attr_name: &Name) -> bool {
        if let Some(attr_base) = self.as_attribute_base(base.clone()) {
            let lookup_result = self.lookup_attr_from_base(attr_base, attr_name);
            return lookup_result.internal_error.is_empty() && lookup_result.not_found.is_empty();
        }
        false
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
        allow_getattr_fallback: bool,
    ) -> Option<Type> {
        let mut not_found = false;
        let mut attr_tys = Vec::new();
        let lookup_result = match self.as_attribute_base(base.clone()) {
            None => {
                LookupResult::internal_error(InternalError::AttributeBaseUndefined(base.clone()))
            }
            Some(base) => {
                let direct_lookup_result = self.lookup_magic_dunder_attr(base.clone(), attr_name);
                if allow_getattr_fallback {
                    self.lookup_attr_from_base_getattr_fallback(attr_name, direct_lookup_result)
                } else {
                    direct_lookup_result
                }
            }
        };
        for (attr, _) in lookup_result.found {
            attr_tys.push(
                self.resolve_get_access(attr_name, attr, range, errors, context)
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
        allow_assign_to_final: bool,
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
            allow_assign_to_final,
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
        for (setattr_attr, _) in setattr_found {
            let result = self
                .resolve_get_access(attr_name, setattr_attr, range, errors, context)
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
        for (delattr_attr, _) in delattr_found {
            let result = self
                .resolve_get_access(attr_name, delattr_attr, range, errors, context)
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
        allow_assign_to_final: bool,
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
        for (attr, found_on) in lookup_found {
            match attr {
                // Attribute setting bypasses `__getattr__` lookup and checks `__setattr__`
                // If the attribute is not found, we fall back to `__setattr__`
                Attribute::GetAttr(not_found, _, _)
                | Attribute::ModuleFallback(not_found, _, _) => {
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
                Attribute::Simple(attr_ty) => {
                    self.check_set_read_write_and_infer_narrow(
                        attr_ty,
                        attr_name,
                        got,
                        range,
                        errors,
                        context,
                        should_narrow,
                        &mut narrowed_types,
                    );
                }
                Attribute::ClassAttribute(class_attr) => {
                    // If we are writing to an instance, we may need access to
                    // the class to special-case dataclass converters.
                    let instance_class = match &found_on {
                        AttributeBase1::ClassInstance(cls) => Some(cls),
                        _ => None,
                    };
                    let class_base = match &found_on {
                        AttributeBase1::ClassObject(cls_base) => Some(cls_base),
                        _ => None,
                    };
                    self.check_class_attr_set_and_infer_narrow(
                        class_attr,
                        instance_class,
                        class_base,
                        attr_name,
                        got,
                        allow_assign_to_final,
                        range,
                        errors,
                        context,
                        &mut should_narrow,
                        &mut narrowed_types,
                    );
                }
            }
        }
        if should_narrow {
            Some(self.unions(narrowed_types))
        } else {
            None
        }
    }

    pub fn check_set_read_write_and_infer_narrow(
        &self,
        attr_ty: Type,
        attr_name: &Name,
        got: TypeOrExpr,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        should_narrow: bool,
        narrowed_types: &mut Vec<Type>,
    ) {
        let ty = match &got {
            TypeOrExpr::Expr(got) => self.expr(
                got,
                Some((&attr_ty, &|| TypeCheckContext {
                    kind: TypeCheckKind::Attribute(attr_name.clone()),
                    context: context.map(|ctx| ctx()),
                })),
                errors,
            ),
            TypeOrExpr::Type(got, _) => {
                self.check_type(got, &attr_ty, range, errors, &|| TypeCheckContext {
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
        for (attr, _) in lookup_found {
            match attr {
                // Attribute deletion bypasses `__getattr__` lookup and checks `__delattr__`
                // If the attribute is not found, we fall back to `__delattr__`
                Attribute::GetAttr(not_found, _, _)
                | Attribute::ModuleFallback(not_found, _, _) => {
                    self.check_delattr(
                        attr_base.clone(),
                        attr_name,
                        not_found,
                        range,
                        errors,
                        context,
                    );
                }
                Attribute::Simple(_) => {
                    // Allow deleting most attributes for now, for compatibility with mypy.
                }
                Attribute::ClassAttribute(class_attr) => {
                    self.check_class_attr_delete(class_attr, attr_name, range, errors, context);
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
        attr_name: &Name,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), SubsetError> {
        if let Some(got_attrs) = self
            .as_attribute_base(got.clone())
            .map(|got_base| {
                let got_base = AttributeBase(
                    got_base
                        .0
                        .mapped(|base| AttributeBase1::ProtocolSubset(Box::new(base))),
                );
                self.lookup_attr_from_base(got_base, attr_name)
            })
            .and_then(|lookup_result| {
                if lookup_result.not_found.is_empty() && lookup_result.internal_error.is_empty() {
                    Some(lookup_result.found)
                } else {
                    None
                }
            })
        {
            // `as_attribute_base` promotes literals, so we should promote here too
            // In the future, we should refactor `get_protocol_attribute` to reuse the `AttributeBase`, to ensure the logic is identical
            let got = got.clone().promote_literals(self.stdlib);
            if (!got_attrs.is_empty())
                && let Some(want) = self.get_protocol_attribute(protocol, got.clone(), attr_name)
            {
                for (got_attr, _) in got_attrs.iter() {
                    self.is_attribute_subset(got_attr, &want, &mut |got, want| {
                        is_subset(got, want)
                    })
                    .map_err(|err| {
                        SubsetError::IncompatibleAttribute(Box::new((
                            protocol.name().clone(),
                            self.for_display(got.clone()),
                            attr_name.clone(),
                            err,
                        )))
                    })?;
                }
                Ok(())
            } else {
                Err(SubsetError::MissingAttribute(
                    protocol.name().clone(),
                    attr_name.clone(),
                ))
            }
        } else {
            Err(SubsetError::MissingAttribute(
                protocol.name().clone(),
                attr_name.clone(),
            ))
        }
    }

    fn is_attribute_subset(
        &self,
        got: &Attribute,
        want: &ClassAttribute,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), AttrSubsetError> {
        match got {
            Attribute::ClassAttribute(got_class_attr) => {
                self.is_class_attribute_subset(got_class_attr, want, is_subset)
            }
            Attribute::Simple(got_ty) => {
                // Treat Simple attributes (which come up for cases like attribute access
                // on modules, Any, and Never) as if they were read-write class attributes
                // for the purpose of protocol subtyping.
                let synthetic_got = ClassAttribute::read_write(got_ty.clone());
                self.is_class_attribute_subset(&synthetic_got, want, is_subset)
            }
            Attribute::GetAttr(..) => {
                // NOTE(grievejia): `__getattr__` does not participate in structural subtyping
                // check for now. We may revisit this in the future if the need comes.
                Err(AttrSubsetError::Getattr)
            }
            Attribute::ModuleFallback(..) => Err(AttrSubsetError::ModuleFallback),
        }
    }

    fn resolve_get_access(
        &self,
        attr_name: &Name,
        attr: Attribute,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Result<Type, NoAccessReason> {
        match attr {
            Attribute::ClassAttribute(class_attr) => {
                self.resolve_get_class_attr(attr_name, class_attr, range, errors, context)
            }
            Attribute::Simple(ty) => Ok(ty),
            Attribute::ModuleFallback(_, name, ty) => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::ImplicitImport, context),
                    format!("Module `{name}` exists, but was not imported explicitly. You are relying on other modules to load it."),
                );
                Ok(ty)
            }
            Attribute::GetAttr(_, getattr_attr, name) => self
                .resolve_get_access(attr_name, *getattr_attr, range, errors, context)
                .map(|getattr_ty| {
                    self.call_getattr_or_delattr(getattr_ty, name, range, errors, context)
                }),
        }
    }
    fn lookup_attr_from_attribute_base(
        &self,
        base: AttributeBase,
        attr_name: &Name,
    ) -> LookupResult {
        let mut acc = LookupResult::empty();
        for base1 in base.0 {
            self.lookup_attr_from_attribute_base1(base1, attr_name, &mut acc);
        }
        acc
    }

    /// Try looking up this attribute as a magic dunder method that follows special attribute
    /// lookup rules, returning the attribute if we succeed.
    fn try_get_magic_dunder_attr(
        &self,
        cls: &ClassBase,
        attr_name: &Name,
    ) -> Option<ClassAttribute> {
        if !is_dunder(attr_name)
            // Constructors and the dataclass __post_init__ method are special-cased elsewhere and
            // should not go through magic dunder lookup.
            || [dunder::NEW, dunder::INIT, dunder::POST_INIT]
                .iter()
                .any(|constructor| constructor == attr_name)
        {
            return None;
        }
        let metadata = self.get_metadata_for_class(cls.class_object());
        let metaclass = metadata.metaclass(self.stdlib);
        let attr = self.get_metaclass_attribute(cls, metaclass, attr_name)?;
        attr.clone().as_instance_method().map(|_| attr)
    }

    /// Helper for looking up attributes on `type[T]` wrappers.
    /// First checks the builtin `type` class, then falls back to the provided fallback.
    fn lookup_attr_from_type_wrapper(
        &self,
        attr_name: &Name,
        fallback: impl FnOnce() -> Type,
    ) -> Type {
        let builtins_type_classtype = self.stdlib.builtins_type();
        self.get_instance_attribute(builtins_type_classtype, attr_name)
            .and_then(|attr| attr.as_instance_method())
            .unwrap_or_else(fallback)
    }

    fn lookup_attr_from_attribute_base1(
        &self,
        base: AttributeBase1,
        attr_name: &Name,
        acc: &mut LookupResult,
    ) {
        match &base {
            AttributeBase1::Any(style) => acc.found_type(style.propagate(), base),
            AttributeBase1::TypeAny(style) => {
                let ty = self.lookup_attr_from_type_wrapper(attr_name, || style.propagate());
                acc.found_type(ty, base);
            }
            AttributeBase1::TypeNever => {
                let ty = self.lookup_attr_from_type_wrapper(attr_name, Type::never);
                acc.found_type(ty, base);
            }
            AttributeBase1::Never => acc.found_type(Type::never(), base),
            AttributeBase1::EnumLiteral(e) if matches!(attr_name.as_str(), "name" | "_name_") => {
                acc.found_type(Type::Literal(Lit::Str(e.member.as_str().into())), base)
            }
            AttributeBase1::LiteralString => match self.get_literal_string_attribute(attr_name) {
                Some(attr) => acc.found_class_attribute(attr, base),
                None => acc.not_found(NotFoundOn::ClassInstance(
                    self.stdlib.str().class_object().dupe(),
                    base,
                )),
            },
            AttributeBase1::ClassInstance(class) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let attr_lookup_result =
                    self.get_enum_or_instance_attribute(class, &metadata, attr_name);
                match attr_lookup_result {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None if metadata.has_base_any() => {
                        acc.found_type(Type::Any(AnyStyle::Implicit), base)
                    }
                    None => {
                        acc.not_found(NotFoundOn::ClassInstance(class.class_object().dupe(), base))
                    }
                }
            }
            AttributeBase1::EnumLiteral(lit @ LitEnum { class, .. }) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let attr_lookup_result =
                    self.get_enum_literal_or_instance_attribute(lit, &metadata, attr_name);
                match attr_lookup_result {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None if metadata.has_base_any() => {
                        acc.found_type(Type::Any(AnyStyle::Implicit), base)
                    }
                    None => {
                        acc.not_found(NotFoundOn::ClassInstance(class.class_object().dupe(), base))
                    }
                }
            }
            AttributeBase1::SuperInstance(cls, obj) => {
                match self.get_super_attribute(cls, obj, attr_name) {
                    Some(attr) => acc.found_class_attribute(
                        attr.read_only_equivalent(ReadOnlyReason::Super),
                        base,
                    ),
                    None if let SuperObj::Instance(cls) = obj
                        && self.extends_any(cls.class_object()) =>
                    {
                        acc.found_class_attribute(
                            ClassAttribute::read_only(
                                Type::Any(AnyStyle::Implicit),
                                ReadOnlyReason::Super,
                            ),
                            base,
                        )
                    }
                    None if let SuperObj::Class(cls) = obj
                        && self.extends_any(cls.class_object()) =>
                    {
                        acc.found_class_attribute(
                            ClassAttribute::read_only(
                                Type::Any(AnyStyle::Implicit),
                                ReadOnlyReason::Super,
                            ),
                            base,
                        )
                    }
                    None => {
                        acc.not_found(NotFoundOn::ClassInstance(cls.class_object().dupe(), base))
                    }
                }
            }
            AttributeBase1::QuantifiedValue(quantified) => {
                match (quantified.kind(), attr_name.as_str()) {
                    (QuantifiedKind::ParamSpec, "args") => {
                        acc.found_type(Type::ArgsValue(Box::new(quantified.clone())), base)
                    }
                    (QuantifiedKind::ParamSpec, "kwargs") => {
                        acc.found_type(Type::KwargsValue(Box::new(quantified.clone())), base)
                    }
                    _ => self.lookup_attr_from_attribute_base1(
                        AttributeBase1::ClassInstance(quantified.class_type(self.stdlib).clone()),
                        attr_name,
                        acc,
                    ),
                }
            }
            AttributeBase1::ProtocolSubset(protocol_base) => {
                if let AttributeBase1::ClassObject(class) = &**protocol_base
                    && let Some(attr) = self.try_get_magic_dunder_attr(class, attr_name)
                {
                    // When looking up a magic dunder method as part of checking a class object
                    // against a protocol, we prefer methods on the metaclass over methods on the
                    // class object. See test::enums::test_iterate for why we need to do this.
                    acc.found_class_attribute(attr, base)
                } else {
                    self.lookup_attr_from_attribute_base1((**protocol_base).clone(), attr_name, acc)
                }
            }
            AttributeBase1::BoundMethod(bound_func) => {
                let method_type_base =
                    AttributeBase1::ClassInstance(self.stdlib.method_type().clone());
                let found_len = acc.found.len();
                let not_found_len = acc.not_found.len();
                let error_len = acc.internal_error.len();
                self.lookup_attr_from_attribute_base1(method_type_base, attr_name, acc);
                if acc.found.len() == found_len {
                    acc.not_found.truncate(not_found_len);
                    acc.internal_error.truncate(error_len);
                    let mut func_bases = Vec::new();
                    self.as_attribute_base1(bound_func.clone().as_type(), &mut func_bases);
                    for base1 in func_bases {
                        self.lookup_attr_from_attribute_base1(base1, attr_name, acc);
                    }
                } else {
                    acc.not_found.truncate(not_found_len);
                    acc.internal_error.truncate(error_len);
                }
            }
            AttributeBase1::ClassObject(class) => {
                let attr = match class {
                    ClassBase::Quantified(quantified, class) => self
                        .get_bounded_quantified_class_attribute(
                            quantified.clone(),
                            class,
                            attr_name,
                        ),
                    _ => self.get_class_attribute(class, attr_name),
                };
                match attr {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None => {
                        // Classes are instances of their metaclass, which defaults to `builtins.type`.
                        // NOTE(grievejia): This lookup serves as fallback for normal class attribute lookup for regular
                        // attributes, but for magic dunder methods it needs to supersede normal class attribute lookup.
                        // See `lookup_magic_dunder_attr()`.
                        let metadata = self.get_metadata_for_class(class.class_object());
                        if metadata.is_new_type() {
                            // NewType values are runtime Python objects (functions). They should behave like ordinary
                            // objects for attribute access even though they don't expose class-level APIs such as `mro`.
                            self.lookup_attr_from_attribute_base1(
                                AttributeBase1::ClassInstance(self.stdlib.object().clone()),
                                attr_name,
                                acc,
                            );
                        } else {
                            let instance_attr = self.get_metaclass_attribute(
                                class,
                                metadata.metaclass(self.stdlib),
                                attr_name,
                            );
                            match instance_attr {
                                Some(attr) => acc.found_class_attribute(attr, base),
                                None if metadata.has_base_any() => {
                                    // We can't immediately fall back to Any in this case -- `type[Any]` is actually a special
                                    // AttributeBase which requires additional lookup on `type` itself before the Any fallback.
                                    self.lookup_attr_from_attribute_base1(
                                        AttributeBase1::TypeAny(AnyStyle::Implicit),
                                        attr_name,
                                        acc,
                                    )
                                }
                                None => acc.not_found(NotFoundOn::ClassObject(
                                    class.class_object().dupe(),
                                    base,
                                )),
                            }
                        }
                    }
                }
            }
            AttributeBase1::Module(module) => match self.get_module_attr(module, attr_name) {
                // TODO(samzhou19815): Support module attribute go-to-definition
                Some(attr) => acc.found(attr, base),
                None => acc.not_found(NotFoundOn::Module(module.clone())),
            },
            AttributeBase1::Quantified(q, bound) => {
                match self.get_bounded_quantified_attribute(q.clone(), bound, attr_name) {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None => {
                        acc.not_found(NotFoundOn::ClassInstance(bound.class_object().dupe(), base))
                    }
                }
            }
            AttributeBase1::Property(property) => {
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
                    let mut getter = property.getter.clone();
                    let metadata_getter = property.getter.without_property_metadata();
                    let metadata_setter = property
                        .setter
                        .as_ref()
                        .map(|setter| setter.without_property_metadata());
                    getter.transform_toplevel_func_metadata(|meta: &mut FuncMetadata| {
                        meta.flags.property_metadata = Some(PropertyMetadata {
                            role: PropertyRole::SetterDecorator,
                            getter: metadata_getter.clone(),
                            setter: metadata_setter.clone(),
                            has_deleter: property.deleter,
                        });
                    });
                    acc.found_type(
                        // TODO(samzhou19815): Support go-to-definition for @property applied symbols
                        getter, base,
                    )
                } else if attr_name == "deleter" {
                    let mut getter = property.getter.clone();
                    getter.transform_toplevel_func_metadata(|meta: &mut FuncMetadata| {
                        meta.flags.property_metadata = Some(PropertyMetadata {
                            role: PropertyRole::DeleterDecorator,
                            getter: property.getter.clone(),
                            setter: property.setter.clone(),
                            has_deleter: true,
                        });
                    });
                    acc.found_type(getter, base)
                } else {
                    let class = self.stdlib.property();
                    match self.get_instance_attribute(class, attr_name) {
                        Some(attr) => acc.found_class_attribute(attr, base),
                        None => acc.not_found(NotFoundOn::ClassInstance(
                            class.class_object().dupe(),
                            base,
                        )),
                    }
                }
            }
            AttributeBase1::TypedDict(typed_dict) => {
                match self.get_typed_dict_attribute(typed_dict, attr_name) {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None => acc.not_found(NotFoundOn::ClassInstance(
                        typed_dict.class_object().dupe(),
                        base,
                    )),
                }
            }
            AttributeBase1::SelfType(cls) => match self.get_self_attribute(cls, attr_name) {
                Some(attr) => acc.found_class_attribute(attr, base),
                None => acc.not_found(NotFoundOn::ClassInstance(cls.class_object().dupe(), base)),
            },
            AttributeBase1::Intersect(bases, fallback) => {
                // For now, only handle the simplest case: if exactly one base has a successful lookup, use it.
                let mut candidates = Vec::new();
                for b in bases {
                    let mut acc_candidate = LookupResult::empty();
                    self.lookup_attr_from_attribute_base1(b.clone(), attr_name, &mut acc_candidate);
                    if acc_candidate.not_found.is_empty() && acc_candidate.internal_error.is_empty()
                    {
                        candidates.push(acc_candidate.found);
                    }
                }
                if candidates.len() == 1 {
                    acc.found.extend(candidates.into_iter().next().unwrap());
                } else {
                    // TODO: Intersect the candidates instead of using the fallback.
                    for b in fallback {
                        self.lookup_attr_from_attribute_base1(b.clone(), attr_name, acc);
                    }
                }
            }
        }
    }

    /// A magic dunder attribute differs from a normal attribute in one crucial aspect:
    /// if looked up from a base of `type[A]` directly, the attribute needs to be defined
    /// on the metaclass instead of class `A` (i.e. we are looking for `type.__magic_dunder_attr__`
    /// instead of `A.__magic_dunder_attr__`).
    fn lookup_magic_dunder_attr(&self, base: AttributeBase, dunder_name: &Name) -> LookupResult {
        let mut acc = LookupResult::empty();
        for base1 in base.0 {
            self.lookup_magic_dunder_attr1(base1, dunder_name, &mut acc);
        }
        acc
    }

    fn lookup_magic_dunder_attr1(
        &self,
        base: AttributeBase1,
        dunder_name: &Name,
        acc: &mut LookupResult,
    ) {
        match &base {
            AttributeBase1::ClassObject(class) => {
                let metadata = self.get_metadata_for_class(class.class_object());
                let metaclass = metadata.metaclass(self.stdlib);
                if *dunder_name == dunder::GETATTRIBUTE
                    && self.field_is_inherited_from(
                        metaclass.class_object(),
                        dunder_name,
                        (ModuleName::builtins().as_str(), "object"),
                    )
                {
                    acc.not_found(NotFoundOn::ClassInstance(
                        metaclass.class_object().clone(),
                        base,
                    ));
                    return;
                }
                match self.get_metaclass_attribute(class, metaclass, dunder_name) {
                    Some(attr) => acc.found_class_attribute(attr, base),
                    None => acc.not_found(NotFoundOn::ClassInstance(
                        metaclass.class_object().clone(),
                        base,
                    )),
                }
            }
            AttributeBase1::ClassInstance(cls)
            | AttributeBase1::SelfType(cls)
            | AttributeBase1::EnumLiteral(LitEnum { class: cls, .. })
            | AttributeBase1::Quantified(_, cls)
            | AttributeBase1::SuperInstance(cls, _)
                if (*dunder_name == dunder::SETATTR
                    || *dunder_name == dunder::DELATTR
                    || *dunder_name == dunder::GETATTRIBUTE)
                    && self.field_is_inherited_from(
                        cls.class_object(),
                        dunder_name,
                        (ModuleName::builtins().as_str(), "object"),
                    ) =>
            {
                acc.not_found(NotFoundOn::ClassInstance(cls.class_object().clone(), base))
            }
            AttributeBase1::LiteralString
                if *dunder_name == dunder::SETATTR
                    || *dunder_name == dunder::DELATTR
                    || *dunder_name == dunder::GETATTRIBUTE =>
            {
                acc.not_found(NotFoundOn::ClassInstance(
                    self.stdlib.str().class_object().clone(),
                    base,
                ))
            }
            AttributeBase1::TypedDict(typed_dict) if *dunder_name == dunder::GETATTRIBUTE => acc
                .not_found(NotFoundOn::ClassInstance(
                    typed_dict.class_object().clone(),
                    base,
                )),
            _ => self.lookup_attr_from_attribute_base1(base, dunder_name, acc),
        }
    }

    fn lookup_attr_from_base_getattr_fallback(
        &self,
        attr_name: &Name,
        mut result: LookupResult,
    ) -> LookupResult {
        let direct_lookup_not_found = std::mem::take(&mut result.not_found);
        for not_found in direct_lookup_not_found {
            let (getattr_found, getattr_not_found, getattr_internal_error) = self
                .lookup_magic_dunder_attr(not_found.attr_base1().to_attr_base(), &dunder::GETATTR)
                .decompose();
            if !(getattr_not_found.is_empty() && getattr_internal_error.is_empty()) {
                // If the `__getattr__` lookup fails, we fall back to `__getattribute__`
                // Note: at runtime, `__getattribute__` is checked BEFORE looking up the attribute by name,
                // but because the declaration is on `object` and returns `Any`, all attribute accesses
                // would return `Any`.
                let (getattribute_found, getattribute_not_found, getattribute_internal_error) =
                    self.lookup_magic_dunder_attr(
                        not_found.attr_base1().to_attr_base(),
                        &dunder::GETATTRIBUTE,
                    )
                    .decompose();
                if !(getattribute_not_found.is_empty() && getattribute_internal_error.is_empty()) {
                    result.not_found.push(not_found.clone())
                } else {
                    for (attr, found_on) in getattribute_found {
                        result.found(
                            Attribute::getattr(not_found.clone(), attr, attr_name.clone()),
                            found_on,
                        );
                    }
                }
            } else {
                for (attr, found_on) in getattr_found {
                    result.found(
                        Attribute::getattr(not_found.clone(), attr, attr_name.clone()),
                        found_on,
                    );
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

    fn get_module_exports(&self, module_name: ModuleName) -> Option<Exports> {
        self.exports.get(module_name).finding()
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
            return Some(Attribute::simple(submodule.to_type()));
        }

        let module_name = ModuleName::from_parts(module.parts());
        let module_exports = match self.get_module_exports(module_name) {
            Some(x) => x,
            None => return Some(Attribute::simple(Type::any_error())), // This module doesn't exist, we must have already errored
        };

        if module_exports.exports(self.exports).contains_key(attr_name) {
            Some(Attribute::simple(
                self.get_from_export(module_name, None, &KeyExport(attr_name.clone()))
                    .arc_clone(),
            ))
        } else if module_exports.is_submodule_imported_implicitly(attr_name)
            && self
                .get_module_exports(module_name.append(attr_name))
                .is_some()
        {
            Some(Attribute::simple(submodule.to_type()))
        } else if self
            .get_module_exports(module_name.append(attr_name))
            .is_some()
        {
            // The module isn't imported, but does exist on disk, so user must
            // be observing someone else's import.
            Some(Attribute::ModuleFallback(
                NotFoundOn::Module(module.clone()),
                module_name.append(attr_name),
                submodule.to_type(),
            ))
        } else {
            None
        }
    }

    fn force_var_for_attribute_base(&self, var: Var, f: impl FnOnce(Type)) {
        if let Some(_guard) = self.recurse(var) {
            // Ensure that the guard is still held when we call `f`, to avoid
            // non-termination when the var appears inside itself.
            f(self.solver().force_var(var))
        } else {
            f(Type::any_implicit())
        }
    }

    fn as_attribute_base(&self, ty: Type) -> Option<AttributeBase> {
        let mut acc = Vec::new();
        self.as_attribute_base1(ty, &mut acc);
        Vec1::try_from_vec(acc).map(AttributeBase).ok()
    }

    fn as_attribute_base1(&self, ty: Type, acc: &mut Vec<AttributeBase1>) {
        match ty {
            Type::ClassType(class_type) => acc.push(AttributeBase1::ClassInstance(class_type)),
            Type::ClassDef(cls) => acc.push(AttributeBase1::ClassObject(ClassBase::ClassDef(
                self.as_class_type_unchecked(&cls),
            ))),
            Type::SelfType(class_type) => acc.push(AttributeBase1::SelfType(class_type)),
            Type::Type(box Type::SelfType(class_type)) => {
                acc.push(AttributeBase1::ClassObject(ClassBase::SelfType(class_type)))
            }
            Type::TypedDict(TypedDict::TypedDict(td))
            | Type::PartialTypedDict(TypedDict::TypedDict(td)) => {
                acc.push(AttributeBase1::TypedDict(td.clone()))
            }
            Type::TypedDict(td @ TypedDict::Anonymous(_))
            | Type::PartialTypedDict(td @ TypedDict::Anonymous(_)) => {
                let value_ty = self.get_typed_dict_value_type(&td);
                acc.push(AttributeBase1::ClassInstance(
                    self.stdlib
                        .dict(self.stdlib.str().clone().to_type(), value_ty),
                ))
            }
            Type::Type(box (Type::TypedDict(_) | Type::PartialTypedDict(_))) => {
                acc.push(AttributeBase1::ClassObject(ClassBase::ClassDef(
                    self.stdlib.typed_dict_fallback().clone(),
                )))
            }
            Type::Tuple(tuple) => {
                acc.push(AttributeBase1::ClassInstance(self.erase_tuple_type(tuple)))
            }
            Type::LiteralString | Type::Literal(Lit::Str(_)) => {
                acc.push(AttributeBase1::LiteralString)
            }
            Type::Literal(Lit::Enum(lit_enum)) => acc.push(AttributeBase1::EnumLiteral(*lit_enum)),
            Type::Literal(lit) => acc.push(AttributeBase1::ClassInstance(
                lit.general_class_type(self.stdlib).clone(),
            )),
            Type::TypeGuard(_) | Type::TypeIs(_) => {
                acc.push(AttributeBase1::ClassInstance(self.stdlib.bool().clone()))
            }
            Type::Any(style) => acc.push(AttributeBase1::Any(style)),
            Type::TypeAlias(ta) => self.as_attribute_base1(ta.as_value(self.stdlib), acc),
            Type::Type(box Type::Tuple(tuple)) => self
                .as_attribute_base1(Type::type_form(self.erase_tuple_type(tuple).to_type()), acc),
            Type::Type(box Type::ClassType(class)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(class.clone()),
            )),
            Type::QuantifiedValue(q) => acc.push(AttributeBase1::QuantifiedValue(*q)),
            Type::Type(box Type::Quantified(quantified)) => match quantified.restriction() {
                Restriction::Bound(ty) => {
                    let mut use_fallback = false;
                    if let Some(base) = self.as_attribute_base(ty.clone()) {
                        for base1 in base.0 {
                            if let AttributeBase1::ClassInstance(cls) = base1 {
                                acc.push(AttributeBase1::ClassObject(ClassBase::Quantified(
                                    (*quantified).clone(),
                                    cls,
                                )));
                            } else {
                                use_fallback = true;
                            }
                        }
                    }
                    if use_fallback {
                        acc.push(AttributeBase1::ClassObject(ClassBase::Quantified(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        )));
                    }
                }
                Restriction::Constraints(constraints) => {
                    let mut use_fallback = false;
                    for ty in constraints {
                        if let Some(base) = self.as_attribute_base(ty.clone()) {
                            for base1 in base.0 {
                                if let AttributeBase1::ClassInstance(cls) = base1 {
                                    acc.push(AttributeBase1::ClassObject(ClassBase::Quantified(
                                        (*quantified).clone(),
                                        cls,
                                    )));
                                } else {
                                    use_fallback = true;
                                }
                            }
                        }
                    }
                    if use_fallback {
                        acc.push(AttributeBase1::ClassObject(ClassBase::Quantified(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        )));
                    }
                }
                Restriction::Unrestricted => acc.push(AttributeBase1::ClassObject(
                    ClassBase::Quantified((*quantified).clone(), self.stdlib.object().clone()),
                )),
            },
            Type::Type(box Type::Any(style)) => acc.push(AttributeBase1::TypeAny(style)),
            Type::Type(box Type::Never(_)) => acc.push(AttributeBase1::TypeNever),
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
            )) => acc.push(AttributeBase1::TypeAny(AnyStyle::Implicit)),
            Type::Type(box Type::SpecialForm(SpecialForm::Type)) => {
                acc.push(AttributeBase1::ClassObject(ClassBase::ClassDef(
                    self.stdlib.builtins_type().clone(),
                )))
            }
            Type::Module(module) => acc.push(AttributeBase1::Module(module)),
            Type::TypeVar(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.type_var().clone(),
            )),
            Type::ParamSpec(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.param_spec().clone(),
            )),
            Type::TypeVarTuple(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.type_var_tuple().clone(),
            )),
            Type::Args(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.param_spec_args_as_tuple(),
            )),
            Type::Kwargs(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.param_spec_kwargs_as_dict(),
            )),
            Type::ArgsValue(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.param_spec_args().clone(),
            )),
            Type::KwargsValue(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.param_spec_kwargs().clone(),
            )),
            Type::Type(box Type::TypeVar(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.type_var().clone()),
            )),
            Type::Type(box Type::ParamSpec(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.param_spec().clone()),
            )),
            Type::Type(box Type::TypeVarTuple(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.type_var_tuple().clone()),
            )),
            Type::Type(box Type::QuantifiedValue(q)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(q.class_type(self.stdlib).clone()),
            )),
            Type::Type(box Type::Args(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.param_spec_args_as_tuple()),
            )),
            Type::Type(box Type::Kwargs(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.param_spec_kwargs_as_dict()),
            )),
            Type::Type(box Type::ArgsValue(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.param_spec_args().clone()),
            )),
            Type::Type(box Type::KwargsValue(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.param_spec_kwargs().clone()),
            )),
            Type::None => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.none_type().clone(),
            )),
            Type::Type(box Type::None) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.none_type().clone()),
            )),
            Type::Type(
                box (Type::Function(_)
                | Type::Callable(_)
                | Type::Overload(_)
                | Type::Forall(box Forall {
                    tparams: _,
                    body: Forallable::Function(_) | Forallable::Callable(_),
                })),
            ) => acc.push(AttributeBase1::ClassObject(ClassBase::ClassType(
                self.stdlib.function_type().clone(),
            ))),
            Type::Type(box Type::BoundMethod(_)) => acc.push(AttributeBase1::ClassObject(
                ClassBase::ClassType(self.stdlib.method_type().clone()),
            )),
            Type::Never(_) => acc.push(AttributeBase1::Never),
            _ if ty.is_property_getter() => {
                let deleter = ty
                    .property_metadata()
                    .map(|meta| meta.has_deleter)
                    .unwrap_or(false);
                acc.push(AttributeBase1::Property(PropertyAttr {
                    getter: ty.clone(),
                    setter: None,
                    deleter,
                }));
            }
            _ if let Some(metadata) = ty.property_metadata()
                && matches!(metadata.role, PropertyRole::Setter) =>
            {
                acc.push(AttributeBase1::Property(PropertyAttr {
                    getter: metadata.getter.clone(),
                    setter: Some(ty.clone()),
                    deleter: metadata.has_deleter,
                }));
            }
            Type::Callable(_) => acc.push(AttributeBase1::ClassInstance(
                self.stdlib.function_type().clone(),
            )),
            Type::KwCall(call) => self.as_attribute_base1(call.return_ty, acc),
            Type::Function(box Function {
                signature: _,
                metadata,
            })
            | Type::Overload(Overload {
                signatures: _,
                metadata: box metadata,
            }) => acc.push(AttributeBase1::ClassInstance(
                if let FunctionKind::CallbackProtocol(cls) = metadata.kind {
                    *cls
                } else {
                    self.stdlib.function_type().clone()
                },
            )),
            Type::BoundMethod(bound_method) => {
                acc.push(AttributeBase1::BoundMethod(bound_method.func.clone()));
            }
            Type::Ellipsis => {
                if let Some(cls) = self.stdlib.ellipsis_type() {
                    acc.push(AttributeBase1::ClassInstance(cls.clone()))
                }
            }
            Type::Forall(forall) => self.as_attribute_base1(forall.body.as_type(), acc),
            Type::Var(v) => {
                self.force_var_for_attribute_base(v, |ty| self.as_attribute_base1(ty, acc))
            }
            Type::Type(box Type::Var(v)) => self.force_var_for_attribute_base(v, |ty| {
                self.as_attribute_base1(Type::type_form(ty), acc)
            }),
            Type::SuperInstance(box (cls, obj)) => {
                acc.push(AttributeBase1::SuperInstance(cls, obj))
            }
            Type::Union(box Union { members, .. }) => {
                for ty in members {
                    self.as_attribute_base1(ty, acc)
                }
            }
            Type::Type(box Type::Union(box Union { members, .. })) => {
                for ty in members {
                    self.as_attribute_base1(Type::type_form(ty), acc)
                }
            }
            Type::Type(box Type::Intersect(box (_, fallback))) => {
                // TODO(rechen): implement attribute access on `type[A & B]`
                self.as_attribute_base1(Type::type_form(fallback), acc)
            }
            Type::Quantified(quantified) => match quantified.restriction() {
                Restriction::Bound(ty) => {
                    let mut use_fallback = false;
                    if let Some(base) = self.as_attribute_base(ty.clone()) {
                        for base1 in base.0 {
                            if let AttributeBase1::ClassInstance(cls) = base1 {
                                acc.push(AttributeBase1::Quantified((*quantified).clone(), cls));
                            } else {
                                use_fallback = true;
                            }
                        }
                    }
                    if use_fallback {
                        acc.push(AttributeBase1::Quantified(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        ));
                    }
                }
                Restriction::Constraints(constraints) => {
                    let mut use_fallback = false;
                    for ty in constraints {
                        if let Some(base) = self.as_attribute_base(ty.clone()) {
                            for base1 in base.0 {
                                if let AttributeBase1::ClassInstance(cls) = base1 {
                                    acc.push(AttributeBase1::Quantified(
                                        (*quantified).clone(),
                                        cls,
                                    ));
                                } else {
                                    use_fallback = true;
                                }
                            }
                        }
                    }
                    if use_fallback {
                        acc.push(AttributeBase1::Quantified(
                            (*quantified).clone(),
                            self.stdlib.object().clone(),
                        ));
                    }
                }
                Restriction::Unrestricted => acc.push(AttributeBase1::Quantified(
                    (*quantified).clone(),
                    self.stdlib.object().clone(),
                )),
            },
            Type::Intersect(x) => {
                let mut acc_intersect = Vec::new();
                for t in x.0 {
                    self.as_attribute_base1(t, &mut acc_intersect);
                }
                let mut acc_fallback = Vec::new();
                self.as_attribute_base1(x.1, &mut acc_fallback);
                acc.push(AttributeBase1::Intersect(acc_intersect, acc_fallback));
            }
            Type::ElementOfTypeVarTuple(_) => {
                acc.push(AttributeBase1::ClassInstance(self.stdlib.object().clone()))
            }
            // TODO: check to see which ones should have class representations
            Type::SpecialForm(_)
            | Type::Type(_)
            | Type::Unpack(_)
            | Type::Concatenate(_, _)
            | Type::ParamSpecValue(_)
            | Type::Materialization => {}
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
        for (attr, _) in found {
            let found_ty = match self.resolve_get_access(attr_name, attr, range, errors, None) {
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

    // When determining the boolean value of some term used in a boolean context
    // to bool, check that either it does not override __bool__, or the
    // condition_type.__bool__ is callable.
    //
    // This allows users to mark a class as not allowing truthiness checks by
    // explicitly setting `__bool__` to any non-callable type.
    pub fn check_dunder_bool_is_callable(
        &self,
        type_of_term_used_as_bool: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        // TODO(stroxler): Ideally, we would collect up the error messages and produce a single
        // error here. But non-callable `__bool__` failures are likely to be rare in most
        // codebases so this is not urgent unless we get complaints.
        let f = |union_member_ty: &Type| {
            let dunder_bool_ty = self.type_of_magic_dunder_attr(
                union_member_ty,
                &dunder::BOOL,
                range,
                errors,
                None,
                "__bool__",
                false,
            );

            if let Some(dunder_bool_ty) = dunder_bool_ty
                && !dunder_bool_ty.is_never()
                && self.as_call_target(dunder_bool_ty.clone()).is_error()
            {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!(
                        "The `__bool__` attribute of `{}` has type `{}`, which is not callable",
                        self.for_display(union_member_ty.clone()),
                        self.for_display(dunder_bool_ty.clone()),
                    ),
                );
            }
        };
        self.map_over_union(type_of_term_used_as_bool, f)
    }
}

#[derive(Debug, Clone)]
pub enum AttrDefinition {
    FullyResolved(TextRangeWithModule),
    PartiallyResolvedImportedModuleAttribute { module_name: ModuleName },
}

#[derive(Debug)]
pub struct AttrInfo {
    pub name: Name,
    pub ty: Option<Type>,
    pub is_deprecated: bool,
    pub definition: Option<AttrDefinition>,
    pub docstring_range: Option<TextRange>,
    /// is this defined in another module (true) or in this module (false)?
    pub is_reexport: bool,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn completions_mro<T>(
        &self,
        mro: T,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) where
        T: Iterator<Item = &'a Class>,
    {
        let mut seen = SmallSet::new();
        for c in mro {
            match expected_attribute_name {
                None => {
                    for fld in c.fields() {
                        if seen.insert(fld)
                            && let Some(range) = c.field_decl_range(fld)
                        {
                            res.push(AttrInfo {
                                name: fld.clone(),
                                ty: None,
                                is_deprecated: false,
                                definition: Some(AttrDefinition::FullyResolved(
                                    TextRangeWithModule::new(c.module().dupe(), range),
                                )),
                                docstring_range: c.field_docstring_range(fld),
                                is_reexport: false,
                            });
                        }
                    }
                }
                Some(expected_attribute_name) => {
                    if let Some(range) = c.field_decl_range(expected_attribute_name) {
                        res.push(AttrInfo {
                            name: expected_attribute_name.clone(),
                            ty: None,
                            is_deprecated: false,
                            definition: Some(AttrDefinition::FullyResolved(
                                TextRangeWithModule::new(c.module().dupe(), range),
                            )),
                            docstring_range: c.field_docstring_range(expected_attribute_name),
                            is_reexport: false,
                        });
                    }
                }
            }
        }
    }

    fn completions_class(
        &self,
        cls: &Class,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        // NOTE: We do not provide completions from object, to avoid noise like __hash__. Maybe we should?
        let mro = self.get_mro_for_class(cls);
        let ancestors =
            iter::once(cls).chain(mro.ancestors_no_object().iter().map(|x| x.class_object()));
        self.completions_mro(ancestors, expected_attribute_name, res)
    }

    fn completions_super(
        &self,
        cls: &Class,
        start_lookup_cls: &ClassType,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        let mro = self.get_mro_for_class(cls);
        let mro = mro
            .ancestors_no_object()
            .iter()
            .skip_while(|ancestor| *ancestor != start_lookup_cls)
            .map(|x| x.class_object());
        self.completions_mro(mro, expected_attribute_name, res)
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
                    res.extend(
                        exports
                            .exports(self.exports)
                            .iter()
                            .map(|(x, export_location)| AttrInfo {
                                name: x.clone(),
                                ty: None,
                                is_deprecated: matches!(
                                    export_location,
                                    ExportLocation::ThisModule(Export {
                                        deprecation: Some(_),
                                        ..
                                    })
                                ),
                                definition: Some(
                                    AttrDefinition::PartiallyResolvedImportedModuleAttribute {
                                        module_name,
                                    },
                                ),
                                docstring_range: match export_location {
                                    ExportLocation::ThisModule(Export {
                                        docstring_range, ..
                                    }) => *docstring_range,
                                    _ => None,
                                },
                                is_reexport: matches!(
                                    export_location,
                                    ExportLocation::OtherModule(..)
                                ),
                            }),
                    );
                }
                Some(expected_attribute_name) => {
                    if let Some(export_location) =
                        exports.exports(self.exports).get(expected_attribute_name)
                    {
                        res.push(AttrInfo {
                            name: expected_attribute_name.clone(),
                            ty: None,
                            is_deprecated: matches!(
                                export_location,
                                ExportLocation::ThisModule(Export {
                                    deprecation: Some(_),
                                    ..
                                })
                            ),
                            definition: Some(
                                AttrDefinition::PartiallyResolvedImportedModuleAttribute {
                                    module_name,
                                },
                            ),
                            docstring_range: match export_location {
                                ExportLocation::ThisModule(Export {
                                    docstring_range, ..
                                }) => *docstring_range,
                                _ => None,
                            },
                            is_reexport: matches!(export_location, ExportLocation::OtherModule(..)),
                        });
                    }
                }
            }
        }
    }

    fn completions_inner(
        &self,
        base: AttributeBase,
        expected_attribute_name: Option<&Name>,
        include_types: bool,
        res: &mut Vec<AttrInfo>,
    ) {
        for base1 in &base.0 {
            self.completions_inner1(base1, expected_attribute_name, res);
        }
        if include_types {
            for info in res {
                if let Some(definition) = &info.definition
                    && matches!(definition, AttrDefinition::FullyResolved(..))
                {
                    let found_attrs = self
                        .lookup_attr_from_attribute_base(base.clone(), &info.name)
                        .found;
                    let mut is_deprecated = false;
                    let found_types: Vec<_> = found_attrs
                        .into_iter()
                        .filter_map(|(attr, _)| {
                            match &attr {
                                Attribute::ClassAttribute(ClassAttribute::ReadWrite(ty))
                                | Attribute::ClassAttribute(ClassAttribute::ReadOnly(ty, _))
                                | Attribute::Simple(ty)
                                | Attribute::ClassAttribute(ClassAttribute::Property(ty, _, _))
                                    if ty.function_deprecation().is_some() =>
                                {
                                    is_deprecated = true;
                                }
                                _ => {}
                            }
                            self.resolve_get_access(
                                &info.name,
                                attr,
                                // Important we do not use the resolved TextRange, as it might be in a different module.
                                // Whereas the empty TextRange is valid for all modules.
                                TextRange::default(),
                                &self.error_swallower(),
                                None,
                            )
                            .ok()
                        })
                        .collect();
                    if !found_types.is_empty() {
                        info.ty = Some(self.unions(found_types));
                    }
                    info.is_deprecated = is_deprecated;
                }
            }
        }
    }

    fn completions_inner1(
        &self,
        base1: &AttributeBase1,
        expected_attribute_name: Option<&Name>,
        res: &mut Vec<AttrInfo>,
    ) {
        match base1 {
            AttributeBase1::ClassInstance(class)
            | AttributeBase1::SelfType(class)
            | AttributeBase1::EnumLiteral(LitEnum { class, .. })
            | AttributeBase1::Quantified(_, class) => {
                self.completions_class_type(class, expected_attribute_name, res)
            }
            AttributeBase1::LiteralString => {
                self.completions_class_type(self.stdlib.str(), expected_attribute_name, res)
            }
            AttributeBase1::TypedDict(_) => self.completions_class_type(
                self.stdlib.typed_dict_fallback(),
                expected_attribute_name,
                res,
            ),
            AttributeBase1::SuperInstance(start_lookup_cls, obj) => {
                let cls = match obj {
                    SuperObj::Instance(c) | SuperObj::Class(c) => c.class_object(),
                };
                self.completions_super(cls, start_lookup_cls, expected_attribute_name, res)
            }
            AttributeBase1::QuantifiedValue(q) => {
                self.completions_class_type(q.class_type(self.stdlib), expected_attribute_name, res)
            }
            AttributeBase1::ClassObject(class) => {
                self.completions_class(class.class_object(), expected_attribute_name, res)
            }
            AttributeBase1::BoundMethod(bound_func) => {
                self.completions_class_type(
                    self.stdlib.method_type(),
                    expected_attribute_name,
                    res,
                );
                let mut func_bases = Vec::new();
                self.as_attribute_base1(bound_func.clone().as_type(), &mut func_bases);
                for base1 in func_bases {
                    self.completions_inner1(&base1, expected_attribute_name, res);
                }
            }
            AttributeBase1::TypeAny(_) | AttributeBase1::TypeNever => self.completions_class_type(
                self.stdlib.builtins_type(),
                expected_attribute_name,
                res,
            ),
            AttributeBase1::Module(module) => {
                self.completions_module(module, expected_attribute_name, res);
            }
            AttributeBase1::ProtocolSubset(protocol_base) => {
                self.completions_inner1(protocol_base, expected_attribute_name, res)
            }
            AttributeBase1::Any(_) => {}
            AttributeBase1::Never => {}
            AttributeBase1::Property(_) => {
                // TODO(samzhou19815): Support autocomplete for properties
                {}
            }
            AttributeBase1::Intersect(bases, _) => {
                for b in bases {
                    self.completions_inner1(b, expected_attribute_name, res);
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
        if let Some(base) = self.as_attribute_base(base) {
            self.completions_inner(base, expected_attribute_name, include_types, &mut res);
        }
        res
    }
}
