/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;

use crate::callable::Callable;
use crate::callable::IdentityIgnored;
use crate::class::Class;
use crate::class::ClassType;
use crate::equality::TypeEq;
use crate::keywords::DataclassTransformMetadata;
use crate::meta_shape_dsl::ShapeDslFunction;
use crate::meta_shape_dsl::ShapeTransform;
use crate::type_level_dsl::TypeShapeDslDomain;
use crate::type_level_dsl::ValidatedTypeShapeDslFunction;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Function {
    pub signature: Callable,
    pub metadata: FuncMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncMetadata {
    pub kind: FunctionKind,
    pub flags: FuncFlags,
}

impl FuncMetadata {
    pub fn def(module: &Module, cls: Option<&Class>, name: Name) -> Self {
        Self {
            kind: FunctionKind::Def(Arc::new(FuncId {
                module: module.dupe(),
                cls: cls.map(Dupe::dupe),
                name,
                def_index: None,
                outer_funcs: None,
            })),
            flags: FuncFlags::default(),
        }
    }

    pub fn method(cls: &Class, name: Name) -> Self {
        Self::def(cls.module(), Some(cls), name)
    }
}

/// Metadata extracted from a `@deprecated` decorator.
#[derive(
    Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq, PartialOrd, Ord, Hash
)]
pub struct Deprecation {
    pub message: Option<String>,
}

impl Deprecation {
    pub fn new(message: Option<String>) -> Self {
        Self { message }
    }

    /// Format deprecation metadata for error reporting.
    pub fn as_error_detail(&self) -> Option<String> {
        match self.message.as_ref().map(|s| s.trim()) {
            Some(msg) if !msg.is_empty() => Some(msg.to_owned()),
            _ => None,
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub enum PropertyRole {
    Getter,
    Setter,
    SetterDecorator,
    DeleterDecorator,
}

/// Shape of a function body.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub enum BodyKind {
    /// Body is exactly `raise NotImplementedError(...)`. This is the canonical
    /// "abstract-ish" placeholder; concrete subclasses override it.
    RaiseNotImplementedError,
    /// Body is exactly `return NotImplemented`. This is the dunder-protocol
    /// signal to defer to the other operand and is not an override placeholder.
    ReturnNotImplemented,
    /// Body is exactly `...`. This is the canonical placeholder for an elided
    /// function body in a stub-like context.
    Ellipsis,
    /// Body is `pass` or a docstring.
    Trivial,
    #[default]
    Other,
}

impl BodyKind {
    pub fn is_placeholder_or_trivial(&self) -> bool {
        matches!(
            self,
            Self::RaiseNotImplementedError
                | Self::ReturnNotImplemented
                | Self::Ellipsis
                | Self::Trivial
        )
    }
}

/// Ephemeral struct for computing facts about a function definition.
#[derive(Debug)]
pub struct FuncFacts {
    pub body_kind: BodyKind,
    pub is_in_protocol_class: bool,
    pub is_in_type_checking_block: bool,
    pub is_abstract_method: bool,
    pub is_overload: bool,
}

impl FuncFacts {
    pub fn allows_missing_implementation(&self) -> bool {
        self.is_in_protocol_class || self.is_in_type_checking_block || self.is_abstract_method
    }

    /// Is this function defined in an interface (i.e., .pyi) -like context?
    pub fn is_in_interface_like_context(&self) -> bool {
        self.allows_missing_implementation() || self.is_overload
    }

    pub fn is_stub(&self) -> bool {
        // A `...` body is always interpreted as a stub function.
        // Functions with other trivial bodies are interpreted as stubs in some contexts.
        self.body_kind == BodyKind::Ellipsis
            || (self.is_in_interface_like_context() && self.body_kind == BodyKind::Trivial)
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub struct PropertyMetadata {
    pub role: PropertyRole,
    pub getter: Type,
    pub setter: Option<Type>,
    pub has_deleter: bool,
}

impl PropertyMetadata {
    /// Build a PropertyMetadata that stores sanitized (metadata-free) copies of getter/setter.
    pub fn from_components(
        role: PropertyRole,
        getter: &Type,
        setter: Option<&Type>,
        has_deleter: bool,
    ) -> Self {
        Self {
            role,
            getter: getter.without_property_metadata(),
            setter: setter.map(|s| s.without_property_metadata()),
            has_deleter,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncFlags {
    pub is_overload: bool,
    pub is_staticmethod: bool,
    pub is_classmethod: bool,
    /// A function decorated with `@deprecated`
    pub deprecation: Option<Deprecation>,
    /// Metadata for `@property`, `@foo.setter`, and `@foo.deleter`.
    pub property_metadata: Option<PropertyMetadata>,
    /// A function decorated with `functools.cached_property` or equivalent.
    pub is_cached_property: bool,
    pub has_enum_member_decoration: bool,
    pub is_override: bool,
    pub has_final_decoration: bool,
    /// A function decorated with `@abc.abstractmethod`
    pub is_abstract_method: bool,
    /// A function decorated with `@typing.no_type_check` or `@typing_extensions.no_type_check`
    pub has_no_type_check: bool,
    /// Style of the module the function is defined in
    pub module_style: ModuleStyle,
    /// Set when the function was declared with `async def` (NOT when a regular
    /// `def` happens to return a `Coroutine[...]`-typed value). Used to
    /// distinguish async-def placeholders from sync functions explicitly
    /// annotated to return a coroutine, which look identical at the type level
    /// once the async-wrapping into `Coroutine[Any, Any, T]` has happened.
    pub is_async: bool,
    /// Tracks special function body shapes such as placeholder statements
    /// and trivial bodies (see `BodyKind`).
    pub body_kind: BodyKind,
    /// Set when the function's return type has no user-supplied annotation and
    /// was inferred from the body (corresponds to
    /// `ReturnTypeKind::ShouldInferType`). Used to distinguish a return type
    /// the user wrote (e.g. an explicit `-> Never`) from one Pyrefly inferred,
    /// which lets override-consistency logic relax inferred placeholder returns
    /// without overriding what the user explicitly declared.
    pub is_return_inferred: bool,
    /// Whether the function body directly calls `super(...).<this function>(...)`.
    pub calls_super_method: bool,
    /// A function decorated with `typing.dataclass_transform(...)`, turning it into a
    /// `dataclasses.dataclass`-like decorator. Stores the keyword values passed to the
    /// `dataclass_transform` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    pub dataclass_transform_metadata: Option<DataclassTransformMetadata>,
    /// A function decorated with `@uses_shape_dsl`, whose return type should be
    /// refined by evaluating the referenced shape-DSL function at call sites.
    pub shape_transform: Option<Arc<ShapeTransform>>,
    /// A function decorated with `@defines_assert_shape`.
    pub is_assert_shape: bool,
    /// A method directly inside a `Protocol` class.
    pub is_in_protocol_class: bool,
    pub is_in_type_checking_block: bool,
    /// Set when the definition has both `*args` and `**kwargs` and both are typed
    /// `Any` — explicitly, or implicitly because they are unannotated. Per the
    /// typing spec such a signature is equivalent to `...`. Captured at definition
    /// time so that an `Any` introduced later by type-parameter substitution (e.g.
    /// `Proto[Any]` where the params are `*args: T, **kwargs: T`) is not mistaken
    /// for a gradual signature.
    pub has_gradual_variadic_params: bool,
}

impl FuncFlags {
    pub fn facts(&self) -> FuncFacts {
        FuncFacts {
            body_kind: self.body_kind,
            is_in_protocol_class: self.is_in_protocol_class,
            is_in_type_checking_block: self.is_in_type_checking_block,
            is_abstract_method: self.is_abstract_method,
            is_overload: self.is_overload,
        }
    }
}

/// The index of a function definition (`def ..():` statement) within the module,
/// used as a reference to data associated with the function.
#[derive(Debug, Clone, Dupe, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[derive(Display, Visit, VisitMut, TypeEq)]
pub struct FuncDefIndex(pub u32);

#[derive(Debug, Clone)]
pub struct FuncId {
    pub module: Module,
    pub cls: Option<Class>,
    pub name: Name,
    pub def_index: Option<FuncDefIndex>,
    /// Dot-separated path of enclosing function names (e.g. `"f1"` for a function nested inside `f1`).
    /// `None` for top-level and class-method functions.
    pub outer_funcs: Option<Name>,
}

impl PartialEq for FuncId {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq().eq(&other.key_eq())
    }
}

impl Eq for FuncId {}
impl TypeEq for FuncId {}

impl Ord for FuncId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key_ord().cmp(&other.key_ord())
    }
}

impl PartialOrd for FuncId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for FuncId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key_eq().hash(state)
    }
}

impl VisitMut<Type> for FuncId {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for FuncId {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// FuncId contains no Type fields, so visiting through Arc is a no-op.
impl VisitMut<Type> for Arc<FuncId> {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for Arc<FuncId> {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl FuncId {
    /// Identity tuple for equality and hashing. `outer_funcs` is intentionally
    /// excluded because it is display-only metadata (the dotted path of enclosing
    /// function names) and does not affect the logical identity of a function.
    fn key_eq(
        &self,
    ) -> (
        ModuleName,
        ModulePath,
        Option<Class>,
        &Name,
        Option<FuncDefIndex>,
    ) {
        (
            self.module.name(),
            self.module.path().to_key_eq(),
            self.cls.clone(),
            &self.name,
            self.def_index,
        )
    }

    fn key_ord(
        &self,
    ) -> (
        ModuleName,
        ModulePath,
        Option<Class>,
        &Name,
        Option<FuncDefIndex>,
    ) {
        self.key_eq()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum FunctionKind {
    IsInstance,
    IsSubclass,
    /// The builtin `len`. Special-cased so that when the argument's `__len__`
    /// returns a subtype of `int` (e.g. a shaped array's `Int[N]`), `len(x)`
    /// yields that type instead of typeshed's plain `int`.
    Len,
    Dataclass,
    DataclassField,
    DataclassReplace,
    CopyReplace,
    DataclassAsdict,
    /// `attr.fields(C)` / `attrs.fields(C)`.
    AttrsFields,
    /// `attr.fields_dict(C)` / `attrs.fields_dict(C)`.
    AttrsFieldsDict,
    /// `attr.evolve` / `attrs.evolve`: validated like `dataclasses.replace`.
    AttrsEvolve,
    /// `attr.assoc` / `attrs.assoc`: validated against attribute names, including `init=False`.
    AttrsAssoc,
    /// `typing.dataclass_transform`. Note that this is `dataclass_transform` itself, *not* the
    /// decorator created by a `dataclass_transform(...)` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    DataclassTransform,
    ClassMethod,
    Overload,
    Override,
    Cast,
    AssertType,
    AssertShape,
    RevealType,
    Final,
    RuntimeCheckable,
    Def(Arc<FuncId>),
    AbstractMethod,
    /// A function decorated with `typing.no_type_check` or `typing_extensions.no_type_check`.
    NoTypeCheck,
    /// Instance of a protocol with a `__call__` method. The function has the `__call__` signature.
    CallbackProtocol(Box<ClassType>),
    /// The `register` method of a `functools.singledispatch` dispatcher, tagged with the fallback's
    /// first-parameter type so the registered dispatch type can be validated against it.
    SingleDispatchRegister(Box<Type>),
    TotalOrdering,
    DisjointBase,
    /// `numba.jit()`
    NumbaJit,
    /// `numba.njit()`
    NumbaNjit,
    /// A function whose return type is computed by a shape DSL definition.
    /// The `FuncId` provides identity (module, class, name) for display and
    /// lookup; the `ShapeDslFunction` carries the parsed DSL IR.
    ShapeDsl(
        Arc<FuncId>,
        Arc<ShapeDslFunction>,
        IdentityIgnored<Arc<Vec<Arc<ShapeDslFunction>>>>,
    ),
    /// A validated user-defined type-level shape DSL function.
    TypeShapeDsl(
        Arc<FuncId>,
        TypeShapeDslDomain,
        Arc<ValidatedTypeShapeDslFunction>,
    ),
    /// The `shape_extensions.uses_shape_dsl` decorator function itself.
    UsesShapeDsl,
    /// The `shape_extensions.defines_assert_shape` decorator function itself.
    DefinesAssertShape,
}

impl FunctionKind {
    /// Return source-definition identity for variants that preserve an ordinary function def.
    pub fn definition_id(&self) -> Option<&FuncId> {
        match self {
            Self::Def(func_id) | Self::ShapeDsl(func_id, ..) | Self::TypeShapeDsl(func_id, ..) => {
                Some(func_id)
            }
            _ => None,
        }
    }

    pub fn from_name(
        module: Module,
        cls: Option<Class>,
        func: &Name,
        def_index: Option<FuncDefIndex>,
        outer_funcs: Option<Name>,
    ) -> Self {
        match (module.name().as_str(), cls.as_ref(), func.as_str()) {
            ("builtins", None, "isinstance") => Self::IsInstance,
            ("builtins", None, "issubclass") => Self::IsSubclass,
            ("builtins", None, "len") => Self::Len,
            ("builtins", None, "classmethod") => Self::ClassMethod,
            ("dataclasses", None, "dataclass") => Self::Dataclass,
            ("dataclasses", None, "field") => Self::DataclassField,
            ("dataclasses", None, "replace") => Self::DataclassReplace,
            ("copy", None, "replace") => Self::CopyReplace,
            ("dataclasses", None, "asdict") => Self::DataclassAsdict,
            ("attr" | "attrs", None, "fields") => Self::AttrsFields,
            ("attr" | "attrs", None, "fields_dict") => Self::AttrsFieldsDict,
            ("attr" | "attrs", None, "evolve") => Self::AttrsEvolve,
            ("attr" | "attrs", None, "assoc") => Self::AttrsAssoc,
            ("typing" | "typing_extensions", None, "overload") => Self::Overload,
            ("typing" | "typing_extensions", None, "override") => Self::Override,
            ("typing" | "typing_extensions", None, "cast") => Self::Cast,
            ("typing" | "typing_extensions", None, "assert_type") => Self::AssertType,
            ("shape_extensions", None, "assert_shape") => Self::AssertShape,
            ("typing" | "typing_extensions", None, "reveal_type") => Self::RevealType,
            ("typing" | "typing_extensions", None, "final") => Self::Final,
            ("typing" | "typing_extensions", None, "runtime_checkable") => Self::RuntimeCheckable,
            ("typing" | "typing_extensions", None, "dataclass_transform") => {
                Self::DataclassTransform
            }
            ("abc", None, "abstractmethod") => Self::AbstractMethod,
            ("typing" | "typing_extensions", None, "no_type_check") => Self::NoTypeCheck,
            ("functools", None, "total_ordering") => Self::TotalOrdering,
            ("typing" | "typing_extensions", None, "disjoint_base") => Self::DisjointBase,
            ("numba.core.decorators", None, "jit") => Self::NumbaJit,
            ("numba.core.decorators", None, "njit") => Self::NumbaNjit,
            ("shape_extensions", None, "uses_shape_dsl") => Self::UsesShapeDsl,
            ("shape_extensions", None, "defines_assert_shape") => Self::DefinesAssertShape,
            _ => Self::Def(Arc::new(FuncId {
                module,
                cls,
                name: func.clone(),
                def_index,
                outer_funcs,
            })),
        }
    }

    pub fn module_name(&self) -> ModuleName {
        match self {
            Self::IsInstance => ModuleName::builtins(),
            Self::IsSubclass => ModuleName::builtins(),
            Self::Len => ModuleName::builtins(),
            Self::ClassMethod => ModuleName::builtins(),
            Self::Dataclass => ModuleName::dataclasses(),
            Self::DataclassField => ModuleName::dataclasses(),
            Self::DataclassReplace => ModuleName::dataclasses(),
            Self::CopyReplace => ModuleName::from_str("copy"),
            Self::DataclassAsdict => ModuleName::dataclasses(),
            Self::AttrsFields => ModuleName::attr(),
            Self::AttrsFieldsDict => ModuleName::attr(),
            Self::AttrsEvolve => ModuleName::attr(),
            Self::AttrsAssoc => ModuleName::attr(),
            Self::DataclassTransform => ModuleName::typing(),
            Self::Final => ModuleName::typing(),
            Self::Overload => ModuleName::typing(),
            Self::Override => ModuleName::typing(),
            Self::Cast => ModuleName::typing(),
            Self::AssertType => ModuleName::typing(),
            Self::AssertShape => ModuleName::from_str("shape_extensions"),
            Self::RevealType => ModuleName::typing(),
            Self::RuntimeCheckable => ModuleName::typing(),
            Self::CallbackProtocol(cls) => cls.qname().module_name(),
            Self::SingleDispatchRegister(_) => ModuleName::functools(),
            Self::AbstractMethod => ModuleName::abc(),
            Self::NoTypeCheck => ModuleName::typing(),
            Self::TotalOrdering => ModuleName::functools(),
            Self::DisjointBase => ModuleName::typing(),
            Self::NumbaJit => ModuleName::from_str("numba"),
            Self::NumbaNjit => ModuleName::from_str("numba"),
            Self::Def(func_id) => func_id.module.name().dupe(),
            Self::ShapeDsl(id, _, _) | Self::TypeShapeDsl(id, _, _) => id.module.name().dupe(),
            Self::UsesShapeDsl => ModuleName::from_str("shape_extensions"),
            Self::DefinesAssertShape => ModuleName::from_str("shape_extensions"),
        }
    }

    pub fn function_name(&self) -> Cow<'_, Name> {
        match self {
            Self::IsInstance => Cow::Owned(Name::new_static("isinstance")),
            Self::IsSubclass => Cow::Owned(Name::new_static("issubclass")),
            Self::Len => Cow::Owned(Name::new_static("len")),
            Self::ClassMethod => Cow::Owned(Name::new_static("classmethod")),
            Self::Dataclass => Cow::Owned(Name::new_static("dataclass")),
            Self::DataclassField => Cow::Owned(Name::new_static("field")),
            Self::DataclassReplace => Cow::Owned(Name::new_static("replace")),
            Self::CopyReplace => Cow::Owned(Name::new_static("replace")),
            Self::DataclassAsdict => Cow::Owned(Name::new_static("asdict")),
            Self::AttrsFields => Cow::Owned(Name::new_static("fields")),
            Self::AttrsFieldsDict => Cow::Owned(Name::new_static("fields_dict")),
            Self::AttrsEvolve => Cow::Owned(Name::new_static("evolve")),
            Self::AttrsAssoc => Cow::Owned(Name::new_static("assoc")),
            Self::DataclassTransform => Cow::Owned(Name::new_static("dataclass_transform")),
            Self::Final => Cow::Owned(Name::new_static("final")),
            Self::Overload => Cow::Owned(Name::new_static("overload")),
            Self::Override => Cow::Owned(Name::new_static("override")),
            Self::Cast => Cow::Owned(Name::new_static("cast")),
            Self::AssertType => Cow::Owned(Name::new_static("assert_type")),
            Self::AssertShape => Cow::Owned(Name::new_static("assert_shape")),
            Self::RevealType => Cow::Owned(Name::new_static("reveal_type")),
            Self::RuntimeCheckable => Cow::Owned(Name::new_static("runtime_checkable")),
            Self::CallbackProtocol(_) => Cow::Owned(dunder::CALL),
            Self::SingleDispatchRegister(_) => Cow::Owned(Name::new_static("register")),
            Self::AbstractMethod => Cow::Owned(Name::new_static("abstractmethod")),
            Self::NoTypeCheck => Cow::Owned(Name::new_static("no_type_check")),
            Self::TotalOrdering => Cow::Owned(Name::new_static("total_ordering")),
            Self::DisjointBase => Cow::Owned(Name::new_static("disjoint_base")),
            Self::NumbaJit => Cow::Owned(Name::new_static("jit")),
            Self::NumbaNjit => Cow::Owned(Name::new_static("njit")),
            Self::Def(func_id) => Cow::Borrowed(&func_id.name),
            Self::ShapeDsl(id, _, _) | Self::TypeShapeDsl(id, _, _) => Cow::Borrowed(&id.name),
            Self::UsesShapeDsl => Cow::Owned(Name::new_static("uses_shape_dsl")),
            Self::DefinesAssertShape => Cow::Owned(Name::new_static("defines_assert_shape")),
        }
    }

    pub fn class(&self) -> Option<Class> {
        match self {
            Self::IsInstance => None,
            Self::IsSubclass => None,
            Self::Len => None,
            Self::ClassMethod => None,
            Self::Dataclass => None,
            Self::DataclassField => None,
            Self::DataclassReplace => None,
            Self::CopyReplace => None,
            Self::DataclassAsdict => None,
            Self::AttrsFields => None,
            Self::AttrsFieldsDict => None,
            Self::AttrsEvolve => None,
            Self::AttrsAssoc => None,
            Self::DataclassTransform => None,
            Self::Final => None,
            Self::Overload => None,
            Self::Override => None,
            Self::Cast => None,
            Self::AssertType => None,
            Self::AssertShape => None,
            Self::RevealType => None,
            Self::RuntimeCheckable => None,
            Self::NumbaJit => None,
            Self::NumbaNjit => None,
            Self::CallbackProtocol(cls) => Some(cls.class_object().dupe()),
            Self::SingleDispatchRegister(_) => None,
            Self::AbstractMethod => None,
            Self::NoTypeCheck => None,
            Self::TotalOrdering => None,
            Self::DisjointBase => None,
            Self::Def(func_id) => func_id.cls.clone(),
            Self::ShapeDsl(id, _, _) | Self::TypeShapeDsl(id, _, _) => id.cls.clone(),
            Self::UsesShapeDsl => None,
            Self::DefinesAssertShape => None,
        }
    }

    pub fn outer_funcs(&self) -> Option<&Name> {
        match self {
            Self::Def(func_id)
            | Self::ShapeDsl(func_id, _, _)
            | Self::TypeShapeDsl(func_id, _, _) => func_id.outer_funcs.as_ref(),
            _ => None,
        }
    }

    pub fn format(&self, current_module: ModuleName) -> String {
        let func_module = self.module_name();
        let module_prefix =
            if func_module == current_module || func_module == ModuleName::builtins() {
                "".to_owned()
            } else {
                format!("{}.", func_module)
            };
        let class_prefix = match &self.class() {
            Some(cls) => {
                format!("{}.", cls.name())
            }
            None => "".to_owned(),
        };
        format!(
            "{module_prefix}{class_prefix}{}",
            self.function_name().as_ref()
        )
    }

    /// Does this decorator require special-casing to be signature-preserving?
    pub fn is_signature_preserving_decorator(&self) -> bool {
        match self {
            Self::NumbaJit | Self::NumbaNjit | Self::SingleDispatchRegister(_) => true,
            _ => false,
        }
    }
}
