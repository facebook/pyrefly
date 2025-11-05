/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FunctionKind;
use ruff_python_ast::name::Name;

use crate::binding::binding::AnnotationTarget;
use crate::config::error_kind::ErrorKind;
use crate::types::types::Type;

/// General context for an error. For many errors, the root cause is some steps removed from what
/// the user sees. For example:
///   class C:
///     def __lt__(self, other: C) -> bool:
///   C() < 0  # ERROR: expected C, got 0
/// The root cause is `C.__lt__` being called with the wrong type, but the user sees a `<`
/// comparison. ErrorContext stores this context that the user sees, to make it easier to connect
/// it back to the root cause.
#[derive(Clone, Debug)]
pub enum ErrorContext {
    /// with x: ...
    BadContextManager(Type),
    /// Unary operation like `+x`
    UnaryOp(String, Type),
    /// Binary operation like `x + y`
    BinaryOp(String, Type, Type),
    /// In-place binary operation like `x += y`
    InplaceBinaryOp(String, Type, Type),
    /// for x in y: ...
    Iteration(Type),
    /// async for x in y: ...
    AsyncIteration(Type),
    /// await x
    Await(Type),
    /// x[y]
    Index(Type),
    /// x[y] = ...
    SetItem(Type),
    /// del x[y]
    DelItem(Type),
    /// match x: case Foo(y): ...
    MatchPositional(Type),
    ImportNotFound(ModuleName),
    ImportNotTyped(ModuleName),
}

impl ErrorContext {
    pub fn as_error_kind(&self) -> ErrorKind {
        match self {
            Self::BadContextManager(..) => ErrorKind::BadContextManager,
            Self::UnaryOp(..) => ErrorKind::UnsupportedOperation,
            Self::BinaryOp(..) => ErrorKind::UnsupportedOperation,
            Self::InplaceBinaryOp(..) => ErrorKind::UnsupportedOperation,
            Self::Iteration(..) => ErrorKind::NotIterable,
            Self::AsyncIteration(..) => ErrorKind::NotIterable,
            Self::Await(..) => ErrorKind::NotAsync,
            Self::Index(..) => ErrorKind::BadIndex,
            Self::SetItem(..) => ErrorKind::UnsupportedOperation,
            Self::DelItem(..) => ErrorKind::UnsupportedOperation,
            Self::MatchPositional(..) => ErrorKind::BadMatch,
            Self::ImportNotFound(..) => ErrorKind::MissingImport,
            Self::ImportNotTyped(..) => ErrorKind::UntypedImport,
        }
    }
}

/// Info about an error. All errors have a kind; some also have a context (see ErrorContext).
/// Use ErrorInfo::Context for errors with both a kind and a context (the kind will be looked up
/// from the context); use ErrorInfo::Kind for errors with a kind but no context.
pub enum ErrorInfo<'a> {
    Context(&'a dyn Fn() -> ErrorContext),
    Kind(ErrorKind),
}

impl<'a> ErrorInfo<'a> {
    /// Build ErrorInfo from a kind and context. Note that the kind is used only when the context is None.
    pub fn new(error_kind: ErrorKind, context: Option<&'a dyn Fn() -> ErrorContext>) -> Self {
        if let Some(ctx) = context {
            Self::Context(ctx)
        } else {
            Self::Kind(error_kind)
        }
    }
}

/// The context in which a got <: want type check occurs. This differs from ErrorContext in that
/// TypeCheckContext applies specifically to type mismatches. For example:
///   class C:
///     def __lt__(self, other: C) -> bool:
///   C() < 0  # ERROR: expected C, got 0
/// The TypeCheckContext contains a TypeCheckKind::CallArgument, recording that
/// the mismatch is in the `other` parameter of `C.__lt__`, and an
/// ErrorContext::BinaryOp, recording that the type mismatch occurs in the context of a `<` comparison.
#[derive(Debug)]
pub struct TypeCheckContext {
    pub kind: TypeCheckKind,
    pub context: Option<ErrorContext>,
}

impl TypeCheckContext {
    pub fn of_kind(kind: TypeCheckKind) -> Self {
        Self {
            kind,
            context: None,
        }
    }
}

#[derive(Debug)]
pub enum TypeCheckKind {
    /// Check on a magic method that is expected to return a particular type; e.g., a context
    /// manager's `__exit__` method must return `bool | None`.
    MagicMethodReturn(Type, Name),
    /// Check that the return of an augmented assignment method call is still assignable to the original value
    AugmentedAssignment,
    /// Implicit return via a path with no explicit return statement. The bool indicates whether
    /// the function has *any* explicit return.
    ImplicitFunctionReturn(bool),
    /// Explicit return statement in a function body.
    ExplicitFunctionReturn,
    /// Return in a type guard function.
    TypeGuardReturn,
    /// Function call argument against parameter type.
    CallArgument(Option<Name>, Option<FunctionKind>),
    /// Function call argument against *arg parameter type. The bool indicates whether the argument is unpacked.
    CallVarArgs(bool, Option<Name>, Option<FunctionKind>),
    /// Keyword argument against parameter or **kwargs type, as (argument name, parameter name, function name).
    CallKwArgs(Option<Name>, Option<Name>, Option<FunctionKind>),
    /// Unpacked keyword argument against named parameter.
    CallUnpackKwArg(Name, Option<FunctionKind>),
    /// Check of a parameter's default value against its type annotation.
    FunctionParameterDefault(Name),
    /// Check against type of a TypedDict key. The name may be None if the type comes from
    /// `extra_items` or some other non-literal-key source.
    TypedDictKey(Option<Name>),
    /// Check an unpacked dict against a TypedDict, e.g., `x: MyTypedDict = {**unpacked_dict}`.
    TypedDictUnpacking,
    /// Check unpacking of an open TypedDict into a TypedDict. Used to report instances of
    /// TypedDictUnpacking that are specifically caused by the open TypedDict potentially
    /// containing extra keys via inheritance.
    TypedDictOpenUnpacking,
    /// Check of an attribute assignment against its type.
    Attribute(Name),
    /// A check against a user-declared type annotation on a variable name.
    AnnotatedName(Name),
    /// Check the type of an iteration variable (the `x` in `for x in seq`) against the iterable.
    // When checking iteration variables (the `x` in `for x in seq`), we transform the type annotation of the variable if it exists.
    // We need to carry around the actual un-transformed type of `x` to avoid a confusing error message.
    IterationVariableMismatch(Name, Type),
    /// var: SomeType = some_value check. This is separate from AnnotatedName because we can
    /// emit a more compact error message for this case.
    AnnAssign,
    /// Check one portion of an unpacked assignment (e.g. `x, y = foo()`) against the expected type.
    UnpackedAssign,
    /// We break cycles using recursive `Var`s, which the solver might pin. When we record a final
    /// answer, if the solver pinned the Var to an incompatible type, we record a type error with this
    /// kind. This is hard to understand and should be avoided when possible.
    CycleBreaking,
    /// Class used in an `except C` clause.
    ExceptionClass,
    /// Yielding a value that conflicts with the return annotation.
    YieldValue,
    /// Yielding from an iterator that conflicts with the return annotation.
    YieldFrom,
    /// Bare yield when the return annotation expects an actual value.
    UnexpectedBareYield,
    /// Check on the type of the dataclass `__post_init__` method.
    PostInit,
    /// Consistency check for overload return types.
    OverloadReturn,
    /// Consistency check for overload input signature, as (overload_signature, implementation_signature)
    OverloadInput(Callable, Callable),
    /// Check that the type a TypeVar is specialized with is compatible with its type restriction.
    TypeVarSpecialization(Name),
    /// An `x in y` check
    Container,
}

impl TypeCheckKind {
    pub fn from_annotation_target(target: &AnnotationTarget) -> Self {
        match target {
            AnnotationTarget::Param(name)
            | AnnotationTarget::ArgsParam(name)
            | AnnotationTarget::KwargsParam(name) => Self::CallArgument(Some(name.clone()), None),
            AnnotationTarget::Return(_func) => Self::ExplicitFunctionReturn,
            AnnotationTarget::Assign(name, _is_initialized) => Self::AnnotatedName(name.clone()),
            AnnotationTarget::ClassMember(member) => Self::Attribute(member.clone()),
        }
    }

    pub fn as_error_kind(&self) -> ErrorKind {
        match self {
            Self::MagicMethodReturn(..) => ErrorKind::BadReturn,
            Self::AugmentedAssignment => ErrorKind::BadAssignment,
            Self::ImplicitFunctionReturn(..) => ErrorKind::BadReturn,
            Self::ExplicitFunctionReturn => ErrorKind::BadReturn,
            Self::TypeGuardReturn => ErrorKind::BadReturn,
            Self::CallArgument(..) => ErrorKind::BadArgumentType,
            Self::CallVarArgs(..) => ErrorKind::BadArgumentType,
            Self::CallKwArgs(..) => ErrorKind::BadArgumentType,
            Self::CallUnpackKwArg(..) => ErrorKind::BadArgumentType,
            Self::FunctionParameterDefault(..) => ErrorKind::BadFunctionDefinition,
            Self::TypedDictKey(..) => ErrorKind::BadTypedDictKey,
            Self::TypedDictUnpacking => ErrorKind::BadUnpacking,
            Self::TypedDictOpenUnpacking => ErrorKind::OpenUnpacking,
            Self::Attribute(..) => ErrorKind::BadAssignment,
            Self::AnnotatedName(..) => ErrorKind::BadAssignment,
            Self::IterationVariableMismatch(..) => ErrorKind::BadAssignment,
            Self::AnnAssign => ErrorKind::BadAssignment,
            Self::UnpackedAssign => ErrorKind::BadAssignment,
            Self::CycleBreaking => ErrorKind::BadAssignment,
            Self::ExceptionClass => ErrorKind::InvalidInheritance,
            Self::YieldValue => ErrorKind::InvalidYield,
            Self::YieldFrom => ErrorKind::InvalidYield,
            Self::UnexpectedBareYield => ErrorKind::InvalidYield,
            Self::PostInit => ErrorKind::BadFunctionDefinition,
            Self::OverloadReturn => ErrorKind::InconsistentOverload,
            Self::OverloadInput(..) => ErrorKind::InconsistentOverload,
            Self::TypeVarSpecialization(..) => ErrorKind::BadSpecialization,
            Self::Container => ErrorKind::UnsupportedOperation,
        }
    }
}
