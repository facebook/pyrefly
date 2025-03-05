/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;

use crate::types::types::Type;

/// General context for an error. For many errors, the root cause is some steps removed from what
/// the user sees. For example:
///   class C:
///     def __lt__(self, other: C) -> bool:
///   C() < 0  # ERROR: expected C, got 0
/// The root cause is `C.__lt__` being called with the wrong type, but the user sees a `<`
/// comparison. ErrorContext stores this context that the user sees, to make it easier to connect
/// it back to the root cause.
pub enum ErrorContext {
    BadContextManager(Type),
}

/// The context in which a got <: want type check occurs. This differs from ErrorContext in that
/// TypeCheckContext applies specifically to type mismatches. For example:
///   class C:
///     def __lt__(self, other: C) -> bool:
///   C() < 0  # ERROR: expected C, got 0
/// The TypeCheckContext contains a TypeCheckKind, recording that the mismatch is in the `other` parameter of `C.__lt__`,
/// and an ErrorContext, recording that the type mismatch occurs in the context of a `<` comparison.
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

    /// Temporary helper to label errors as Unknown before we properly classify them.
    pub fn unknown() -> Self {
        Self::of_kind(TypeCheckKind::Unknown)
    }

    /// Temporary helper to label errors as Test to aid classification.
    #[allow(dead_code)]
    pub fn test() -> Self {
        Self::of_kind(TypeCheckKind::Test)
    }
}

pub enum TypeCheckKind {
    /// Check on a magic method that is expected to return a particular type; e.g., a context
    /// manager's `__exit__` method must return `bool | None`.
    MagicMethodReturn(Type, Name),
    /// Implicit return via a path with no explicit return statement. The bool indicates whether
    /// the function has *any* explicit return.
    ImplicitFunctionReturn(bool),
    /// Explicit return statement in a function body.
    ExplicitFunctionReturn,
    /// Return in a type guard function.
    TypeGuardReturn,
    // TODO: categorize all type checks and remove Unknown and Test designations
    Unknown,
    #[allow(dead_code)]
    Test,
}
