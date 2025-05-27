/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use ruff_python_ast::name::Name;

use crate::module::module_name::ModuleName;

/// These are names that are exported from the stdlib, but which take on
/// a more keyword-like quality. E.g. `x: TypeAlias = ...` meaningfully
/// changes the sense of the binding.
#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq)]
pub enum SpecialExport {
    TypeAlias,
    TypeVar,
    ParamSpec,
    TypeVarTuple,
    Annotated,
    Literal,
    Enum,
    StrEnum,
    IntEnum,
    TypedDict,
    CollectionsNamedTuple,
    TypingNamedTuple,
    AssertType,
    NewType,
    Union,
    Optional,
    Cast,
    Super,
    Exit,
    Quit,
    OsExit,
    Len,
    NoTypeCheck,
}

impl SpecialExport {
    pub fn new(name: &Name) -> Option<Self> {
        match name.as_str() {
            "TypeAlias" => Some(Self::TypeAlias),
            "TypeVar" => Some(Self::TypeVar),
            "ParamSpec" => Some(Self::ParamSpec),
            "TypeVarTuple" => Some(Self::TypeVarTuple),
            "Annotated" => Some(Self::Annotated),
            "Literal" => Some(Self::Literal),
            "Enum" => Some(Self::Enum),
            "StrEnum" => Some(Self::StrEnum),
            "IntEnum" => Some(Self::IntEnum),
            "TypedDict" => Some(Self::TypedDict),
            "namedtuple" => Some(Self::CollectionsNamedTuple),
            "NamedTuple" => Some(Self::TypingNamedTuple),
            "assert_type" => Some(Self::AssertType),
            "NewType" => Some(Self::NewType),
            "Union" => Some(Self::Union),
            "Optional" => Some(Self::Optional),
            "cast" => Some(Self::Cast),
            "super" => Some(Self::Super),
            "exit" => Some(Self::Exit),
            "quit" => Some(Self::Quit),
            "_exit" => Some(Self::OsExit),
            "len" => Some(Self::Len),
            "no_type_check" => Some(Self::NoTypeCheck),
            _ => None,
        }
    }

    pub fn defined_in(self, m: ModuleName) -> bool {
        match self {
            Self::TypeAlias
            | Self::TypeVar
            | Self::ParamSpec
            | Self::TypeVarTuple
            | Self::Annotated
            | Self::Literal
            | Self::TypedDict
            | Self::TypingNamedTuple
            | Self::NewType
            | Self::Union
            | Self::Optional
            | Self::AssertType
            | Self::NoTypeCheck
            | Self::Cast => {
                matches!(m.as_str(), "typing" | "typing_extensions")
            }
            Self::CollectionsNamedTuple => matches!(m.as_str(), "collections"),
            Self::Enum | Self::StrEnum | Self::IntEnum => matches!(m.as_str(), "enum"),
            Self::Super | Self::Len => matches!(m.as_str(), "builtins"),
            Self::Exit => matches!(m.as_str(), "sys" | "builtins"),
            Self::Quit => matches!(m.as_str(), "builtins"),
            Self::OsExit => matches!(m.as_str(), "os"),
        }
    }
}
