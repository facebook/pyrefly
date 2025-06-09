/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
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
    Overload,
}

#[derive(Debug)]
pub enum SpecialEntry<'a> {
    ImportModule(ModuleName),
    ImportName(ModuleName, &'a Name),
    Local,
}

pub trait SpecialEnv {
    fn current_module(&self) -> ModuleName;
    fn lookup_special(&self, name: &Name) -> Option<SpecialEntry>;
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
            "overload" => Some(Self::Overload),
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
            | Self::Overload
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

    pub fn as_special_export(env: &impl SpecialEnv, e: &Expr) -> Option<SpecialExport> {
        // Only works for things with `Foo`, or `source.Foo`, or `F` where `from module import Foo as F`.
        // Does not work for things with nested modules - but no SpecialExport's have that.
        match e {
            Expr::Name(name) => {
                let name = &name.id;
                match env.lookup_special(name)? {
                    SpecialEntry::ImportName(m, name2) => {
                        let special = SpecialExport::new(name2)?;
                        if special.defined_in(m) {
                            Some(special)
                        } else {
                            None
                        }
                    }
                    SpecialEntry::Local => {
                        let special = SpecialExport::new(name)?;
                        if special.defined_in(env.current_module()) {
                            Some(special)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            Expr::Attribute(ExprAttribute {
                value: box Expr::Name(module),
                attr: name,
                ..
            }) => {
                let special = SpecialExport::new(&name.id)?;
                match env.lookup_special(&module.id)? {
                    SpecialEntry::ImportModule(m) if special.defined_in(m) => Some(special),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
