/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::char;
use std::fmt;
use std::fmt::Display;

use compact_str::CompactString;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::assert_words;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprBytesLiteral;
use ruff_python_ast::ExprFString;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::FStringPart;
use ruff_python_ast::Int;
use ruff_python_ast::InterpolatedStringElement;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::lit_int::LitInt;
use crate::stdlib::Stdlib;
use crate::types::Type;

assert_words!(Lit, 3);

/// A literal value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Lit {
    Str(CompactString),
    Int(LitInt),
    Bool(bool),
    Bytes(Box<[u8]>),
    Enum(Box<LitEnum>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct LitEnum {
    pub class: ClassType,
    pub member: Name,
    /// Raw type assigned to name in class def.
    /// We store the raw type so we can return it when the value or _value_ attribute is accessed.
    pub ty: Type,
}

impl Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lit::Str(x) => write!(f, "'{x}'"),
            Lit::Int(x) => write!(f, "{x}"),
            Lit::Bool(x) => {
                let s = if *x { "True" } else { "False" };
                write!(f, "{s}")
            }
            Lit::Bytes(bytes) => {
                write!(f, "b'")?;
                for byte in bytes {
                    match char::from_u32(*byte as u32) {
                        Some(ch) => write!(f, "{ch}")?,
                        None => write!(f, "\\x{byte:02x}")?,
                    }
                }
                write!(f, "'")
            }
            Lit::Enum(lit_enum) => {
                write!(f, "{}.{}", lit_enum.class.name(), lit_enum.member)
            }
        }
    }
}

impl Lit {
    /// Returns the negated type, or None if literal can't be negated.
    pub fn negate(&self) -> Option<Type> {
        match self {
            Lit::Int(x) => Some(Lit::Int(x.negate()).to_type()),
            _ => None,
        }
    }

    /// Returns `+self` if the `+` operation is allowed, None otherwise.
    pub fn positive(&self) -> Option<Type> {
        match self {
            Lit::Int(_) => Some(self.clone().to_type()),
            Lit::Bool(true) => Some(Lit::Int(LitInt::new(1)).to_type()),
            Lit::Bool(false) => Some(Lit::Int(LitInt::new(0)).to_type()),
            _ => None,
        }
    }

    /// Returns the inverted type, or None if literal can't be inverted.
    pub fn invert(&self) -> Option<Type> {
        match self {
            Lit::Int(x) => {
                let x = x.invert();
                Some(Lit::Int(x).to_type())
            }
            _ => None,
        }
    }

    pub fn from_string_literal(x: &ExprStringLiteral) -> Self {
        Lit::Str(x.value.to_str().into())
    }

    pub fn from_bytes_literal(x: &ExprBytesLiteral) -> Self {
        Lit::Bytes(x.value.bytes().collect())
    }

    pub fn from_fstring(x: &ExprFString) -> Option<Self> {
        let mut collected_literals = Vec::new();
        for fstring_part in x.value.as_slice() {
            match fstring_part {
                FStringPart::Literal(x) => collected_literals.push(x.value.clone()),
                FStringPart::FString(x) => {
                    for fstring_part in x.elements.iter() {
                        match fstring_part {
                            InterpolatedStringElement::Literal(x) => {
                                collected_literals.push(x.value.clone())
                            }
                            _ => return None,
                        }
                    }
                }
            }
        }
        Some(Lit::Str(collected_literals.join("").into()))
    }

    pub fn from_int(x: &Int) -> Self {
        Lit::Int(LitInt::from_ast(x))
    }

    pub fn from_boolean_literal(x: &ExprBooleanLiteral) -> Self {
        Lit::Bool(x.value)
    }

    /// Convert a literal to a `Type::Literal`.
    pub fn to_type(self) -> Type {
        Type::Literal(self)
    }

    /// Convert a literal to a `ClassType` that is the general class_type of the literal.
    /// For example, `1` is converted to `int`, and `"foo"` is converted to `str`.
    pub fn general_class_type<'a>(&'a self, stdlib: &'a Stdlib) -> &'a ClassType {
        match self {
            Lit::Str(_) => stdlib.str(),
            Lit::Int(_) => stdlib.int(),
            Lit::Bool(_) => stdlib.bool(),
            Lit::Bytes(_) => stdlib.bytes(),
            Lit::Enum(lit_enum) => &lit_enum.class,
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Lit::Str(_))
    }

    pub fn as_index_i64(&self) -> Option<i64> {
        match self {
            Lit::Int(x) => x.as_i64(),
            Lit::Bool(true) => Some(1),
            Lit::Bool(false) => Some(0),
            _ => None,
        }
    }
}
