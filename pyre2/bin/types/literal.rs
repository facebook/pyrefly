/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::char;
use std::fmt;
use std::fmt::Display;

use ruff_python_ast::name::Name;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprBytesLiteral;
use ruff_python_ast::ExprFString;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::FStringElement;
use ruff_python_ast::FStringPart;
use ruff_python_ast::Identifier;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;
use static_assertions::assert_eq_size;

use crate::ast::Ast;
use crate::error::collector::ErrorCollector;
use crate::module::module_info::ModuleInfo;
use crate::types::class::ClassType;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

assert_eq_size!(Lit, [usize; 3]);

/// A literal value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lit {
    String(Box<str>),
    Int(i64),
    Bool(bool),
    Bytes(Box<[u8]>),
    Enum(Box<(ClassType, Name)>),
}

impl Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lit::String(x) => write!(f, "'{x}'"),
            Lit::Int(x) => write!(f, "{x}"),
            Lit::Bool(x) => {
                let s = if *x { "True" } else { "False" };
                write!(f, "{s}")
            }
            Lit::Bytes(xs) => {
                write!(f, "b'")?;
                for x in xs {
                    match char::from_u32(*x as u32) {
                        Some(x) => write!(f, "{x}")?,
                        None => write!(f, "\\x{:02x}", x)?,
                    }
                }
                write!(f, "'")
            }
            Lit::Enum(box (enumeration, member)) => {
                let name = &enumeration.name();
                write!(f, "{name}.{member}")
            }
        }
    }
}

impl Lit {
    pub fn from_expr(
        x: &Expr,
        module_info: &ModuleInfo,
        get_enum_from_name: &dyn Fn(Identifier) -> Option<Enum>,
        errors: &ErrorCollector,
    ) -> Type {
        match x {
            Expr::UnaryOp(x) => match x.op {
                UnaryOp::UAdd => {
                    Self::from_expr(&x.operand, module_info, get_enum_from_name, errors)
                }
                UnaryOp::USub => {
                    match Self::from_expr(&x.operand, module_info, get_enum_from_name, errors) {
                        Type::Literal(l) => l.negate(module_info, x.range, errors),
                        x => x,
                    }
                }
                _ => {
                    errors.todo(module_info, "Lit::from_expr", x);
                    Type::any_error()
                }
            },
            Expr::StringLiteral(x) => Self::from_string_literal(x).to_type(),
            Expr::BytesLiteral(x) => Self::from_bytes_literal(x).to_type(),
            Expr::NumberLiteral(x) => Self::from_number_literal(x, module_info, errors),
            Expr::BooleanLiteral(x) => Self::from_boolean_literal(x).to_type(),
            Expr::Attribute(ExprAttribute {
                range: _,
                value: box Expr::Name(maybe_enum_name),
                attr: member_name,
                ctx: _,
            }) => match get_enum_from_name(Ast::expr_name_identifier(maybe_enum_name.clone())) {
                Some(e) if let Some(lit) = e.get_member(&member_name.id) => lit.to_type(),
                _ => {
                    errors.todo(module_info, "Lit::from_expr", x);
                    Type::any_error()
                }
            },
            _ => {
                errors.todo(module_info, "Lit::from_expr", x);
                Type::any_error()
            }
        }
    }

    pub fn negate(
        &self,
        module_info: &ModuleInfo,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match self {
            Lit::Int(x) if let Some(x) = x.checked_neg() => Lit::Int(x).to_type(),
            _ => {
                errors.add(module_info, range, format!("Cannot negate type {self}"));
                Type::any_error()
            }
        }
    }

    pub fn invert(
        &self,
        module_info: &ModuleInfo,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match self {
            Lit::Int(x) => {
                let x = !x;
                Lit::Int(x).to_type()
            }
            _ => {
                errors.add(module_info, range, format!("Cannot invert type {self}"));
                Type::any_error()
            }
        }
    }

    pub fn from_string_literal(x: &ExprStringLiteral) -> Self {
        Lit::String(x.value.to_str().into())
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
                            FStringElement::Literal(x) => collected_literals.push(x.value.clone()),
                            _ => return None,
                        }
                    }
                }
            }
        }
        Some(Lit::String(collected_literals.join("").into_boxed_str()))
    }

    pub fn from_number_literal(
        x: &ExprNumberLiteral,
        module_info: &ModuleInfo,
        errors: &ErrorCollector,
    ) -> Type {
        match &x.value {
            Number::Int(x) if let Some(x) = x.as_i64() => Lit::Int(x).to_type(),
            Number::Float(v) => {
                errors.add(
                    module_info,
                    x.range,
                    format!("Float literals are disallowed by the spec, got `{v}`"),
                );
                Type::any_error()
            }
            Number::Complex { real, imag } => {
                errors.add(
                    module_info,
                    x.range,
                    format!("Complex literals are not allowed, got `{real} + {imag}j`"),
                );
                Type::any_error()
            }
            _ => {
                errors.todo(module_info, "Lit::from_number_literal", x);
                Type::any_error()
            }
        }
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
    pub fn general_class_type(&self, stdlib: &Stdlib) -> ClassType {
        match self {
            Lit::String(_) => stdlib.str(),
            Lit::Int(_) => stdlib.int(),
            Lit::Bool(_) => stdlib.bool(),
            Lit::Bytes(_) => stdlib.bytes(),
            Lit::Enum(box (class_type, _)) => class_type.clone(),
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Lit::String(_))
    }
}

pub struct Enum {
    pub cls: ClassType,
}

impl Enum {
    pub fn get_member(&self, name: &Name) -> Option<Lit> {
        // TODO(stroxler, yangdanny) Enums can contain attributes that are not
        // members, we eventually need to implement enough checks to know the
        // difference.
        //
        // Instance-only attributes are one case of this and are correctly handled
        // upstream, but there are other cases as well.

        // Names starting but not ending with __ are private
        // Names starting and ending with _ are reserved by the enum
        if name.starts_with("__") && !name.ends_with("__")
            || name.starts_with("_") && name.ends_with("_")
        {
            None
        } else if self.cls.class_object().contains(name) {
            Some(Lit::Enum(Box::new((self.cls.clone(), name.clone()))))
        } else {
            None
        }
    }

    pub fn get_members(&self) -> SmallSet<Lit> {
        self.cls
            .class_object()
            .fields()
            .iter()
            .filter_map(|f| self.get_member(f))
            .collect()
    }
}
