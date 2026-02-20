/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::binding::KeyClass;
use crate::binding::bindings::BindingsBuilder;
use crate::export::special::SpecialExport;

/// We only recognize a small subset of syntactical forms for expression that may appear on base classes
#[derive(Debug, Clone)]
pub enum BaseClassExpr {
    Name(ExprName),
    Attribute {
        value: Box<BaseClassExpr>,
        attr: Identifier,
        range: TextRange,
    },
    Subscript {
        value: Box<BaseClassExpr>,
        slice: Box<Expr>,
        range: TextRange,
    },
}

impl Ranged for BaseClassExpr {
    fn range(&self) -> TextRange {
        match self {
            BaseClassExpr::Name(x) => x.range(),
            BaseClassExpr::Attribute { range, .. } => *range,
            BaseClassExpr::Subscript { range, .. } => *range,
        }
    }
}

impl BaseClassExpr {
    pub fn from_expr(expr: &Expr) -> Option<Self> {
        match expr {
            Expr::Name(x) => Some(Self::Name(x.clone())),
            Expr::Attribute(x) => {
                let value = Box::new(Self::from_expr(&x.value)?);
                let attr = x.attr.clone();
                let range = x.range();
                Some(Self::Attribute { value, attr, range })
            }
            Expr::Subscript(x) => {
                let value = Box::new(Self::from_expr(x.value.as_ref())?);
                let slice = x.slice.clone();
                let range = x.range();
                Some(Self::Subscript {
                    value,
                    slice,
                    range,
                })
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BaseClassGenericKind {
    Generic,
    Protocol,
}

#[derive(Debug, Clone)]
pub struct BaseClassGeneric {
    pub kind: BaseClassGenericKind,
    pub args: Box<[Expr]>,
    pub range: TextRange,
}

impl Ranged for BaseClassGeneric {
    fn range(&self) -> TextRange {
        self.range
    }
}

/// Helper type used to share part of the logic needed for the
/// binding-level work of finding legacy type parameters versus the type-level
/// work of computing inheritance information and the MRO.
#[derive(Debug, Clone)]
pub enum BaseClass {
    TypedDict(TextRange),
    Generic(BaseClassGeneric),
    BaseClassExpr(BaseClassExpr),
    InvalidExpr(Expr),
    NamedTuple(TextRange),
    /// A namedtuple class synthesized anonymously as a base class,
    /// e.g. `class Foo(namedtuple("Foo", ["a", "b"]))`.
    SynthesizedBase(Idx<KeyClass>, TextRange),
    /// `type(X)` where X is a statically determinable expression.
    /// At runtime, evaluates to the metaclass of X (if X is a class)
    /// or the class of X (if X is an instance).
    TypeOf(BaseClassExpr, TextRange),
}

impl BaseClass {
    pub fn is_generic(&self) -> bool {
        match self {
            BaseClass::Generic(x) if !x.args.is_empty() => true,
            _ => false,
        }
    }

    pub fn is_typed_dict(&self) -> bool {
        match self {
            BaseClass::TypedDict(_) => true,
            _ => false,
        }
    }
}

impl Ranged for BaseClass {
    fn range(&self) -> TextRange {
        match self {
            BaseClass::TypedDict(range) => *range,
            BaseClass::Generic(x) => x.range(),
            BaseClass::BaseClassExpr(base_expr) => base_expr.range(),
            BaseClass::InvalidExpr(expr) => expr.range(),
            BaseClass::NamedTuple(range) => *range,
            BaseClass::SynthesizedBase(_, range) => *range,
            BaseClass::TypeOf(_, range) => *range,
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    pub fn base_class_of(&self, base_expr: Expr) -> BaseClass {
        match self.as_special_export(&base_expr) {
            Some(SpecialExport::TypedDict) => BaseClass::TypedDict(base_expr.range()),
            Some(SpecialExport::TypingNamedTuple) | Some(SpecialExport::CollectionsNamedTuple) => {
                BaseClass::NamedTuple(base_expr.range())
            }
            Some(SpecialExport::Protocol) => BaseClass::Generic(BaseClassGeneric {
                kind: BaseClassGenericKind::Protocol,
                args: Box::new([]),
                range: base_expr.range(),
            }),
            Some(SpecialExport::Generic) => BaseClass::Generic(BaseClassGeneric {
                kind: BaseClassGenericKind::Generic,
                args: Box::new([]),
                range: base_expr.range(),
            }),
            _ => {
                if let Expr::Subscript(subscript) = &base_expr {
                    match self.as_special_export(&subscript.value) {
                        Some(SpecialExport::Protocol) => {
                            return BaseClass::Generic(BaseClassGeneric {
                                kind: BaseClassGenericKind::Protocol,
                                args: Ast::unpack_slice(&subscript.slice).into(),
                                range: base_expr.range(),
                            });
                        }
                        Some(SpecialExport::Generic) => {
                            return BaseClass::Generic(BaseClassGeneric {
                                kind: BaseClassGenericKind::Generic,
                                args: Ast::unpack_slice(&subscript.slice).into(),
                                range: base_expr.range(),
                            });
                        }
                        _ => {}
                    }
                }
                if let Expr::Call(call) = &base_expr
                    && call.arguments.args.len() == 1
                    && call.arguments.keywords.is_empty()
                    && self.as_special_export(&call.func) == Some(SpecialExport::BuiltinsType)
                    && let Some(inner) = BaseClassExpr::from_expr(&call.arguments.args[0])
                {
                    return BaseClass::TypeOf(inner, base_expr.range());
                }
                if let Some(valid_expr) = BaseClassExpr::from_expr(&base_expr) {
                    BaseClass::BaseClassExpr(valid_expr)
                } else {
                    BaseClass::InvalidExpr(base_expr)
                }
            }
        }
    }
}
