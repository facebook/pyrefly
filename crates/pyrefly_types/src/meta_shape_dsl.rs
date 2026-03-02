/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Pythonic DSL for defining tensor shape functions.
//!
//! This module provides the complete DSL infrastructure:
//!
//! - `Val` — runtime values (ints, dims, shapes, lists, etc.)
//! - `MetaShapeFunction` — trait implemented by all shape functions
//! - `extract` — helpers for converting `Type` to concrete values
//! - DSL grammar types (`DslFnDef`, `DslBody`, `DslExpr`, etc.)
//! - Parser (Python AST → grammar-aligned data types)
//! - Interpreter (evaluates DSL function bodies)
//! - `DslMetaShapeFunction` — `MetaShapeFunction` impl backed by a parsed DSL definition
//!
//! The data types mirror the DSL grammar defined in `meta_shape_pythonic.md`.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use pyrefly_python::ast::Ast;
use ruff_python_ast::BoolOp as RuffBoolOp;
use ruff_python_ast::CmpOp as RuffCmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator as RuffOperator;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp as RuffUnaryOp;

use crate::dimension::canonicalize;
use crate::lit_int::LitInt;
use crate::literal::Lit;
use crate::tensor::ShapeError;
use crate::tensor::SizeExpr;
use crate::tensor::TensorShape;
use crate::tensor::TensorType;
use crate::tuple::Tuple;
use crate::types::Type;

// ============================================================================
// Runtime Values
// ============================================================================

/// Runtime value produced by parameter extraction and manipulated by the
/// interpreter. Bridges between `Type` (the type-checker's representation)
/// and the shape computation domain.
#[derive(Debug, Clone)]
pub(crate) enum Val {
    /// Concrete integer (e.g., dim=0, stride=1).
    Int(i64),
    /// Boolean flag (e.g., keepdim=False).
    Bool(bool),
    /// String literal (e.g., einsum spec).
    Str(String),
    /// Single tensor dimension — a symbolic `Type` (SizeExpr, Quantified, etc.).
    Dim(Type),
    /// Full tensor shape with concrete rank.
    Shape(TensorShape),
    /// Homogeneous list. Elements are all the same variant (Int, Dim, Shape, …).
    List(Vec<Val>),
    /// Python None (for optional parameters).
    None,
}

impl Val {
    /// Extract as `i64`. Panics if not `Int` — the DSL type checker guarantees
    /// this won't happen for well-typed DSL code.
    pub fn as_int(&self) -> i64 {
        match self {
            Val::Int(n) => *n,
            _ => panic!("IR bug: expected Int, got {}", self.variant_name()),
        }
    }

    /// Extract as `bool`. Panics if not `Bool` — the DSL type checker guarantees
    /// this won't happen for well-typed DSL code.
    pub fn as_bool(&self) -> bool {
        match self {
            Val::Bool(b) => *b,
            _ => panic!("IR bug: expected Bool, got {}", self.variant_name()),
        }
    }

    /// Convert to a `Type::Size` for use in dimension arithmetic within the DSL evaluator.
    /// `Int(n)` becomes `Size(Literal(n))`; `Dim(ty)` passes through as-is.
    /// This is for *internal* symbolic computation, not for producing user-facing types
    /// (see `val_to_scalar_type` for that).
    pub fn as_size(&self) -> Type {
        match self {
            Val::Dim(ty) => ty.clone(),
            Val::Int(n) => Type::Size(SizeExpr::Literal(*n)),
            _ => panic!("IR bug: expected Dim or Int, got {}", self.variant_name()),
        }
    }

    /// Extract as `&[Val]`. Panics if not `List` — the DSL type checker guarantees
    /// this won't happen for well-typed DSL code.
    pub fn as_list(&self) -> &[Val] {
        match self {
            Val::List(items) => items,
            _ => panic!("IR bug: expected List, got {}", self.variant_name()),
        }
    }

    /// Extract as `&TensorShape`. Panics if not `Shape` — the DSL type checker
    /// guarantees this won't happen for well-typed DSL code.
    pub fn as_shape(&self) -> &TensorShape {
        match self {
            Val::Shape(s) => s,
            _ => panic!("IR bug: expected Shape, got {}", self.variant_name()),
        }
    }

    /// Extract as `&str`. Panics if not `Str` — the DSL type checker guarantees
    /// this won't happen for well-typed DSL code.
    pub fn as_str_val(&self) -> &str {
        match self {
            Val::Str(s) => s.as_str(),
            _ => panic!("IR bug: expected Str, got {}", self.variant_name()),
        }
    }

    /// Extract a list of `Type::Size` values from a `Val::List`, for use in shape arithmetic.
    pub fn as_size_list(&self) -> Vec<Type> {
        self.as_list().iter().map(|v| v.as_size()).collect()
    }

    /// Short name for error messages.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Val::Int(_) => "Int",
            Val::Bool(_) => "Bool",
            Val::Str(_) => "Str",
            Val::Dim(_) => "Dim",
            Val::Shape(_) => "Shape",
            Val::List(_) => "List",
            Val::None => "None",
        }
    }
}

// ============================================================================
// Extraction Helpers
// ============================================================================

/// Helper functions for extracting typed values from `Type`.
///
/// These are used in `bind_dsl_params()` to convert bound Python types
/// to runtime values. Each returns `None` if the type doesn't match.
pub(crate) mod extract {
    use crate::literal::Lit;
    use crate::literal::Literal;
    use crate::tensor::TensorShape;
    use crate::tuple::Tuple;
    use crate::types::Type;

    /// Extract a concrete TensorShape from a Type.
    /// Returns None for non-tensors, shapeless tensors, and variadic (Unpacked) shapes.
    /// Use this when the caller needs to iterate over individual dimensions.
    pub fn concrete_tensor_shape(ty: &Type) -> Option<TensorShape> {
        match ty {
            Type::Tensor(tensor) => match &tensor.shape {
                TensorShape::Concrete(_) => Some(tensor.shape.clone()),
                TensorShape::Unpacked(_) => None,
            },
            Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                // Extract shape from ClassType targs
                let targs = cls.targs();
                if targs.is_empty() {
                    return None;
                }
                // First targ should be the shape tuple
                if let Type::Tuple(Tuple::Concrete(elems)) = &targs.as_slice()[0] {
                    let dims: Vec<Type> = elems.iter().filter_map(dimension).collect();
                    if dims.len() == elems.len() && !dims.is_empty() {
                        return Some(TensorShape::from_types(dims));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract literal int from Type::Literal(Lit::Int(...)).
    pub fn literal_int(ty: &Type) -> Option<i64> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Int(n), ..
            }) => n.as_i64(),
            _ => None,
        }
    }

    /// Extract symbolic dimension from Type.
    /// Handles Dim[N], SizeExpr, Quantified, Var, etc.
    pub fn dimension(ty: &Type) -> Option<Type> {
        match ty {
            // Dim[inner] -> extract inner (could be Quantified, SizeExpr, etc.)
            Type::Dim(inner) => Some((**inner).clone()),
            // Already a SizeExpr
            Type::Size(_) => Some(ty.clone()),
            // Type variable or quantified
            Type::Quantified(_) | Type::Var(_) => Some(ty.clone()),
            // Literal int -> wrap in SizeExpr
            Type::Literal(box Literal {
                value: Lit::Int(n), ..
            }) => n
                .as_i64()
                .map(|v| Type::Size(crate::tensor::SizeExpr::Literal(v))),
            _ => None,
        }
    }

    /// Extract int list from tuple of literal ints.
    /// Returns None if any element is not a literal int.
    /// Also handles nested tuples (e.g., from variadic binding of tuple args).
    pub fn int_list(ty: &Type) -> Option<Vec<i64>> {
        match ty {
            Type::Tuple(Tuple::Concrete(elts)) => {
                // First, try to extract ints directly
                let result = elts.iter().map(literal_int).collect::<Option<Vec<i64>>>();
                if result.is_some() {
                    result
                } else if elts.len() == 1 {
                    // Handle nested tuple case: tuple[tuple[ints...]] -> extract inner tuple
                    int_list(&elts[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract dimension list from tuple of Dims or dimension types.
    /// Returns None if any element is not a valid dimension type.
    /// Also handles nested tuples (e.g., from variadic binding of tuple args).
    pub fn dim_list(ty: &Type) -> Option<Vec<Type>> {
        match ty {
            Type::Tuple(Tuple::Concrete(elts)) => {
                // First, try to extract dimensions directly
                let result = elts.iter().map(dimension).collect::<Option<Vec<Type>>>();
                if result.is_some() {
                    result
                } else if elts.len() == 1 {
                    // Handle nested tuple case: tuple[tuple[dims...]] -> extract inner tuple
                    dim_list(&elts[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract bool literal from Type::Literal(Lit::Bool(...)).
    pub fn bool_arg(ty: &Type) -> Option<bool> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Bool(b),
                ..
            }) => Some(*b),
            _ => None,
        }
    }

    /// Extract string literal from Type::Literal(Lit::Str(...)).
    pub fn string_arg(ty: &Type) -> Option<String> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Str(s), ..
            }) => Some(s.to_string()),
            _ => None,
        }
    }

    /// Extract list or tuple of tensor shapes.
    /// Handles both list[Tensor[...]] and tuple[Tensor[...], ...].
    /// Returns None for list types (can't determine element count) or unbounded tuples.
    pub fn tensor_list(ty: &Type) -> Option<Vec<TensorShape>> {
        use crate::tuple::Tuple;

        match ty {
            // list[Tensor[...]] - can't determine element count, return None
            Type::ClassType(class_type) if class_type.has_qname("builtins", "list") => {
                // Lists don't preserve element count in the type system
                // Fall back to fixture for now
                None
            }
            // tuple[Tensor[...], ...] - unbounded, can't determine count
            Type::Tuple(Tuple::Unbounded(_)) => None,
            // tuple[Tensor[...], Tensor[...], ...] - concrete, extract all
            Type::Tuple(Tuple::Concrete(elems)) => {
                // Check if first element is a tensor
                if let Some(first) = elems.first() {
                    let is_tensor = match first {
                        Type::Tensor(_) => true,
                        Type::ClassType(ct) => ct.has_qname("torch", "Tensor"),
                        _ => false,
                    };

                    if is_tensor {
                        // Tuple of tensors - extract all
                        let shapes: Option<Vec<TensorShape>> =
                            elems.iter().map(concrete_tensor_shape).collect();
                        return shapes;
                    }
                }
                None
            }
            _ => None,
        }
    }
}

// ============================================================================
// Meta-Shape Function Trait
// ============================================================================

/// A function that computes output shapes from input shapes.
///
/// The `evaluate` method takes bound arguments (from the call site) and the
/// fixture return type, and produces the refined return type directly.
/// This is symmetric with parameter binding: on the way in, `(Type, DslType) → Val`;
/// on the way out, `(Val, DslType, Type) → Type`.
pub trait MetaShapeFunction: Debug + Send + Sync {
    /// Name of this meta-shape function (for error messages).
    fn name(&self) -> &str;

    /// Evaluate bound arguments and return the refined return type directly.
    ///
    /// `ret_type` is the fixture return type (provides base class info for tensors,
    /// tuple structure, etc.).
    ///
    /// Returns `None` to fall back to `ret_type` (e.g., missing shape info).
    /// Returns `Some(Ok(ty))` on success, `Some(Err(e))` on shape error.
    fn evaluate(
        &self,
        bound_args: &HashMap<String, Type>,
        ret_type: &Type,
    ) -> Option<Result<Type, ShapeError>>;
}

// ============================================================================
// Grammar-aligned data types
// ============================================================================

/// Binary operators: arithmetic, comparison, and logical.
/// Corresponds to OP in `<expr> OP <expr>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DslOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    // Comparison
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    // Logical
    And,
    Or,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DslUnaryOp {
    Not,
    Neg,
}

/// Well-known builtin functions, validated at parse time.
/// A typo in the DSL source (e.g., `prodd`) will be caught immediately
/// as an undefined user-defined function rather than silently falling through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DslBuiltin {
    Len,
    Range,
    Prod,
    Sum,
    Str,
    ParseEinsumEquation,
    Enumerate,
    Zip,
}

/// Target of a function call — either a builtin or a user-defined function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DslCallTarget {
    Builtin(DslBuiltin),
    UserDefined(String),
}

/// Type constructors for `isinstance` checks.
/// These are nullary: `isinstance(x, list)` checks the constructor, not the element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DslTypeCon {
    Int,
    Str,
    Bool,
    SymInt,
    List,
    Tensor,
}

/// Types in the DSL. Corresponds to `<type>` in the grammar,
/// extended with Tuple for return type annotations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DslType {
    Int,
    SymInt,
    Bool,
    Str,
    Tensor,
    None,
    /// `list[T]`
    List(Box<DslType>),
    /// `T1 | T2 | ...`
    Union(Vec<DslType>),
    /// `[T1, T2]` — fixed-size tuple (used in return type annotations).
    Tuple(Vec<DslType>),
}

/// Constant values. Corresponds to `<const>` in the grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DslConst {
    None,
    Int(i64),
    Bool(bool),
    Str(String),
}

/// Function parameter. Corresponds to `<param>` in the grammar.
#[derive(Debug, Clone)]
pub(crate) struct DslParam {
    pub(crate) name: String,
    pub(crate) ty: DslType,
    pub(crate) default: Option<DslConst>,
}

/// Function body. Corresponds to `<body>` in the grammar.
/// This is a recursive (linked-list) structure where Assign and If
/// have a `rest` continuation, and Return/Raise are terminals.
#[derive(Debug, Clone)]
pub(crate) enum DslBody {
    /// `x, ..., x = <expr>; <body>`
    Assign {
        vars: Vec<String>,
        expr: DslExpr,
        rest: Box<DslBody>,
    },
    /// `if <expr>: <body>; <body>`
    If {
        cond: DslExpr,
        then_body: Box<DslBody>,
        rest: Box<DslBody>,
    },
    /// `return <expr>`
    Return(DslExpr),
    /// `raise Error(expr)`
    Raise(DslExpr),
}

/// Expressions. Corresponds to `<expr>` in the grammar.
#[derive(Debug, Clone)]
pub(crate) enum DslExpr {
    /// Literal constant.
    Const(DslConst),
    /// Variable reference.
    Var(String),
    /// List literal `[e1, e2, ...]`.
    List(Vec<DslExpr>),
    /// List comprehension `[elt for vars in iter if cond]`.
    ListComp {
        elt: Box<DslExpr>,
        vars: Vec<String>,
        iter: Box<DslExpr>,
        cond: Option<Box<DslExpr>>,
    },
    /// Indexing `base[index]`.
    Index {
        base: Box<DslExpr>,
        index: Box<DslExpr>,
    },
    /// Slicing `base[lower:upper]`.
    Slice {
        base: Box<DslExpr>,
        lower: Option<Box<DslExpr>>,
        upper: Option<Box<DslExpr>>,
    },
    /// Binary operation `left OP right`.
    BinOp {
        left: Box<DslExpr>,
        op: DslOp,
        right: Box<DslExpr>,
    },
    /// Unary operation.
    UnaryOp {
        op: DslUnaryOp,
        operand: Box<DslExpr>,
    },
    /// Function call `f(args...)`.
    Call {
        func: DslCallTarget,
        args: Vec<DslExpr>,
    },
    /// `isinstance(expr, type_constructor)`
    IsInstance { expr: Box<DslExpr>, ty: DslTypeCon },
    /// `expr in expr` (membership test).
    In {
        left: Box<DslExpr>,
        right: Box<DslExpr>,
    },
    /// `expr.shape` (extract tensor dimensions).
    Shape(Box<DslExpr>),
    /// `Tensor(shape=expr)` (construct result tensor).
    TensorNew(Box<DslExpr>),
    /// Ternary `body if test else orelse`.
    IfExpr {
        body: Box<DslExpr>,
        test: Box<DslExpr>,
        orelse: Box<DslExpr>,
    },
    /// `...` (Ellipsis sentinel for unbounded tuples).
    Ellipsis,
    /// `Unknown` (sentinel for fixture fallback).
    Unknown,
}

/// Function definition. Corresponds to `<fndef>` in the grammar.
#[derive(Debug, Clone)]
pub(crate) struct DslFnDef {
    pub(crate) name: String,
    pub(crate) params: Vec<DslParam>,
    pub(crate) return_type: Option<DslType>,
    pub(crate) body: DslBody,
}

// ============================================================================
// Display implementations
// ============================================================================

impl fmt::Display for DslOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslOp::Add => write!(f, "+"),
            DslOp::Sub => write!(f, "-"),
            DslOp::Mul => write!(f, "*"),
            DslOp::FloorDiv => write!(f, "//"),
            DslOp::Mod => write!(f, "%"),
            DslOp::Eq => write!(f, "=="),
            DslOp::NotEq => write!(f, "!="),
            DslOp::Lt => write!(f, "<"),
            DslOp::LtE => write!(f, "<="),
            DslOp::Gt => write!(f, ">"),
            DslOp::GtE => write!(f, ">="),
            DslOp::And => write!(f, "and"),
            DslOp::Or => write!(f, "or"),
        }
    }
}

impl fmt::Display for DslUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslUnaryOp::Not => write!(f, "not "),
            DslUnaryOp::Neg => write!(f, "-"),
        }
    }
}

impl fmt::Display for DslBuiltin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslBuiltin::Len => write!(f, "len"),
            DslBuiltin::Range => write!(f, "range"),
            DslBuiltin::Prod => write!(f, "torch_shapes.prod"),
            DslBuiltin::Sum => write!(f, "torch_shapes.sum"),
            DslBuiltin::Str => write!(f, "str"),
            DslBuiltin::ParseEinsumEquation => write!(f, "torch_shapes.parse_einsum_equation"),
            DslBuiltin::Enumerate => write!(f, "enumerate"),
            DslBuiltin::Zip => write!(f, "zip"),
        }
    }
}

impl fmt::Display for DslCallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslCallTarget::Builtin(b) => write!(f, "{}", b),
            DslCallTarget::UserDefined(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for DslTypeCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslTypeCon::Int => write!(f, "int"),
            DslTypeCon::Str => write!(f, "str"),
            DslTypeCon::Bool => write!(f, "bool"),
            DslTypeCon::SymInt => write!(f, "symint"),
            DslTypeCon::List => write!(f, "list"),
            DslTypeCon::Tensor => write!(f, "Tensor"),
        }
    }
}

impl fmt::Display for DslType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslType::Int => write!(f, "int"),
            DslType::SymInt => write!(f, "symint"),
            DslType::Bool => write!(f, "bool"),
            DslType::Str => write!(f, "str"),
            DslType::Tensor => write!(f, "Tensor"),
            DslType::None => write!(f, "None"),
            DslType::List(inner) => write!(f, "list[{}]", inner),
            DslType::Union(types) => {
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                Ok(())
            }
            DslType::Tuple(types) => {
                write!(f, "[")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl fmt::Display for DslConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslConst::None => write!(f, "None"),
            DslConst::Int(i) => write!(f, "{}", i),
            DslConst::Bool(b) => {
                if *b {
                    write!(f, "True")
                } else {
                    write!(f, "False")
                }
            }
            DslConst::Str(s) => write!(f, "\"{}\"", s),
        }
    }
}

impl fmt::Display for DslExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslExpr::Const(c) => write!(f, "{}", c),
            DslExpr::Var(name) => write!(f, "{}", name),
            DslExpr::List(elts) => {
                write!(f, "[")?;
                for (i, e) in elts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            DslExpr::ListComp {
                elt,
                vars,
                iter,
                cond,
            } => {
                write!(f, "[{} for {} in {}", elt, vars.join(", "), iter)?;
                if let Some(c) = cond {
                    write!(f, " if {}", c)?;
                }
                write!(f, "]")
            }
            DslExpr::Index { base, index } => write!(f, "{}[{}]", base, index),
            DslExpr::Slice { base, lower, upper } => {
                write!(f, "{}[", base)?;
                if let Some(l) = lower {
                    write!(f, "{}", l)?;
                }
                write!(f, ":")?;
                if let Some(u) = upper {
                    write!(f, "{}", u)?;
                }
                write!(f, "]")
            }
            DslExpr::BinOp { left, op, right } => {
                write!(f, "{} {} {}", left, op, right)
            }
            DslExpr::UnaryOp { op, operand } => write!(f, "{}{}", op, operand),
            DslExpr::Call { func, args } => {
                write!(f, "{}(", func)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            DslExpr::IsInstance { expr, ty } => write!(f, "isinstance({}, {})", expr, ty),
            DslExpr::In { left, right } => write!(f, "{} in {}", left, right),
            DslExpr::Shape(expr) => write!(f, "{}.shape", expr),
            DslExpr::TensorNew(expr) => write!(f, "Tensor(shape={})", expr),
            DslExpr::IfExpr { body, test, orelse } => {
                write!(f, "{} if {} else {}", body, test, orelse)
            }
            DslExpr::Ellipsis => write!(f, "..."),
            DslExpr::Unknown => write!(f, "Unknown"),
        }
    }
}

impl fmt::Display for DslParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)?;
        if let Some(d) = &self.default {
            write!(f, " = {}", d)?;
        }
        Ok(())
    }
}

// ============================================================================
// AST conversion: ruff Python AST → DSL grammar types
// ============================================================================

/// Convert an isinstance type argument to a DslTypeCon.
fn convert_type_constructor(expr: &Expr) -> Result<DslTypeCon, String> {
    match expr {
        Expr::Name(n) => match n.id.as_str() {
            "int" => Ok(DslTypeCon::Int),
            "str" => Ok(DslTypeCon::Str),
            "bool" => Ok(DslTypeCon::Bool),
            "symint" => Ok(DslTypeCon::SymInt),
            "list" => Ok(DslTypeCon::List),
            "Tensor" => Ok(DslTypeCon::Tensor),
            other => Err(format!(
                "unknown type constructor '{}' in isinstance. \
                 Expected one of: int, str, bool, symint, list, Tensor",
                other
            )),
        },
        _ => Err(format!(
            "expected type name in isinstance, got {:?}",
            std::mem::discriminant(expr)
        )),
    }
}

/// Convert a ruff type annotation expression to a DslType.
fn convert_type_annotation(expr: &Expr) -> Result<DslType, String> {
    match expr {
        Expr::Name(n) => match n.id.as_str() {
            "int" => Ok(DslType::Int),
            "symint" => Ok(DslType::SymInt),
            "bool" => Ok(DslType::Bool),
            "str" => Ok(DslType::Str),
            "Tensor" => Ok(DslType::Tensor),
            "None" => Ok(DslType::None),
            other => Err(format!(
                "unknown type '{}' in annotation. \
                 Expected one of: int, symint, bool, str, Tensor, None",
                other
            )),
        },
        Expr::NoneLiteral(_) => Ok(DslType::None),
        // list[T]
        Expr::Subscript(sub) => {
            if let Expr::Name(n) = sub.value.as_ref() {
                if n.id.as_str() != "list" {
                    return Err(format!(
                        "only list[T] subscripts are supported in DSL types, got {}[...]",
                        n.id
                    ));
                }
                Ok(DslType::List(Box::new(convert_type_annotation(
                    &sub.slice,
                )?)))
            } else {
                Err("unexpected subscript base in DSL type: expected a name like 'list'".to_owned())
            }
        }
        // T1 | T2 (union) — BinOp with BitOr
        Expr::BinOp(binop) => {
            if !matches!(binop.op, RuffOperator::BitOr) {
                return Err(format!(
                    "only | operator is supported in DSL type unions, got {:?}",
                    binop.op
                ));
            }
            let mut types = Vec::new();
            flatten_union(&binop.left, &mut types)?;
            flatten_union(&binop.right, &mut types)?;
            Ok(DslType::Union(types))
        }
        // [T1, T2] — list literal used as fixed-size tuple type annotation
        Expr::List(list) => {
            let types: Vec<DslType> = list
                .elts
                .iter()
                .map(convert_type_annotation)
                .collect::<Result<_, _>>()?;
            Ok(DslType::Tuple(types))
        }
        _ => Err(format!(
            "unexpected expression in DSL type annotation: {:?}",
            std::mem::discriminant(expr)
        )),
    }
}

/// Flatten nested BitOr unions into a vec: `(a | b) | c` → [a, b, c].
fn flatten_union(expr: &Expr, out: &mut Vec<DslType>) -> Result<(), String> {
    if let Expr::BinOp(binop) = expr
        && matches!(binop.op, RuffOperator::BitOr)
    {
        flatten_union(&binop.left, out)?;
        flatten_union(&binop.right, out)?;
        return Ok(());
    }
    out.push(convert_type_annotation(expr)?);
    Ok(())
}

/// Convert a default value expression to a DslConst.
fn convert_default(expr: &Expr) -> Result<DslConst, String> {
    match expr {
        Expr::NoneLiteral(_) => Ok(DslConst::None),
        Expr::BooleanLiteral(b) => Ok(DslConst::Bool(b.value)),
        Expr::NumberLiteral(n) => match &n.value {
            Number::Int(i) => {
                Ok(DslConst::Int(i.as_i64().ok_or_else(|| {
                    format!("default int literal too large: {}", i)
                })?))
            }
            _ => Err("non-int number as default value in DSL".to_owned()),
        },
        Expr::UnaryOp(u) if matches!(u.op, RuffUnaryOp::USub) => {
            if let Expr::NumberLiteral(n) = u.operand.as_ref()
                && let Number::Int(i) = &n.value
            {
                return Ok(DslConst::Int(
                    -i.as_i64()
                        .ok_or_else(|| format!("default int literal too large: {}", i))?,
                ));
            }
            Err("unexpected unary expression as default value in DSL".to_owned())
        }
        Expr::StringLiteral(s) => Ok(DslConst::Str(s.value.to_str().to_owned())),
        Expr::Name(n) if n.id.as_str() == "None" => Ok(DslConst::None),
        _ => Err(format!(
            "unexpected default value in DSL: {:?}",
            std::mem::discriminant(expr)
        )),
    }
}

/// Extract variable names from an assignment target.
fn extract_assign_vars(targets: &[Expr]) -> Result<Vec<String>, String> {
    if targets.len() != 1 {
        return Err(format!(
            "DSL assignment must have exactly one target, got {}",
            targets.len()
        ));
    }
    match &targets[0] {
        Expr::Name(n) => Ok(vec![n.id.to_string()]),
        Expr::Tuple(t) => t
            .elts
            .iter()
            .map(|e| match e {
                Expr::Name(n) => Ok(n.id.to_string()),
                _ => Err("expected name in assignment tuple target".to_owned()),
            })
            .collect(),
        _ => Err("unexpected assignment target in DSL".to_owned()),
    }
}

/// Extract variable names from a comprehension target.
fn extract_comp_vars(target: &Expr) -> Result<Vec<String>, String> {
    match target {
        Expr::Name(n) => Ok(vec![n.id.to_string()]),
        Expr::Tuple(t) => t
            .elts
            .iter()
            .map(|e| match e {
                Expr::Name(n) => Ok(n.id.to_string()),
                _ => Err("expected name in comprehension target".to_owned()),
            })
            .collect(),
        _ => Err("unexpected comprehension target in DSL".to_owned()),
    }
}

/// Extract the error message expression from a `raise Error(expr)` statement.
/// The expression is converted to a DslExpr and evaluated at runtime.
fn extract_error_expr(exc: &Expr) -> Result<DslExpr, String> {
    if let Expr::Call(call) = exc
        && let Expr::Name(n) = call.func.as_ref()
    {
        if n.id.as_str() != "Error" {
            return Err(format!("DSL raise must use Error(), got {}()", n.id));
        }
        if call.arguments.args.len() != 1 {
            return Err(format!(
                "Error() must have exactly one argument, got {}",
                call.arguments.args.len()
            ));
        }
        return convert_expr(&call.arguments.args[0]);
    }
    Err("expected raise Error(expr) in DSL".to_owned())
}

/// Convert a sequence of Python statements into a DslBody.
/// The grammar's body is a recursive structure: assignments and ifs have
/// continuations, while return and raise are terminals.
fn convert_body(stmts: &[Stmt]) -> Result<DslBody, String> {
    if stmts.is_empty() {
        return Err("empty body in DSL function".to_owned());
    }

    match &stmts[0] {
        Stmt::Assign(assign) => {
            let vars = extract_assign_vars(&assign.targets)?;
            let expr = convert_expr(&assign.value)?;
            let rest = convert_body(&stmts[1..])?;
            Ok(DslBody::Assign {
                vars,
                expr,
                rest: Box::new(rest),
            })
        }
        Stmt::If(if_stmt) => {
            if !if_stmt.elif_else_clauses.is_empty() {
                return Err("DSL if must not have elif/else (use early returns)".to_owned());
            }
            let cond = convert_expr(&if_stmt.test)?;
            let then_body = convert_body(&if_stmt.body)?;
            let rest = convert_body(&stmts[1..])?;
            Ok(DslBody::If {
                cond,
                then_body: Box::new(then_body),
                rest: Box::new(rest),
            })
        }
        Stmt::Return(ret) => {
            let value = ret
                .value
                .as_ref()
                .ok_or_else(|| "DSL return must have a value".to_owned())?;
            Ok(DslBody::Return(convert_expr(value)?))
        }
        Stmt::Raise(raise) => {
            let exc = raise
                .exc
                .as_ref()
                .ok_or_else(|| "DSL raise must have an exception".to_owned())?;
            Ok(DslBody::Raise(extract_error_expr(exc)?))
        }
        _ => Err(format!(
            "unexpected statement in DSL body: {:?}",
            std::mem::discriminant(&stmts[0])
        )),
    }
}

/// Convert a ruff expression into a DslExpr.
fn convert_expr(expr: &Expr) -> Result<DslExpr, String> {
    match expr {
        // Constants
        Expr::NoneLiteral(_) => Ok(DslExpr::Const(DslConst::None)),
        Expr::BooleanLiteral(b) => Ok(DslExpr::Const(DslConst::Bool(b.value))),
        Expr::NumberLiteral(n) => match &n.value {
            Number::Int(i) => Ok(DslExpr::Const(DslConst::Int(
                i.as_i64()
                    .ok_or_else(|| format!("int literal too large: {}", i))?,
            ))),
            _ => Err("non-int number in DSL expression".to_owned()),
        },
        Expr::StringLiteral(s) => Ok(DslExpr::Const(DslConst::Str(s.value.to_str().to_owned()))),
        Expr::EllipsisLiteral(_) => Ok(DslExpr::Ellipsis),

        // Variable reference or special names
        Expr::Name(n) => Ok(match n.id.as_str() {
            "Unknown" => DslExpr::Unknown,
            "None" => DslExpr::Const(DslConst::None),
            "True" => DslExpr::Const(DslConst::Bool(true)),
            "False" => DslExpr::Const(DslConst::Bool(false)),
            other => DslExpr::Var(other.to_owned()),
        }),

        // Attribute access: only x.shape is supported
        Expr::Attribute(attr) => {
            if attr.attr.as_str() != "shape" {
                return Err(format!(
                    "only .shape attribute access is supported in DSL, got .{}",
                    attr.attr
                ));
            }
            Ok(DslExpr::Shape(Box::new(convert_expr(&attr.value)?)))
        }

        // Subscript: either indexing or slicing
        Expr::Subscript(sub) => {
            let base = convert_expr(&sub.value)?;
            match sub.slice.as_ref() {
                Expr::Slice(slice) => {
                    let lower = slice
                        .lower
                        .as_ref()
                        .map(|e| convert_expr(e).map(Box::new))
                        .transpose()?;
                    let upper = slice
                        .upper
                        .as_ref()
                        .map(|e| convert_expr(e).map(Box::new))
                        .transpose()?;
                    Ok(DslExpr::Slice {
                        base: Box::new(base),
                        lower,
                        upper,
                    })
                }
                other => Ok(DslExpr::Index {
                    base: Box::new(base),
                    index: Box::new(convert_expr(other)?),
                }),
            }
        }

        // List literal
        Expr::List(list) => {
            let elts: Vec<DslExpr> = list
                .elts
                .iter()
                .map(convert_expr)
                .collect::<Result<_, _>>()?;
            Ok(DslExpr::List(elts))
        }

        // List comprehension
        Expr::ListComp(comp) => {
            if comp.generators.len() != 1 {
                return Err(format!(
                    "DSL list comprehension must have exactly one generator, got {}",
                    comp.generators.len()
                ));
            }
            let generator = &comp.generators[0];
            let vars = extract_comp_vars(&generator.target)?;
            let iter = convert_expr(&generator.iter)?;
            let cond = if generator.ifs.is_empty() {
                None
            } else {
                // Multiple if-clauses are ANDed together
                let mut combined = convert_expr(&generator.ifs[0])?;
                for if_clause in &generator.ifs[1..] {
                    combined = DslExpr::BinOp {
                        left: Box::new(combined),
                        op: DslOp::And,
                        right: Box::new(convert_expr(if_clause)?),
                    };
                }
                Some(Box::new(combined))
            };
            Ok(DslExpr::ListComp {
                elt: Box::new(convert_expr(&comp.elt)?),
                vars,
                iter: Box::new(iter),
                cond,
            })
        }

        // Tuple (used as expression, e.g. in parenthesized expressions)
        Expr::Tuple(t) => {
            // In our DSL, tuples in expression context are treated as lists
            let elts: Vec<DslExpr> = t.elts.iter().map(convert_expr).collect::<Result<_, _>>()?;
            Ok(DslExpr::List(elts))
        }

        // Function call: dispatch to special forms or general Call
        Expr::Call(call) => convert_call(call),

        // Binary operation
        Expr::BinOp(binop) => {
            let op = match binop.op {
                RuffOperator::Add => DslOp::Add,
                RuffOperator::Sub => DslOp::Sub,
                RuffOperator::Mult => DslOp::Mul,
                RuffOperator::FloorDiv => DslOp::FloorDiv,
                RuffOperator::Mod => DslOp::Mod,
                _ => {
                    return Err(format!(
                        "unsupported binary operator in DSL: {:?}",
                        binop.op
                    ));
                }
            };
            Ok(DslExpr::BinOp {
                left: Box::new(convert_expr(&binop.left)?),
                op,
                right: Box::new(convert_expr(&binop.right)?),
            })
        }

        // Unary operation
        Expr::UnaryOp(unary) => {
            let op = match unary.op {
                RuffUnaryOp::Not => DslUnaryOp::Not,
                RuffUnaryOp::USub => DslUnaryOp::Neg,
                _ => return Err(format!("unsupported unary operator in DSL: {:?}", unary.op)),
            };
            Ok(DslExpr::UnaryOp {
                op,
                operand: Box::new(convert_expr(&unary.operand)?),
            })
        }

        // Boolean operation (and/or): chain into binary ops
        Expr::BoolOp(boolop) => {
            let op = match boolop.op {
                RuffBoolOp::And => DslOp::And,
                RuffBoolOp::Or => DslOp::Or,
            };
            if boolop.values.len() < 2 {
                return Err("BoolOp must have at least 2 values".to_owned());
            }
            let mut result = convert_expr(&boolop.values[0])?;
            for value in &boolop.values[1..] {
                result = DslExpr::BinOp {
                    left: Box::new(result),
                    op,
                    right: Box::new(convert_expr(value)?),
                };
            }
            Ok(result)
        }

        // Comparison: dispatch to BinOp or In
        Expr::Compare(cmp) => {
            if cmp.ops.len() != 1 {
                return Err("DSL does not support chained comparisons".to_owned());
            }
            let left = convert_expr(&cmp.left)?;
            let right = convert_expr(&cmp.comparators[0])?;
            match cmp.ops[0] {
                RuffCmpOp::In => Ok(DslExpr::In {
                    left: Box::new(left),
                    right: Box::new(right),
                }),
                RuffCmpOp::NotIn => Ok(DslExpr::UnaryOp {
                    op: DslUnaryOp::Not,
                    operand: Box::new(DslExpr::In {
                        left: Box::new(left),
                        right: Box::new(right),
                    }),
                }),
                _ => {
                    let op = match cmp.ops[0] {
                        RuffCmpOp::Eq => DslOp::Eq,
                        RuffCmpOp::NotEq => DslOp::NotEq,
                        RuffCmpOp::Lt => DslOp::Lt,
                        RuffCmpOp::LtE => DslOp::LtE,
                        RuffCmpOp::Gt => DslOp::Gt,
                        RuffCmpOp::GtE => DslOp::GtE,
                        _ => {
                            return Err(format!(
                                "unsupported comparison op in DSL: {:?}",
                                cmp.ops[0]
                            ));
                        }
                    };
                    Ok(DslExpr::BinOp {
                        left: Box::new(left),
                        op,
                        right: Box::new(right),
                    })
                }
            }
        }

        // Ternary expression: body if test else orelse
        Expr::If(if_expr) => Ok(DslExpr::IfExpr {
            body: Box::new(convert_expr(&if_expr.body)?),
            test: Box::new(convert_expr(&if_expr.test)?),
            orelse: Box::new(convert_expr(&if_expr.orelse)?),
        }),

        _ => Err(format!(
            "unexpected expression in DSL: {:?}",
            std::mem::discriminant(expr)
        )),
    }
}

/// Convert a function call expression, dispatching special forms.
fn convert_call(call: &ruff_python_ast::ExprCall) -> Result<DslExpr, String> {
    // Extract function name for dispatch. Supports both simple names (`len`)
    // and dotted names (`torch_shapes.prod`).
    let func_name = match call.func.as_ref() {
        Expr::Name(n) => n.id.to_string(),
        Expr::Attribute(a) => {
            let prefix = match a.value.as_ref() {
                Expr::Name(n) => n.id.as_str(),
                _ => return Err(format!("unsupported call target: {:?}", call.func)),
            };
            format!("{}.{}", prefix, a.attr)
        }
        _ => return Err(format!("unsupported call target: {:?}", call.func)),
    };

    match func_name.as_str() {
        // Special forms with non-call syntax — keep as dedicated DslExpr variants
        "isinstance" => {
            if call.arguments.args.len() != 2 {
                return Err(format!(
                    "isinstance() takes exactly 2 arguments, got {}",
                    call.arguments.args.len()
                ));
            }
            Ok(DslExpr::IsInstance {
                expr: Box::new(convert_expr(&call.arguments.args[0])?),
                ty: convert_type_constructor(&call.arguments.args[1])?,
            })
        }
        "Tensor" => {
            // Tensor(shape=expr) — keyword argument
            if !call.arguments.args.is_empty() {
                return Err("Tensor() uses keyword arg shape=, not positional args".to_owned());
            }
            if call.arguments.keywords.len() != 1 {
                return Err(format!(
                    "Tensor() takes exactly one keyword arg, got {}",
                    call.arguments.keywords.len()
                ));
            }
            let kw = &call.arguments.keywords[0];
            let kw_name = kw
                .arg
                .as_ref()
                .ok_or_else(|| "Tensor keyword must be named".to_owned())?
                .as_str();
            if kw_name != "shape" {
                return Err(format!(
                    "Tensor() keyword must be 'shape', got '{}'",
                    kw_name
                ));
            }
            Ok(DslExpr::TensorNew(Box::new(convert_expr(&kw.value)?)))
        }

        // Builtins validated at parse time
        "len" => {
            if call.arguments.args.len() != 1 {
                return Err(format!(
                    "len() takes exactly 1 argument, got {}",
                    call.arguments.args.len()
                ));
            }
            Ok(DslExpr::Call {
                func: DslCallTarget::Builtin(DslBuiltin::Len),
                args: vec![convert_expr(&call.arguments.args[0])?],
            })
        }
        "range" => {
            if call.arguments.args.len() != 1 {
                return Err(format!(
                    "range() takes exactly 1 argument in DSL, got {}",
                    call.arguments.args.len()
                ));
            }
            Ok(DslExpr::Call {
                func: DslCallTarget::Builtin(DslBuiltin::Range),
                args: vec![convert_expr(&call.arguments.args[0])?],
            })
        }
        "str"
        | "enumerate"
        | "zip"
        | "torch_shapes.prod"
        | "torch_shapes.sum"
        | "torch_shapes.parse_einsum_equation" => {
            let builtin = match func_name.as_str() {
                "torch_shapes.prod" => DslBuiltin::Prod,
                "torch_shapes.sum" => DslBuiltin::Sum,
                "str" => DslBuiltin::Str,
                "torch_shapes.parse_einsum_equation" => DslBuiltin::ParseEinsumEquation,
                "enumerate" => DslBuiltin::Enumerate,
                "zip" => DslBuiltin::Zip,
                _ => unreachable!(),
            };
            let args = call
                .arguments
                .args
                .iter()
                .map(convert_expr)
                .collect::<Result<_, _>>()?;
            Ok(DslExpr::Call {
                func: DslCallTarget::Builtin(builtin),
                args,
            })
        }

        // User-defined function call
        _ => {
            let args: Vec<DslExpr> = call
                .arguments
                .args
                .iter()
                .map(convert_expr)
                .collect::<Result<_, _>>()?;
            Ok(DslExpr::Call {
                func: DslCallTarget::UserDefined(func_name.to_owned()),
                args,
            })
        }
    }
}

/// Convert a ruff StmtFunctionDef into a DslFnDef.
fn convert_fndef(func: &ruff_python_ast::StmtFunctionDef) -> Result<DslFnDef, String> {
    let name = func.name.to_string();

    let params: Vec<DslParam> = func
        .parameters
        .args
        .iter()
        .map(|p| {
            let param_name = p.parameter.name.to_string();
            let ty = p
                .parameter
                .annotation
                .as_ref()
                .map(|a| convert_type_annotation(a))
                .transpose()?
                .ok_or_else(|| {
                    format!(
                        "DSL parameter '{}' in function '{}' must have a type annotation",
                        param_name, name
                    )
                })?;
            let default = p.default.as_ref().map(|d| convert_default(d)).transpose()?;
            Ok(DslParam {
                name: param_name,
                ty,
                default,
            })
        })
        .collect::<Result<_, String>>()?;

    let return_type = func
        .returns
        .as_ref()
        .map(|r| convert_type_annotation(r))
        .transpose()?;

    let body = convert_body(&func.body)?;

    Ok(DslFnDef {
        name,
        params,
        return_type,
        body,
    })
}

// ============================================================================
// Type Inference
// ============================================================================
//
// Infers types for all expressions and variables in the DSL. The key function
// is `infer_expr`, which the type checker uses to resolve overloaded operations
// (e.g. list concatenation vs numeric addition).
//
// Narrowing handles:
// - isinstance(x, con) → narrows x to/from the constructor in then/else
// - x == None / x != None → narrows x to/from None in then/else

/// Type environment mapping variable names to their inferred types.
type TypeEnv = HashMap<String, DslType>;

/// Function return types, keyed by function name.
type FnRetTypes = HashMap<String, DslType>;

/// The type `int | symint`, representing a tensor dimension.
fn dim_type() -> DslType {
    DslType::Union(vec![DslType::Int, DslType::SymInt])
}

/// True if this type contains `symint` (possibly inside a union).
fn contains_symint(ty: &DslType) -> bool {
    match ty {
        DslType::SymInt => true,
        DslType::Union(types) => types.iter().any(contains_symint),
        _ => false,
    }
}

/// True if `ty` matches the given type constructor.
fn matches_constructor(ty: &DslType, con: DslTypeCon) -> bool {
    matches!(
        (ty, con),
        (DslType::Int, DslTypeCon::Int)
            | (DslType::SymInt, DslTypeCon::SymInt)
            | (DslType::Bool, DslTypeCon::Bool)
            | (DslType::Str, DslTypeCon::Str)
            | (DslType::Tensor, DslTypeCon::Tensor)
            | (DslType::List(_), DslTypeCon::List)
    )
}

/// Flatten a DslType into its atomic constituents (expanding unions).
fn flatten_type(ty: &DslType, out: &mut Vec<DslType>) {
    match ty {
        DslType::Union(types) => {
            for t in types {
                flatten_type(t, out);
            }
        }
        _ => {
            if !out.contains(ty) {
                out.push(ty.clone());
            }
        }
    }
}

/// Join (least upper bound) of two types.
fn join(a: &DslType, b: &DslType) -> DslType {
    if a == b {
        return a.clone();
    }
    let mut types = Vec::new();
    flatten_type(a, &mut types);
    flatten_type(b, &mut types);
    if types.len() == 1 {
        types.pop().unwrap()
    } else {
        DslType::Union(types)
    }
}

/// Extract the element type from a list type. Panics on non-list (IR bug).
fn element_type(ty: &DslType) -> DslType {
    match ty {
        DslType::List(inner) => *inner.clone(),
        _ => unreachable!("expected list type, got {}", ty),
    }
}

/// Narrow a type to only variants matching a constructor.
fn narrow_to(ty: &DslType, con: DslTypeCon) -> DslType {
    match ty {
        DslType::Union(types) => {
            let matching: Vec<_> = types
                .iter()
                .filter(|t| matches_constructor(t, con))
                .cloned()
                .collect();
            match matching.len() {
                1 => matching.into_iter().next().unwrap(),
                0 => unreachable!("isinstance narrowing: no variant of {} matches {}", ty, con),
                _ => DslType::Union(matching),
            }
        }
        _ if matches_constructor(ty, con) => ty.clone(),
        _ => unreachable!("isinstance narrowing: {} does not match {}", ty, con),
    }
}

/// Narrow a type to exclude variants matching a constructor.
fn narrow_away(ty: &DslType, con: DslTypeCon) -> DslType {
    match ty {
        DslType::Union(types) => {
            let remaining: Vec<_> = types
                .iter()
                .filter(|t| !matches_constructor(t, con))
                .cloned()
                .collect();
            match remaining.len() {
                1 => remaining.into_iter().next().unwrap(),
                0 => unreachable!("narrowed away all variants of {}", ty),
                _ => DslType::Union(remaining),
            }
        }
        _ => ty.clone(),
    }
}

/// Narrow a type to exclude None.
fn narrow_away_none(ty: &DslType) -> DslType {
    match ty {
        DslType::Union(types) => {
            let remaining: Vec<_> = types
                .iter()
                .filter(|t| !matches!(t, DslType::None))
                .cloned()
                .collect();
            match remaining.len() {
                1 => remaining.into_iter().next().unwrap(),
                0 => unreachable!("narrowed away all variants of {}", ty),
                _ => DslType::Union(remaining),
            }
        }
        _ => ty.clone(),
    }
}

/// Analyze a condition expression for type narrowing.
/// Returns (then_env, else_env) — the environments for the true and false branches.
fn narrow(cond: &DslExpr, env: &TypeEnv) -> (TypeEnv, TypeEnv) {
    match cond {
        // isinstance(x, con)
        DslExpr::IsInstance { expr, ty } => {
            if let DslExpr::Var(name) = expr.as_ref()
                && let Some(var_ty) = env.get(name)
            {
                let mut then_env = env.clone();
                let mut else_env = env.clone();
                then_env.insert(name.clone(), narrow_to(var_ty, *ty));
                else_env.insert(name.clone(), narrow_away(var_ty, *ty));
                return (then_env, else_env);
            }
            (env.clone(), env.clone())
        }
        // x == None
        DslExpr::BinOp {
            left,
            op: DslOp::Eq,
            right,
        } => {
            if let (DslExpr::Var(name), DslExpr::Const(DslConst::None)) =
                (left.as_ref(), right.as_ref())
                && let Some(var_ty) = env.get(name)
            {
                let mut then_env = env.clone();
                let mut else_env = env.clone();
                then_env.insert(name.clone(), DslType::None);
                else_env.insert(name.clone(), narrow_away_none(var_ty));
                return (then_env, else_env);
            }
            (env.clone(), env.clone())
        }
        // x != None
        DslExpr::BinOp {
            left,
            op: DslOp::NotEq,
            right,
        } => {
            if let (DslExpr::Var(name), DslExpr::Const(DslConst::None)) =
                (left.as_ref(), right.as_ref())
                && let Some(var_ty) = env.get(name)
            {
                let mut then_env = env.clone();
                let mut else_env = env.clone();
                then_env.insert(name.clone(), narrow_away_none(var_ty));
                else_env.insert(name.clone(), DslType::None);
                return (then_env, else_env);
            }
            (env.clone(), env.clone())
        }
        // not cond — swap then/else
        DslExpr::UnaryOp {
            op: DslUnaryOp::Not,
            operand,
        } => {
            let (then_env, else_env) = narrow(operand, env);
            (else_env, then_env)
        }
        // cond1 and cond2 — narrow both in then-branch, conservative in else
        DslExpr::BinOp {
            left,
            op: DslOp::And,
            right,
        } => {
            let (then1, _) = narrow(left, env);
            let (then2, _) = narrow(right, &then1);
            (then2, env.clone())
        }
        _ => (env.clone(), env.clone()),
    }
}

/// Build function return type map from DSL definitions.
fn build_fn_ret_types(fndefs: &[DslFnDef]) -> FnRetTypes {
    fndefs
        .iter()
        .map(|f| {
            let return_type = f
                .return_type
                .clone()
                .unwrap_or_else(|| unreachable!("DSL function {} must have a return type", f.name));
            (f.name.clone(), return_type)
        })
        .collect()
}

/// Result type of numeric arithmetic. If either operand contains symint,
/// the result is `int | symint` (will generate DimAdd etc.); otherwise
/// the result is `int` (will generate IntAdd etc.).
fn arithmetic_result(a: &DslType, b: &DslType) -> DslType {
    if contains_symint(a) || contains_symint(b) {
        dim_type()
    } else {
        DslType::Int
    }
}

/// Infer the element type of a list literal from its elements.
fn infer_list_elem_type(elts: &[DslExpr], env: &TypeEnv, sigs: &FnRetTypes) -> DslType {
    assert!(
        !elts.is_empty(),
        "infer_list_elem_type called with empty list"
    );
    let mut result = infer_expr(&elts[0], env, sigs);
    for elt in &elts[1..] {
        result = join(&result, &infer_expr(elt, env, sigs));
    }
    result
}

/// Bind comprehension variables based on the iterator expression.
/// Handles zip (multi-list iteration) and enumerate (index + element).
fn bind_comp_vars(vars: &[String], iter: &DslExpr, env: &TypeEnv, sigs: &FnRetTypes) -> TypeEnv {
    let mut new_env = env.clone();
    match iter {
        DslExpr::Call {
            func: DslCallTarget::Builtin(DslBuiltin::Zip),
            args,
        } => {
            assert_eq!(
                vars.len(),
                args.len(),
                "zip: {} vars but {} args",
                vars.len(),
                args.len()
            );
            for (var, arg) in vars.iter().zip(args.iter()) {
                let arg_ty = infer_expr(arg, env, sigs);
                new_env.insert(var.clone(), element_type(&arg_ty));
            }
        }
        DslExpr::Call {
            func: DslCallTarget::Builtin(DslBuiltin::Enumerate),
            args,
        } => {
            assert_eq!(args.len(), 1, "enumerate takes exactly 1 argument");
            assert_eq!(vars.len(), 2, "enumerate requires exactly 2 variables");
            let list_ty = infer_expr(&args[0], env, sigs);
            new_env.insert(vars[0].clone(), DslType::Int);
            new_env.insert(vars[1].clone(), element_type(&list_ty));
        }
        _ => {
            let iter_ty = infer_expr(iter, env, sigs);
            if vars.len() == 1 {
                new_env.insert(vars[0].clone(), element_type(&iter_ty));
            } else {
                // Multiple vars iterating over a single list — each gets element type.
                let elem = element_type(&iter_ty);
                for var in vars {
                    new_env.insert(var.clone(), elem.clone());
                }
            }
        }
    }
    new_env
}

/// Infer the return type of a function call.
fn infer_call(func: &DslCallTarget, args: &[DslExpr], env: &TypeEnv, sigs: &FnRetTypes) -> DslType {
    // Infer all arguments for logging, regardless of whether we need them.
    for arg in args {
        infer_expr(arg, env, sigs);
    }
    match func {
        DslCallTarget::Builtin(builtin) => match builtin {
            // prod/sum reduce a list of dims to a single dim.
            DslBuiltin::Prod | DslBuiltin::Sum => {
                let arg_ty = infer_expr(&args[0], env, sigs);
                element_type(&arg_ty)
            }
            DslBuiltin::Str => DslType::Str,
            DslBuiltin::ParseEinsumEquation => DslType::List(Box::new(DslType::List(Box::new(
                DslType::List(Box::new(DslType::Int)),
            )))),
            DslBuiltin::Len => DslType::Int,
            DslBuiltin::Range => DslType::List(Box::new(DslType::Int)),
            DslBuiltin::Zip | DslBuiltin::Enumerate => {
                unreachable!("{} should only appear as comprehension iterator", builtin)
            }
        },
        DslCallTarget::UserDefined(name) => sigs
            .get(name)
            .unwrap_or_else(|| unreachable!("undefined function: {}", name))
            .clone(),
    }
}

/// Infer the type of a DSL expression.
fn infer_expr(expr: &DslExpr, env: &TypeEnv, sigs: &FnRetTypes) -> DslType {
    match expr {
        DslExpr::Const(c) => match c {
            DslConst::None => DslType::None,
            DslConst::Int(_) => DslType::Int,
            DslConst::Bool(_) => DslType::Bool,
            DslConst::Str(_) => DslType::Str,
        },
        DslExpr::Var(name) => env
            .get(name)
            .cloned()
            .unwrap_or_else(|| unreachable!("undefined variable: {}", name)),
        DslExpr::List(elts) => {
            if elts.is_empty() {
                // All empty list literals in the DSL are dimension lists.
                DslType::List(Box::new(dim_type()))
            } else if matches!(elts.last(), Some(DslExpr::Ellipsis)) {
                // [expr, ...] — unbounded list sentinel.
                let elem_ty = infer_list_elem_type(&elts[..elts.len() - 1], env, sigs);
                DslType::List(Box::new(elem_ty))
            } else {
                let elem_ty = infer_list_elem_type(elts, env, sigs);
                DslType::List(Box::new(elem_ty))
            }
        }
        DslExpr::ListComp {
            elt, vars, iter, ..
        } => {
            let comp_env = bind_comp_vars(vars, iter, env, sigs);
            let elt_ty = infer_expr(elt, &comp_env, sigs);
            DslType::List(Box::new(elt_ty))
        }
        DslExpr::Index { base, index } => {
            let base_ty = infer_expr(base, env, sigs);
            infer_expr(index, env, sigs);
            element_type(&base_ty)
        }
        DslExpr::Slice { base, lower, upper } => {
            let base_ty = infer_expr(base, env, sigs);
            if let Some(l) = lower {
                infer_expr(l, env, sigs);
            }
            if let Some(u) = upper {
                infer_expr(u, env, sigs);
            }
            base_ty
        }
        DslExpr::BinOp { left, op, right } => {
            let lt = infer_expr(left, env, sigs);
            let rt = infer_expr(right, env, sigs);
            match op {
                DslOp::Add => {
                    // List concatenation, string concatenation, or numeric addition.
                    if let DslType::List(a) = &lt {
                        let DslType::List(b) = &rt else {
                            unreachable!("+ with list and non-list: {} + {}", lt, rt)
                        };
                        DslType::List(Box::new(join(a, b)))
                    } else if matches!(lt, DslType::Str) {
                        assert!(
                            matches!(rt, DslType::Str),
                            "+ with str and non-str: {} + {}",
                            lt,
                            rt
                        );
                        DslType::Str
                    } else {
                        arithmetic_result(&lt, &rt)
                    }
                }
                DslOp::Sub | DslOp::Mul | DslOp::FloorDiv | DslOp::Mod => {
                    arithmetic_result(&lt, &rt)
                }
                DslOp::Eq
                | DslOp::NotEq
                | DslOp::Lt
                | DslOp::LtE
                | DslOp::Gt
                | DslOp::GtE
                | DslOp::And
                | DslOp::Or => DslType::Bool,
            }
        }
        DslExpr::UnaryOp { op, operand } => match op {
            DslUnaryOp::Not => {
                infer_expr(operand, env, sigs);
                DslType::Bool
            }
            DslUnaryOp::Neg => infer_expr(operand, env, sigs),
        },
        DslExpr::Call { func, args } => infer_call(func, args, env, sigs),
        DslExpr::IsInstance { expr, .. } => {
            infer_expr(expr, env, sigs);
            DslType::Bool
        }
        DslExpr::In { left, right } => {
            infer_expr(left, env, sigs);
            infer_expr(right, env, sigs);
            DslType::Bool
        }
        DslExpr::Shape(inner) => {
            infer_expr(inner, env, sigs);
            DslType::List(Box::new(dim_type()))
        }
        DslExpr::TensorNew(inner) => {
            infer_expr(inner, env, sigs);
            DslType::Tensor
        }
        DslExpr::IfExpr { body, test, orelse } => {
            let (then_env, else_env) = narrow(test, env);
            let body_ty = infer_expr(body, &then_env, sigs);
            let else_ty = infer_expr(orelse, &else_env, sigs);
            join(&body_ty, &else_ty)
        }
        DslExpr::Ellipsis => unreachable!("Ellipsis should be handled by List"),
        DslExpr::Unknown => DslType::None, // sentinel for fixture fallback
    }
}

/// Type-check a function body, updating the environment through assignments
/// and narrowing through conditionals.
fn check_body(body: &DslBody, env: &TypeEnv, sigs: &FnRetTypes) {
    match body {
        DslBody::Assign { vars, expr, rest } => {
            let ty = infer_expr(expr, env, sigs);
            let mut new_env = env.clone();
            if vars.len() == 1 {
                new_env.insert(vars[0].clone(), ty);
            } else {
                let elem = element_type(&ty);
                for var in vars {
                    new_env.insert(var.clone(), elem.clone());
                }
            }
            check_body(rest, &new_env, sigs);
        }
        DslBody::If {
            cond,
            then_body,
            rest,
        } => {
            let (then_env, else_env) = narrow(cond, env);
            check_body(then_body, &then_env, sigs);
            check_body(rest, &else_env, sigs);
        }
        DslBody::Return(expr) => {
            infer_expr(expr, env, sigs);
        }
        DslBody::Raise(expr) => {
            infer_expr(expr, env, sigs);
        }
    }
}

/// Type-check all DSL function definitions.
fn type_check_program(fndefs: &[DslFnDef]) {
    let sigs = build_fn_ret_types(fndefs);
    for fndef in fndefs {
        let mut env = TypeEnv::new();
        for param in &fndef.params {
            env.insert(param.name.clone(), param.ty.clone());
        }
        check_body(&fndef.body, &env, &sigs);
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Parse DSL source code, convert to grammar-aligned types, and return the
/// list of function definitions.
pub(crate) fn parse_dsl(source: &str) -> Result<Vec<DslFnDef>, String> {
    let (module, errors, _unsupported) = Ast::parse(source, PySourceType::Python);
    if !errors.is_empty() {
        return Err(format!(
            "DSL syntax errors:\n{}",
            errors
                .iter()
                .map(|e| format!("  {}", e))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let fndefs: Vec<DslFnDef> = module
        .body
        .iter()
        .filter_map(|stmt| {
            if let Stmt::FunctionDef(f) = stmt {
                Some(convert_fndef(f))
            } else {
                None // skip comments, blank lines (not in AST anyway)
            }
        })
        .collect::<Result<_, _>>()?;

    type_check_program(&fndefs);

    Ok(fndefs)
}

// ============================================================================
// Interpreter — evaluate DSL directly against runtime Val values
// ============================================================================

/// Extract a runtime `Val` from a type-checker `Type` based on the declared `DslType`.
/// `actual_arg_type` is the type the user passed; `expected_param_type` is the DSL
/// function's parameter annotation. Returns `None` if the type doesn't match
/// (causes fallback to fixture).
fn extract_dsl_val(actual_arg_type: &Type, expected_param_type: &DslType) -> Option<Val> {
    match expected_param_type {
        DslType::Int => Some(Val::Int(extract::literal_int(actual_arg_type)?)),
        DslType::SymInt => {
            let d = extract::dimension(actual_arg_type)?;
            Some(dim_val(d))
        }
        DslType::Bool => Some(Val::Bool(extract::bool_arg(actual_arg_type)?)),
        DslType::Str => Some(Val::Str(extract::string_arg(actual_arg_type)?)),
        DslType::Tensor => Some(Val::Shape(extract::concrete_tensor_shape(actual_arg_type)?)),
        DslType::None => actual_arg_type.is_none().then_some(Val::None),
        DslType::List(inner) => match inner.as_ref() {
            DslType::Int => Some(Val::List(
                extract::int_list(actual_arg_type)?
                    .iter()
                    .map(|&i| Val::Int(i))
                    .collect(),
            )),
            DslType::Tensor => Some(Val::List(
                extract::tensor_list(actual_arg_type)?
                    .into_iter()
                    .map(Val::Shape)
                    .collect(),
            )),
            DslType::SymInt => Some(Val::List(
                extract::dim_list(actual_arg_type)?
                    .into_iter()
                    .map(dim_val)
                    .collect(),
            )),
            // list[int | symint] — try dim_list (covers both concrete and symbolic).
            DslType::Union(variants)
                if variants
                    .iter()
                    .all(|v| matches!(v, DslType::Int | DslType::SymInt)) =>
            {
                // dim_list handles Tuple(Concrete([Literal, Size, ...])) for
                // both literal ints and symbolic dims.  Convert concrete
                // Size(Literal(n)) → Val::Int(n) so == comparisons against
                // literal ints work naturally in the interpreter.
                let dims = extract::dim_list(actual_arg_type)?;
                Some(Val::List(dims.into_iter().map(dim_val).collect()))
            }
            _ => None,
        },
        DslType::Union(variants) => {
            for v in variants {
                if let Some(val) = extract_dsl_val(actual_arg_type, v) {
                    return Some(val);
                }
            }
            None
        }
        DslType::Tuple(_) => None, // Tuples are return-type-only, not params
    }
}

/// Convert a `DslConst` default value to a `Val`.
fn const_to_val(c: &DslConst) -> Val {
    match c {
        DslConst::None => Val::None,
        DslConst::Int(n) => Val::Int(*n),
        DslConst::Bool(b) => Val::Bool(*b),
        DslConst::Str(s) => Val::Str(s.clone()),
    }
}

/// Bind DSL function parameters to runtime `Val` values.
/// Returns `None` if any required parameter can't be extracted (falls back to fixture).
fn bind_dsl_params(
    params: &[DslParam],
    bound_args: &HashMap<String, Type>,
) -> Option<HashMap<String, Val>> {
    let mut env = HashMap::new();
    for param in params {
        match bound_args.get(&param.name) {
            Some(ty) => {
                match extract_dsl_val(ty, &param.ty) {
                    Some(val) => {
                        env.insert(param.name.clone(), val);
                    }
                    None => {
                        // Type doesn't match the declared DSL type.
                        // If the param has a default, use it (e.g. `mean: Tensor | None`
                        // receiving a scalar float → treat as None).
                        // Otherwise the binding fails entirely.
                        let default = param.default.as_ref()?;
                        env.insert(param.name.clone(), const_to_val(default));
                    }
                }
            }
            None => {
                let default = param.default.as_ref()?;
                env.insert(param.name.clone(), const_to_val(default));
            }
        }
    }
    Some(env)
}

/// Evaluate a DSL expression against a runtime environment.
fn eval_dsl_expr(
    expr: &DslExpr,
    env: &HashMap<String, Val>,
    fns: &HashMap<String, Arc<DslFnDef>>,
    op_name: &str,
) -> Result<Val, ShapeError> {
    match expr {
        DslExpr::Const(c) => Ok(const_to_val(c)),

        DslExpr::Var(name) => Ok(env
            .get(name)
            .unwrap_or_else(|| panic!("DSL bug: {op_name}: undefined variable `{name}`"))
            .clone()),

        DslExpr::List(elems) => {
            // Skip Ellipsis elements — they are AST-level markers for
            // unbounded-tuple return types, not runtime values.
            let vals: Vec<Val> = elems
                .iter()
                .filter(|e| !matches!(e, DslExpr::Ellipsis))
                .map(|e| eval_dsl_expr(e, env, fns, op_name))
                .collect::<Result<_, _>>()?;
            Ok(Val::List(vals))
        }

        DslExpr::ListComp {
            elt,
            vars,
            iter,
            cond,
        } => {
            let iter_val = eval_dsl_expr(iter, env, fns, op_name)?;
            let items = iter_val.as_list();
            let mut result = Vec::new();
            let mut inner_env = env.clone();
            for item in items {
                if vars.len() == 1 {
                    inner_env.insert(vars[0].clone(), item.clone());
                } else {
                    // Tuple unpacking: item must be a List
                    let sub_items = item.as_list();
                    assert_eq!(
                        sub_items.len(),
                        vars.len(),
                        "DSL bug: {op_name}: tuple unpack length mismatch"
                    );
                    for (var, val) in vars.iter().zip(sub_items.iter()) {
                        inner_env.insert(var.clone(), val.clone());
                    }
                }
                if let Some(cond_expr) = cond
                    && !eval_dsl_expr(cond_expr, &inner_env, fns, op_name)?.as_bool()
                {
                    continue;
                }
                result.push(eval_dsl_expr(elt, &inner_env, fns, op_name)?);
            }
            Ok(Val::List(result))
        }

        DslExpr::Index { base, index } => {
            let base_val = eval_dsl_expr(base, env, fns, op_name)?;
            let items = base_val.as_list();
            let mut idx = eval_dsl_expr(index, env, fns, op_name)?.as_int();
            let len = items.len() as i64;
            if idx < 0 {
                idx += len;
            }
            if idx < 0 || idx >= len {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "{op_name}: index {idx} out of bounds for list of length {len}"
                    ),
                });
            }
            Ok(items[idx as usize].clone())
        }

        DslExpr::Slice { base, lower, upper } => {
            let base_val = eval_dsl_expr(base, env, fns, op_name)?;
            let items = base_val.as_list();
            let len = items.len() as i64;
            let lo = match lower {
                Some(e) => {
                    let v = eval_dsl_expr(e, env, fns, op_name)?.as_int();
                    if v < 0 { (v + len).max(0) } else { v.min(len) }
                }
                None => 0,
            };
            let hi = match upper {
                Some(e) => {
                    let v = eval_dsl_expr(e, env, fns, op_name)?.as_int();
                    if v < 0 { (v + len).max(0) } else { v.min(len) }
                }
                None => len,
            };
            let lo = lo as usize;
            let hi = hi as usize;
            if lo >= hi {
                Ok(Val::List(vec![]))
            } else {
                Ok(Val::List(items[lo..hi].to_vec()))
            }
        }

        DslExpr::BinOp { left, op, right } => {
            let lval = eval_dsl_expr(left, env, fns, op_name)?;
            // Short-circuit and/or (Python semantics).
            match op {
                DslOp::And => {
                    if !lval.as_bool() {
                        return Ok(Val::Bool(false));
                    }
                    let rval = eval_dsl_expr(right, env, fns, op_name)?;
                    return Ok(Val::Bool(rval.as_bool()));
                }
                DslOp::Or => {
                    if lval.as_bool() {
                        return Ok(Val::Bool(true));
                    }
                    let rval = eval_dsl_expr(right, env, fns, op_name)?;
                    return Ok(Val::Bool(rval.as_bool()));
                }
                _ => {}
            }
            let rval = eval_dsl_expr(right, env, fns, op_name)?;
            eval_binop(&lval, *op, &rval, op_name)
        }

        DslExpr::UnaryOp { op, operand } => {
            let val = eval_dsl_expr(operand, env, fns, op_name)?;
            match op {
                DslUnaryOp::Not => Ok(Val::Bool(!val.as_bool())),
                DslUnaryOp::Neg => {
                    match &val {
                        Val::Int(n) => Ok(Val::Int(-n)),
                        Val::Dim(ty) => {
                            // Negate symbolic: 0 - ty
                            let zero = Type::Size(SizeExpr::Literal(0));
                            Ok(Val::Dim(canonicalize(Type::Size(SizeExpr::sub(
                                zero,
                                ty.clone(),
                            )))))
                        }
                        _ => panic!("DSL bug: {op_name}: cannot negate {}", val.variant_name()),
                    }
                }
            }
        }

        DslExpr::Call { func, args } => {
            let arg_vals: Vec<Val> = args
                .iter()
                .map(|a| eval_dsl_expr(a, env, fns, op_name))
                .collect::<Result<_, _>>()?;
            eval_call(func, &arg_vals, fns, op_name)
        }

        DslExpr::IsInstance { expr, ty } => {
            let val = eval_dsl_expr(expr, env, fns, op_name)?;
            let matches = match (ty, &val) {
                (DslTypeCon::Int, Val::Int(_)) => true,
                (DslTypeCon::Str, Val::Str(_)) => true,
                (DslTypeCon::Bool, Val::Bool(_)) => true,
                (DslTypeCon::SymInt, Val::Dim(_)) => true,
                (DslTypeCon::List, Val::List(_)) => true,
                (DslTypeCon::Tensor, Val::Shape(_)) => true,
                _ => false,
            };
            Ok(Val::Bool(matches))
        }

        DslExpr::In { left, right } => {
            let needle = eval_dsl_expr(left, env, fns, op_name)?;
            let haystack = eval_dsl_expr(right, env, fns, op_name)?;
            let items = haystack.as_list();
            let found = items.iter().any(|item| val_eq(&needle, item));
            Ok(Val::Bool(found))
        }

        DslExpr::Shape(inner) => {
            let val = eval_dsl_expr(inner, env, fns, op_name)?;
            let shape = val.as_shape();
            // Use dim_val to convert concrete Size(Literal(n)) to Val::Int(n)
            // so comparisons against literal ints (e.g., `d != 1` in squeeze)
            // work naturally.
            let dims: Vec<Val> = shape.dims().iter().map(|d| dim_val(d.clone())).collect();
            Ok(Val::List(dims))
        }

        DslExpr::TensorNew(shape_expr) => {
            let val = eval_dsl_expr(shape_expr, env, fns, op_name)?;
            let dims = val.as_size_list();
            Ok(Val::Shape(TensorShape::from_types(dims)))
        }

        DslExpr::IfExpr { body, test, orelse } => {
            if eval_dsl_expr(test, env, fns, op_name)?.as_bool() {
                eval_dsl_expr(body, env, fns, op_name)
            } else {
                eval_dsl_expr(orelse, env, fns, op_name)
            }
        }

        DslExpr::Ellipsis => {
            // Ellipsis is only meaningful in return-position list literals;
            // it should never be evaluated as a standalone expression.
            unreachable!("DSL bug: {op_name}: Ellipsis should not be evaluated directly")
        }

        DslExpr::Unknown => Ok(Val::None), // sentinel for fixture fallback
    }
}

/// Normalize a canonical `Type` to `Val`. If it's a concrete `Size(Literal(n))`,
/// produce `Val::Int(n)` so equality checks against literal ints work naturally.
/// Otherwise produce `Val::Dim(ty)`.
fn dim_val(ty: Type) -> Val {
    match &ty {
        Type::Size(SizeExpr::Literal(n)) => Val::Int(*n),
        _ => Val::Dim(ty),
    }
}

/// Evaluate a binary operation, dispatching on runtime Val variants.
///
/// Arithmetic: both Int → concrete i64; either Dim → symbolic SizeExpr; + on Lists → concat.
/// Comparison: concrete on Int; == None → is_none(); on Str → string compare.
/// Note: And/Or are short-circuited in eval_dsl_expr and never reach here.
fn eval_binop(lval: &Val, op: DslOp, rval: &Val, op_name: &str) -> Result<Val, ShapeError> {
    match op {
        // --- Arithmetic ---
        DslOp::Add => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Int(a + b)),
            (Val::Str(a), Val::Str(b)) => Ok(Val::Str(format!("{}{}", a, b))),
            (Val::List(a), Val::List(b)) => {
                let mut result = a.clone();
                result.extend(b.iter().cloned());
                Ok(Val::List(result))
            }
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Size(SizeExpr::add(a, b)))))
            }
        },
        DslOp::Sub => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Int(a - b)),
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Size(SizeExpr::sub(a, b)))))
            }
        },
        DslOp::Mul => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Int(a * b)),
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Size(SizeExpr::mul(a, b)))))
            }
        },
        DslOp::FloorDiv => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => {
                if *b == 0 {
                    return Err(ShapeError::ShapeComputation {
                        message: format!("{op_name}: division by zero"),
                    });
                }
                Ok(Val::Int(a / b))
            }
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Size(SizeExpr::floor_div(a, b)))))
            }
        },
        DslOp::Mod => {
            let a = lval.as_int();
            let b = rval.as_int();
            if b == 0 {
                return Err(ShapeError::ShapeComputation {
                    message: format!("{op_name}: modulo by zero"),
                });
            }
            Ok(Val::Int(a % b))
        }

        // --- Comparison ---
        DslOp::Eq => Ok(Val::Bool(val_eq(lval, rval))),
        DslOp::NotEq => Ok(Val::Bool(!val_eq(lval, rval))),
        DslOp::Lt => Ok(Val::Bool(lval.as_int() < rval.as_int())),
        DslOp::LtE => Ok(Val::Bool(lval.as_int() <= rval.as_int())),
        DslOp::Gt => Ok(Val::Bool(lval.as_int() > rval.as_int())),
        DslOp::GtE => Ok(Val::Bool(lval.as_int() >= rval.as_int())),

        // And/Or are short-circuited in eval_dsl_expr, never reach here.
        DslOp::And | DslOp::Or => unreachable!("and/or are short-circuited in eval_dsl_expr"),
    }
}

/// Structural equality for Val. Used by `==`, `!=`, and `in`.
fn val_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::None, Val::None) => true,
        _ => false,
    }
}

/// Evaluate a function call — builtins handled inline, user-defined looked up in `fns`.
fn eval_call(
    func: &DslCallTarget,
    args: &[Val],
    fns: &HashMap<String, Arc<DslFnDef>>,
    op_name: &str,
) -> Result<Val, ShapeError> {
    match func {
        DslCallTarget::Builtin(builtin) => match builtin {
            DslBuiltin::Prod => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: prod takes 1 arg");
                let items = args[0].as_list();
                // Check if all items are Int — if so, concrete product.
                // If any is Dim, use symbolic product.
                let all_int = items.iter().all(|v| matches!(v, Val::Int(_)));
                if all_int {
                    let product: i64 = items.iter().map(|v| v.as_int()).product();
                    Ok(Val::Int(product))
                } else {
                    let dims = args[0].as_size_list();
                    let mut product = Type::Size(SizeExpr::Literal(1));
                    for d in dims {
                        product = canonicalize(Type::Size(SizeExpr::mul(product, d)));
                    }
                    Ok(dim_val(product))
                }
            }
            DslBuiltin::Sum => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: sum takes 1 arg");
                let items = args[0].as_list();
                let all_int = items.iter().all(|v| matches!(v, Val::Int(_)));
                if all_int {
                    let total: i64 = items.iter().map(|v| v.as_int()).sum();
                    Ok(Val::Int(total))
                } else {
                    let dims = args[0].as_size_list();
                    let mut total = Type::Size(SizeExpr::Literal(0));
                    for d in dims {
                        total = canonicalize(Type::Size(SizeExpr::add(total, d)));
                    }
                    Ok(dim_val(total))
                }
            }
            DslBuiltin::Str => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: str takes 1 arg");
                let s = match &args[0] {
                    Val::Int(n) => n.to_string(),
                    Val::Str(s) => s.clone(),
                    Val::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
                    Val::Dim(ty) => ty.to_string(),
                    Val::Shape(shape) => shape.to_string(),
                    Val::List(items) => format!(
                        "[{}]",
                        items
                            .iter()
                            .map(|v| match v {
                                Val::Int(n) => n.to_string(),
                                Val::Str(s) => format!("\"{}\"", s),
                                Val::Dim(ty) => ty.to_string(),
                                other => other.variant_name().to_owned(),
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Val::None => "None".to_owned(),
                };
                Ok(Val::Str(s))
            }
            DslBuiltin::ParseEinsumEquation => {
                assert_eq!(
                    args.len(),
                    1,
                    "DSL bug: {op_name}: einsum_parse takes 1 arg"
                );
                let spec = args[0].as_str_val();

                // Parse spec: "ij,jk->ik"
                let parts: Vec<&str> = spec.split("->").collect();
                if parts.len() != 2 {
                    return Err(ShapeError::ShapeComputation {
                        message: format!("einsum spec must contain '->', got: {}", spec),
                    });
                }
                let input_specs: Vec<Vec<char>> = parts[0]
                    .split(',')
                    .map(|s| s.trim().chars().filter(|c| c.is_alphanumeric()).collect())
                    .collect();
                let output_spec: Vec<char> = parts[1]
                    .trim()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();

                // Build char → (input_idx, dim_pos) map, collecting consistency
                // check pairs for labels that appear in multiple positions.
                let mut char_to_location: HashMap<char, (usize, usize)> = HashMap::new();
                let mut check_pairs: Vec<Val> = Vec::new();
                for (input_idx, spec_chars) in input_specs.iter().enumerate() {
                    for (pos, ch) in spec_chars.iter().enumerate() {
                        if let Some(&(prev_input, prev_pos)) = char_to_location.get(ch) {
                            // Each check pair is [idx1, pos1, idx2, pos2].
                            check_pairs.push(Val::List(vec![
                                Val::Int(prev_input as i64),
                                Val::Int(prev_pos as i64),
                                Val::Int(input_idx as i64),
                                Val::Int(pos as i64),
                            ]));
                        } else {
                            char_to_location.insert(*ch, (input_idx, pos));
                        }
                    }
                }

                // Build output_map: for each output char, resolve to [input_idx, dim_pos].
                let mut output_map: Vec<Val> = Vec::new();
                for ch in &output_spec {
                    let &(input_idx, dim_pos) =
                        char_to_location
                            .get(ch)
                            .ok_or_else(|| ShapeError::ShapeComputation {
                                message: format!(
                                    "einsum: output index '{}' not found in inputs",
                                    ch
                                ),
                            })?;
                    output_map.push(Val::List(vec![
                        Val::Int(input_idx as i64),
                        Val::Int(dim_pos as i64),
                    ]));
                }

                Ok(Val::List(vec![
                    Val::List(output_map),
                    Val::List(check_pairs),
                ]))
            }
            DslBuiltin::Enumerate => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: enumerate takes 1 arg");
                let items = args[0].as_list();
                Ok(Val::List(
                    items
                        .iter()
                        .enumerate()
                        .map(|(i, v)| Val::List(vec![Val::Int(i as i64), v.clone()]))
                        .collect(),
                ))
            }
            DslBuiltin::Zip => {
                assert!(args.len() >= 2, "DSL bug: {op_name}: zip takes 2+ args");
                let lists: Vec<Vec<Val>> = args.iter().map(|a| a.as_list().to_vec()).collect();
                let min_len = lists.iter().map(|l| l.len()).min().unwrap_or(0);
                Ok(Val::List(
                    (0..min_len)
                        .map(|i| Val::List(lists.iter().map(|l| l[i].clone()).collect()))
                        .collect(),
                ))
            }
            DslBuiltin::Len => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: len takes 1 arg");
                let items = args[0].as_list();
                Ok(Val::Int(items.len() as i64))
            }
            DslBuiltin::Range => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: range takes 1 arg");
                let n = args[0].as_int();
                Ok(Val::List((0..n).map(Val::Int).collect()))
            }
        },
        DslCallTarget::UserDefined(name) => {
            let fn_def = fns
                .get(name)
                .unwrap_or_else(|| panic!("DSL bug: {op_name}: undefined function `{name}`"));
            let mut call_env = HashMap::new();
            for (param, arg) in fn_def.params.iter().zip(args.iter()) {
                call_env.insert(param.name.clone(), arg.clone());
            }
            // Fill defaults for remaining params
            for param in fn_def.params.iter().skip(args.len()) {
                let default = param.default.as_ref().unwrap_or_else(|| {
                    panic!(
                        "DSL bug: {op_name}: missing arg `{}` for function `{name}`",
                        param.name
                    )
                });
                call_env.insert(param.name.clone(), const_to_val(default));
            }
            eval_dsl_body(&fn_def.body, &mut call_env, fns, op_name).map(|(val, _is_unbounded)| val)
        }
    }
}

/// Evaluate a DSL function body. Walks the linked-list structure:
/// Assign → bind vars, continue to rest.
/// If → eval cond, branch to then_body or continue to rest.
/// Return → eval expr, return value + whether it's an ellipsis list.
/// Raise → return ShapeError.
///
/// The second element of the returned tuple is `true` when the Return
/// expression is a list literal ending with `...` (the unbounded-tuple marker).
fn eval_dsl_body(
    body: &DslBody,
    env: &mut HashMap<String, Val>,
    fns: &HashMap<String, Arc<DslFnDef>>,
    op_name: &str,
) -> Result<(Val, bool), ShapeError> {
    match body {
        DslBody::Assign { vars, expr, rest } => {
            let val = eval_dsl_expr(expr, env, fns, op_name)?;
            if vars.len() == 1 {
                env.insert(vars[0].clone(), val);
            } else {
                // Tuple unpacking
                let items = val.as_list();
                assert_eq!(
                    items.len(),
                    vars.len(),
                    "DSL bug: {op_name}: unpack length mismatch"
                );
                for (var, item) in vars.iter().zip(items.iter()) {
                    env.insert(var.clone(), item.clone());
                }
            }
            eval_dsl_body(rest, env, fns, op_name)
        }
        DslBody::If {
            cond,
            then_body,
            rest,
        } => {
            if eval_dsl_expr(cond, env, fns, op_name)?.as_bool() {
                let mut then_env = env.clone();
                eval_dsl_body(then_body, &mut then_env, fns, op_name)
            } else {
                eval_dsl_body(rest, env, fns, op_name)
            }
        }
        DslBody::Return(expr) => {
            let has_ellipsis = matches!(expr, DslExpr::List(elems)
                if elems.len() >= 2 && matches!(elems.last(), Some(DslExpr::Ellipsis)));
            Ok((eval_dsl_expr(expr, env, fns, op_name)?, has_ellipsis))
        }
        DslBody::Raise(expr) => {
            let msg = eval_dsl_expr(expr, env, fns, op_name)?;
            Err(ShapeError::ShapeComputation {
                message: msg.as_str_val().to_owned(),
            })
        }
    }
}

/// Inject a computed `TensorShape` into a fixture `Type`, preserving the base class.
///
/// Handles both `Type::Tensor(t)` and `Type::ClassType("torch.Tensor")` fixture types.
/// Falls back to `ret_type` unchanged if it isn't a tensor type.
fn inject_shape(shape: TensorShape, ret_type: &Type) -> Type {
    match ret_type {
        Type::Tensor(t) => TensorType::new(t.base_class.clone(), shape).to_type(),
        Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
            TensorType::new(cls.clone(), shape).to_type()
        }
        _ => ret_type.clone(),
    }
}

/// Convert a DSL `Val` to a user-facing `Type` for output from a shape function.
/// Unlike `Val::as_size` (which produces `Type::Size` for internal arithmetic),
/// this produces the types that appear in type-checker diagnostics:
/// - `Val::Int(n)` → `Literal[n]`
/// - `Val::Dim(SizeExpr::Literal(n))` → `Literal[n]` (concrete dims become literals)
/// - `Val::Dim(symbolic)` → `Dim[symbolic]`
fn val_to_scalar_type(val: &Val) -> Type {
    match val {
        Val::Int(n) => Lit::Int(LitInt::new(*n)).to_implicit_type(),
        Val::Dim(ty) => {
            if let Some(n) = ty.as_shape_literal() {
                Lit::Int(LitInt::new(n)).to_implicit_type()
            } else {
                Type::Dim(Box::new(ty.clone()))
            }
        }
        _ => unreachable!(
            "DSL bug: expected Int or Dim for scalar type, got {}",
            val.variant_name()
        ),
    }
}

/// Inject a list of computed shapes into the fixture return type's tuple structure.
/// Returns `None` if shapes is empty or the fixture type doesn't match.
fn inject_shapes_into_tuple(shapes: Vec<TensorShape>, expected_return_type: &Type) -> Option<Type> {
    if shapes.is_empty() {
        return None;
    }
    match expected_return_type {
        Type::Tuple(Tuple::Concrete(elems)) if elems.len() == shapes.len() => {
            Some(Type::concrete_tuple(
                elems
                    .iter()
                    .zip(shapes)
                    .map(|(elem, shape)| inject_shape(shape, elem))
                    .collect(),
            ))
        }
        Type::Tuple(Tuple::Unbounded(elem)) => Some(Type::concrete_tuple(
            shapes
                .into_iter()
                .map(|shape| inject_shape(shape, elem))
                .collect(),
        )),
        _ => None,
    }
}

/// Convert a return `Val` directly to a `Type` using the DSL return type annotation
/// and the fixture return type.
///
/// This is symmetric with input binding: on the way in, `(Type, DslType) → Val`;
/// on the way out, `(Val, DslType, Type) → Type`.
///
/// `is_unbounded` is true when the return expression was `[..., ...]` (ellipsis list).
fn val_to_type(
    val: Val,
    is_unbounded: bool,
    actual_result_type: &DslType,
    expected_return_type: &Type,
    op_name: &str,
) -> Type {
    match actual_result_type {
        DslType::Tensor => match val {
            Val::Shape(s) => inject_shape(s, expected_return_type),
            // Some ops conditionally return a tuple of tensors despite
            // declaring `-> Tensor` (e.g. min_max_median_ir returns
            // [Tensor, Tensor] when dim is given).
            Val::List(items) => {
                let shapes: Vec<TensorShape> = items.iter().map(|v| v.as_shape().clone()).collect();
                if shapes.len() == 1 {
                    return inject_shape(shapes.into_iter().next().unwrap(), expected_return_type);
                }
                inject_shapes_into_tuple(shapes, expected_return_type)
                    .unwrap_or_else(|| expected_return_type.clone())
            }
            _ => panic!(
                "DSL bug: {op_name}: expected Shape for Tensor return, got {}",
                val.variant_name()
            ),
        },

        DslType::Int => match val {
            Val::Int(n) => Lit::Int(LitInt::new(n)).to_implicit_type(),
            _ => panic!(
                "DSL bug: {op_name}: expected Int for int return, got {}",
                val.variant_name()
            ),
        },

        DslType::SymInt => val_to_scalar_type(&val),

        DslType::Bool => match val {
            Val::Bool(b) => Lit::Bool(b).to_implicit_type(),
            _ => panic!(
                "DSL bug: {op_name}: expected Bool for bool return, got {}",
                val.variant_name()
            ),
        },

        DslType::Union(variants) => {
            // List of int/symint → per-element normalization (e.g. size() with no dim arg).
            if let Val::List(items) = &val
                && variants
                    .iter()
                    .all(|v| matches!(v, DslType::Int | DslType::SymInt))
            {
                return Type::concrete_tuple(items.iter().map(val_to_scalar_type).collect());
            }
            // Try to match the val against each variant.
            // For int | symint, Int or Dim are both valid.
            for v in variants {
                match (v, &val) {
                    (DslType::Int, Val::Int(_))
                    | (DslType::SymInt, Val::Int(_))
                    | (DslType::SymInt, Val::Dim(_))
                    | (DslType::Tensor, Val::Shape(_))
                    | (DslType::Bool, Val::Bool(_)) => {
                        return val_to_type(val, is_unbounded, v, expected_return_type, op_name);
                    }
                    _ => continue,
                }
            }
            panic!(
                "DSL bug: {op_name}: no union variant matched return val {}",
                val.variant_name()
            );
        }

        DslType::List(inner) => match inner.as_ref() {
            DslType::Tensor => {
                let items = val.as_list();
                let shapes: Vec<TensorShape> = items.iter().map(|v| v.as_shape().clone()).collect();
                if is_unbounded {
                    // Unbounded: build Tuple::Unbounded with computed element shape
                    if let (Some(first), Type::Tuple(Tuple::Unbounded(elem))) =
                        (shapes.first(), expected_return_type)
                    {
                        Type::Tuple(Tuple::Unbounded(Box::new(inject_shape(
                            first.clone(),
                            elem,
                        ))))
                    } else {
                        expected_return_type.clone()
                    }
                } else {
                    inject_shapes_into_tuple(shapes, expected_return_type)
                        .unwrap_or_else(|| expected_return_type.clone())
                }
            }
            _ => expected_return_type.clone(),
        },

        DslType::Tuple(elems) => {
            let items = val.as_list();
            let all_tensor = elems.iter().all(|e| matches!(e, DslType::Tensor));
            if all_tensor {
                let shapes: Vec<TensorShape> = items.iter().map(|v| v.as_shape().clone()).collect();
                inject_shapes_into_tuple(shapes, expected_return_type)
                    .unwrap_or_else(|| expected_return_type.clone())
            } else {
                // All ints/symints → per-element normalization
                let all_int_like = elems
                    .iter()
                    .all(|e| matches!(e, DslType::Int | DslType::SymInt));
                if all_int_like {
                    Type::concrete_tuple(items.iter().map(val_to_scalar_type).collect())
                } else {
                    expected_return_type.clone()
                }
            }
        }

        DslType::None | DslType::Str => expected_return_type.clone(),
    }
}

/// A `MetaShapeFunction` backed by a parsed DSL function definition.
/// The DSL is interpreted directly — no IR conversion.
#[derive(Debug)]
pub(crate) struct DslMetaShapeFunction {
    /// The primary function to evaluate.
    pub(crate) fn_def: Arc<DslFnDef>,
    /// Precomputed lookup table mapping function names to definitions.
    /// Shared across all instances — built once at registry init.
    pub(crate) fn_lookup: Arc<HashMap<String, Arc<DslFnDef>>>,
}

impl MetaShapeFunction for DslMetaShapeFunction {
    fn name(&self) -> &str {
        &self.fn_def.name
    }

    fn evaluate(
        &self,
        bound_args: &HashMap<String, Type>,
        ret_type: &Type,
    ) -> Option<Result<Type, ShapeError>> {
        let mut env = bind_dsl_params(&self.fn_def.params, bound_args)?;

        let result = eval_dsl_body(
            &self.fn_def.body,
            &mut env,
            &self.fn_lookup,
            &self.fn_def.name,
        );

        match result {
            Ok((val, is_unbounded)) => {
                // `Unknown` in the DSL evaluates to Val::None.
                // This signals a fixture-type fallback.
                if matches!(val, Val::None) {
                    return Some(Ok(ret_type.clone()));
                }

                let actual_result_type = self.fn_def.return_type.as_ref().unwrap_or_else(|| {
                    panic!(
                        "DSL bug: {}: function has no return type annotation",
                        self.fn_def.name
                    )
                });
                Some(Ok(val_to_type(
                    val,
                    is_unbounded,
                    actual_result_type,
                    ret_type,
                    &self.fn_def.name,
                )))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
