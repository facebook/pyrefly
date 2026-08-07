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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::BoolOp as RuffBoolOp;
use ruff_python_ast::CmpOp as RuffCmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator as RuffOperator;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp as RuffUnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::callable::IdentityIgnored;
use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::dimension::canonicalize;
use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::lit_int::LitInt;
use crate::literal::Lit;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::tuple::Tuple;
use crate::types::Type;

// Section: Runtime Values

/// Runtime value produced by parameter extraction and manipulated by the
/// interpreter. Bridges between `Type` (the type-checker's representation)
/// and the shape computation domain.
#[derive(Debug, Clone)]
enum Val {
    /// Concrete integer (e.g., dim=0, stride=1).
    Int(i64),
    /// Boolean flag (e.g., keepdim=False).
    Bool(bool),
    /// String literal (e.g., einsum spec).
    Str(String),
    /// Single tensor dimension: a symbolic `Type` (`Int`, `Quantified`, etc.).
    Dim(Type),
    /// Full tensor shape with concrete rank.
    Shape(IntTuple),
    /// Homogeneous list. Elements are all the same variant (Int, Dim, Shape, …).
    List(Vec<Val>),
    /// Variadic shape: prefix dims + variadic middle + suffix dims.
    /// Used when the tensor shape is `Unpacked` (e.g., `Tensor[*Bs, Features]`).
    /// Slicing and concatenation work on this variant; operations that require
    /// a concrete length (like `len()` or `enumerate()`) produce a soft error.
    Unpacked {
        prefix: Vec<Val>,
        middle: Type,
        suffix: Vec<Val>,
    },
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

    /// Convert to a `Type::Int` for use in dimension arithmetic within the DSL evaluator.
    /// `Int(n)` becomes `Type::Int(Int::Literal(n))`; `Dim(ty)` passes through as-is.
    /// This is for *internal* symbolic computation, not for producing user-facing types
    /// (see `val_to_scalar_type` for that).
    pub fn as_size(&self) -> Type {
        match self {
            Val::Dim(ty) => ty.clone(),
            Val::Int(n) => Type::Int(Int::Literal(*n)),
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

    /// Extract as `&IntTuple`. Panics if not `Shape` — the DSL type checker
    /// guarantees this won't happen for well-typed DSL code.
    pub fn as_shape(&self) -> &IntTuple {
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

    /// Extract a list of `Type::Int` values from a `Val::List`, for use in shape arithmetic.
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
            Val::Unpacked { .. } => "Unpacked",
            Val::None => "None",
        }
    }
}

// Section: Extraction Helpers

/// Helper functions for extracting typed values from `Type`.
///
/// These are used in `bind_dsl_params()` to convert bound Python types
/// to runtime values. Each returns `None` if the type doesn't match.
mod extract {
    use crate::dimension::Int;
    use crate::literal::Lit;
    use crate::shaped_array::IntTuple;
    use crate::tuple::Tuple;
    use crate::types::Type;

    /// Extract a IntTuple from a Type.
    /// Returns None for non-shaped-arrays and shapeless arrays.
    /// Allows both Concrete and Unpacked shapes through so DSL ops that
    /// support variadic shapes (e.g., slicing) can operate on them.
    pub fn shaped_array_shape(ty: &Type) -> Option<IntTuple> {
        match ty {
            Type::ShapedArray(shaped_array) => {
                if shaped_array.is_shapeless() {
                    None
                } else {
                    Some(shaped_array.shape().clone())
                }
            }
            Type::Union(union) => {
                let mut shapes = union.members.iter().map(shaped_array_shape);
                let first = shapes.next()??;
                if shapes.all(|shape| shape.as_ref() == Some(&first)) {
                    Some(first)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract literal int from Type::Literal(Lit::Int(...)).
    pub fn literal_int(ty: &Type) -> Option<i64> {
        match ty {
            Type::Literal(lit) if let Lit::Int(n) = &lit.value => n.as_i64(),
            _ => None,
        }
    }

    /// Extract symbolic dimension from Type.
    /// Handles `Int`, `Quantified`, `Var`, etc.
    pub fn dimension(ty: &Type) -> Option<Type> {
        match ty {
            // Already a symbolic integer expression.
            Type::Int(_) => Some(ty.clone()),
            // Type variable or quantified
            Type::Quantified(_) | Type::Var(_) => Some(ty.clone()),
            // Literal int -> wrap in `Int`.
            Type::Literal(lit) if let Lit::Int(n) = &lit.value => {
                n.as_i64().map(|v| Type::Int(Int::Literal(v)))
            }
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
            Type::IntTuple(shape) => dim_list(&shape.to_tuple_type()),
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
            Type::Literal(lit) if let Lit::Bool(b) = &lit.value => Some(*b),
            _ => None,
        }
    }

    /// Extract string literal from Type::Literal(Lit::Str(...)).
    pub fn string_arg(ty: &Type) -> Option<String> {
        match ty {
            Type::Literal(lit) if let Lit::Str(s) = &lit.value => Some(s.to_string()),
            _ => None,
        }
    }

    /// Extract list or tuple of shaped-array shapes.
    /// Handles tuple[Array[...], ...].
    /// Returns None for list types (can't determine element count) or unbounded tuples.
    pub fn shaped_array_list(ty: &Type) -> Option<Vec<IntTuple>> {
        use crate::tuple::Tuple;

        match ty {
            // list[Array[...]] - can't determine element count, return None
            Type::ClassType(class_type) if class_type.has_qname("builtins", "list") => {
                // Lists don't preserve element count in the type system
                // Fall back to fixture for now
                None
            }
            // tuple[Array[...], ...] - unbounded, can't determine count
            Type::Tuple(Tuple::Unbounded(_)) => None,
            // tuple[Array[...], Array[...], ...] - concrete, extract all
            Type::Tuple(Tuple::Concrete(elems)) => {
                if matches!(elems.first(), Some(Type::ShapedArray(_))) {
                    return elems.iter().map(shaped_array_shape).collect();
                }
                None
            }
            _ => None,
        }
    }
}

// Section: Meta-Shape Function Trait

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

    /// Return the names of all parameters in this DSL function.
    ///
    /// Used by the caller to auto-inject module field values: if a DSL parameter
    /// name is not in `bound_args` but matches a field on `self`, the caller
    /// resolves `self.<param_name>` and injects it before evaluation.
    fn param_names(&self) -> Vec<&str> {
        vec![]
    }
}

// Section: Compile error type

/// A structured error from DSL compilation (parsing or type-checking), carrying
/// the source range of the problematic construct so callers can emit precise
/// diagnostics without resorting to a function-wide fallback range.
#[derive(Debug, Clone)]
pub struct DslCompileError {
    /// Source span of the construct that caused the error.
    pub range: TextRange,
    /// Human-readable description of the error.
    pub message: String,
}

// Section: Grammar-aligned data types

/// Binary operators: arithmetic, comparison, and logical.
/// Corresponds to OP in `<expr> OP <expr>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DslOp {
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
enum DslUnaryOp {
    Not,
    Neg,
}

/// Well-known builtin functions, validated at parse time.
/// A typo in the DSL source (e.g., `prodd`) will be caught immediately
/// as an undefined user-defined function rather than silently falling through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DslBuiltin {
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
enum DslCallTarget {
    Builtin(DslBuiltin),
    UserDefined(String),
}

/// Type constructors for `isinstance` checks.
/// These are nullary: `isinstance(x, list)` checks the constructor, not the element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DslTypeCon {
    Int,
    Str,
    Bool,
    SymInt,
    List,
    ShapedArray,
}

/// Types in the DSL. Corresponds to `<type>` in the grammar,
/// extended with Tuple for return type annotations.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DslType {
    Int,
    SymInt,
    Bool,
    Str,
    ShapedArray,
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
enum DslConst {
    None,
    Int(i64),
    Bool(bool),
    Str(String),
}

/// Function parameter. Corresponds to `<param>` in the grammar.
#[derive(Debug, Clone)]
struct DslParam {
    name: String,
    ty: DslType,
    default: Option<DslConst>,
}

/// Function body. Corresponds to `<body>` in the grammar.
/// This is a recursive (linked-list) structure where Assign and If
/// have a `rest` continuation, and Return/Raise are terminals.
#[derive(Debug, Clone)]
enum DslBody {
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
enum DslExpr {
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
    /// `ShapedArray(shape=expr)` (construct result shaped array).
    ShapedArrayNew(Box<DslExpr>),
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
    name: String,
    /// Source range of the function name identifier; used to attach type-check
    /// errors to a precise location even when working from DSL IR.
    name_range: TextRange,
    params: Vec<DslParam>,
    return_type: Option<DslType>,
    body: DslBody,
}

// Section: Display implementations

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
            DslBuiltin::Prod => write!(f, "shape_extensions.dsl.prod"),
            DslBuiltin::Sum => write!(f, "shape_extensions.dsl.sum"),
            DslBuiltin::Str => write!(f, "str"),
            DslBuiltin::ParseEinsumEquation => {
                write!(f, "shape_extensions.dsl.parse_einsum_equation")
            }
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
            DslTypeCon::ShapedArray => write!(f, "ShapedArray"),
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
            DslType::ShapedArray => write!(f, "ShapedArray"),
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
            DslExpr::ShapedArrayNew(expr) => write!(f, "ShapedArray(shape={})", expr),
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

// Section: AST conversion: ruff Python AST → DSL grammar types

/// Convert an isinstance type argument to a DslTypeCon.
fn convert_type_constructor(expr: &Expr) -> Result<DslTypeCon, String> {
    match expr {
        Expr::Name(n) => match n.id.as_str() {
            "int" => Ok(DslTypeCon::Int),
            "str" => Ok(DslTypeCon::Str),
            "bool" => Ok(DslTypeCon::Bool),
            "symint" => Ok(DslTypeCon::SymInt),
            "list" => Ok(DslTypeCon::List),
            "ShapedArray" => Ok(DslTypeCon::ShapedArray),
            other => Err(format!(
                "unknown type constructor '{}' in isinstance. \
                 Expected one of: int, str, bool, symint, list, ShapedArray",
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
            "ShapedArray" => Ok(DslType::ShapedArray),
            "None" => Ok(DslType::None),
            other => Err(format!(
                "unknown type '{}' in annotation. \
                 Expected one of: int, symint, bool, str, ShapedArray, None",
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
fn extract_error_expr(exc: &Expr) -> Result<DslExpr, DslCompileError> {
    if let Expr::Call(call) = exc
        && let Expr::Name(n) = call.func.as_ref()
    {
        if n.id.as_str() != "Error" {
            return Err(DslCompileError {
                range: exc.range(),
                message: format!("DSL raise must use Error(), got {}()", n.id),
            });
        }
        if call.arguments.args.len() != 1 {
            return Err(DslCompileError {
                range: exc.range(),
                message: format!(
                    "Error() must have exactly one argument, got {}",
                    call.arguments.args.len()
                ),
            });
        }
        return convert_expr(&call.arguments.args[0]);
    }
    Err(DslCompileError {
        range: exc.range(),
        message: "expected raise Error(expr) in DSL".to_owned(),
    })
}

/// Convert a sequence of Python statements into a DslBody.
/// The grammar's body is a recursive structure: assignments and ifs have
/// continuations, while return and raise are terminals.
fn convert_body(stmts: &[Stmt]) -> Result<DslBody, DslCompileError> {
    if stmts.is_empty() {
        return Err(DslCompileError {
            range: TextRange::default(),
            message: "empty body in DSL function".to_owned(),
        });
    }

    let range = stmts[0].range();
    match &stmts[0] {
        Stmt::Assign(assign) => {
            let vars = extract_assign_vars(&assign.targets).map_err(|msg| DslCompileError {
                range,
                message: msg,
            })?;
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
                return Err(DslCompileError {
                    range,
                    message: "DSL if must not have elif/else (use early returns)".to_owned(),
                });
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
            let value = ret.value.as_ref().ok_or_else(|| DslCompileError {
                range,
                message: "DSL return must have a value".to_owned(),
            })?;
            Ok(DslBody::Return(convert_expr(value)?))
        }
        Stmt::Raise(raise) => {
            let exc = raise.exc.as_ref().ok_or_else(|| DslCompileError {
                range,
                message: "DSL raise must have an exception".to_owned(),
            })?;
            Ok(DslBody::Raise(extract_error_expr(exc)?))
        }
        _ => Err(DslCompileError {
            range,
            message: format!(
                "unexpected statement in DSL body: {:?}",
                std::mem::discriminant(&stmts[0])
            ),
        }),
    }
}

/// Convert a ruff expression into a DslExpr.
fn convert_expr(expr: &Expr) -> Result<DslExpr, DslCompileError> {
    let range = expr.range();
    match expr {
        // Constants
        Expr::NoneLiteral(_) => Ok(DslExpr::Const(DslConst::None)),
        Expr::BooleanLiteral(b) => Ok(DslExpr::Const(DslConst::Bool(b.value))),
        Expr::NumberLiteral(n) => match &n.value {
            Number::Int(i) => Ok(DslExpr::Const(DslConst::Int(i.as_i64().ok_or_else(
                || DslCompileError {
                    range,
                    message: format!("int literal too large: {}", i),
                },
            )?))),
            _ => Err(DslCompileError {
                range,
                message: "non-int number in DSL expression".to_owned(),
            }),
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
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "only .shape attribute access is supported in DSL, got .{}",
                        attr.attr
                    ),
                });
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
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "DSL list comprehension must have exactly one generator, got {}",
                        comp.generators.len()
                    ),
                });
            }
            let generator = &comp.generators[0];
            let vars = extract_comp_vars(&generator.target).map_err(|msg| DslCompileError {
                range: generator.target.range(),
                message: msg,
            })?;
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
                    return Err(DslCompileError {
                        range,
                        message: format!("unsupported binary operator in DSL: {:?}", binop.op),
                    });
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
                _ => {
                    return Err(DslCompileError {
                        range,
                        message: format!("unsupported unary operator in DSL: {:?}", unary.op),
                    });
                }
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
                return Err(DslCompileError {
                    range,
                    message: "BoolOp must have at least 2 values".to_owned(),
                });
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
                return Err(DslCompileError {
                    range,
                    message: "DSL does not support chained comparisons".to_owned(),
                });
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
                            return Err(DslCompileError {
                                range,
                                message: format!(
                                    "unsupported comparison op in DSL: {:?}",
                                    cmp.ops[0]
                                ),
                            });
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

        _ => Err(DslCompileError {
            range,
            message: format!(
                "unexpected expression in DSL: {:?}",
                std::mem::discriminant(expr)
            ),
        }),
    }
}

/// Recursively extract a dotted name from an expression (e.g. `shape_extensions.dsl.prod`).
fn dotted_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(n) => Some(n.id.to_string()),
        Expr::Attribute(a) => {
            let prefix = dotted_name(&a.value)?;
            Some(format!("{}.{}", prefix, a.attr))
        }
        _ => None,
    }
}

/// Convert a function call expression, dispatching special forms.
fn convert_call(call: &ruff_python_ast::ExprCall) -> Result<DslExpr, DslCompileError> {
    let range = call.range();
    let func_name = dotted_name(&call.func).ok_or_else(|| DslCompileError {
        range,
        message: format!("unsupported call target: {:?}", call.func),
    })?;

    match func_name.as_str() {
        // Special forms with non-call syntax — keep as dedicated DslExpr variants
        "isinstance" => {
            if call.arguments.args.len() != 2 {
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "isinstance() takes exactly 2 arguments, got {}",
                        call.arguments.args.len()
                    ),
                });
            }
            Ok(DslExpr::IsInstance {
                expr: Box::new(convert_expr(&call.arguments.args[0])?),
                ty: convert_type_constructor(&call.arguments.args[1]).map_err(|msg| {
                    DslCompileError {
                        range: call.arguments.args[1].range(),
                        message: msg,
                    }
                })?,
            })
        }
        "ShapedArray" => {
            // ShapedArray(shape=expr) - keyword argument.
            if !call.arguments.args.is_empty() {
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "{}() uses keyword arg shape=, not positional args",
                        func_name
                    ),
                });
            }
            if call.arguments.keywords.len() != 1 {
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "{}() takes exactly one keyword arg, got {}",
                        func_name,
                        call.arguments.keywords.len()
                    ),
                });
            }
            let kw = &call.arguments.keywords[0];
            let kw_name = kw
                .arg
                .as_ref()
                .ok_or_else(|| DslCompileError {
                    range,
                    message: format!("{} keyword must be named", func_name),
                })?
                .as_str();
            if kw_name != "shape" {
                return Err(DslCompileError {
                    range,
                    message: format!("{}() keyword must be 'shape', got '{}'", func_name, kw_name),
                });
            }
            Ok(DslExpr::ShapedArrayNew(Box::new(convert_expr(&kw.value)?)))
        }

        // Builtins validated at parse time
        "len" => {
            if call.arguments.args.len() != 1 {
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "len() takes exactly 1 argument, got {}",
                        call.arguments.args.len()
                    ),
                });
            }
            Ok(DslExpr::Call {
                func: DslCallTarget::Builtin(DslBuiltin::Len),
                args: vec![convert_expr(&call.arguments.args[0])?],
            })
        }
        "range" => {
            if call.arguments.args.len() != 1 {
                return Err(DslCompileError {
                    range,
                    message: format!(
                        "range() takes exactly 1 argument in DSL, got {}",
                        call.arguments.args.len()
                    ),
                });
            }
            Ok(DslExpr::Call {
                func: DslCallTarget::Builtin(DslBuiltin::Range),
                args: vec![convert_expr(&call.arguments.args[0])?],
            })
        }
        "str"
        | "enumerate"
        | "zip"
        | "prod"
        | "shape_extensions.dsl.prod"
        | "sum"
        | "shape_extensions.dsl.sum"
        | "parse_einsum_equation"
        | "shape_extensions.dsl.parse_einsum_equation" => {
            let builtin = match func_name.as_str() {
                "prod" | "shape_extensions.dsl.prod" => DslBuiltin::Prod,
                "sum" | "shape_extensions.dsl.sum" => DslBuiltin::Sum,
                "str" => DslBuiltin::Str,
                "parse_einsum_equation" | "shape_extensions.dsl.parse_einsum_equation" => {
                    DslBuiltin::ParseEinsumEquation
                }
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
fn convert_fndef(func: &ruff_python_ast::StmtFunctionDef) -> Result<DslFnDef, DslCompileError> {
    let name = func.name.to_string();
    let name_range = func.name.range();

    let params: Vec<DslParam> = func
        .parameters
        .args
        .iter()
        .map(|p| {
            let param_name = p.parameter.name.to_string();
            let param_range = p.parameter.name.range();
            let ty = p
                .parameter
                .annotation
                .as_ref()
                .map(|a| {
                    convert_type_annotation(a).map_err(|msg| DslCompileError {
                        range: a.range(),
                        message: msg,
                    })
                })
                .transpose()?
                .ok_or_else(|| DslCompileError {
                    range: param_range,
                    message: format!(
                        "DSL parameter '{}' in function '{}' must have a type annotation",
                        param_name, name
                    ),
                })?;
            let default = p
                .default
                .as_ref()
                .map(|d| {
                    convert_default(d).map_err(|msg| DslCompileError {
                        range: d.range(),
                        message: msg,
                    })
                })
                .transpose()?;
            Ok(DslParam {
                name: param_name,
                ty,
                default,
            })
        })
        .collect::<Result<_, DslCompileError>>()?;

    let return_type = func
        .returns
        .as_ref()
        .map(|r| {
            convert_type_annotation(r).map_err(|msg| DslCompileError {
                range: r.range(),
                message: msg,
            })
        })
        .transpose()?;

    let body = convert_body(&func.body)?;

    Ok(DslFnDef {
        name,
        name_range,
        params,
        return_type,
        body,
    })
}

// Section: Type Inference
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
///
/// Each entry carries the declared return type plus the min/max argument
/// count so that call-site arg-count mismatches can be caught at compile time.
struct FnSig {
    ret_type: DslType,
    /// Number of required parameters (those without a default value).
    min_args: usize,
    /// Total number of parameters (required + optional).
    max_args: usize,
}

type FnRetTypes = HashMap<String, FnSig>;

/// The type `int | symint`, representing a tensor dimension.
fn dim_type() -> DslType {
    DslType::Union(vec![DslType::Int, DslType::SymInt])
}

/// True if this type contains `symint` (possibly inside a union).
fn contains_int(ty: &DslType) -> bool {
    match ty {
        DslType::SymInt => true,
        DslType::Union(types) => types.iter().any(contains_int),
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
            | (DslType::ShapedArray, DslTypeCon::ShapedArray)
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

/// Extract the element type from a list type. Pushes an error on non-list.
fn element_type(ty: &DslType, errors: &mut Vec<String>) -> DslType {
    match ty {
        DslType::List(inner) => *inner.clone(),
        _ => {
            errors.push(format!("expected list type, got {}", ty));
            DslType::Int
        }
    }
}

/// Narrow a type to only variants matching a constructor.
fn narrow_to(ty: &DslType, con: DslTypeCon, errors: &mut Vec<String>) -> DslType {
    match ty {
        DslType::Union(types) => {
            let matching: Vec<_> = types
                .iter()
                .filter(|t| matches_constructor(t, con))
                .cloned()
                .collect();
            match matching.len() {
                1 => matching.into_iter().next().unwrap(),
                0 => {
                    errors.push(format!(
                        "isinstance narrowing: no variant of {} matches {}",
                        ty, con
                    ));
                    ty.clone()
                }
                _ => DslType::Union(matching),
            }
        }
        _ if matches_constructor(ty, con) => ty.clone(),
        _ => {
            errors.push(format!(
                "isinstance narrowing: {} does not match {}",
                ty, con
            ));
            ty.clone()
        }
    }
}

/// Narrow a type to exclude variants matching a constructor.
fn narrow_away(ty: &DslType, con: DslTypeCon, errors: &mut Vec<String>) -> DslType {
    match ty {
        DslType::Union(types) => {
            let remaining: Vec<_> = types
                .iter()
                .filter(|t| !matches_constructor(t, con))
                .cloned()
                .collect();
            match remaining.len() {
                1 => remaining.into_iter().next().unwrap(),
                0 => {
                    errors.push(format!("narrowed away all variants of {}", ty));
                    ty.clone()
                }
                _ => DslType::Union(remaining),
            }
        }
        _ => ty.clone(),
    }
}

/// Check whether an inferred return-expression type is compatible with the
/// function's declared return type, at the granularity needed by `val_to_type`.
///
/// "Compatible" means: if the evaluator produces a `Val` consistent with
/// `inferred`, then `val_to_type(..., declared, ...)` will not hit an
/// `unreachable!` arm.  Used by `check_body` to validate `return`
/// expressions at compile time, making those arms sound.
fn return_type_compatible(inferred: &DslType, declared: &DslType) -> bool {
    // Union inferred: every possible runtime Val must be covered by `declared`.
    if let DslType::Union(variants) = inferred {
        return variants.iter().all(|v| return_type_compatible(v, declared));
    }
    // list[int | symint] -> int | symint: existing dynamic dimension tuple
    // behaviour for APIs such as Tensor.size() with no dim argument.
    if let (DslType::List(inner), DslType::Union(variants)) = (inferred, declared)
        && variants
            .iter()
            .all(|v| matches!(v, DslType::Int | DslType::SymInt))
    {
        return return_type_compatible(inner, declared);
    }
    // Union declared: the inferred Val must match at least one variant.
    if let DslType::Union(variants) = declared {
        return variants.iter().any(|v| return_type_compatible(inferred, v));
    }
    match (inferred, declared) {
        _ if inferred == declared => true,
        // Val::Int is valid for a declared Int via val_to_scalar_type.
        (DslType::Int, DslType::SymInt) => true,
        (DslType::List(inferred_inner), DslType::List(declared_inner)) => {
            return_type_compatible(inferred_inner, declared_inner)
        }
        // list[ShapedArray] -> ShapedArray: existing dynamic multi-array return
        // behaviour where some ops return [t1, t2] despite declaring a single
        // shaped-array return.
        (DslType::List(inner), DslType::ShapedArray)
            if matches!(inner.as_ref(), DslType::ShapedArray) =>
        {
            true
        }
        // List expressions are backed by Val::List, same as Tuple at runtime.
        // Check that the list element type is compatible with every tuple element.
        (DslType::List(inner), DslType::Tuple(elements)) => {
            elements.iter().all(|e| return_type_compatible(inner, e))
        }
        _ => false,
    }
}

/// Narrow a type to exclude None.
fn narrow_away_none(ty: &DslType, errors: &mut Vec<String>) -> DslType {
    match ty {
        DslType::Union(types) => {
            let remaining: Vec<_> = types
                .iter()
                .filter(|t| !matches!(t, DslType::None))
                .cloned()
                .collect();
            match remaining.len() {
                1 => remaining.into_iter().next().unwrap(),
                0 => {
                    errors.push(format!("narrowed away all variants of {}", ty));
                    ty.clone()
                }
                _ => DslType::Union(remaining),
            }
        }
        _ => ty.clone(),
    }
}

/// Analyze a condition expression for type narrowing.
/// Returns (then_env, else_env) — the environments for the true and false branches.
fn narrow(cond: &DslExpr, env: &TypeEnv, errors: &mut Vec<String>) -> (TypeEnv, TypeEnv) {
    match cond {
        // isinstance(x, con)
        DslExpr::IsInstance { expr, ty } => {
            if let DslExpr::Var(name) = expr.as_ref()
                && let Some(var_ty) = env.get(name)
            {
                let mut then_env = env.clone();
                let mut else_env = env.clone();
                then_env.insert(name.clone(), narrow_to(var_ty, *ty, errors));
                else_env.insert(name.clone(), narrow_away(var_ty, *ty, errors));
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
                else_env.insert(name.clone(), narrow_away_none(var_ty, errors));
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
                then_env.insert(name.clone(), narrow_away_none(var_ty, errors));
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
            let (then_env, else_env) = narrow(operand, env, errors);
            (else_env, then_env)
        }
        // cond1 and cond2 — narrow both in then-branch, conservative in else
        DslExpr::BinOp {
            left,
            op: DslOp::And,
            right,
        } => {
            let (then1, _) = narrow(left, env, errors);
            let (then2, _) = narrow(right, &then1, errors);
            (then2, env.clone())
        }
        _ => (env.clone(), env.clone()),
    }
}

/// Build function signature map from DSL definitions.
///
/// Records the return type plus the min/max argument count (for compile-time
/// arg-count validation in `infer_call`).
fn build_fn_ret_types(fndefs: &[DslFnDef], errors: &mut Vec<DslCompileError>) -> FnRetTypes {
    fndefs
        .iter()
        .filter_map(|f| match &f.return_type {
            Some(rt) => {
                let min_args = f.params.iter().filter(|p| p.default.is_none()).count();
                let max_args = f.params.len();
                Some((
                    f.name.clone(),
                    FnSig {
                        ret_type: rt.clone(),
                        min_args,
                        max_args,
                    },
                ))
            }
            None => {
                errors.push(DslCompileError {
                    range: f.name_range,
                    message: format!("DSL function {} must have a return type", f.name),
                });
                None
            }
        })
        .collect()
}

/// Result type of numeric arithmetic. If either operand contains symint,
/// the result is `int | symint` (will generate DimAdd etc.); otherwise
/// the result is `int` (will generate IntAdd etc.).
fn arithmetic_result(a: &DslType, b: &DslType) -> DslType {
    if contains_int(a) || contains_int(b) {
        dim_type()
    } else {
        DslType::Int
    }
}

/// Infer the element type of a list literal from its elements.
fn infer_list_elem_type(
    elts: &[DslExpr],
    env: &TypeEnv,
    sigs: &FnRetTypes,
    errors: &mut Vec<String>,
) -> DslType {
    if elts.is_empty() {
        errors.push("infer_list_elem_type called with empty list".to_owned());
        return DslType::Int;
    }
    let mut result = infer_expr(&elts[0], env, sigs, errors);
    for elt in &elts[1..] {
        result = join(&result, &infer_expr(elt, env, sigs, errors));
    }
    result
}

/// Bind comprehension variables based on the iterator expression.
/// Handles zip (multi-list iteration) and enumerate (index + element).
fn bind_comp_vars(
    vars: &[String],
    iter: &DslExpr,
    env: &TypeEnv,
    sigs: &FnRetTypes,
    errors: &mut Vec<String>,
) -> TypeEnv {
    let mut new_env = env.clone();
    match iter {
        DslExpr::Call {
            func: DslCallTarget::Builtin(DslBuiltin::Zip),
            args,
        } => {
            if vars.len() != args.len() {
                errors.push(format!("zip: {} vars but {} args", vars.len(), args.len()));
            }
            for (var, arg) in vars.iter().zip(args.iter()) {
                let arg_ty = infer_expr(arg, env, sigs, errors);
                new_env.insert(var.clone(), element_type(&arg_ty, errors));
            }
        }
        DslExpr::Call {
            func: DslCallTarget::Builtin(DslBuiltin::Enumerate),
            args,
        } => {
            if args.len() != 1 {
                errors.push(format!(
                    "enumerate takes exactly 1 argument, got {}",
                    args.len()
                ));
            }
            if vars.len() != 2 {
                errors.push(format!(
                    "enumerate requires exactly 2 variables, got {}",
                    vars.len()
                ));
            }
            if !args.is_empty() && vars.len() >= 2 {
                let list_ty = infer_expr(&args[0], env, sigs, errors);
                new_env.insert(vars[0].clone(), DslType::Int);
                new_env.insert(vars[1].clone(), element_type(&list_ty, errors));
            }
        }
        _ => {
            let iter_ty = infer_expr(iter, env, sigs, errors);
            if vars.len() == 1 {
                new_env.insert(vars[0].clone(), element_type(&iter_ty, errors));
            } else {
                // Multiple vars iterating over a single list — each gets element type.
                let elem = element_type(&iter_ty, errors);
                for var in vars {
                    new_env.insert(var.clone(), elem.clone());
                }
            }
        }
    }
    new_env
}

/// Infer the return type of a function call.
fn infer_call(
    func: &DslCallTarget,
    args: &[DslExpr],
    env: &TypeEnv,
    sigs: &FnRetTypes,
    errors: &mut Vec<String>,
) -> DslType {
    // Infer all arguments (undefined-variable detection, etc.).
    let arg_tys: Vec<DslType> = args
        .iter()
        .map(|a| infer_expr(a, env, sigs, errors))
        .collect();
    match func {
        DslCallTarget::Builtin(builtin) => match builtin {
            // prod/sum reduce a list of dims to a single dim.
            DslBuiltin::Prod | DslBuiltin::Sum => {
                if arg_tys.len() != 1 {
                    errors.push(format!(
                        "{} takes exactly 1 argument, got {}",
                        builtin,
                        arg_tys.len()
                    ));
                    return DslType::Int;
                }
                element_type(&arg_tys[0], errors)
            }
            DslBuiltin::Str => {
                if arg_tys.len() != 1 {
                    errors.push(format!(
                        "str() takes exactly 1 argument, got {}",
                        arg_tys.len()
                    ));
                }
                DslType::Str
            }
            DslBuiltin::ParseEinsumEquation => {
                if arg_tys.len() != 1 {
                    errors.push(format!(
                        "parse_einsum_equation() takes exactly 1 argument, got {}",
                        arg_tys.len()
                    ));
                }
                DslType::List(Box::new(DslType::List(Box::new(DslType::List(Box::new(
                    DslType::Int,
                ))))))
            }
            DslBuiltin::Len => {
                if arg_tys.len() != 1 {
                    errors.push(format!(
                        "len() takes exactly 1 argument, got {}",
                        arg_tys.len()
                    ));
                    return DslType::Int;
                }
                if !matches!(arg_tys[0], DslType::List(_)) {
                    errors.push(format!(
                        "len() requires a list argument, got {}",
                        arg_tys[0]
                    ));
                }
                DslType::Int
            }
            DslBuiltin::Range => {
                if arg_tys.len() != 1 {
                    errors.push(format!(
                        "range() takes exactly 1 argument, got {}",
                        arg_tys.len()
                    ));
                }
                DslType::List(Box::new(DslType::Int))
            }
            DslBuiltin::Zip | DslBuiltin::Enumerate => {
                errors.push(format!(
                    "{} should only appear as comprehension iterator",
                    builtin
                ));
                DslType::Int
            }
        },
        DslCallTarget::UserDefined(name) => match sigs.get(name) {
            Some(sig) => {
                let n = args.len();
                if n < sig.min_args {
                    if sig.min_args == sig.max_args {
                        errors.push(format!(
                            "'{}' takes exactly {} argument(s), got {}",
                            name, sig.min_args, n
                        ));
                    } else {
                        errors.push(format!(
                            "'{}' requires at least {} argument(s), got {}",
                            name, sig.min_args, n
                        ));
                    }
                } else if n > sig.max_args {
                    errors.push(format!(
                        "'{}' takes at most {} argument(s), got {}",
                        name, sig.max_args, n
                    ));
                }
                sig.ret_type.clone()
            }
            None => {
                errors.push(format!("undefined function: {}", name));
                DslType::Int
            }
        },
    }
}

/// Infer the type of a DSL expression.
fn infer_expr(
    expr: &DslExpr,
    env: &TypeEnv,
    sigs: &FnRetTypes,
    errors: &mut Vec<String>,
) -> DslType {
    match expr {
        DslExpr::Const(c) => match c {
            DslConst::None => DslType::None,
            DslConst::Int(_) => DslType::Int,
            DslConst::Bool(_) => DslType::Bool,
            DslConst::Str(_) => DslType::Str,
        },
        DslExpr::Var(name) => match env.get(name) {
            Some(ty) => ty.clone(),
            None => {
                errors.push(format!("undefined variable: {}", name));
                DslType::Int
            }
        },
        DslExpr::List(elts) => {
            if elts.is_empty() {
                // All empty list literals in the DSL are dimension lists.
                DslType::List(Box::new(dim_type()))
            } else if matches!(elts.last(), Some(DslExpr::Ellipsis)) {
                // [expr, ...] — unbounded list sentinel.
                let elem_ty = infer_list_elem_type(&elts[..elts.len() - 1], env, sigs, errors);
                DslType::List(Box::new(elem_ty))
            } else {
                let elem_ty = infer_list_elem_type(elts, env, sigs, errors);
                DslType::List(Box::new(elem_ty))
            }
        }
        DslExpr::ListComp {
            elt, vars, iter, ..
        } => {
            let comp_env = bind_comp_vars(vars, iter, env, sigs, errors);
            let elt_ty = infer_expr(elt, &comp_env, sigs, errors);
            DslType::List(Box::new(elt_ty))
        }
        DslExpr::Index { base, index } => {
            let base_ty = infer_expr(base, env, sigs, errors);
            infer_expr(index, env, sigs, errors);
            element_type(&base_ty, errors)
        }
        DslExpr::Slice { base, lower, upper } => {
            let base_ty = infer_expr(base, env, sigs, errors);
            if !matches!(base_ty, DslType::List(_)) {
                errors.push(format!("slice requires a list operand, got {}", base_ty));
            }
            if let Some(l) = lower {
                infer_expr(l, env, sigs, errors);
            }
            if let Some(u) = upper {
                infer_expr(u, env, sigs, errors);
            }
            base_ty
        }
        DslExpr::BinOp { left, op, right } => {
            let lt = infer_expr(left, env, sigs, errors);
            let rt = infer_expr(right, env, sigs, errors);
            match op {
                DslOp::Add => {
                    // List concatenation, string concatenation, or numeric addition.
                    if let DslType::List(a) = &lt {
                        if let DslType::List(b) = &rt {
                            DslType::List(Box::new(join(a, b)))
                        } else {
                            errors.push(format!("+ with list and non-list: {} + {}", lt, rt));
                            DslType::Int
                        }
                    } else if matches!(lt, DslType::Str) {
                        if !matches!(rt, DslType::Str) {
                            errors.push(format!("+ with str and non-str: {} + {}", lt, rt));
                        }
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
                infer_expr(operand, env, sigs, errors);
                DslType::Bool
            }
            DslUnaryOp::Neg => infer_expr(operand, env, sigs, errors),
        },
        DslExpr::Call { func, args } => infer_call(func, args, env, sigs, errors),
        DslExpr::IsInstance { expr, .. } => {
            infer_expr(expr, env, sigs, errors);
            DslType::Bool
        }
        DslExpr::In { left, right } => {
            infer_expr(left, env, sigs, errors);
            infer_expr(right, env, sigs, errors);
            DslType::Bool
        }
        DslExpr::Shape(inner) => {
            infer_expr(inner, env, sigs, errors);
            DslType::List(Box::new(dim_type()))
        }
        DslExpr::ShapedArrayNew(inner) => {
            infer_expr(inner, env, sigs, errors);
            DslType::ShapedArray
        }
        DslExpr::IfExpr { body, test, orelse } => {
            let (then_env, else_env) = narrow(test, env, errors);
            let body_ty = infer_expr(body, &then_env, sigs, errors);
            let else_ty = infer_expr(orelse, &else_env, sigs, errors);
            join(&body_ty, &else_ty)
        }
        DslExpr::Ellipsis => {
            errors.push("Ellipsis should be handled by List".to_owned());
            DslType::Int
        }
        DslExpr::Unknown => DslType::None, // sentinel for fixture fallback
    }
}

/// Type-check a function body, updating the environment through assignments
/// and narrowing through conditionals.
///
/// `ret_ty` is the declared return type of the enclosing function, used to
/// validate `return` expressions at compile time.  Pass `None` when the
/// function has no return annotation (an error already reported elsewhere).
fn check_body(
    body: &DslBody,
    env: &TypeEnv,
    sigs: &FnRetTypes,
    ret_ty: Option<&DslType>,
    errors: &mut Vec<String>,
) {
    match body {
        DslBody::Assign { vars, expr, rest } => {
            let ty = infer_expr(expr, env, sigs, errors);
            let mut new_env = env.clone();
            if vars.len() == 1 {
                new_env.insert(vars[0].clone(), ty);
            } else {
                let elem = element_type(&ty, errors);
                for var in vars {
                    new_env.insert(var.clone(), elem.clone());
                }
            }
            check_body(rest, &new_env, sigs, ret_ty, errors);
        }
        DslBody::If {
            cond,
            then_body,
            rest,
        } => {
            let (then_env, else_env) = narrow(cond, env, errors);
            check_body(then_body, &then_env, sigs, ret_ty, errors);
            check_body(rest, &else_env, sigs, ret_ty, errors);
        }
        DslBody::Return(expr) => {
            let expr_ty = infer_expr(expr, env, sigs, errors);
            if let Some(ret_ty) = ret_ty
                && !matches!(expr, DslExpr::Unknown)
                && !return_type_compatible(&expr_ty, ret_ty)
            {
                errors.push(format!(
                    "return expression type {} is not compatible with declared return type {}",
                    expr_ty, ret_ty
                ));
            }
        }
        DslBody::Raise(expr) => {
            let ty = infer_expr(expr, env, sigs, errors);
            if !matches!(ty, DslType::Str) {
                errors.push(format!("raise expression must be a string, got {}", ty));
            }
        }
    }
}

/// Type-check all DSL function definitions. Returns `Err` with collected
/// error messages if type errors are found.
fn type_check_program(fndefs: &[DslFnDef]) -> Result<(), Vec<DslCompileError>> {
    let mut errors: Vec<DslCompileError> = Vec::new();
    let sigs = build_fn_ret_types(fndefs, &mut errors);
    for fndef in fndefs {
        let mut fn_errors: Vec<String> = Vec::new();
        let mut env = TypeEnv::new();
        for param in &fndef.params {
            env.insert(param.name.clone(), param.ty.clone());
        }
        // Pass the declared return type so check_body can validate return exprs.
        // Functions without a return annotation already have an error from
        // build_fn_ret_types; we still type-check the body for other errors.
        let ret_ty = fndef.return_type.as_ref();
        check_body(&fndef.body, &env, &sigs, ret_ty, &mut fn_errors);
        errors.extend(fn_errors.into_iter().map(|message| DslCompileError {
            range: fndef.name_range,
            message,
        }));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// Section: Interpreter — evaluate DSL directly against runtime Val values

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
        DslType::ShapedArray => Some(Val::Shape(extract::shaped_array_shape(actual_arg_type)?)),
        DslType::None => actual_arg_type.is_none().then_some(Val::None),
        DslType::List(inner) => match inner.as_ref() {
            DslType::Int => Some(Val::List(
                extract::int_list(actual_arg_type)?
                    .iter()
                    .map(|&i| Val::Int(i))
                    .collect(),
            )),
            DslType::ShapedArray => Some(Val::List(
                extract::shaped_array_list(actual_arg_type)?
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
                // dim_list handles Tuple(Concrete([Literal, Type::Int, ...])) for
                // both literal ints and symbolic dims.  Convert concrete
                // Type::Int(Int::Literal(n)) → Val::Int(n) so == comparisons against
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
            // Iterating over a variadic shape is not supported — soft error.
            if matches!(iter_val, Val::Unpacked { .. }) {
                return Err(ShapeError::Unsupported {
                    message: format!(
                        "{op_name}: cannot iterate over variadic shape in list comprehension"
                    ),
                });
            }
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
            if matches!(base_val, Val::Unpacked { .. }) {
                return Err(ShapeError::Unsupported {
                    message: format!("{op_name}: cannot index variadic shape"),
                });
            }
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
            match base_val {
                Val::Unpacked {
                    ref prefix,
                    ref middle,
                    ref suffix,
                } => {
                    // Evaluate bounds — at most one of lower/upper is present for
                    // the slice patterns used in the DSL (e.g., dims[:-1], dims[1:]).
                    let lo_val = match lower {
                        Some(e) => Some(eval_dsl_expr(e, env, fns, op_name)?.as_int()),
                        None => None,
                    };
                    let hi_val = match upper {
                        Some(e) => Some(eval_dsl_expr(e, env, fns, op_name)?.as_int()),
                        None => None,
                    };
                    eval_unpacked_slice(prefix, middle, suffix, lo_val, hi_val, op_name)
                }
                Val::List(_) => {
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
                _ => Err(ShapeError::Unsupported {
                    message: format!("{op_name}: cannot slice {}", base_val.variant_name()),
                }),
            }
        }

        DslExpr::BinOp { left, op, right } => {
            let lval = eval_dsl_expr(left, env, fns, op_name)?;
            // Short-circuit and/or (Python semantics).
            match op {
                DslOp::And => {
                    if !val_as_bool(&lval, op_name)? {
                        return Ok(Val::Bool(false));
                    }
                    let rval = eval_dsl_expr(right, env, fns, op_name)?;
                    return Ok(Val::Bool(val_as_bool(&rval, op_name)?));
                }
                DslOp::Or => {
                    if val_as_bool(&lval, op_name)? {
                        return Ok(Val::Bool(true));
                    }
                    let rval = eval_dsl_expr(right, env, fns, op_name)?;
                    return Ok(Val::Bool(val_as_bool(&rval, op_name)?));
                }
                _ => {}
            }
            let rval = eval_dsl_expr(right, env, fns, op_name)?;
            eval_binop(&lval, *op, &rval, op_name)
        }

        DslExpr::UnaryOp { op, operand } => {
            let val = eval_dsl_expr(operand, env, fns, op_name)?;
            match op {
                DslUnaryOp::Not => Ok(Val::Bool(!val_as_bool(&val, op_name)?)),
                DslUnaryOp::Neg => {
                    match &val {
                        Val::Int(n) => Ok(Val::Int(-n)),
                        Val::Dim(ty) => {
                            // Negate symbolic: 0 - ty
                            let zero = Type::Int(Int::Literal(0));
                            Ok(Val::Dim(canonicalize(Type::Int(Int::sub(
                                zero,
                                ty.clone(),
                            )))))
                        }
                        _ => Err(ShapeError::Unsupported {
                            message: format!("{op_name}: cannot negate {}", val.variant_name()),
                        }),
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
                (DslTypeCon::List, Val::Unpacked { .. }) => true,
                (DslTypeCon::ShapedArray, Val::Shape(_)) => true,
                _ => false,
            };
            Ok(Val::Bool(matches))
        }

        DslExpr::In { left, right } => {
            let needle = eval_dsl_expr(left, env, fns, op_name)?;
            let haystack = eval_dsl_expr(right, env, fns, op_name)?;
            if matches!(haystack, Val::Unpacked { .. }) {
                return Err(ShapeError::Unsupported {
                    message: format!("{op_name}: 'in' not supported on variadic shape"),
                });
            }
            let items = haystack.as_list();
            let found = items.iter().any(|item| val_eq(&needle, item));
            Ok(Val::Bool(found))
        }

        DslExpr::Shape(inner) => {
            let val = eval_dsl_expr(inner, env, fns, op_name)?;
            let shape = val.as_shape();
            match shape.view() {
                IntTupleView::Concrete(dims) => {
                    // Use dim_val to convert concrete
                    // Type::Int(Int::Literal(n)) to Val::Int(n) so comparisons
                    // against literal ints (e.g., `d != 1` in squeeze) work naturally.
                    let vals: Vec<Val> =
                        dims.iter().map(|d| dim_val(Type::Int(d.clone()))).collect();
                    Ok(Val::List(vals))
                }
                IntTupleView::Gradual => Ok(Val::Unpacked {
                    prefix: Vec::new(),
                    middle: IntTuple::shapeless().to_shape_arg_type(),
                    suffix: Vec::new(),
                }),
                IntTupleView::Unpacked {
                    prefix,
                    middle,
                    suffix,
                } => Ok(Val::Unpacked {
                    prefix: prefix
                        .iter()
                        .map(|d| dim_val(Type::Int(d.clone())))
                        .collect(),
                    middle: middle.clone(),
                    suffix: suffix
                        .iter()
                        .map(|d| dim_val(Type::Int(d.clone())))
                        .collect(),
                }),
            }
        }

        DslExpr::ShapedArrayNew(shape_expr) => {
            let val = eval_dsl_expr(shape_expr, env, fns, op_name)?;
            match val {
                Val::Unpacked {
                    prefix,
                    middle,
                    suffix,
                } => {
                    let prefix_types = prefix.iter().map(|v| v.as_size()).collect();
                    let suffix_types = suffix.iter().map(|v| v.as_size()).collect();
                    Ok(Val::Shape(IntTuple::unpacked_from_types(
                        prefix_types,
                        middle,
                        suffix_types,
                    )))
                }
                _ => {
                    let dims = val.as_size_list();
                    Ok(Val::Shape(IntTuple::from_types(dims)))
                }
            }
        }

        DslExpr::IfExpr { body, test, orelse } => {
            if val_as_bool(&eval_dsl_expr(test, env, fns, op_name)?, op_name)? {
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

/// Return the inner bool from a `Val::Bool`, or `Err(ShapeError::Unsupported)` if
/// the value is not boolean.  Used in place of `Val::as_bool()` throughout the
/// evaluator so that ill-typed DSL code produces a graceful fallback rather than
/// a Rust panic.
fn val_as_bool(val: &Val, op_name: &str) -> Result<bool, ShapeError> {
    match val {
        Val::Bool(b) => Ok(*b),
        other => Err(ShapeError::Unsupported {
            message: format!(
                "{op_name}: expected a bool value, got {}",
                other.variant_name()
            ),
        }),
    }
}

/// Normalize a canonical `Type` to `Val`. If it's a concrete
/// `Type::Int(Int::Literal(n))`,
/// produce `Val::Int(n)` so equality checks against literal ints work naturally.
/// Otherwise produce `Val::Dim(ty)`.
fn dim_val(ty: Type) -> Val {
    match &ty {
        Type::Int(Int::Literal(n)) => Val::Int(*n),
        _ => Val::Dim(ty),
    }
}

/// Evaluate a slice operation on a `Val::Unpacked { prefix, middle, suffix }`.
///
/// Supports the slice patterns used in the DSL for variadic shapes:
/// - `dims[:n]` (positive n): takes from prefix if n <= prefix.len()
/// - `dims[n:]` (positive n): drops from prefix if n <= prefix.len()
/// - `dims[:-n]` (negative upper): drops from suffix if n <= suffix.len()
/// - `dims[-n:]` (negative lower): takes from suffix if n <= suffix.len()
///
/// Returns `Err(Unsupported)` if the slice crosses the variadic middle.
fn eval_unpacked_slice(
    prefix: &[Val],
    middle: &Type,
    suffix: &[Val],
    lo_val: Option<i64>,
    hi_val: Option<i64>,
    op_name: &str,
) -> Result<Val, ShapeError> {
    match (lo_val, hi_val) {
        // dims[:n] where n >= 0 — take first n from prefix
        (None, Some(n)) if n >= 0 => {
            let n = n as usize;
            if n <= prefix.len() {
                Ok(Val::List(prefix[..n].to_vec()))
            } else {
                Err(ShapeError::Unsupported {
                    message: format!(
                        "{op_name}: slice [:{}] crosses variadic middle (prefix len {})",
                        n,
                        prefix.len()
                    ),
                })
            }
        }
        // dims[:-n] where n > 0 — drop last n from suffix
        (None, Some(n)) if n < 0 => {
            let drop = (-n) as usize;
            if drop <= suffix.len() {
                let new_suffix = suffix[..suffix.len() - drop].to_vec();
                Ok(Val::Unpacked {
                    prefix: prefix.to_vec(),
                    middle: middle.clone(),
                    suffix: new_suffix,
                })
            } else {
                Err(ShapeError::Unsupported {
                    message: format!(
                        "{op_name}: slice [:-{}] crosses variadic middle (suffix len {})",
                        drop,
                        suffix.len()
                    ),
                })
            }
        }
        // dims[n:] where n >= 0 — drop first n from prefix
        (Some(n), None) if n >= 0 => {
            let n = n as usize;
            if n <= prefix.len() {
                Ok(Val::Unpacked {
                    prefix: prefix[n..].to_vec(),
                    middle: middle.clone(),
                    suffix: suffix.to_vec(),
                })
            } else {
                Err(ShapeError::Unsupported {
                    message: format!(
                        "{op_name}: slice [{}:] crosses variadic middle (prefix len {})",
                        n,
                        prefix.len()
                    ),
                })
            }
        }
        // dims[-n:] where n > 0 — take last n from suffix
        (Some(n), None) if n < 0 => {
            let take = (-n) as usize;
            if take <= suffix.len() {
                Ok(Val::List(suffix[suffix.len() - take..].to_vec()))
            } else {
                Err(ShapeError::Unsupported {
                    message: format!(
                        "{op_name}: slice [-{}:] crosses variadic middle (suffix len {})",
                        take,
                        suffix.len()
                    ),
                })
            }
        }
        _ => Err(ShapeError::Unsupported {
            message: format!("{op_name}: unsupported slice pattern on variadic shape"),
        }),
    }
}

/// Evaluate a binary operation, dispatching on runtime Val variants.
///
/// Arithmetic: both Int → concrete i64; either Dim → symbolic integer expression; + on Lists → concat.
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
            // Unpacked + List → append list elements to suffix
            (
                Val::Unpacked {
                    prefix,
                    middle,
                    suffix,
                },
                Val::List(b),
            ) => {
                let mut new_suffix = suffix.clone();
                new_suffix.extend(b.iter().cloned());
                Ok(Val::Unpacked {
                    prefix: prefix.clone(),
                    middle: middle.clone(),
                    suffix: new_suffix,
                })
            }
            // List + Unpacked → prepend list elements to prefix
            (
                Val::List(a),
                Val::Unpacked {
                    prefix,
                    middle,
                    suffix,
                },
            ) => {
                let mut new_prefix = a.clone();
                new_prefix.extend(prefix.iter().cloned());
                Ok(Val::Unpacked {
                    prefix: new_prefix,
                    middle: middle.clone(),
                    suffix: suffix.clone(),
                })
            }
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Int(Int::add(a, b)))))
            }
        },
        DslOp::Sub => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Int(a - b)),
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Int(Int::sub(a, b)))))
            }
        },
        DslOp::Mul => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Int(a * b)),
            _ => {
                let a = lval.as_size();
                let b = rval.as_size();
                Ok(dim_val(canonicalize(Type::Int(Int::mul(a, b)))))
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
                Ok(dim_val(canonicalize(Type::Int(Int::floor_div(a, b)))))
            }
        },
        DslOp::Mod => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => {
                if *b == 0 {
                    return Err(ShapeError::ShapeComputation {
                        message: format!("{op_name}: modulo by zero"),
                    });
                }
                Ok(Val::Int(a % b))
            }
            _ => Err(ShapeError::Unsupported {
                message: format!("{op_name}: % not supported on non-integer values"),
            }),
        },

        // --- Comparison ---
        DslOp::Eq => Ok(Val::Bool(val_eq(lval, rval))),
        DslOp::NotEq => Ok(Val::Bool(!val_eq(lval, rval))),
        DslOp::Lt => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Bool(a < b)),
            _ => Err(ShapeError::Unsupported {
                message: format!("{op_name}: < not supported on non-integer values"),
            }),
        },
        DslOp::LtE => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Bool(a <= b)),
            _ => Err(ShapeError::Unsupported {
                message: format!("{op_name}: <= not supported on non-integer values"),
            }),
        },
        DslOp::Gt => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Bool(a > b)),
            _ => Err(ShapeError::Unsupported {
                message: format!("{op_name}: > not supported on non-integer values"),
            }),
        },
        DslOp::GtE => match (lval, rval) {
            (Val::Int(a), Val::Int(b)) => Ok(Val::Bool(a >= b)),
            _ => Err(ShapeError::Unsupported {
                message: format!("{op_name}: >= not supported on non-integer values"),
            }),
        },

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
        // Structurally equal symbolic dims compare equal (e.g. `int_min(N, N) == N`),
        // matching how a literal dim compares equal to the same integer.
        (Val::Dim(x), Val::Dim(y)) => x == y,
        (Val::None, Val::None) => true,
        (Val::Int(x), Val::Dim(y)) | (Val::Dim(y), Val::Int(x)) => {
            *y == Type::Int(Int::Literal(*x))
        }
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
                if matches!(args[0], Val::Unpacked { .. }) {
                    return Err(ShapeError::Unsupported {
                        message: format!("{op_name}: prod() not supported on variadic shape"),
                    });
                }
                let items = args[0].as_list();
                // Check if all items are Int — if so, concrete product.
                // If any is Dim, use symbolic product.
                let all_int = items.iter().all(|v| matches!(v, Val::Int(_)));
                if all_int {
                    let product: i64 = items.iter().map(|v| v.as_int()).product();
                    Ok(Val::Int(product))
                } else {
                    let dims = args[0].as_size_list();
                    let mut product = Type::Int(Int::Literal(1));
                    for d in dims {
                        product = canonicalize(Type::Int(Int::mul(product, d)));
                    }
                    Ok(dim_val(product))
                }
            }
            DslBuiltin::Sum => {
                assert_eq!(args.len(), 1, "DSL bug: {op_name}: sum takes 1 arg");
                if matches!(args[0], Val::Unpacked { .. }) {
                    return Err(ShapeError::Unsupported {
                        message: format!("{op_name}: sum() not supported on variadic shape"),
                    });
                }
                let items = args[0].as_list();
                let all_int = items.iter().all(|v| matches!(v, Val::Int(_)));
                if all_int {
                    let total: i64 = items.iter().map(|v| v.as_int()).sum();
                    Ok(Val::Int(total))
                } else {
                    let dims = args[0].as_size_list();
                    let mut total = Type::Int(Int::Literal(0));
                    for d in dims {
                        total = canonicalize(Type::Int(Int::add(total, d)));
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
                    Val::Unpacked { .. } => "Unpacked".to_owned(),
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
                match &args[0] {
                    Val::Unpacked { .. } => Err(ShapeError::Unsupported {
                        message: format!("{op_name}: enumerate() not supported on variadic shape"),
                    }),
                    _ => {
                        let items = args[0].as_list();
                        Ok(Val::List(
                            items
                                .iter()
                                .enumerate()
                                .map(|(i, v)| Val::List(vec![Val::Int(i as i64), v.clone()]))
                                .collect(),
                        ))
                    }
                }
            }
            DslBuiltin::Zip => {
                assert!(args.len() >= 2, "DSL bug: {op_name}: zip takes 2+ args");
                if args.iter().any(|a| matches!(a, Val::Unpacked { .. })) {
                    return Err(ShapeError::Unsupported {
                        message: format!("{op_name}: zip() not supported on variadic shape"),
                    });
                }
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
                match &args[0] {
                    Val::List(items) => Ok(Val::Int(items.len() as i64)),
                    Val::Unpacked { .. } => Err(ShapeError::Unsupported {
                        message: format!("{op_name}: len() not supported on variadic shape"),
                    }),
                    _ => panic!(
                        "DSL bug: {op_name}: len() expected List or Unpacked, got {}",
                        args[0].variant_name()
                    ),
                }
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
            // Fill defaults for remaining params; missing-required-arg errors
            // are already caught at compile time by infer_call, so reaching
            // here with too few args is a DSL bug rather than a user error.
            for param in fn_def.params.iter().skip(args.len()) {
                let default = param.default.as_ref().unwrap_or_else(|| {
                    unreachable!(
                        "DSL bug: {op_name}: missing required argument \
                             for function `{name}` — should have been caught by infer_call"
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
                if items.len() != vars.len() {
                    return Err(ShapeError::ShapeComputation {
                        message: format!(
                            "{op_name}: tuple unpack expected {} values but got {}",
                            vars.len(),
                            items.len()
                        ),
                    });
                }
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
            if val_as_bool(&eval_dsl_expr(cond, env, fns, op_name)?, op_name)? {
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
            let text = match msg {
                Val::Str(s) => s,
                other => {
                    return Err(ShapeError::Unsupported {
                        message: format!(
                            "{op_name}: raise value must be a string, got {}",
                            other.variant_name()
                        ),
                    });
                }
            };
            Err(ShapeError::ShapeComputation { message: text })
        }
    }
}

/// Inject a computed `IntTuple` into a fixture `Type`, preserving the base class.
/// Falls back to `ret_type` unchanged if it isn't a shaped-array type.
fn inject_shape(shape: IntTuple, ret_type: &Type) -> Type {
    match ret_type {
        Type::ShapedArray(t) => {
            let mut t = (**t).clone();
            t.set_shape(shape);
            t.to_type()
        }
        _ => ret_type.clone(),
    }
}

/// Convert a DSL scalar `Val` to a `Type` for output from a shape function:
/// - `Val::Int(n)` → `Literal[n]`
/// - `Val::Dim(Int::Literal(n))` → `Literal[n]` (concrete dims become literals)
/// - `Val::Dim(symbolic)` → canonical `Int[...]`
fn val_to_scalar_type(val: &Val) -> Type {
    match val {
        Val::Int(n) => Lit::Int(LitInt::new(*n)).to_implicit_type(),
        Val::Dim(ty) => {
            let dim = Int::from_type(ty).unwrap_or_else(|| Int::Symbolic(Box::new(ty.clone())));
            match dim {
                Int::Literal(n) => Lit::Int(LitInt::new(n)).to_implicit_type(),
                dim => canonicalize(Type::Int(dim)),
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
fn inject_shapes_into_tuple(shapes: Vec<IntTuple>, expected_return_type: &Type) -> Option<Type> {
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
        DslType::ShapedArray => match val {
            Val::Shape(s) => inject_shape(s, expected_return_type),
            // Some ops conditionally return a tuple of tensors despite
            // declaring `-> ShapedArray` (e.g. min_max_median_ir returns
            // [Tensor, Tensor] when dim is given).  return_type_compatible
            // allows list[ShapedArray] for a declared ShapedArray return, so
            // this path is guarded by check_body's static validation.
            Val::List(items) => {
                let shapes: Vec<IntTuple> = items.iter().map(|v| v.as_shape().clone()).collect();
                if shapes.len() == 1 {
                    return inject_shape(shapes.into_iter().next().unwrap(), expected_return_type);
                }
                inject_shapes_into_tuple(shapes, expected_return_type)
                    .unwrap_or_else(|| expected_return_type.clone())
            }
            _ => unreachable!(
                "validated by check_body: ShapedArray return but got {:?} (op: {})",
                val.variant_name(),
                op_name,
            ),
        },

        // Int and Bool synthesize Literal[n] / Literal[bool] from the DSL's
        // traced runtime value, just like Int does via `val_to_scalar_type`.
        // This is intentionally load-bearing: functions like `dim_ir`,
        // `numel_ir`, and `size_ir(dim=N)` trace exact integer results, and
        // downstream consumers (assert_type, reshape validation, shape
        // inference) rely on the literal precision. Returning
        // `expected_return_type` here would discard the traced value and
        // produce `int` instead of e.g. `Literal[3]`.
        //
        // This differs from the Tensor/List/Tuple/None/Str branches, which
        // return `expected_return_type.clone()`. Those branches are correct
        // because their `expected_return_type` already carries the refined
        // structure (e.g. `Tensor[B, C, H, W]` with shape injected). For
        // scalars, the fixture return type is just `int` — the literal value
        // comes solely from DSL evaluation.
        DslType::Int => match val {
            Val::Int(n) => Lit::Int(LitInt::new(n)).to_implicit_type(),
            _ => unreachable!(
                "validated by check_body: expected Int for int return, got {:?} (op: {})",
                val.variant_name(),
                op_name,
            ),
        },

        // Int synthesizes a type from the traced `Val`: `val_to_scalar_type`
        // returns canonical `Int` for symbolic dims and `Literal[n]` for ints.
        // The trace value is load-bearing for shape inference — downstream
        // tensor shape types are built from these dimension representations.
        DslType::SymInt => val_to_scalar_type(&val),

        DslType::Bool => match val {
            Val::Bool(b) => Lit::Bool(b).to_implicit_type(),
            _ => unreachable!(
                "validated by check_body: expected Bool for bool return, got {:?} (op: {})",
                val.variant_name(),
                op_name,
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
                    | (DslType::ShapedArray, Val::Shape(_))
                    | (DslType::Bool, Val::Bool(_))
                    | (DslType::Str, Val::Str(_))
                    | (DslType::None, Val::None) => {
                        return val_to_type(val, is_unbounded, v, expected_return_type, op_name);
                    }
                    _ => continue,
                }
            }
            unreachable!(
                "validated by check_body: no union variant matched return val {:?} (declared -> {}, op: {})",
                val.variant_name(),
                actual_result_type,
                op_name,
            );
        }

        DslType::List(inner) => match inner.as_ref() {
            DslType::ShapedArray => {
                let items = val.as_list();
                let shapes: Vec<IntTuple> = items.iter().map(|v| v.as_shape().clone()).collect();
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
            let all_shaped_array = elems.iter().all(|e| matches!(e, DslType::ShapedArray));
            if all_shaped_array {
                let shapes: Vec<IntTuple> = items.iter().map(|v| v.as_shape().clone()).collect();
                inject_shapes_into_tuple(shapes, expected_return_type)
                    .unwrap_or_else(|| expected_return_type.clone())
            } else {
                // All ints/ints → per-element normalization
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
struct DslMetaShapeFunction {
    /// The primary function to evaluate.
    fn_def: Arc<DslFnDef>,
    /// Precomputed lookup table mapping function names to definitions.
    /// Shared across all instances — built once at registry init.
    fn_lookup: Arc<HashMap<String, Arc<DslFnDef>>>,
}

impl MetaShapeFunction for DslMetaShapeFunction {
    fn name(&self) -> &str {
        &self.fn_def.name
    }

    fn param_names(&self) -> Vec<&str> {
        self.fn_def.params.iter().map(|p| p.name.as_str()).collect()
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
            // Unsupported errors on variadic shapes → fixture fallback (None),
            // not a user-visible error.
            Err(ShapeError::Unsupported { .. }) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
// Section: Public wrapper API
//
// These wrappers form the public surface of
// `pyrefly_types::meta_shape_dsl`. They let callers outside this module (the
// binder and solver in `pyrefly/lib`) drive the DSL pipeline without exposing
// the grammar-aligned `DslFnDef` internals.
//
// Constraint: the public surface never returns `Arc<DslFnDef>`. Callers store
// `ShapeDslFunction` opaquely; only code inside `pyrefly_types` may reach the
// underlying `DslFnDef` via the `pub(crate)` fields.

/// A single DSL function that has been lowered from its Python AST.
///
/// This is a cheap (one `Arc`) opaque handle produced by
/// [`convert_shape_dsl_function`] and consumed by [`validate_shape_dsl_functions`].
#[derive(Debug, Clone)]
pub struct ShapeDslFunction {
    pub(crate) inner: Arc<DslFnDef>,
}

/// Pointer identity: two `ShapeDslFunction`s are equal iff they point to the
/// same `DslFnDef` allocation.
impl PartialEq for ShapeDslFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for ShapeDslFunction {}

impl Hash for ShapeDslFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.inner) as *const () as usize).hash(state);
    }
}

impl PartialOrd for ShapeDslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShapeDslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_ptr = Arc::as_ptr(&self.inner) as *const () as usize;
        let other_ptr = Arc::as_ptr(&other.inner) as *const () as usize;
        self_ptr.cmp(&other_ptr)
    }
}

/// DSL IR contains no `Type` values, so visiting is a no-op.
impl Visit<Type> for ShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// DSL IR contains no `Type` values, so visiting is a no-op.
impl VisitMut<Type> for ShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

/// DSL IR contains no `Type` values, so visiting through `Arc` is also a no-op.
impl Visit<Type> for Arc<ShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// DSL IR contains no `Type` values, so visiting through `Arc` is also a no-op.
impl VisitMut<Type> for Arc<ShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeEq for ShapeDslFunction {
    fn type_eq(&self, other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        self == other
    }
}

impl ShapeDslFunction {
    /// The function name from the DSL definition.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns the set of user-defined function names called in this function's body.
    pub fn call_targets(&self) -> HashSet<String> {
        let mut targets = HashSet::new();
        collect_call_targets_body(&self.inner.body, &mut targets);
        targets
    }
}

/// Walk a `DslBody` and collect all `DslCallTarget::UserDefined` names.
fn collect_call_targets_body(body: &DslBody, targets: &mut HashSet<String>) {
    match body {
        DslBody::Assign { expr, rest, .. } => {
            collect_call_targets_expr(expr, targets);
            collect_call_targets_body(rest, targets);
        }
        DslBody::If {
            cond,
            then_body,
            rest,
        } => {
            collect_call_targets_expr(cond, targets);
            collect_call_targets_body(then_body, targets);
            collect_call_targets_body(rest, targets);
        }
        DslBody::Return(expr) | DslBody::Raise(expr) => {
            collect_call_targets_expr(expr, targets);
        }
    }
}

/// Walk a `DslExpr` and collect all `DslCallTarget::UserDefined` names.
fn collect_call_targets_expr(expr: &DslExpr, targets: &mut HashSet<String>) {
    match expr {
        DslExpr::Call {
            func: DslCallTarget::UserDefined(name),
            args,
        } => {
            targets.insert(name.clone());
            for arg in args {
                collect_call_targets_expr(arg, targets);
            }
        }
        DslExpr::Call { args, .. } => {
            for arg in args {
                collect_call_targets_expr(arg, targets);
            }
        }
        DslExpr::List(items) => {
            for item in items {
                collect_call_targets_expr(item, targets);
            }
        }
        DslExpr::ListComp {
            elt, iter, cond, ..
        } => {
            collect_call_targets_expr(elt, targets);
            collect_call_targets_expr(iter, targets);
            if let Some(c) = cond {
                collect_call_targets_expr(c, targets);
            }
        }
        DslExpr::Index { base, index } => {
            collect_call_targets_expr(base, targets);
            collect_call_targets_expr(index, targets);
        }
        DslExpr::Slice { base, lower, upper } => {
            collect_call_targets_expr(base, targets);
            if let Some(l) = lower {
                collect_call_targets_expr(l, targets);
            }
            if let Some(u) = upper {
                collect_call_targets_expr(u, targets);
            }
        }
        DslExpr::BinOp { left, right, .. } => {
            collect_call_targets_expr(left, targets);
            collect_call_targets_expr(right, targets);
        }
        DslExpr::UnaryOp { operand, .. } => {
            collect_call_targets_expr(operand, targets);
        }
        DslExpr::IsInstance { expr, .. } => {
            collect_call_targets_expr(expr, targets);
        }
        DslExpr::In { left, right } => {
            collect_call_targets_expr(left, targets);
            collect_call_targets_expr(right, targets);
        }
        DslExpr::Shape(expr) | DslExpr::ShapedArrayNew(expr) => {
            collect_call_targets_expr(expr, targets);
        }
        DslExpr::IfExpr { body, test, orelse } => {
            collect_call_targets_expr(body, targets);
            collect_call_targets_expr(test, targets);
            collect_call_targets_expr(orelse, targets);
        }
        DslExpr::Const(_) | DslExpr::Var(_) | DslExpr::Ellipsis | DslExpr::Unknown => {}
    }
}

/// Validate a set of `ShapeDslFunction`s as a program.
///
/// Runs `type_check_program` on the inner `DslFnDef`s, verifying that
/// cross-function calls have consistent signatures. Returns collected
/// type error messages on failure.
///
/// Also rejects programs whose call graph (restricted to `fns`) contains
/// a cycle, since the DSL evaluator does not support recursion.
///
/// Intended to be called with a per-caller transitive closure (root +
/// its resolved helpers), not the full module.
pub fn validate_shape_dsl_functions(
    fns: &[Arc<ShapeDslFunction>],
) -> Result<(), Vec<DslCompileError>> {
    // Reject recursive cycles before running the type checker. The DSL
    // evaluator does not support recursion and would infinite-loop at a
    // call site if a cycle reached the interpreter.
    if has_dsl_cycle(fns) {
        let root_fn = fns
            .first()
            // has_dsl_cycle returned true ⟹ fns is non-empty.
            .expect("non-empty closure when a cycle is detected");
        return Err(vec![DslCompileError {
            range: root_fn.inner.name_range,
            message: format!(
                "DSL function '{}' is recursive (or part of a recursive cycle); \
                 recursion is not supported in shape DSL",
                root_fn.name()
            ),
        }]);
    }
    let defs: Vec<DslFnDef> = fns.iter().map(|f| (*f.inner).clone()).collect();
    type_check_program(&defs)
}

/// Returns `true` if the call graph of `fns` — restricted to functions
/// within `fns` — contains a cycle.
///
/// Uses a DFS reachability check: a function is considered cyclic iff it
/// can reach itself through one or more hops.  Both direct self-recursion
/// and mutual recursion are detected.
fn has_dsl_cycle(fns: &[Arc<ShapeDslFunction>]) -> bool {
    let fn_names: HashSet<String> = fns.iter().map(|f| f.name().to_owned()).collect();
    // Build adjacency lists restricted to functions in this closure.
    let adj: HashMap<String, Vec<String>> = fns
        .iter()
        .map(|f| {
            let callees: Vec<String> = f
                .call_targets()
                .into_iter()
                .filter(|t| fn_names.contains(t))
                .collect();
            (f.name().to_owned(), callees)
        })
        .collect();
    fns.iter().any(|f| is_self_reachable(f.name(), &adj))
}

/// Returns `true` if `start` can reach itself through one or more edges in `adj`.
fn is_self_reachable(start: &str, adj: &HashMap<String, Vec<String>>) -> bool {
    let neighbors = match adj.get(start) {
        Some(n) => n,
        None => return false,
    };
    let mut visited: HashSet<&str> = HashSet::new();
    // Use a Vec as a DFS stack seeded with the direct callees of `start`.
    let mut stack: Vec<&str> = neighbors.iter().map(String::as_str).collect();
    while let Some(curr) = stack.pop() {
        if curr == start {
            return true;
        }
        if visited.insert(curr)
            && let Some(nexts) = adj.get(curr)
        {
            stack.extend(nexts.iter().map(String::as_str));
        }
    }
    false
}

/// Reference to a shape-DSL function that refines a callable's return type.
/// Carried on `FuncFlags` for functions decorated with `@uses_shape_dsl`.
#[derive(Debug, Clone)]
pub struct ShapeTransform {
    pub dsl_fn: Arc<ShapeDslFunction>,
    /// Transitive closure (including `dsl_fn` itself) of user-defined
    /// functions reachable from `dsl_fn`. Identity-ignored for hashing/eq.
    pub fn_closure: IdentityIgnored<Arc<Vec<Arc<ShapeDslFunction>>>>,
}

/// Pointer identity: delegates to `ShapeDslFunction`'s pointer-identity equality.
impl PartialEq for ShapeTransform {
    fn eq(&self, other: &Self) -> bool {
        self.dsl_fn == other.dsl_fn
    }
}

impl Eq for ShapeTransform {}

impl Hash for ShapeTransform {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dsl_fn.hash(state);
    }
}

impl PartialOrd for ShapeTransform {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShapeTransform {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dsl_fn.cmp(&other.dsl_fn)
    }
}

impl Visit<Type> for ShapeTransform {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for ShapeTransform {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<ShapeTransform> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Arc<ShapeTransform> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeEq for ShapeTransform {
    fn type_eq(&self, other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        self == other
    }
}

impl ShapeTransform {
    /// Build a `MetaShapeFunction` evaluator from this shape transform.
    /// Populates `fn_lookup` with all functions in `fn_closure` so that
    /// cross-function DSL calls resolve correctly.
    pub fn to_meta_shape_function(&self) -> Box<dyn MetaShapeFunction> {
        // fn_closure contains self and its transitive callees.
        let fn_lookup: Arc<HashMap<String, Arc<DslFnDef>>> = Arc::new(
            self.fn_closure
                .iter()
                .map(|h| (h.inner.name.clone(), h.inner.clone()))
                .collect(),
        );
        Box::new(DslMetaShapeFunction {
            fn_def: self.dsl_fn.inner.clone(),
            fn_lookup,
        })
    }
}

/// Convert a single Python function definition into a [`ShapeDslFunction`].
///
/// This is pure AST-to-IR lowering — it does not parse source text or run
/// the type checker. The output is a single opaque handle; the caller is
/// expected to combine handles from this function (and possibly other
/// modules) via [`validate_shape_dsl_functions`].
///
/// Returns `Err` with a terse description if the function body uses Python
/// syntax outside the DSL subset.
pub fn convert_shape_dsl_function(
    func: &ruff_python_ast::StmtFunctionDef,
) -> Result<ShapeDslFunction, DslCompileError> {
    let fndef = convert_fndef(func)?;
    Ok(ShapeDslFunction {
        inner: Arc::new(fndef),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::ast::Ast;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::PySourceType;
    use ruff_python_ast::Stmt;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::shaped_array::ShapedArrayType;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::AnyStyle;
    use crate::types::TArgs;
    use crate::types::Union;
    use crate::types::Var;

    fn parse_dsl_functions(source: &str) -> Vec<ShapeDslFunction> {
        let (module, _, _) = Ast::parse(source, PySourceType::Stub);
        module
            .body
            .iter()
            .filter_map(|stmt| {
                if let Stmt::FunctionDef(func) = stmt {
                    Some(convert_shape_dsl_function(func).unwrap())
                } else {
                    None
                }
            })
            .collect()
    }

    fn fake_class(name: &str, module: &str) -> Class {
        let module = Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("1234567890".to_owned()),
        );
        Class::new(
            ClassDefIndex(0),
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            NestingContext::toplevel(),
            module,
            None,
            false,
        )
    }

    fn fake_intvar(name: &str) -> Type {
        let index = name.bytes().fold(0, |acc: u32, byte| {
            acc.wrapping_mul(16777619) ^ u32::from(byte)
        });
        Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), index),
                QuantifiedOrigin::Pep695,
            ),
            Name::new(name),
            QuantifiedKind::IntVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )))
    }

    #[test]
    fn test_val_to_scalar_type_uses_canonical_size_for_symbolic_dims() {
        let gradual = Type::Int(Int::Symbolic(Box::new(Type::Int(Int::Int))));
        assert_eq!(val_to_scalar_type(&Val::Dim(gradual)), Type::Int(Int::Int));

        let quantified = fake_intvar("N");
        assert_eq!(
            val_to_scalar_type(&Val::Dim(quantified)),
            Type::Int(Int::Symbolic(Box::new(fake_intvar("N"))))
        );

        assert_eq!(
            val_to_scalar_type(&Val::Dim(Type::Var(Var::ZERO))),
            Type::Int(Int::Symbolic(Box::new(Type::Var(Var::ZERO))))
        );

        let any = Type::Any(AnyStyle::Error);
        assert_eq!(
            val_to_scalar_type(&Val::Dim(any.clone())),
            Type::Int(Int::Symbolic(Box::new(any)))
        );
    }

    #[test]
    fn test_val_to_scalar_type_preserves_symbolic_size_arithmetic() {
        let n = Int::Symbolic(Box::new(fake_intvar("N")));
        let m = Int::Symbolic(Box::new(fake_intvar("M")));
        assert_eq!(
            val_to_scalar_type(&Val::Dim(Type::Int(Int::Mul(
                Box::new(n.clone()),
                Box::new(m.clone()),
            )))),
            Type::Int(Int::Mul(Box::new(m), Box::new(n)))
        );
    }

    #[test]
    fn test_val_to_scalar_type_preserves_concrete_dim_literals() {
        let literal = Lit::Int(LitInt::new(3)).to_implicit_type();
        assert_eq!(
            val_to_scalar_type(&Val::Dim(Type::Int(Int::Literal(3)))),
            literal
        );
    }

    #[test]
    fn test_extract_dim_list_accepts_int_tuple() {
        let shape = Type::IntTuple(Box::new(IntTuple::new(vec![
            Int::Literal(2),
            Int::Literal(3),
        ])));
        assert_eq!(
            extract::dim_list(&shape),
            Some(vec![Type::Int(Int::Literal(2)), Type::Int(Int::Literal(3)),])
        );
    }

    #[test]
    fn test_shape_dsl_extract_ignores_ordinary_torch_class_type() {
        let torch_tensor = Type::ClassType(ClassType::new(
            fake_class("Tensor", "torch"),
            TArgs::default(),
        ));
        let shape = IntTuple::new(vec![Int::Literal(2)]);

        assert!(extract::shaped_array_shape(&torch_tensor).is_none());
        assert!(
            extract::shaped_array_list(&Type::Tuple(Tuple::Concrete(vec![torch_tensor.clone()])))
                .is_none()
        );
        assert_eq!(inject_shape(shape, &torch_tensor), torch_tensor);
    }

    #[test]
    fn test_shape_dsl_extracts_registered_shaped_array_tuple() {
        let array = ClassType::new(fake_class("Array", "arrays"), TArgs::default());
        let first_shape = IntTuple::new(vec![Int::Literal(2)]);
        let second_shape = IntTuple::new(vec![Int::Literal(3)]);
        let first = ShapedArrayType::new(array.clone(), first_shape.clone()).to_type();
        let second = ShapedArrayType::new(array.clone(), second_shape.clone()).to_type();

        assert_eq!(
            extract::shaped_array_shape(&first),
            Some(first_shape.clone())
        );
        assert_eq!(
            extract::shaped_array_list(&Type::Tuple(Tuple::Concrete(vec![first, second]))),
            Some(vec![first_shape, second_shape.clone()])
        );
        assert_eq!(
            inject_shape(
                second_shape.clone(),
                &ShapedArrayType::shapeless(array.clone()).to_type()
            ),
            ShapedArrayType::new(array, second_shape).to_type()
        );
    }

    #[test]
    fn test_shape_dsl_extracts_same_shape_union() {
        let array = ClassType::new(fake_class("Array", "arrays"), TArgs::default());
        let shape = IntTuple::new(vec![Int::Literal(2)]);
        let other_shape = IntTuple::new(vec![Int::Literal(3)]);
        let union = Type::Union(Box::new(Union {
            members: vec![
                ShapedArrayType::new(array.clone(), shape.clone()).to_type(),
                ShapedArrayType::new(array.clone(), shape.clone()).to_type(),
            ],
            display_name: None,
        }));
        let mismatched_union = Type::Union(Box::new(Union {
            members: vec![
                ShapedArrayType::new(array.clone(), shape.clone()).to_type(),
                ShapedArrayType::new(array, other_shape).to_type(),
            ],
            display_name: None,
        }));

        assert_eq!(extract::shaped_array_shape(&union), Some(shape));
        assert!(extract::shaped_array_shape(&mismatched_union).is_none());
    }

    #[test]
    fn test_shaped_array_dsl_spelling_is_canonical() {
        let fns = parse_dsl_functions(
            r#"
def new_spelling(x: ShapedArray) -> ShapedArray:
    return ShapedArray(shape=x.shape)
"#,
        );

        let f = fns.iter().find(|f| f.name() == "new_spelling").unwrap();
        assert_eq!(f.inner.params[0].ty.to_string(), "ShapedArray");
        assert_eq!(
            f.inner.return_type.as_ref().unwrap().to_string(),
            "ShapedArray"
        );
        let DslBody::Return(expr) = &f.inner.body else {
            panic!("new_spelling should have a return body");
        };
        assert_eq!(expr.to_string(), "ShapedArray(shape=x.shape)");
    }

    #[test]
    fn test_tensor_dsl_spelling_is_rejected() {
        let (module, _, _) = Ast::parse(
            r#"
def legacy_spelling(x: Tensor) -> Tensor:
    return Tensor(shape=x.shape)
"#,
            PySourceType::Stub,
        );
        let Stmt::FunctionDef(func) = &module.body[0] else {
            panic!("expected a function");
        };

        let err = convert_shape_dsl_function(func).unwrap_err();
        assert!(
            err.message.contains("unknown type 'Tensor' in annotation")
                || err
                    .message
                    .contains("unknown function or constructor 'Tensor'"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn test_call_targets_disjoint_closures() {
        let fns = parse_dsl_functions(
            r#"
def helper_a(x: int) -> int:
    return x + 1

def helper_b(x: int) -> int:
    return x + 2

def calls_a(x: int) -> int:
    return helper_a(x)

def calls_b(x: int) -> int:
    return helper_b(x)

def leaf(x: int) -> int:
    return x
"#,
        );
        assert_eq!(fns.len(), 5);

        let calls_a = fns.iter().find(|f| f.name() == "calls_a").unwrap();
        let calls_b = fns.iter().find(|f| f.name() == "calls_b").unwrap();
        let leaf = fns.iter().find(|f| f.name() == "leaf").unwrap();

        assert_eq!(
            calls_a.call_targets(),
            HashSet::from(["helper_a".to_owned()])
        );
        assert_eq!(
            calls_b.call_targets(),
            HashSet::from(["helper_b".to_owned()])
        );
        assert!(leaf.call_targets().is_empty());
    }

    #[test]
    fn test_call_targets_transitive() {
        let fns = parse_dsl_functions(
            r#"
def deep(x: int) -> int:
    return x

def mid(x: int) -> int:
    return deep(x)

def top(x: int) -> int:
    return mid(x)
"#,
        );
        let top = fns.iter().find(|f| f.name() == "top").unwrap();
        let mid = fns.iter().find(|f| f.name() == "mid").unwrap();

        // call_targets is direct only, not transitive
        assert_eq!(top.call_targets(), HashSet::from(["mid".to_owned()]));
        assert_eq!(mid.call_targets(), HashSet::from(["deep".to_owned()]));
    }
}
