/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Special handling for Polars `DataFrame` construction.
//!
//! Polars' stubs type a `DataFrame` as one opaque blob, so we synthesize a
//! `Type::DataFrame` carrying an inferred column schema when a DataFrame is built
//! from a dict literal. This is the entry point for column-aware checking.

use std::sync::Arc;

use pyrefly_types::data_frame::DataFrameKind;
use pyrefly_types::data_frame::DataFrameSchema;
use pyrefly_types::data_frame::SchemaCompleteness;
use pyrefly_types::polars_dtype::PolarsDType;
use pyrefly_types::series::SeriesSchema;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use ruff_python_ast::Arguments;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Keyword;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FuncId;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::literal::Lit;

pub fn is_polars_dataframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.dataframe.frame", "DataFrame")
}

pub fn is_polars_series(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.series.series", "Series")
}

fn is_polars_expr(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.expr.expr", "Expr")
}

pub fn is_polars_lazyframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.lazyframe.frame", "LazyFrame")
}

/// The receiver schema for a column transform whose method takes only positional
/// arguments: `base` must carry a schema and `func` must name `method` with no
/// keywords. Shared preamble so each transform states only what is unique to it.
fn column_transform_schema<'b>(
    base: &'b Type,
    func: &ExprAttribute,
    method: &str,
    args: &Arguments,
) -> Option<&'b DataFrameSchema> {
    let Type::DataFrame(schema) = base else {
        return None;
    };
    (func.attr.id.as_str() == method && args.keywords.is_empty()).then_some(&**schema)
}

pub fn is_pandas_dataframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("pandas.core.frame", "DataFrame")
}

/// DataFrame methods whose arguments may refer to existing columns.
pub fn is_dataframe_column_method(method: &str) -> bool {
    matches!(
        method,
        "select" | "drop" | "with_columns" | "filter" | "sort" | "group_by" | "groupby"
    )
}

/// How an in-place mutation changes a Polars frame's tracked column set.
#[derive(Clone, Debug)]
pub enum PolarsMutationKind {
    /// Adds a column with an unknowable name, so every known column survives but exhaustiveness is lost.
    Add,
    /// Overwrites a column at an index we cannot map to a name.
    Replace,
    /// May insert a statically-known column name at a known index. The callee is resolved before the
    /// column set stays exhaustive.
    Insert(Name, usize, Box<Expr>),
}

/// The literal name, index, and unresolved callee of an `insert_column` candidate. The callee is kept
/// so schema application can verify it is `pl.Series`; `None` when the call shape is not static.
fn insert_column_spec(args: &Arguments) -> Option<(Name, usize, Box<Expr>)> {
    let [index_expr, column_expr] = &args.args[..] else {
        return None;
    };
    if !args.keywords.is_empty() {
        return None;
    }
    let Expr::NumberLiteral(ExprNumberLiteral {
        value: Number::Int(i),
        ..
    }) = index_expr
    else {
        return None;
    };
    let (name, callee) = series_literal_name(column_expr)?;
    Some((name, i.to_string().parse::<usize>().ok()?, callee))
}

/// The literal `name` and unresolved callee of a possible `pl.Series` call, or `None` when it is not a
/// call with a string-literal name. The callee is resolved when the mutation is applied.
fn series_literal_name(expr: &Expr) -> Option<(Name, Box<Expr>)> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let name = if let Some(kw) = call
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_ref().is_some_and(|a| a.id.as_str() == "name"))
    {
        match &kw.value {
            Expr::StringLiteral(s) => Name::new(s.value.to_str()),
            _ => return None,
        }
    } else {
        match call.arguments.args.first() {
            Some(Expr::StringLiteral(s)) => Name::new(s.value.to_str()),
            _ => return None,
        }
    };
    Some((name, call.func.clone()))
}

/// Classify an in-place column-mutation method, or `None` for anything else. `insert_column` with a
/// literal index and `pl.Series` name inserts that exact column; otherwise it only adds an unknowable
/// one. `hstack` counts only when an `in_place` keyword is present and not the literal `False`.
pub fn polars_column_mutation(method: &str, args: &Arguments) -> Option<PolarsMutationKind> {
    match method {
        "insert_column" => Some(match insert_column_spec(args) {
            Some((name, index, callee)) => PolarsMutationKind::Insert(name, index, callee),
            None => PolarsMutationKind::Add,
        }),
        "replace_column" => Some(PolarsMutationKind::Replace),
        "hstack"
            if args.keywords.iter().any(|kw| {
                kw.arg.as_ref().is_some_and(|a| a.id.as_str() == "in_place")
                    && !matches!(&kw.value, Expr::BooleanLiteral(b) if !b.value)
            }) =>
        {
            Some(PolarsMutationKind::Add)
        }
        _ => None,
    }
}

/// Degrade a Polars frame's schema for an in-place column mutation, or identity on any other type.
/// `Insert` adds the known column and stays exhaustive, `Add` keeps the known columns but drops
/// exhaustiveness, and `Replace` falls back to opaque.
pub fn polars_degrade_for_mutation(
    ty: &Type,
    kind: &PolarsMutationKind,
    is_polars_series: impl Fn(&Expr) -> bool,
) -> Type {
    let Type::DataFrame(schema) = ty else {
        return ty.clone();
    };
    if schema.kind != DataFrameKind::Polars {
        return ty.clone();
    }
    match kind {
        PolarsMutationKind::Replace => schema.underlying_type(),
        PolarsMutationKind::Insert(name, index, callee)
            if schema.is_complete() && is_polars_series(callee) =>
        {
            let mut columns = schema.columns.clone();
            columns.insert(
                (*index).min(columns.len()),
                (name.clone(), PolarsDType::Unknown),
            );
            DataFrameSchema {
                columns,
                ..(**schema).clone()
            }
            .to_type()
        }
        PolarsMutationKind::Add | PolarsMutationKind::Insert(..) if schema.is_complete() => {
            DataFrameSchema {
                completeness: SchemaCompleteness::Partial,
                ..(**schema).clone()
            }
            .to_type()
        }
        PolarsMutationKind::Add | PolarsMutationKind::Insert(..) => ty.clone(),
    }
}

/// Whether `ty` is a Polars `DataFrame`, schema-carrying or an opaque class instance.
fn is_polars_dataframe_type(ty: &Type) -> bool {
    match ty {
        Type::DataFrame(schema) => schema.kind == DataFrameKind::Polars,
        Type::ClassType(ct) => is_polars_dataframe(ct.class_object()),
        _ => false,
    }
}

/// Map a Python scalar value type to its Polars construction dtype, e.g. `int` to `Int64`. A `bool`
/// is exact and distinct from `int`. Returns `None` for any other type so the caller degrades.
fn polars_dtype_from_scalar_type(ty: &Type) -> Option<PolarsDType> {
    // Integers past i64 have a data-shape-dependent runtime dtype, so do not claim Int64.
    if let Type::Literal(lit) = ty {
        return Some(match &lit.value {
            Lit::Int(i) => return i.as_i64().map(|_| PolarsDType::Int64),
            Lit::Bool(_) => PolarsDType::Boolean,
            Lit::Str(_) => PolarsDType::String,
            Lit::Bytes(_) => PolarsDType::Binary,
            Lit::Enum(_) => return None,
        });
    }
    let Type::ClassType(cls) = ty else {
        return None;
    };
    Some(if cls.is_builtin("bool") {
        PolarsDType::Boolean
    } else if cls.is_builtin("int") {
        PolarsDType::Int64
    } else if cls.is_builtin("float") {
        PolarsDType::Float64
    } else if cls.is_builtin("str") {
        PolarsDType::String
    } else if cls.is_builtin("bytes") {
        PolarsDType::Binary
    } else {
        return None;
    })
}

/// Map a resolved Polars dtype class or instance to its modeled dtype, or `None` otherwise.
fn polars_dtype_from_type(ty: &Type) -> Option<PolarsDType> {
    let cls = match ty {
        Type::ClassDef(cls) => cls,
        Type::ClassType(cls) => cls.class_object(),
        _ => return None,
    };
    let module = cls.module_name();
    if module.as_str() != "polars" && !module.as_str().starts_with("polars.") {
        return None;
    }
    Some(match cls.name().as_str() {
        "Int8" => PolarsDType::Int8,
        "Int16" => PolarsDType::Int16,
        "Int32" => PolarsDType::Int32,
        "Int64" => PolarsDType::Int64,
        "Int128" => PolarsDType::Int128,
        "UInt8" => PolarsDType::UInt8,
        "UInt16" => PolarsDType::UInt16,
        "UInt32" => PolarsDType::UInt32,
        "UInt64" => PolarsDType::UInt64,
        "UInt128" => PolarsDType::UInt128,
        "Float32" => PolarsDType::Float32,
        "Float64" => PolarsDType::Float64,
        "Boolean" => PolarsDType::Boolean,
        "String" => PolarsDType::String,
        "Binary" => PolarsDType::Binary,
        "Date" => PolarsDType::Date,
        "Datetime" => PolarsDType::Datetime,
        "Duration" => PolarsDType::Duration,
        "Time" => PolarsDType::Time,
        _ => return None,
    })
}

/// Recognize module-level `polars.concat` through callable wrappers.
pub fn is_polars_concat(callee: &Type) -> bool {
    matches!(
        callee.callee_kind(),
        Some(CalleeKind::Function(FunctionKind::Def(id)))
            if id.name.as_str() == "concat"
                && id.module.name().as_str() == "polars.functions.eager"
    )
}

/// The two `pl.concat` strategies column inference models. `Vertical` requires identical schemas;
/// `VerticalRelaxed` requires the same ordered names and folds `supertype()` per column.
#[derive(Clone, Copy)]
enum ConcatHow {
    Vertical,
    VerticalRelaxed,
}

/// A dict-of-columns literal with its resolved columns and source range.
#[derive(Clone)]
struct PolarsDictData<'b> {
    columns: SmallMap<Name, &'b Expr>,
    range: TextRange,
}

/// The data shapes column inference models: a dict-of-columns literal, list-of-dict rows, or a value
/// whose type is a TypedDict, whose already-resolved columns ride inline.
#[derive(Clone)]
enum PolarsData<'b> {
    Dict(PolarsDictData<'b>),
    Records(SmallMap<Name, Vec<&'b Expr>>),
    TypedDict(Vec<(Name, PolarsDType)>, SchemaCompleteness),
}

/// A DataFrame constructor call reduced to the pieces column inference needs.
/// `data` is `None` when absent or empty; `schema` is the ordered `schema=` column set, each
/// column pinned to a dtype or `None` to defer to data inference; `columns` is the pandas
/// `columns=` selection, applied only for a pandas frame.
pub struct PolarsConstruct<'b> {
    data: Option<PolarsData<'b>>,
    schema: Option<Vec<(Name, Option<PolarsDType>)>>,
    columns: Option<Vec<Name>>,
    overrides: SmallMap<Name, PolarsDType>,
    strict: bool,
}

/// A `pl.Series(...)` call reduced to what element inference needs. The name does not affect the
/// dtype, so only the `values` expression, an optional `dtype=` override, and `strict` are kept.
struct SeriesConstruct<'b> {
    values: Option<&'b Expr>,
    dtype: Option<PolarsDType>,
    strict: bool,
}

/// A data dict literal as an order-preserving name-to-value map, or `None` if a key is not a
/// string literal or repeats (Python keeps only the last value for a repeated key).
fn dataframe_data_map(dict: &ExprDict) -> Option<SmallMap<Name, &Expr>> {
    let mut map = SmallMap::with_capacity(dict.items.len());
    for item in &dict.items {
        let Some(Expr::StringLiteral(key)) = &item.key else {
            return None;
        };
        if map
            .insert(Name::new(key.value.to_str()), &item.value)
            .is_some()
        {
            return None;
        }
    }
    Some(map)
}

/// A list-of-dicts as an ordered column-to-per-row-values map (first-appearance order), or `None`
/// if not a record literal we model. Only the first 100 rows are read, matching Polars'
/// `infer_schema_length` default, so a key that first appears past row 100 is absent at runtime too.
fn dataframe_records_map(list: &ExprList) -> Option<SmallMap<Name, Vec<&Expr>>> {
    let mut columns: SmallMap<Name, Vec<&Expr>> = SmallMap::new();
    for elt in list.elts.iter().take(100) {
        let Expr::Dict(dict) = elt else {
            return None;
        };
        for (name, value) in dataframe_data_map(dict)? {
            columns.entry(name).or_default().push(value);
        }
    }
    (!columns.is_empty()).then_some(columns)
}

/// How a `schema=` was written: a bare dict literal treats a `None` value as "defer to data
/// inference", but an inline `pl.Schema({...})` forbids `None` (runtime error). They diverge on `None`.
#[derive(Clone, Copy, PartialEq)]
enum SchemaForm {
    Dict,
    SchemaClass,
}

/// The join strategies column inference models. They differ only in which key columns survive and
/// which name overlaps get suffixed.
#[derive(Clone, Copy)]
enum JoinHow {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
    Cross,
}

impl JoinHow {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "inner" => Self::Inner,
            "left" => Self::Left,
            "right" => Self::Right,
            "full" => Self::Full,
            "semi" => Self::Semi,
            "anti" => Self::Anti,
            "cross" => Self::Cross,
            _ => return None,
        })
    }

    /// Whether a paired key coalesces into one primary-side column by default, dropping the other
    /// side's key. `full` and `cross` keep every key; `semi`/`anti` output only the left columns.
    fn coalesces(self) -> bool {
        matches!(self, Self::Inner | Self::Left | Self::Right)
    }
}

/// Return a list or tuple literal's elements, or `arg` as a one-element slice.
fn unpack_list_or_tuple_literal(arg: &Expr) -> &[Expr] {
    match arg {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        _ => std::slice::from_ref(arg),
    }
}

/// Whether a `pl.col` name is a selector (`"*"` or `"^regex$"`) rather than an exact name.
fn is_polars_selector_name(name: &Name) -> bool {
    let value = name.as_str();
    value == "*" || (value.starts_with('^') && value.ends_with('$'))
}

/// A resolved Polars expression, either a pinned dtype or a flexible numeric literal that adapts to
/// the operand it meets. Only a literal stays flexible.
#[derive(Clone, Copy)]
enum ExprValue {
    Dtype(PolarsDType),
    IntLit(i128),
    FloatLit,
}

/// A column argument classified as one exact name, a data-dependent string, or a Polars expression.
enum ColumnArg {
    Named(Name),
    Opaque,
    Expr,
}

impl ExprValue {
    /// The dtype a value materializes to as a standalone column. A float literal is `Float64`, and an
    /// integer literal takes the narrowest signed width that holds it.
    fn dtype(self) -> PolarsDType {
        match self {
            ExprValue::Dtype(d) => d,
            ExprValue::FloatLit => PolarsDType::Float64,
            ExprValue::IntLit(v) => {
                if i32::try_from(v).is_ok() {
                    PolarsDType::Int32
                } else if i64::try_from(v).is_ok() {
                    PolarsDType::Int64
                } else {
                    PolarsDType::Int128
                }
            }
        }
    }

    fn is_numeric(self) -> bool {
        match self {
            ExprValue::IntLit(_) | ExprValue::FloatLit => true,
            ExprValue::Dtype(d) => d.is_numeric(),
        }
    }

    fn is_integer(self) -> bool {
        match self {
            ExprValue::IntLit(_) => true,
            ExprValue::FloatLit => false,
            ExprValue::Dtype(d) => d.is_integer(),
        }
    }
}

/// A Python scalar literal as an `ExprValue`, or `None` for a non-literal. A numeric literal is
/// flexible, every other literal pins a dtype, and an integer past `i128` falls back like the constructor.
fn literal_value(expr: &Expr) -> Option<ExprValue> {
    match expr {
        Expr::NumberLiteral(ExprNumberLiteral {
            value: Number::Int(i),
            ..
        }) => i.to_string().parse::<i128>().ok().map(ExprValue::IntLit),
        Expr::NumberLiteral(ExprNumberLiteral {
            value: Number::Float(_),
            ..
        }) => Some(ExprValue::FloatLit),
        Expr::BooleanLiteral(_) => Some(ExprValue::Dtype(PolarsDType::Boolean)),
        Expr::StringLiteral(_) => Some(ExprValue::Dtype(PolarsDType::String)),
        Expr::BytesLiteral(_) => Some(ExprValue::Dtype(PolarsDType::Binary)),
        Expr::NoneLiteral(_) => Some(ExprValue::Dtype(PolarsDType::Null)),
        _ => None,
    }
}

/// A `Series`-returning DataFrame method, modeled only on a Complete Polars schema where a column
/// proves its exact dtype. `None` for a Partial schema or a pandas frame.
fn series_method_schema<'b>(
    base: &'b Type,
    func: &ExprAttribute,
    method: &str,
) -> Option<&'b DataFrameSchema> {
    let Type::DataFrame(schema) = base else {
        return None;
    };
    (func.attr.id.as_str() == method
        && schema.kind == DataFrameKind::Polars
        && schema.completeness == SchemaCompleteness::Complete)
        .then_some(&**schema)
}

/// The sole column-name argument of a `get_column` call, positional or `name=`, or `None` to fall
/// back. A `default=`, a name supplied both ways, extra positionals, or any other keyword falls back.
fn get_column_name_arg(args: &Arguments) -> Option<&Expr> {
    let mut name_keyword = None;
    for kw in &args.keywords {
        match kw.arg.as_ref()?.id.as_str() {
            "name" => name_keyword = Some(&kw.value),
            _ => return None,
        }
    }
    let name_positional = match &args.args[..] {
        [] => None,
        [name] => Some(name),
        _ => return None,
    };
    match (name_positional, name_keyword) {
        (Some(_), Some(_)) => None,
        (e, None) | (None, e) => e,
    }
}

/// The static index of a `to_series` call. An absent argument means `0`, otherwise one integer
/// literal, positional or `index=`. A non-integer, extra positionals, both ways, or any other keyword
/// falls back with `None`.
fn to_series_index(args: &Arguments) -> Option<i128> {
    let mut index_keyword = None;
    for kw in &args.keywords {
        match kw.arg.as_ref()?.id.as_str() {
            "index" => index_keyword = Some(&kw.value),
            _ => return None,
        }
    }
    let index_positional = match &args.args[..] {
        [] => None,
        [index] => Some(index),
        _ => return None,
    };
    match (index_positional, index_keyword) {
        (Some(_), Some(_)) => None,
        (None, None) => Some(0),
        (Some(e), None) | (None, Some(e)) => integer_literal(e),
    }
}

/// An integer literal's value, possibly behind a unary `+`/`-`, or `None` otherwise. Negative
/// indexing means `-1` must resolve to its signed value.
fn integer_literal(expr: &Expr) -> Option<i128> {
    match expr {
        Expr::NumberLiteral(ExprNumberLiteral {
            value: Number::Int(i),
            ..
        }) => i.to_string().parse::<i128>().ok(),
        Expr::UnaryOp(u) if u.op == UnaryOp::USub => integer_literal(&u.operand).map(|v| -v),
        Expr::UnaryOp(u) if u.op == UnaryOp::UAdd => integer_literal(&u.operand),
        _ => None,
    }
}

/// The dtype of a column read by name against a receiver schema, or `None` when it cannot resolve.
/// Reports `UnknownColumn` at `range` only on a `Complete` schema, since a `Partial` one may hold the
/// name untracked.
fn resolve_column(
    schema: &DataFrameSchema,
    name: &Name,
    range: TextRange,
    errors: &ErrorCollector,
) -> Option<PolarsDType> {
    match schema.columns.iter().find(|(c, _)| c == name) {
        Some((_, ty)) => Some(*ty),
        None => {
            if schema.completeness == SchemaCompleteness::Complete {
                errors
                    .error_builder(
                        range,
                        ErrorKind::UnknownColumn,
                        format!("Column `{name}` is not in the DataFrame schema"),
                    )
                    .emit();
            }
            None
        }
    }
}

/// Combine two numeric operands under a supertype-forming operator. A flexible integer literal keeps
/// a narrower integer column's dtype only when its value fits that column's range.
fn arith(a: ExprValue, b: ExprValue) -> Option<ExprValue> {
    use ExprValue::*;
    match (a, b) {
        (Dtype(da), Dtype(db)) if da.is_numeric() && db.is_numeric() => da.supertype(db).map(Dtype),
        (Dtype(_), Dtype(_)) => None,
        (Dtype(d), IntLit(v)) | (IntLit(v), Dtype(d)) => int_lit_with_dtype(d, v),
        (Dtype(d), FloatLit) | (FloatLit, Dtype(d)) => float_lit_with_dtype(d),
        (IntLit(_), IntLit(_)) => a.dtype().supertype(b.dtype()).map(Dtype),
        (IntLit(_), FloatLit) | (FloatLit, IntLit(_)) | (FloatLit, FloatLit) => {
            Some(Dtype(PolarsDType::Float64))
        }
    }
}

/// A flexible integer literal keeps a float column's dtype, and an integer column's dtype only when
/// the value fits its range. Anything else falls back.
fn int_lit_with_dtype(d: PolarsDType, v: i128) -> Option<ExprValue> {
    if d.is_float() {
        return Some(ExprValue::Dtype(d));
    }
    match d.int_bounds() {
        Some((lo, hi)) if (lo..=hi).contains(&v) => Some(ExprValue::Dtype(d)),
        _ => None,
    }
}

/// A flexible float literal keeps `Float32` and widens a `Float64` or integer column to `Float64`. A
/// boolean or non-numeric column falls back.
fn float_lit_with_dtype(d: PolarsDType) -> Option<ExprValue> {
    use PolarsDType::*;
    match d {
        Float32 => Some(ExprValue::Dtype(Float32)),
        Float64 => Some(ExprValue::Dtype(Float64)),
        d if d.is_integer() => Some(ExprValue::Dtype(Float64)),
        _ => None,
    }
}

/// Resolve a binary operator over two resolved operands, or `None` when the result is data-dependent
/// or the operands are incompatible. Division promotes an integer result to `Float64`, bitwise ops
/// act on two booleans or two integers, and the rest form the numeric supertype.
fn combine_binop(op: Operator, a: ExprValue, b: ExprValue) -> Option<ExprValue> {
    use ExprValue::*;
    match op {
        Operator::Div => {
            let result = arith(a, b)?;
            Some(if result.is_integer() {
                Dtype(PolarsDType::Float64)
            } else {
                result
            })
        }
        Operator::BitAnd | Operator::BitOr | Operator::BitXor => {
            if matches!(
                (a, b),
                (Dtype(PolarsDType::Boolean), Dtype(PolarsDType::Boolean))
            ) {
                Some(Dtype(PolarsDType::Boolean))
            } else if a.is_integer() && b.is_integer() {
                arith(a, b)
            } else {
                None
            }
        }
        Operator::Add
        | Operator::Sub
        | Operator::Mult
        | Operator::FloorDiv
        | Operator::Mod
        | Operator::Pow => arith(a, b),
        Operator::LShift | Operator::RShift | Operator::MatMult => None,
    }
}

/// Resolve a unary operator over a resolved operand. Negation keeps a signed-int or float dtype and
/// negates a literal but falls back for an unsigned column. `~` yields `Boolean` for a boolean and
/// the operand's dtype for an integer.
fn unary_value(op: UnaryOp, a: ExprValue) -> Option<ExprValue> {
    use ExprValue::*;
    match op {
        UnaryOp::USub | UnaryOp::UAdd => match a {
            IntLit(v) if op == UnaryOp::USub => v.checked_neg().map(IntLit),
            IntLit(v) => Some(IntLit(v)),
            FloatLit => Some(FloatLit),
            Dtype(d) if d.is_signed_int() || d.is_float() => Some(Dtype(d)),
            Dtype(_) => None,
        },
        UnaryOp::Invert => match a {
            Dtype(PolarsDType::Boolean) => Some(Dtype(PolarsDType::Boolean)),
            v if v.is_integer() => Some(Dtype(v.dtype())),
            _ => None,
        },
        UnaryOp::Not => None,
    }
}

/// A comparison yields `Boolean` when both operands are comparable, meaning both numeric or the same
/// pinned dtype. Otherwise it falls back rather than fabricate a `Boolean` the runtime would reject.
fn comparison_value(a: ExprValue, b: ExprValue) -> Option<ExprValue> {
    use ExprValue::*;
    let comparable =
        (a.is_numeric() && b.is_numeric()) || matches!((a, b), (Dtype(x), Dtype(y)) if x == y);
    comparable.then_some(Dtype(PolarsDType::Boolean))
}

#[derive(Clone, Copy)]
enum Reducer {
    Identity,
    FloatPromote,
    Count,
    Sum,
    Product,
}

impl Reducer {
    fn parse(method: &str) -> Option<Self> {
        Some(match method {
            "min" | "max" | "first" | "last" => Self::Identity,
            "mean" | "median" | "std" | "var" => Self::FloatPromote,
            "count" | "n_unique" => Self::Count,
            "sum" => Self::Sum,
            "product" => Self::Product,
            _ => return None,
        })
    }

    /// Polars' `Int128` and `UInt128` sum/product results have not been runtime-verified.
    fn output_dtype(self, d: PolarsDType) -> Option<PolarsDType> {
        use PolarsDType::*;
        match self {
            Reducer::Identity => Some(d),
            Reducer::Count => Some(UInt32),
            Reducer::FloatPromote => match d {
                Float32 => Some(Float32),
                Boolean => Some(Float64),
                d if d.is_numeric() => Some(Float64),
                _ => None,
            },
            Reducer::Sum => match d {
                Boolean => Some(UInt32),
                Int8 | Int16 | UInt8 | UInt16 => Some(Int64),
                Int32 | Int64 | UInt32 | UInt64 | Float32 | Float64 => Some(d),
                _ => None,
            },
            Reducer::Product => match d {
                UInt64 | Float32 | Float64 => Some(d),
                Boolean | Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 => Some(Int64),
                _ => None,
            },
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Map a `pl.<DType>` expression such as `pl.Int8` or `pl.Datetime("us")` to its dtype, or `None`
    /// to fall back. The class must resolve into the `polars` package, so an unrelated symbol reusing a
    /// dtype name like `other.Float64` is not mistaken for one.
    fn polars_dtype_from_expr(&self, e: &Expr) -> Option<PolarsDType> {
        // Swallow errors here, since the fallback call path re-infers this and is the sole reporter.
        let ty = self.expr_infer(e, &self.error_swallower());
        polars_dtype_from_type(&ty)
    }

    /// Classify a column argument without reporting inference errors.
    fn polars_column_arg(&self, expr: &Expr) -> ColumnArg {
        let ty = self.expr_infer(expr, &self.error_swallower());
        if let Type::Literal(lit) = &ty
            && let Lit::Str(value) = &lit.value
        {
            let value = value.as_str();
            // A `"*"` or `"^regex$"` string is a `pl.col` selector, not an exact name.
            if value == "*" || (value.starts_with('^') && value.ends_with('$')) {
                return ColumnArg::Opaque;
            }
            return ColumnArg::Named(Name::new(value));
        }
        // Any other string type is a name or selector we cannot pin to a single column.
        if ty.is_literal_string() || polars_dtype_from_scalar_type(&ty) == Some(PolarsDType::String)
        {
            return ColumnArg::Opaque;
        }
        ColumnArg::Expr
    }

    /// A statically known column name. Selector-shaped strings remain literal in name-only contexts.
    pub fn polars_column_name(&self, expr: &Expr) -> Option<Name> {
        let ty = self.expr_infer(expr, &self.error_swallower());
        if let Type::Literal(lit) = &ty
            && let Lit::Str(value) = &lit.value
        {
            return Some(Name::new(value.as_str()));
        }
        None
    }

    /// Reduce a `pl.Series(...)` call to a `SeriesConstruct`, or `None` to fall back to the opaque
    /// Series. Positional slots are `(name, values, dtype)`, but a non-string first slot is itself the
    /// values. An ambiguous name slot, a positional-plus-keyword clash, or an unrecognized keyword
    /// falls back, while `name` and `nan_to_null` are tolerated.
    fn polars_series_options<'b>(&self, arguments: &'b Arguments) -> Option<SeriesConstruct<'b>> {
        let mut values_keyword: Option<&Expr> = None;
        let mut dtype_keyword: Option<&Expr> = None;
        let mut strict = true;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "name" | "nan_to_null" => {}
                "values" => values_keyword = Some(&kw.value),
                "dtype" => dtype_keyword = Some(&kw.value),
                "strict" => match &kw.value {
                    Expr::BooleanLiteral(b) => strict = b.value,
                    _ => return None,
                },
                _ => return None,
            }
        }
        let (values_positional, dtype_positional) = match &arguments.args[..] {
            [] => (None, None),
            [Expr::StringLiteral(_)] => (None, None),
            [values] => (Some(values), None),
            [Expr::StringLiteral(_), values] => (Some(values), None),
            [Expr::StringLiteral(_), values, dtype] => (Some(values), Some(dtype)),
            _ => return None,
        };
        let values = match (values_positional, values_keyword) {
            (Some(_), Some(_)) => return None,
            (e, None) | (None, e) => e,
        };
        let dtype = match (dtype_positional, dtype_keyword) {
            (Some(_), Some(_)) => return None,
            (Some(e), None) | (None, Some(e)) => Some(self.polars_dtype_from_expr(e)?),
            (None, None) => None,
        };
        Some(SeriesConstruct {
            values,
            dtype,
            strict,
        })
    }

    /// A `schema=` dict literal as an ordered column list, each pinned to a `pl.<DType>` or `None` to
    /// defer to data inference. Falls back for an empty dict, an unrecognized entry, or a `None` under
    /// the `pl.Schema` form, which forbids it.
    fn schema_dict_entries(
        &self,
        form: SchemaForm,
        dict: &ExprDict,
    ) -> Option<Vec<(Name, Option<PolarsDType>)>> {
        if dict.items.is_empty() {
            return None;
        }
        let mut entries = Vec::with_capacity(dict.items.len());
        let mut seen = SmallSet::new();
        for item in &dict.items {
            let Some(Expr::StringLiteral(key)) = &item.key else {
                return None;
            };
            let name = Name::new(key.value.to_str());
            if !seen.insert(name.clone()) {
                return None;
            }
            let dtype = match &item.value {
                Expr::NoneLiteral(_) if form == SchemaForm::Dict => None,
                Expr::NoneLiteral(_) => return None,
                value => Some(self.polars_dtype_from_expr(value)?),
            };
            entries.push((name, dtype));
        }
        Some(entries)
    }

    /// Build the `Type::DataFrame` for a `pl.DataFrame[Schema]` annotation, or `None` when `base` is
    /// not the Polars DataFrame class or `arg` is not a schema class. The schema is Complete because
    /// an annotation states the full column set, and it stands for a real DataFrame instance so
    /// unmodeled behavior still resolves against the class.
    pub fn polars_dataframe_schema_annotation(&self, base: &Class, arg: &Expr) -> Option<Type> {
        if !is_polars_dataframe(base) {
            return None;
        }
        // Swallow errors here, since a bad annotation is reported by the ordinary subscript path.
        let ty = self.expr_infer(arg, &self.error_swallower());
        let Type::ClassDef(schema_cls) = &ty else {
            return None;
        };
        let columns = self.schema_class_columns(schema_cls)?;
        let Type::ClassType(underlying) = self.promote_silently(base) else {
            return None;
        };
        Some(
            DataFrameSchema {
                underlying,
                columns,
                completeness: SchemaCompleteness::Complete,
                kind: DataFrameKind::Polars,
            }
            .to_type(),
        )
    }

    /// Read a class named by `schema=` as an ordered column list, one column per `field: pl.<DType>`
    /// annotation, the way patito and dataframely model classes declare a frame's schema. A field
    /// whose annotation is not a modeled Polars dtype falls back, so a non-schema class is never misread.
    fn schema_class_entries(&self, expr: &Expr) -> Option<Vec<(Name, Option<PolarsDType>)>> {
        // Swallow errors here, since the fallback call path re-infers this and is the sole reporter.
        let ty = self.expr_infer(expr, &self.error_swallower());
        let Type::ClassDef(cls) = &ty else {
            return None;
        };
        Some(
            self.schema_class_columns(cls)?
                .into_iter()
                .map(|(name, dtype)| (name, Some(dtype)))
                .collect(),
        )
    }

    /// Read a schema class's columns as `(name, dtype)` in declaration order, one per
    /// `field: pl.<DType>` annotation. A field whose annotation is not a modeled Polars dtype
    /// falls back, so a non-schema class is never misread as an empty or partial schema.
    fn schema_class_columns(&self, cls: &Class) -> Option<Vec<(Name, PolarsDType)>> {
        let fields = self.get_class_field_map(cls);
        if fields.is_empty() {
            return None;
        }
        let mut columns = Vec::with_capacity(fields.len());
        for (name, field) in &fields {
            columns.push((name.clone(), polars_dtype_from_type(&field.ty())?));
        }
        Some(columns)
    }

    /// Read the required columns of a `data=` value whose type is a TypedDict. Each required field
    /// must be a `Sequence` of a primitive scalar, unwrapped to the column dtype (`Sequence[int]`
    /// gives Int64). Optional fields are omitted and make the schema Partial because they may be
    /// present at runtime. A non-TypedDict value, an empty TypedDict, or an unmodeled required field
    /// falls back with `None`.
    fn typed_dict_data_columns(
        &self,
        expr: &Expr,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        // Swallow errors here, since the fallback call path re-infers this and is the sole reporter.
        let ty = self.expr_infer(expr, &self.error_swallower());
        let Type::TypedDict(typed_dict) = &ty else {
            return None;
        };
        let fields = self.typed_dict_fields(typed_dict);
        if fields.is_empty() {
            return None;
        }
        let completeness = if fields.values().all(|field| field.required) {
            SchemaCompleteness::Complete
        } else {
            SchemaCompleteness::Partial
        };
        let sequence = self.stdlib.sequence(Type::any_implicit());
        let mut columns = Vec::with_capacity(fields.len());
        for (name, field) in fields.iter().filter(|(_, field)| field.required) {
            let Type::ClassType(cls) = &field.ty else {
                return None;
            };
            let sequence = self
                .type_order()
                .as_superclass(cls, sequence.class_object())?;
            let [element] = sequence.targs().as_slice() else {
                return None;
            };
            columns.push((name.clone(), polars_dtype_from_scalar_type(element)?));
        }
        Some((columns, completeness))
    }

    /// Infer a column schema for a DataFrame constructor call, or `None` to fall back to plain
    /// construction. Purely syntactic; never infers the element expressions.
    ///
    /// With `schema=` the column set is authoritative and ordered: each column takes its
    /// `schema_overrides` dtype, else its schema dtype, else defers to data inference. Data must name
    /// the same columns, else we report the mismatch and fall back. Without `schema=`, data order
    /// defines the columns.
    pub fn infer_dataframe_schema(
        &self,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
        errors: &ErrorCollector,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        // `columns=` is a pandas-only selector; a Polars call with it is a runtime error, so fall back.
        if construct.columns.is_some() && kind != DataFrameKind::Pandas {
            return None;
        }
        match (&construct.data, &construct.schema) {
            // Record schemas do not follow the exact-name rules for column-oriented dict data.
            (Some(PolarsData::Records(_)), Some(_)) => None,
            (Some(PolarsData::Records(records)), None) => {
                self.infer_dataframe_records_schema(records, construct, kind, errors)
            }
            // TypedDict data carries resolved columns, but combining it with `schema=` is not modeled.
            (Some(PolarsData::TypedDict(_, _)), Some(_)) => None,
            (Some(PolarsData::TypedDict(columns, completeness)), None) => {
                self.infer_dataframe_typed_dict_schema(columns, completeness, construct, kind)
            }
            (Some(PolarsData::Dict(data)), _) => {
                self.infer_dataframe_dict_schema(Some(data), construct, kind, errors)
            }
            (None, _) => self.infer_dataframe_dict_schema(None, construct, kind, errors),
        }
    }

    fn infer_dataframe_typed_dict_schema(
        &self,
        columns: &[(Name, PolarsDType)],
        completeness: &SchemaCompleteness,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        if kind != DataFrameKind::Polars {
            return None;
        }
        Some((
            columns
                .iter()
                .map(|(name, dtype)| {
                    (
                        name.clone(),
                        construct.overrides.get(name).copied().unwrap_or(*dtype),
                    )
                })
                .collect(),
            completeness.clone(),
        ))
    }

    /// Infer a Polars list-of-dicts after parsing it into per-column row values.
    fn infer_dataframe_records_schema(
        &self,
        records: &SmallMap<Name, Vec<&Expr>>,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
        errors: &ErrorCollector,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        if kind != DataFrameKind::Polars {
            return None;
        }
        Some((
            records
                .iter()
                .map(|(name, values)| {
                    let element = match construct.overrides.get(name) {
                        Some(dtype) => *dtype,
                        None => self
                            .dataframe_list_element_type(
                                name,
                                values.iter().copied(),
                                kind.clone(),
                                false,
                                errors,
                            )
                            .unwrap_or(PolarsDType::Unknown),
                    };
                    (name.clone(), element)
                })
                .collect(),
            SchemaCompleteness::Complete,
        ))
    }

    /// Infer column-oriented dict data, or a schema-only construction when `data` is absent.
    fn infer_dataframe_dict_schema(
        &self,
        data: Option<&PolarsDictData>,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
        errors: &ErrorCollector,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        let completeness = if kind == DataFrameKind::Polars {
            SchemaCompleteness::Complete
        } else {
            SchemaCompleteness::Partial
        };
        // Only list literals have modeled dtypes.
        let element_from_data = |name: &Name, value: &Expr| match value {
            Expr::List(ExprList { elts, .. }) => self.dataframe_list_element_type(
                name,
                elts.iter(),
                kind.clone(),
                construct.strict,
                errors,
            ),
            _ => None,
        };
        let Some(schema) = &construct.schema else {
            let data = data?;
            // A pandas `columns=` selects and orders the output columns; absent it, use data order.
            let names: Vec<&Name> = match &construct.columns {
                Some(cols) => cols.iter().collect(),
                None => data.columns.keys().collect(),
            };
            let mut result = Vec::with_capacity(names.len());
            for name in names {
                // A `columns=` name absent from the data is an all-NaN column; fall back rather than model it.
                let value = data.columns.get(name).copied()?;
                let element = if let Some(dtype) = construct.overrides.get(name) {
                    *dtype
                } else {
                    match element_from_data(name, value) {
                        Some(dtype) => dtype,
                        None if kind == DataFrameKind::Polars => PolarsDType::Unknown,
                        None => return None,
                    }
                };
                result.push((name.clone(), element));
            }
            return Some((result, completeness));
        };
        if kind != DataFrameKind::Polars {
            return None;
        }
        // Declared schemas require the exact runtime column set.
        if let Some(data) = &data {
            let missing: Vec<&Name> = schema
                .iter()
                .map(|(n, _)| n)
                .filter(|n| !data.columns.contains_key(*n))
                .collect();
            let unexpected: Vec<&Name> = data
                .columns
                .keys()
                .filter(|n| !schema.iter().any(|(s, _)| s == *n))
                .collect();
            if !missing.is_empty() || !unexpected.is_empty() {
                let show = |ns: &[&Name]| {
                    ns.iter()
                        .map(|n| format!("`{n}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let detail = [
                    (!missing.is_empty()).then(|| format!("missing {}", show(&missing))),
                    (!unexpected.is_empty()).then(|| format!("unexpected {}", show(&unexpected))),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(", ");
                self.error(
                    errors,
                    data.range,
                    ErrorKind::ColumnSchemaMismatch,
                    format!("DataFrame data columns do not match the declared schema ({detail})"),
                );
                return None;
            }
        }
        let columns = schema
            .iter()
            .map(|(name, dtype)| {
                let element = if let Some(dtype) = construct.overrides.get(name) {
                    *dtype
                } else if let Some(dtype) = dtype {
                    *dtype
                } else {
                    match data.and_then(|d| d.columns.get(name).copied()) {
                        Some(value) => {
                            element_from_data(name, value).unwrap_or(PolarsDType::Unknown)
                        }
                        None => PolarsDType::Null,
                    }
                };
                (name.clone(), element)
            })
            .collect();
        Some((columns, completeness))
    }

    /// The single expression for a constructor parameter from its positional and keyword slots.
    /// Supplying both is a runtime error that ordinary call checking reports on the fallback path,
    /// so this returns `None` to defer to it rather than emit a second diagnostic.
    fn positional_or_keyword<'b>(
        positional: Option<&'b Expr>,
        keyword: Option<&'b Expr>,
    ) -> Option<Option<&'b Expr>> {
        match (positional, keyword) {
            (Some(_), Some(_)) => None,
            (e, None) | (None, e) => Some(e),
        }
    }

    /// The dict literal a `schema=` reduces to, tagged with its form: a bare dict literal or an inline
    /// call to `polars.schema.Schema`. A bound value, non-dict, or list form is not a static column set.
    fn schema_literal_dict<'b>(&self, expr: &'b Expr) -> Option<(SchemaForm, &'b ExprDict)> {
        match expr {
            Expr::Dict(dict) => Some((SchemaForm::Dict, dict)),
            Expr::Call(call) => {
                let [Expr::Dict(dict)] = &call.arguments.args[..] else {
                    return None;
                };
                if !call.arguments.keywords.is_empty() {
                    return None;
                }
                match self.expr_infer(&call.func, &self.error_swallower()) {
                    Type::ClassDef(cls) if cls.has_toplevel_qname("polars.schema", "Schema") => {
                        Some((SchemaForm::SchemaClass, dict))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The element dtype of a `pl.Series(...)` call, or `None` to fall back to the opaque Series. A
    /// `dtype=` override wins, a literal list/tuple `values` resolves through the DataFrame column
    /// fold, an absent `values` is `Null`, and a non-literal `values` falls back.
    pub fn infer_series_dtype(&self, arguments: &Arguments) -> Option<PolarsDType> {
        let construct = self.polars_series_options(arguments)?;
        if let Some(dtype) = construct.dtype {
            return Some(dtype);
        }
        let elts = match construct.values {
            None => return Some(PolarsDType::Null),
            Some(Expr::List(ExprList { elts, .. })) | Some(Expr::Tuple(ExprTuple { elts, .. })) => {
                elts
            }
            Some(_) => return None,
        };
        self.dataframe_list_element_type(
            &Name::new_static("values"),
            elts.iter(),
            DataFrameKind::Polars,
            construct.strict,
            &self.error_swallower(),
        )
    }

    /// A DataFrame constructor call reduced to the pieces column inference needs, or `None` to fall
    /// back to plain construction. `data` and `schema` each come from their positional slot or keyword,
    /// not both.
    pub fn polars_construct_options<'b>(
        &self,
        arguments: &'b Arguments,
    ) -> Option<PolarsConstruct<'b>> {
        let mut overrides = SmallMap::new();
        let mut strict = true;
        let mut data_keyword: Option<&Expr> = None;
        let mut schema_keyword: Option<&Expr> = None;
        let mut columns = None;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "data" => data_keyword = Some(&kw.value),
                "schema" => schema_keyword = Some(&kw.value),
                "columns" => {
                    let Expr::List(list) = &kw.value else {
                        return None;
                    };
                    let mut names = Vec::with_capacity(list.elts.len());
                    for elt in &list.elts {
                        let Expr::StringLiteral(s) = elt else {
                            return None;
                        };
                        let name = Name::new(s.value.to_str());
                        // Duplicate output columns are not modeled; fall back.
                        if names.contains(&name) {
                            return None;
                        }
                        names.push(name);
                    }
                    columns = Some(names);
                }
                "schema_overrides" => {
                    let Expr::Dict(dict) = &kw.value else {
                        return None;
                    };
                    for item in &dict.items {
                        let (Some(Expr::StringLiteral(key)), value) = (&item.key, &item.value)
                        else {
                            return None;
                        };
                        overrides.insert(
                            Name::new(key.value.to_str()),
                            self.polars_dtype_from_expr(value)?,
                        );
                    }
                }
                "strict" => match &kw.value {
                    Expr::BooleanLiteral(b) => strict = b.value,
                    _ => return None,
                },
                _ => return None,
            }
        }
        let (data_positional, schema_positional) = match &arguments.args[..] {
            [] => (None, None),
            [data] => (Some(data), None),
            [data, schema] => (Some(data), Some(schema)),
            _ => return None,
        };
        let data_expr = Self::positional_or_keyword(data_positional, data_keyword)?;
        let schema_expr = Self::positional_or_keyword(schema_positional, schema_keyword)?;
        let data = match data_expr {
            None | Some(Expr::NoneLiteral(_)) => None,
            Some(Expr::Dict(dict)) if dict.items.is_empty() => None,
            Some(Expr::Dict(dict)) => Some(PolarsData::Dict(PolarsDictData {
                columns: dataframe_data_map(dict)?,
                range: dict.range(),
            })),
            Some(Expr::List(list)) => Some(PolarsData::Records(dataframe_records_map(list)?)),
            // A non-literal `data=` whose type is a TypedDict names its columns through its fields.
            Some(expr) => {
                let (columns, completeness) = self.typed_dict_data_columns(expr)?;
                Some(PolarsData::TypedDict(columns, completeness))
            }
        };
        let schema = match schema_expr {
            None | Some(Expr::NoneLiteral(_)) => None,
            Some(expr) if let Some(entries) = self.schema_class_entries(expr) => Some(entries),
            Some(expr) => {
                let (form, dict) = self.schema_literal_dict(expr)?;
                Some(self.schema_dict_entries(form, dict)?)
            }
        };
        Some(PolarsConstruct {
            data,
            schema,
            columns,
            overrides,
            strict,
        })
    }

    /// The modeled `how=` value, defaulting to `Vertical` when absent, or `None` for a value we do not
    /// model or a `**spread` that could carry `how`, so the opaque frame is the safe fallback. The value
    /// is read from its inferred type, so a `Literal["vertical"]` variable resolves like a bare literal.
    fn polars_concat_how(&self, keywords: &[Keyword]) -> Option<ConcatHow> {
        let mut how = ConcatHow::Vertical;
        for kw in keywords {
            let arg = kw.arg.as_ref()?;
            if arg.id.as_str() == "how" {
                // Swallow errors here, since the fallback call path re-infers this and is the sole reporter.
                let ty = self.expr_infer(&kw.value, &self.error_swallower());
                let Type::Literal(lit) = &ty else {
                    return None;
                };
                let Lit::Str(value) = &lit.value else {
                    return None;
                };
                how = match value.as_str() {
                    "vertical" => ConcatHow::Vertical,
                    "vertical_relaxed" => ConcatHow::VerticalRelaxed,
                    _ => return None,
                };
            }
        }
        Some(how)
    }

    /// Infers literal vertical concatenations; any schema or supertype mismatch falls back.
    pub fn infer_polars_concat(&self, arguments: &Arguments) -> Option<Vec<(Name, PolarsDType)>> {
        let [items] = &arguments.args[..] else {
            return None;
        };
        let elts = match items {
            Expr::List(list) => &list.elts,
            Expr::Tuple(tuple) => &tuple.elts,
            _ => return None,
        };
        let how = self.polars_concat_how(&arguments.keywords)?;
        let schemas = elts
            .iter()
            .map(|e| match self.expr_infer(e, &self.error_swallower()) {
                Type::DataFrame(schema) => Some(schema.columns),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let (first, rest) = schemas.split_first()?;
        match how {
            ConcatHow::Vertical => rest
                .iter()
                .all(|columns| columns == first)
                .then(|| first.clone()),
            ConcatHow::VerticalRelaxed => {
                let names_match = rest.iter().all(|columns| {
                    columns.len() == first.len()
                        && columns.iter().zip(first).all(|((n, _), (m, _))| n == m)
                });
                if !names_match {
                    return None;
                }
                first
                    .iter()
                    .enumerate()
                    .map(|(i, (name, dtype))| {
                        let folded = rest
                            .iter()
                            .try_fold(*dtype, |acc, columns| acc.supertype(columns[i].1))?;
                        Some((name.clone(), folded))
                    })
                    .collect()
            }
        }
    }

    /// Anchors on the first non-null element; only Polars reports later mismatches.
    fn dataframe_list_element_type<'e>(
        &self,
        name: &Name,
        elts: impl Iterator<Item = &'e Expr> + Clone,
        kind: DataFrameKind,
        strict: bool,
        errors: &ErrorCollector,
    ) -> Option<PolarsDType> {
        let scalar = |e: &Expr| {
            // A float literal's type is the plain `float`, indistinguishable from a `float` variable,
            // so a written float literal is the one element read from syntax to keep pandas floats pinned.
            if let Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Float(_),
                ..
            }) = e
            {
                return Some(PolarsDType::Float64);
            }
            let ty = self.expr_infer(e, &self.error_swallower());
            if matches!(ty, Type::Literal(_))
                && let Some(dtype) = polars_dtype_from_scalar_type(&ty)
            {
                return Some(dtype);
            }
            // Beyond a literal, pandas coerces mixed and null-bearing columns in ways we do not model
            // (int-with-`None` becomes float64), so only Polars pins the dtype of a non-literal element.
            if kind != DataFrameKind::Polars {
                return None;
            }
            // A datetime-constructor call resolves by its callee class, not its element type, because a
            // variable typed `date` may hold a `datetime` subclass at runtime and so does not pin the
            // dtype. Only a direct constructor call is specific enough.
            if let Expr::Call(call) = e {
                let temporal = match self.expr_infer(&call.func, &self.error_swallower()) {
                    Type::ClassDef(cls) if cls.has_toplevel_qname("datetime", "date") => {
                        Some(PolarsDType::Date)
                    }
                    Type::ClassDef(cls) if cls.has_toplevel_qname("datetime", "datetime") => {
                        Some(PolarsDType::Datetime)
                    }
                    Type::ClassDef(cls) if cls.has_toplevel_qname("datetime", "time") => {
                        Some(PolarsDType::Time)
                    }
                    Type::ClassDef(cls) if cls.has_toplevel_qname("datetime", "timedelta") => {
                        Some(PolarsDType::Duration)
                    }
                    _ => None,
                };
                if temporal.is_some() {
                    return temporal;
                }
            }
            if matches!(ty, Type::None) {
                return Some(PolarsDType::Null);
            }
            polars_dtype_from_scalar_type(&ty)
        };
        let mut rest = elts.clone();
        let Some(first) = rest.next() else {
            return Some(PolarsDType::Unknown);
        };
        if !strict {
            // strict=False widens to the elements' common supertype instead of erroring.
            let mut acc = scalar(first)?;
            let mut any_rest = false;
            for e in rest {
                any_rest = true;
                acc = acc.supertype(scalar(e)?)?;
            }
            // We do not model timezones, so a naive/tz-aware mix (which Polars rejects under
            // strict=False) is indistinguishable here; fall back rather than assert `Datetime`.
            if acc == PolarsDType::Datetime && any_rest {
                return None;
            }
            return Some(acc);
        }
        // Anchor to the first non-null element; a `None` never anchors and always fits
        // (`supertype(Null, anchor) == anchor`), so an all-null column stays `Null`.
        let mut column = PolarsDType::Null;
        for e in elts {
            let element = scalar(e)?;
            if column == PolarsDType::Null {
                column = element;
                continue;
            }
            // The element fits only if it coerces into the column dtype without widening it.
            if element.supertype(column) != Some(column) {
                if kind == DataFrameKind::Polars {
                    self.error(
                        errors,
                        e.range(),
                        ErrorKind::ColumnTypeMismatch,
                        format!(
                            "Polars builds column `{name}` with type `{column}` from its first non-null element, so a `{element}` element does not fit. Use one dtype for the column or pass an explicit `schema`.",
                        ),
                    );
                }
                return None;
            }
        }
        Some(column)
    }

    /// Narrow a schema to the columns named in a `df[[...]]` list literal, keeping list order.
    /// Falls back with `None` when an element does not resolve to a single column name or when a name
    /// repeats, since Polars rejects duplicate column selection at runtime. An absent name reports the
    /// same `UnknownColumn` error as a single-column read.
    pub fn polars_select_columns(
        &self,
        schema: &DataFrameSchema,
        elts: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let mut names = Vec::with_capacity(elts.len());
        let mut seen = SmallSet::new();
        for elt in elts {
            let name = self.polars_column_name(elt)?;
            if !seen.insert(name.clone()) {
                return None;
            }
            names.push((name, elt.range()));
        }
        // Committed to a schema, so report the argument errors swallowed by name resolution.
        for elt in elts {
            self.expr_infer(elt, errors);
        }
        let columns = names
            .into_iter()
            .filter_map(
                |(name, range)| match schema.columns.iter().find(|(c, _)| *c == name) {
                    Some((_, ty)) => Some((name, *ty)),
                    None => {
                        // Only a Complete schema can prove a column absent, since a Partial one may
                        // hold it untracked.
                        if schema.is_complete() {
                            errors
                                .error_builder(
                                    range,
                                    ErrorKind::UnknownColumn,
                                    format!("Column `{name}` is not in the DataFrame schema"),
                                )
                                .emit();
                        }
                        None
                    }
                },
            )
            .collect();
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Model `df.select(...)` as a new schema from the positional arguments in order, unpacking list
    /// and tuple literals. A string names a column, an expression contributes its output name and
    /// inferred dtype, and a lone `"*"` keeps the schema. Falls back if an output name is unknowable
    /// or two collide.
    pub fn polars_select(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "select", args)?;
        // Column selection is Polars-only; pandas has no `select` method and uses `[]`.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let positional = args
            .args
            .iter()
            .flat_map(unpack_list_or_tuple_literal)
            .collect::<Vec<_>>();
        // A lone `"*"` selects every column, so the schema is unchanged.
        if let [arg] = &positional[..]
            && let Type::Literal(lit) = &self.expr_infer(arg, &self.error_swallower())
            && let Lit::Str(value) = &lit.value
            && value.as_str() == "*"
        {
            // Committed to a schema, so report the argument errors swallowed by name resolution.
            self.expr_infer(arg, errors);
            return Some(base.clone());
        }
        // Resolve every output name first without emitting errors. A missing or duplicate name returns
        // `None`, so ordinary call checking validates the arguments and reports any errors.
        let mut names = Vec::with_capacity(positional.len());
        let mut seen = SmallSet::new();
        for &arg in &positional {
            let (name, is_column) = match self.polars_column_arg(arg) {
                ColumnArg::Named(name) => (name, true),
                // A selector or wider `str` names columns whose identity is data-dependent.
                ColumnArg::Opaque => return None,
                ColumnArg::Expr => (self.polars_expr_output_name(arg)?, false),
            };
            if !seen.insert(name.clone()) {
                return None;
            }
            names.push((name, is_column));
        }
        // Once every output name is known, resolve each dtype here and report the argument errors
        // swallowed by name resolution.
        let mut columns = Vec::with_capacity(names.len());
        for ((name, is_column), arg) in names.into_iter().zip(positional) {
            self.expr_infer(arg, errors);
            if is_column {
                // A name reference selects a column, and an absent one is reported and dropped.
                if let Some(dtype) = resolve_column(schema, &name, arg.range(), errors) {
                    columns.push((name, dtype));
                }
            } else {
                // An expression yields one output column, degrading its dtype to `Unknown` when
                // unresolved, like `with_columns`.
                let dtype = self
                    .eval_polars_expr(arg, schema, errors)
                    .map_or(PolarsDType::Unknown, ExprValue::dtype);
                columns.push((name, dtype));
            }
        }
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Model `df.drop("a", "b")` as a new schema with the named columns removed, order preserved,
    /// unpacking list and tuple literals. Falls back with `None` unless every element resolves to one
    /// column name, and an unknown name errors only after a schema is committed. Duplicate names are
    /// de-duplicated, unlike `select`.
    pub fn polars_drop(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "drop", args)?;
        // Column drop is Polars-only; pandas `drop` defaults to the index axis.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let positional = args
            .args
            .iter()
            .flat_map(unpack_list_or_tuple_literal)
            .collect::<Vec<_>>();
        let mut dropped: Vec<(Name, TextRange)> = Vec::with_capacity(positional.len());
        let mut seen = SmallSet::new();
        for arg in positional {
            let ColumnArg::Named(name) = self.polars_column_arg(arg) else {
                return None;
            };
            if seen.insert(name.clone()) {
                dropped.push((name, arg.range()));
            }
        }
        // Committed to a schema, so report the argument errors swallowed by name resolution.
        for arg in &args.args {
            self.expr_infer(arg, errors);
        }
        for (name, range) in &dropped {
            if !schema.has_column(name) && schema.is_complete() {
                errors
                    .error_builder(
                        *range,
                        ErrorKind::UnknownColumn,
                        format!("Column `{name}` is not in the DataFrame schema"),
                    )
                    .emit();
            }
        }
        let columns = schema
            .columns
            .iter()
            .filter(|(c, _)| !seen.contains(c))
            .cloned()
            .collect();
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Model `df.rename({"a": "b"})` as a new schema whose renamed columns keep their type and order.
    /// Falls back with `None` unless the sole argument is a dict literal of string-literal pairs, or if
    /// the rename would collide two columns. An unknown source name errors only after a schema is committed.
    pub fn polars_rename(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "rename", args)?;
        let [Expr::Dict(mapping)] = &args.args[..] else {
            return None;
        };
        // Column rename is Polars-only; pandas `rename` defaults to the index axis.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let mut renames: SmallMap<Name, (Name, TextRange)> =
            SmallMap::with_capacity(mapping.items.len());
        let mut name_exprs = Vec::with_capacity(mapping.items.len());
        for item in &mapping.items {
            let (Some(key), value) = (&item.key, &item.value) else {
                return None;
            };
            let (Some(source), Some(dest)) =
                (self.polars_column_name(key), self.polars_column_name(value))
            else {
                return None;
            };
            if renames.insert(source, (dest, key.range())).is_some() {
                return None;
            }
            name_exprs.push((key, value));
        }
        let target = |name: &Name| {
            renames
                .get(name)
                .map_or_else(|| name.clone(), |(dest, _)| dest.clone())
        };
        let mut resulting = SmallSet::new();
        for (name, _) in &schema.columns {
            if !resulting.insert(target(name)) {
                return None;
            }
        }
        // Committed to a schema, so report the argument errors swallowed by name resolution.
        for (key, value) in name_exprs {
            self.expr_infer(key, errors);
            self.expr_infer(value, errors);
        }
        for (source, (_, range)) in &renames {
            if !schema.has_column(source) && schema.is_complete() {
                errors
                    .error_builder(
                        *range,
                        ErrorKind::UnknownColumn,
                        format!("Column `{source}` is not in the DataFrame schema"),
                    )
                    .emit();
            }
        }
        let columns = schema
            .columns
            .iter()
            .map(|(name, ty)| (target(name), *ty))
            .collect();
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Resolve a Polars expression to an `ExprValue` against the receiver schema, or `None` to leave
    /// the output column `Unknown`. A bare string here is a literal, not a column reference.
    fn eval_polars_expr(
        &self,
        expr: &Expr,
        schema: &DataFrameSchema,
        errors: &ErrorCollector,
    ) -> Option<ExprValue> {
        match expr {
            Expr::Call(call) => {
                if let Expr::Attribute(attr) = &*call.func {
                    match attr.attr.id.as_str() {
                        "cast" => {
                            self.eval_polars_expr(&attr.value, schema, errors)?;
                            let [target] = &call.arguments.args[..] else {
                                return None;
                            };
                            return self.polars_dtype_from_expr(target).map(ExprValue::Dtype);
                        }
                        "alias" => return self.eval_polars_expr(&attr.value, schema, errors),
                        method => {
                            if let Some(reducer) = Reducer::parse(method) {
                                let inner = self.eval_polars_expr(&attr.value, schema, errors)?;
                                return Some(ExprValue::Dtype(
                                    reducer.output_dtype(inner.dtype())?,
                                ));
                            }
                        }
                    }
                }
                let id = self.polars_function_id(&call.func)?;
                match (id.name.as_str(), id.module.name().as_str()) {
                    ("len", "polars.functions.len") => Some(ExprValue::Dtype(PolarsDType::UInt32)),
                    ("col", "polars.functions.col") => {
                        let [arg] = &call.arguments.args[..] else {
                            return None;
                        };
                        let name = self.polars_column_name(arg)?;
                        // `"*"` and a regex select many columns whose names are data-dependent.
                        if is_polars_selector_name(&name) {
                            return None;
                        }
                        resolve_column(schema, &name, arg.range(), errors).map(ExprValue::Dtype)
                    }
                    ("lit", "polars.functions.lit") => {
                        for kw in &call.arguments.keywords {
                            if kw.arg.as_ref().is_some_and(|a| a.id.as_str() == "dtype") {
                                return self
                                    .polars_dtype_from_expr(&kw.value)
                                    .map(ExprValue::Dtype);
                            }
                        }
                        let [value] = &call.arguments.args[..] else {
                            return None;
                        };
                        literal_value(value)
                    }
                    _ => None,
                }
            }
            Expr::BinOp(binop) => {
                let a = self.eval_polars_expr(&binop.left, schema, errors)?;
                let b = self.eval_polars_expr(&binop.right, schema, errors)?;
                combine_binop(binop.op, a, b)
            }
            Expr::UnaryOp(unary) => {
                let a = self.eval_polars_expr(&unary.operand, schema, errors)?;
                unary_value(unary.op, a)
            }
            Expr::Compare(cmp) => {
                let ([op], [right]) = (&*cmp.ops, &*cmp.comparators) else {
                    return None;
                };
                if !matches!(
                    op,
                    CmpOp::Eq | CmpOp::NotEq | CmpOp::Lt | CmpOp::LtE | CmpOp::Gt | CmpOp::GtE
                ) {
                    return None;
                }
                let a = self.eval_polars_expr(&cmp.left, schema, errors)?;
                let b = self.eval_polars_expr(right, schema, errors)?;
                comparison_value(a, b)
            }
            _ => literal_value(expr),
        }
    }

    /// The `FuncId` of a call whose callee resolves to a top-level function, or `None`. Inference
    /// swallows errors since each caller reports through its own path.
    fn polars_function_id(&self, func: &Expr) -> Option<Arc<FuncId>> {
        match self.expr_infer(func, &self.error_swallower()).callee_kind() {
            Some(CalleeKind::Function(FunctionKind::Def(id))) => Some(id),
            _ => None,
        }
    }

    /// The output column name Polars gives a `select` expression, which is its leftmost leaf column
    /// overridden by the outermost `.alias(<literal>)`, with a bare `lit` named `"literal"`. Returns
    /// `None` for any data-dependent or unmodeled form so the whole `select` falls back.
    fn polars_expr_output_name(&self, expr: &Expr) -> Option<Name> {
        match expr {
            Expr::Call(call) => {
                if let Expr::Attribute(attr) = &*call.func {
                    match attr.attr.id.as_str() {
                        "cast" => return self.polars_expr_output_name(&attr.value),
                        "alias" => {
                            let [Expr::StringLiteral(s)] = &call.arguments.args[..] else {
                                return None;
                            };
                            return Some(Name::new(s.value.to_str()));
                        }
                        method if Reducer::parse(method).is_some() => {
                            return self.polars_expr_output_name(&attr.value);
                        }
                        _ => {}
                    }
                }
                let id = self.polars_function_id(&call.func)?;
                match (id.name.as_str(), id.module.name().as_str()) {
                    ("len", "polars.functions.len") => Some(Name::new("len")),
                    ("col", "polars.functions.col") => {
                        let [arg] = &call.arguments.args[..] else {
                            return None;
                        };
                        let name = self.polars_column_name(arg)?;
                        // A `"*"` or a regex selects many columns whose names are data-dependent.
                        if is_polars_selector_name(&name) {
                            return None;
                        }
                        Some(name)
                    }
                    ("lit", "polars.functions.lit") => {
                        let [value] = &call.arguments.args[..] else {
                            return None;
                        };
                        // Only `pl.lit(scalar)` is named "literal". A `pl.lit(series)` takes the
                        // series name, which is not statically knowable, so fall back.
                        literal_value(value).map(|_| Name::new("literal"))
                    }
                    _ => None,
                }
            }
            Expr::BinOp(binop) => self.polars_expr_output_name(&binop.left),
            Expr::UnaryOp(unary) => self.polars_expr_output_name(&unary.operand),
            Expr::Compare(cmp) => {
                let ([_], [right]) = (&*cmp.ops, &*cmp.comparators) else {
                    return None;
                };
                // Python reflects a comparison when the left operand is a Python scalar rather than a
                // Polars expression, so the right expression becomes the leftmost leaf and names the
                // output. Decide by the operand's inferred type, since a scalar can be held in a
                // variable, not only written as a literal.
                if self.is_polars_expr_value(&cmp.left) {
                    self.polars_expr_output_name(&cmp.left)
                } else {
                    self.polars_expr_output_name(right)
                }
            }
            // A bare Python scalar literal is promoted to a `lit`, which Polars names `"literal"`.
            // Every other form is unmodeled, so fall back rather than guess an output name.
            Expr::NumberLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::NoneLiteral(_) => literal_value(expr).map(|_| Name::new("literal")),
            _ => None,
        }
    }

    /// Whether `expr` infers to a Polars `Expr` instance rather than a Python scalar, so a comparison
    /// against it does not reflect. Errors are swallowed since the caller reports through its own path.
    fn is_polars_expr_value(&self, expr: &Expr) -> bool {
        matches!(
            self.expr_infer(expr, &self.error_swallower()),
            Type::ClassType(cls) if is_polars_expr(cls.class_object())
        )
    }

    /// Model `df.with_columns(x=..., y=...)` as a new schema, overwriting or appending each named
    /// column with the dtype inferred from its Polars expression. Falls back unless every argument is
    /// a named keyword.
    pub fn polars_with_columns(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if func.attr.id.as_str() != "with_columns" || !args.args.is_empty() {
            return None;
        }
        // Adding columns this way is Polars-only; pandas uses `assign`, not `with_columns`.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Validate syntactically before inferring anything: a `**mapping` spread bails here, so the
        // fallback path stays the sole checker and never double-reports.
        let mut named = Vec::with_capacity(args.keywords.len());
        for kw in &args.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            named.push((arg.id.clone(), &kw.value));
        }
        // Polars evaluates every keyword expression against the receiver's original schema in
        // parallel, so a sibling's new column is not visible; resolve all values before applying.
        let mut columns = schema.columns.clone();
        for (name, value) in named {
            // Infer the value to surface type errors inside it; `eval_polars_expr` is the sole
            // reporter of column errors and uses only the receiver schema.
            self.expr_infer(value, errors);
            let dtype = match self.polars_column_arg(value) {
                // A string keyword value names a column to copy.
                ColumnArg::Named(name) => resolve_column(schema, &name, value.range(), errors),
                // A regex, wildcard, or wider `str` selects a data-dependent column set, so fall back
                // rather than report the selector as a missing column.
                ColumnArg::Opaque => None,
                // A string inside an expression is a literal; any other value is a Polars expression.
                ColumnArg::Expr => self
                    .eval_polars_expr(value, schema, errors)
                    .map(ExprValue::dtype),
            }
            .unwrap_or(PolarsDType::Unknown);
            match columns.iter_mut().find(|(c, _)| *c == name) {
                Some((_, ty)) => *ty = dtype,
                None => columns.push((name, dtype)),
            }
        }
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// A bound `GroupBy` does not expose its receiver schema, so only an inline chain is modeled.
    pub fn polars_group_by_agg(
        &self,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if func.attr.id.as_str() != "agg" {
            return None;
        }
        let Expr::Call(group_by) = &*func.value else {
            return None;
        };
        let Expr::Attribute(group_by_func) = &*group_by.func else {
            return None;
        };
        if group_by_func.attr.id.as_str() != "group_by" {
            return None;
        }
        // Read the already-inferred receiver schema without reporting its errors again.
        let Type::DataFrame(schema) =
            self.expr_infer(&group_by_func.value, &self.error_swallower())
        else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Validate all names before dtype inference so a collision cannot leak diagnostics.
        enum ColumnKind {
            Key,
            Agg,
        }
        let mut outputs = Vec::new();
        for arg in &group_by.arguments.args {
            for elt in unpack_list_or_tuple_literal(arg) {
                outputs.push((self.polars_group_output_name(elt)?, elt, ColumnKind::Key));
            }
        }
        for kw in &group_by.arguments.keywords {
            let Some(name) = &kw.arg else {
                return None;
            };
            if name.id.as_str() == "maintain_order" {
                continue;
            }
            outputs.push((name.id.clone(), &kw.value, ColumnKind::Key));
        }
        for arg in &args.args {
            for elt in unpack_list_or_tuple_literal(arg) {
                outputs.push((self.polars_group_output_name(elt)?, elt, ColumnKind::Agg));
            }
        }
        for kw in &args.keywords {
            let Some(name) = &kw.arg else {
                return None;
            };
            self.polars_group_output_name(&kw.value)?;
            outputs.push((name.id.clone(), &kw.value, ColumnKind::Agg));
        }
        let mut seen = SmallSet::new();
        if outputs
            .iter()
            .any(|(name, _, _)| !seen.insert(name.clone()))
        {
            return None;
        }

        let columns = outputs
            .into_iter()
            .map(|(name, expr, kind)| {
                let dtype = match kind {
                    ColumnKind::Key => self.polars_group_key_dtype(&schema, expr, errors)?,
                    ColumnKind::Agg => self.polars_agg_dtype(&schema, expr, errors)?,
                };
                Some((name, dtype))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    fn polars_group_output_name(&self, expr: &Expr) -> Option<Name> {
        match self.polars_column_arg(expr) {
            ColumnArg::Named(name) => Some(name),
            ColumnArg::Opaque => None,
            ColumnArg::Expr => self.polars_expr_output_name(expr),
        }
    }

    fn polars_group_key_dtype(
        &self,
        schema: &DataFrameSchema,
        expr: &Expr,
        errors: &ErrorCollector,
    ) -> Option<PolarsDType> {
        match self.polars_column_arg(expr) {
            ColumnArg::Named(name) => resolve_column(schema, &name, expr.range(), errors),
            ColumnArg::Opaque => None,
            ColumnArg::Expr => Some(
                self.eval_polars_expr(expr, schema, errors)
                    .map_or(PolarsDType::Unknown, ExprValue::dtype),
            ),
        }
    }

    fn polars_agg_dtype(
        &self,
        schema: &DataFrameSchema,
        expr: &Expr,
        errors: &ErrorCollector,
    ) -> Option<PolarsDType> {
        match self.polars_column_arg(expr) {
            ColumnArg::Named(name) => {
                self.expr_infer(expr, errors);
                resolve_column(schema, &name, expr.range(), errors);
                Some(PolarsDType::Unknown)
            }
            ColumnArg::Opaque => None,
            ColumnArg::Expr => {
                self.expr_infer(expr, errors);
                let dtype = if self.polars_expr_aggregates(expr) {
                    self.eval_polars_expr(expr, schema, errors)
                        .map_or(PolarsDType::Unknown, ExprValue::dtype)
                } else {
                    // Evaluate for column-existence errors, but the list dtype is unmodeled.
                    self.eval_polars_expr(expr, schema, errors);
                    PolarsDType::Unknown
                };
                Some(dtype)
            }
        }
    }

    fn polars_expr_aggregates(&self, expr: &Expr) -> bool {
        let Expr::Call(call) = expr else {
            return false;
        };
        if let Some(id) = self.polars_function_id(&call.func) {
            return id.name.as_str() == "len"
                && id.module.name().as_str() == "polars.functions.len";
        }
        match &*call.func {
            Expr::Attribute(attr) => match attr.attr.id.as_str() {
                "alias" | "cast" => self.polars_expr_aggregates(&attr.value),
                method => Reducer::parse(method).is_some(),
            },
            _ => false,
        }
    }

    /// Model row-only transforms as returning the receiver's schema unchanged; they drop, reorder,
    /// deduplicate, window, or replace rows without touching the column set. `None` if no schema.
    pub fn polars_row_transform(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if !matches!(
            func.attr.id.as_str(),
            "filter" | "sort" | "fill_null" | "head" | "slice" | "unique" | "drop_nulls"
        ) {
            return None;
        }
        // Pandas `filter` selects columns and has no `sort`/`fill_null`, so this is Polars-only.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Infer the arguments so type errors inside them surface; the schema is unchanged. A `*args`
        // spread is an `Expr::Starred`; infer its inner value, since `expr_infer` treats a bare
        // starred expression as a type-form and would wrongly report the iterable as not-a-type.
        for arg in args.args.iter() {
            let value = match arg {
                Expr::Starred(starred) => &starred.value,
                _ => arg,
            };
            self.expr_infer(value, errors);
        }
        for kw in args.keywords.iter() {
            self.expr_infer(&kw.value, errors);
        }
        Some(base.clone())
    }

    /// Model `df.vstack(other)`/`df.extend(other)` as the receiver schema unchanged: both append rows
    /// and raise unless `other` has the identical schema, so `other` needs no inspection. `other` must
    /// be a Polars `DataFrame`; a non-frame, pandas, or any keyword (`in_place=` unmodeled) falls back.
    pub fn polars_row_append(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if !matches!(func.attr.id.as_str(), "vstack" | "extend")
            || schema.kind != DataFrameKind::Polars
            || !args.keywords.is_empty()
        {
            return None;
        }
        let [other_expr] = &args.args[..] else {
            return None;
        };
        if !is_polars_dataframe_type(&self.expr_infer(other_expr, &self.error_swallower())) {
            return None;
        }
        // Now committed to a schema, so infer `other` with real errors as the sole reporter of any
        // error inside it, since returning here bypasses ordinary call-checking.
        self.expr_infer(other_expr, errors);
        Some(base.clone())
    }

    /// Model `df.lazy()` and `lf.collect()`, which move the column set unchanged onto the converted
    /// frame class. A receiver without a schema, a non-Polars frame, or any positional argument falls back.
    pub fn polars_lazy_collect(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if !matches!(func.attr.id.as_str(), "lazy" | "collect")
            || schema.kind != DataFrameKind::Polars
            || !args.args.is_empty()
        {
            return None;
        }
        // The stub is the single source of truth for the converted class and any keyword errors.
        let call_kws: Vec<CallKeyword> = args.keywords.iter().map(CallKeyword::new).collect();
        let result = self.call_method_or_error(
            &schema.underlying_type(),
            &func.attr.id,
            func.range(),
            &[],
            &call_kws,
            errors,
            None,
        );
        // An unexpected converted class keeps the opaque result rather than carrying a wrong schema.
        match (func.attr.id.as_str(), result) {
            ("lazy", Type::ClassType(cls)) if is_polars_lazyframe(cls.class_object()) => Some(
                DataFrameSchema {
                    underlying: cls,
                    ..(**schema).clone()
                }
                .to_type(),
            ),
            ("collect", Type::ClassType(cls)) if is_polars_dataframe(cls.class_object()) => Some(
                DataFrameSchema {
                    underlying: cls,
                    ..(**schema).clone()
                }
                .to_type(),
            ),
            (_, result) => Some(result),
        }
    }

    /// Model `df.cast(...)`, which rewrites column dtypes. A single `pl.<DType>` casts every column;
    /// a `{name: pl.<DType>}` dict casts the named ones and reports a name absent from the schema.
    /// An unrecognized dtype falls back with `None` before any error is emitted.
    pub fn polars_cast(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "cast", args)?;
        let [arg] = &args.args[..] else {
            return None;
        };
        // Pandas `astype` casts columns; `cast` is Polars-only.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let columns = match arg {
            Expr::Dict(mapping) => {
                let mut casts: SmallMap<Name, (TextRange, PolarsDType)> =
                    SmallMap::with_capacity(mapping.items.len());
                for item in &mapping.items {
                    let (Some(Expr::StringLiteral(key)), value) = (&item.key, &item.value) else {
                        return None;
                    };
                    casts.insert(
                        Name::new(key.value.to_str()),
                        (key.range(), self.polars_dtype_from_expr(value)?),
                    );
                }
                for (name, (range, _)) in &casts {
                    if !schema.has_column(name) && schema.is_complete() {
                        errors
                            .error_builder(
                                *range,
                                ErrorKind::UnknownColumn,
                                format!("Column `{name}` is not in the DataFrame schema"),
                            )
                            .emit();
                    }
                }
                schema
                    .columns
                    .iter()
                    .map(|(name, ty)| (name.clone(), casts.get(name).map_or(*ty, |(_, d)| *d)))
                    .collect()
            }
            _ => {
                let dtype = self.polars_dtype_from_expr(arg)?;
                schema
                    .columns
                    .iter()
                    .map(|(name, _)| (name.clone(), dtype))
                    .collect()
            }
        };
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// The key names and ranges from a single name or a list/tuple literal of names.
    fn join_key_names(&self, on: &Expr) -> Option<Vec<(Name, TextRange)>> {
        if let Some(name) = self.polars_column_name(on) {
            return Some(vec![(name, on.range())]);
        }
        let elts = match on {
            Expr::List(list) => &list.elts,
            Expr::Tuple(tuple) => &tuple.elts,
            _ => return None,
        };
        elts.iter()
            .map(|elt| self.polars_column_name(elt).map(|name| (name, elt.range())))
            .collect()
    }

    /// Model `df.join(other, on=..., how=...)` as the merged schema of the two frames; dtypes copy
    /// straight from each side, so the result is a pure column-name computation (see [JoinStrategy]).
    /// Only same-name `on=` keys with the default coalesce and suffix are modeled; `left_on`/`right_on`,
    /// an explicit `coalesce=`, or a custom `suffix=` fall back.
    pub fn polars_join(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if func.attr.id.as_str() != "join" || schema.kind != DataFrameKind::Polars {
            return None;
        }
        let [other_expr] = &args.args[..] else {
            return None;
        };
        let mut on = None;
        let mut how = JoinHow::Inner;
        for kw in &args.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "on" => on = Some(&kw.value),
                "how" => {
                    let Expr::StringLiteral(s) = &kw.value else {
                        return None;
                    };
                    how = JoinHow::parse(s.value.to_str())?;
                }
                _ => return None,
            }
        }
        // `cross` takes no keys and every other strategy needs `on=` here since `left_on`/`right_on`
        // are not yet modeled; both mismatches raise at runtime, so fall back.
        let keys = match (how, on) {
            (JoinHow::Cross, None) => Vec::new(),
            (JoinHow::Cross, Some(_)) | (_, None) => return None,
            (_, Some(on)) => self.join_key_names(on)?,
        };
        let Type::DataFrame(other) = self.expr_infer(other_expr, &self.error_swallower()) else {
            return None;
        };
        // A key absent from either frame makes the join malformed, so report it and fall back. Only a
        // Complete side can prove a key absent, since a Partial one may hold it untracked.
        for (name, range) in &keys {
            let base_missing = !schema.has_column(name);
            let other_missing = !other.has_column(name);
            if base_missing || other_missing {
                if (base_missing && schema.is_complete()) || (other_missing && other.is_complete())
                {
                    errors
                        .error_builder(
                            *range,
                            ErrorKind::UnknownColumn,
                            format!("Column `{name}` is not in the DataFrame schema"),
                        )
                        .emit();
                }
                return None;
            }
        }
        let key_set: SmallSet<Name> = keys.into_iter().map(|(name, _)| name).collect();
        let column_dtype = |columns: &[(Name, PolarsDType)], name: &Name| {
            columns.iter().find(|(c, _)| c == name).map(|(_, t)| *t)
        };
        // A coalesced key keeps the primary side's dtype, so paired keys with differing dtypes could
        // be cast or rejected at runtime; fall back rather than pick one side.
        if how.coalesces()
            && key_set.iter().any(|name| {
                column_dtype(&schema.columns, name) != column_dtype(&other.columns, name)
            })
        {
            return None;
        }
        let not_key = |(name, _): &&(Name, PolarsDType)| !key_set.contains(name);
        let (base_columns, other_columns): (Vec<_>, Vec<_>) = match how {
            JoinHow::Semi | JoinHow::Anti => (schema.columns.clone(), Vec::new()),
            JoinHow::Inner | JoinHow::Left => (
                schema.columns.clone(),
                other.columns.iter().filter(not_key).cloned().collect(),
            ),
            JoinHow::Full | JoinHow::Cross => (schema.columns.clone(), other.columns.clone()),
            JoinHow::Right => (
                schema.columns.iter().filter(not_key).cloned().collect(),
                other.columns.clone(),
            ),
        };
        let base_names: SmallSet<Name> =
            base_columns.iter().map(|(name, _)| name.clone()).collect();
        let mut columns = base_columns;
        for (name, ty) in other_columns {
            let out = if base_names.contains(&name) {
                Name::new(format!("{name}_right"))
            } else {
                name
            };
            columns.push((out, ty));
        }
        // A suffixed name that already exists is a runtime `DuplicateError`, so fall back rather than
        // emit a schema with a duplicate column.
        let mut seen = SmallSet::new();
        if columns.iter().any(|(name, _)| !seen.insert(name.clone())) {
            return None;
        }
        // Committed to a schema, so report the errors inside `other` and the `on=` keys; returning
        // here bypasses the ordinary call-checking that would otherwise report them.
        self.expr_infer(other_expr, errors);
        if let Some(on) = on {
            self.expr_infer(on, errors);
        }
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Model `df.hstack(other)` as the receiver columns followed by `other`'s (dtypes copy from each
    /// side). Only a schema-carrying Polars `DataFrame` is modeled; a non-frame, keyword, or
    /// overlapping name (runtime `DuplicateError`) falls back.
    pub fn polars_hstack(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "hstack", args)?;
        // Appending columns this way is Polars-only; pandas has no `hstack`.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let [other_expr] = &args.args[..] else {
            return None;
        };
        let Type::DataFrame(other) = self.expr_infer(other_expr, &self.error_swallower()) else {
            return None;
        };
        // A pandas frame raises `AttributeError` at runtime, so fall back rather than fabricate a
        // merged Polars schema and swallow the argument-type error.
        if other.kind != DataFrameKind::Polars {
            return None;
        }
        let mut columns = schema.columns.clone();
        columns.extend(other.columns.iter().cloned());
        let mut seen = SmallSet::new();
        if columns.iter().any(|(name, _)| !seen.insert(name.clone())) {
            return None;
        }
        // Now committed to a schema, so infer `other` with real errors as the sole reporter of any
        // error inside it, since returning here bypasses ordinary call-checking.
        self.expr_infer(other_expr, errors);
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
                kind: schema.kind.clone(),
            }
            .to_type(),
        )
    }

    /// Model an in-place column mutation as the receiver schema degraded for that mutation. The frame
    /// is rebound in the binding phase, so a discarded return value still degrades it.
    pub fn polars_in_place_column_mutation(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        let kind = polars_column_mutation(func.attr.id.as_str(), args)?;
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Infer the arguments so type errors inside them surface, though the schema ignores them.
        for arg in &args.args {
            self.expr_infer(arg, errors);
        }
        for kw in &args.keywords {
            self.expr_infer(&kw.value, errors);
        }
        Some(polars_degrade_for_mutation(base, &kind, |callee| {
            self.polars_series_constructor(callee)
        }))
    }

    pub(crate) fn polars_series_constructor(&self, callee: &Expr) -> bool {
        matches!(
            self.expr_infer(callee, &self.error_swallower()),
            Type::ClassDef(cls)
                if cls.has_toplevel_qname("polars.series.series", "Series")
                    || cls.has_toplevel_qname("polars.dataframe.frame", "Series")
        )
    }

    /// Model `df.get_column("a")` as `Series[dtype]` of the named column. Only a single string-literal
    /// name with no `default=` is modeled, and anything else falls back. A column absent from the
    /// Complete schema is reported and keeps the opaque `Series`.
    pub fn polars_get_column(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = series_method_schema(base, func, "get_column")?;
        let name_expr = get_column_name_arg(args)?;
        let name = self.polars_column_name(name_expr)?;
        let dtype = resolve_column(schema, &name, name_expr.range(), errors);
        Some(self.wrap_series_method(schema, func, args, dtype, errors))
    }

    /// Model `df.to_series(i)` as `Series[dtype]` of the column at position `i`, with negative indexing.
    /// Only a static integer index is modeled, and anything else falls back. An index out of the
    /// column count raises `IndexError` at runtime, so it is reported and keeps the opaque `Series`.
    pub fn polars_to_series(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = series_method_schema(base, func, "to_series")?;
        let index = to_series_index(args)?;
        let len = schema.columns.len() as i128;
        let resolved = if index < 0 { index + len } else { index };
        let dtype = if (0..len).contains(&resolved) {
            Some(schema.columns[resolved as usize].1)
        } else {
            errors
                .error_builder(
                    args.range(),
                    ErrorKind::UnknownColumn,
                    format!("Index {index} is out of bounds for a DataFrame with {len} columns"),
                )
                .emit();
            None
        };
        Some(self.wrap_series_method(schema, func, args, dtype, errors))
    }

    /// Call a `Series`-returning method on the opaque underlying frame so argument errors surface,
    /// wrapping the result as `Series[dtype]` when the column resolved. An unresolved column keeps the
    /// opaque `Series`.
    fn wrap_series_method(
        &self,
        schema: &DataFrameSchema,
        func: &ExprAttribute,
        args: &Arguments,
        dtype: Option<PolarsDType>,
        errors: &ErrorCollector,
    ) -> Type {
        let call_args: Vec<CallArg> = args.args.iter().map(CallArg::expr_maybe_starred).collect();
        let call_kws: Vec<CallKeyword> = args.keywords.iter().map(CallKeyword::new).collect();
        let result = self.call_method_or_error(
            &schema.underlying_type(),
            &func.attr.id,
            func.range(),
            &call_args,
            &call_kws,
            errors,
            None,
        );
        match (dtype, result) {
            (Some(dtype), Type::ClassType(cls)) if is_polars_series(cls.class_object()) => {
                SeriesSchema {
                    underlying: cls,
                    dtype,
                }
                .to_type()
            }
            (_, result) => result,
        }
    }
}
