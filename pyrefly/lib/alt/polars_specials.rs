/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Column-aware typing for Polars and pandas DataFrames.

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
use crate::binding::polars::PolarsMutationKind;
use crate::binding::polars::PolarsMutationMethod;
use crate::binding::polars::polars_column_mutation;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::class::Class;
use crate::types::function::FuncId;
use crate::types::function::FunctionKind;
use crate::types::literal::Lit;

const POLARS_MODULE: &str = "polars";
const POLARS_MODULE_PREFIX: &str = "polars.";
const POLARS_ALL_COLUMNS: &str = "*";
const POLARS_DEFAULT_INFER_SCHEMA_LENGTH: usize = 100;
const POLARS_DEFAULT_JOIN_SUFFIX: &str = "_right";
const POLARS_LEN_OUTPUT_NAME: &str = "len";
const POLARS_LITERAL_OUTPUT_NAME: &str = "literal";

#[derive(Clone, Copy)]
enum RuntimeClass {
    PolarsDataFrame,
    PolarsSeries,
    PolarsDataFrameSeries,
    PolarsExpr,
    PolarsLazyFrame,
    PolarsCol,
    PolarsSchema,
    PandasDataFrame,
    Date,
    Datetime,
    Time,
    Timedelta,
}

impl RuntimeClass {
    fn matches(self, cls: &Class) -> bool {
        let (module, name) = match self {
            Self::PolarsDataFrame => ("polars.dataframe.frame", "DataFrame"),
            Self::PolarsSeries => ("polars.series.series", "Series"),
            Self::PolarsDataFrameSeries => ("polars.dataframe.frame", "Series"),
            Self::PolarsExpr => ("polars.expr.expr", "Expr"),
            Self::PolarsLazyFrame => ("polars.lazyframe.frame", "LazyFrame"),
            Self::PolarsCol => ("polars.functions.col", "Col"),
            Self::PolarsSchema => ("polars.schema", "Schema"),
            Self::PandasDataFrame => ("pandas.core.frame", "DataFrame"),
            Self::Date => ("datetime", "date"),
            Self::Datetime => ("datetime", "datetime"),
            Self::Time => ("datetime", "time"),
            Self::Timedelta => ("datetime", "timedelta"),
        };
        cls.has_toplevel_qname(module, name)
    }
}

fn is_polars_dataframe(cls: &Class) -> bool {
    RuntimeClass::PolarsDataFrame.matches(cls)
}

pub fn is_polars_series(cls: &Class) -> bool {
    RuntimeClass::PolarsSeries.matches(cls)
}

fn is_polars_expr(cls: &Class) -> bool {
    RuntimeClass::PolarsExpr.matches(cls)
}

fn is_polars_lazyframe(cls: &Class) -> bool {
    RuntimeClass::PolarsLazyFrame.matches(cls)
}

fn column_transform_schema<'b>(base: &'b Type, args: &Arguments) -> Option<&'b DataFrameSchema> {
    let Type::DataFrame(schema) = base else {
        return None;
    };
    args.keywords.is_empty().then_some(&**schema)
}

fn dataframe_type_with_columns(
    schema: &DataFrameSchema,
    columns: Vec<(Name, PolarsDType)>,
) -> Type {
    dataframe_type_with_columns_and_completeness(schema, columns, schema.completeness)
}

fn dataframe_type_with_columns_and_completeness(
    schema: &DataFrameSchema,
    columns: Vec<(Name, PolarsDType)>,
    completeness: SchemaCompleteness,
) -> Type {
    DataFrameSchema {
        underlying: schema.underlying.clone(),
        columns,
        completeness,
        ..schema.clone()
    }
    .to_type()
}

fn is_pandas_dataframe(cls: &Class) -> bool {
    RuntimeClass::PandasDataFrame.matches(cls)
}

/// Methods whose arguments may contain column references.
pub fn is_dataframe_column_method(method: &str) -> bool {
    matches!(
        method,
        "select" | "drop" | "with_columns" | "filter" | "sort" | "group_by" | "groupby"
    )
}

/// Apply a binding-time mutation to a tracked frame schema.
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

fn is_polars_dataframe_type(ty: &Type) -> bool {
    match ty {
        Type::DataFrame(schema) => schema.kind == DataFrameKind::Polars,
        Type::ClassType(ct) => is_polars_dataframe(ct.class_object()),
        _ => false,
    }
}

fn polars_dtype_from_scalar_type(ty: &Type) -> Option<PolarsDType> {
    // Integers past i64 have a data-shape-dependent runtime dtype, so do not claim Int64.
    if let Type::Literal(lit) = ty {
        return match &lit.value {
            Lit::Int(i) => i.as_i64().map(|_| PolarsDType::Int64),
            Lit::Bool(_) => Some(PolarsDType::Boolean),
            Lit::Str(_) => Some(PolarsDType::String),
            Lit::Bytes(_) => Some(PolarsDType::Binary),
            Lit::Enum(_) => None,
        };
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

fn is_string_type(ty: &Type) -> bool {
    ty.is_literal_string() || polars_dtype_from_scalar_type(ty) == Some(PolarsDType::String)
}

fn is_polars_selector_name(name: &Name) -> bool {
    let name = name.as_str();
    name == POLARS_ALL_COLUMNS || (name.starts_with('^') && name.ends_with('$'))
}

/// Map a resolved type to the Polars dtype it names, e.g. the `pl.Float64` class to `Float64`.
/// Only the modeled scalar dtypes from the `polars` package are recognized; anything else is `None`.
fn polars_dtype_from_type(ty: &Type) -> Option<PolarsDType> {
    let cls = match ty {
        Type::ClassDef(cls) => cls,
        Type::ClassType(cls) => cls.class_object(),
        _ => return None,
    };
    let module = cls.module_name();
    if module.as_str() != POLARS_MODULE && !module.as_str().starts_with(POLARS_MODULE_PREFIX) {
        return None;
    }
    PolarsDType::from_polars_name(cls.name().as_str())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PolarsFunction {
    Col,
    Concat,
    Csv(PolarsCsvFunction),
    Len,
    Lit,
    Unmodeled,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PolarsCsvFunction {
    Read,
    Scan,
}

#[derive(Clone, Copy)]
enum PolarsMethod {
    Select,
    Drop,
    Rename,
    WithColumns,
    FillNull,
    RowTransform,
    RowAppend,
    Cast,
    FrameConversion(PolarsFrameConversion),
    Join,
    Hstack,
    GroupByAgg,
    InPlaceMutation,
    GetColumn,
    ToSeries,
}

impl PolarsMethod {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "select" => Self::Select,
            "drop" => Self::Drop,
            "rename" => Self::Rename,
            "with_columns" => Self::WithColumns,
            "fill_null" => Self::FillNull,
            "filter" | "sort" | "head" | "slice" | "unique" | "drop_nulls" => Self::RowTransform,
            "vstack" | "extend" => Self::RowAppend,
            "cast" => Self::Cast,
            "lazy" => Self::FrameConversion(PolarsFrameConversion::Lazy),
            "collect" => Self::FrameConversion(PolarsFrameConversion::Collect),
            "join" => Self::Join,
            "agg" => Self::GroupByAgg,
            "get_column" => Self::GetColumn,
            "to_series" => Self::ToSeries,
            name => match PolarsMutationMethod::parse(name)? {
                PolarsMutationMethod::InsertColumn | PolarsMutationMethod::ReplaceColumn => {
                    Self::InPlaceMutation
                }
                PolarsMutationMethod::Hstack => Self::Hstack,
            },
        })
    }
}

#[derive(Clone, Copy)]
enum PolarsFrameConversion {
    Lazy,
    Collect,
}

impl PolarsFunction {
    fn from_id(id: &FuncId) -> Self {
        match (id.name.as_str(), id.module.name().as_str()) {
            ("col", "polars.functions.col") => Self::Col,
            ("concat", "polars.functions.eager") => Self::Concat,
            ("len", "polars.functions.len") => Self::Len,
            ("lit", "polars.functions.lit") => Self::Lit,
            ("read_csv", "polars.io.csv.functions") => Self::Csv(PolarsCsvFunction::Read),
            ("scan_csv", "polars.io.csv.functions") => Self::Csv(PolarsCsvFunction::Scan),
            _ => Self::Unmodeled,
        }
    }

    fn from_callee(callee: &Type) -> Option<Self> {
        if let Type::ClassType(cls) = callee
            && RuntimeClass::PolarsCol.matches(cls.class_object())
        {
            return Some(Self::Col);
        }
        match callee.callee_kind() {
            Some(CalleeKind::Function(FunctionKind::Def(id))) => Some(Self::from_id(&id)),
            _ => None,
        }
    }
}

enum CsvColumnSelection {
    All,
    Names(SmallSet<Name>),
    Indices(SmallSet<usize>),
}

enum CsvDtypeOverrides {
    Unchanged,
    Sequence(Vec<PolarsDType>),
}

impl CsvDtypeOverrides {
    /// A complete schema changes only for positional dtype overrides.
    fn parse(
        expr: Option<&Expr>,
        parse_dtype: impl FnMut(&Expr) -> Option<PolarsDType>,
    ) -> Option<Self> {
        Some(match expr {
            None | Some(Expr::NoneLiteral(_) | Expr::Dict(_)) => Self::Unchanged,
            Some(expr) => Self::Sequence(
                literal_sequence(expr)?
                    .iter()
                    .map(parse_dtype)
                    .collect::<Option<Vec<_>>>()?,
            ),
        })
    }
}

struct PolarsCsvCommonOptions {
    schema: Vec<(Name, PolarsDType)>,
    overrides: CsvDtypeOverrides,
    new_columns: Vec<Name>,
    row_index_name: Option<Name>,
}

enum PolarsCsvOptions {
    Read {
        common: PolarsCsvCommonOptions,
        selection: CsvColumnSelection,
    },
    Scan {
        common: PolarsCsvCommonOptions,
        include_file_paths: Option<Name>,
    },
}

#[derive(Clone, Copy)]
enum ConcatHow {
    Vertical,
    VerticalRelaxed,
}

impl ConcatHow {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "vertical" => Self::Vertical,
            "vertical_relaxed" => Self::VerticalRelaxed,
            _ => return None,
        })
    }
}

#[derive(Clone)]
struct PolarsDictData<'b> {
    columns: SmallMap<Name, &'b Expr>,
    range: TextRange,
}

#[derive(Clone)]
enum PolarsData<'b> {
    Dict(PolarsDictData<'b>),
    Records(SmallMap<Name, Vec<&'b Expr>>),
    TypedDict(Vec<(Name, PolarsDType)>, SchemaCompleteness),
}

/// Parsed DataFrame constructor inputs.
struct PolarsConstruct<'b> {
    data: Option<PolarsData<'b>>,
    schema: Option<Vec<(Name, Option<PolarsDType>)>>,
    columns: Option<Vec<Name>>,
    overrides: SmallMap<Name, PolarsDType>,
    strict: bool,
}

pub(crate) enum PolarsCallSpecialization {
    DataFrame {
        columns: Vec<(Name, PolarsDType)>,
        kind: DataFrameKind,
        completeness: SchemaCompleteness,
    },
    Series(PolarsDType),
}

struct SeriesConstruct<'b> {
    values: Option<&'b Expr>,
    dtype: Option<PolarsDType>,
    strict: bool,
}

/// A bare dict defers `None` to data inference; `pl.Schema` rejects it.
#[derive(Clone, Copy, PartialEq)]
enum SchemaForm {
    Dict,
    SchemaClass,
}

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

    fn coalesces(self) -> bool {
        matches!(self, Self::Inner | Self::Left | Self::Right)
    }
}

fn literal_sequence(expr: &Expr) -> Option<&[Expr]> {
    match expr {
        Expr::List(list) => Some(&list.elts),
        Expr::Tuple(tuple) => Some(&tuple.elts),
        _ => None,
    }
}

fn positional_elements(arg: &Expr) -> &[Expr] {
    literal_sequence(arg).unwrap_or_else(|| std::slice::from_ref(arg))
}

/// A pinned dtype or a numeric literal that can adapt to its other operand.
#[derive(Clone, Copy)]
enum ExprValue {
    Dtype(PolarsDType),
    IntLit(i128),
    FloatLit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FillNullValue {
    IntegerLiteral(i128),
    /// The literal exceeds the i128 range used by inference.
    IntegerOutsideModel,
    Float,
    /// The integer value is not statically known.
    DynamicInteger,
    /// The value does not change an integer column dtype.
    Other,
}

impl FillNullValue {
    fn from_type(ty: &Type) -> Self {
        if let Type::Literal(lit) = ty
            && let Lit::Int(value) = &lit.value
        {
            return value
                .to_string()
                .parse::<i128>()
                .map_or(Self::IntegerOutsideModel, Self::IntegerLiteral);
        }
        match polars_dtype_from_scalar_type(ty) {
            Some(PolarsDType::Float64) => Self::Float,
            Some(PolarsDType::Int64) => Self::DynamicInteger,
            _ => Self::Other,
        }
    }

    fn widen_integer(self, dtype: PolarsDType) -> PolarsDType {
        if !dtype.is_integer() {
            return dtype;
        }
        match self {
            Self::Float => PolarsDType::Float64,
            Self::IntegerLiteral(value) => {
                integer_dtype_with_literal(dtype, value).unwrap_or(PolarsDType::Unknown)
            }
            Self::DynamicInteger => PolarsDType::Unknown,
            Self::IntegerOutsideModel | Self::Other => dtype,
        }
    }
}

enum ColumnArg {
    Named(Name),
    Opaque,
    Expr,
}

#[derive(Clone, Copy)]
enum ArgumentValue<'b> {
    Missing,
    Present(&'b Expr),
}

impl<'b> ArgumentValue<'b> {
    fn into_option(self) -> Option<&'b Expr> {
        match self {
            Self::Missing => None,
            Self::Present(expr) => Some(expr),
        }
    }
}

/// Extract the last named form or its positional form and reject conflicts.
fn extract_argument<'b>(
    arguments: &'b Arguments,
    position: usize,
    name: &str,
) -> Option<ArgumentValue<'b>> {
    let positional = arguments.args.get(position);
    let keyword = arguments
        .keywords
        .iter()
        .rev()
        .find(|kw| kw.arg.as_ref().is_some_and(|arg| arg.id.as_str() == name))
        .map(|kw| &kw.value);
    match (positional, keyword) {
        (Some(_), Some(_)) => None,
        (Some(expr), None) | (None, Some(expr)) => Some(ArgumentValue::Present(expr)),
        (None, None) => Some(ArgumentValue::Missing),
    }
}

/// Check positional arity and optionally restrict keyword names.
fn arguments_are_valid(
    arguments: &Arguments,
    max_positional: usize,
    allowed_keywords: Option<&[&str]>,
) -> bool {
    arguments.args.len() <= max_positional
        && allowed_keywords.is_none_or(|allowed| {
            arguments.keywords.iter().all(|kw| {
                kw.arg
                    .as_ref()
                    .is_some_and(|arg| allowed.contains(&arg.id.as_str()))
            })
        })
}

impl ExprValue {
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

fn series_method_schema(base: &Type) -> Option<&DataFrameSchema> {
    let Type::DataFrame(schema) = base else {
        return None;
    };
    (schema.kind == DataFrameKind::Polars && schema.is_complete()).then_some(&**schema)
}

fn get_column_name_arg(args: &Arguments) -> Option<&Expr> {
    if !arguments_are_valid(args, 1, Some(&["name"])) {
        return None;
    }
    extract_argument(args, 0, "name")?.into_option()
}

fn resolve_column(
    schema: &DataFrameSchema,
    name: &Name,
    range: TextRange,
    errors: &ErrorCollector,
) -> Option<PolarsDType> {
    match schema.columns.iter().find(|(c, _)| c == name) {
        Some((_, ty)) => Some(*ty),
        None => {
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
    }
}

fn report_duplicate_column(name: &Name, range: TextRange, errors: &ErrorCollector) {
    errors
        .error_builder(
            range,
            ErrorKind::DuplicateColumn,
            format!("Projection produces duplicate column `{name}`"),
        )
        .emit();
}

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

fn int_lit_with_dtype(d: PolarsDType, v: i128) -> Option<ExprValue> {
    if d.is_float() {
        return Some(ExprValue::Dtype(d));
    }
    match d.int_bounds() {
        Some((lo, hi)) if (lo..=hi).contains(&v) => Some(ExprValue::Dtype(d)),
        _ => None,
    }
}

fn float_lit_with_dtype(d: PolarsDType) -> Option<ExprValue> {
    use PolarsDType::*;
    match d {
        Float32 => Some(ExprValue::Dtype(Float32)),
        Float64 => Some(ExprValue::Dtype(Float64)),
        d if d.is_integer() => Some(ExprValue::Dtype(Float64)),
        _ => None,
    }
}

fn pow(a: ExprValue, b: ExprValue) -> Option<ExprValue> {
    if !a.is_numeric() || !b.is_numeric() {
        return None;
    }
    let left = a.dtype();
    let right = b.dtype();
    Some(ExprValue::Dtype(if left.is_float() {
        left
    } else if right.is_float() {
        right
    } else {
        left
    }))
}

fn integer_dtype_with_literal(dtype: PolarsDType, value: i128) -> Option<PolarsDType> {
    use PolarsDType::*;
    let (lower, upper) = dtype.int_bounds()?;
    if (lower..=upper).contains(&value)
        || value < i64::MIN as i128
        || value > u64::MAX as i128
        || dtype == UInt128
    {
        return Some(dtype);
    }
    if dtype == UInt64 && value < 0 {
        return Some(Int64);
    }
    let smallest = |candidates: [PolarsDType; 4]| {
        candidates.into_iter().find(|candidate| {
            candidate
                .int_bounds()
                .is_some_and(|(lower, upper)| (lower..=upper).contains(&value))
        })
    };
    let literal = if value < 0 {
        smallest([Int8, Int16, Int32, Int64])
            .expect("negative literal within i64 bounds must fit Int64")
    } else if dtype.is_signed_int() {
        smallest([Int8, Int16, Int32, Int64]).unwrap_or(UInt64)
    } else {
        smallest([UInt8, UInt16, UInt32, UInt64])
            .expect("nonnegative literal within u64 bounds must fit UInt64")
    };
    dtype.supertype(literal)
}

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
        Operator::Add | Operator::Sub | Operator::Mult | Operator::FloorDiv | Operator::Mod => {
            arith(a, b)
        }
        Operator::Pow => pow(a, b),
        Operator::LShift | Operator::RShift | Operator::MatMult => None,
    }
}

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

#[derive(Clone, Copy)]
enum PolarsExprMethod {
    Alias,
    Cast,
    Reducer(Reducer),
}

impl PolarsExprMethod {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "alias" => Some(Self::Alias),
            "cast" => Some(Self::Cast),
            name => Reducer::parse(name).map(Self::Reducer),
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn polars_method_call(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if matches!(base, Type::DataFrame(schema) if schema.kind == DataFrameKind::Pandas) {
            return None;
        }
        match PolarsMethod::parse(func.attr.id.as_str())? {
            PolarsMethod::Select => self.polars_select(base, args, errors),
            PolarsMethod::Drop => self.polars_drop(base, args, errors),
            PolarsMethod::Rename => self.polars_rename(base, args, errors),
            PolarsMethod::WithColumns => self.polars_with_columns(base, args, errors),
            PolarsMethod::FillNull => self.polars_fill_null(base, args, errors),
            PolarsMethod::RowTransform => self.polars_row_transform(base, args, errors),
            PolarsMethod::RowAppend => self.polars_row_append(base, args, errors),
            PolarsMethod::Cast => self.polars_cast(base, args, errors),
            PolarsMethod::FrameConversion(conversion) => {
                self.polars_lazy_collect(base, func, args, errors, conversion)
            }
            PolarsMethod::Join => self.polars_join(base, args, errors),
            PolarsMethod::Hstack => self
                .polars_hstack(base, args, errors)
                .or_else(|| self.polars_in_place_column_mutation(base, func, args, errors)),
            PolarsMethod::GroupByAgg => self.polars_group_by_agg(func, args, errors),
            PolarsMethod::InPlaceMutation => {
                self.polars_in_place_column_mutation(base, func, args, errors)
            }
            PolarsMethod::GetColumn => self.polars_get_column(base, func, args, errors),
            PolarsMethod::ToSeries => self.polars_to_series(base, func, args, errors),
        }
    }

    pub(crate) fn infer_polars_call_specialization(
        &self,
        callee: &Type,
        arguments: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<PolarsCallSpecialization> {
        if let Type::ClassDef(cls) = callee
            && (is_polars_dataframe(cls) || is_pandas_dataframe(cls))
            && let Some(construct) = self.polars_construct_options(arguments)
        {
            let kind = if is_polars_dataframe(cls) {
                DataFrameKind::Polars
            } else {
                DataFrameKind::Pandas
            };
            return self.infer_dataframe_schema(&construct, kind, errors).map(
                |(columns, completeness)| PolarsCallSpecialization::DataFrame {
                    columns,
                    kind,
                    completeness,
                },
            );
        }
        match PolarsFunction::from_callee(callee) {
            Some(PolarsFunction::Csv(function)) => {
                return self
                    .infer_polars_csv_schema(arguments, function)
                    .map(|columns| PolarsCallSpecialization::DataFrame {
                        columns,
                        kind: DataFrameKind::Polars,
                        completeness: SchemaCompleteness::Complete,
                    });
            }
            Some(PolarsFunction::Concat) => {
                return self
                    .infer_polars_concat(arguments)
                    .map(
                        |(columns, completeness)| PolarsCallSpecialization::DataFrame {
                            columns,
                            kind: DataFrameKind::Polars,
                            completeness,
                        },
                    );
            }
            _ => {}
        }
        if let Type::ClassDef(cls) = callee
            && is_polars_series(cls)
        {
            return self
                .infer_series_dtype(arguments)
                .map(PolarsCallSpecialization::Series);
        }
        None
    }

    pub(crate) fn apply_polars_call_specialization(
        &self,
        result: Type,
        specialization: Option<PolarsCallSpecialization>,
    ) -> Type {
        match (specialization, result) {
            (
                Some(PolarsCallSpecialization::DataFrame {
                    columns,
                    kind,
                    completeness,
                }),
                Type::ClassType(underlying),
            ) => DataFrameSchema {
                underlying,
                columns,
                completeness,
                kind,
            }
            .to_type(),
            (Some(PolarsCallSpecialization::Series(dtype)), Type::ClassType(underlying)) => {
                SeriesSchema { underlying, dtype }.to_type()
            }
            (_, result) => result,
        }
    }

    fn polars_dtype_from_expr(&self, e: &Expr) -> Option<PolarsDType> {
        let ty = self.expr_infer(e, &self.error_swallower());
        polars_dtype_from_type(&ty)
    }

    fn polars_column_arg(&self, expr: &Expr) -> ColumnArg {
        let ty = self.expr_infer(expr, &self.error_swallower());
        if let Type::Literal(lit) = &ty
            && let Lit::Str(value) = &lit.value
        {
            let name = Name::new(value.as_str());
            if is_polars_selector_name(&name) {
                return ColumnArg::Opaque;
            }
            return ColumnArg::Named(name);
        }
        if is_string_type(&ty) {
            return ColumnArg::Opaque;
        }
        ColumnArg::Expr
    }

    pub fn polars_column_name(&self, expr: &Expr) -> Option<Name> {
        self.polars_string_literal(expr).map(Name::new)
    }

    fn polars_literal<T>(&self, expr: &Expr, get: impl FnOnce(&Lit) -> Option<T>) -> Option<T> {
        let Type::Literal(lit) = self.expr_infer(expr, &self.error_swallower()) else {
            return None;
        };
        get(&lit.value)
    }

    fn polars_bool_literal(&self, expr: &Expr) -> Option<bool> {
        self.polars_literal(expr, |lit| match lit {
            Lit::Bool(value) => Some(*value),
            _ => None,
        })
    }

    fn polars_string_literal(&self, expr: &Expr) -> Option<String> {
        self.polars_literal(expr, |lit| match lit {
            Lit::Str(value) => Some(value.to_string()),
            _ => None,
        })
    }

    fn polars_int_literal(&self, expr: &Expr) -> Option<i64> {
        self.polars_literal(expr, |lit| match lit {
            Lit::Int(value) => value.as_i64(),
            _ => None,
        })
    }

    /// Parse optional new column names and treat absence as an empty rename set.
    fn polars_csv_names(&self, expr: Option<&Expr>) -> Option<Vec<Name>> {
        let Some(expr) = Self::non_none_csv_option(expr) else {
            return Some(Vec::new());
        };
        literal_sequence(expr)?
            .iter()
            .map(|expr| self.polars_column_name(expr))
            .collect()
    }

    /// Distinguish an absent name from a name that cannot be resolved.
    fn polars_optional_csv_name(&self, expr: Option<&Expr>) -> Option<Option<Name>> {
        let expr = Self::non_none_csv_option(expr);
        let name = expr.and_then(|expr| self.polars_column_name(expr));
        (expr.is_none() || name.is_some()).then_some(name)
    }

    /// Parse static projections; an empty sequence selects every column.
    fn polars_csv_selection(&self, expr: &Expr, width: usize) -> Option<CsvColumnSelection> {
        let values = literal_sequence(expr)?;
        if values.is_empty() {
            return Some(CsvColumnSelection::All);
        }
        if let Some(names) = values
            .iter()
            .map(|expr| self.polars_column_name(expr))
            .collect::<Option<SmallSet<_>>>()
            && names.len() == values.len()
        {
            return Some(CsvColumnSelection::Names(names));
        }
        let indices = values
            .iter()
            .map(|expr| usize::try_from(self.polars_int_literal(expr)?).ok())
            .collect::<Option<SmallSet<_>>>()?;
        (indices.len() == values.len() && indices.iter().all(|index| *index < width))
            .then_some(CsvColumnSelection::Indices(indices))
    }

    fn polars_csv_schema(&self, expr: &Expr) -> Option<Vec<(Name, PolarsDType)>> {
        let (form, dict) = self.schema_literal_dict(expr)?;
        if dict.items.is_empty() {
            return Some(Vec::new());
        }
        self.schema_dict_entries(form, dict)?
            .into_iter()
            .map(|(name, dtype)| Some((name, dtype?)))
            .collect()
    }

    /// An omitted option and explicit `None` both represent semantic absence.
    fn non_none_csv_option(expr: Option<&Expr>) -> Option<&Expr> {
        match expr {
            None | Some(Expr::NoneLiteral(_)) => None,
            Some(expr) => Some(expr),
        }
    }

    /// The requested row index is prepended with the UInt32 dtype.
    fn apply_polars_csv_row_index(
        columns: &mut Vec<(Name, PolarsDType)>,
        row_index_name: Option<Name>,
    ) -> bool {
        let Some(name) = row_index_name else {
            return true;
        };
        if columns.iter().any(|(column, _)| *column == name) {
            return false;
        }
        columns.insert(0, (name, PolarsDType::UInt32));
        true
    }

    /// Positional dtypes replace leading column dtypes unless the sequence is too large.
    fn apply_polars_csv_sequence_overrides(
        columns: &mut [(Name, PolarsDType)],
        overrides: CsvDtypeOverrides,
    ) -> bool {
        let CsvDtypeOverrides::Sequence(dtypes) = overrides else {
            return true;
        };
        if dtypes.len() > columns.len() {
            return false;
        }
        for ((_, dtype), override_dtype) in columns.iter_mut().zip(dtypes) {
            *dtype = override_dtype;
        }
        true
    }

    /// New names replace leading columns unless they exceed the width or create duplicates.
    fn apply_polars_csv_new_columns(
        columns: &mut [(Name, PolarsDType)],
        new_columns: Vec<Name>,
    ) -> bool {
        if new_columns.len() > columns.len() {
            return false;
        }
        for ((name, _), replacement) in columns.iter_mut().zip(new_columns) {
            *name = replacement;
        }
        let mut seen = SmallSet::new();
        !columns.iter().any(|(name, _)| !seen.insert(name.clone()))
    }

    fn infer_polars_csv_schema(
        &self,
        arguments: &Arguments,
        function: PolarsCsvFunction,
    ) -> Option<Vec<(Name, PolarsDType)>> {
        match self.polars_csv_options(arguments, function)? {
            PolarsCsvOptions::Read { common, selection } => {
                self.infer_read_csv_schema(common, selection)
            }
            PolarsCsvOptions::Scan {
                common,
                include_file_paths,
            } => self.infer_scan_csv_schema(common, include_file_paths),
        }
    }

    /// This parser converts CSV arguments into their schema transformations.
    fn polars_csv_options(
        &self,
        arguments: &Arguments,
        function: PolarsCsvFunction,
    ) -> Option<PolarsCsvOptions> {
        let mut source_keyword = false;
        let mut schema = None;
        let mut overrides = None;
        let mut selection = None;
        let mut new_columns = None;
        let mut row_index_name = None;
        let mut with_column_names = None;
        let mut include_file_paths = None;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            let slot = match arg.id.as_str() {
                "source" => {
                    if source_keyword {
                        return None;
                    }
                    source_keyword = true;
                    continue;
                }
                "schema" => &mut schema,
                "schema_overrides" => &mut overrides,
                "columns" => &mut selection,
                "new_columns" => &mut new_columns,
                "row_index_name" => &mut row_index_name,
                "with_column_names" => &mut with_column_names,
                "include_file_paths" => &mut include_file_paths,
                _ => continue,
            };
            if slot.replace(&kw.value).is_some() {
                return None;
            }
        }
        if !arguments_are_valid(arguments, 1, None)
            || !matches!(
                extract_argument(arguments, 0, "source")?,
                ArgumentValue::Present(_)
            )
        {
            return None;
        }

        let schema = self.polars_csv_schema(schema?)?;
        let overrides =
            CsvDtypeOverrides::parse(overrides, |expr| self.polars_dtype_from_expr(expr))?;
        let new_columns = self.polars_csv_names(new_columns)?;
        let row_index_name = self.polars_optional_csv_name(row_index_name)?;
        let common = PolarsCsvCommonOptions {
            schema,
            overrides,
            new_columns,
            row_index_name,
        };

        Some(match function {
            PolarsCsvFunction::Read => {
                let selection = match Self::non_none_csv_option(selection) {
                    None => CsvColumnSelection::All,
                    Some(expr) => self.polars_csv_selection(expr, common.schema.len())?,
                };
                PolarsCsvOptions::Read { common, selection }
            }
            PolarsCsvFunction::Scan => {
                if Self::non_none_csv_option(with_column_names).is_some() {
                    return None;
                }
                let include_file_paths = self.polars_optional_csv_name(include_file_paths)?;
                PolarsCsvOptions::Scan {
                    common,
                    include_file_paths,
                }
            }
        })
    }

    /// Eager CSV inference applies overrides before projection and inserts the row index before renaming.
    fn infer_read_csv_schema(
        &self,
        common: PolarsCsvCommonOptions,
        selection: CsvColumnSelection,
    ) -> Option<Vec<(Name, PolarsDType)>> {
        let PolarsCsvCommonOptions {
            schema: mut columns,
            overrides,
            new_columns,
            row_index_name,
        } = common;
        if matches!(&overrides, CsvDtypeOverrides::Sequence(_))
            && !matches!(selection, CsvColumnSelection::All)
        {
            return None;
        }
        if !Self::apply_polars_csv_sequence_overrides(&mut columns, overrides) {
            return None;
        }
        if !matches!(selection, CsvColumnSelection::All) {
            if let CsvColumnSelection::Names(names) = &selection
                && !names
                    .iter()
                    .all(|name| columns.iter().any(|(column, _)| column == name))
            {
                return None;
            }
            columns = columns
                .into_iter()
                .enumerate()
                .filter(|(index, (name, _))| match &selection {
                    CsvColumnSelection::All => true,
                    CsvColumnSelection::Names(names) => names.contains(name),
                    CsvColumnSelection::Indices(indices) => indices.contains(index),
                })
                .map(|(_, column)| column)
                .collect();
        }
        if !Self::apply_polars_csv_row_index(&mut columns, row_index_name)
            || !Self::apply_polars_csv_new_columns(&mut columns, new_columns)
        {
            return None;
        }
        Some(columns)
    }

    /// Lazy CSV inference applies overrides before inserting the row index.
    /// It renames columns before adding the file path.
    fn infer_scan_csv_schema(
        &self,
        common: PolarsCsvCommonOptions,
        include_file_paths: Option<Name>,
    ) -> Option<Vec<(Name, PolarsDType)>> {
        let PolarsCsvCommonOptions {
            schema: mut columns,
            overrides,
            new_columns,
            row_index_name,
        } = common;
        if !Self::apply_polars_csv_sequence_overrides(&mut columns, overrides)
            || !Self::apply_polars_csv_row_index(&mut columns, row_index_name)
            || !Self::apply_polars_csv_new_columns(&mut columns, new_columns)
        {
            return None;
        }
        if let Some(name) = include_file_paths {
            if columns.iter().any(|(column, _)| *column == name) {
                return None;
            }
            columns.push((name, PolarsDType::String));
        }
        Some(columns)
    }

    fn dataframe_data_map<'b>(&self, dict: &'b ExprDict) -> Option<SmallMap<Name, &'b Expr>> {
        let mut map = SmallMap::with_capacity(dict.items.len());
        for item in &dict.items {
            let name = self.polars_column_name(item.key.as_ref()?)?;
            if map.insert(name, &item.value).is_some() {
                return None;
            }
        }
        Some(map)
    }

    fn dataframe_records_map<'b>(
        &self,
        list: &'b ExprList,
    ) -> Option<SmallMap<Name, Vec<&'b Expr>>> {
        let mut columns: SmallMap<Name, Vec<&Expr>> = SmallMap::new();
        for elt in list.elts.iter().take(POLARS_DEFAULT_INFER_SCHEMA_LENGTH) {
            let Expr::Dict(dict) = elt else {
                return None;
            };
            for (name, value) in self.dataframe_data_map(dict)? {
                columns.entry(name).or_default().push(value);
            }
        }
        (!columns.is_empty()).then_some(columns)
    }

    fn is_string_typed(&self, expr: &Expr) -> bool {
        let ty = self.expr_infer(expr, &self.error_swallower());
        is_string_type(&ty)
    }

    fn polars_series_options<'b>(&self, arguments: &'b Arguments) -> Option<SeriesConstruct<'b>> {
        let mut strict = true;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "name" | "values" | "dtype" | "nan_to_null" => {}
                "strict" => strict = self.polars_bool_literal(&kw.value)?,
                _ => return None,
            }
        }
        let values_position = match &arguments.args[..] {
            [first] | [first, _] | [first, _, _] if self.is_string_typed(first) => 1,
            [] | [_] => 0,
            _ => return None,
        };
        let values = extract_argument(arguments, values_position, "values")?.into_option();
        let dtype = match extract_argument(arguments, values_position + 1, "dtype")? {
            ArgumentValue::Present(expr) => Some(self.polars_dtype_from_expr(expr)?),
            ArgumentValue::Missing => None,
        };
        Some(SeriesConstruct {
            values,
            dtype,
            strict,
        })
    }

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
            let name = self.polars_column_name(item.key.as_ref()?)?;
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

    /// Build the instance type represented by `pl.DataFrame[Schema]`.
    pub fn polars_dataframe_schema_annotation(&self, base: &Class, arg: &Expr) -> Option<Type> {
        if !is_polars_dataframe(base) {
            return None;
        }
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

    fn schema_class_entries(&self, expr: &Expr) -> Option<Vec<(Name, Option<PolarsDType>)>> {
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

    /// Optional TypedDict fields are omitted and make the resulting schema partial.
    fn typed_dict_data_columns(
        &self,
        expr: &Expr,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
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

    /// Infer the ordered columns of a DataFrame constructor.
    fn infer_dataframe_schema(
        &self,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
        errors: &ErrorCollector,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
        if construct.columns.is_some() && kind != DataFrameKind::Pandas {
            return None;
        }
        match (&construct.data, &construct.schema) {
            (Some(PolarsData::Records(_)), Some(_)) => None,
            (Some(PolarsData::Records(records)), None) => {
                self.infer_dataframe_records_schema(records, construct, kind, errors)
            }
            (Some(PolarsData::TypedDict(_, _)), Some(_)) => None,
            (Some(PolarsData::TypedDict(columns, completeness)), None) => {
                self.infer_dataframe_typed_dict_schema(columns, *completeness, construct, kind)
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
        completeness: SchemaCompleteness,
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
            completeness,
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
                                kind,
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
        let element_from_data = |name: &Name, value: &Expr| match value {
            Expr::List(ExprList { elts, .. }) => {
                self.dataframe_list_element_type(name, elts.iter(), kind, construct.strict, errors)
            }
            _ => None,
        };
        let Some(schema) = &construct.schema else {
            let data = data?;
            let names: Vec<&Name> = match &construct.columns {
                Some(cols) => cols.iter().collect(),
                None => data.columns.keys().collect(),
            };
            let mut result = Vec::with_capacity(names.len());
            for name in names {
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

    /// Recognize a dict literal or an inline call resolving to `polars.Schema`.
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
                    Type::ClassDef(cls) if RuntimeClass::PolarsSchema.matches(&cls) => {
                        Some((SchemaForm::SchemaClass, dict))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn infer_series_dtype(&self, arguments: &Arguments) -> Option<PolarsDType> {
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

    fn polars_construct_options<'b>(
        &self,
        arguments: &'b Arguments,
    ) -> Option<PolarsConstruct<'b>> {
        let mut overrides = SmallMap::new();
        let mut strict = true;
        let mut columns = None;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "data" | "schema" => {}
                "columns" => {
                    let Expr::List(list) = &kw.value else {
                        return None;
                    };
                    let mut names = Vec::with_capacity(list.elts.len());
                    for elt in &list.elts {
                        let name = self.polars_column_name(elt)?;
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
                        let (Some(key), value) = (&item.key, &item.value) else {
                            return None;
                        };
                        overrides.insert(
                            self.polars_column_name(key)?,
                            self.polars_dtype_from_expr(value)?,
                        );
                    }
                }
                "strict" => strict = self.polars_bool_literal(&kw.value)?,
                _ => return None,
            }
        }
        if !arguments_are_valid(arguments, 2, None) {
            return None;
        }
        let data_expr = extract_argument(arguments, 0, "data")?.into_option();
        let schema_expr = extract_argument(arguments, 1, "schema")?.into_option();
        let data = match data_expr {
            None | Some(Expr::NoneLiteral(_)) => None,
            Some(Expr::Dict(dict)) if dict.items.is_empty() => None,
            Some(Expr::Dict(dict)) => Some(PolarsData::Dict(PolarsDictData {
                columns: self.dataframe_data_map(dict)?,
                range: dict.range(),
            })),
            Some(Expr::List(list)) => Some(PolarsData::Records(self.dataframe_records_map(list)?)),
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

    fn polars_concat_how(&self, keywords: &[Keyword]) -> Option<ConcatHow> {
        let mut how = ConcatHow::Vertical;
        for kw in keywords {
            let arg = kw.arg.as_ref()?;
            if arg.id.as_str() == "how" {
                how = ConcatHow::parse(self.polars_string_literal(&kw.value)?.as_str())?;
            }
        }
        Some(how)
    }

    fn infer_polars_concat(
        &self,
        arguments: &Arguments,
    ) -> Option<(Vec<(Name, PolarsDType)>, SchemaCompleteness)> {
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
                Type::DataFrame(schema) if schema.kind == DataFrameKind::Polars => {
                    Some((schema.columns, schema.completeness))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let completeness = schemas
            .iter()
            .fold(SchemaCompleteness::Complete, |completeness, (_, next)| {
                completeness.combine(*next)
            });
        let (first, rest) = schemas.split_first()?;
        let columns = match how {
            ConcatHow::Vertical => rest
                .iter()
                .all(|(columns, _)| columns == &first.0)
                .then(|| first.0.clone())?,
            ConcatHow::VerticalRelaxed => {
                let names_match = rest.iter().all(|(columns, _)| {
                    columns.len() == first.0.len()
                        && columns.iter().zip(&first.0).all(|((n, _), (m, _))| n == m)
                });
                if !names_match {
                    return None;
                }
                first
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, (name, dtype))| {
                        let folded = rest
                            .iter()
                            .try_fold(*dtype, |acc, (columns, _)| acc.supertype(columns[i].1))?;
                        Some((name.clone(), folded))
                    })
                    .collect::<Option<Vec<_>>>()?
            }
        };
        Some((columns, completeness))
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
                    Type::ClassDef(cls) if RuntimeClass::Date.matches(&cls) => {
                        Some(PolarsDType::Date)
                    }
                    Type::ClassDef(cls) if RuntimeClass::Datetime.matches(&cls) => {
                        Some(PolarsDType::Datetime)
                    }
                    Type::ClassDef(cls) if RuntimeClass::Time.matches(&cls) => {
                        Some(PolarsDType::Time)
                    }
                    Type::ClassDef(cls) if RuntimeClass::Timedelta.matches(&cls) => {
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
        let mut column = PolarsDType::Null;
        for e in elts {
            let element = scalar(e)?;
            if column == PolarsDType::Null {
                column = element;
                continue;
            }
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

    /// Select list-literal columns while rejecting duplicates as Polars does.
    pub fn polars_select_columns(
        &self,
        schema: &DataFrameSchema,
        elts: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let mut names = Vec::with_capacity(elts.len());
        let mut output_names = SmallSet::new();
        let mut has_repeated_name = false;
        for elt in elts {
            let name = self.polars_column_name(elt)?;
            if !output_names.insert(name.clone()) {
                has_repeated_name = true;
            }
            names.push((name, elt.range()));
        }
        for elt in elts {
            self.expr_infer(elt, errors);
        }
        let mut columns = Vec::with_capacity(names.len());
        let mut resolved_names = SmallSet::new();
        for (name, range) in names {
            let Some(dtype) = resolve_column(schema, &name, range, errors) else {
                continue;
            };
            if !resolved_names.insert(name.clone()) {
                report_duplicate_column(&name, range, errors);
            }
            columns.push((name, dtype));
        }
        if has_repeated_name {
            Some(schema.underlying_type())
        } else {
            Some(dataframe_type_with_columns(schema, columns))
        }
    }

    /// Infer the ordered output columns of a Polars `select` call.
    fn polars_select(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, args)?;
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let positional = args
            .args
            .iter()
            .flat_map(positional_elements)
            .collect::<Vec<_>>();
        if let [arg] = &positional[..]
            && let Type::Literal(lit) = &self.expr_infer(arg, &self.error_swallower())
            && let Lit::Str(value) = &lit.value
            && value.as_str() == POLARS_ALL_COLUMNS
        {
            self.expr_infer(arg, errors);
            return Some(base.clone());
        }
        let mut names = Vec::with_capacity(positional.len());
        let mut output_names = SmallSet::new();
        let mut has_opaque = false;
        let mut has_repeated_name = false;
        for &arg in &positional {
            let output = match self.polars_column_arg(arg) {
                ColumnArg::Named(name) => Some((name, true)),
                ColumnArg::Opaque => None,
                ColumnArg::Expr => self.polars_expr_output_name(arg).map(|name| (name, false)),
            };
            let Some((name, is_column)) = output else {
                has_opaque = true;
                names.push(None);
                continue;
            };
            if !output_names.insert(name.clone()) {
                has_repeated_name = true;
            }
            names.push(Some((name, is_column)));
        }
        let mut columns = Vec::with_capacity(names.len());
        let mut resolved_names = SmallSet::new();
        for (output, arg) in names.iter().zip(positional) {
            let Some((name, is_column)) = output else {
                continue;
            };
            let resolved = if *is_column {
                resolve_column(schema, name, arg.range(), errors)
            } else {
                self.eval_polars_expr(arg, schema, errors)
                    .map(ExprValue::dtype)
            };
            if resolved.is_some() && !resolved_names.insert(name.clone()) {
                report_duplicate_column(name, arg.range(), errors);
            }
            if let Some(dtype) = resolved {
                columns.push((name.clone(), dtype));
            } else if !*is_column {
                columns.push((name.clone(), PolarsDType::Unknown));
            }
        }
        if has_opaque {
            None
        } else {
            for arg in &args.args {
                self.expr_infer(arg, errors);
            }
            if !has_repeated_name {
                return Some(dataframe_type_with_columns(schema, columns));
            }
            Some(schema.underlying_type())
        }
    }

    /// Remove statically named columns while preserving order.
    fn polars_drop(&self, base: &Type, args: &Arguments, errors: &ErrorCollector) -> Option<Type> {
        let schema = column_transform_schema(base, args)?;
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let positional = args
            .args
            .iter()
            .flat_map(positional_elements)
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
        for arg in &args.args {
            self.expr_infer(arg, errors);
        }
        for (name, range) in &dropped {
            let _ = resolve_column(schema, name, *range, errors);
        }
        let columns = schema
            .columns
            .iter()
            .filter(|(c, _)| !seen.contains(c))
            .cloned()
            .collect();
        Some(dataframe_type_with_columns(schema, columns))
    }

    /// Rename statically named columns while preserving dtype and order.
    fn polars_rename(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, args)?;
        let [Expr::Dict(mapping)] = &args.args[..] else {
            return None;
        };
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
        for (key, value) in name_exprs {
            self.expr_infer(key, errors);
            self.expr_infer(value, errors);
        }
        for (source, (_, range)) in &renames {
            let _ = resolve_column(schema, source, *range, errors);
        }
        let columns = schema
            .columns
            .iter()
            .map(|(name, ty)| (target(name), *ty))
            .collect();
        Some(dataframe_type_with_columns(schema, columns))
    }

    fn eval_polars_expr(
        &self,
        expr: &Expr,
        schema: &DataFrameSchema,
        errors: &ErrorCollector,
    ) -> Option<ExprValue> {
        match expr {
            Expr::Call(call) => {
                if let Expr::Attribute(attr) = &*call.func {
                    match PolarsExprMethod::parse(attr.attr.id.as_str()) {
                        Some(PolarsExprMethod::Cast) => {
                            self.eval_polars_expr(&attr.value, schema, errors)?;
                            let [target] = &call.arguments.args[..] else {
                                return None;
                            };
                            return self.polars_dtype_from_expr(target).map(ExprValue::Dtype);
                        }
                        Some(PolarsExprMethod::Alias) => {
                            return self.eval_polars_expr(&attr.value, schema, errors);
                        }
                        Some(PolarsExprMethod::Reducer(reducer)) => {
                            let inner = self.eval_polars_expr(&attr.value, schema, errors)?;
                            return Some(ExprValue::Dtype(reducer.output_dtype(inner.dtype())?));
                        }
                        None => {}
                    }
                }
                match self.polars_function(&call.func)? {
                    PolarsFunction::Len => Some(ExprValue::Dtype(PolarsDType::UInt32)),
                    PolarsFunction::Col => {
                        let [arg] = &call.arguments.args[..] else {
                            return None;
                        };
                        let ColumnArg::Named(name) = self.polars_column_arg(arg) else {
                            return None;
                        };
                        resolve_column(schema, &name, arg.range(), errors).map(ExprValue::Dtype)
                    }
                    PolarsFunction::Lit => {
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
                    PolarsFunction::Concat | PolarsFunction::Csv(_) | PolarsFunction::Unmodeled => {
                        None
                    }
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

    fn polars_function(&self, func: &Expr) -> Option<PolarsFunction> {
        PolarsFunction::from_callee(&self.expr_infer(func, &self.error_swallower()))
    }

    /// Prove an expression produces one column, so its output name is well-defined.
    fn polars_expr_has_single_output(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Call(call) => {
                if let Expr::Attribute(attr) = &*call.func
                    && PolarsExprMethod::parse(attr.attr.id.as_str()).is_some()
                {
                    return self.polars_expr_has_single_output(&attr.value);
                }
                match self.polars_function(&call.func) {
                    Some(PolarsFunction::Len) => true,
                    Some(PolarsFunction::Col) => {
                        matches!(&call.arguments.args[..], [arg] if matches!(self.polars_column_arg(arg), ColumnArg::Named(_)))
                    }
                    Some(PolarsFunction::Lit) => matches!(&call.arguments.args[..], [_]),
                    _ => false,
                }
            }
            Expr::BinOp(binop) => {
                self.polars_expr_has_single_output(&binop.left)
                    && self.polars_expr_has_single_output(&binop.right)
            }
            Expr::UnaryOp(unary) => self.polars_expr_has_single_output(&unary.operand),
            Expr::Compare(cmp) => {
                matches!((&*cmp.ops, &*cmp.comparators), ([_], [right]) if self.polars_expr_has_single_output(&cmp.left) && self.polars_expr_has_single_output(right))
            }
            Expr::NumberLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::NoneLiteral(_) => true,
            _ => !self.is_polars_expr_value(expr),
        }
    }

    /// Follow Polars' leftmost-leaf naming, overridden by an outer `alias`.
    fn polars_expr_output_name(&self, expr: &Expr) -> Option<Name> {
        match expr {
            Expr::Call(call) => {
                if let Expr::Attribute(attr) = &*call.func {
                    match PolarsExprMethod::parse(attr.attr.id.as_str()) {
                        Some(PolarsExprMethod::Cast) => {
                            return self.polars_expr_output_name(&attr.value);
                        }
                        Some(PolarsExprMethod::Alias) => {
                            if !self.polars_expr_has_single_output(&attr.value) {
                                return None;
                            }
                            let [arg] = &call.arguments.args[..] else {
                                return None;
                            };
                            return self.polars_string_literal(arg).map(Name::new);
                        }
                        Some(PolarsExprMethod::Reducer(_)) => {
                            return self.polars_expr_output_name(&attr.value);
                        }
                        None => {}
                    }
                }
                match self.polars_function(&call.func)? {
                    PolarsFunction::Len => Some(Name::new_static(POLARS_LEN_OUTPUT_NAME)),
                    PolarsFunction::Col => {
                        let [arg] = &call.arguments.args[..] else {
                            return None;
                        };
                        let ColumnArg::Named(name) = self.polars_column_arg(arg) else {
                            return None;
                        };
                        Some(name)
                    }
                    PolarsFunction::Lit => {
                        let [value] = &call.arguments.args[..] else {
                            return None;
                        };
                        // `pl.lit(series)` takes the runtime Series name.
                        literal_value(value).map(|_| Name::new_static(POLARS_LITERAL_OUTPUT_NAME))
                    }
                    PolarsFunction::Concat | PolarsFunction::Csv(_) | PolarsFunction::Unmodeled => {
                        None
                    }
                }
            }
            Expr::BinOp(binop) => {
                if !self.polars_expr_has_single_output(expr) {
                    return None;
                }
                self.polars_expr_output_name(&binop.left)
            }
            Expr::UnaryOp(unary) => self.polars_expr_output_name(&unary.operand),
            Expr::Compare(cmp) => {
                let ([_], [right]) = (&*cmp.ops, &*cmp.comparators) else {
                    return None;
                };
                if !self.polars_expr_has_single_output(expr) {
                    return None;
                }
                // Python reflects comparisons whose left operand is not a Polars expression.
                if self.is_polars_expr_value(&cmp.left) {
                    self.polars_expr_output_name(&cmp.left)
                } else {
                    self.polars_expr_output_name(right)
                }
            }
            Expr::NumberLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::NoneLiteral(_) => {
                literal_value(expr).map(|_| Name::new_static(POLARS_LITERAL_OUTPUT_NAME))
            }
            _ => None,
        }
    }

    fn is_polars_expr_value(&self, expr: &Expr) -> bool {
        matches!(
            self.expr_infer(expr, &self.error_swallower()),
            Type::ClassType(cls) if is_polars_expr(cls.class_object())
        )
    }

    /// Infer named `with_columns` outputs against the receiver schema.
    fn polars_with_columns(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if !args.args.is_empty() {
            return None;
        }
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Validate names before inference so fallback does not duplicate diagnostics.
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
            self.expr_infer(value, errors);
            let dtype = match self.polars_column_arg(value) {
                ColumnArg::Named(name) => resolve_column(schema, &name, value.range(), errors),
                ColumnArg::Opaque => None,
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
        Some(dataframe_type_with_columns(schema, columns))
    }

    /// A bound `GroupBy` does not expose its receiver schema, so only an inline chain is modeled.
    fn polars_group_by_agg(
        &self,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Expr::Call(group_by) = &*func.value else {
            return None;
        };
        let Expr::Attribute(group_by_func) = &*group_by.func else {
            return None;
        };
        if group_by_func.attr.id.as_str() != "group_by" {
            return None;
        }
        let Type::DataFrame(schema) =
            self.expr_infer(&group_by_func.value, &self.error_swallower())
        else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Validate every output name before emitting diagnostics.
        enum ColumnKind {
            Key,
            Agg,
        }
        let mut outputs = Vec::new();
        for arg in &group_by.arguments.args {
            for elt in positional_elements(arg) {
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
            for elt in positional_elements(arg) {
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
        Some(dataframe_type_with_columns(&schema, columns))
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
                // The aggregated list dtype is unmodeled.
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
        if let Some(function) = self.polars_function(&call.func) {
            return function == PolarsFunction::Len;
        }
        match &*call.func {
            Expr::Attribute(attr) => match PolarsExprMethod::parse(attr.attr.id.as_str()) {
                Some(PolarsExprMethod::Alias | PolarsExprMethod::Cast) => {
                    self.polars_expr_aggregates(&attr.value)
                }
                Some(PolarsExprMethod::Reducer(_)) => true,
                None => false,
            },
            _ => false,
        }
    }

    /// Model the deterministic dtype widening performed by `DataFrame.fill_null`.
    fn polars_fill_null(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let mut arguments_are_static = args.args.len() <= 1;
        let mut value_type = None;
        for arg in &args.args {
            let ty = match arg {
                Expr::Starred(starred) => {
                    arguments_are_static = false;
                    self.expr_infer(&starred.value, errors)
                }
                _ => self.expr_infer(arg, errors),
            };
            if args.args.len() == 1 && !matches!(arg, Expr::Starred(_)) {
                value_type = Some(ty);
            }
        }
        let mut matches_supertype = Some(true);
        let mut seen_matches_supertype = false;
        for kw in &args.keywords {
            let ty = self.expr_infer(&kw.value, errors);
            let Some(arg) = &kw.arg else {
                arguments_are_static = false;
                continue;
            };
            match arg.id.as_str() {
                "value" => {
                    if value_type.is_some() {
                        arguments_are_static = false;
                    } else {
                        value_type = Some(ty);
                    }
                }
                "strategy" | "limit" => {}
                "matches_supertype" => {
                    if seen_matches_supertype {
                        arguments_are_static = false;
                    }
                    seen_matches_supertype = true;
                    match &ty {
                        Type::Literal(lit) => match &lit.value {
                            Lit::Bool(value) => matches_supertype = Some(*value),
                            _ => matches_supertype = None,
                        },
                        _ => matches_supertype = None,
                    }
                }
                _ => arguments_are_static = false,
            }
        }
        if !arguments_are_static || matches_supertype.is_none() {
            return Some(schema.underlying_type());
        }
        if matches_supertype == Some(false) {
            return Some(base.clone());
        }
        let Some(value_type) = value_type.as_ref() else {
            return Some(base.clone());
        };
        let value = FillNullValue::from_type(value_type);
        if value == FillNullValue::IntegerOutsideModel {
            return Some(base.clone());
        }
        let columns = schema
            .columns
            .iter()
            .map(|(name, dtype)| (name.clone(), value.widen_integer(*dtype)))
            .collect();
        Some(dataframe_type_with_columns(schema, columns))
    }

    /// Preserve the schema through row-only transforms.
    fn polars_row_transform(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // A bare `Expr::Starred` is treated as a type form, so infer its value instead.
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

    /// Preserve the receiver schema through `vstack` and `extend`.
    fn polars_row_append(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars || !args.keywords.is_empty() {
            return None;
        }
        let [other_expr] = &args.args[..] else {
            return None;
        };
        if !is_polars_dataframe_type(&self.expr_infer(other_expr, &self.error_swallower())) {
            return None;
        }
        self.expr_infer(other_expr, errors);
        Some(base.clone())
    }

    /// Carry columns between eager and lazy Polars frames.
    fn polars_lazy_collect(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
        conversion: PolarsFrameConversion,
    ) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars || !args.args.is_empty() {
            return None;
        }
        // Delegate keyword validation and the result class to the stub.
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
        match (conversion, result) {
            (PolarsFrameConversion::Lazy, Type::ClassType(cls))
                if is_polars_lazyframe(cls.class_object()) =>
            {
                Some(
                    DataFrameSchema {
                        underlying: cls,
                        ..(**schema).clone()
                    }
                    .to_type(),
                )
            }
            (PolarsFrameConversion::Collect, Type::ClassType(cls))
                if is_polars_dataframe(cls.class_object()) =>
            {
                Some(
                    DataFrameSchema {
                        underlying: cls,
                        ..(**schema).clone()
                    }
                    .to_type(),
                )
            }
            (_, result) => Some(result),
        }
    }

    /// Rewrite all or statically named column dtypes through `DataFrame.cast`.
    fn polars_cast(&self, base: &Type, args: &Arguments, errors: &ErrorCollector) -> Option<Type> {
        let schema = column_transform_schema(base, args)?;
        let [arg] = &args.args[..] else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let columns = match arg {
            Expr::Dict(mapping) => {
                let mut casts: SmallMap<Name, (TextRange, PolarsDType)> =
                    SmallMap::with_capacity(mapping.items.len());
                for item in &mapping.items {
                    let (Some(key), value) = (&item.key, &item.value) else {
                        return None;
                    };
                    let name = self.polars_column_name(key)?;
                    casts.insert(name, (key.range(), self.polars_dtype_from_expr(value)?));
                }
                for (name, (range, _)) in &casts {
                    let _ = resolve_column(schema, name, *range, errors);
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
        Some(dataframe_type_with_columns(schema, columns))
    }

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

    /// Merge schemas for joins with same-name keys and default coalescing.
    fn polars_join(&self, base: &Type, args: &Arguments, errors: &ErrorCollector) -> Option<Type> {
        let Type::DataFrame(schema) = base else {
            return None;
        };
        if schema.kind != DataFrameKind::Polars {
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
                    how = JoinHow::parse(self.polars_string_literal(&kw.value)?.as_str())?;
                }
                _ => return None,
            }
        }
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
        let completeness = match how {
            JoinHow::Semi | JoinHow::Anti => schema.completeness,
            _ => schema.completeness.combine(other.completeness),
        };
        let base_names: SmallSet<Name> =
            base_columns.iter().map(|(name, _)| name.clone()).collect();
        let mut columns = base_columns;
        for (name, ty) in other_columns {
            let out = if base_names.contains(&name) {
                Name::new(format!("{name}{POLARS_DEFAULT_JOIN_SUFFIX}"))
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
        self.expr_infer(other_expr, errors);
        if let Some(on) = on {
            self.expr_infer(on, errors);
        }
        Some(dataframe_type_with_columns_and_completeness(
            schema,
            columns,
            completeness,
        ))
    }

    /// Append another Polars frame's non-overlapping columns.
    fn polars_hstack(
        &self,
        base: &Type,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, args)?;
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        let [other_expr] = &args.args[..] else {
            return None;
        };
        let Type::DataFrame(other) = self.expr_infer(other_expr, &self.error_swallower()) else {
            return None;
        };
        if other.kind != DataFrameKind::Polars {
            return None;
        }
        let mut columns = schema.columns.clone();
        columns.extend(other.columns.iter().cloned());
        let mut seen = SmallSet::new();
        if columns.iter().any(|(name, _)| !seen.insert(name.clone())) {
            return None;
        }
        self.expr_infer(other_expr, errors);
        let completeness = schema.completeness.combine(other.completeness);
        Some(dataframe_type_with_columns_and_completeness(
            schema,
            columns,
            completeness,
        ))
    }

    /// Apply a binding-time column mutation to the receiver schema.
    fn polars_in_place_column_mutation(
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
                if is_polars_series(&cls) || RuntimeClass::PolarsDataFrameSeries.matches(&cls)
        )
    }

    /// Return the statically named column's element dtype.
    fn polars_get_column(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = series_method_schema(base)?;
        let name_expr = get_column_name_arg(args)?;
        let name = self.polars_column_name(name_expr)?;
        let dtype = resolve_column(schema, &name, name_expr.range(), errors);
        Some(self.wrap_series_method(schema, func, args, dtype, errors))
    }

    fn to_series_index(&self, args: &Arguments) -> Option<i128> {
        if !arguments_are_valid(args, 1, Some(&["index"])) {
            return None;
        }
        match extract_argument(args, 0, "index")? {
            ArgumentValue::Missing => Some(0),
            ArgumentValue::Present(expr) => self.polars_int_literal(expr).map(i128::from),
        }
    }

    /// Return a statically indexed column's element dtype.
    fn polars_to_series(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = series_method_schema(base)?;
        let index = self.to_series_index(args)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pow_dtype(left: PolarsDType, right: PolarsDType) -> Option<PolarsDType> {
        pow(ExprValue::Dtype(left), ExprValue::Dtype(right)).map(ExprValue::dtype)
    }

    #[test]
    fn test_pow_dtype_matrix_matches_polars_runtime() {
        use PolarsDType::*;

        let dtypes = [
            Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, Float32,
            Float64,
        ];
        let expected = [
            [
                Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Float32, Float64,
            ],
            [
                Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Float32,
                Float64,
            ],
            [
                Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Float32,
                Float64,
            ],
            [
                Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Float32,
                Float64,
            ],
            [
                Int128, Int128, Int128, Int128, Int128, Int128, Int128, Int128, Int128, Int128,
                Float32, Float64,
            ],
            [
                UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, Float32,
                Float64,
            ],
            [
                UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16,
                Float32, Float64,
            ],
            [
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                Float32, Float64,
            ],
            [
                UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                Float32, Float64,
            ],
            [
                UInt128, UInt128, UInt128, UInt128, UInt128, UInt128, UInt128, UInt128, UInt128,
                UInt128, Float32, Float64,
            ],
            [
                Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
                Float32, Float32, Float32,
            ],
            [
                Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
                Float64, Float64, Float64,
            ],
        ];

        for (left, expected_row) in dtypes.into_iter().zip(expected) {
            for (right, expected_dtype) in dtypes.into_iter().zip(expected_row) {
                assert_eq!(
                    pow_dtype(left, right),
                    Some(expected_dtype),
                    "{} ** {}",
                    left.name(),
                    right.name(),
                );
            }
        }
    }

    #[test]
    fn test_pow_rejects_every_nonnumeric_dtype() {
        use PolarsDType::*;

        let nonnumeric = [
            Boolean, String, Binary, Date, Datetime, Duration, Time, Null, Unknown,
        ];
        for dtype in nonnumeric {
            assert_eq!(pow_dtype(dtype, Int8), None);
            assert_eq!(pow_dtype(Int8, dtype), None);
        }
    }
}
