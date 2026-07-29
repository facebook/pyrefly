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

use pyrefly_types::data_frame::DataFrameKind;
use pyrefly_types::data_frame::DataFrameSchema;
use pyrefly_types::polars_dtype::PolarsDType;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Keyword;
use ruff_python_ast::Number;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::literal::Lit;

pub fn is_polars_dataframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.dataframe.frame", "DataFrame")
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

/// Map a `pl.<DType>` expression such as `pl.Int8` or `pl.Datetime("us")` to its dtype, or `None`
/// for anything that is not a recognized dtype so the caller falls back rather than guessing.
fn polars_dtype_from_expr(e: &Expr) -> Option<PolarsDType> {
    let name = match e {
        Expr::Attribute(a) => a.attr.id.as_str(),
        Expr::Call(c) => match &*c.func {
            Expr::Attribute(a) => a.attr.id.as_str(),
            _ => return None,
        },
        _ => return None,
    };
    Some(match name {
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
        "String" | "Utf8" => PolarsDType::String,
        "Binary" => PolarsDType::Binary,
        "Date" => PolarsDType::Date,
        "Datetime" => PolarsDType::Datetime,
        "Duration" => PolarsDType::Duration,
        "Time" => PolarsDType::Time,
        _ => return None,
    })
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

/// Whether a callee is the module-level `polars.concat`, seen through any `Forall`/`Overload`
/// wrapper via `callee_kind`.
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

/// The two data shapes column inference models: a dict-of-columns or list-of-dict rows.
#[derive(Clone, Copy)]
enum PolarsData<'b> {
    Dict(&'b ExprDict),
    Records(&'b ExprList),
}

/// A DataFrame constructor call reduced to the pieces column inference needs.
/// `data` is `None` when absent or empty; `schema` is the ordered `schema=` column set, each
/// column pinned to a dtype or `None` to defer to data inference.
pub struct PolarsConstruct<'b> {
    data: Option<PolarsData<'b>>,
    schema: Option<Vec<(Name, Option<PolarsDType>)>>,
    overrides: SmallMap<Name, PolarsDType>,
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

/// A `schema=` dict literal as an ordered column list, each pinned to a `pl.<DType>` or `None`
/// to defer to data inference. `None` (fall back) for an empty dict or an unrecognized entry.
fn schema_dict_entries(dict: &ExprDict) -> Option<Vec<(Name, Option<PolarsDType>)>> {
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
            Expr::NoneLiteral(_) => None,
            value => Some(polars_dtype_from_expr(value)?),
        };
        entries.push((name, dtype));
    }
    Some(entries)
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

/// The key names from an `on=` argument, each with its literal range for error reporting, or `None`
/// when it is not a string literal or a list/tuple of string literals.
fn join_key_names(on: &Expr) -> Option<Vec<(Name, TextRange)>> {
    let elts = match on {
        Expr::StringLiteral(s) => return Some(vec![(Name::new(s.value.to_str()), on.range())]),
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        _ => return None,
    };
    elts.iter()
        .map(|elt| match elt {
            Expr::StringLiteral(s) => Some((Name::new(s.value.to_str()), elt.range())),
            _ => None,
        })
        .collect()
}

/// Whether a `select`/`drop` string is a `pl.col` selector (`"*"` or `"^regex$"`) rather than an
/// exact column name.
fn is_polars_selector_string(arg: &Expr) -> bool {
    let Expr::StringLiteral(s) = arg else {
        return false;
    };
    let value = s.value.to_str();
    value == "*" || (value.starts_with('^') && value.ends_with('$'))
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Infer a column schema for a DataFrame constructor call, or `None` to fall back to plain
    /// construction. Purely syntactic; never infers the element expressions.
    ///
    /// With `schema=` the column set is authoritative and ordered: each column takes its
    /// `schema_overrides` dtype, else its schema dtype, else defers to data inference. Data must
    /// name the same columns. Without `schema=`, data order defines the columns.
    pub fn infer_dataframe_schema(
        &self,
        construct: &PolarsConstruct,
        kind: DataFrameKind,
        errors: &ErrorCollector,
    ) -> Option<Vec<(Name, PolarsDType)>> {
        let data = match construct.data {
            Some(PolarsData::Dict(dict)) => Some(dataframe_data_map(dict)?),
            None => None,
            // Records fold the supertype over each column's per-row values (as if strict=False), so a
            // column never errors. Polars-only, and records with a `schema=` are not modeled: fall back.
            Some(PolarsData::Records(list)) => {
                if kind != DataFrameKind::Polars || construct.schema.is_some() {
                    return None;
                }
                let columns = dataframe_records_map(list)?
                    .into_iter()
                    .map(|(name, values)| {
                        let element = match construct.overrides.get(&name) {
                            Some(dtype) => *dtype,
                            None => self
                                .dataframe_list_element_type(
                                    &name,
                                    values.iter().copied(),
                                    kind.clone(),
                                    false,
                                    errors,
                                )
                                .unwrap_or(PolarsDType::Unknown),
                        };
                        (name, element)
                    })
                    .collect();
                return Some(columns);
            }
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
            let mut columns = Vec::with_capacity(data.len());
            for (name, value) in &data {
                let element = if let Some(dtype) = construct.overrides.get(name) {
                    *dtype
                } else {
                    match element_from_data(name, value) {
                        Some(dtype) => dtype,
                        None if kind == DataFrameKind::Polars => PolarsDType::Unknown,
                        None => return None,
                    }
                };
                columns.push((name.clone(), element));
            }
            return Some(columns);
        };
        if kind != DataFrameKind::Polars {
            return None;
        }
        // Data and schema must name the same columns at runtime.
        if let Some(data) = &data
            && (data.len() != schema.len() || schema.iter().any(|(n, _)| !data.contains_key(n)))
        {
            return None;
        }
        let columns = schema
            .iter()
            .map(|(name, dtype)| {
                let element = if let Some(dtype) = construct.overrides.get(name) {
                    *dtype
                } else if let Some(dtype) = dtype {
                    *dtype
                } else {
                    match data.as_ref().and_then(|m| m.get(name).copied()) {
                        Some(value) => {
                            element_from_data(name, value).unwrap_or(PolarsDType::Unknown)
                        }
                        None => PolarsDType::Null,
                    }
                };
                (name.clone(), element)
            })
            .collect();
        Some(columns)
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

    /// Reduce a DataFrame constructor call to a `PolarsConstruct`, or `None` to fall back to plain
    /// construction. `data` and `schema` each come from their positional slot or keyword, not both.
    pub fn polars_construct_options<'b>(
        &self,
        arguments: &'b Arguments,
    ) -> Option<PolarsConstruct<'b>> {
        let mut overrides = SmallMap::new();
        let mut strict = true;
        let mut data_keyword: Option<&Expr> = None;
        let mut schema_keyword: Option<&Expr> = None;
        for kw in &arguments.keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
                "data" => data_keyword = Some(&kw.value),
                "schema" => schema_keyword = Some(&kw.value),
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
                            polars_dtype_from_expr(value)?,
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
            Some(Expr::Dict(dict)) => Some(PolarsData::Dict(dict)),
            Some(Expr::List(list)) => Some(PolarsData::Records(list)),
            Some(_) => return None,
        };
        let schema = match schema_expr {
            None | Some(Expr::NoneLiteral(_)) => None,
            Some(Expr::Dict(dict)) => Some(schema_dict_entries(dict)?),
            Some(_) => return None,
        };
        Some(PolarsConstruct {
            data,
            schema,
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
        let scalar = |e: &Expr| match e {
            // An integer literal that fits in i64 is Int64; past i64 the runtime dtype is data-shape
            // dependent (UInt64 or Int128), so we degrade rather than claim a wrong Int64.
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(i),
                ..
            }) => i.as_i64().map(|_| PolarsDType::Int64),
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Float(_),
                ..
            }) => Some(PolarsDType::Float64),
            Expr::BooleanLiteral(_) => Some(PolarsDType::Boolean),
            Expr::StringLiteral(_) => Some(PolarsDType::String),
            Expr::BytesLiteral(_) => Some(PolarsDType::Binary),
            // `None` is `Null` in Polars only; pandas coerces it (int-with-`None` → `float64`),
            // which we do not model, so fall back there.
            Expr::NoneLiteral(_) if kind == DataFrameKind::Polars => Some(PolarsDType::Null),
            // Resolve the callee class, not the element type: a variable typed `date` may hold a
            // `datetime` subclass at runtime, so only a direct constructor call pins the dtype.
            Expr::Call(call) if kind == DataFrameKind::Polars => {
                match self.expr_infer(&call.func, &self.error_swallower()) {
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
                }
            }
            _ => None,
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
    /// Falls back with `None` when an element is not a string literal or when a name repeats,
    /// since Polars rejects duplicate column selection at runtime. An absent name reports the
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
            let Expr::StringLiteral(key) = elt else {
                return None;
            };
            let name = Name::new(key.value.to_str());
            if !seen.insert(name.clone()) {
                return None;
            }
            names.push((name, elt.range()));
        }
        let columns = names
            .into_iter()
            .filter_map(
                |(name, range)| match schema.columns.iter().find(|(c, _)| *c == name) {
                    Some((_, ty)) => Some((name, *ty)),
                    None => {
                        errors
                            .error_builder(
                                range,
                                ErrorKind::UnknownColumn,
                                format!("Column `{name}` is not in the DataFrame schema"),
                            )
                            .emit();
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

    /// Model `df.select("a", "b")` as a new schema with the named columns in argument order. A lone
    /// `"*"` keeps the schema unchanged since it selects every column. Falls back with `None` unless
    /// every argument is a positional string literal.
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
        if let [Expr::StringLiteral(s)] = &args.args[..]
            && s.value.to_str() == "*"
        {
            return Some(base.clone());
        }
        if args.args.iter().any(is_polars_selector_string) {
            return None;
        }
        self.polars_select_columns(schema, &args.args, errors)
    }

    /// Model `df.drop("a", "b")` as a new schema with the named columns removed, order preserved.
    /// Falls back with `None` unless every argument is a positional string literal, and an unknown
    /// name errors only after a schema is committed. Duplicate names are de-duplicated, unlike `select`.
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
        if args.args.iter().any(is_polars_selector_string) {
            return None;
        }
        let mut dropped: Vec<(Name, TextRange)> = Vec::with_capacity(args.args.len());
        let mut seen = SmallSet::new();
        for arg in &args.args {
            let Expr::StringLiteral(key) = arg else {
                return None;
            };
            let name = Name::new(key.value.to_str());
            if seen.insert(name.clone()) {
                dropped.push((name, arg.range()));
            }
        }
        for (name, range) in &dropped {
            if !schema.has_column(name) {
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
        for item in &mapping.items {
            let (Some(Expr::StringLiteral(src)), Expr::StringLiteral(dest)) =
                (&item.key, &item.value)
            else {
                return None;
            };
            let source = Name::new(src.value.to_str());
            if renames
                .insert(source, (Name::new(dest.value.to_str()), src.range()))
                .is_some()
            {
                return None;
            }
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
        for (source, (_, range)) in &renames {
            if !schema.has_column(source) {
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

    /// Model `df.with_columns(x=..., y=...)` as a new schema, overwriting a matching column
    /// in place or appending a new one with an `Unknown` element type since the value type is
    /// not modeled. Falls back with `None` unless every argument is a keyword with a name.
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
        let mut columns = schema.columns.clone();
        for (name, value) in named {
            // Infer the value to surface type errors inside it; its type is unused.
            self.expr_infer(value, errors);
            let unknown = PolarsDType::Unknown;
            match columns.iter_mut().find(|(c, _)| *c == name) {
                Some((_, ty)) => *ty = unknown,
                None => columns.push((name, unknown)),
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

    /// Model row-only transforms (`filter`, `sort`, `fill_null`) as returning the receiver's
    /// schema unchanged, since they reorder rows or replace values without touching the column
    /// set or its types. Falls back with `None` for a receiver that carries no schema.
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
        if !matches!(func.attr.id.as_str(), "filter" | "sort" | "fill_null") {
            return None;
        }
        // Pandas `filter` selects columns and has no `sort`/`fill_null`, so this is Polars-only.
        if schema.kind != DataFrameKind::Polars {
            return None;
        }
        // Infer the arguments so type errors inside them surface; the schema is unchanged.
        for arg in args.args.iter() {
            self.expr_infer(arg, errors);
        }
        for kw in args.keywords.iter() {
            self.expr_infer(&kw.value, errors);
        }
        Some(base.clone())
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
                        (key.range(), polars_dtype_from_expr(value)?),
                    );
                }
                for (name, (range, _)) in &casts {
                    if !schema.has_column(name) {
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
                let dtype = polars_dtype_from_expr(arg)?;
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
            (_, Some(on)) => join_key_names(on)?,
        };
        let Type::DataFrame(other) = self.expr_infer(other_expr, &self.error_swallower()) else {
            return None;
        };
        // A key absent from either frame makes the join malformed, so report it at its declaration
        // site and fall back, matching the `drop`/`cast`/`rename` convention.
        for (name, range) in &keys {
            if !schema.has_column(name) || !other.has_column(name) {
                errors
                    .error_builder(
                        *range,
                        ErrorKind::UnknownColumn,
                        format!("Column `{name}` is not in the DataFrame schema"),
                    )
                    .emit();
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
}
