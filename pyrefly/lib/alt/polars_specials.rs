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
use crate::types::class::Class;

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
    /// Infer a column schema from a `pl.DataFrame({...})` dict literal, or `None` to
    /// fall back to plain construction.
    ///
    /// Extraction is purely syntactic and never infers the element expressions.
    /// Duplicate keys yield `None`: Python keeps only the last value for a repeated
    /// key, so one column per syntactic entry would misdescribe the runtime schema.
    pub fn infer_dataframe_schema(
        &self,
        dict: &ExprDict,
        kind: DataFrameKind,
        overrides: &SmallMap<Name, PolarsDType>,
        strict: bool,
        errors: &ErrorCollector,
    ) -> Option<Vec<(Name, PolarsDType)>> {
        if dict.items.is_empty() {
            return None;
        }
        let mut columns = Vec::with_capacity(dict.items.len());
        let mut seen = SmallSet::new();
        for item in &dict.items {
            let Some(Expr::StringLiteral(key)) = &item.key else {
                return None;
            };
            let name = Name::new(key.value.to_str());
            if !seen.insert(name.clone()) {
                return None;
            }
            let Expr::List(ExprList { elts, .. }) = &item.value else {
                return None;
            };
            // An explicit `schema_overrides` dtype is authoritative, so Polars casts the column to
            // it regardless of the element values.
            let element = match overrides.get(&name) {
                Some(dtype) => *dtype,
                None => {
                    self.dataframe_list_element_type(&name, elts, kind.clone(), strict, errors)?
                }
            };
            columns.push((name, element));
        }
        Some(columns)
    }

    /// The dtype overrides and strictness requested by a DataFrame call, or `None` to fall back
    /// to ordinary call checking for a form we do not model.
    pub fn polars_construct_options(
        &self,
        keywords: &[Keyword],
    ) -> Option<(SmallMap<Name, PolarsDType>, bool)> {
        let mut overrides = SmallMap::new();
        let mut strict = true;
        for kw in keywords {
            let Some(arg) = &kw.arg else {
                return None;
            };
            match arg.id.as_str() {
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
        Some((overrides, strict))
    }

    /// The column's Polars dtype, or `None` to fall back to plain construction.
    /// The column anchors to its first non-null element; later elements must coerce into it.
    /// Polars reports a mismatch; pandas coerces the column, so it falls back.
    fn dataframe_list_element_type(
        &self,
        name: &Name,
        elts: &[Expr],
        kind: DataFrameKind,
        strict: bool,
        errors: &ErrorCollector,
    ) -> Option<PolarsDType> {
        let scalar = |e: &Expr| match e {
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(_),
                ..
            }) => Some(PolarsDType::Int64),
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
        let Some((first, rest)) = elts.split_first() else {
            return Some(PolarsDType::Unknown);
        };
        if !strict {
            // strict=False widens to the elements' common supertype instead of erroring.
            let mut acc = scalar(first)?;
            for e in rest {
                acc = acc.supertype(scalar(e)?)?;
            }
            // We do not model timezones, so a naive/tz-aware mix (which Polars rejects under
            // strict=False) is indistinguishable here; fall back rather than assert `Datetime`.
            if acc == PolarsDType::Datetime && !rest.is_empty() {
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
}
