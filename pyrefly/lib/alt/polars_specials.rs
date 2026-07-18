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

use pyrefly_types::data_frame::DataFrameSchema;
use pyrefly_types::types::Type;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
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

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Infer a column schema from a `pl.DataFrame({...})` dict literal, or `None` to
    /// fall back to plain construction.
    ///
    /// Extraction is purely syntactic and never infers the element expressions.
    /// Duplicate keys yield `None`: Python keeps only the last value for a repeated
    /// key, so one column per syntactic entry would misdescribe the runtime schema.
    pub fn infer_polars_schema(
        &self,
        dict: &ExprDict,
        errors: &ErrorCollector,
    ) -> Option<Vec<(Name, Type)>> {
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
            let element = self.polars_list_element_type(&name, elts, errors)?;
            columns.push((name, element));
        }
        Some(columns)
    }

    /// The column's modeled element type, or `None` to fall back to plain construction.
    /// Mirrors Polars: the column takes its first element's dtype, and a later element that
    /// does not fit is a runtime error we report before falling back. An empty list is
    /// `Unknown`; a non-literal element falls back silently, as does `complex` since Polars
    /// has no complex dtype.
    fn polars_list_element_type(
        &self,
        name: &Name,
        elts: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let scalar = |e: &Expr| match e {
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(_),
                ..
            }) => Some(self.stdlib.int()),
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Float(_),
                ..
            }) => Some(self.stdlib.float()),
            Expr::BooleanLiteral(_) => Some(self.stdlib.bool()),
            Expr::StringLiteral(_) => Some(self.stdlib.str()),
            Expr::BytesLiteral(_) => Some(self.stdlib.bytes()),
            _ => None,
        };
        let Some((first, rest)) = elts.split_first() else {
            return Some(self.heap.mk_any_implicit());
        };
        let column = self.heap.mk_class_type(scalar(first)?.clone());
        for e in rest {
            let element = self.heap.mk_class_type(scalar(e)?.clone());
            if !self.is_subset_eq(&element, &column) {
                self.error(
                    errors,
                    e.range(),
                    ErrorKind::ColumnTypeMismatch,
                    format!(
                        "Polars builds column `{name}` with type `{}` from its first element, so a `{}` element does not fit. Use one dtype for the column or pass an explicit `schema`.",
                        self.for_display(column.clone()),
                        self.for_display(element.clone()),
                    ),
                );
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
                    Some((_, ty)) => Some((name, ty.clone())),
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
            }
            .to_type(),
        )
    }

    /// Model `df.select("a", "b")` as a new schema with the named columns in argument order.
    /// The caller passes the already-inferred receiver type, so the receiver is never inferred
    /// twice. Falls back with `None` unless the receiver carries a schema, the method is
    /// `select`, and every argument is a positional string literal.
    pub fn polars_select(
        &self,
        base: &Type,
        func: &ExprAttribute,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let schema = column_transform_schema(base, func, "select", args)?;
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
            if !schema.columns.iter().any(|(c, _)| c == name) {
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
            .map(|(name, ty)| (target(name), ty.clone()))
            .collect();
        Some(
            DataFrameSchema {
                underlying: schema.underlying.clone(),
                columns,
                completeness: schema.completeness.clone(),
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
            let unknown = self.heap.mk_any_implicit();
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
            }
            .to_type(),
        )
    }
}
