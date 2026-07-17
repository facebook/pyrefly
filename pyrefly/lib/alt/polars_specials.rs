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
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Number;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::class::Class;

pub fn is_polars_dataframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.dataframe.frame", "DataFrame")
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
}
