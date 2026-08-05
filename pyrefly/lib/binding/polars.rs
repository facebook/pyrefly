/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Binding-time recognition of Polars in-place column mutations.

use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Number;
use ruff_python_ast::name::Name;

/// How an in-place mutation changes a Polars frame's tracked column set.
#[derive(Clone, Debug)]
pub enum PolarsMutationKind {
    /// Preserve known columns but make the schema partial.
    Add,
    /// Discard the schema because the replaced column is unknown.
    Replace,
    /// Insert a known name and index after resolving the callee as `pl.Series`.
    Insert(Name, usize, Box<Expr>),
}

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

/// Classify mutations that may change a bound frame's columns.
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
