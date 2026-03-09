/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Emits `.pyi` stub syntax from ruff AST nodes.
//!
//! Each `emit_*` function writes PEP 484-compliant stub text into an
//! output `String` buffer. Source annotations are preserved verbatim
//! by slicing the original source at AST node ranges.

use std::collections::HashSet;

use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::Operator;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;

use crate::export::definitions::Definitions;
use crate::stubgen::StubgenOptions;
use crate::stubgen::visibility::VisibilityFilter;

/// Extract the source text for an AST node's range.
fn source_at<'a>(source: &'a str, node: &impl Ranged) -> &'a str {
    let range = node.range();
    &source[range.start().to_usize()..range.end().to_usize()]
}

/// Returns `true` when `value` is a call to a type-variable constructor
/// (`TypeVar`, `ParamSpec`, `TypeVarTuple`, `NewType`, `NamedTuple`,
/// `TypedDict`). These assignments are preserved verbatim in stubs.
fn is_type_var_call(value: &Expr) -> bool {
    if let Expr::Call(call) = value
        && let Expr::Name(name) = &*call.func
    {
        return matches!(
            name.id.as_str(),
            "TypeVar" | "ParamSpec" | "TypeVarTuple" | "NewType" | "NamedTuple" | "TypedDict"
        );
    }
    false
}

/// Returns `true` when `value` looks like an old-style type alias RHS --
/// a subscript (`List[int]`) or a union pipe (`int | str`). These are
/// emitted verbatim rather than collapsed to `Any`.
fn is_type_alias_value(value: &Expr) -> bool {
    match value {
        Expr::Subscript(_) => true,
        Expr::BinOp(op) if op.op == Operator::BitOr => true,
        _ => false,
    }
}

fn has_overload_decorator(func: &StmtFunctionDef) -> bool {
    func.decorator_list.iter().any(|d| {
        matches!(
            &d.expression,
            Expr::Name(ExprName { id, .. }) if id.as_str() == "overload"
        )
    })
}

/// Collect the names of all functions that have at least one `@overload`
/// variant in `stmts`. Used to drop the non-overloaded implementation.
pub(crate) fn collect_overloaded_names(stmts: &[Stmt]) -> HashSet<String> {
    let mut names = HashSet::new();
    for stmt in stmts {
        if let Stmt::FunctionDef(func) = stmt
            && has_overload_decorator(func)
        {
            names.insert(func.name.to_string());
        }
    }
    names
}

/// Returns `true` if `stmt` is the non-overloaded implementation of an
/// overloaded function (i.e. it should be dropped from the stub).
pub(crate) fn is_overload_impl(stmt: &Stmt, overloaded: &HashSet<String>) -> bool {
    if let Stmt::FunctionDef(func) = stmt {
        overloaded.contains(func.name.as_str()) && !has_overload_decorator(func)
    } else {
        false
    }
}

/// Emit a single top-level (or class-body) statement into the output buffer.
/// Returns `true` if something was written.
pub(crate) fn emit_stmt(
    source: &str,
    stmt: &Stmt,
    filter: &VisibilityFilter,
    options: &StubgenOptions,
    out: &mut String,
    indent: &str,
    defs: &Definitions,
) -> bool {
    match stmt {
        Stmt::Import(_) | Stmt::ImportFrom(_) => {
            out.push_str(indent);
            out.push_str(source_at(source, stmt));
            out.push('\n');
            true
        }
        Stmt::FunctionDef(func) => {
            let name = func.name.as_str();
            if !filter.should_include(name, options.include_private) {
                return false;
            }
            emit_function(source, func, out, indent);
            true
        }
        Stmt::ClassDef(class) => {
            let name = class.name.as_str();
            if !filter.should_include(name, options.include_private) {
                return false;
            }
            emit_class(source, class, options, out, indent, defs);
            true
        }
        Stmt::AnnAssign(ann) => {
            if let Expr::Name(ExprName { id, .. }) = &*ann.target {
                if !filter.should_include(id.as_str(), options.include_private) {
                    return false;
                }
                out.push_str(indent);
                out.push_str(id.as_str());
                out.push_str(": ");
                out.push_str(source_at(source, &*ann.annotation));
                if ann.value.is_some() {
                    out.push_str(" = ...");
                }
                out.push('\n');
                return true;
            }
            false
        }
        Stmt::Assign(assign) => {
            if let [Expr::Name(ExprName { id, .. })] = assign.targets.as_slice() {
                // __all__, TypeVar calls, and type aliases are type-level
                // declarations that stubs always need regardless of visibility.
                if id.as_str() == "__all__"
                    || is_type_var_call(&assign.value)
                    || is_type_alias_value(&assign.value)
                {
                    out.push_str(indent);
                    out.push_str(source_at(source, stmt));
                    out.push('\n');
                    return true;
                }
                if !filter.should_include(id.as_str(), options.include_private) {
                    return false;
                }
                out.push_str(indent);
                out.push_str(id.as_str());
                out.push_str(": Any = ...\n");
                return true;
            }
            false
        }
        Stmt::TypeAlias(_) => {
            out.push_str(indent);
            out.push_str(source_at(source, stmt));
            out.push('\n');
            true
        }
        _ => false,
    }
}

/// Emit a function stub: `def name(params) -> RetType: ...`
fn emit_function(source: &str, func: &StmtFunctionDef, out: &mut String, indent: &str) {
    for decorator in &func.decorator_list {
        out.push_str(indent);
        out.push('@');
        out.push_str(source_at(source, &decorator.expression));
        out.push('\n');
    }

    out.push_str(indent);
    if func.is_async {
        out.push_str("async ");
    }
    out.push_str("def ");
    out.push_str(func.name.as_str());
    out.push('(');

    emit_parameters(source, &func.parameters, out);

    out.push(')');

    if let Some(returns) = &func.returns {
        out.push_str(" -> ");
        out.push_str(source_at(source, &**returns));
    }

    out.push_str(": ...\n");
}

/// Emit function parameters, preserving source annotations and defaults as `...`.
fn emit_parameters(source: &str, params: &ruff_python_ast::Parameters, out: &mut String) {
    let mut first = true;
    let mut needs_separator = false;

    // Positional-only parameters
    for param in &params.posonlyargs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }
    if !params.posonlyargs.is_empty() {
        needs_separator = true;
    }

    // Regular (positional-or-keyword) parameters
    for param in &params.args {
        if needs_separator {
            out.push_str(", /");
            needs_separator = false;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }
    if needs_separator && params.args.is_empty() {
        if !first {
            out.push_str(", ");
        }
        out.push('/');
        first = false;
    }

    // *args
    if let Some(vararg) = &params.vararg {
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('*');
        out.push_str(vararg.name.as_str());
        if let Some(ann) = &vararg.annotation {
            out.push_str(": ");
            out.push_str(source_at(source, &**ann));
        }
    } else if !params.kwonlyargs.is_empty() {
        // Bare `*` to signal keyword-only params follow
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('*');
    }

    // Keyword-only parameters
    for param in &params.kwonlyargs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }

    // **kwargs
    if let Some(kwarg) = &params.kwarg {
        if !first {
            out.push_str(", ");
        }
        out.push_str("**");
        out.push_str(kwarg.name.as_str());
        if let Some(ann) = &kwarg.annotation {
            out.push_str(": ");
            out.push_str(source_at(source, &**ann));
        }
    }
}

/// Emit a single parameter with its annotation (from source) and default as `...`.
fn emit_param(source: &str, param: &ParameterWithDefault, out: &mut String) {
    out.push_str(param.parameter.name.as_str());
    if let Some(ann) = &param.parameter.annotation {
        out.push_str(": ");
        out.push_str(source_at(source, &**ann));
    }
    if param.default.is_some() {
        if param.parameter.annotation.is_some() {
            out.push_str(" = ...");
        } else {
            out.push_str("=...");
        }
    }
}

/// Emit a class stub with its bases and nested method/variable stubs.
fn emit_class(
    source: &str,
    class: &StmtClassDef,
    options: &StubgenOptions,
    out: &mut String,
    indent: &str,
    defs: &Definitions,
) {
    for decorator in &class.decorator_list {
        out.push_str(indent);
        out.push('@');
        out.push_str(source_at(source, &decorator.expression));
        out.push('\n');
    }

    out.push_str(indent);
    out.push_str("class ");
    out.push_str(class.name.as_str());

    if let Some(type_params) = &class.type_params {
        out.push_str(source_at(source, &**type_params));
    }

    if let Some(arguments) = &class.arguments
        && (!arguments.args.is_empty() || !arguments.keywords.is_empty())
    {
        out.push_str(source_at(source, &**arguments));
    }

    out.push_str(":\n");

    let child_indent = format!("{indent}    ");
    let mut has_body = false;
    let overloaded = collect_overloaded_names(&class.body);

    for stmt in &class.body {
        if is_overload_impl(stmt, &overloaded) {
            continue;
        }
        // Inside a class, we don't filter by module-level __all__ -- all
        // class members are part of the class's stub.
        let class_filter = VisibilityFilter::Inferred;
        if emit_stmt(
            source,
            stmt,
            &class_filter,
            options,
            out,
            &child_indent,
            defs,
        ) {
            has_body = true;
        }
    }

    if !has_body {
        out.push_str(&child_indent);
        out.push_str("...\n");
    }
}

/// Check whether a statement will produce a literal `Any` token in its
/// stub output, so we know to add `from typing import Any` at the top.
///
/// Only unannotated `Assign` statements produce `x: Any = ...`; functions
/// with missing annotations are emitted without `Any` (parameters are left
/// bare, return types are omitted).
pub(crate) fn stmt_uses_any(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Assign(assign) => {
            if let [Expr::Name(ExprName { id, .. })] = assign.targets.as_slice() {
                if id.as_str() == "__all__" {
                    return false;
                }
                // TypeVar calls and type-alias subscripts are emitted verbatim
                // and don't need the `Any` import.
                if is_type_var_call(&assign.value) || is_type_alias_value(&assign.value) {
                    return false;
                }
                true
            } else {
                false
            }
        }
        Stmt::ClassDef(class) => class.body.iter().any(stmt_uses_any),
        _ => false,
    }
}
