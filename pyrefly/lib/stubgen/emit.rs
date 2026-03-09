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

use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;

use crate::export::definitions::Definitions;
use crate::stubgen::visibility::VisibilityFilter;
use crate::stubgen::StubgenOptions;

/// Extract the source text for an AST node's range.
fn source_at<'a>(source: &'a str, node: &impl Ranged) -> &'a str {
    let range = node.range();
    &source[range.start().to_usize()..range.end().to_usize()]
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
                // __all__ assignments are always included -- stub consumers need
                // them regardless of the visibility filter.
                if id.as_str() == "__all__" {
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

    for stmt in &class.body {
        // Inside a class, we don't filter by module-level __all__ -- all
        // class members are part of the class's stub.
        let class_filter = VisibilityFilter::Inferred;
        if emit_stmt(source, stmt, &class_filter, options, out, &child_indent, defs) {
            has_body = true;
        }
    }

    if !has_body {
        out.push_str(&child_indent);
        out.push_str("...\n");
    }
}

/// Check whether a statement will produce `Any` in its stub output, so
/// we know to add `from typing import Any` at the top.
pub(crate) fn stmt_uses_any(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::FunctionDef(func) => {
            if func.returns.is_none() {
                return true;
            }
            for p in func
                .parameters
                .posonlyargs
                .iter()
                .chain(&func.parameters.args)
                .chain(&func.parameters.kwonlyargs)
            {
                let name = p.parameter.name.as_str();
                if name == "self" || name == "cls" {
                    continue;
                }
                if p.parameter.annotation.is_none() {
                    return true;
                }
            }
            false
        }
        Stmt::Assign(assign) => {
            if let [Expr::Name(ExprName { id, .. })] = assign.targets.as_slice() {
                id.as_str() != "__all__"
            } else {
                false
            }
        }
        Stmt::ClassDef(class) => class.body.iter().any(stmt_uses_any),
        _ => false,
    }
}
