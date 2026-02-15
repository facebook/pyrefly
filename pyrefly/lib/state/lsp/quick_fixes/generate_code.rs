/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::find_enclosing_statement_range;
use super::extract_shared::line_indent_and_start;
use crate::state::lsp::Transaction;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

const BODY_INDENT: &str = "    ";

struct InferredParam {
    prefix: &'static str,
    name: String,
    annotation: Option<String>,
}

fn param_name_from_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.id.to_string()),
        Expr::Attribute(attr) => Some(attr.attr.id.to_string()),
        _ => None,
    }
}

fn unique_name(base: String, used: &mut HashSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    let mut i = 1;
    loop {
        let candidate = format!("{base}_{i}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        i += 1;
    }
}

fn type_to_annotation(ty: Type, stdlib: &Stdlib) -> Option<String> {
    let ty = ty.promote_implicit_literals(stdlib);
    if ty.is_any() {
        return None;
    }
    let parts = ty.get_types_with_locations(Some(stdlib));
    let text: String = parts.into_iter().map(|(part, _)| part).collect();
    Some(text)
}

fn infer_annotation(
    transaction: &Transaction<'_>,
    handle: &Handle,
    range: TextRange,
) -> Option<Type> {
    let ty = transaction.get_type_trace(handle, range)?;
    (!ty.is_any()).then_some(ty)
}

fn infer_params_from_call(
    transaction: &Transaction<'_>,
    handle: &Handle,
    stdlib: &Stdlib,
    ast: &ModModule,
    selection: TextRange,
    name: &str,
) -> Vec<InferredParam> {
    let covering_nodes = Ast::locate_node(ast, selection.start());
    let mut call = None;
    for node in covering_nodes {
        if let AnyNodeRef::ExprCall(found_call) = node
            && found_call.func.range().contains_range(selection)
            && matches!(
                found_call.func.as_ref(),
                Expr::Name(expr_name) if expr_name.id.as_str() == name
            )
        {
            call = Some(found_call.clone());
            break;
        }
    }
    let Some(call) = call else {
        return Vec::new();
    };
    let mut params = Vec::new();
    let mut used_names = HashSet::new();
    let mut counter = 1;

    for arg in &call.arguments.args {
        let (prefix, expr) = match arg {
            Expr::Starred(starred) => ("*", starred.value.as_ref()),
            _ => ("", arg),
        };
        let base_name = param_name_from_expr(expr).unwrap_or_else(|| {
            let name = format!("arg{counter}");
            counter += 1;
            name
        });
        let name = unique_name(base_name, &mut used_names);
        let annotation = infer_annotation(transaction, handle, expr.range())
            .and_then(|ty| type_to_annotation(ty, stdlib));
        params.push(InferredParam {
            prefix,
            name,
            annotation,
        });
    }

    let mut kwargs_params = Vec::new();
    for keyword in &call.arguments.keywords {
        let (prefix, base_name, value) = match &keyword.arg {
            Some(arg_name) => ("", arg_name.id.to_string(), &keyword.value),
            None => (
                "**",
                param_name_from_expr(&keyword.value).unwrap_or_else(|| "kwargs".to_owned()),
                &keyword.value,
            ),
        };
        let name = unique_name(base_name, &mut used_names);
        let annotation = infer_annotation(transaction, handle, value.range())
            .and_then(|ty| type_to_annotation(ty, stdlib));
        let param = InferredParam {
            prefix,
            name,
            annotation,
        };
        if prefix == "**" {
            kwargs_params.push(param);
        } else {
            params.push(param);
        }
    }

    params.extend(kwargs_params);
    params
}

fn format_params(params: &[InferredParam]) -> String {
    params
        .iter()
        .map(|param| {
            let base = match &param.annotation {
                Some(annotation) => format!("{}: {}", param.name, annotation),
                None => param.name.clone(),
            };
            format!("{}{}", param.prefix, base)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn generate_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    ast: &ModModule,
    selection: TextRange,
    name: &str,
) -> Option<Vec<(String, Module, TextRange, String)>> {
    if name.is_empty() {
        return None;
    }
    let stdlib = transaction.get_stdlib(handle);
    let (statement_indent, insert_position) = match find_enclosing_statement_range(ast, selection)
        .and_then(|range| line_indent_and_start(module_info.contents(), range.start()))
    {
        Some(result) => result,
        None => line_indent_and_start(module_info.contents(), selection.start())?,
    };
    let insert_range = TextRange::at(insert_position, TextSize::new(0));
    let inferred_params =
        infer_params_from_call(transaction, handle, &stdlib, ast, selection, name);
    let variable_text = match infer_annotation(transaction, handle, selection).map(|ty| {
        let ty = ty.promote_implicit_literals(&stdlib);
        if ty.is_any() {
            None
        } else {
            let ty = Type::optional(ty);
            let parts = ty.get_types_with_locations(Some(&stdlib));
            Some(parts.into_iter().map(|(part, _)| part).collect::<String>())
        }
    }) {
        Some(Some(annotation)) => format!("{statement_indent}{name}: {annotation} = None\n"),
        _ => format!("{statement_indent}{name} = None\n"),
    };
    let params_text = format_params(&inferred_params);
    let function_text = if params_text.is_empty() {
        format!("{statement_indent}def {name}():\n{statement_indent}{BODY_INDENT}pass\n")
    } else {
        format!(
            "{statement_indent}def {name}({params_text}):\n{statement_indent}{BODY_INDENT}pass\n"
        )
    };
    let class_text = if inferred_params.is_empty() {
        format!("{statement_indent}class {name}:\n{statement_indent}{BODY_INDENT}pass\n")
    } else {
        let init_params = format!("self, {params_text}");
        format!(
            "{statement_indent}class {name}:\n{statement_indent}{BODY_INDENT}def __init__({init_params}):\n{statement_indent}{BODY_INDENT}{BODY_INDENT}pass\n"
        )
    };

    Some(vec![
        (
            format!("Generate variable `{name}`"),
            module_info.dupe(),
            insert_range,
            variable_text,
        ),
        (
            format!("Generate function `{name}`"),
            module_info.dupe(),
            insert_range,
            function_text,
        ),
        (
            format!("Generate class `{name}`"),
            module_info.dupe(),
            insert_range,
            class_text,
        ),
    ])
}
