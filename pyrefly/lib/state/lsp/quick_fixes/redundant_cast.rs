/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;

fn expr_needs_parens(expr: &Expr) -> bool {
    !matches!(
        expr,
        Expr::Name(_)
            | Expr::NumberLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::NoneLiteral(_)
            | Expr::EllipsisLiteral(_)
            | Expr::Subscript(_)
            | Expr::Attribute(_)
            | Expr::Call(_)
            | Expr::List(_)
            | Expr::Dict(_)
            | Expr::Set(_)
            | Expr::Tuple(_)
            | Expr::FString(_)
    )
}

fn wrap_if_needed(expr: &Expr, text: &str) -> String {
    if expr_needs_parens(expr) {
        format!("({text})")
    } else {
        text.to_owned()
    }
}

fn find_enclosing_call(ast: &ModModule, selection: TextRange) -> Option<ExprCall> {
    let mut found: Option<ExprCall> = None;
    ast.visit(&mut |expr| {
        if let Expr::Call(call) = expr
            && call.range().contains_range(selection)
        {
            if let Some(existing) = &found {
                if existing.range().contains_range(call.range()) {
                    found = Some(call.clone());
                }
            } else {
                found = Some(call.clone());
            }
        }
    });
    found
}

fn redundant_cast_replacement(module_info: &ModuleInfo, call: &ExprCall) -> Option<String> {
    if call.arguments.args.iter().any(|arg| arg.is_starred_expr())
        || call.arguments.keywords.iter().any(|kw| kw.arg.is_none())
    {
        return None;
    }
    let mut typ = None;
    let mut val = None;
    let mut extra = false;
    match call.arguments.args.as_ref() {
        [] => {}
        [arg1] => {
            typ = Some(arg1);
        }
        [arg1, arg2, tail @ ..] => {
            typ = Some(arg1);
            val = Some(arg2);
            extra = !tail.is_empty();
        }
    }
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        match name.as_str() {
            "typ" => {
                if typ.is_some() {
                    return None;
                }
                typ = Some(&keyword.value);
            }
            "val" => {
                if val.is_some() {
                    return None;
                }
                val = Some(&keyword.value);
            }
            _ => return None,
        }
    }
    if extra || typ.is_none() {
        return None;
    }
    let val_expr = val?;
    if val_expr.is_starred_expr() {
        return None;
    }
    let val_text = module_info.code_at(val_expr.range());
    Some(wrap_if_needed(val_expr, val_text))
}

pub(crate) fn redundant_cast_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error_range: TextRange,
) -> Option<(String, Module, TextRange, String)> {
    let call = find_enclosing_call(ast, error_range)?;
    let replacement = redundant_cast_replacement(module_info, &call)?;
    Some((
        "Remove redundant cast".to_owned(),
        module_info.dupe(),
        call.range(),
        replacement,
    ))
}
