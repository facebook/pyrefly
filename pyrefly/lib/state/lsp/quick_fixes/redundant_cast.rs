/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::state::lsp::quick_fixes::extract_shared::wrap_if_needed;

fn find_enclosing_call(ast: &ModModule, selection: TextRange) -> Option<ExprCall> {
    for node in Ast::locate_node(ast, selection.start()) {
        if let AnyNodeRef::ExprCall(call) = node
            && call.range().contains_range(selection)
        {
            return Some(call.clone());
        }
    }
    None
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
