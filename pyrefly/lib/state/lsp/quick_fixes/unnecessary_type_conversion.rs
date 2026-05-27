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
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::state::lsp::quick_fixes::extract_shared::wrap_if_needed;

pub(crate) fn unnecessary_type_conversion_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error_range: TextRange,
) -> Option<(String, Module, TextRange, String)> {
    let mut call = None;
    for node in Ast::locate_node(ast, error_range.start()) {
        if let AnyNodeRef::ExprCall(candidate) = node
            && candidate.range().contains_range(error_range)
        {
            call = Some(candidate);
            break;
        }
    }
    let call = call?;
    let conversion_name = match call.func.as_ref() {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attribute) => Some(attribute.attr.as_str()),
        _ => None,
    }?;
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [arg] = call.arguments.args.as_ref() else {
        return None;
    };
    if arg.is_starred_expr() {
        return None;
    }
    let arg_text = module_info.code_at(arg.range());
    Some((
        format!("Remove unnecessary `{conversion_name}()` call"),
        module_info.dupe(),
        call.range(),
        wrap_if_needed(arg, arg_text),
    ))
}
