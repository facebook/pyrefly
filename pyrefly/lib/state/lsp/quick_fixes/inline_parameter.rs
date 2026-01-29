/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::NameRefCollector;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_function;

pub(crate) fn inline_parameter_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let position = selection.start();
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let defs = transaction.find_definition(handle, position, FindPreference::default());
    let def = defs.into_iter().find(|def| {
        def.module.path() == module_info.path()
            && matches!(def.metadata.symbol_kind(), Some(SymbolKind::Parameter))
    })?;
    let function_def = find_enclosing_function(ast.as_ref(), def.definition_range)?;
    if !function_def.decorator_list.is_empty() {
        return None;
    }
    if !function_def.parameters.posonlyargs.is_empty()
        || !function_def.parameters.kwonlyargs.is_empty()
        || function_def.parameters.vararg.is_some()
        || function_def.parameters.kwarg.is_some()
    {
        return None;
    }
    let param = function_def
        .parameters
        .args
        .iter()
        .find(|param| param.name().range() == def.definition_range)?;
    let param_name = param.name().id.as_str();
    let calls = collect_calls_to_definition(
        transaction,
        handle,
        ast.as_ref(),
        def.definition_range,
        function_def.range(),
        function_def.name.id.as_str(),
    )?;
    if calls.len() != 1 {
        return None;
    }
    let call = calls.into_iter().next()?;
    if call.arguments.args.iter().any(|arg| arg.is_starred_expr())
        || call.arguments.keywords.iter().any(|kw| kw.arg.is_none())
    {
        return None;
    }
    let param_index = function_def
        .parameters
        .args
        .iter()
        .position(|arg| arg.name().id.as_str() == param_name)?;
    let arg_expr = call
        .arguments
        .find_argument_value(param_name, param_index)?;
    let arg_text = module_info.code_at(arg_expr.range());
    let replacement = format!("({arg_text})");
    let mut edits = Vec::new();
    let mut collector = NameRefCollector::new(param_name.to_owned());
    collector.visit_stmts(&function_def.body);
    if collector.invalid || collector.load_refs.is_empty() {
        return None;
    }
    for range in collector.load_refs {
        edits.push((module_info.dupe(), range, replacement.clone()));
    }
    let param_remove_range = expand_range_to_remove_item(module_info.contents(), param.range());
    edits.push((module_info.dupe(), param_remove_range, String::new()));
    let arg_remove_range =
        argument_remove_range(module_info.contents(), &call, param_name, param_index)?;
    edits.push((module_info.dupe(), arg_remove_range, String::new()));
    Some(vec![LocalRefactorCodeAction {
        title: format!("Inline parameter `{param_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_INLINE,
    }])
}

fn collect_calls_to_definition(
    transaction: &Transaction<'_>,
    handle: &Handle,
    ast: &ModModule,
    definition_range: TextRange,
    function_range: TextRange,
    function_name: &str,
) -> Option<Vec<ExprCall>> {
    let module_info = transaction.get_module_info(handle)?;
    let mut calls = Vec::new();
    let mut name_calls = Vec::new();
    ast.visit(&mut |expr| {
        if let Expr::Call(call) = expr
            && let Expr::Name(name) = call.func.as_ref()
        {
            let defs =
                transaction.find_definition(handle, name.range.start(), FindPreference::default());
            if defs.iter().any(|def| {
                def.module.path() == module_info.path() && def.definition_range == definition_range
            }) {
                calls.push(call.clone());
            } else if name.id.as_str() == function_name {
                name_calls.push(call.clone());
            }
        }
    });
    let calls = if calls.is_empty() { name_calls } else { calls };
    if calls
        .iter()
        .any(|call| function_range.contains_range(call.range()))
    {
        return None;
    }
    Some(calls)
}

fn expand_range_to_remove_item(source: &str, item_range: TextRange) -> TextRange {
    let bytes = source.as_bytes();
    let mut start = item_range.start().to_usize();
    let mut end = item_range.end().to_usize().min(bytes.len());
    let mut idx = end;
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx < bytes.len() && bytes[idx] == b',' {
        idx += 1;
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        end = idx;
    } else {
        idx = start;
        while idx > 0 && bytes[idx - 1].is_ascii_whitespace() {
            idx -= 1;
        }
        if idx > 0 && bytes[idx - 1] == b',' {
            idx -= 1;
            while idx > 0 && bytes[idx - 1].is_ascii_whitespace() {
                idx -= 1;
            }
            start = idx;
        }
    }
    let start = TextSize::try_from(start).unwrap_or(item_range.start());
    let end = TextSize::try_from(end).unwrap_or(item_range.end());
    TextRange::new(start, end)
}

fn argument_remove_range(
    source: &str,
    call: &ExprCall,
    param_name: &str,
    param_index: usize,
) -> Option<TextRange> {
    if let Some(keyword) = call.arguments.find_keyword(param_name) {
        return Some(expand_range_to_remove_item(source, keyword.range()));
    }
    let arg = call.arguments.find_positional(param_index)?;
    Some(expand_range_to_remove_item(source, arg.range()))
}
