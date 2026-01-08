/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_function::LocalRefactorCodeAction;
use crate::state::lsp::quick_fixes::extract_shared::first_parameter_name;
use crate::state::lsp::quick_fixes::extract_shared::function_has_decorator;

pub(crate) fn inline_method_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let call = find_enclosing_call(ast.as_ref(), selection)?;
    let (callee_name, callee_range, receiver_range) = match call.func.as_ref() {
        Expr::Name(name) => (name.id.to_string(), name.range(), None),
        Expr::Attribute(attr) => (
            attr.attr.id.to_string(),
            attr.attr.range(),
            Some(attr.value.range()),
        ),
        _ => return None,
    };
    let defs = transaction.find_definition(handle, callee_range.start(), FindPreference::default());
    let def = defs.into_iter().find(|def| {
        def.module.path() == module_info.path()
            && matches!(
                def.metadata.symbol_kind(),
                Some(SymbolKind::Function | SymbolKind::Method)
            )
    })?;
    let FunctionDefContext {
        function_def,
        in_class,
    } = find_function_def_with_context(ast.as_ref(), def.definition_range)?;
    if !function_def.decorator_list.is_empty() {
        return None;
    }
    if in_class && receiver_range.is_none() {
        return None;
    }
    if !function_def.parameters.posonlyargs.is_empty()
        || !function_def.parameters.kwonlyargs.is_empty()
        || function_def.parameters.vararg.is_some()
        || function_def.parameters.kwarg.is_some()
    {
        return None;
    }
    let receiver_name = if in_class && !is_static_or_class_method(&function_def) {
        first_parameter_name(&function_def.parameters)?
    } else {
        String::new()
    };
    let receiver_text = receiver_range.map(|range| module_info.code_at(range));
    let param_map = build_param_map(
        &function_def,
        &call,
        receiver_name.as_str(),
        receiver_text.as_deref(),
        module_info.contents(),
    )?;
    let return_expr = match function_def.body.as_slice() {
        [Stmt::Return(ret)] => ret.value.as_deref(),
        _ => return None,
    };
    let (expr_range, expr_text, replacements) = if let Some(expr) = return_expr {
        let expr_text = module_info.code_at(expr.range()).to_owned();
        let replacements = collect_param_replacements(expr, &param_map)?;
        (expr.range(), expr_text, replacements)
    } else {
        let none_text = "None".to_owned();
        let none_range = TextRange::at(call.range().start(), TextSize::new(0));
        (none_range, none_text, Vec::new())
    };
    let replaced = apply_replacements_in_text(&expr_text, expr_range.start(), &replacements)?;
    let inline_text = format!("({replaced})");
    let edits = vec![(module_info.dupe(), call.range(), inline_text)];
    Some(vec![LocalRefactorCodeAction {
        title: format!("Inline call to `{callee_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_INLINE,
    }])
}

struct FunctionDefContext {
    function_def: StmtFunctionDef,
    in_class: bool,
}

fn find_function_def_with_context(
    ast: &ModModule,
    definition_range: TextRange,
) -> Option<FunctionDefContext> {
    fn search_in_body(
        body: &[Stmt],
        definition_range: TextRange,
        in_class: bool,
    ) -> Option<FunctionDefContext> {
        for stmt in body {
            match stmt {
                Stmt::FunctionDef(function_def)
                    if function_def.name.range() == definition_range =>
                {
                    return Some(FunctionDefContext {
                        function_def: function_def.clone(),
                        in_class,
                    });
                }
                Stmt::ClassDef(class_def) => {
                    if let Some(found) = search_in_body(&class_def.body, definition_range, true) {
                        return Some(found);
                    }
                }
                Stmt::FunctionDef(function_def) => {
                    if let Some(found) =
                        search_in_body(&function_def.body, definition_range, in_class)
                    {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }

    search_in_body(&ast.body, definition_range, false)
}

fn is_static_or_class_method(function_def: &StmtFunctionDef) -> bool {
    function_has_decorator(function_def, "staticmethod")
        || function_has_decorator(function_def, "classmethod")
}

fn build_param_map(
    function_def: &StmtFunctionDef,
    call: &ExprCall,
    receiver_name: &str,
    receiver_text: Option<&str>,
    source: &str,
) -> Option<HashMap<String, String>> {
    if call.arguments.args.iter().any(|arg| arg.is_starred_expr())
        || call.arguments.keywords.iter().any(|kw| kw.arg.is_none())
    {
        return None;
    }
    let mut map = HashMap::new();
    let mut params: Vec<_> = function_def.parameters.args.iter().collect();
    if !receiver_name.is_empty() {
        let receiver = params.first()?;
        if receiver.name().id.as_str() != receiver_name {
            return None;
        }
        let receiver_text = receiver_text?;
        map.insert(receiver_name.to_owned(), format!("({receiver_text})"));
        params.remove(0);
    }
    let positional_count = call
        .arguments
        .args
        .iter()
        .take_while(|arg| !arg.is_starred_expr())
        .count();
    if positional_count > params.len() {
        return None;
    }
    for keyword in call.arguments.keywords.iter() {
        if let Some(arg) = &keyword.arg
            && !params
                .iter()
                .any(|param| param.name().id.as_str() == arg.id())
        {
            return None;
        }
    }
    let mut positional_index = 0;
    for param in params {
        let param_name = param.name().id.as_str();
        let arg_value = call
            .arguments
            .find_argument_value(param_name, positional_index)
            .or_else(|| param.default.as_deref());
        let arg_expr = arg_value?;
        let start = arg_expr.range().start().to_usize();
        let end = arg_expr.range().end().to_usize().min(source.len());
        let arg_text = if start <= end && end <= source.len() {
            &source[start..end]
        } else {
            return None;
        };
        map.insert(param_name.to_owned(), format!("({arg_text})"));
        positional_index += 1;
    }
    Some(map)
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

fn collect_param_replacements(
    expr: &Expr,
    param_map: &HashMap<String, String>,
) -> Option<Vec<(TextRange, String)>> {
    let mut replacements = Vec::new();
    let mut invalid = false;
    fn visit(
        expr: &Expr,
        param_map: &HashMap<String, String>,
        replacements: &mut Vec<(TextRange, String)>,
        invalid: &mut bool,
    ) {
        if *invalid {
            return;
        }
        match expr {
            Expr::Lambda(_)
            | Expr::ListComp(_)
            | Expr::SetComp(_)
            | Expr::DictComp(_)
            | Expr::Generator(_) => {
                *invalid = true;
                return;
            }
            Expr::Name(name) => {
                if matches!(name.ctx, ExprContext::Store)
                    && param_map.contains_key(name.id.as_str())
                {
                    *invalid = true;
                    return;
                }
                if matches!(name.ctx, ExprContext::Load) {
                    if let Some(replacement) = param_map.get(name.id.as_str()) {
                        replacements.push((name.range(), replacement.clone()));
                    }
                }
            }
            _ => {}
        }
        expr.recurse(&mut |child| visit(child, param_map, replacements, invalid));
    }
    visit(expr, param_map, &mut replacements, &mut invalid);
    if invalid { None } else { Some(replacements) }
}

fn apply_replacements_in_text(
    original: &str,
    range_start: TextSize,
    replacements: &[(TextRange, String)],
) -> Option<String> {
    if replacements.is_empty() {
        return Some(original.to_owned());
    }
    let mut result = original.to_owned();
    let mut sorted: Vec<_> = replacements.to_vec();
    sorted.sort_by_key(|(range, _)| range.start());
    for (range, replacement) in sorted.into_iter().rev() {
        if range.start() < range_start {
            return None;
        }
        let start = (range.start() - range_start).to_usize();
        let end = (range.end() - range_start).to_usize();
        if start > result.len() || end > result.len() || start > end {
            return None;
        }
        result.replace_range(start..end, &replacement);
    }
    Some(result)
}
