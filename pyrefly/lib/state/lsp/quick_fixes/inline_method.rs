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
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::first_parameter_name;
use crate::state::lsp::quick_fixes::extract_shared::is_disallowed_scope_expr;
use crate::state::lsp::quick_fixes::extract_shared::is_static_or_class_method;

pub(crate) fn inline_method_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let call = find_enclosing_call(ast.as_ref(), selection)?;
    let (callee_name, callee_range, receiver_expr) = match call.func.as_ref() {
        Expr::Name(name) => (name.id.to_string(), name.range(), None),
        Expr::Attribute(attr) => (
            attr.attr.id.to_string(),
            attr.attr.range(),
            Some(attr.value.as_ref()),
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
    });
    let mut function_def_ctx =
        def.and_then(|def| find_function_def_with_context(ast.as_ref(), def.definition_range));
    if function_def_ctx.is_none() {
        function_def_ctx =
            find_method_def_for_self_call(ast.as_ref(), call.range(), &callee_name, receiver_expr);
    }
    let FunctionDefContext {
        function_def,
        in_class,
    } = function_def_ctx?;
    if !function_def.decorator_list.is_empty() {
        return None;
    }
    if in_class && receiver_expr.is_none() {
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
    let param_map = build_param_map(
        &function_def,
        &call,
        receiver_name.as_str(),
        receiver_expr,
        module_info.contents(),
    )?;
    let return_expr = match function_def.body.as_slice() {
        [Stmt::Return(ret)] => ret.value.as_deref(),
        _ => return None,
    };
    let (expr_range, expr_text, replacements, needs_outer_parens) = if let Some(expr) = return_expr
    {
        let expr_text = module_info.code_at(expr.range()).to_owned();
        let replacements = collect_param_replacements(expr, &param_map)?;
        (
            expr.range(),
            expr_text,
            replacements,
            expr_needs_parens(expr),
        )
    } else {
        let none_text = "None".to_owned();
        let none_range = TextRange::empty(call.range().start());
        (none_range, none_text, Vec::new(), false)
    };
    let replaced = apply_replacements_in_text(&expr_text, expr_range.start(), &replacements)?;
    let inline_text = if needs_outer_parens {
        format!("({replaced})")
    } else {
        replaced
    };
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

struct EnclosingMethod<'a> {
    class_def: &'a StmtClassDef,
    function_def: &'a StmtFunctionDef,
}

fn find_enclosing_method(ast: &ModModule, selection: TextRange) -> Option<EnclosingMethod<'_>> {
    fn search_in_class<'a>(
        class_def: &'a StmtClassDef,
        selection: TextRange,
    ) -> Option<EnclosingMethod<'a>> {
        for stmt in &class_def.body {
            match stmt {
                Stmt::FunctionDef(function_def)
                    if function_def.range().contains_range(selection) =>
                {
                    return Some(EnclosingMethod {
                        class_def,
                        function_def,
                    });
                }
                Stmt::ClassDef(inner_class) => {
                    if let Some(found) = search_in_class(inner_class, selection) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }

    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && let Some(found) = search_in_class(class_def, selection)
        {
            return Some(found);
        }
    }
    None
}

fn find_method_def_for_self_call(
    ast: &ModModule,
    selection: TextRange,
    callee_name: &str,
    receiver_expr: Option<&Expr>,
) -> Option<FunctionDefContext> {
    let receiver_name = match receiver_expr {
        Some(Expr::Name(name)) => name.id.as_str(),
        _ => return None,
    };
    let EnclosingMethod {
        class_def,
        function_def: enclosing_method,
    } = find_enclosing_method(ast, selection)?;
    if is_static_or_class_method(enclosing_method) {
        return None;
    }
    let enclosing_receiver = first_parameter_name(&enclosing_method.parameters)?;
    if enclosing_receiver != receiver_name {
        return None;
    }
    for stmt in &class_def.body {
        if let Stmt::FunctionDef(function_def) = stmt
            && function_def.name.id.as_str() == callee_name
        {
            return Some(FunctionDefContext {
                function_def: function_def.clone(),
                in_class: true,
            });
        }
    }
    None
}

/// Returns true if the expression needs parentheses when inlined.
/// Simple expressions (literals, names, subscripts, attributes, calls) don't need
/// parentheses because they have high precedence. Complex expressions (binary ops,
/// unary ops, comparisons, etc.) need parentheses to preserve semantics.
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

/// Wraps text in parentheses only if the expression needs them.
fn wrap_if_needed(expr: &Expr, text: &str) -> String {
    if expr_needs_parens(expr) {
        format!("({text})")
    } else {
        text.to_owned()
    }
}

fn build_param_map(
    function_def: &StmtFunctionDef,
    call: &ExprCall,
    receiver_name: &str,
    receiver_expr: Option<&Expr>,
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
        let receiver_expr = receiver_expr?;
        let receiver_text = &source[receiver_expr.range().start().to_usize()
            ..receiver_expr.range().end().to_usize().min(source.len())];
        map.insert(
            receiver_name.to_owned(),
            wrap_if_needed(receiver_expr, receiver_text),
        );
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
    for (positional_index, param) in params.into_iter().enumerate() {
        let param_name = param.name().id.as_str();
        let arg_value = call
            .arguments
            .find_argument_value(param_name, positional_index)
            .or(param.default.as_deref());
        let arg_expr = arg_value?;
        let start = arg_expr.range().start().to_usize();
        let end = arg_expr.range().end().to_usize().min(source.len());
        let arg_text = if start <= end && end <= source.len() {
            &source[start..end]
        } else {
            return None;
        };
        map.insert(param_name.to_owned(), wrap_if_needed(arg_expr, arg_text));
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
        if is_disallowed_scope_expr(expr) {
            *invalid = true;
            return;
        }
        if let Expr::Name(name) = expr {
            if matches!(name.ctx, ExprContext::Store) && param_map.contains_key(name.id.as_str()) {
                *invalid = true;
                return;
            }
            if matches!(name.ctx, ExprContext::Load)
                && let Some(replacement) = param_map.get(name.id.as_str())
            {
                replacements.push((name.range(), replacement.clone()));
            }
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
