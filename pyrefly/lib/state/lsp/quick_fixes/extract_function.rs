/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::dedent_block_preserving_layout;
use pyrefly_python::module::Module;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const HELPER_INDENT: &str = "    ";

/// Description of a refactor edit that stays within the local workspace.
#[derive(Clone, Debug)]
pub struct LocalRefactorCodeAction {
    pub title: String,
    pub edits: Vec<(Module, TextRange, String)>,
    pub kind: CodeActionKind,
}

/// Builds extract-function quick fix code actions for the supplied selection.
pub(crate) fn extract_function_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    if selection.is_empty() {
        return None;
    }
    let module_info = transaction.get_module_info(handle)?;
    let module_source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let selection_text = module_info.code_at(selection);
    if selection_text.trim().is_empty() {
        return None;
    }
    let module_len = TextSize::try_from(module_info.contents().len()).unwrap_or(TextSize::new(0));
    let module_stmt_range =
        find_enclosing_module_statement_range(ast.as_ref(), selection, module_len);
    if selection_contains_disallowed_statements(ast.as_ref(), selection) {
        return None;
    }
    let (load_refs, store_refs) = collect_identifier_refs(ast.as_ref(), selection);
    if load_refs.is_empty() && store_refs.is_empty() {
        return None;
    }
    let post_loads = collect_post_selection_loads(ast.as_ref(), module_stmt_range, selection.end());
    let block_indent = detect_block_indent(selection_text);
    let mut dedented_body = dedent_block_preserving_layout(selection_text)?;
    if dedented_body.ends_with('\n') {
        dedented_body.pop();
        if dedented_body.ends_with('\r') {
            dedented_body.pop();
        }
    }

    let function_helper_name = generate_helper_name(module_source, "extracted_function");
    let mut params = Vec::new();
    let mut seen_params = HashSet::new();
    for ident in load_refs {
        if seen_params.contains(&ident.name) {
            continue;
        }
        if ident.synthetic_load {
            seen_params.insert(ident.name.clone());
            params.push(ident.name.clone());
            continue;
        }
        let defs = transaction.find_definition(handle, ident.position, FindPreference::default());
        let Some(def) = defs.first() else {
            continue;
        };
        if def.module.path() != module_info.path() {
            continue;
        }
        if !module_stmt_range.contains_range(def.definition_range)
            || selection.contains_range(def.definition_range)
            || def.definition_range.start() >= selection.start()
        {
            continue;
        }
        seen_params.insert(ident.name.clone());
        params.push(ident.name.clone());
    }

    let mut returns = Vec::new();
    let mut seen_returns = HashSet::new();
    for ident in store_refs {
        if seen_returns.contains(&ident.name) || !post_loads.contains(&ident.name) {
            continue;
        }
        seen_returns.insert(ident.name.clone());
        returns.push(ident.name.clone());
    }

    let helper_text =
        build_helper_text(&function_helper_name, &params, &returns, &dedented_body, "");
    let call_expr = build_call_expr(&function_helper_name, None, &params);
    let replacement_line = build_call_replacement(&block_indent, &call_expr, &returns);
    let helper_edit = (
        module_info.dupe(),
        TextRange::at(module_stmt_range.start(), TextSize::new(0)),
        helper_text,
    );
    let call_edit = (module_info.dupe(), selection, replacement_line);
    let mut actions = vec![LocalRefactorCodeAction {
        title: format!("Extract into helper `{function_helper_name}`"),
        edits: vec![helper_edit, call_edit],
        kind: CodeActionKind::REFACTOR_EXTRACT,
    }];
    if let Some(method_ctx) = find_enclosing_method(ast.as_ref(), selection, module_source) {
        let method_helper_name = generate_helper_name(module_source, "extracted_method");
        let mut signature_params = Vec::new();
        signature_params.push(method_ctx.receiver_name.clone());
        let method_params = filter_params_excluding(&params, &method_ctx.receiver_name);
        signature_params.extend(method_params.iter().cloned());
        let method_helper_text = build_helper_text(
            &method_helper_name,
            &signature_params,
            &returns,
            &dedented_body,
            &method_ctx.method_indent,
        );
        let method_call_expr = build_call_expr(
            &method_helper_name,
            Some(&method_ctx.receiver_name),
            &method_params,
        );
        let method_replacement = build_call_replacement(&block_indent, &method_call_expr, &returns);
        let method_helper_edit = (
            module_info.dupe(),
            TextRange::at(method_ctx.insert_position, TextSize::new(0)),
            method_helper_text,
        );
        let method_call_edit = (module_info.dupe(), selection, method_replacement);
        actions.push(LocalRefactorCodeAction {
            title: format!(
                "Extract into method `{}` on `{}`",
                method_helper_name, method_ctx.class_name
            ),
            edits: vec![method_helper_edit, method_call_edit],
            kind: CodeActionKind::REFACTOR_EXTRACT,
        });
    }

    Some(actions)
}

#[derive(Clone, Debug)]
struct IdentifierRef {
    /// Identifier string.
    name: String,
    /// Byte offset where the identifier was observed.
    position: TextSize,
    /// True when this "load" came from reading the left-hand side of an augmented assignment.
    synthetic_load: bool,
}

fn collect_identifier_refs(
    ast: &ModModule,
    selection: TextRange,
) -> (Vec<IdentifierRef>, Vec<IdentifierRef>) {
    struct IdentifierCollector {
        selection: TextRange,
        loads: Vec<IdentifierRef>,
        stores: Vec<IdentifierRef>,
    }

    impl<'a> ruff_python_ast::visitor::Visitor<'a> for IdentifierCollector {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if self.selection.contains_range(expr.range())
                && let Expr::Name(name) = expr
            {
                let ident = IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: false,
                };
                match name.ctx {
                    ExprContext::Load => self.loads.push(ident),
                    ExprContext::Store => self.stores.push(ident),
                    ExprContext::Del | ExprContext::Invalid => {}
                }
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }

        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if self.selection.contains_range(stmt.range())
                && let Stmt::AugAssign(aug) = stmt
                && let Expr::Name(name) = aug.target.as_ref()
            {
                self.loads.push(IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: true,
                });
            }
            ruff_python_ast::visitor::walk_stmt(self, stmt);
        }
    }

    let mut collector = IdentifierCollector {
        selection,
        loads: Vec::new(),
        stores: Vec::new(),
    };
    collector.visit_body(&ast.body);
    (collector.loads, collector.stores)
}

#[derive(Clone, Debug)]
/// Context information for extracting a method from a class.
///
/// Contains details about where and how to insert the extracted method,
/// as well as relevant naming and formatting information.
struct MethodContext {
    /// Name of the class from which the method is being extracted.
    class_name: String,
    /// Name of the receiver variable (typically `self`).
    receiver_name: String,
    /// Byte offset in the source code where the extracted method should be inserted.
    insert_position: TextSize,
    /// Indentation string to use for the method definition line.
    method_indent: String,
}

fn selection_contains_disallowed_statements(ast: &ModModule, selection: TextRange) -> bool {
    fn visit_stmt(stmt: &Stmt, selection: TextRange, found: &mut bool) {
        if *found || stmt.range().intersect(selection).is_none() {
            return;
        }
        if selection.contains_range(stmt.range()) {
            match stmt {
                Stmt::Return(_)
                | Stmt::Break(_)
                | Stmt::Continue(_)
                | Stmt::Raise(_)
                | Stmt::FunctionDef(_)
                | Stmt::ClassDef(_) => {
                    *found = true;
                    return;
                }
                _ => {}
            }
        }
        stmt.recurse(&mut |child| visit_stmt(child, selection, found));
    }

    let mut found = false;
    for stmt in &ast.body {
        visit_stmt(stmt, selection, &mut found);
        if found {
            break;
        }
    }
    found
}

fn find_enclosing_module_statement_range(
    ast: &ModModule,
    selection: TextRange,
    module_len: TextSize,
) -> TextRange {
    for stmt in &ast.body {
        if stmt.range().contains_range(selection) {
            return stmt.range();
        }
    }
    TextRange::new(TextSize::new(0), module_len)
}

fn collect_post_selection_loads(
    ast: &ModModule,
    module_stmt_range: TextRange,
    selection_end: TextSize,
) -> HashSet<String> {
    let mut loads = HashSet::new();
    ast.visit(&mut |expr: &Expr| {
        if let Expr::Name(name) = expr
            && matches!(name.ctx, ExprContext::Load)
            && module_stmt_range.contains_range(name.range)
            && name.range.start() > selection_end
        {
            loads.insert(name.id.to_string());
        }
    });
    loads
}

fn find_enclosing_method(
    ast: &ModModule,
    selection: TextRange,
    source: &str,
) -> Option<MethodContext> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && let Some(ctx) = method_context_in_class(class_def, selection, source)
        {
            return Some(ctx);
        }
    }
    None
}

fn method_context_in_class(
    class_def: &StmtClassDef,
    selection: TextRange,
    source: &str,
) -> Option<MethodContext> {
    for stmt in &class_def.body {
        match stmt {
            Stmt::FunctionDef(function_def) if function_def.range().contains_range(selection) => {
                if let Some(ctx) = method_context_from_function(class_def, function_def, source) {
                    return Some(ctx);
                }
            }
            Stmt::ClassDef(inner_class) => {
                if let Some(ctx) = method_context_in_class(inner_class, selection, source) {
                    return Some(ctx);
                }
            }
            _ => {}
        }
    }
    None
}

fn method_context_from_function(
    class_def: &StmtClassDef,
    function_def: &StmtFunctionDef,
    source: &str,
) -> Option<MethodContext> {
    if is_static_or_class_method(function_def) {
        return None;
    }
    let receiver_name = first_parameter_name(&function_def.parameters)?;
    let (method_indent, insert_position) =
        indent_and_line_start(source, function_def.range().start())?;
    Some(MethodContext {
        class_name: class_def.name.id.to_string(),
        receiver_name,
        insert_position,
        method_indent,
    })
}

fn first_parameter_name(parameters: &Parameters) -> Option<String> {
    if let Some(param) = parameters.posonlyargs.first() {
        return Some(param.name().id.to_string());
    }
    parameters
        .args
        .first()
        .map(|param| param.name().id.to_string())
}

fn is_static_or_class_method(function_def: &StmtFunctionDef) -> bool {
    function_def.decorator_list.iter().any(|decorator| {
        decorator_matches_name(&decorator.expression, "staticmethod")
            || decorator_matches_name(&decorator.expression, "classmethod")
    })
}

fn decorator_matches_name(decorator: &Expr, name: &str) -> bool {
    match decorator {
        Expr::Name(identifier) => identifier.id.as_str() == name,
        Expr::Attribute(attribute) => attribute.attr.as_str() == name,
        Expr::Call(call) => decorator_matches_name(call.func.as_ref(), name),
        _ => false,
    }
}

fn detect_block_indent(selection_text: &str) -> String {
    for line in selection_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        return line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();
    }
    String::new()
}

fn indent_and_line_start(source: &str, position: TextSize) -> Option<(String, TextSize)> {
    let mut idx = position.to_usize();
    if idx > source.len() {
        idx = source.len();
    }
    let line_start = source[..idx]
        .rfind('\n')
        .map(|start| start + 1)
        .unwrap_or(0);
    let indent = source[line_start..idx]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let insert_position = TextSize::try_from(line_start).ok()?;
    Some((indent, insert_position))
}

fn build_helper_text(
    helper_name: &str,
    params: &[String],
    returns: &[String],
    dedented_body: &str,
    definition_indent: &str,
) -> String {
    let mut helper_text = if params.is_empty() {
        format!("{definition_indent}def {helper_name}():\n")
    } else {
        let helper_params = params.join(", ");
        format!("{definition_indent}def {helper_name}({helper_params}):\n")
    };
    let body_indent = format!("{definition_indent}{HELPER_INDENT}");
    helper_text.push_str(&prefix_lines_with(dedented_body, &body_indent));
    if !returns.is_empty() && !returns.iter().all(|name| name.is_empty()) {
        let return_expr = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        helper_text.push_str(&format!("{body_indent}return {return_expr}\n"));
    }
    helper_text.push('\n');
    helper_text
}

fn build_call_expr(helper_name: &str, receiver: Option<&str>, params: &[String]) -> String {
    let callee = if let Some(receiver) = receiver {
        format!("{receiver}.{helper_name}")
    } else {
        helper_name.to_owned()
    };
    let call_args = params.join(", ");
    format!("{callee}({call_args})")
}

fn build_call_replacement(block_indent: &str, call_expr: &str, returns: &[String]) -> String {
    if returns.is_empty() {
        format!("{block_indent}{call_expr}\n")
    } else {
        let lhs = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        format!("{block_indent}{lhs} = {call_expr}\n")
    }
}

fn filter_params_excluding(params: &[String], excluded: &str) -> Vec<String> {
    params
        .iter()
        .filter(|name| name.as_str() != excluded)
        .cloned()
        .collect()
}

fn prefix_lines_with(block: &str, indent: &str) -> String {
    let mut result = String::new();
    for line in block.lines() {
        result.push_str(indent);
        result.push_str(line);
        result.push('\n');
    }
    result
}

fn generate_helper_name(source: &str, prefix: &str) -> String {
    let mut counter = 1;
    loop {
        let candidate = if counter == 1 {
            prefix.to_owned()
        } else {
            format!("{prefix}_{counter}")
        };
        let needle = format!("def {candidate}(");
        if !source.contains(&needle) {
            return candidate;
        }
        counter += 1;
    }
}
