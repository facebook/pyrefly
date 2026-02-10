/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprRef;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::MethodInfo;
use super::extract_shared::code_at_range;
use super::extract_shared::first_parameter_name;
use super::extract_shared::function_has_decorator;
use super::extract_shared::is_exact_expression;
use super::extract_shared::is_local_scope_stmt;
use super::extract_shared::split_selection;
use super::extract_shared::unique_name;
use super::extract_shared::validate_non_empty_selection;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const DEFAULT_PARAMETER_PREFIX: &str = "param";

#[derive(Clone, Debug)]
struct MethodContext {
    info: MethodInfo,
    is_staticmethod: bool,
    is_classmethod: bool,
}

#[derive(Clone, Debug)]
struct FunctionContext<'a> {
    function_def: &'a StmtFunctionDef,
    method: Option<MethodContext>,
}

#[derive(Clone, Debug)]
struct ParamInfo {
    positional: Vec<String>,
    kwonly: HashSet<String>,
    vararg: Option<String>,
    kwarg: Option<String>,
}

#[derive(Clone, Debug)]
struct ExpressionTemplate {
    text: String,
    range: TextRange,
    name_refs: Vec<(TextRange, String)>,
    param_names_used: HashSet<String>,
}

#[derive(Clone, Debug)]
struct ArgReplacement {
    text: String,
    needs_parens: bool,
}

#[derive(Clone, Copy, Debug)]
enum ParameterInsertStyle {
    Positional,
    KeywordOnly,
}

/// Builds introduce-parameter refactor actions for a selected expression.
pub(crate) fn introduce_parameter_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let selection_text = validate_non_empty_selection(selection, module_info.code_at(selection))?;
    let (leading_ws, expression_text, trailing_ws, expression_range) =
        split_selection(selection_text, selection)?;
    if !is_exact_expression(ast.as_ref(), expression_range) {
        return None;
    }
    let function_ctx = find_function_context(ast.as_ref(), expression_range)?;
    if function_ctx
        .function_def
        .parameters
        .range()
        .contains_range(expression_range)
    {
        return None;
    }
    let param_info = ParamInfo::from_parameters(&function_ctx.function_def.parameters);
    let param_names = param_info.all_names();
    let name_refs = collect_expression_name_refs(ast.as_ref(), expression_range);
    if expression_uses_local_names(
        transaction,
        handle,
        module_info.path(),
        function_ctx.function_def,
        &name_refs,
        &param_names,
    ) {
        return None;
    }
    if expression_uses_variadic_params(&name_refs, &param_info) {
        return None;
    }

    let template =
        ExpressionTemplate::new(expression_text, expression_range, &name_refs, &param_names);
    let base_name = suggest_parameter_name(ast.as_ref(), expression_range, &param_names);
    let param_name = unique_name(&base_name, |name| param_names.contains(name));
    let insert_style = parameter_insert_style(&function_ctx.function_def.parameters);
    let (signature_range, signature_text) = build_parameter_insertion(
        module_info.contents(),
        &function_ctx.function_def.parameters,
        &param_name,
        insert_style,
    )?;
    let signature_edit = (module_info.dupe(), signature_range, signature_text);
    let call_edits = build_callsite_edits(
        transaction,
        handle,
        &module_info,
        &param_info,
        &template,
        &param_name,
        insert_style,
        &function_ctx,
    )?;

    let selection_replacement = format!("{leading_ws}{param_name}{trailing_ws}");
    let replace_selection_edit = (module_info.dupe(), selection, selection_replacement);
    let mut actions = Vec::new();
    actions.push(LocalRefactorCodeAction {
        title: format!("Introduce parameter `{param_name}`"),
        edits: [
            vec![signature_edit.clone()],
            call_edits.clone(),
            vec![replace_selection_edit],
        ]
        .concat(),
        kind: CodeActionKind::REFACTOR_EXTRACT,
    });

    let occurrence_ranges = collect_matching_expression_ranges(
        module_info.contents(),
        &function_ctx.function_def.body,
        &template.text,
    );
    if occurrence_ranges.len() > 1 {
        let replace_all_edits = occurrence_ranges
            .into_iter()
            .map(|range| (module_info.dupe(), range, param_name.clone()))
            .collect::<Vec<_>>();
        actions.push(LocalRefactorCodeAction {
            title: format!("Introduce parameter `{param_name}` (replace all occurrences)"),
            edits: [vec![signature_edit], call_edits, replace_all_edits].concat(),
            kind: CodeActionKind::REFACTOR_EXTRACT,
        });
    }

    Some(actions)
}

fn find_function_context<'a>(
    ast: &'a ModModule,
    selection: TextRange,
) -> Option<FunctionContext<'a>> {
    let covering_nodes = Ast::locate_node(ast, selection.start());
    for (idx, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(function_def) = node {
            if !function_def.range().contains_range(selection) {
                continue;
            }
            let method = match covering_nodes.get(idx + 1) {
                Some(AnyNodeRef::StmtClassDef(class_def)) => {
                    Some(method_context_from_class(class_def, function_def)?)
                }
                _ => None,
            };
            return Some(FunctionContext {
                function_def,
                method,
            });
        }
    }
    None
}

fn method_context_from_class(
    class_def: &StmtClassDef,
    function_def: &StmtFunctionDef,
) -> Option<MethodContext> {
    let receiver_name = first_parameter_name(&function_def.parameters)?;
    Some(MethodContext {
        info: MethodInfo {
            class_name: class_def.name.id.to_string(),
            receiver_name,
        },
        is_staticmethod: function_has_decorator(function_def, "staticmethod"),
        is_classmethod: function_has_decorator(function_def, "classmethod"),
    })
}

impl ParamInfo {
    fn from_parameters(parameters: &Parameters) -> Self {
        let mut positional = Vec::new();
        for param in &parameters.posonlyargs {
            positional.push(param.name().id.to_string());
        }
        for param in &parameters.args {
            positional.push(param.name().id.to_string());
        }
        let mut kwonly = HashSet::new();
        for param in &parameters.kwonlyargs {
            kwonly.insert(param.name().id.to_string());
        }
        let vararg = parameters
            .vararg
            .as_ref()
            .map(|param| param.name.id.to_string());
        let kwarg = parameters
            .kwarg
            .as_ref()
            .map(|param| param.name.id.to_string());
        Self {
            positional,
            kwonly,
            vararg,
            kwarg,
        }
    }

    fn all_names(&self) -> HashSet<String> {
        let mut names: HashSet<String> = self.positional.iter().cloned().collect();
        names.extend(self.kwonly.iter().cloned());
        if let Some(vararg) = &self.vararg
            && !vararg.is_empty()
        {
            names.insert(vararg.clone());
        }
        if let Some(kwarg) = &self.kwarg
            && !kwarg.is_empty()
        {
            names.insert(kwarg.clone());
        }
        names
    }
}

impl ExpressionTemplate {
    fn new(
        expression_text: &str,
        range: TextRange,
        name_refs: &[(TextRange, String)],
        param_names: &HashSet<String>,
    ) -> Self {
        let param_names_used: HashSet<String> = name_refs
            .iter()
            .filter(|(_, name)| param_names.contains(name))
            .map(|(_, name)| name.clone())
            .collect();
        Self {
            text: expression_text.to_owned(),
            range,
            name_refs: name_refs.to_vec(),
            param_names_used,
        }
    }
}

fn collect_expression_name_refs(ast: &ModModule, selection: TextRange) -> Vec<(TextRange, String)> {
    struct Collector {
        selection: TextRange,
        refs: Vec<(TextRange, String)>,
    }

    impl<'a> Visitor<'a> for Collector {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if self.selection.contains_range(expr.range())
                && let Expr::Name(name) = expr
                && matches!(name.ctx, ExprContext::Load)
            {
                self.refs.push((name.range, name.id.to_string()));
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut collector = Collector {
        selection,
        refs: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.refs
}

fn expression_uses_local_names(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_path: &ModulePath,
    function_def: &StmtFunctionDef,
    name_refs: &[(TextRange, String)],
    param_names: &HashSet<String>,
) -> bool {
    for (range, name) in name_refs {
        if param_names.contains(name) {
            continue;
        }
        let defs = transaction.find_definition(handle, range.start(), FindPreference::default());
        if defs.iter().any(|def| {
            def.module.path() == module_path
                && function_def.range().contains_range(def.definition_range)
        }) {
            return true;
        }
    }
    false
}

fn expression_uses_variadic_params(
    name_refs: &[(TextRange, String)],
    param_info: &ParamInfo,
) -> bool {
    let vararg = param_info.vararg.as_deref();
    let kwarg = param_info.kwarg.as_deref();
    name_refs.iter().any(|(_, name)| {
        vararg.is_some_and(|vararg| vararg == name) || kwarg.is_some_and(|kwarg| kwarg == name)
    })
}

fn suggest_parameter_name(
    ast: &ModModule,
    selection: TextRange,
    existing: &HashSet<String>,
) -> String {
    let base = Ast::locate_node(ast, selection.start())
        .into_iter()
        .filter_map(|node| node.as_expr_ref())
        .find(|expr| expr.range() == selection)
        .and_then(|expr| match expr {
            ExprRef::Name(name) => Some(name.id.to_string()),
            ExprRef::Attribute(attribute) => Some(attribute.attr.id.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| DEFAULT_PARAMETER_PREFIX.to_owned());
    if base.is_empty() || existing.contains(&base) {
        DEFAULT_PARAMETER_PREFIX.to_owned()
    } else {
        base
    }
}

fn parameter_insert_style(parameters: &Parameters) -> ParameterInsertStyle {
    if parameters.vararg.is_some()
        || !parameters.kwonlyargs.is_empty()
        || parameters.kwarg.is_some()
    {
        ParameterInsertStyle::KeywordOnly
    } else {
        ParameterInsertStyle::Positional
    }
}

fn build_parameter_insertion(
    source: &str,
    parameters: &Parameters,
    param_name: &str,
    style: ParameterInsertStyle,
) -> Option<(TextRange, String)> {
    let params_range = parameters.range();
    let insert_pos = match style {
        ParameterInsertStyle::Positional => params_range.end().checked_sub(TextSize::from(1))?,
        ParameterInsertStyle::KeywordOnly => {
            if let Some(first_kwonly) = parameters.kwonlyargs.first() {
                first_kwonly.range().start()
            } else if let Some(kwarg) = parameters.kwarg.as_ref() {
                kwarg.range().start()
            } else if let Some(vararg) = parameters.vararg.as_ref() {
                vararg.range().end()
            } else {
                params_range.end().checked_sub(TextSize::from(1))?
            }
        }
    };
    let has_any = !parameters.posonlyargs.is_empty()
        || !parameters.args.is_empty()
        || parameters.vararg.is_some()
        || !parameters.kwonlyargs.is_empty()
        || parameters.kwarg.is_some();
    let insertion_text = match style {
        ParameterInsertStyle::Positional => {
            if has_any {
                format!(", {param_name}")
            } else {
                param_name.to_owned()
            }
        }
        ParameterInsertStyle::KeywordOnly => {
            if parameters.kwonlyargs.is_empty() && parameters.kwarg.is_none() {
                format!(", {param_name}")
            } else {
                format!("{param_name}, ")
            }
        }
    };
    let insert_range = TextRange::at(insert_pos, TextSize::new(0));
    if insert_pos.to_usize() > source.len() {
        return None;
    }
    Some((insert_range, insertion_text))
}

fn collect_matching_expression_ranges(
    source: &str,
    body: &[Stmt],
    expression_text: &str,
) -> Vec<TextRange> {
    struct Finder<'a> {
        source: &'a str,
        expression_text: &'a str,
        matches: Vec<TextRange>,
    }

    impl<'a> Visitor<'a> for Finder<'a> {
        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if is_local_scope_stmt(stmt) {
                ruff_python_ast::visitor::walk_stmt(self, stmt);
            }
        }

        fn visit_expr(&mut self, expr: &'a Expr) {
            let range = expr.range();
            if code_at_range(self.source, range) == Some(self.expression_text) {
                self.matches.push(range);
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut finder = Finder {
        source,
        expression_text,
        matches: Vec::new(),
    };
    for stmt in body {
        finder.visit_stmt(stmt);
    }
    finder.matches
}

fn build_callsite_edits(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &crate::module::module_info::ModuleInfo,
    param_info: &ParamInfo,
    template: &ExpressionTemplate,
    new_param_name: &str,
    insert_style: ParameterInsertStyle,
    function_ctx: &FunctionContext<'_>,
) -> Option<Vec<(pyrefly_python::module::Module, TextRange, String)>> {
    let definition = transaction
        .find_definition(
            handle,
            function_ctx.function_def.name.range.start(),
            FindPreference::default(),
        )
        .into_iter()
        .find(|def| {
            def.module.path() == module_info.path()
                && def
                    .definition_range
                    .contains_range(function_ctx.function_def.name.range)
        })?;

    let mut edits = Vec::new();
    for module_handle in transaction.handles() {
        let Some(other_module_info) = transaction.get_module_info(&module_handle) else {
            continue;
        };
        let Some(refs) = transaction.local_references_from_definition(
            &module_handle,
            definition.metadata.clone(),
            definition.definition_range,
            &definition.module,
        ) else {
            continue;
        };
        if refs.is_empty() {
            continue;
        }
        let Some(ast) = transaction.get_ast(&module_handle) else {
            continue;
        };
        let ref_set: HashSet<TextRange> = refs.into_iter().collect();
        let mut module_edits = Vec::new();
        let mut failed = false;
        ast.as_ref().visit(&mut |expr| {
            if failed {
                return;
            }
            let Expr::Call(call) = expr else {
                return;
            };
            if !ref_set
                .iter()
                .any(|range| call.func.range().contains(range.start()))
            {
                return;
            }
            let Some(argument_text) = build_argument_expression_for_call(
                call,
                other_module_info.contents(),
                param_info,
                template,
                function_ctx.method.as_ref(),
            ) else {
                failed = true;
                return;
            };
            let Some((range, text)) = build_call_argument_insertion(
                call,
                other_module_info.contents(),
                new_param_name,
                &argument_text,
                insert_style,
            ) else {
                failed = true;
                return;
            };
            module_edits.push((other_module_info.dupe(), range, text));
        });
        if failed {
            return None;
        }
        edits.extend(module_edits);
    }
    Some(edits)
}

fn build_argument_expression_for_call(
    call: &ExprCall,
    source: &str,
    param_info: &ParamInfo,
    template: &ExpressionTemplate,
    method_ctx: Option<&MethodContext>,
) -> Option<String> {
    let replacements = build_param_replacements(call, source, param_info, method_ctx)?;
    if template.param_names_used.is_empty() {
        return Some(template.text.clone());
    }
    build_argument_expression(template, &replacements)
}

fn build_param_replacements(
    call: &ExprCall,
    source: &str,
    param_info: &ParamInfo,
    method_ctx: Option<&MethodContext>,
) -> Option<HashMap<String, ArgReplacement>> {
    let mut keyword_args: HashMap<String, &Expr> = HashMap::new();
    for kw in &call.arguments.keywords {
        let Some(arg) = &kw.arg else {
            return None;
        };
        keyword_args.insert(arg.id.to_string(), &kw.value);
    }
    if call
        .arguments
        .args
        .iter()
        .any(|arg| matches!(arg, Expr::Starred(_)))
    {
        return None;
    }

    let (implicit_receiver, implicit_offset) = bound_method_receiver(call, method_ctx)?;
    let mut replacements: HashMap<String, ArgReplacement> = HashMap::new();
    if let Some((receiver_name, receiver_expr)) = implicit_receiver {
        replacements.insert(
            receiver_name,
            build_replacement_from_expr(receiver_expr, source),
        );
    }

    for (idx, arg) in call.arguments.args.iter().enumerate() {
        let param_idx = idx + implicit_offset;
        if let Some(param_name) = param_info.positional.get(param_idx) {
            replacements.insert(param_name.clone(), build_replacement_from_expr(arg, source));
        }
    }

    for (name, expr) in keyword_args {
        replacements.insert(name, build_replacement_from_expr(expr, source));
    }

    Some(replacements)
}

fn bound_method_receiver<'a>(
    call: &'a ExprCall,
    method_ctx: Option<&'a MethodContext>,
) -> Option<(Option<(String, &'a Expr)>, usize)> {
    let Some(ctx) = method_ctx else {
        return Some((None, 0));
    };
    if ctx.is_staticmethod {
        return Some((None, 0));
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return Some((None, 0));
    };
    if !ctx.is_classmethod
        && let Expr::Name(name) = attribute.value.as_ref()
        && name.id.as_str() == ctx.info.class_name
    {
        return Some((None, 0));
    }
    Some((
        Some((ctx.info.receiver_name.clone(), attribute.value.as_ref())),
        1,
    ))
}

fn build_replacement_from_expr(expr: &Expr, source: &str) -> ArgReplacement {
    let range = expr.range();
    let text = code_at_range(source, range)
        .map(|s| s.to_owned())
        .unwrap_or_default();
    let needs_parens = !matches!(expr, Expr::Name(_) | Expr::Attribute(_));
    ArgReplacement { text, needs_parens }
}

fn build_argument_expression(
    template: &ExpressionTemplate,
    replacements: &HashMap<String, ArgReplacement>,
) -> Option<String> {
    for name in &template.param_names_used {
        if !replacements.contains_key(name) {
            return None;
        }
    }
    let mut result = template.text.clone();
    let mut occurrences: Vec<(TextRange, String)> = template
        .name_refs
        .iter()
        .filter(|(_, name)| template.param_names_used.contains(name))
        .cloned()
        .collect();
    occurrences.sort_by_key(|(range, _)| range.start());
    for (range, name) in occurrences.into_iter().rev() {
        let replacement = replacements.get(&name)?;
        let replacement_text = if replacement.needs_parens {
            format!("({})", replacement.text)
        } else {
            replacement.text.clone()
        };
        let start = (range.start() - template.range.start()).to_usize();
        let end = (range.end() - template.range.start()).to_usize();
        result.replace_range(start..end, &replacement_text);
    }
    Some(result)
}

fn build_call_argument_insertion(
    call: &ExprCall,
    source: &str,
    param_name: &str,
    argument_text: &str,
    insert_style: ParameterInsertStyle,
) -> Option<(TextRange, String)> {
    let args = &call.arguments.args;
    let keywords = &call.arguments.keywords;
    let mut insertion_point = None;
    let insertion_text;
    match insert_style {
        ParameterInsertStyle::Positional => {
            if let Some(first_kw) = keywords.first() {
                let keyword_argument = format!("{param_name}={argument_text}");
                insertion_point = Some(first_kw.range().start());
                insertion_text = format!("{keyword_argument}, ");
            } else if let Some(last_arg) = args.last() {
                insertion_point = Some(last_arg.range().end());
                insertion_text = format!(", {argument_text}");
            } else {
                insertion_text = argument_text.to_owned();
            }
        }
        ParameterInsertStyle::KeywordOnly => {
            let keyword_argument = format!("{param_name}={argument_text}");
            if let Some(first_kw) = keywords.first() {
                insertion_point = Some(first_kw.range().start());
                insertion_text = format!("{keyword_argument}, ");
            } else if let Some(last_arg) = args.last() {
                insertion_point = Some(last_arg.range().end());
                insertion_text = format!(", {keyword_argument}");
            } else {
                insertion_text = keyword_argument;
            }
        }
    }

    let insert_pos = if let Some(pos) = insertion_point {
        pos
    } else {
        call.range().end().checked_sub(TextSize::from(1))?
    };
    if insert_pos.to_usize() > source.len() {
        return None;
    }
    Some((TextRange::at(insert_pos, TextSize::new(0)), insertion_text))
}
