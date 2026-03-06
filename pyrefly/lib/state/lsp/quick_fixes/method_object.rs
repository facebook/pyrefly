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
use pyrefly_python::module_path::ModulePath;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::member_name_from_stmt;
use super::extract_shared::prepare_insertion_text;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const METHOD_OBJECT_CLASS_BASENAME: &str = "NewMethodObject";
const METHOD_OBJECT_MEMBER_INDENT: &str = "    ";
const METHOD_OBJECT_BODY_INDENT: &str = "        ";
const SELF_HOST_BASENAME: &str = "host";

/// Builds a method-object rewrite action for a selected sync function or method.
pub(crate) fn method_object_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let selection_point = selection_anchor(source, selection);
    let selected = find_selected_function(ast.as_ref(), source, selection_point)?;
    if selected.function_def.is_async {
        return None;
    }

    let first_body_stmt = selected.function_def.body.first()?;
    let body_range = function_body_range(source, selected.function_def)?;
    let (body_indent, _) = line_indent_and_start(source, first_body_stmt.range().start())?;
    let parameter_infos = collect_parameter_infos(&selected.function_def.parameters);
    let class_name = unique_name(METHOD_OBJECT_CLASS_BASENAME, |name| {
        ast.body
            .iter()
            .filter_map(member_name_from_stmt)
            .any(|existing| existing == name)
    });
    let renamed_body = rename_parameter_references(
        transaction,
        handle,
        module_info.path(),
        &selected.function_def.body,
        source,
        body_range,
        &parameter_infos,
    )?;
    let call_replacement = build_wrapper_body(&body_indent, &class_name, &parameter_infos);
    let class_text =
        build_method_object_class(&class_name, &parameter_infos, &body_indent, &renamed_body);
    let class_insert_text = prepare_insertion_text(
        source,
        selected.insertion_position,
        &format!("\n{class_text}"),
    );

    Some(vec![LocalRefactorCodeAction {
        title: format!(
            "Convert `{}` to method object `{class_name}`",
            selected.function_def.name.id
        ),
        edits: vec![
            (module_info.dupe(), body_range, call_replacement),
            (
                module_info.dupe(),
                TextRange::at(selected.insertion_position, TextSize::new(0)),
                class_insert_text,
            ),
        ],
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

#[derive(Clone, Copy, Debug)]
struct SelectedFunction<'a> {
    function_def: &'a StmtFunctionDef,
    insertion_position: TextSize,
}

fn find_selected_function<'a>(
    ast: &'a ModModule,
    source: &str,
    point: TextSize,
) -> Option<SelectedFunction<'a>> {
    for stmt in &ast.body {
        if stmt.range().contains(point)
            && let Some(function_def) = find_selected_function_in_stmt(stmt, point)
        {
            return Some(SelectedFunction {
                function_def,
                insertion_position: line_end_position(source, stmt.range().end()),
            });
        }
    }
    None
}

fn find_selected_function_in_stmt<'a>(
    stmt: &'a Stmt,
    point: TextSize,
) -> Option<&'a StmtFunctionDef> {
    match stmt {
        Stmt::FunctionDef(function_def) => {
            for child in &function_def.body {
                if child.range().contains(point)
                    && let Some(found) = find_selected_function_in_stmt(child, point)
                {
                    return Some(found);
                }
            }
            function_def
                .name
                .range()
                .contains(point)
                .then_some(function_def)
        }
        Stmt::ClassDef(class_def) => find_selected_function_in_body(&class_def.body, point),
        _ => None,
    }
}

fn find_selected_function_in_body<'a>(
    body: &'a [Stmt],
    point: TextSize,
) -> Option<&'a StmtFunctionDef> {
    for stmt in body {
        if stmt.range().contains(point)
            && let Some(found) = find_selected_function_in_stmt(stmt, point)
        {
            return Some(found);
        }
    }
    None
}

fn function_body_range(source: &str, function_def: &StmtFunctionDef) -> Option<TextRange> {
    let first_stmt = function_def.body.first()?;
    let last_stmt = function_def.body.last()?;
    let (_, start) = line_indent_and_start(source, first_stmt.range().start())?;
    let end = line_end_position(source, last_stmt.range().end());
    Some(TextRange::new(start, end))
}

#[derive(Clone, Debug)]
struct ParameterInfo {
    stored_name: String,
    init_name: String,
    definition_range: TextRange,
}

fn collect_parameter_infos(parameters: &Parameters) -> Vec<ParameterInfo> {
    let parameter_names = parameter_names(parameters);
    let host_name = unique_name(SELF_HOST_BASENAME, |candidate| {
        parameter_names
            .iter()
            .any(|name| name != "self" && name == candidate)
    });
    let mut infos = Vec::new();
    for param in &parameters.posonlyargs {
        infos.push(ParameterInfo {
            stored_name: param.name().id.to_string(),
            init_name: param.name().id.to_string(),
            definition_range: param.name().range(),
        });
    }
    for param in &parameters.args {
        let stored_name = param.name().id.to_string();
        let init_name = if stored_name == "self" {
            host_name.clone()
        } else {
            stored_name.clone()
        };
        infos.push(ParameterInfo {
            stored_name,
            init_name,
            definition_range: param.name().range(),
        });
    }
    if let Some(param) = &parameters.vararg {
        infos.push(ParameterInfo {
            stored_name: param.name.id.to_string(),
            init_name: param.name.id.to_string(),
            definition_range: param.name.range(),
        });
    }
    for param in &parameters.kwonlyargs {
        infos.push(ParameterInfo {
            stored_name: param.name().id.to_string(),
            init_name: param.name().id.to_string(),
            definition_range: param.name().range(),
        });
    }
    if let Some(param) = &parameters.kwarg {
        infos.push(ParameterInfo {
            stored_name: param.name.id.to_string(),
            init_name: param.name.id.to_string(),
            definition_range: param.name.range(),
        });
    }
    infos
}

fn parameter_names(parameters: &Parameters) -> Vec<String> {
    let mut names = Vec::new();
    for param in &parameters.posonlyargs {
        names.push(param.name().id.to_string());
    }
    for param in &parameters.args {
        names.push(param.name().id.to_string());
    }
    if let Some(param) = &parameters.vararg {
        names.push(param.name.id.to_string());
    }
    for param in &parameters.kwonlyargs {
        names.push(param.name().id.to_string());
    }
    if let Some(param) = &parameters.kwarg {
        names.push(param.name.id.to_string());
    }
    names
}

fn rename_parameter_references(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_path: &ModulePath,
    body: &[Stmt],
    source: &str,
    body_range: TextRange,
    parameter_infos: &[ParameterInfo],
) -> Option<String> {
    let original = source.get(body_range.start().to_usize()..body_range.end().to_usize())?;
    if parameter_infos.is_empty() {
        return Some(original.to_owned());
    }
    let definition_ranges = parameter_infos
        .iter()
        .map(|param| (param.stored_name.clone(), param.definition_range))
        .collect::<HashMap<_, _>>();
    let mut collector = ParameterReferenceCollector {
        transaction,
        handle,
        module_path,
        definition_ranges: &definition_ranges,
        replacements: Vec::new(),
    };
    for stmt in body {
        collector.visit_stmt(stmt);
    }
    apply_replacements_in_text(original, body_range.start(), &collector.replacements)
}

struct ParameterReferenceCollector<'a, 'txn> {
    transaction: &'a Transaction<'txn>,
    handle: &'a Handle,
    module_path: &'a ModulePath,
    definition_ranges: &'a HashMap<String, TextRange>,
    replacements: Vec<(TextRange, String)>,
}

impl Visitor<'_> for ParameterReferenceCollector<'_, '_> {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if let Expr::Name(name) = expr
            && let Some(expected_definition) = self.definition_ranges.get(name.id.as_str())
        {
            let definitions = self.transaction.find_definition(
                self.handle,
                name.range().start(),
                FindPreference::default(),
            );
            if definitions.iter().any(|definition| {
                definition.module.path() == self.module_path
                    && definition.definition_range == *expected_definition
            }) {
                self.replacements
                    .push((name.range(), format!("self.{}", name.id)));
            }
        }
        walk_expr(self, expr);
    }
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
    let mut sorted = replacements.to_vec();
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

fn build_wrapper_body(indent: &str, class_name: &str, parameter_infos: &[ParameterInfo]) -> String {
    let call_args = parameter_infos
        .iter()
        .map(|param| param.stored_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if call_args.is_empty() {
        format!("{indent}return {class_name}()()\n")
    } else {
        format!("{indent}return {class_name}({call_args})()\n")
    }
}

fn build_method_object_class(
    class_name: &str,
    parameter_infos: &[ParameterInfo],
    body_indent: &str,
    renamed_body: &str,
) -> String {
    let mut text = format!("class {class_name}(object):\n\n");
    if !parameter_infos.is_empty() {
        text.push_str("    def __init__(self");
        for parameter in parameter_infos {
            text.push_str(", ");
            text.push_str(&parameter.init_name);
        }
        text.push_str("):\n");
        for parameter in parameter_infos {
            text.push_str(METHOD_OBJECT_BODY_INDENT);
            text.push_str("self.");
            text.push_str(&parameter.stored_name);
            text.push_str(" = ");
            text.push_str(&parameter.init_name);
            text.push('\n');
        }
        text.push('\n');
    }
    text.push_str(METHOD_OBJECT_MEMBER_INDENT);
    text.push_str("def __call__(self):\n");
    let reindented_body = reindent_block(renamed_body, body_indent, METHOD_OBJECT_BODY_INDENT);
    text.push_str(&reindented_body);
    if !reindented_body.ends_with('\n') {
        text.push('\n');
    }
    text
}
