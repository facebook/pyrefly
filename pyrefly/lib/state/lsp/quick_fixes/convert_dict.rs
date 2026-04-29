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
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ModModule;
use ruff_python_stdlib::identifiers::is_identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::build_from_import_edit;
use super::extract_shared::build_from_import_line;
use super::extract_shared::code_at_range;
use super::extract_shared::find_enclosing_statement_range;
use super::extract_shared::import_insertion_point;
use super::extract_shared::is_exact_expression;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use super::extract_shared::split_selection;
use super::extract_shared::type_to_annotation;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;
use crate::types::stdlib::Stdlib;

const BODY_INDENT: &str = "    ";
const DEFAULT_MODEL_NAME: &str = "Model";

struct DictField {
    name: String,
    annotation: String,
}

/// Builds code actions that generate a TypedDict, dataclass, or Pydantic model
/// definition from a selected dict literal.
pub(crate) fn convert_dict_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let (dict_range, dict_expr) = find_dict_expression(ast.as_ref(), source, selection)?;
    let (statement_indent, insert_position) =
        match find_enclosing_statement_range(ast.as_ref(), dict_range)
            .and_then(|range| line_indent_and_start(source, range.start()))
        {
            Some(result) => result,
            None => line_indent_and_start(source, dict_range.start())?,
        };
    let insert_range = TextRange::at(insert_position, TextSize::new(0));
    let stdlib = transaction.get_stdlib(handle);
    let fields = collect_dict_fields(transaction, handle, &stdlib, dict_expr)?;
    let requires_any = fields.iter().any(|field| field.annotation == "Any");
    let base_name = suggest_base_name(ast.as_ref(), dict_range);
    let class_name = unique_name(&to_pascal_case(&base_name), |candidate| {
        name_conflicts(source, candidate)
    });

    Some(vec![
        build_typed_dict_action(
            &module_info,
            ast.as_ref(),
            &class_name,
            &fields,
            requires_any,
            insert_range,
            &statement_indent,
        )?,
        build_dataclass_action(
            &module_info,
            ast.as_ref(),
            &class_name,
            &fields,
            requires_any,
            insert_range,
            &statement_indent,
        )?,
        build_pydantic_action(
            &module_info,
            ast.as_ref(),
            &class_name,
            &fields,
            requires_any,
            insert_range,
            &statement_indent,
        )?,
    ])
}

fn find_dict_expression<'a>(
    ast: &'a ModModule,
    source: &str,
    selection: TextRange,
) -> Option<(TextRange, &'a ExprDict)> {
    if selection.is_empty() {
        let anchor = selection_anchor(source, selection);
        for node in Ast::locate_node(ast, anchor) {
            if let AnyNodeRef::ExprDict(dict_expr) = node {
                return Some((dict_expr.range(), dict_expr));
            }
        }
        return None;
    }

    let selection_text = code_at_range(source, selection)?;
    let (_, _, _, expr_range) = split_selection(selection_text, selection)?;
    if !is_exact_expression(ast, expr_range) {
        return None;
    }
    for node in Ast::locate_node(ast, expr_range.start()) {
        if let AnyNodeRef::ExprDict(dict_expr) = node
            && dict_expr.range() == expr_range
        {
            return Some((expr_range, dict_expr));
        }
    }
    None
}

fn collect_dict_fields(
    transaction: &Transaction<'_>,
    handle: &Handle,
    stdlib: &Stdlib,
    dict_expr: &ExprDict,
) -> Option<Vec<DictField>> {
    let mut fields: Vec<DictField> = Vec::new();
    let mut field_positions: HashMap<String, usize> = HashMap::new();
    for item in &dict_expr.items {
        let key_expr = item.key.as_ref()?;
        let key = string_literal_key(key_expr)?;
        if !is_identifier(&key) {
            return None;
        }
        let annotation = infer_field_annotation(transaction, handle, stdlib, item.value.range());
        if let Some(index) = field_positions.get(&key).copied() {
            fields[index].annotation = annotation;
            continue;
        }
        field_positions.insert(key.clone(), fields.len());
        fields.push(DictField {
            name: key,
            annotation,
        });
    }
    Some(fields)
}

fn string_literal_key(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => literal
            .as_single_part_string()
            .map(|part| part.as_str().to_owned()),
        _ => None,
    }
}

fn infer_field_annotation(
    transaction: &Transaction<'_>,
    handle: &Handle,
    stdlib: &Stdlib,
    range: TextRange,
) -> String {
    match transaction.get_type_trace(handle, range) {
        Some(ty) => type_to_annotation(ty, stdlib).unwrap_or_else(|| "Any".to_owned()),
        None => "Any".to_owned(),
    }
}

fn suggest_base_name(ast: &ModModule, dict_range: TextRange) -> String {
    let nodes = Ast::locate_node(ast, dict_range.start());
    let mut function_name = None;
    for node in &nodes {
        if let AnyNodeRef::StmtFunctionDef(def) = node {
            function_name = Some(def.name.id.to_string());
            break;
        }
    }
    for node in &nodes {
        match node {
            AnyNodeRef::StmtAssign(assign)
                if assign.value.range() == dict_range && assign.targets.len() == 1 =>
            {
                if let Expr::Name(name) = &assign.targets[0] {
                    return name.id.to_string();
                }
            }
            AnyNodeRef::StmtAnnAssign(assign)
                if assign
                    .value
                    .as_ref()
                    .is_some_and(|value| value.range() == dict_range) =>
            {
                if let Expr::Name(name) = assign.target.as_ref() {
                    return name.id.to_string();
                }
            }
            _ => {}
        }
    }
    for node in &nodes {
        if let AnyNodeRef::StmtReturn(ret) = node
            && ret
                .value
                .as_ref()
                .is_some_and(|value| value.range() == dict_range)
            && let Some(function_name) = &function_name
        {
            return function_name.clone();
        }
    }
    DEFAULT_MODEL_NAME.to_owned()
}

fn to_pascal_case(name: &str) -> String {
    let mut result = String::new();
    let mut capitalize = true;
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            if capitalize {
                result.extend(ch.to_uppercase());
                capitalize = false;
            } else {
                result.push(ch);
            }
        } else {
            capitalize = true;
        }
    }
    if result.is_empty() {
        result = DEFAULT_MODEL_NAME.to_owned();
    }
    if !result
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
    {
        result = format!("{DEFAULT_MODEL_NAME}{result}");
    }
    result
}

fn name_conflicts(source: &str, name: &str) -> bool {
    let patterns = [
        format!("class {name}"),
        format!("def {name}"),
        format!("{name} ="),
        format!("{name}\t="),
    ];
    patterns.iter().any(|pattern| source.contains(pattern))
}

fn build_typed_dict_action(
    module_info: &Module,
    ast: &ModModule,
    class_name: &str,
    fields: &[DictField],
    requires_any: bool,
    insert_range: TextRange,
    indent: &str,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    let mut imports = vec!["TypedDict"];
    if requires_any {
        imports.push("Any");
    }
    if let Some(import_edit) = build_import_edit(module_info, ast, "typing", &imports) {
        edits.push(import_edit);
    }
    let class_text = build_class_body(indent, class_name, "TypedDict", fields, None);
    edits.push((module_info.dupe(), insert_range, class_text));
    Some(LocalRefactorCodeAction {
        title: format!("Create TypedDict `{class_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    })
}

fn build_dataclass_action(
    module_info: &Module,
    ast: &ModModule,
    class_name: &str,
    fields: &[DictField],
    requires_any: bool,
    insert_range: TextRange,
    indent: &str,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    let position = import_insertion_point(ast);
    let mut import_text = String::new();
    if requires_any && let Some(line) = build_from_import_line(ast, "typing", &["Any"]) {
        import_text.push_str(&line);
    }
    if let Some(line) = build_from_import_line(ast, "dataclasses", &["dataclass"]) {
        import_text.push_str(&line);
    }
    if !import_text.is_empty() {
        edits.push((
            module_info.dupe(),
            TextRange::at(position, TextSize::new(0)),
            import_text,
        ));
    }
    let class_text = build_class_body(indent, class_name, "", fields, Some("@dataclass"));
    edits.push((module_info.dupe(), insert_range, class_text));
    Some(LocalRefactorCodeAction {
        title: format!("Create dataclass `{class_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    })
}

fn build_pydantic_action(
    module_info: &Module,
    ast: &ModModule,
    class_name: &str,
    fields: &[DictField],
    requires_any: bool,
    insert_range: TextRange,
    indent: &str,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    let position = import_insertion_point(ast);
    let mut import_text = String::new();
    if requires_any && let Some(line) = build_from_import_line(ast, "typing", &["Any"]) {
        import_text.push_str(&line);
    }
    if let Some(line) = build_from_import_line(ast, "pydantic", &["BaseModel"]) {
        import_text.push_str(&line);
    }
    if !import_text.is_empty() {
        edits.push((
            module_info.dupe(),
            TextRange::at(position, TextSize::new(0)),
            import_text,
        ));
    }
    let class_text = build_class_body(indent, class_name, "BaseModel", fields, None);
    edits.push((module_info.dupe(), insert_range, class_text));
    Some(LocalRefactorCodeAction {
        title: format!("Create pydantic model `{class_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    })
}

fn build_class_body(
    indent: &str,
    class_name: &str,
    base: &str,
    fields: &[DictField],
    decorator: Option<&str>,
) -> String {
    let mut output = String::new();
    if let Some(decorator) = decorator {
        output.push_str(indent);
        output.push_str(decorator);
        output.push('\n');
    }
    output.push_str(indent);
    output.push_str("class ");
    output.push_str(class_name);
    if !base.is_empty() {
        output.push('(');
        output.push_str(base);
        output.push(')');
    }
    output.push_str(":\n");
    if fields.is_empty() {
        output.push_str(indent);
        output.push_str(BODY_INDENT);
        output.push_str("pass\n");
        return output;
    }
    for field in fields {
        output.push_str(indent);
        output.push_str(BODY_INDENT);
        output.push_str(&field.name);
        output.push_str(": ");
        output.push_str(&field.annotation);
        output.push('\n');
    }
    output
}

fn build_import_edit(
    module_info: &Module,
    ast: &ModModule,
    module_name: &str,
    names: &[&str],
) -> Option<(Module, TextRange, String)> {
    build_from_import_edit(module_info, ast, module_name, names)
}
