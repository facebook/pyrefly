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
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::code_at_range;
use super::extract_shared::find_enclosing_statement_range;
use super::extract_shared::is_exact_expression;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use super::extract_shared::split_selection;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

const BODY_INDENT: &str = "    ";
const DEFAULT_MODEL_NAME: &str = "Model";

struct DictField {
    name: String,
    annotation: String,
}

/// Builds code actions that generate a TypedDict, dataclass, or Pydantic model
/// definition from a selected dict literal.
pub(crate) fn dict_definition_code_actions(
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
    let mut requires_any = false;
    let fields = collect_dict_fields(transaction, handle, &stdlib, dict_expr, &mut requires_any)?;
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
    requires_any: &mut bool,
) -> Option<Vec<DictField>> {
    let mut fields = Vec::new();
    let mut seen_names = HashSet::new();
    for item in &dict_expr.items {
        let key_expr = item.key.as_ref()?;
        let key = string_literal_key(key_expr)?;
        if !is_valid_identifier(&key) {
            return None;
        }
        if !seen_names.insert(key.clone()) {
            continue;
        }
        let annotation = infer_field_annotation(transaction, handle, stdlib, item.value.range());
        if annotation == "Any" {
            *requires_any = true;
        }
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

fn type_to_annotation(ty: Type, stdlib: &Stdlib) -> Option<String> {
    let ty = ty.promote_implicit_literals(stdlib);
    if ty.is_any() {
        return None;
    }
    let parts = ty.get_types_with_locations(Some(stdlib));
    Some(parts.into_iter().map(|(part, _)| part).collect())
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

fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_alphanumeric() || c == '_') {
        return false;
    }
    !is_keyword(name)
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "match"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "type"
            | "while"
            | "with"
            | "yield"
            | "case"
    )
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
    if requires_any && let Some(line) = build_import_line(ast, "typing", &["Any"]) {
        import_text.push_str(&line);
    }
    if let Some(line) = build_import_line(ast, "dataclasses", &["dataclass"]) {
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
    if requires_any && let Some(line) = build_import_line(ast, "typing", &["Any"]) {
        import_text.push_str(&line);
    }
    if let Some(line) = build_import_line(ast, "pydantic", &["BaseModel"]) {
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
    let import_text = build_import_line(ast, module_name, names)?;
    let position = import_insertion_point(ast);
    Some((
        module_info.dupe(),
        TextRange::at(position, TextSize::new(0)),
        import_text,
    ))
}

fn build_import_line(ast: &ModModule, module_name: &str, names: &[&str]) -> Option<String> {
    let missing = missing_imports(ast, module_name, names);
    if missing.is_empty() {
        return None;
    }
    Some(format!(
        "from {module_name} import {}\n",
        missing.join(", ")
    ))
}

fn missing_imports<'a>(ast: &ModModule, module_name: &str, names: &'a [&'a str]) -> Vec<&'a str> {
    let mut missing: Vec<&str> = names.to_vec();
    for stmt in &ast.body {
        let Stmt::ImportFrom(import_from) = stmt else {
            continue;
        };
        let Some(module) = &import_from.module else {
            continue;
        };
        if module.id.as_str() != module_name {
            continue;
        }
        for alias in &import_from.names {
            let name = alias.name.id.as_str();
            let is_same_binding = match &alias.asname {
                None => true,
                Some(asname) => asname.id.as_str() == name,
            };
            if !is_same_binding {
                continue;
            }
            if let Some(index) = missing.iter().position(|candidate| *candidate == name) {
                missing.remove(index);
            }
        }
    }
    missing
}

fn import_insertion_point(ast: &ModModule) -> TextSize {
    if let Some(first_stmt) = ast.body.iter().find(|stmt| !is_docstring_stmt(stmt)) {
        first_stmt.range().start()
    } else {
        ast.range.end()
    }
}
