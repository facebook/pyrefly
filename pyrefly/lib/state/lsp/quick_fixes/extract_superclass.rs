/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::code_at_range;
use super::extract_shared::is_member_stmt;
use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::prepare_insertion_text;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

const DEFAULT_INDENT: &str = "    ";
const DEFAULT_SUPERCLASS_PREFIX: &str = "Base";

/// Builds extract-superclass refactor actions for the supplied selection.
pub(crate) fn extract_superclass_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let selection_point = selection_anchor(source, selection);
    let class_def = find_class_at_selection(ast.as_ref(), selection_point)?;
    let members = selected_members(class_def, selection);
    if members.is_empty() {
        return None;
    }

    let (class_indent, insert_position) = class_insertion_point(class_def, source)?;
    let superclass_name = generate_superclass_name(source, class_def.name.id.as_str());
    let superclass_text = build_superclass_text(source, &class_indent, &superclass_name, &members)?;
    let insert_text = prepare_insertion_text(source, insert_position, &superclass_text);
    let insert_edit = (
        module_info.dupe(),
        TextRange::at(insert_position, TextSize::new(0)),
        insert_text,
    );

    let base_edit = build_base_insertion_edit(&module_info, class_def, &superclass_name)?;
    let removal_edits = build_removal_edits(&module_info, class_def, source, &members)?;

    let mut edits = Vec::new();
    edits.push(insert_edit);
    edits.push(base_edit);
    edits.extend(removal_edits);

    Some(vec![LocalRefactorCodeAction {
        title: format!("Extract superclass `{superclass_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_EXTRACT,
    }])
}

fn find_class_at_selection<'a>(
    ast: &'a ModModule,
    selection_point: TextSize,
) -> Option<&'a StmtClassDef> {
    Ast::locate_node(ast, selection_point)
        .into_iter()
        .find_map(|node| node.as_stmt_class_def().copied())
}

fn selected_members<'a>(class_def: &'a StmtClassDef, selection: TextRange) -> Vec<&'a Stmt> {
    let members: Vec<&Stmt> = class_def
        .body
        .iter()
        .filter(|stmt| is_member_stmt(stmt))
        .collect();
    if members.is_empty() {
        return Vec::new();
    }
    if selection.is_empty() {
        return members;
    }
    members
        .into_iter()
        .filter(|stmt| ranges_overlap(selection, stmt.range()))
        .collect()
}

fn ranges_overlap(a: TextRange, b: TextRange) -> bool {
    a.start() < b.end() && b.start() < a.end()
}

fn class_insertion_point(class_def: &StmtClassDef, source: &str) -> Option<(String, TextSize)> {
    let anchor = class_def
        .decorator_list
        .iter()
        .map(|decorator| decorator.range().start())
        .min()
        .unwrap_or_else(|| class_def.range().start());
    line_indent_and_start(source, anchor)
}

fn generate_superclass_name(source: &str, class_name: &str) -> String {
    let base = format!("{DEFAULT_SUPERCLASS_PREFIX}{class_name}");
    unique_name(&base, |name| source.contains(&format!("class {name}")))
}

fn build_superclass_text(
    source: &str,
    class_indent: &str,
    superclass_name: &str,
    members: &[&Stmt],
) -> Option<String> {
    let mut text = format!("{class_indent}class {superclass_name}:\n");
    let member_indent = format!("{class_indent}{DEFAULT_INDENT}");
    if members.is_empty() {
        text.push_str(&format!("{member_indent}pass\n\n"));
        return Some(text);
    }
    for member in members {
        let removal_range = statement_removal_range(source, member)?;
        let raw = code_at_range(source, removal_range)?;
        let (from_indent, _) = line_indent_and_start(source, member.range().start())?;
        let mut reindented = reindent_block(raw, &from_indent, &member_indent);
        if !reindented.ends_with('\n') {
            reindented.push('\n');
        }
        text.push_str(&reindented);
    }
    text.push('\n');
    Some(text)
}

fn build_base_insertion_edit(
    module_info: &pyrefly_python::module::Module,
    class_def: &StmtClassDef,
    superclass_name: &str,
) -> Option<(pyrefly_python::module::Module, TextRange, String)> {
    if let Some(arguments) = &class_def.arguments {
        if let Some(first_keyword) = arguments.keywords.first() {
            let insert_pos = first_keyword.range().start();
            let insert_text = format!("{superclass_name}, ");
            return Some((
                module_info.dupe(),
                TextRange::at(insert_pos, TextSize::new(0)),
                insert_text,
            ));
        }
        let insert_pos = arguments.range.end().saturating_sub(TextSize::new(1));
        let insert_text = if arguments.args.is_empty() {
            superclass_name.to_owned()
        } else {
            format!(", {superclass_name}")
        };
        return Some((
            module_info.dupe(),
            TextRange::at(insert_pos, TextSize::new(0)),
            insert_text,
        ));
    }
    let insert_pos = if let Some(type_params) = &class_def.type_params {
        type_params.range.end()
    } else {
        class_def.name.range.end()
    };
    Some((
        module_info.dupe(),
        TextRange::at(insert_pos, TextSize::new(0)),
        format!("({superclass_name})"),
    ))
}

fn build_removal_edits(
    module_info: &pyrefly_python::module::Module,
    class_def: &StmtClassDef,
    source: &str,
    members: &[&Stmt],
) -> Option<Vec<(pyrefly_python::module::Module, TextRange, String)>> {
    let selected_ranges: Vec<TextRange> = members.iter().map(|stmt| stmt.range()).collect();
    let remaining_non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt))
        .filter(|stmt| !selected_ranges.iter().any(|range| *range == stmt.range()))
        .count();
    let needs_pass = remaining_non_docstring == 0;
    let mut edits = Vec::new();
    let mut pass_inserted = false;
    for stmt in members {
        let removal_range = statement_removal_range(source, stmt)?;
        let replacement = if needs_pass && !pass_inserted {
            let (indent, _) = line_indent_and_start(source, stmt.range().start())?;
            pass_inserted = true;
            format!("{indent}pass\n")
        } else {
            String::new()
        };
        edits.push((module_info.dupe(), removal_range, replacement));
    }
    Some(edits)
}

/// Remove the full statement line, including leading indent and trailing newline.
fn statement_removal_range(source: &str, stmt: &Stmt) -> Option<TextRange> {
    let (_, line_start) = line_indent_and_start(source, stmt.range().start())?;
    let line_end = line_end_position(source, stmt.range().end());
    Some(TextRange::new(line_start, line_end))
}
