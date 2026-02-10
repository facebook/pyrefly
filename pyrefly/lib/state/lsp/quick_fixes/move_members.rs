/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::is_member_stmt;
use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::member_name_from_stmt;
use super::extract_shared::prepare_insertion_text;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

const DEFAULT_INDENT: &str = "    ";

/// Builds pull-members-up refactor actions for the supplied selection.
pub(crate) fn pull_members_up_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    with_move_context(
        transaction,
        handle,
        selection,
        |module_info, ast, class_def, member_stmt, member_name, member_indent| {
            let base_classes = base_classes_in_module(ast, class_def);
            let mut actions = Vec::new();
            for base_class in base_classes {
                if class_has_member_named(base_class, member_name) {
                    continue;
                }
                if let Some(action) = build_move_action(
                    module_info,
                    member_stmt,
                    member_indent,
                    class_def,
                    base_class,
                    format!("Pull `{member_name}` up to `{}`", base_class.name.id),
                ) {
                    actions.push(action);
                }
            }
            actions.sort_by(|a, b| a.title.cmp(&b.title));
            if actions.is_empty() {
                None
            } else {
                Some(actions)
            }
        },
    )
}

/// Builds push-members-down refactor actions for the supplied selection.
pub(crate) fn push_members_down_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    with_move_context(
        transaction,
        handle,
        selection,
        |module_info, ast, class_def, member_stmt, member_name, member_indent| {
            let subclasses = subclasses_in_module(ast, class_def);
            let mut eligible_subclasses: Vec<&StmtClassDef> = Vec::new();
            let mut actions = Vec::new();
            for subclass in subclasses {
                if class_has_member_named(subclass, member_name) {
                    continue;
                }
                eligible_subclasses.push(subclass);
                if let Some(action) = build_move_action(
                    module_info,
                    member_stmt,
                    member_indent,
                    class_def,
                    subclass,
                    format!("Push `{member_name}` down to `{}`", subclass.name.id),
                ) {
                    actions.push(action);
                }
            }
            if eligible_subclasses.len() > 1
                && let Some(action) = build_move_action_multi_target(
                    module_info,
                    member_stmt,
                    member_indent,
                    class_def,
                    &eligible_subclasses,
                    format!("Push `{member_name}` down to all subclasses"),
                )
            {
                actions.push(action);
            }
            actions.sort_by(|a, b| a.title.cmp(&b.title));
            if actions.is_empty() {
                None
            } else {
                Some(actions)
            }
        },
    )
}

fn with_move_context(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
    build: impl FnOnce(
        &pyrefly_python::module::Module,
        &ModModule,
        &StmtClassDef,
        &Stmt,
        &str,
        &str,
    ) -> Option<Vec<LocalRefactorCodeAction>>,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let (class_def, member_stmt) = find_member_context(ast.as_ref(), selection_point)?;
    let member_name = member_name_from_stmt(member_stmt)?;
    let member_indent = line_indent_and_start(source, member_stmt.range().start())?.0;
    build(
        &module_info,
        ast.as_ref(),
        class_def,
        member_stmt,
        &member_name,
        &member_indent,
    )
}

/// Locate the class member statement targeted by the selection.
fn find_member_context<'a>(
    ast: &'a ModModule,
    selection: TextSize,
) -> Option<(&'a StmtClassDef, &'a Stmt)> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
            && let Some(found) = find_member_in_class(class_def, selection)
        {
            return Some(found);
        }
    }
    None
}

/// Search within a class body (recursing into nested classes) for a member
/// statement that contains the selection.
fn find_member_in_class<'a>(
    class_def: &'a StmtClassDef,
    selection: TextSize,
) -> Option<(&'a StmtClassDef, &'a Stmt)> {
    for stmt in &class_def.body {
        if let Stmt::ClassDef(inner_class) = stmt
            && inner_class.range().contains(selection)
            && let Some(found) = find_member_in_class(inner_class, selection)
        {
            return Some(found);
        }
    }
    for stmt in &class_def.body {
        if stmt.range().contains(selection) && is_member_stmt(stmt) {
            return Some((class_def, stmt));
        }
    }
    None
}

fn class_has_member_named(class_def: &StmtClassDef, name: &str) -> bool {
    class_def
        .body
        .iter()
        .any(|stmt| member_name_from_stmt(stmt).is_some_and(|n| n == name))
}

fn base_classes_in_module<'a>(
    ast: &'a ModModule,
    class_def: &StmtClassDef,
) -> Vec<&'a StmtClassDef> {
    let Some(arguments) = &class_def.arguments else {
        return Vec::new();
    };
    // Only handle direct name bases for now.
    let base_names = arguments
        .args
        .iter()
        .filter_map(|expr| match expr {
            Expr::Name(name) => Some(name.id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if base_names.is_empty() {
        return Vec::new();
    }
    let mut classes = Vec::new();
    collect_class_defs(&ast.body, &mut classes);
    classes
        .into_iter()
        .filter(|cls| base_names.iter().any(|base| *base == cls.name.id.as_str()))
        .collect()
}

fn subclasses_in_module<'a>(ast: &'a ModModule, class_def: &StmtClassDef) -> Vec<&'a StmtClassDef> {
    let mut classes = Vec::new();
    collect_class_defs(&ast.body, &mut classes);
    classes
        .into_iter()
        .filter(|cls| {
            let Some(arguments) = &cls.arguments else {
                return false;
            };
            // Only handle direct name bases for now.
            arguments.args.iter().any(|expr| {
                matches!(expr, Expr::Name(name) if name.id.as_str() == class_def.name.id.as_str())
            })
        })
        .collect()
}

fn collect_class_defs<'a>(body: &'a [Stmt], out: &mut Vec<&'a StmtClassDef>) {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt {
            out.push(class_def);
            collect_class_defs(&class_def.body, out);
        }
    }
}

fn build_move_action(
    module_info: &pyrefly_python::module::Module,
    member_stmt: &Stmt,
    member_indent: &str,
    origin_class: &StmtClassDef,
    target_class: &StmtClassDef,
    title: String,
) -> Option<LocalRefactorCodeAction> {
    let insert_edit = build_insertion_edit(module_info, member_stmt, member_indent, target_class)?;
    let removal_edit = build_removal_edit(module_info, member_stmt, member_indent, origin_class)?;

    Some(LocalRefactorCodeAction {
        title,
        edits: vec![insert_edit, removal_edit],
        kind: CodeActionKind::new("refactor.move"),
    })
}

fn build_move_action_multi_target(
    module_info: &pyrefly_python::module::Module,
    member_stmt: &Stmt,
    member_indent: &str,
    origin_class: &StmtClassDef,
    target_classes: &[&StmtClassDef],
    title: String,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    for target_class in target_classes {
        let insert_edit =
            build_insertion_edit(module_info, member_stmt, member_indent, target_class)?;
        edits.push(insert_edit);
    }
    let removal_edit = build_removal_edit(module_info, member_stmt, member_indent, origin_class)?;
    edits.push(removal_edit);
    Some(LocalRefactorCodeAction {
        title,
        edits,
        kind: CodeActionKind::new("refactor.move"),
    })
}

/// Build an insertion edit for placing the member into the target class.
fn build_insertion_edit(
    module_info: &pyrefly_python::module::Module,
    member_stmt: &Stmt,
    member_indent: &str,
    target_class: &StmtClassDef,
) -> Option<(pyrefly_python::module::Module, TextRange, String)> {
    let source = module_info.contents();
    let (target_indent, insert_range, replaces_pass) =
        target_insertion_point(target_class, source)?;
    let member_text =
        member_text_for_target(source, member_stmt.range(), member_indent, &target_indent);
    let insert_text = if replaces_pass {
        member_text
    } else {
        prepare_insertion_text(source, insert_range.start(), &member_text)
    };
    Some((module_info.dupe(), insert_range, insert_text))
}

/// Build a removal edit for the member in its origin class, inserting `pass` if needed.
fn build_removal_edit(
    module_info: &pyrefly_python::module::Module,
    member_stmt: &Stmt,
    member_indent: &str,
    origin_class: &StmtClassDef,
) -> Option<(pyrefly_python::module::Module, TextRange, String)> {
    let source = module_info.contents();
    let removal_range = statement_removal_range(source, member_stmt)?;
    let removal_text = if needs_pass_after_removal(origin_class, member_stmt) {
        format!("{member_indent}pass\n")
    } else {
        String::new()
    };
    Some((module_info.dupe(), removal_range, removal_text))
}

/// Determine target insertion indentation and range.
/// Returns `(indent, insert_range, replaces_pass)`.
fn target_insertion_point(
    class_def: &StmtClassDef,
    source: &str,
) -> Option<(String, TextRange, bool)> {
    if let Some(pass_stmt) = replaceable_pass_stmt(class_def) {
        let (indent, line_start) = line_indent_and_start(source, pass_stmt.range().start())?;
        let line_end = line_end_position(source, pass_stmt.range().end());
        return Some((indent, TextRange::new(line_start, line_end), true));
    }
    if let Some(last_stmt) = class_def.body.last() {
        let (indent, _) = line_indent_and_start(source, last_stmt.range().start())?;
        let insert_position = line_end_position(source, last_stmt.range().end());
        return Some((
            indent,
            TextRange::at(insert_position, TextSize::new(0)),
            false,
        ));
    }
    let (class_indent, _) = line_indent_and_start(source, class_def.range().start())?;
    let insert_position = line_end_position(source, class_def.range().end());
    Some((
        format!("{class_indent}{DEFAULT_INDENT}"),
        TextRange::at(insert_position, TextSize::new(0)),
        false,
    ))
}

/// Return the `pass` statement if the class body contains only a docstring and a single `pass`.
fn replaceable_pass_stmt(class_def: &StmtClassDef) -> Option<&Stmt> {
    let mut non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next()?;
    if non_docstring.next().is_some() {
        return None;
    }
    match only_stmt {
        Stmt::Pass(_) => Some(only_stmt),
        _ => None,
    }
}

/// Determine whether removing `member_stmt` would leave an empty class body.
fn needs_pass_after_removal(class_def: &StmtClassDef, member_stmt: &Stmt) -> bool {
    let mut non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next();
    non_docstring.next().is_none()
        && only_stmt.is_some_and(|stmt| stmt.range() == member_stmt.range())
}

/// Remove the full statement line, including leading indent and trailing newline.
fn statement_removal_range(source: &str, stmt: &Stmt) -> Option<TextRange> {
    let (_, line_start) = line_indent_and_start(source, stmt.range().start())?;
    let line_end = line_end_position(source, stmt.range().end());
    Some(TextRange::new(line_start, line_end))
}

/// Reindent the member block so it aligns with the target class indentation.
fn member_text_for_target(
    source: &str,
    range: TextRange,
    from_indent: &str,
    to_indent: &str,
) -> String {
    let start = range.start().to_usize().min(source.len());
    let end = range.end().to_usize().min(source.len());
    let raw = if start < end { &source[start..end] } else { "" };
    let mut text = reindent_block(raw, from_indent, to_indent);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}
