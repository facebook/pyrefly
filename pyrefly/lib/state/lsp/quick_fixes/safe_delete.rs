/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParentKind {
    Module,
    Function,
    Class,
}

struct DefinitionContext<'a> {
    stmt_range: TextRange,
    parent_body: &'a [Stmt],
    parent: ParentKind,
}

/// Builds safe-delete refactor actions for the supplied selection.
pub(crate) fn safe_delete_code_actions(
    transaction: &mut Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let position = selection.start();
    let identifier = transaction.identifier_at(handle, position)?;
    let definitions = transaction.find_definition(handle, position, FindPreference::default());
    if definitions.len() != 1 {
        return None;
    }
    let definition = definitions.into_iter().next()?;
    if definition.module.path() != module_info.path() {
        return None;
    }
    let symbol_kind = definition.metadata.symbol_kind()?;
    if !is_supported_symbol_kind(symbol_kind) {
        return None;
    }
    let context = find_definition_context(ast.as_ref(), definition.definition_range)?;
    let references = transaction
        .find_global_references_from_definition(
            handle.sys_info(),
            definition.metadata.clone(),
            TextRangeWithModule::new(definition.module.dupe(), definition.definition_range),
        )
        .ok()?;
    if has_non_definition_reference(&references, &definition.module, definition.definition_range) {
        return None;
    }
    let removal_range =
        statement_removal_range_from_range(module_info.contents(), context.stmt_range)?;
    let replacement = if context.parent == ParentKind::Module {
        String::new()
    } else if needs_pass_after_removal(context.parent_body, context.stmt_range) {
        let (indent, _) =
            line_indent_and_start(module_info.contents(), context.stmt_range.start())?;
        format!("{indent}pass\n")
    } else {
        String::new()
    };
    let title_name = definition
        .display_name
        .clone()
        .unwrap_or_else(|| identifier.identifier.id.to_string());
    Some(vec![LocalRefactorCodeAction {
        title: format!("Safe delete `{}`", title_name),
        edits: vec![(module_info.dupe(), removal_range, replacement)],
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

fn is_supported_symbol_kind(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Function
            | SymbolKind::Method
            | SymbolKind::Class
            | SymbolKind::Variable
            | SymbolKind::Constant
            | SymbolKind::TypeAlias
    )
}

fn has_non_definition_reference(
    references: &[(Module, Vec<TextRange>)],
    definition_module: &Module,
    definition_range: TextRange,
) -> bool {
    let mut saw_definition = false;
    for (module, ranges) in references {
        for range in ranges {
            if module.path() == definition_module.path() && *range == definition_range {
                saw_definition = true;
            } else {
                return true;
            }
        }
    }
    !saw_definition
}

fn find_definition_context<'a>(
    ast: &'a ModModule,
    definition_range: TextRange,
) -> Option<DefinitionContext<'a>> {
    find_definition_context_in_body(&ast.body, definition_range, ParentKind::Module)
}

fn find_definition_context_in_body<'a>(
    body: &'a [Stmt],
    definition_range: TextRange,
    parent: ParentKind,
) -> Option<DefinitionContext<'a>> {
    for stmt in body {
        if matches_definition(stmt, definition_range) {
            return Some(DefinitionContext {
                stmt_range: stmt.range(),
                parent_body: body,
                parent,
            });
        }
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if let Some(found) = find_definition_context_in_body(
                    &function_def.body,
                    definition_range,
                    ParentKind::Function,
                ) {
                    return Some(found);
                }
            }
            Stmt::ClassDef(class_def) => {
                if let Some(found) = find_definition_context_in_body(
                    &class_def.body,
                    definition_range,
                    ParentKind::Class,
                ) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

fn matches_definition(stmt: &Stmt, definition_range: TextRange) -> bool {
    match stmt {
        Stmt::FunctionDef(function_def) => function_def.name.range() == definition_range,
        Stmt::ClassDef(class_def) => class_def.name.range() == definition_range,
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return false;
            }
            matches!(&assign.targets[0], Expr::Name(name) if name.range() == definition_range)
        }
        Stmt::AnnAssign(assign) => matches!(
            assign.target.as_ref(),
            Expr::Name(name) if name.range() == definition_range
        ),
        Stmt::TypeAlias(type_alias) => type_alias.name.range() == definition_range,
        _ => false,
    }
}

fn statement_removal_range_from_range(source: &str, range: TextRange) -> Option<TextRange> {
    let (_, line_start) = line_indent_and_start(source, range.start())?;
    let line_end = line_end_position(source, range.end());
    Some(TextRange::new(line_start, line_end))
}

fn needs_pass_after_removal(body: &[Stmt], removed_range: TextRange) -> bool {
    let mut non_docstring = body.iter().filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next();
    non_docstring.next().is_none() && only_stmt.is_some_and(|stmt| stmt.range() == removed_range)
}
