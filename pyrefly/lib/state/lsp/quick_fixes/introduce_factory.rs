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
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::member_name_from_stmt;
use super::extract_shared::selection_anchor;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const DEFAULT_FACTORY_NAME: &str = "create";
const DEFAULT_INDENT: &str = "    ";

/// Builds introduce-factory refactor actions for a selected class name.
pub(crate) fn introduce_factory_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let class_def = find_class_at_selection(ast.as_ref(), selection_point)?;
    if !contains_point(class_def.name.range(), selection_point) {
        return None;
    }

    let definition = transaction
        .find_definition(
            handle,
            class_def.name.range().start(),
            FindPreference::default(),
        )
        .map(Vec1::into_vec)
        .unwrap_or_default()
        .into_iter()
        .find(|def| {
            def.module.path() == module_info.path()
                && def.definition_range.contains_range(class_def.name.range())
                && def.metadata.symbol_kind() == Some(SymbolKind::Class)
        })?;

    let factory_name = generate_factory_name(class_def);
    let factory_edit =
        build_factory_insertion_edit(&module_info, class_def, source, factory_name.as_str())?;
    let callsite_edits =
        build_constructor_callsite_edits(transaction, &definition, factory_name.as_str())?;

    let mut edits = vec![factory_edit];
    edits.extend(callsite_edits);
    Some(vec![LocalRefactorCodeAction {
        title: format!(
            "Introduce factory `{factory_name}` for `{}`",
            class_def.name.id
        ),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

fn find_class_at_selection(ast: &ModModule, selection_point: TextSize) -> Option<&StmtClassDef> {
    Ast::locate_node(ast, selection_point)
        .into_iter()
        .find_map(|node| node.as_stmt_class_def().copied())
}

fn contains_point(range: TextRange, point: TextSize) -> bool {
    range.start() <= point && point < range.end()
}

fn generate_factory_name(class_def: &StmtClassDef) -> String {
    let existing_names: HashSet<String> = class_def
        .body
        .iter()
        .filter_map(member_name_from_stmt)
        .collect();
    unique_name(DEFAULT_FACTORY_NAME, |name| existing_names.contains(name))
}

fn build_factory_insertion_edit(
    module_info: &pyrefly_python::module::Module,
    class_def: &StmtClassDef,
    source: &str,
    factory_name: &str,
) -> Option<(pyrefly_python::module::Module, TextRange, String)> {
    let (class_indent, _) = line_indent_and_start(source, class_def.range().start())?;
    let insert_position = line_end_position(source, class_def.body.last()?.range().end());
    let method_indent = format!("{class_indent}{DEFAULT_INDENT}");
    let body_indent = format!("{method_indent}{DEFAULT_INDENT}");
    let insert_text = format!(
        "\n{method_indent}@staticmethod\n{method_indent}def {factory_name}(*args, **kwargs):\n{body_indent}return {}(*args, **kwargs)\n",
        class_def.name.id
    );
    Some((
        module_info.dupe(),
        TextRange::at(insert_position, TextSize::new(0)),
        insert_text,
    ))
}

fn build_constructor_callsite_edits(
    transaction: &Transaction<'_>,
    definition: &crate::state::lsp::FindDefinitionItemWithDocstring,
    factory_name: &str,
) -> Option<Vec<(pyrefly_python::module::Module, TextRange, String)>> {
    let mut edits = Vec::new();
    for module_handle in transaction.handles() {
        let Some(module_info) = transaction.get_module_info(&module_handle) else {
            continue;
        };
        let Some(refs) = transaction.local_references_from_definition(
            &module_handle,
            definition.metadata.clone(),
            definition.definition_range,
            &definition.module,
            false,
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
            match constructor_call_kind(call, &ref_set) {
                ConstructorCallKind::Unrelated => {}
                ConstructorCallKind::Unsupported => failed = true,
                ConstructorCallKind::Rewritable => {
                    let Some((range, text)) = build_factory_call_edit(call, &ref_set, factory_name)
                    else {
                        failed = true;
                        return;
                    };
                    module_edits.push((module_info.dupe(), range, text));
                }
            }
        });
        if failed {
            return None;
        }
        edits.extend(module_edits);
    }
    Some(edits)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstructorCallKind {
    Unrelated,
    Rewritable,
    Unsupported,
}

fn constructor_call_kind(call: &ExprCall, ref_set: &HashSet<TextRange>) -> ConstructorCallKind {
    match call.func.as_ref() {
        Expr::Name(name) => ref_set
            .contains(&name.range())
            .then_some(ConstructorCallKind::Rewritable)
            .unwrap_or(ConstructorCallKind::Unrelated),
        Expr::Attribute(attribute) => ref_set
            .contains(&attribute.attr.range())
            .then_some(ConstructorCallKind::Rewritable)
            .unwrap_or(ConstructorCallKind::Unrelated),
        _ => ref_set
            .iter()
            .any(|range| call.func.range().contains(range.start()))
            .then_some(ConstructorCallKind::Unsupported)
            .unwrap_or(ConstructorCallKind::Unrelated),
    }
}

fn build_factory_call_edit(
    call: &ExprCall,
    ref_set: &HashSet<TextRange>,
    factory_name: &str,
) -> Option<(TextRange, String)> {
    match call.func.as_ref() {
        Expr::Name(name) if ref_set.contains(&name.range()) => {
            Some((name.range(), format!("{}.{}", name.id, factory_name)))
        }
        Expr::Attribute(attribute) if ref_set.contains(&attribute.attr.range()) => Some((
            attribute.attr.range(),
            format!("{}.{}", attribute.attr.id, factory_name),
        )),
        _ => None,
    }
}
