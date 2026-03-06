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
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtDelete;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::member_name_from_stmt;
use super::extract_shared::prepare_insertion_text;
use super::extract_shared::selection_anchor;
use super::extract_shared::unique_name;
use super::types::LocalRefactorCodeAction;
use crate::module::module_info::ModuleInfo;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::Transaction;

/// Builds an encapsulate-field refactor for an attribute under the cursor.
pub(crate) fn encapsulate_field_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let position = selection_anchor(source, selection);
    let identifier = transaction.identifier_at(handle, position)?;
    if !matches!(identifier.context, IdentifierContext::Attribute { .. }) {
        return None;
    }
    let field_name = identifier.identifier.id.to_string();
    let definition = transaction
        .find_definition(handle, position, FindPreference::default())
        .into_iter()
        .find(|definition| {
            definition.module.path() == module_info.path()
                && matches!(
                    definition.metadata,
                    DefinitionMetadata::Attribute | DefinitionMetadata::VariableOrAttribute(_)
                )
                && !matches!(
                    definition.metadata.symbol_kind(),
                    Some(
                        SymbolKind::Module
                            | SymbolKind::Function
                            | SymbolKind::Method
                            | SymbolKind::Class
                    )
                )
        })?;
    let ast = transaction.get_ast(handle)?;
    if definition_is_callable_member(ast.as_ref(), definition.definition_range) {
        return None;
    }
    let class_def = enclosing_class(ast.as_ref(), definition.definition_range)?;
    let getter_name = unique_name(&format!("get_{field_name}"), |name| {
        class_has_member_named(class_def, name)
    });
    let setter_name = unique_name(&format!("set_{field_name}"), |name| {
        name == getter_name || class_has_member_named(class_def, name)
    });

    let references =
        transaction.find_local_references(handle, definition.definition_range.start(), true);
    if references.is_empty() {
        return None;
    }

    let mut edits = Vec::new();
    let mut handled_write_stmt_ranges = Vec::new();
    for reference in references {
        let occurrence = classify_occurrence(ast.as_ref(), reference)?;
        if occurrence.is_definition_write()
            && handled_write_stmt_ranges.contains(&occurrence.stmt_range)
        {
            continue;
        }
        if reference == definition.definition_range && occurrence.is_definition_write() {
            continue;
        }
        match occurrence.kind {
            OccurrenceKind::Read(attribute) => {
                let receiver = module_info.code_at(attribute.value.range());
                edits.push((
                    module_info.dupe(),
                    attribute.range(),
                    format!("{receiver}.{getter_name}()"),
                ));
            }
            OccurrenceKind::SimpleWrite { attribute, value } => {
                handled_write_stmt_ranges.push(occurrence.stmt_range);
                let receiver = module_info.code_at(attribute.value.range());
                edits.push((
                    module_info.dupe(),
                    attribute.range(),
                    format!("{receiver}.{setter_name}("),
                ));
                let gap_range = TextRange::new(attribute.range().end(), value.range().start());
                if !gap_range.is_empty() {
                    edits.push((module_info.dupe(), gap_range, String::new()));
                }
                edits.push((
                    module_info.dupe(),
                    TextRange::at(value.range().end(), TextSize::new(0)),
                    ")".to_owned(),
                ));
            }
        }
    }

    let methods_edit = build_methods_insertion_edit(
        &module_info,
        class_def,
        &field_name,
        &getter_name,
        &setter_name,
    )?;
    edits.push(methods_edit);

    Some(vec![LocalRefactorCodeAction {
        title: format!("Encapsulate field `{field_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

#[derive(Clone, Copy)]
struct Occurrence<'a> {
    stmt_range: TextRange,
    kind: OccurrenceKind<'a>,
}

impl Occurrence<'_> {
    fn is_definition_write(self) -> bool {
        matches!(self.kind, OccurrenceKind::SimpleWrite { .. })
    }
}

#[derive(Clone, Copy)]
enum OccurrenceKind<'a> {
    Read(&'a ExprAttribute),
    SimpleWrite {
        attribute: &'a ExprAttribute,
        value: &'a Expr,
    },
}

fn classify_occurrence(ast: &ModModule, reference: TextRange) -> Option<Occurrence<'_>> {
    let covering_nodes = Ast::locate_node(ast, reference.start());
    let attribute = covering_nodes.iter().find_map(|node| match node {
        AnyNodeRef::ExprAttribute(attribute) if attribute.attr.range() == reference => {
            Some(*attribute)
        }
        _ => None,
    })?;
    let stmt = find_enclosing_statement(&ast.body, reference)?;
    match attribute.ctx {
        ExprContext::Load => Some(Occurrence {
            stmt_range: stmt.range(),
            kind: OccurrenceKind::Read(attribute),
        }),
        ExprContext::Store => classify_write_occurrence(stmt, attribute),
        ExprContext::Del | ExprContext::Invalid => None,
    }
}

fn find_enclosing_statement<'a>(body: &'a [Stmt], reference: TextRange) -> Option<&'a Stmt> {
    for stmt in body {
        if !stmt.range().contains_range(reference) {
            continue;
        }
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if let Some(found) = find_enclosing_statement(&function_def.body, reference) {
                    return Some(found);
                }
            }
            Stmt::ClassDef(class_def) => {
                if let Some(found) = find_enclosing_statement(&class_def.body, reference) {
                    return Some(found);
                }
            }
            _ => {}
        }
        return Some(stmt);
    }
    None
}

fn classify_write_occurrence<'a>(
    stmt: &'a Stmt,
    attribute: &'a ExprAttribute,
) -> Option<Occurrence<'a>> {
    match stmt {
        Stmt::Assign(assign) => classify_assign_occurrence(stmt, assign, attribute),
        Stmt::AnnAssign(assign) => {
            direct_attribute_target(assign.target.as_ref(), attribute)?;
            let value = assign.value.as_deref()?;
            Some(Occurrence {
                stmt_range: stmt.range(),
                kind: OccurrenceKind::SimpleWrite { attribute, value },
            })
        }
        Stmt::AugAssign(assign) => classify_augassign_occurrence(stmt, assign, attribute),
        Stmt::Delete(delete) => classify_delete_occurrence(delete, attribute),
        _ => None,
    }
}

fn classify_assign_occurrence<'a>(
    stmt: &'a Stmt,
    assign: &'a StmtAssign,
    attribute: &'a ExprAttribute,
) -> Option<Occurrence<'a>> {
    if assign.targets.len() != 1 {
        return None;
    }
    direct_attribute_target(&assign.targets[0], attribute)?;
    Some(Occurrence {
        stmt_range: stmt.range(),
        kind: OccurrenceKind::SimpleWrite {
            attribute,
            value: assign.value.as_ref(),
        },
    })
}

fn classify_augassign_occurrence<'a>(
    _stmt: &'a Stmt,
    _assign: &'a StmtAugAssign,
    attribute: &'a ExprAttribute,
) -> Option<Occurrence<'a>> {
    direct_attribute_target(_assign.target.as_ref(), attribute)?;
    // Reject augmented assignments for now. Rewriting `x += y` as `set_x(get_x() + y)`
    // is not semantics-preserving for in-place operators like `list.__iadd__`.
    None
}

fn classify_delete_occurrence<'a>(
    delete: &'a StmtDelete,
    attribute: &'a ExprAttribute,
) -> Option<Occurrence<'a>> {
    if delete
        .targets
        .iter()
        .any(|target| direct_attribute_target(target, attribute).is_some())
    {
        return None;
    }
    unreachable!("delete target attributes should always use ExprContext::Del")
}

fn direct_attribute_target<'a>(
    expr: &'a Expr,
    attribute: &ExprAttribute,
) -> Option<&'a ExprAttribute> {
    match expr {
        Expr::Attribute(target) if target.range() == attribute.range() => Some(target),
        _ => None,
    }
}

fn definition_is_callable_member(ast: &ModModule, definition_range: TextRange) -> bool {
    Ast::locate_node(ast, definition_range.start())
        .into_iter()
        .any(|node| {
            matches!(
                node,
                AnyNodeRef::StmtFunctionDef(function_def)
                    if function_def.name.range() == definition_range
            ) || matches!(
                node,
                AnyNodeRef::StmtClassDef(class_def)
                    if class_def.name.range() == definition_range
            )
        })
}

fn enclosing_class<'a>(
    ast: &'a ModModule,
    definition_range: TextRange,
) -> Option<&'a StmtClassDef> {
    Ast::locate_node(ast, definition_range.start())
        .into_iter()
        .find_map(|node| match node {
            AnyNodeRef::StmtClassDef(class_def)
                if class_def.range().contains_range(definition_range) =>
            {
                Some(class_def)
            }
            _ => None,
        })
}

fn class_has_member_named(class_def: &StmtClassDef, name: &str) -> bool {
    class_def
        .body
        .iter()
        .any(|stmt| member_name_from_stmt(stmt).is_some_and(|member| member == name))
}

fn build_methods_insertion_edit(
    module_info: &ModuleInfo,
    class_def: &StmtClassDef,
    field_name: &str,
    getter_name: &str,
    setter_name: &str,
) -> Option<(pyrefly_python::module::Module, TextRange, String)> {
    let source = module_info.contents();
    let (indent, insert_range, replaces_pass) = class_method_insertion_point(class_def, source)?;
    let methods = format!(
        "{indent}def {getter_name}(self):\n{indent}    return self.{field_name}\n\n{indent}def {setter_name}(self, value):\n{indent}    self.{field_name} = value\n"
    );
    let mut insert_text = String::new();
    if !replaces_pass && !class_def.body.is_empty() {
        insert_text.push('\n');
    }
    insert_text.push_str(&methods);
    let insert_text = if replaces_pass {
        insert_text
    } else {
        prepare_insertion_text(source, insert_range.start(), &insert_text)
    };
    Some((module_info.dupe(), insert_range, insert_text))
}

fn class_method_insertion_point(
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
        format!("{class_indent}    "),
        TextRange::at(insert_position, TextSize::new(0)),
        false,
    ))
}

fn replaceable_pass_stmt(class_def: &StmtClassDef) -> Option<&Stmt> {
    let mut non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt));
    let pass_stmt = non_docstring.next()?;
    if !matches!(pass_stmt, Stmt::Pass(_)) {
        return None;
    }
    if non_docstring.next().is_some() {
        return None;
    }
    Some(pass_stmt)
}
