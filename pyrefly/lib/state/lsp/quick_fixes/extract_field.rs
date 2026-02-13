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
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::first_parameter_name;
use super::extract_shared::function_has_decorator;
use super::extract_shared::is_exact_expression;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::split_selection;
use super::extract_shared::validate_non_empty_selection;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const DEFAULT_FIELD_PREFIX: &str = "extracted_field";

pub(crate) fn extract_field_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let selection_text = validate_non_empty_selection(selection, module_info.code_at(selection))?;
    let (leading_ws, expression_text, trailing_ws, expression_range) =
        split_selection(selection_text, selection)?;
    if !is_exact_expression(ast.as_ref(), expression_range) {
        return None;
    }
    let context = find_field_context(ast.as_ref(), expression_range, source)?;
    let loads = collect_expression_loads(ast.as_ref(), expression_range);
    let module_path = module_info.path();
    for ident in &loads {
        if ident.name == context.receiver_name {
            return None;
        }
        let definitions =
            transaction.find_definition(handle, ident.position, FindPreference::default());
        if definitions.iter().any(|definition| {
            definition.module.path() == module_path
                && context
                    .method_range
                    .contains_range(definition.definition_range)
        }) {
            return None;
        }
    }
    let field_name = generate_field_name(source, context.class_range);
    let assignment = build_field_assignment(
        source,
        context.insert_position,
        &context.field_indent,
        &field_name,
        expression_text,
    );
    let insert_edit = (
        module_info.dupe(),
        TextRange::at(context.insert_position, TextSize::new(0)),
        assignment,
    );
    let replacement = format!(
        "{leading_ws}{receiver}.{field_name}{trailing_ws}",
        receiver = context.receiver_name
    );
    let replace_edit = (module_info.dupe(), selection, replacement);
    Some(vec![LocalRefactorCodeAction {
        title: format!(
            "Extract into field `{field_name}` on `{}`",
            context.class_name
        ),
        edits: vec![insert_edit, replace_edit],
        kind: CodeActionKind::REFACTOR_EXTRACT,
    }])
}

struct FieldContext {
    class_name: String,
    receiver_name: String,
    class_range: TextRange,
    method_range: TextRange,
    field_indent: String,
    insert_position: TextSize,
}

fn find_field_context(ast: &ModModule, selection: TextRange, source: &str) -> Option<FieldContext> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains_range(selection)
            && let Some(context) = find_field_context_in_class(class_def, selection, source)
        {
            return Some(context);
        }
    }
    None
}

fn find_field_context_in_class(
    class_def: &StmtClassDef,
    selection: TextRange,
    source: &str,
) -> Option<FieldContext> {
    for stmt in &class_def.body {
        match stmt {
            Stmt::FunctionDef(function_def) if function_def.range().contains_range(selection) => {
                if function_has_decorator(function_def, "staticmethod") {
                    return None;
                }
                let receiver_name = first_parameter_name(&function_def.parameters)?;
                let (field_indent, insert_position) = field_insertion_point(class_def, source)?;
                return Some(FieldContext {
                    class_name: class_def.name.id.to_string(),
                    receiver_name,
                    class_range: class_def.range(),
                    method_range: function_def.range(),
                    field_indent,
                    insert_position,
                });
            }
            Stmt::ClassDef(inner_class) if inner_class.range().contains_range(selection) => {
                if let Some(context) = find_field_context_in_class(inner_class, selection, source) {
                    return Some(context);
                }
            }
            _ => {}
        }
    }
    None
}

fn field_insertion_point(class_def: &StmtClassDef, source: &str) -> Option<(String, TextSize)> {
    for stmt in &class_def.body {
        if is_docstring_stmt(stmt) {
            continue;
        }
        let (indent, line_start) = line_indent_and_start(source, stmt.range().start())?;
        return Some((indent, line_start));
    }
    if let Some(docstring) = class_def.body.first() {
        let (indent, _) = line_indent_and_start(source, docstring.range().start())?;
        return Some((indent, docstring.range().end()));
    }
    let (class_indent, _) = line_indent_and_start(source, class_def.range().start())?;
    Some((format!("{class_indent}    "), class_def.range().end()))
}

struct IdentifierRef {
    name: String,
    position: TextSize,
}

fn collect_expression_loads(ast: &ModModule, selection: TextRange) -> Vec<IdentifierRef> {
    struct Collector {
        selection: TextRange,
        loads: Vec<IdentifierRef>,
    }
    impl<'a> Visitor<'a> for Collector {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if self.selection.contains_range(expr.range())
                && let Expr::Name(name) = expr
                && matches!(name.ctx, ExprContext::Load)
            {
                self.loads.push(IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                });
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }
    let mut collector = Collector {
        selection,
        loads: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.loads
}

fn generate_field_name(source: &str, class_range: TextRange) -> String {
    let start = class_range.start().to_usize().min(source.len());
    let end = class_range.end().to_usize().min(source.len());
    let class_source = if start < end { &source[start..end] } else { "" };
    let mut counter = 1;
    loop {
        let candidate = if counter == 1 {
            DEFAULT_FIELD_PREFIX.to_owned()
        } else {
            format!("{DEFAULT_FIELD_PREFIX}_{counter}")
        };
        let check_space = format!("{candidate} =");
        let check_tab = format!("{candidate}\t=");
        if !class_source.contains(&check_space) && !class_source.contains(&check_tab) {
            return candidate;
        }
        counter += 1;
    }
}

fn build_field_assignment(
    source: &str,
    insert_position: TextSize,
    indent: &str,
    field_name: &str,
    expression: &str,
) -> String {
    let mut assignment = String::new();
    if needs_leading_newline(source, insert_position) {
        assignment.push('\n');
    }
    assignment.push_str(indent);
    assignment.push_str(field_name);
    assignment.push_str(" = ");
    assignment.push_str(expression);
    assignment.push('\n');
    assignment
}

fn needs_leading_newline(source: &str, insert_position: TextSize) -> bool {
    let idx = insert_position.to_usize();
    idx > 0 && !source[..idx].ends_with('\n')
}
