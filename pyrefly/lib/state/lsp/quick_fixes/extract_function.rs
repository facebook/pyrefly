/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use pyrefly_python::docstring::dedent_block_preserving_layout;
use pyrefly_python::module::Module;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

const HELPER_INDENT: &str = "    ";

/// Description of a refactor edit that stays within the local workspace.
#[derive(Clone, Debug)]
pub struct LocalRefactorCodeAction {
    pub title: String,
    pub edits: Vec<(Module, TextRange, String)>,
    pub kind: CodeActionKind,
}

/// Builds extract-function quick fix code actions for the supplied selection.
pub(crate) fn extract_function_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    if selection.is_empty() {
        return None;
    }
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let selection_text = module_info.code_at(selection);
    if selection_text.trim().is_empty() {
        return None;
    }
    let module_len = TextSize::try_from(module_info.contents().len()).unwrap_or(TextSize::new(0));
    let module_stmt_range =
        find_enclosing_module_statement_range(ast.as_ref(), selection, module_len);
    if selection_contains_disallowed_statements(ast.as_ref(), selection) {
        return None;
    }
    let (load_refs, store_refs) = collect_identifier_refs(ast.as_ref(), selection);
    if load_refs.is_empty() && store_refs.is_empty() {
        return None;
    }
    let post_loads = collect_post_selection_loads(ast.as_ref(), module_stmt_range, selection.end());
    let block_indent = detect_block_indent(selection_text);
    let mut dedented_body = dedent_block_preserving_layout(selection_text)?;
    if dedented_body.ends_with('\n') {
        dedented_body.pop();
        if dedented_body.ends_with('\r') {
            dedented_body.pop();
        }
    }

    let helper_name = generate_helper_name(module_info.contents());
    let mut params = Vec::new();
    let mut seen_params = HashSet::new();
    for ident in load_refs {
        if seen_params.contains(&ident.name) {
            continue;
        }
        if ident.synthetic_load {
            seen_params.insert(ident.name.clone());
            params.push(ident.name.clone());
            continue;
        }
        let defs = transaction.find_definition(handle, ident.position, FindPreference::default());
        let Some(def) = defs.first() else {
            continue;
        };
        if def.module.path() != module_info.path() {
            continue;
        }
        if !module_stmt_range.contains_range(def.definition_range)
            || selection.contains_range(def.definition_range)
            || def.definition_range.start() >= selection.start()
        {
            continue;
        }
        seen_params.insert(ident.name.clone());
        params.push(ident.name.clone());
    }

    let mut returns = Vec::new();
    let mut seen_returns = HashSet::new();
    for ident in store_refs {
        if seen_returns.contains(&ident.name) || !post_loads.contains(&ident.name) {
            continue;
        }
        seen_returns.insert(ident.name.clone());
        returns.push(ident.name.clone());
    }

    let indented_body = prefix_lines_with(&dedented_body, HELPER_INDENT);

    let mut helper_text = if params.is_empty() {
        format!("def {helper_name}():\n{indented_body}")
    } else {
        let helper_params = params.join(", ");
        format!("def {helper_name}({helper_params}):\n{indented_body}")
    };

    if !returns.is_empty() && !returns.iter().all(|name| name.is_empty()) {
        let return_expr = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        helper_text.push_str(&format!("{HELPER_INDENT}return {return_expr}\n"));
    }
    helper_text.push('\n');

    let call_args = params.join(", ");
    let call_expr = format!("{helper_name}({call_args})");
    let replacement_line = if returns.is_empty() {
        format!("{block_indent}{call_expr}\n")
    } else {
        let lhs = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        format!("{block_indent}{lhs} = {call_expr}\n")
    };

    let helper_edit = (
        module_info.dupe(),
        TextRange::at(module_stmt_range.start(), TextSize::new(0)),
        helper_text,
    );
    let call_edit = (module_info.dupe(), selection, replacement_line);
    let action = LocalRefactorCodeAction {
        title: format!("Extract into helper `{helper_name}`"),
        edits: vec![helper_edit, call_edit],
        kind: CodeActionKind::REFACTOR_EXTRACT,
    };
    Some(vec![action])
}

#[derive(Clone, Debug)]
struct IdentifierRef {
    /// Identifier string.
    name: String,
    /// Byte offset where the identifier was observed.
    position: TextSize,
    /// True when this "load" came from reading the left-hand side of an augmented assignment.
    synthetic_load: bool,
}

fn collect_identifier_refs(
    ast: &ModModule,
    selection: TextRange,
) -> (Vec<IdentifierRef>, Vec<IdentifierRef>) {
    struct IdentifierCollector {
        selection: TextRange,
        loads: Vec<IdentifierRef>,
        stores: Vec<IdentifierRef>,
    }

    impl<'a> ruff_python_ast::visitor::Visitor<'a> for IdentifierCollector {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if self.selection.contains_range(expr.range())
                && let Expr::Name(name) = expr
            {
                let ident = IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: false,
                };
                match name.ctx {
                    ExprContext::Load => self.loads.push(ident),
                    ExprContext::Store => self.stores.push(ident),
                    ExprContext::Del | ExprContext::Invalid => {}
                }
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }

        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if self.selection.contains_range(stmt.range())
                && let Stmt::AugAssign(aug) = stmt
                && let Expr::Name(name) = aug.target.as_ref()
            {
                self.loads.push(IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: true,
                });
            }
            ruff_python_ast::visitor::walk_stmt(self, stmt);
        }
    }

    let mut collector = IdentifierCollector {
        selection,
        loads: Vec::new(),
        stores: Vec::new(),
    };
    collector.visit_body(&ast.body);
    (collector.loads, collector.stores)
}

fn selection_contains_disallowed_statements(ast: &ModModule, selection: TextRange) -> bool {
    fn visit_stmt(stmt: &Stmt, selection: TextRange, found: &mut bool) {
        if *found || stmt.range().intersect(selection).is_none() {
            return;
        }
        if selection.contains_range(stmt.range()) {
            match stmt {
                Stmt::Return(_)
                | Stmt::Break(_)
                | Stmt::Continue(_)
                | Stmt::Raise(_)
                | Stmt::FunctionDef(_)
                | Stmt::ClassDef(_) => {
                    *found = true;
                    return;
                }
                _ => {}
            }
        }
        stmt.recurse(&mut |child| visit_stmt(child, selection, found));
    }

    let mut found = false;
    for stmt in &ast.body {
        visit_stmt(stmt, selection, &mut found);
        if found {
            break;
        }
    }
    found
}

fn find_enclosing_module_statement_range(
    ast: &ModModule,
    selection: TextRange,
    module_len: TextSize,
) -> TextRange {
    for stmt in &ast.body {
        if stmt.range().contains_range(selection) {
            return stmt.range();
        }
    }
    TextRange::new(TextSize::new(0), module_len)
}

fn collect_post_selection_loads(
    ast: &ModModule,
    module_stmt_range: TextRange,
    selection_end: TextSize,
) -> HashSet<String> {
    let mut loads = HashSet::new();
    ast.visit(&mut |expr: &Expr| {
        if let Expr::Name(name) = expr
            && matches!(name.ctx, ExprContext::Load)
            && module_stmt_range.contains_range(name.range)
            && name.range.start() > selection_end
        {
            loads.insert(name.id.to_string());
        }
    });
    loads
}

fn detect_block_indent(selection_text: &str) -> String {
    for line in selection_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        return line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();
    }
    String::new()
}

fn prefix_lines_with(block: &str, indent: &str) -> String {
    let mut result = String::new();
    for line in block.lines() {
        result.push_str(indent);
        result.push_str(line);
        result.push('\n');
    }
    result
}

fn generate_helper_name(source: &str) -> String {
    let mut counter = 1;
    loop {
        let candidate = if counter == 1 {
            "extracted_function".to_owned()
        } else {
            format!("extracted_function_{counter}")
        };
        let needle = format!("def {candidate}(");
        if !source.contains(&needle) {
            return candidate;
        }
        counter += 1;
    }
}
