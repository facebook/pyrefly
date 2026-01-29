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
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

#[derive(Clone, Debug)]
struct LoadEdit {
    range: TextRange,
    replacement: String,
}

#[derive(Clone, Debug)]
struct AssignmentPlan {
    target_range: TextRange,
    value_range: TextRange,
    value_kind: AssignmentValueKind,
}

#[derive(Clone, Debug)]
enum AssignmentValueKind {
    UnaryNot { operand_range: TextRange },
    BooleanLiteral { value: bool },
    Other,
}

/// Builds invert-boolean refactor actions for the supplied selection.
pub(crate) fn invert_boolean_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let position = selection.start();
    let identifier = transaction.identifier_at(handle, position)?;
    if !matches!(identifier.context, IdentifierContext::Expr(_)) {
        return None;
    }
    let target_name = identifier.identifier.id.to_string();
    let definitions = transaction.find_definition(handle, position, FindPreference::default());
    let definition = definitions.first()?;
    if !matches!(definition.metadata, DefinitionMetadata::Variable(_)) {
        return None;
    }
    if definition.module.path() != module_info.path() {
        return None;
    }
    let references = transaction.local_references_from_definition(
        handle,
        definition.metadata.clone(),
        definition.definition_range,
        &definition.module,
    )?;
    let reference_set: HashSet<TextRange> = references.into_iter().collect();

    let assignment_plans = collect_assignment_plans(ast.as_ref(), &target_name, &reference_set);
    if assignment_plans.is_empty() {
        return None;
    }
    let allowed_store_ranges: HashSet<TextRange> = assignment_plans
        .iter()
        .map(|plan| plan.target_range)
        .collect();
    let store_info = collect_store_ranges(ast.as_ref(), &target_name, &reference_set);
    if store_info.has_del
        || store_info
            .stores
            .iter()
            .any(|range| !allowed_store_ranges.contains(range))
    {
        return None;
    }

    let load_edits = collect_load_edits(ast.as_ref(), &target_name, &reference_set);
    let assignment_value_ranges: Vec<TextRange> = assignment_plans
        .iter()
        .map(|plan| plan.value_range)
        .collect();
    let source = module_info.contents();
    let mut edits = Vec::new();
    for plan in &assignment_plans {
        let replacement = build_assignment_replacement(source, plan, &load_edits);
        edits.push((module_info.dupe(), plan.value_range, replacement));
    }
    for edit in load_edits {
        if assignment_value_ranges
            .iter()
            .any(|range| range.contains_range(edit.range))
        {
            continue;
        }
        edits.push((module_info.dupe(), edit.range, edit.replacement));
    }
    if edits.is_empty() {
        return None;
    }
    Some(vec![LocalRefactorCodeAction {
        title: format!("Invert boolean `{target_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

struct StoreInfo {
    stores: Vec<TextRange>,
    has_del: bool,
}

fn collect_store_ranges(
    ast: &ModModule,
    target_name: &str,
    references: &HashSet<TextRange>,
) -> StoreInfo {
    struct StoreCollector<'a> {
        target_name: &'a str,
        references: &'a HashSet<TextRange>,
        stores: Vec<TextRange>,
        has_del: bool,
    }

    impl<'a> Visitor<'a> for StoreCollector<'a> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(name) = expr
                && name.id == self.target_name
                && self.references.contains(&name.range())
            {
                match name.ctx {
                    ExprContext::Store => self.stores.push(name.range()),
                    ExprContext::Del => self.has_del = true,
                    ExprContext::Load | ExprContext::Invalid => {}
                }
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut collector = StoreCollector {
        target_name,
        references,
        stores: Vec::new(),
        has_del: false,
    };
    collector.visit_body(&ast.body);
    StoreInfo {
        stores: collector.stores,
        has_del: collector.has_del,
    }
}

fn collect_assignment_plans(
    ast: &ModModule,
    target_name: &str,
    references: &HashSet<TextRange>,
) -> Vec<AssignmentPlan> {
    struct AssignmentCollector<'a> {
        target_name: &'a str,
        references: &'a HashSet<TextRange>,
        plans: Vec<AssignmentPlan>,
    }

    impl<'a> Visitor<'a> for AssignmentCollector<'a> {
        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            match stmt {
                Stmt::Assign(assign) => {
                    if assign.targets.len() == 1
                        && let Expr::Name(name) = &assign.targets[0]
                        && name.id == self.target_name
                        && self.references.contains(&name.range())
                    {
                        let value_kind = assignment_value_kind(&assign.value);
                        self.plans.push(AssignmentPlan {
                            target_range: name.range(),
                            value_range: assign.value.range(),
                            value_kind,
                        });
                    }
                }
                Stmt::AnnAssign(assign) => {
                    if let Expr::Name(name) = assign.target.as_ref()
                        && name.id == self.target_name
                        && self.references.contains(&name.range())
                        && let Some(value) = assign.value.as_ref()
                    {
                        let value_kind = assignment_value_kind(value);
                        self.plans.push(AssignmentPlan {
                            target_range: name.range(),
                            value_range: value.range(),
                            value_kind,
                        });
                    }
                }
                _ => {}
            }
            ruff_python_ast::visitor::walk_stmt(self, stmt);
        }
    }

    let mut collector = AssignmentCollector {
        target_name,
        references,
        plans: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.plans
}

fn assignment_value_kind(value: &Expr) -> AssignmentValueKind {
    match value {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => AssignmentValueKind::UnaryNot {
            operand_range: unary.operand.range(),
        },
        Expr::BooleanLiteral(boolean_literal) => AssignmentValueKind::BooleanLiteral {
            value: boolean_literal.value,
        },
        _ => AssignmentValueKind::Other,
    }
}

fn collect_load_edits(
    ast: &ModModule,
    target_name: &str,
    references: &HashSet<TextRange>,
) -> Vec<LoadEdit> {
    struct LoadCollector<'a> {
        target_name: &'a str,
        references: &'a HashSet<TextRange>,
        edits: Vec<LoadEdit>,
    }

    impl<'a> Visitor<'a> for LoadCollector<'a> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::UnaryOp(unary) = expr
                && unary.op == UnaryOp::Not
                && let Expr::Name(name) = unary.operand.as_ref()
                && name.id == self.target_name
                && name.ctx == ExprContext::Load
                && self.references.contains(&name.range())
            {
                self.edits.push(LoadEdit {
                    range: unary.range(),
                    replacement: name.id.to_string(),
                });
                // Avoid also visiting the inner `abc` node and producing overlapping edits
                // (`not abc` -> `abc` and `abc` -> `not abc`) for the same source span.
                return;
            }
            if let Expr::Name(name) = expr
                && name.id == self.target_name
                && name.ctx == ExprContext::Load
                && self.references.contains(&name.range())
            {
                self.edits.push(LoadEdit {
                    range: name.range(),
                    replacement: format!("(not {})", name.id),
                });
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut collector = LoadCollector {
        target_name,
        references,
        edits: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.edits
}

fn build_assignment_replacement(
    source: &str,
    plan: &AssignmentPlan,
    load_edits: &[LoadEdit],
) -> String {
    match plan.value_kind {
        AssignmentValueKind::UnaryNot { operand_range } => {
            apply_edits_to_slice(source, operand_range, load_edits)
        }
        AssignmentValueKind::BooleanLiteral { value } => {
            if value {
                "False".to_owned()
            } else {
                "True".to_owned()
            }
        }
        AssignmentValueKind::Other => {
            let transformed = apply_edits_to_slice(source, plan.value_range, load_edits);
            format!("not ({transformed})")
        }
    }
}

fn apply_edits_to_slice(source: &str, base_range: TextRange, edits: &[LoadEdit]) -> String {
    let start = base_range.start().to_usize();
    let end = base_range.end().to_usize();
    let mut result = source[start..end].to_owned();
    let mut relevant_edits: Vec<(usize, usize, &str)> = edits
        .iter()
        .filter(|edit| base_range.contains_range(edit.range))
        .map(|edit| {
            let relative_start = edit.range.start().to_usize().saturating_sub(start);
            let relative_end = edit.range.end().to_usize().saturating_sub(start);
            (relative_start, relative_end, edit.replacement.as_str())
        })
        .collect();
    relevant_edits.sort_by_key(|(start, _, _)| *start);
    for (relative_start, relative_end, replacement) in relevant_edits.into_iter().rev() {
        result.replace_range(relative_start..relative_end, replacement);
    }
    result
}
