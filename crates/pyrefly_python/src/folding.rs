/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::FoldingRangeKind;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_body;
use ruff_text_size::TextRange;

use crate::docstring::Docstring;
use crate::module::Module;

/// Find the folding ranges (where you can collapse the code) in a module, given the AST.
pub fn folding_ranges(
    module: &Module,
    body: &[Stmt],
) -> Vec<(TextRange, Option<FoldingRangeKind>)> {
    use ruff_python_ast::ExceptHandler;
    use ruff_text_size::Ranged;

    fn range_without_decorators(
        range: TextRange,
        decorators: &[ruff_python_ast::Decorator],
    ) -> TextRange {
        let decorators_range = decorators
            .first()
            .map(|first| first.range().cover(decorators.last().unwrap().range()));

        decorators_range.map_or(range, |x| {
            range.add_start(x.len() + ruff_text_size::TextSize::from(1))
        })
    }

    struct FoldingRangeCollector<'a> {
        ranges: Vec<(TextRange, Option<FoldingRangeKind>)>,
        module: &'a Module,
    }

    impl Visitor<'_> for FoldingRangeCollector<'_> {
        fn visit_body(&mut self, body: &[Stmt]) {
            if let Some(range) = Docstring::range_from_stmts(body) {
                self.ranges.push((range, Some(FoldingRangeKind::Comment)));
            }
            walk_body(self, body);
        }

        fn visit_stmt(&mut self, stmt: &Stmt) {
            match stmt {
                Stmt::FunctionDef(func) => {
                    if !func.body.is_empty() {
                        let range = range_without_decorators(func.range, &func.decorator_list);
                        self.ranges.push((range, None));
                    }
                }
                Stmt::ClassDef(class) => {
                    if !class.body.is_empty() {
                        let range = range_without_decorators(class.range, &class.decorator_list);
                        self.ranges.push((range, None));
                    }
                }
                Stmt::If(if_stmt) => {
                    if !if_stmt.body.is_empty() {
                        self.ranges.push((if_stmt.range, None));
                    }
                    for elif_else in &if_stmt.elif_else_clauses {
                        if !elif_else.body.is_empty() {
                            self.ranges.push((elif_else.range, None));
                        }
                    }
                }
                Stmt::For(for_stmt) => {
                    if !for_stmt.body.is_empty() {
                        self.ranges.push((for_stmt.range, None));
                    }
                }
                Stmt::While(while_stmt) => {
                    if !while_stmt.body.is_empty() {
                        self.ranges.push((while_stmt.range, None));
                    }
                }
                Stmt::With(with_stmt) => {
                    if !with_stmt.body.is_empty() {
                        self.ranges.push((with_stmt.range, None));
                    }
                }
                Stmt::Match(match_stmt) => {
                    self.ranges.push((match_stmt.range, None));
                    for case in &match_stmt.cases {
                        if !case.body.is_empty() {
                            self.ranges.push((case.range, None));
                        }
                    }
                }
                Stmt::Try(try_stmt) => {
                    if !try_stmt.body.is_empty() {
                        self.ranges.push((try_stmt.range, None));
                    }
                    for handler in &try_stmt.handlers {
                        let ExceptHandler::ExceptHandler(handler_inner) = handler;
                        if !handler_inner.body.is_empty() {
                            self.ranges.push((handler_inner.range(), None));
                        }
                    }
                }
                _ => {}
            }
            ruff_python_ast::visitor::walk_stmt(self, stmt);
        }

        fn visit_expr(&mut self, expr: &Expr) {
            let range = match expr {
                Expr::Call(call) => Some(call.arguments.range),
                Expr::Dict(dict) => Some(dict.range),
                Expr::List(list) => Some(list.range),
                Expr::Set(set) => Some(set.range),
                Expr::Tuple(tuple) => Some(tuple.range),
                _ => None,
            };

            if let Some(range) = range {
                let lsp_range = self.module.to_lsp_range(range);
                if lsp_range.start.line != lsp_range.end.line {
                    self.ranges.push((range, None));
                }
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut collector = FoldingRangeCollector {
        ranges: Vec::new(),
        module,
    };

    if let Some(range) = Docstring::range_from_stmts(body) {
        collector
            .ranges
            .push((range, Some(FoldingRangeKind::Comment)));
    }

    for stmt in body {
        Visitor::visit_stmt(&mut collector, stmt);
    }

    collector.ranges.sort_by_key(|(range, _)| range.start());
    collector.ranges.dedup();
    collector.ranges
}
