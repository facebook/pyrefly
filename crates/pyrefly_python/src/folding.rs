/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::lined_buffer::LineNumber;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_body;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::comment_section::CommentSection;
use crate::docstring::Docstring;
use crate::module::Module;

/// Semantic category of a folding range before conversion to LSP kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldKind {
    Code,
    Comment,
    CommentSection,
    Region,
}

/// Find the folding ranges (where you can collapse the code) in a module, given the AST.
pub fn folding_ranges(module: &Module, body: &[Stmt]) -> Vec<(TextRange, FoldKind)> {
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
        ranges: Vec<(TextRange, FoldKind)>,
        module: &'a Module,
    }

    impl Visitor<'_> for FoldingRangeCollector<'_> {
        fn visit_body(&mut self, body: &[Stmt]) {
            if let Some(range) = Docstring::range_from_stmts(body) {
                self.ranges.push((range, FoldKind::Comment));
            }
            walk_body(self, body);
        }

        fn visit_stmt(&mut self, stmt: &Stmt) {
            match stmt {
                Stmt::FunctionDef(func) if !func.body.is_empty() => {
                    let range = range_without_decorators(func.range, &func.decorator_list);
                    self.ranges.push((range, FoldKind::Code));
                }
                Stmt::ClassDef(class) if !class.body.is_empty() => {
                    let range = range_without_decorators(class.range, &class.decorator_list);
                    self.ranges.push((range, FoldKind::Code));
                }
                Stmt::If(if_stmt) => {
                    if !if_stmt.body.is_empty() {
                        self.ranges.push((if_stmt.range, FoldKind::Code));
                    }
                    for elif_else in &if_stmt.elif_else_clauses {
                        if !elif_else.body.is_empty() {
                            self.ranges.push((elif_else.range, FoldKind::Code));
                        }
                    }
                }
                Stmt::For(for_stmt) if !for_stmt.body.is_empty() => {
                    self.ranges.push((for_stmt.range, FoldKind::Code));
                }
                Stmt::While(while_stmt) if !while_stmt.body.is_empty() => {
                    self.ranges.push((while_stmt.range, FoldKind::Code));
                }
                Stmt::With(with_stmt) if !with_stmt.body.is_empty() => {
                    self.ranges.push((with_stmt.range, FoldKind::Code));
                }
                Stmt::Match(match_stmt) => {
                    self.ranges.push((match_stmt.range, FoldKind::Code));
                    for case in &match_stmt.cases {
                        if !case.body.is_empty() {
                            self.ranges.push((case.range, FoldKind::Code));
                        }
                    }
                }
                Stmt::Try(try_stmt) => {
                    if !try_stmt.body.is_empty() {
                        self.ranges.push((try_stmt.range, FoldKind::Code));
                    }
                    for handler in &try_stmt.handlers {
                        let ExceptHandler::ExceptHandler(handler_inner) = handler;
                        if !handler_inner.body.is_empty() {
                            self.ranges.push((handler_inner.range(), FoldKind::Code));
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
                    self.ranges.push((range, FoldKind::Code));
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
        collector.ranges.push((range, FoldKind::Comment));
    }

    for stmt in body {
        Visitor::visit_stmt(&mut collector, stmt);
    }

    // Add comment section folding ranges
    add_comment_section_ranges(&mut collector.ranges, module);

    // Explicit regions follow VS Code's Python folding marker syntax.
    let lined_buffer = module.lined_buffer();
    let mut region_starts = Vec::new();
    for (line_number, line) in lined_buffer.lines().enumerate() {
        let line_number = u32::try_from(line_number).expect("module line number should fit in u32");
        let Some(marker) = line.trim_start().strip_prefix('#').map(str::trim_start) else {
            continue;
        };
        let has_marker = |prefix| {
            marker.strip_prefix(prefix).is_some_and(|rest| {
                rest.chars()
                    .next()
                    .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
            })
        };
        if has_marker("region") {
            region_starts.push(lined_buffer.line_start(LineNumber::from_zero_indexed(line_number)));
        } else if has_marker("endregion")
            && let Some(start) = region_starts.pop()
        {
            let next_line = line_number
                .checked_add(1)
                .expect("module line count should fit in u32");
            let end = if usize::try_from(next_line).expect("u32 should fit in usize")
                < lined_buffer.line_count()
            {
                lined_buffer.line_start(LineNumber::from_zero_indexed(next_line))
            } else {
                TextSize::try_from(module.contents().len())
                    .expect("module contents should fit in TextSize")
            };
            collector
                .ranges
                .push((TextRange::new(start, end), FoldKind::Region));
        }
    }

    collector.ranges.sort_by_key(|(range, _)| range.start());
    collector.ranges.dedup();
    collector.ranges
}

/// Add folding ranges for comment sections.
/// Each section folds from its line to the line before the next section at the same or higher level.
fn add_comment_section_ranges(ranges: &mut Vec<(TextRange, FoldKind)>, module: &Module) {
    let sections = CommentSection::extract_from_module(module);

    for (i, section) in sections.iter().enumerate() {
        // Find the end of this section by looking for the next section at the same or higher level
        let end_line = if let Some(next_section_idx) = sections[i + 1..]
            .iter()
            .position(|s| s.level <= section.level)
        {
            // End at the line before the next section
            let next_section = &sections[i + 1 + next_section_idx];
            if next_section.line_number > 0 {
                next_section.line_number - 1
            } else {
                next_section.line_number
            }
        } else {
            // No next section at same/higher level, fold to end of file
            module.lined_buffer().line_count() as u32 - 1
        };

        // Only create a folding range if there's at least one line to fold
        if end_line > section.line_number {
            let line_start = module
                .lined_buffer()
                .line_start(LineNumber::from_zero_indexed(section.line_number));
            let line_end = if (end_line as usize) < module.lined_buffer().line_count() {
                module
                    .lined_buffer()
                    .line_start(LineNumber::from_zero_indexed(end_line + 1))
            } else {
                TextSize::try_from(module.contents().len())
                    .expect("module contents should fit in TextSize")
            };

            let range = TextRange::new(line_start, line_end);
            ranges.push((range, FoldKind::CommentSection));
        }
    }
}
