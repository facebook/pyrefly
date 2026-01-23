/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

pub(super) fn split_selection<'a>(
    selection_text: &'a str,
    selection_range: TextRange,
) -> Option<(&'a str, &'a str, &'a str, TextRange)> {
    let trimmed_start = selection_text.trim_start_matches(char::is_whitespace);
    let leading_len = selection_text.len() - trimmed_start.len();
    let trimmed = trimmed_start.trim_end_matches(char::is_whitespace);
    let trailing_len = trimmed_start.len() - trimmed.len();
    if trimmed.is_empty() || trimmed.contains('\n') {
        return None;
    }
    let leading_ws = &selection_text[..leading_len];
    let trailing_ws = &selection_text[selection_text.len() - trailing_len..];
    let leading_size = TextSize::try_from(leading_len).ok()?;
    let trailing_size = TextSize::try_from(trailing_len).ok()?;
    let expr_start = selection_range.start() + leading_size;
    let expr_end = selection_range.end() - trailing_size;
    if expr_start >= expr_end {
        return None;
    }
    Some((
        leading_ws,
        trimmed,
        trailing_ws,
        TextRange::new(expr_start, expr_end),
    ))
}

pub(super) fn is_exact_expression(ast: &ModModule, selection: TextRange) -> bool {
    Ast::locate_node(ast, selection.start())
        .into_iter()
        .any(|node| node.as_expr_ref().is_some() && node.range() == selection)
}

pub(super) fn line_indent_and_start(
    source: &str,
    position: TextSize,
) -> Option<(String, TextSize)> {
    let mut idx = position.to_usize();
    if idx > source.len() {
        idx = source.len();
    }
    let line_start = source[..idx]
        .rfind('\n')
        .map(|start| start + 1)
        .unwrap_or(0);
    let indent = source[line_start..idx]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let insert_position = TextSize::try_from(line_start).ok()?;
    Some((indent, insert_position))
}

pub(super) fn first_parameter_name(parameters: &Parameters) -> Option<String> {
    if let Some(param) = parameters.posonlyargs.first() {
        return Some(param.name().id.to_string());
    }
    parameters
        .args
        .first()
        .map(|param| param.name().id.to_string())
}

pub(super) fn function_has_decorator(function_def: &StmtFunctionDef, decorator: &str) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|d| decorator_matches_name(&d.expression, decorator))
}

pub(super) fn decorator_matches_name(decorator: &Expr, expected: &str) -> bool {
    match decorator {
        Expr::Name(identifier) => identifier.id.as_str() == expected,
        Expr::Attribute(attribute) => attribute.attr.as_str() == expected,
        Expr::Call(call) => decorator_matches_name(call.func.as_ref(), expected),
        _ => false,
    }
}

/// Given a selection range, returns the first non-whitespace position within it.
/// If the selection is empty, returns the start position.
pub(super) fn selection_anchor(source: &str, selection: TextRange) -> TextSize {
    if selection.is_empty() {
        return selection.start();
    }
    let start = selection.start().to_usize().min(source.len());
    let end = selection.end().to_usize().min(source.len());
    if start >= end {
        return selection.start();
    }
    if let Some(offset) = source[start..end]
        .char_indices()
        .find(|(_, ch)| !matches!(ch, ' ' | '\t' | '\n' | '\r'))
        .map(|(idx, _)| idx)
    {
        TextSize::try_from(start + offset).unwrap_or(selection.start())
    } else {
        selection.start()
    }
}

/// Extracts the name from a statement that defines a named symbol.
/// Returns `None` for statements that don't define a single named symbol.
pub(super) fn member_name_from_stmt(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::FunctionDef(func_def) => Some(func_def.name.id.to_string()),
        Stmt::ClassDef(class_def) => Some(class_def.name.id.to_string()),
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            if let Expr::Name(name) = &assign.targets[0] {
                Some(name.id.to_string())
            } else {
                None
            }
        }
        Stmt::AnnAssign(assign) => {
            if let Expr::Name(name) = assign.target.as_ref() {
                Some(name.id.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Checks if an expression creates a new scope where variable semantics differ.
/// These include lambdas and comprehensions, where inlining could change behavior.
pub(super) fn is_disallowed_scope_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Lambda(_)
            | Expr::ListComp(_)
            | Expr::SetComp(_)
            | Expr::DictComp(_)
            | Expr::Generator(_)
    )
}

/// Checks if a reference position is inside a disallowed scope expression.
pub(super) fn reference_in_disallowed_scope(ast: &ModModule, reference: TextRange) -> bool {
    Ast::locate_node(ast, reference.start())
        .into_iter()
        .any(|node| {
            matches!(
                node,
                AnyNodeRef::ExprLambda(_)
                    | AnyNodeRef::ExprListComp(_)
                    | AnyNodeRef::ExprSetComp(_)
                    | AnyNodeRef::ExprDictComp(_)
                    | AnyNodeRef::ExprGenerator(_)
            )
        })
}

/// Collects references to a named identifier in statements or expressions.
///
/// This utility consolidates the common pattern of:
/// 1. Finding Load-context references to a name
/// 2. Detecting Store-context references (which invalidate inline operations)
/// 3. Skipping nested function/class definitions (different scope)
/// 4. Rejecting if references appear in disallowed expressions (lambdas, comprehensions)
///
/// The collector becomes "invalid" if:
/// - A Store-context reference to the name is found
/// - A Load-context reference appears inside a lambda or comprehension
pub(super) struct NameRefCollector {
    pub name: String,
    pub load_refs: Vec<TextRange>,
    pub invalid: bool,
}

impl NameRefCollector {
    /// Creates a new collector for the given identifier name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            load_refs: Vec::new(),
            invalid: false,
        }
    }

    /// Collects references from statements, skipping nested function/class definitions.
    pub fn visit_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            if self.invalid {
                return;
            }
            self.visit_stmt(stmt);
        }
    }
}

impl Visitor<'_> for NameRefCollector {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.invalid {
            return;
        }
        // Skip nested function and class definitions to avoid capturing
        // references in different scopes.
        match stmt {
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if self.invalid {
            return;
        }
        // Disallow references inside lambdas and comprehensions where
        // inlining could change semantics.
        if is_disallowed_scope_expr(expr) {
            self.invalid = true;
            return;
        }
        if let Expr::Name(name) = expr
            && name.id.as_str() == self.name
        {
            match name.ctx {
                ExprContext::Store => {
                    self.invalid = true;
                    return;
                }
                ExprContext::Load => {
                    self.load_refs.push(name.range());
                }
                _ => {}
            }
        }
        walk_expr(self, expr);
    }
}

/// Generates a unique name by appending `_N` suffixes until the name is not in use.
/// The `exists` predicate should return true if a name is already taken.
pub(super) fn unique_name(base: &str, exists: impl Fn(&str) -> bool) -> String {
    if !exists(base) {
        return base.to_owned();
    }
    let mut counter = 1;
    loop {
        let candidate = format!("{base}_{counter}");
        if !exists(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

/// Finds the innermost function definition that contains the given range.
pub(super) fn find_enclosing_function(
    ast: &ModModule,
    range: TextRange,
) -> Option<&StmtFunctionDef> {
    let covering_nodes = Ast::locate_node(ast, range.start());
    for node in covering_nodes {
        if let Some(function_def) = node.as_stmt_function_def()
            && function_def.range().contains_range(range)
        {
            return Some(function_def);
        }
    }
    None
}

/// Returns true if a visitor should recurse into this statement for intra-function analysis.
/// Returns false for nested function and class definitions, which create new scopes.
pub(super) fn is_local_scope_stmt(stmt: &Stmt) -> bool {
    !matches!(stmt, Stmt::FunctionDef(_) | Stmt::ClassDef(_))
}

/// Core information about a method within a class.
/// Contains the common fields needed by various refactoring operations.
#[derive(Clone, Debug)]
pub(super) struct MethodInfo {
    /// Name of the class containing the method.
    pub class_name: String,
    /// Name of the receiver parameter (typically `self` or `cls`).
    pub receiver_name: String,
}

/// Extracts the text at the given range from a source string.
/// Returns `None` if the range extends beyond the source bounds.
pub(super) fn code_at_range<'a>(source: &'a str, range: TextRange) -> Option<&'a str> {
    let start = range.start().to_usize();
    let end = range.end().to_usize();
    if end <= source.len() {
        Some(&source[start..end])
    } else {
        None
    }
}
