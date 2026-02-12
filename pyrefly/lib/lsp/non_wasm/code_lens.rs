/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::TextRange;

use crate::state::state::Transaction;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeLensKind {
    Run,
    Test,
}

#[derive(Clone, Copy, Debug)]
pub struct CodeLensEntry {
    pub range: TextRange,
    pub kind: CodeLensKind,
}

impl<'a> Transaction<'a> {
    pub fn code_lens_entries(&self, handle: &Handle) -> Option<Vec<CodeLensEntry>> {
        let ast = self.get_ast(handle)?;
        let mut entries = Vec::new();
        collect_module_entries(&ast.body, &mut entries);
        Some(entries)
    }
}

fn collect_module_entries(stmts: &[Stmt], entries: &mut Vec<CodeLensEntry>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(func) => {
                maybe_push_test(entries, func.name.as_str(), func.name.range);
            }
            Stmt::ClassDef(class_def) => {
                if is_test_class(class_def) {
                    entries.push(CodeLensEntry {
                        range: class_def.name.range,
                        kind: CodeLensKind::Test,
                    });
                }
                collect_class_entries(&class_def.body, entries);
            }
            Stmt::If(stmt_if) => {
                if is_main_guard(&stmt_if.test) {
                    entries.push(CodeLensEntry {
                        range: stmt_if.range,
                        kind: CodeLensKind::Run,
                    });
                }
            }
            _ => {}
        }
    }
}

fn collect_class_entries(stmts: &[Stmt], entries: &mut Vec<CodeLensEntry>) {
    for stmt in stmts {
        if let Stmt::FunctionDef(func) = stmt {
            maybe_push_test(entries, func.name.as_str(), func.name.range);
        }
    }
}

fn maybe_push_test(entries: &mut Vec<CodeLensEntry>, name: &str, range: TextRange) {
    if is_test_name(name) {
        entries.push(CodeLensEntry {
            range,
            kind: CodeLensKind::Test,
        });
    }
}

fn is_test_name(name: &str) -> bool {
    name.starts_with("test_")
}

fn is_test_class(class_def: &StmtClassDef) -> bool {
    if class_def.name.as_str().starts_with("Test") {
        return true;
    }
    class_def.bases().iter().any(is_unittest_base)
}

fn is_unittest_base(base: &Expr) -> bool {
    match base {
        Expr::Name(name) => name.id.as_str().ends_with("TestCase"),
        Expr::Attribute(ExprAttribute { attr, .. }) => attr.id.as_str().ends_with("TestCase"),
        _ => false,
    }
}

fn is_main_guard(test: &Expr) -> bool {
    let Expr::Compare(ExprCompare {
        left,
        ops,
        comparators,
        ..
    }) = test
    else {
        return false;
    };

    if ops.len() != 1 || comparators.len() != 1 {
        return false;
    }

    let op = ops[0];
    if !matches!(op, CmpOp::Eq | CmpOp::Is) {
        return false;
    }

    let left = left.as_ref();
    let right = &comparators[0];
    (is_name_dunder_name(left) && is_main_string(right))
        || (is_main_string(left) && is_name_dunder_name(right))
}

fn is_name_dunder_name(expr: &Expr) -> bool {
    matches!(expr, Expr::Name(name) if name.id.as_str() == "__name__")
}

fn is_main_string(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::StringLiteral(ExprStringLiteral { value, .. }) if value.to_string() == "__main__"
    )
}
