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

#[derive(Clone, Debug)]
pub struct CodeLensEntry {
    pub range: TextRange,
    pub kind: CodeLensKind,
    pub test_name: Option<String>,
    pub class_name: Option<String>,
    pub is_unittest: bool,
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
                maybe_push_test(entries, func.name.as_str(), func.name.range, None, false);
            }
            Stmt::ClassDef(class_def) => {
                let is_unittest = is_unittest_class(class_def);
                if is_test_class(class_def, is_unittest) {
                    entries.push(CodeLensEntry {
                        range: class_def.name.range,
                        kind: CodeLensKind::Test,
                        test_name: None,
                        class_name: Some(class_def.name.as_str().to_owned()),
                        is_unittest,
                    });
                }
                collect_class_entries(
                    &class_def.body,
                    entries,
                    class_def.name.as_str(),
                    is_unittest,
                );
            }
            Stmt::If(stmt_if) => {
                if is_main_guard(&stmt_if.test) {
                    entries.push(CodeLensEntry {
                        range: stmt_if.range,
                        kind: CodeLensKind::Run,
                        test_name: None,
                        class_name: None,
                        is_unittest: false,
                    });
                }
            }
            _ => {}
        }
    }
}

fn collect_class_entries(
    stmts: &[Stmt],
    entries: &mut Vec<CodeLensEntry>,
    class_name: &str,
    is_unittest: bool,
) {
    for stmt in stmts {
        if let Stmt::FunctionDef(func) = stmt {
            maybe_push_test(
                entries,
                func.name.as_str(),
                func.name.range,
                Some(class_name.to_owned()),
                is_unittest,
            );
        }
    }
}

fn maybe_push_test(
    entries: &mut Vec<CodeLensEntry>,
    name: &str,
    range: TextRange,
    class_name: Option<String>,
    is_unittest: bool,
) {
    if is_test_name(name) {
        entries.push(CodeLensEntry {
            range,
            kind: CodeLensKind::Test,
            test_name: Some(name.to_owned()),
            class_name,
            is_unittest,
        });
    }
}

fn is_test_name(name: &str) -> bool {
    name.starts_with("test_")
}

fn is_test_class(class_def: &StmtClassDef, is_unittest: bool) -> bool {
    if class_def.name.as_str().starts_with("Test") {
        return true;
    }
    is_unittest
}

fn is_unittest_class(class_def: &StmtClassDef) -> bool {
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
