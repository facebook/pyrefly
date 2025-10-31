/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;

use crate::alt::types::class_metadata::ClassMro;
use crate::report::pysa::class::get_all_classes;
use crate::report::pysa::class::get_class_mro;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::get_all_decorated_functions;

fn is_unittest_module(context: &ModuleContext) -> bool {
    get_all_classes(context).any(|class| match &*get_class_mro(&class, context) {
        ClassMro::Resolved(mro) => mro
            .iter()
            .any(|base| base.has_qname("unittest.case", "TestCase")),
        ClassMro::Cyclic => false,
    })
}

fn is_pytest_module(context: &ModuleContext) -> bool {
    fn has_pytest_prefix(name: &Name) -> bool {
        name == "pytest" || name.starts_with("pytest.")
    }
    fn imports_pytest(ast: &ModModule) -> bool {
        ast.body.iter().any(|stmt| match stmt {
            Stmt::Import(import_stmt) => import_stmt
                .names
                .iter()
                .any(|alias| has_pytest_prefix(&alias.name.id)),
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(module),
                ..
            }) => has_pytest_prefix(&module.id),
            _ => false,
        })
    }
    fn has_test_function(context: &ModuleContext) -> bool {
        get_all_decorated_functions(context).any(|function| {
            function
                .metadata()
                .kind
                .function_name()
                .as_ref()
                .starts_with("test_")
        })
    }
    imports_pytest(&context.ast) && has_test_function(context)
}

/// Returns true if the module is considered a test module for Pysa.
/// In that case, we won't analyze the module at all.
///
/// We currently use the following heuristics:
/// - If a class inherits from `unittest.TestCase`, we assume this is a test file.
/// - If `pytest` is imported and at least one function starts with `test_`, we assume this is a test file.
pub fn is_test_module(context: &ModuleContext) -> bool {
    is_unittest_module(context) || is_pytest_module(context)
}
