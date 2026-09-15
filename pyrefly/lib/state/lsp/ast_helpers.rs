/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::TextSize;

/// A function definition plus its enclosing class (if any).
pub(crate) struct ContainingFunction<'a> {
    pub function_def: &'a StmtFunctionDef,
    pub class_def: Option<&'a StmtClassDef>,
}

/// Finds the innermost function definition covering `position`.
///
/// Returns the function and its enclosing class when the function is a method.
pub(crate) fn find_containing_function_def(
    ast: &ModModule,
    position: TextSize,
) -> Option<ContainingFunction<'_>> {
    let covering_nodes = Ast::locate_node(ast, position);
    for (idx, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(function_def) = node {
            let class_def = match covering_nodes.get(idx + 1) {
                Some(AnyNodeRef::StmtClassDef(class_def)) => Some(*class_def),
                _ => None,
            };
            return Some(ContainingFunction {
                function_def,
                class_def,
            });
        }
    }
    None
}
