/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;

use crate::report::pysa::location::PysaLocation;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ScopeParent {
    Function { location: PysaLocation },
    Class { location: PysaLocation },
    TopLevel,
}

pub fn get_scope_parent(ast: &ModModule, module_info: &Module, range: TextRange) -> ScopeParent {
    Ast::locate_node(ast, range.start())
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::Identifier(id) if id.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) if class_def.name.range() == range => None,
            AnyNodeRef::StmtFunctionDef(fun_def) if fun_def.name.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) => Some(ScopeParent::Class {
                location: PysaLocation::new(module_info.display_range(class_def.name.range())),
            }),
            AnyNodeRef::StmtFunctionDef(fun_def) => Some(ScopeParent::Function {
                location: PysaLocation::new(module_info.display_range(fun_def.name.range())),
            }),
            _ => None,
        })
        .unwrap_or(ScopeParent::TopLevel)
}
