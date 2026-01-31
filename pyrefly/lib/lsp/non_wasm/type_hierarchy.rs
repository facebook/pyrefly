/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;
use lsp_types::TypeHierarchyItem;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

/// Finds a class definition at a specific position in an AST.
pub fn find_class_at_position_in_ast(ast: &ModModule, position: TextSize) -> Option<&StmtClassDef> {
    let covering_nodes = Ast::locate_node(ast, position);
    for node in covering_nodes.iter() {
        if let AnyNodeRef::StmtClassDef(class_def) = node {
            return Some(class_def);
        }
    }
    None
}

/// Creates the LSP TypeHierarchyItem representation for a class definition.
pub fn prepare_type_hierarchy_item(
    class_def: &StmtClassDef,
    module: &Module,
    uri: lsp_types::Url,
) -> TypeHierarchyItem {
    TypeHierarchyItem {
        name: class_def.name.id.to_string(),
        kind: SymbolKind::CLASS,
        tags: None,
        detail: Some(format!("{}.{}", module.name(), class_def.name.id)),
        uri,
        range: module.to_lsp_range(class_def.range()),
        selection_range: module.to_lsp_range(class_def.name.range),
        data: None,
    }
}

/// Collect all class definitions from a statement body, including nested classes.
pub fn collect_class_defs<'a>(body: &'a [Stmt], out: &mut Vec<&'a StmtClassDef>) {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt {
            out.push(class_def);
            collect_class_defs(class_def.body.as_slice(), out);
        }
    }
}
