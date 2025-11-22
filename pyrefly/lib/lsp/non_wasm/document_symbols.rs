/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbol;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;

use crate::state::state::Transaction;

impl<'a> Transaction<'a> {
    #[allow(deprecated)] // The `deprecated` field
    pub fn symbols(&self, handle: &Handle) -> Option<Vec<DocumentSymbol>> {
        let ast = self.get_ast(handle)?;
        let module_info = self.get_module_info(handle)?;

        let mut result = Vec::new();
        ast.body
            .visit(&mut |stmt| recurse_stmt_adding_symbols(stmt, &mut result, &module_info));
        Some(result)
    }
}

#[allow(deprecated)] // The `deprecated` field
fn recurse_stmt_adding_symbols<'a>(
    stmt: &'a Stmt,
    symbols: &'a mut Vec<DocumentSymbol>,
    module_info: &Module,
) {
    let mut recursed_symbols = Vec::new();
    stmt.recurse(&mut |stmt| recurse_stmt_adding_symbols(stmt, &mut recursed_symbols, module_info));

    match stmt {
        Stmt::FunctionDef(stmt_function_def) => {
            let mut children = Vec::new();
            children.append(&mut recursed_symbols);
            // todo(kylei): better approach to filtering out "" for all symbols
            let name = match stmt_function_def.name.as_str() {
                "" => "unknown".to_owned(),
                name => name.to_owned(),
            };
            symbols.push(DocumentSymbol {
                name,
                detail: None,
                kind: lsp_types::SymbolKind::FUNCTION,
                tags: None,
                deprecated: None,
                range: module_info.to_lsp_range(stmt_function_def.range),
                selection_range: module_info.to_lsp_range(stmt_function_def.name.range),

                children: Some(children),
            });
        }
        Stmt::ClassDef(stmt_class_def) => {
            let mut children = Vec::new();
            children.append(&mut recursed_symbols);
            let name = match stmt_class_def.name.as_str() {
                "" => "unknown".to_owned(),
                name => name.to_owned(),
            };
            symbols.push(DocumentSymbol {
                name,
                detail: None,
                kind: lsp_types::SymbolKind::CLASS,
                tags: None,
                deprecated: None,
                range: module_info.to_lsp_range(stmt_class_def.range),
                selection_range: module_info.to_lsp_range(stmt_class_def.name.range),
                children: Some(children),
            });
        }
        Stmt::Assign(stmt_assign) => {
            for target in &stmt_assign.targets {
                if let Expr::Name(name) = target {
                    // todo(jvansch): Try to resuse DefinitionMetadata here.
                    symbols.push(DocumentSymbol {
                        name: name.id.to_string(),
                        detail: None, // Todo(jvansch): Could add type info here later
                        kind: lsp_types::SymbolKind::VARIABLE,
                        tags: None,
                        deprecated: None,
                        range: module_info.to_lsp_range(stmt_assign.range),
                        selection_range: module_info.to_lsp_range(name.range),
                        children: None,
                    });
                }
            }
        }
        Stmt::AnnAssign(stmt_ann_assign) => {
            if let Expr::Name(name) = &*stmt_ann_assign.target {
                symbols.push(DocumentSymbol {
                    name: name.id.to_string(),
                    detail: Some(
                        module_info
                            .code_at(stmt_ann_assign.annotation.range())
                            .to_owned(),
                    ),
                    kind: lsp_types::SymbolKind::VARIABLE,
                    tags: None,
                    deprecated: None,
                    range: module_info.to_lsp_range(stmt_ann_assign.range),
                    selection_range: module_info.to_lsp_range(name.range),
                    children: None,
                });
            }
        }
        _ => {}
    };
    symbols.append(&mut recursed_symbols);
}
