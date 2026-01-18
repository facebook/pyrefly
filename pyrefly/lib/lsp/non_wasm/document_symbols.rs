/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbol;
use pyrefly_build::handle::Handle;
use pyrefly_python::comment_section::CommentSection;
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

        // Extract comment sections
        let sections = CommentSection::extract_from_module(&module_info);

        // Build symbols with comment sections and AST symbols integrated
        build_symbols_with_sections(&ast.body, &sections, &mut result, &module_info);

        Some(result)
    }
}

/// Build document symbols integrating comment sections and AST symbols.
/// AST symbols (functions, classes, variables) are added as children of the
/// comment section that precedes them.
#[allow(deprecated)] // The `deprecated` field
fn build_symbols_with_sections(
    stmts: &[Stmt],
    sections: &[CommentSection],
    result: &mut Vec<DocumentSymbol>,
    module_info: &Module,
) {
    use ruff_text_size::Ranged;

    // Build a hierarchical structure tracking current section context
    // Stack contains (level, path to section in result tree)
    let mut section_stack: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut section_idx = 0;

    for stmt in stmts {
        let stmt_line = module_info.to_lsp_range(stmt.range()).start.line;

        // Process any comment sections that come before this statement
        while section_idx < sections.len() && sections[section_idx].line_number <= stmt_line {
            let section = &sections[section_idx];

            // Pop sections from stack that are at the same or higher level
            while let Some((level, _)) = section_stack.last() {
                if *level >= section.level {
                    section_stack.pop();
                } else {
                    break;
                }
            }

            let symbol = DocumentSymbol {
                name: section.title.clone(),
                detail: None,
                kind: lsp_types::SymbolKind::STRING,
                tags: None,
                deprecated: None,
                range: module_info.to_lsp_range(section.range),
                selection_range: module_info.to_lsp_range(section.range),
                children: Some(Vec::new()),
            };

            if let Some((_, path)) = section_stack.last() {
                // Add as child of parent section
                let current = navigate_to_path_mut(result, path);
                let new_idx = current.len();
                current.push(symbol);

                let mut new_path = path.clone();
                new_path.push(new_idx);
                section_stack.push((section.level, new_path));
            } else {
                // Top-level section
                let new_idx = result.len();
                result.push(symbol);
                section_stack.push((section.level, vec![new_idx]));
            }

            section_idx += 1;
        }

        // Add the AST symbol as a child of the current section (if any)
        if let Some((_, path)) = section_stack.last() {
            // Navigate to the current section and add symbol as its child
            let current = navigate_to_path_mut(result, path);
            recurse_stmt_adding_symbols(stmt, current, module_info);
        } else {
            // No section context, add at top level
            recurse_stmt_adding_symbols(stmt, result, module_info);
        }
    }

    // Process any remaining comment sections at the end of the file
    while section_idx < sections.len() {
        let section = &sections[section_idx];

        while let Some((level, _)) = section_stack.last() {
            if *level >= section.level {
                section_stack.pop();
            } else {
                break;
            }
        }

        let symbol = DocumentSymbol {
            name: section.title.clone(),
            detail: None,
            kind: lsp_types::SymbolKind::STRING,
            tags: None,
            deprecated: None,
            range: module_info.to_lsp_range(section.range),
            selection_range: module_info.to_lsp_range(section.range),
            children: Some(Vec::new()),
        };

        if let Some((_, path)) = section_stack.last() {
            let current = navigate_to_path_mut(result, path);
            let new_idx = current.len();
            current.push(symbol);

            let mut new_path = path.clone();
            new_path.push(new_idx);
            section_stack.push((section.level, new_path));
        } else {
            let new_idx = result.len();
            result.push(symbol);
            section_stack.push((section.level, vec![new_idx]));
        }

        section_idx += 1;
    }
}

/// Navigate to a specific position in the document symbol tree using a path of indices.
fn navigate_to_path_mut<'a>(
    symbols: &'a mut Vec<DocumentSymbol>,
    path: &[usize],
) -> &'a mut Vec<DocumentSymbol> {
    let mut current = symbols;
    for &idx in path {
        current = current[idx].children.as_mut().unwrap();
    }
    current
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

            // Functions defined inside a class are methods.
            for child in &mut children {
                if child.kind == lsp_types::SymbolKind::FUNCTION {
                    child.kind = lsp_types::SymbolKind::METHOD;
                }
            }

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
                    // todo(jvansch): Try to reuse DefinitionMetadata here.
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
