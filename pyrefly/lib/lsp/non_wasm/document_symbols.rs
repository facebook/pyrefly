/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentSymbol;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::comment_section::CommentSection;
use pyrefly_python::module::Module;
use pyrefly_python::symbol_kind::SymbolKind as PyreflySymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;

use crate::export::symbols::ScopeKind;
use crate::export::symbols::assignment_kind;
use crate::state::state::Transaction;

impl<'a> Transaction<'a> {
    /// Return document symbols for the file behind `handle`.
    /// When `limit_cell_idx` is `Some`, only symbols whose range falls within that
    /// notebook cell are returned (mirroring semantic-token cell filtering).
    #[allow(deprecated)] // The `deprecated` field
    pub fn symbols(
        &self,
        handle: &Handle,
        limit_cell_idx: Option<usize>,
    ) -> Option<Vec<DocumentSymbol>> {
        let ast = self.get_ast(handle)?;
        let module_info = self.get_module_info(handle)?;

        let mut result = Vec::new();

        // Extract comment sections (only relevant for non-notebook files)
        let sections = if limit_cell_idx.is_none() {
            CommentSection::extract_from_module(&module_info)
        } else {
            Vec::new()
        };

        // Build symbols with comment sections and AST symbols integrated
        build_symbols_with_sections(
            &ast.body,
            &sections,
            &mut result,
            &module_info,
            limit_cell_idx,
        );

        Some(result)
    }
}

/// Build document symbols integrating comment sections and AST symbols.
/// AST symbols (functions, classes, variables) are added as children of the
/// comment section that precedes them.
/// When `limit_cell_idx` is `Some`, only top-level statements belonging to that
/// notebook cell are included.
#[allow(deprecated)] // The `deprecated` field
fn build_symbols_with_sections(
    stmts: &[Stmt],
    sections: &[CommentSection],
    result: &mut Vec<DocumentSymbol>,
    module_info: &Module,
    limit_cell_idx: Option<usize>,
) {
    use ruff_text_size::Ranged;

    // Build a hierarchical structure tracking current section context
    // Stack contains (level, path to section in result tree)
    let mut section_stack: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut section_idx = 0;

    for stmt in stmts {
        // Skip statements that belong to a different notebook cell
        if limit_cell_idx.is_some()
            && module_info.to_cell_for_lsp(stmt.range().start()) != limit_cell_idx
        {
            continue;
        }
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
            recurse_stmt_adding_symbols(stmt, current, module_info, ScopeKind::Module);
        } else {
            // No section context, add at top level
            recurse_stmt_adding_symbols(stmt, result, module_info, ScopeKind::Module);
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
fn recurse_stmt_adding_symbols(
    stmt: &Stmt,
    symbols: &mut Vec<DocumentSymbol>,
    module_info: &Module,
    scope: ScopeKind,
) {
    let nested_scope = match stmt {
        Stmt::FunctionDef(_) => ScopeKind::Function,
        Stmt::ClassDef(_) => ScopeKind::Class,
        _ => scope,
    };
    let mut recursed_symbols = Vec::new();
    stmt.recurse(&mut |stmt| {
        recurse_stmt_adding_symbols(stmt, &mut recursed_symbols, module_info, nested_scope)
    });

    match stmt {
        Stmt::FunctionDef(stmt_function_def) => {
            let name = if Ast::is_synthesized_empty_identifier(&stmt_function_def.name) {
                "unknown".to_owned()
            } else {
                stmt_function_def.name.to_string()
            };
            symbols.push(DocumentSymbol {
                name,
                detail: None,
                kind: if scope == ScopeKind::Class {
                    PyreflySymbolKind::Method
                } else {
                    PyreflySymbolKind::Function
                }
                .to_lsp_symbol_kind(),
                tags: None,
                deprecated: None,
                range: module_info.to_lsp_range(stmt_function_def.range),
                selection_range: module_info.to_lsp_range(stmt_function_def.name.range),
                children: Some(recursed_symbols),
            });
        }
        Stmt::ClassDef(stmt_class_def) => {
            let name = if Ast::is_synthesized_empty_identifier(&stmt_class_def.name) {
                "unknown".to_owned()
            } else {
                stmt_class_def.name.to_string()
            };
            symbols.push(DocumentSymbol {
                name,
                detail: None,
                kind: PyreflySymbolKind::Class.to_lsp_symbol_kind(),
                tags: None,
                deprecated: None,
                range: module_info.to_lsp_range(stmt_class_def.range),
                selection_range: module_info.to_lsp_range(stmt_class_def.name.range),
                children: Some(recursed_symbols),
            });
        }
        Stmt::Assign(stmt_assign) => {
            for target in &stmt_assign.targets {
                Ast::expr_lvalue(target, &mut |name| {
                    symbols.push(DocumentSymbol {
                        name: name.id.to_string(),
                        detail: None,
                        kind: assignment_kind(&name.id, scope).to_lsp_symbol_kind(),
                        tags: None,
                        deprecated: None,
                        range: module_info.to_lsp_range(stmt_assign.range),
                        selection_range: module_info.to_lsp_range(name.range),
                        children: None,
                    });
                });
            }
            symbols.append(&mut recursed_symbols);
        }
        Stmt::AnnAssign(stmt_ann_assign) => {
            if let Expr::Name(name) = &*stmt_ann_assign.target
                && !Ast::is_synthesized_empty_name(name)
            {
                symbols.push(DocumentSymbol {
                    name: name.id.to_string(),
                    detail: Some(
                        module_info
                            .code_at(stmt_ann_assign.annotation.range())
                            .to_owned(),
                    ),
                    kind: assignment_kind(&name.id, scope).to_lsp_symbol_kind(),
                    tags: None,
                    deprecated: None,
                    range: module_info.to_lsp_range(stmt_ann_assign.range),
                    selection_range: module_info.to_lsp_range(name.range),
                    children: None,
                });
            }
            symbols.append(&mut recursed_symbols);
        }
        Stmt::TypeAlias(stmt_type_alias) => {
            if let Expr::Name(name) = &*stmt_type_alias.name
                && !Ast::is_synthesized_empty_name(name)
            {
                symbols.push(DocumentSymbol {
                    name: name.id.to_string(),
                    detail: None,
                    kind: PyreflySymbolKind::TypeAlias.to_lsp_symbol_kind(),
                    tags: None,
                    deprecated: None,
                    range: module_info.to_lsp_range(stmt_type_alias.range),
                    selection_range: module_info.to_lsp_range(name.range),
                    children: None,
                });
            }
            symbols.append(&mut recursed_symbols);
        }
        _ => symbols.append(&mut recursed_symbols),
    }
}

pub fn flatten_to_symbol_information(
    symbols: Vec<DocumentSymbol>,
    uri: &lsp_types::Url,
) -> Vec<lsp_types::SymbolInformation> {
    let mut results = Vec::new();
    flatten_recursive(symbols, uri, None, &mut results);
    results
}

fn flatten_recursive(
    symbols: Vec<DocumentSymbol>,
    uri: &lsp_types::Url,
    container_name: Option<String>,
    result: &mut Vec<lsp_types::SymbolInformation>,
) {
    for sym in symbols {
        let children = sym.children.unwrap_or_default();
        let qualified_name = match &container_name {
            Some(parent) => format!("{}.{}", parent, sym.name),
            None => sym.name.clone(),
        };

        #[allow(deprecated)]
        result.push(lsp_types::SymbolInformation {
            name: sym.name,
            kind: sym.kind,
            tags: sym.tags,
            deprecated: sym.deprecated,
            location: lsp_types::Location {
                uri: uri.clone(),
                range: sym.range,
            },
            container_name: container_name.clone(),
        });

        flatten_recursive(children, uri, Some(qualified_name), result);
    }
}
