/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use ruff_python_ast::Alias;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_function::LocalRefactorCodeAction;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use crate::state::ide::handle_require_absolute_import;
use crate::state::lsp::Transaction;

#[derive(Clone, Copy, Debug)]
enum ConvertTarget {
    Relative,
    Absolute,
}

#[derive(Clone, Copy, Debug)]
enum ImportStmtRef<'a> {
    Import(&'a StmtImport),
    ImportFrom(&'a StmtImportFrom),
}

pub(crate) fn convert_import_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let import_stmt = find_import_stmt(ast.as_ref(), selection_point)?;

    let mut actions = Vec::new();
    if let Some(action) = convert_single_import_action(
        transaction,
        handle,
        &module_info,
        source,
        import_stmt,
        ConvertTarget::Relative,
    ) {
        actions.push(action);
    }
    if let Some(action) = convert_single_import_action(
        transaction,
        handle,
        &module_info,
        source,
        import_stmt,
        ConvertTarget::Absolute,
    ) {
        actions.push(action);
    }
    if let Some(action) = convert_all_imports_action(
        transaction,
        handle,
        &module_info,
        source,
        ast.as_ref(),
        ConvertTarget::Relative,
    ) {
        actions.push(action);
    }
    if let Some(action) = convert_all_imports_action(
        transaction,
        handle,
        &module_info,
        source,
        ast.as_ref(),
        ConvertTarget::Absolute,
    ) {
        actions.push(action);
    }

    if actions.is_empty() {
        None
    } else {
        actions.sort_by(|a, b| a.title.cmp(&b.title));
        Some(actions)
    }
}

fn convert_single_import_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    source: &str,
    stmt: ImportStmtRef<'_>,
    target: ConvertTarget,
) -> Option<LocalRefactorCodeAction> {
    let edit = convert_import_stmt(transaction, handle, source, stmt, target)?;
    let title = match target {
        ConvertTarget::Relative => "Convert import to relative path",
        ConvertTarget::Absolute => "Convert import to absolute path",
    };
    Some(LocalRefactorCodeAction {
        title: title.to_owned(),
        edits: vec![(module_info.dupe(), edit.0, edit.1)],
        kind: CodeActionKind::REFACTOR_REWRITE,
    })
}

fn convert_all_imports_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    source: &str,
    ast: &ModModule,
    target: ConvertTarget,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    for stmt in collect_import_stmts(ast) {
        if let Some(edit) = convert_import_stmt(transaction, handle, source, stmt, target) {
            edits.push((module_info.dupe(), edit.0, edit.1));
        }
    }
    if edits.is_empty() {
        return None;
    }
    let title = match target {
        ConvertTarget::Relative => "Convert all imports to relative path",
        ConvertTarget::Absolute => "Convert all imports to absolute path",
    };
    Some(LocalRefactorCodeAction {
        title: title.to_owned(),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    })
}

fn convert_import_stmt(
    transaction: &Transaction<'_>,
    handle: &Handle,
    source: &str,
    stmt: ImportStmtRef<'_>,
    target: ConvertTarget,
) -> Option<(TextRange, String)> {
    let (stmt_range, new_text) = match (stmt, target) {
        (ImportStmtRef::Import(import), ConvertTarget::Relative) => {
            let stmt_range = import.range();
            let indent = line_indent_and_start(source, import.range().start())?.0;
            let lines = convert_import_to_relative(transaction, handle, import)?;
            let new_text = join_import_lines(&indent, lines);
            (stmt_range, new_text)
        }
        (ImportStmtRef::ImportFrom(import_from), ConvertTarget::Relative)
            if import_from.level == 0 =>
        {
            let stmt_range = import_from.range();
            let indent = line_indent_and_start(source, import_from.range().start())?.0;
            let line = convert_from_import_to_relative(transaction, handle, import_from)?;
            let new_text = join_import_lines(&indent, vec![line]);
            (stmt_range, new_text)
        }
        (ImportStmtRef::ImportFrom(import_from), ConvertTarget::Absolute)
            if import_from.level > 0 =>
        {
            let stmt_range = import_from.range();
            let indent = line_indent_and_start(source, import_from.range().start())?.0;
            let line = convert_from_import_to_absolute(handle, import_from)?;
            let new_text = join_import_lines(&indent, vec![line]);
            (stmt_range, new_text)
        }
        _ => return None,
    };
    Some((stmt_range, new_text))
}

fn convert_import_to_relative(
    transaction: &Transaction<'_>,
    handle: &Handle,
    import: &StmtImport,
) -> Option<Vec<String>> {
    let mut grouped: Vec<(String, Vec<String>)> = Vec::new();
    for alias in &import.names {
        let module_str = alias.name.id.as_str();
        if module_str.contains('.') && alias.asname.is_none() {
            return None;
        }
        let module_name = ModuleName::from_name(&alias.name.id);
        let target_handle = transaction
            .import_handle(handle, module_name, None)
            .finding()?;
        if handle_require_absolute_import(transaction.config_finder(), &target_handle) {
            return None;
        }
        let relative_module = relative_module_string(handle, &target_handle)?;
        let (base, leaf) = split_relative_module(&relative_module, module_str)?;
        let name_text = render_alias(leaf.as_str(), alias);
        push_grouped_import(&mut grouped, base, name_text);
    }
    if grouped.is_empty() {
        return None;
    }
    Some(
        grouped
            .into_iter()
            .map(|(base, names)| format!("from {base} import {}", names.join(", ")))
            .collect(),
    )
}

fn convert_from_import_to_relative(
    transaction: &Transaction<'_>,
    handle: &Handle,
    import_from: &StmtImportFrom,
) -> Option<String> {
    let module = import_from.module.as_ref()?;
    let module_name = ModuleName::from_str(module.as_str());
    let target_handle = transaction
        .import_handle(handle, module_name, None)
        .finding()?;
    if handle_require_absolute_import(transaction.config_finder(), &target_handle) {
        return None;
    }
    let relative_module = relative_module_string(handle, &target_handle)?;
    let names = render_imported_names(&import_from.names);
    Some(format!("from {relative_module} import {names}"))
}

fn convert_from_import_to_absolute(
    handle: &Handle,
    import_from: &StmtImportFrom,
) -> Option<String> {
    let module = handle.module().new_maybe_relative(
        handle.path().is_init(),
        import_from.level,
        import_from.module.as_ref().map(|module| &module.id),
    )?;
    let module_str = module.as_str();
    if module_str.is_empty() {
        return None;
    }
    let names = render_imported_names(&import_from.names);
    Some(format!("from {module_str} import {names}"))
}

fn render_imported_names(names: &[Alias]) -> String {
    names
        .iter()
        .map(|alias| render_alias(alias.name.id.as_str(), alias))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_alias(default_name: &str, alias: &Alias) -> String {
    if let Some(asname) = &alias.asname {
        format!("{default_name} as {}", asname.id.as_str())
    } else {
        default_name.to_owned()
    }
}

fn relative_module_string(from: &Handle, to: &Handle) -> Option<String> {
    let module =
        ModuleName::relative_module_name_between(from.path().as_path(), to.path().as_path())?;
    let raw = module.as_str();
    if raw.is_empty() {
        Some(".".to_owned())
    } else {
        Some(raw.to_owned())
    }
}

fn split_relative_module(relative_module: &str, absolute_module: &str) -> Option<(String, String)> {
    if relative_module == "." {
        let leaf = absolute_module
            .rsplit_once('.')
            .map_or(absolute_module, |(_, leaf)| leaf);
        return Some((".".to_owned(), leaf.to_owned()));
    }
    if let Some((base, leaf)) = relative_module.rsplit_once('.') {
        let base = if base.is_empty() { "." } else { base };
        return Some((base.to_owned(), leaf.to_owned()));
    }
    Some((".".to_owned(), relative_module.to_owned()))
}

fn push_grouped_import(grouped: &mut Vec<(String, Vec<String>)>, base: String, name: String) {
    if let Some((_, names)) = grouped.iter_mut().find(|(key, _)| *key == base) {
        names.push(name);
    } else {
        grouped.push((base, vec![name]));
    }
}

fn join_import_lines(indent: &str, lines: Vec<String>) -> String {
    let mut iter = lines.into_iter();
    let Some(first) = iter.next() else {
        return String::new();
    };
    if indent.is_empty() {
        return iter.fold(first, |mut acc, line| {
            acc.push('\n');
            acc.push_str(&line);
            acc
        });
    }
    iter.fold(first, |mut acc, line| {
        acc.push('\n');
        acc.push_str(indent);
        acc.push_str(&line);
        acc
    })
}

fn find_import_stmt(ast: &ModModule, position: TextSize) -> Option<ImportStmtRef<'_>> {
    for node in Ast::locate_node(ast, position) {
        match node {
            AnyNodeRef::StmtImport(import) => return Some(ImportStmtRef::Import(import)),
            AnyNodeRef::StmtImportFrom(import_from) => {
                return Some(ImportStmtRef::ImportFrom(import_from));
            }
            _ => {}
        }
    }
    None
}

fn collect_import_stmts(ast: &ModModule) -> Vec<ImportStmtRef<'_>> {
    struct ImportCollector<'a> {
        imports: Vec<ImportStmtRef<'a>>,
    }

    impl<'a> Visitor<'a> for ImportCollector<'a> {
        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            match stmt {
                Stmt::Import(import) => self.imports.push(ImportStmtRef::Import(import)),
                Stmt::ImportFrom(import_from) => {
                    self.imports.push(ImportStmtRef::ImportFrom(import_from));
                }
                _ => {}
            }
            ruff_python_ast::visitor::walk_stmt(self, stmt);
        }
    }

    let mut collector = ImportCollector {
        imports: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.imports
}
