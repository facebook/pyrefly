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
use vec1::Vec1;

use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use crate::state::ide::handle_require_absolute_import;
use crate::state::lsp::LocalRefactorCodeAction;
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

/// Returns actions only when the selection identifies an import and at least one
/// selected or file-wide conversion is applicable.
pub(crate) fn convert_import_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec1<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let import_stmt = find_import_stmt(ast.as_ref(), selection_point)?;

    let mut actions = Vec::new();
    for target in [ConvertTarget::Relative, ConvertTarget::Absolute] {
        actions.extend(build_selected_import_action(
            transaction,
            handle,
            &module_info,
            import_stmt,
            target,
        ));
        actions.extend(build_all_imports_action(
            transaction,
            handle,
            &module_info,
            ast.as_ref(),
            target,
        ));
    }

    actions.sort_by(|a, b| a.title.cmp(&b.title));
    Vec1::try_from_vec(actions).ok()
}

/// Builds an action when the selected statement can be converted to `target`.
fn build_selected_import_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    stmt: ImportStmtRef<'_>,
    target: ConvertTarget,
) -> Option<LocalRefactorCodeAction> {
    let edit = rewrite_import(transaction, handle, module_info, stmt, target)?;
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

/// Builds an action when at least one import in the file can be converted to `target`.
fn build_all_imports_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    ast: &ModModule,
    target: ConvertTarget,
) -> Option<LocalRefactorCodeAction> {
    let mut edits = Vec::new();
    for stmt in collect_import_stmts(ast) {
        if let Some(edit) = rewrite_import(transaction, handle, module_info, stmt, target) {
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

/// Returns a replacement when the statement has a semantics-preserving conversion to `target`.
fn rewrite_import(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    stmt: ImportStmtRef<'_>,
    target: ConvertTarget,
) -> Option<(TextRange, String)> {
    let source = module_info.contents();
    let stmt_range = match stmt {
        ImportStmtRef::Import(import) => import.range(),
        ImportStmtRef::ImportFrom(import_from) => import_from.range(),
    };
    // Import statements cannot contain string literals, so any `#` inside the AST range starts a
    // comment that re-rendering would discard.
    if module_info.code_at(stmt_range).contains('#') {
        return None;
    }
    let new_text = match (stmt, target) {
        (ImportStmtRef::Import(import), ConvertTarget::Relative) => {
            let indent = line_indent_and_start(source, import.range().start())?.0;
            let lines = convert_plain_import_to_relative(transaction, handle, import)?;
            join_import_lines(&indent, lines)
        }
        (ImportStmtRef::ImportFrom(import_from), ConvertTarget::Relative)
            if import_from.level == 0 =>
        {
            let indent = line_indent_and_start(source, import_from.range().start())?.0;
            let line = convert_from_import_to_relative(transaction, handle, import_from)?;
            join_import_lines(&indent, Vec1::new(line))
        }
        (ImportStmtRef::ImportFrom(import_from), ConvertTarget::Absolute)
            if import_from.level > 0 =>
        {
            let indent = line_indent_and_start(source, import_from.range().start())?.0;
            let line = convert_from_import_to_absolute(handle, import_from)?;
            join_import_lines(&indent, Vec1::new(line))
        }
        _ => return None,
    };
    Some((stmt_range, new_text))
}

/// Converts a plain `import` statement into one or more relative `from` import lines.
/// Returns `None` if any alias cannot be resolved or safely represented as a relative import.
fn convert_plain_import_to_relative(
    transaction: &Transaction<'_>,
    handle: &Handle,
    import: &StmtImport,
) -> Option<Vec1<String>> {
    let mut grouped: Vec<(String, Vec<String>)> = Vec::new();
    for alias in &import.names {
        let module_str = alias.name.id.as_str();
        if module_str.contains('.') && alias.asname.is_none() {
            return None;
        }
        let module_name = ModuleName::from_name(&alias.name.id);
        let relative_module = relative_module_for_import(transaction, handle, module_name)?;
        let (base, leaf) = split_relative_module(&relative_module)?;
        let name_text = render_alias(leaf.as_str(), alias);
        push_grouped_import(&mut grouped, base, name_text);
    }
    Some(
        Vec1::try_from_vec(
            grouped
                .into_iter()
                .map(|(base, names)| format!("from {base} import {}", names.join(", ")))
                .collect(),
        )
        .expect("an import statement has at least one alias"),
    )
}

fn convert_from_import_to_relative(
    transaction: &Transaction<'_>,
    handle: &Handle,
    import_from: &StmtImportFrom,
) -> Option<String> {
    let module = import_from.module.as_ref()?;
    let module_name = ModuleName::from_str(module.as_str());
    let relative_module = relative_module_for_import(transaction, handle, module_name)?;
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
    // A top-level `from . import ...` has no absolute package name.
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

fn relative_module_for_import(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_name: ModuleName,
) -> Option<String> {
    let target_handle = transaction
        .import_handle(handle, module_name, None)
        .finding()?;
    if handle_require_absolute_import(transaction.config_finder(), &target_handle) {
        return None;
    }
    Some(
        ModuleName::relative_module_name_between(
            handle.path().as_path(),
            target_handle.path().as_path(),
        )?
        .as_str()
        .to_owned(),
    )
}

fn split_relative_module(relative_module: &str) -> Option<(String, String)> {
    let rest = relative_module.trim_start_matches('.');
    assert!(
        rest.len() < relative_module.len(),
        "relative module name must start with a dot"
    );
    // A plain import of the current package has no equivalent relative-import spelling.
    if rest.is_empty() {
        return None;
    }
    let dots = &relative_module[..relative_module.len() - rest.len()];
    let (package, leaf) = rest.rsplit_once('.').unwrap_or(("", rest));
    Some((format!("{dots}{package}"), leaf.to_owned()))
}

fn push_grouped_import(grouped: &mut Vec<(String, Vec<String>)>, base: String, name: String) {
    if let Some((_, names)) = grouped.iter_mut().find(|(key, _)| *key == base) {
        names.push(name);
    } else {
        grouped.push((base, vec![name]));
    }
}

fn join_import_lines(indent: &str, lines: Vec1<String>) -> String {
    let (first, remaining) = lines.split_off_first();
    if indent.is_empty() {
        return remaining.into_iter().fold(first, |mut acc, line| {
            acc.push('\n');
            acc.push_str(&line);
            acc
        });
    }
    remaining.into_iter().fold(first, |mut acc, line| {
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
