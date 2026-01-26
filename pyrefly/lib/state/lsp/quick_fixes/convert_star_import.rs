/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use crate::binding::binding::Key;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

fn rewrite_kind() -> CodeActionKind {
    CodeActionKind::new("refactor.rewrite")
}

/// Builds convert-star-import refactor actions for the supplied selection.
pub(crate) fn convert_star_import_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let (import_from, star_range) = find_star_import(ast.as_ref(), selection_point)?;
    let module_name = resolve_import_module_name(&module_info, import_from)?;
    let bindings = transaction.get_bindings(handle)?;

    let names = collect_star_imported_names(ast.as_ref(), &bindings, module_name, star_range);
    if names.is_empty() {
        return None;
    }

    let (indent, line_start) = line_indent_and_start(source, import_from.range().start())?;
    let line_end = line_end_position(source, import_from.range().end());
    let line_range = TextRange::new(line_start, line_end);
    let line_text =
        &source[line_start.to_usize().min(source.len())..line_end.to_usize().min(source.len())];
    let comment = trailing_comment(line_text);

    let from_module = import_from_module_text(import_from);
    let import_list = names.join(", ");
    let mut replacement = format!("{indent}from {from_module} import {import_list}");
    if let Some(comment) = comment {
        replacement.push(' ');
        replacement.push_str(comment.trim_start());
    }
    replacement.push('\n');

    Some(vec![LocalRefactorCodeAction {
        title: format!(
            "Convert to explicit imports from `{}`",
            module_name.as_str()
        ),
        edits: vec![(module_info.dupe(), line_range, replacement)],
        kind: rewrite_kind(),
    }])
}

fn find_star_import<'a>(
    ast: &'a ModModule,
    selection: TextSize,
) -> Option<(&'a StmtImportFrom, TextRange)> {
    ast.body.iter().find_map(|stmt| match stmt {
        Stmt::ImportFrom(import_from) if import_from.range().contains(selection) => {
            let star = import_from.names.iter().find(|alias| &alias.name == "*")?;
            Some((import_from, star.range))
        }
        _ => None,
    })
}

fn resolve_import_module_name(
    module_info: &pyrefly_python::module::Module,
    import_from: &StmtImportFrom,
) -> Option<ModuleName> {
    module_info.name().new_maybe_relative(
        module_info.path().is_init(),
        import_from.level,
        import_from.module.as_ref().map(|module| &module.id),
    )
}

fn import_from_module_text(import_from: &StmtImportFrom) -> String {
    let mut module_text = ".".repeat(import_from.level as usize);
    if let Some(module) = &import_from.module {
        module_text.push_str(module.id.as_str());
    }
    module_text
}

fn collect_star_imported_names(
    ast: &ModModule,
    bindings: &crate::binding::bindings::Bindings,
    module_name: ModuleName,
    star_range: TextRange,
) -> Vec<String> {
    struct NameCollector<'a> {
        bindings: &'a crate::binding::bindings::Bindings,
        module_name: ModuleName,
        star_range: TextRange,
        names: BTreeSet<String>,
    }

    impl<'a> Visitor<'a> for NameCollector<'a> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(name) = expr
                && matches!(name.ctx, ExprContext::Load | ExprContext::Del)
            {
                let key = Key::BoundName(ShortIdentifier::expr_name(name));
                if self.bindings.is_valid_key(&key)
                    && let Some(intermediate) = key_to_intermediate_definition(self.bindings, &key)
                    && let IntermediateDefinition::NamedImport(
                        import_range,
                        import_module,
                        import_name,
                        _,
                    ) = intermediate
                    && import_range == self.star_range
                    && import_module == self.module_name
                {
                    self.names.insert(import_name.as_str().to_owned());
                }
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let mut collector = NameCollector {
        bindings,
        module_name,
        star_range,
        names: BTreeSet::new(),
    };
    collector.visit_body(&ast.body);
    collector.names.into_iter().collect()
}

fn trailing_comment(line: &str) -> Option<&str> {
    let trimmed = line.strip_suffix("\n").unwrap_or(line);
    let trimmed = trimmed.strip_suffix("\r").unwrap_or(trimmed);
    trimmed.find('#').map(|idx| &trimmed[idx..])
}

fn line_end_position(source: &str, position: TextSize) -> TextSize {
    let idx = position.to_usize().min(source.len());
    if let Some(offset) = source[idx..].find('\n') {
        TextSize::try_from(idx + offset + 1).unwrap_or(position)
    } else {
        TextSize::try_from(source.len()).unwrap_or(position)
    }
}
