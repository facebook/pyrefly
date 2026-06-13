/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::Alias;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use crate::ModuleInfo;
use crate::binding::scope::UnusedImport;

pub(crate) fn remove_unused_import_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    unused: &UnusedImport,
) -> Option<(String, Module, TextRange, String)> {
    let range = import_removal_range(module_info.contents(), ast, unused)?;
    Some((
        format!("Remove unused import: `{}`", unused.name.as_str()),
        module_info.dupe(),
        range,
        String::new(),
    ))
}

fn import_removal_range(source: &str, ast: &ModModule, unused: &UnusedImport) -> Option<TextRange> {
    for stmt in &ast.body {
        match stmt {
            Stmt::Import(import) if import.range().contains_range(unused.range) => {
                return alias_removal_range(source, import.range(), &import.names, unused.range);
            }
            Stmt::ImportFrom(import_from) if import_from.range().contains_range(unused.range) => {
                return alias_removal_range(
                    source,
                    import_from.range(),
                    &import_from.names,
                    unused.range,
                );
            }
            _ => {}
        }
    }
    None
}

fn alias_removal_range(
    source: &str,
    stmt_range: TextRange,
    aliases: &[Alias],
    unused_range: TextRange,
) -> Option<TextRange> {
    let index = aliases
        .iter()
        .position(|alias| alias.range().contains_range(unused_range))?;
    if aliases.len() == 1 {
        let (_, line_start) = line_indent_and_start(source, stmt_range.start())?;
        return Some(TextRange::new(
            line_start,
            line_end_position(source, stmt_range.end()),
        ));
    }
    if index == 0 {
        Some(TextRange::new(
            aliases[index].range().start(),
            aliases[index + 1].range().start(),
        ))
    } else {
        Some(TextRange::new(
            aliases[index - 1].range().end(),
            aliases[index].range().end(),
        ))
    }
}
