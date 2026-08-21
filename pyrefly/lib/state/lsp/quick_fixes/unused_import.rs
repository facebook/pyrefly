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
use ruff_text_size::TextSize;

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
        return statement_removal_range(source, stmt_range);
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

/// Range covering an import statement that is being removed in its entirety.
///
/// A statement usually owns its whole line, and taking the line avoids leaving a
/// blank one behind. A `;` separator means the line also holds unrelated
/// statements that must survive, so in that case the range covers the import and
/// the one separator that joins it to its neighbours, keeping what is left a
/// valid statement list. A trailing comment describes the import and is removed
/// along with it.
fn statement_removal_range(source: &str, stmt_range: TextRange) -> Option<TextRange> {
    let (_, line_start) = line_indent_and_start(source, stmt_range.start())?;
    let line_end = line_end_position(source, stmt_range.end());
    // `line_indent_and_start` and `line_end_position` return offsets of `\n`
    // boundaries clamped to the source, so these slices are always in range.
    let before = &source[line_start.to_usize()..stmt_range.start().to_usize()];
    let after = &source[stmt_range.end().to_usize()..line_end.to_usize()];
    // Only a `;` outside a comment separates statements.
    let after_code = after.find('#').map_or(after, |hash| &after[..hash]);

    if before.trim().is_empty() && !after_code.contains(';') {
        return Some(TextRange::new(line_start, line_end));
    }

    let (start, end) = match after_code.find(';') {
        Some(semicolon) => {
            let after_semicolon = stmt_range.end().to_usize() + semicolon + 1;
            let rest = source[after_semicolon..line_end.to_usize()].trim_start_matches([' ', '\t']);
            (
                stmt_range.start().to_usize(),
                line_end.to_usize() - rest.len(),
            )
        }
        // A statement preceded on its line by something other than a `;` is the
        // body of a one-line compound statement, which cannot be removed on its
        // own.
        None => (
            line_start.to_usize() + before.rfind(';')?,
            stmt_range.end().to_usize(),
        ),
    };
    Some(TextRange::new(
        TextSize::try_from(start).ok()?,
        TextSize::try_from(end).ok()?,
    ))
}
