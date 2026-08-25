/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::Alias;
use ruff_python_ast::ExceptHandler;
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
    let (range, replacement) = import_removal_edit(module_info.contents(), ast, unused)?;
    Some((
        format!("Remove unused import: `{}`", unused.name.as_str()),
        module_info.dupe(),
        range,
        replacement,
    ))
}

/// A block enclosing the import, and the statement that owns the block. The
/// module body owns itself, so it is not stored: running off the end of the
/// chain is what reaching the module looks like.
struct EnclosingBlock<'a> {
    owner: &'a Stmt,
    body: &'a [Stmt],
}

struct FoundImport<'a> {
    stmt_range: TextRange,
    aliases: &'a [Alias],
    /// The blocks containing the import, innermost first.
    blocks: Vec<EnclosingBlock<'a>>,
}

/// The range to replace and the text to put there. The text is empty for a plain
/// removal, and `pass` when a block would otherwise be left without a statement.
fn import_removal_edit(
    source: &str,
    ast: &ModModule,
    unused: &UnusedImport,
) -> Option<(TextRange, String)> {
    let found = find_import(&ast.body, unused)?;
    let index = found
        .aliases
        .iter()
        .position(|alias| alias.range().contains_range(unused.range))?;

    if found.aliases.len() > 1 {
        let range = if index == 0 {
            TextRange::new(
                found.aliases[index].range().start(),
                found.aliases[index + 1].range().start(),
            )
        } else {
            TextRange::new(
                found.aliases[index - 1].range().end(),
                found.aliases[index].range().end(),
            )
        };
        return Some((range, String::new()));
    }

    // Removing the only alias removes the statement, which would leave an empty
    // block behind if it is the only statement there. An `if` with no other
    // clauses exists solely for its body, so remove it too, repeatedly: the
    // import may be nested several such blocks deep.
    let mut stmt_range = found.stmt_range;
    for block in &found.blocks {
        if block.body.len() > 1 {
            break;
        }
        match block.owner {
            Stmt::If(if_stmt) if if_stmt.elif_else_clauses.is_empty() => {
                stmt_range = if_stmt.range();
            }
            // The owner keeps running whether or not the block does anything, so
            // it needs a body. Replacing just the statement leaves the
            // indentation, and any trailing comment, in place.
            _ => return Some((stmt_range, "pass".to_owned())),
        }
    }
    Some((statement_removal_range(source, stmt_range)?, String::new()))
}

/// Find the import binding `unused`, along with the chain of blocks containing
/// it. Only blocks that share the module scope are searched, since an import in
/// a function or class body binds elsewhere and is never reported as unused.
fn find_import<'a>(body: &'a [Stmt], unused: &UnusedImport) -> Option<FoundImport<'a>> {
    let stmt = body
        .iter()
        .find(|stmt| stmt.range().contains_range(unused.range))?;
    let (stmt_range, aliases) = match stmt {
        Stmt::Import(import) => (import.range(), import.names.as_slice()),
        Stmt::ImportFrom(import) => (import.range(), import.names.as_slice()),
        _ => {
            let (nested, mut found) = nested_bodies(stmt)
                .into_iter()
                .find_map(|nested| find_import(nested, unused).map(|found| (nested, found)))?;
            found.blocks.push(EnclosingBlock {
                owner: stmt,
                body: nested,
            });
            return Some(found);
        }
    };
    Some(FoundImport {
        stmt_range,
        aliases,
        blocks: Vec::new(),
    })
}

/// The blocks directly inside `stmt` that share the scope `stmt` is written in.
fn nested_bodies(stmt: &Stmt) -> Vec<&[Stmt]> {
    match stmt {
        Stmt::If(x) => iter::once(x.body.as_slice())
            .chain(x.elif_else_clauses.iter().map(|c| c.body.as_slice()))
            .collect(),
        Stmt::Try(x) => [
            x.body.as_slice(),
            x.orelse.as_slice(),
            x.finalbody.as_slice(),
        ]
        .into_iter()
        .chain(
            x.handlers
                .iter()
                .map(|ExceptHandler::ExceptHandler(handler)| handler.body.as_slice()),
        )
        .collect(),
        Stmt::For(x) => vec![x.body.as_slice(), x.orelse.as_slice()],
        Stmt::While(x) => vec![x.body.as_slice(), x.orelse.as_slice()],
        Stmt::With(x) => vec![x.body.as_slice()],
        Stmt::Match(x) => x.cases.iter().map(|case| case.body.as_slice()).collect(),
        _ => Vec::new(),
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
        // Reached only when another statement precedes this one on the line,
        // which requires a `;` to separate them.
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
