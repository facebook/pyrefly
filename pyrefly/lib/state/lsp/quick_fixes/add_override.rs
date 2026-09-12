/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_function;
use crate::state::lsp::quick_fixes::extract_shared::function_has_decorator;
use crate::state::lsp::quick_fixes::extract_shared::line_indent_and_start;

/// Builds a quick fix for the `MissingOverrideDecorator` diagnostic: a method
/// that overrides a parent-class member but lacks an `@override` decorator.
///
/// Returns `(title, module, range, insert_text)` for a single text edit that
/// inserts `@override` on its own line above the method, matching the method's
/// indentation. Returns `None` when the enclosing function can't be found or
/// already has an `@override` decorator.
///
/// This only builds the decorator edit. The caller pairs it with an import edit
/// (see `override_in_scope`) when `override` is not already in scope, so the two
/// edits apply together as a single quick fix.
pub(crate) fn add_override_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error_range: TextRange,
) -> Option<(String, Module, TextRange, String)> {
    let function_def = find_enclosing_function(ast, error_range)?;
    if function_has_decorator(function_def, "override") {
        return None;
    }
    // Anchor above the first decorator (so `@override` becomes the outermost
    // decorator) if any are present, otherwise above the `def` line itself.
    let anchor = function_def
        .decorator_list
        .first()
        .map_or_else(|| function_def.range().start(), |d| d.range().start());
    let (indent, line_start) = line_indent_and_start(module_info.contents().as_str(), anchor)?;
    let insert_text = format!("{indent}@override\n");
    Some((
        "Add `@override` decorator".to_owned(),
        module_info.dupe(),
        TextRange::new(line_start, line_start),
        insert_text,
    ))
}

/// Returns whether `override` is already importable as a bare name in this module
/// (via `from typing import override` or `from typing_extensions import override`),
/// in which case the decorator fix does not need to add an import.
pub(crate) fn override_in_scope(ast: &ModModule) -> bool {
    ast.body.iter().any(|stmt| {
        let Stmt::ImportFrom(import_from) = stmt else {
            return false;
        };
        if import_from.level != 0 {
            return false;
        }
        let Some(module) = &import_from.module else {
            return false;
        };
        if module.id.as_str() != "typing" && module.id.as_str() != "typing_extensions" {
            return false;
        }
        import_from.names.iter().any(|alias| {
            if alias.name.id.as_str() != "override" {
                return false;
            }
            match &alias.asname {
                None => true,
                Some(asname) => asname.id.as_str() == "override",
            }
        })
    })
}
