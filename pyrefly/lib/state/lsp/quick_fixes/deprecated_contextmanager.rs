/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::ModuleInfo;
use crate::error::error::Error;
use crate::error::error::ErrorQuickFix;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_function;

/// Replaces the deprecated iterator return annotation on a context manager.
///
/// The diagnostic identifies the semantic replacement, while the AST locates
/// the spelling used by the decorated function (including imported aliases).
pub(crate) fn replace_return_code_action(
    module_info: &ModuleInfo,
    ast: &ModModule,
    error: &Error,
) -> Option<(String, Module, TextRange, String, bool)> {
    let (from, to) = error.quick_fixes().iter().find_map(|fix| match fix {
        ErrorQuickFix::ReplaceDeprecatedContextManagerReturn { from, to } => {
            Some((from.as_str(), to.as_str()))
        }
        _ => None,
    })?;
    let function_def = find_enclosing_function(ast, error.range())?;
    let annotation = function_def.returns.as_deref()?;
    let annotation_base = match annotation {
        Expr::Subscript(subscript) => subscript.value.as_ref(),
        annotation_base => annotation_base,
    };
    let (range, needs_import) = match annotation_base {
        Expr::Name(name) => (name.range(), true),
        Expr::Attribute(attribute) => (attribute.attr.range(), false),
        _ => return None,
    };
    Some((
        format!("Replace `{from}` with `{to}`"),
        module_info.dupe(),
        range,
        to.to_owned(),
        needs_import && !bare_name_in_scope(ast, to),
    ))
}

fn bare_name_in_scope(ast: &ModModule, name: &str) -> bool {
    ast.body.iter().any(|stmt| {
        let Stmt::ImportFrom(import_from) = stmt else {
            return false;
        };
        if import_from.level != 0
            || !import_from
                .module
                .as_ref()
                .is_some_and(|module| matches!(module.id.as_str(), "typing" | "collections.abc"))
        {
            return false;
        }
        import_from.names.iter().any(|alias| {
            alias.name.id.as_str() == "*"
                || (alias.name.id.as_str() == name
                    && alias
                        .asname
                        .as_ref()
                        .is_none_or(|asname| asname.id.as_str() == name))
        })
    })
}
