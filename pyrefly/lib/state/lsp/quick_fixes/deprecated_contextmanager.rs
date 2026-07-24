/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_python_ast::token::parenthesized_range;
use ruff_python_ast::visitor::source_order::SourceOrderVisitor;
use ruff_python_ast::visitor::source_order::walk_expr;
use ruff_python_ast::visitor::source_order::walk_stmt;
use ruff_python_parser::parse_expression_range;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ModuleInfo;
use crate::error::error::ContextManagerKind;
use crate::error::error::Error;
use crate::error::error::ErrorQuickFix;
use crate::state::ide::prepare_import_insertion;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_function;
use crate::state::lsp::quick_fixes::extract_shared::unique_name;
use crate::types::types::Type;

/// Replaces the deprecated iterator return annotation on a context manager.
///
/// Only direct iterator annotations on the diagnosed decorator are editable.
pub(crate) fn replace_return_code_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &ModuleInfo,
    ast: &ModModule,
    error: &Error,
) -> Option<(String, Vec<(Module, TextRange, String)>)> {
    let kind = error.quick_fixes().iter().find_map(|fix| match fix {
        ErrorQuickFix::ReplaceDeprecatedContextManagerReturn(kind) => Some(*kind),
        ErrorQuickFix::ReplaceWithEnumMember { .. }
        | ErrorQuickFix::AssertNotNone
        | ErrorQuickFix::RemoveUnusedSuppression(_)
        | ErrorQuickFix::ImportModule { .. } => None,
    })?;
    let (from, to, defaults) = kind.return_annotation_replacement();
    let function_def = find_enclosing_function(ast, error.range())?;
    if !function_def
        .decorator_list
        .iter()
        .any(|decorator| decorator.range() == error.range())
        || !Ast::body_contains_yield(&function_def.body)
        || function_def.is_async != matches!(kind, ContextManagerKind::Async)
    {
        return None;
    }
    let annotation = function_def.returns.as_deref()?;
    let annotation_base = match annotation {
        Expr::Subscript(subscript) => subscript.value.as_ref(),
        annotation_base => annotation_base,
    };
    let position = match annotation_base {
        Expr::Name(name) => name.range().start(),
        Expr::Attribute(attribute) => attribute.attr.range().start(),
        _ => return None,
    };
    // A bound alias can already contain type arguments. Its outer name is not
    // interchangeable with the iterator class, even if its value is an iterator.
    let Type::ClassDef(class) = transaction.get_type_at(handle, position)? else {
        return None;
    };
    if !class.has_toplevel_qname("typing", from)
        && !class.has_toplevel_qname("collections.abc", from)
    {
        return None;
    }

    // These standard modules export both iterator and generator classes. Other
    // modules can re-export just the iterator, so use a fresh typing import.
    let qualifier = if let Expr::Attribute(attribute) = annotation_base
        && let Some(Type::Module(module)) =
            transaction.get_type_at(handle, attribute.value.range().end() - TextSize::new(1))
        && matches!(module.to_string().as_str(), "typing" | "collections.abc")
    {
        Some(attribute)
    } else {
        None
    };
    let mut edits = Vec::new();
    let (name_range, binding) = if let Some(attribute) = qualifier {
        // Preserve the qualifier and every byte around the iterator name.
        (attribute.attr.range(), to.to_owned())
    } else {
        let mut names = BindingNames::default();
        names.visit_body(&ast.body);
        if names.wildcard_import {
            return None;
        }
        // A sole, earlier standard import cannot be shadowed by another binding.
        let existing = ast.body.iter().find_map(|stmt| {
            let Stmt::ImportFrom(import) = stmt else {
                return None;
            };
            if import.end() > function_def.start()
                || import.level != 0
                || !import.module.as_ref().is_some_and(|module| {
                    matches!(module.id.as_str(), "typing" | "collections.abc")
                })
            {
                return None;
            }
            import.names.iter().find_map(|alias| {
                let binding = alias.asname.as_ref().unwrap_or(&alias.name);
                (alias.name.id == to
                    && names
                        .sites
                        .get(&binding.id)
                        .is_some_and(|sites| sites.as_slice() == [binding.range()]))
                .then(|| binding.id.to_string())
            })
        });
        let binding = if let Some(existing) = existing {
            existing
        } else {
            let binding = unique_name(to, |name| names.sites.contains_key(name));
            let import = if binding == to {
                format!("from typing import {to}\n")
            } else {
                format!("from typing import {to} as {binding}\n")
            };
            let (position, import) = prepare_import_insertion(ast, import);
            edits.push((module_info.dupe(), TextRange::empty(position), import));
            binding
        };
        (annotation_base.range(), binding)
    };
    edits.push((module_info.dupe(), name_range, binding));
    if let Expr::Subscript(subscript) = annotation {
        let argument = match subscript.slice.as_ref() {
            Expr::Tuple(tuple) if tuple.elts.len() == 1 => &tuple.elts[0],
            Expr::Tuple(_) | Expr::Starred(_) => return None,
            argument => argument,
        };
        if handle.sys_info().version() < PythonVersion::new(3, 13, 0) {
            let parsed = parse_expression_range(module_info.contents(), annotation.range()).ok()?;
            // Grouping parentheses belong to the original argument. Insert before
            // its trailing comma, after all closing parentheses, without copying source.
            let range = parenthesized_range(argument.into(), subscript.into(), parsed.tokens())
                .unwrap_or(argument.range());
            edits.push((
                module_info.dupe(),
                TextRange::empty(range.end()),
                defaults.to_owned(),
            ));
        }
    }
    Some((format!("Replace `{from}` with `{to}`"), edits))
}

/// Binding sites use the parser's normalized identifiers, across enclosing and
/// nested scopes. Loads, attribute names, and comments do not bind a fresh name.
#[derive(Default)]
struct BindingNames {
    sites: HashMap<Name, Vec<TextRange>>,
    wildcard_import: bool,
}

impl<'a> SourceOrderVisitor<'a> for BindingNames {
    fn visit_identifier(&mut self, identifier: &'a Identifier) {
        self.sites
            .entry(identifier.id.clone())
            .or_default()
            .push(identifier.range());
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Name(name) => {
                if name.ctx != ExprContext::Load {
                    self.sites
                        .entry(name.id.clone())
                        .or_default()
                        .push(name.range());
                }
            }
            Expr::Attribute(attribute) => self.visit_expr(&attribute.value),
            _ => walk_expr(self, expr),
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::ImportFrom(import) => {
                for alias in &import.names {
                    self.wildcard_import |= alias.name.id == "*";
                    self.visit_identifier(alias.asname.as_ref().unwrap_or(&alias.name));
                }
            }
            Stmt::Import(import) => {
                for alias in &import.names {
                    if let Some(name) = &alias.asname {
                        self.visit_identifier(name);
                    } else {
                        let path = alias.name.id.as_str();
                        let root = path.split_once('.').map_or(path, |(root, _)| root);
                        self.sites
                            .entry(Name::new(root))
                            .or_default()
                            .push(alias.name.range());
                    }
                }
            }
            // Compound statements remain in the traversal, including module-level
            // conditionals, exception handlers, and loops containing wildcard imports.
            _ => walk_stmt(self, stmt),
        }
    }
}
