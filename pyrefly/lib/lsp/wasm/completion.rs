/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::keywords::get_keywords;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use ruff_python_ast::Identifier;
use ruff_text_size::TextSize;

use crate::alt::attr::AttrInfo;
use crate::export::exports::Export;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::types::Type;

impl Transaction<'_> {
    /// Adds completion items for literal types (e.g., `Literal["foo", "bar"]`).
    pub(crate) fn add_literal_completions_from_type(
        param_type: &Type,
        completions: &mut Vec<CompletionItem>,
        in_string_literal: bool,
    ) {
        match param_type {
            Type::Literal(lit) => {
                // TODO: Pass the flag correctly for whether literal string is single quoted or double quoted
                let label = lit.value.to_string_escaped(true);
                let insert_text = if in_string_literal {
                    if let Lit::Str(s) = &lit.value {
                        s.to_string()
                    } else {
                        label.clone()
                    }
                } else {
                    label.clone()
                };
                completions.push(CompletionItem {
                    label,
                    kind: Some(CompletionItemKind::VALUE),
                    detail: Some(format!("{param_type}")),
                    insert_text: Some(insert_text),
                    ..Default::default()
                });
            }
            Type::Union(box Union { members, .. }) => {
                for member in members {
                    Self::add_literal_completions_from_type(member, completions, in_string_literal);
                }
            }
            _ => {}
        }
    }

    /// Adds completions for magic methods (dunder methods like `__init__`, `__str__`, etc.).
    pub(crate) fn add_magic_method_completions(
        identifier: &Identifier,
        completions: &mut Vec<CompletionItem>,
    ) {
        let typed = identifier.as_str();
        if !typed.is_empty() && !typed.starts_with("__") {
            return;
        }
        for name in dunder::MAGIC_METHOD_NAMES {
            if name.starts_with(typed) {
                completions.push(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::METHOD),
                    ..Default::default()
                });
            }
        }
    }

    /// Adds completions for Python keywords (e.g., `if`, `for`, `class`, etc.).
    pub(crate) fn add_keyword_completions(handle: &Handle, completions: &mut Vec<CompletionItem>) {
        get_keywords(handle.sys_info().version())
            .iter()
            .for_each(|name| {
                completions.push(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    ..Default::default()
                })
            });
    }

    /// Retrieves documentation for an export to display in completion items.
    pub(crate) fn get_documentation_from_export(
        &self,
        export_info: Option<(Handle, Export)>,
    ) -> Option<lsp_types::Documentation> {
        let (definition_handle, export) = export_info?;
        let docstring_range = export.docstring_range?;
        let def_module = self.get_module_info(&definition_handle)?;
        let docstring = Docstring(docstring_range, def_module.clone()).resolve();
        let documentation = lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: docstring,
        });
        Some(documentation)
    }

    /// Adds keyword argument completions (e.g., `arg=`) for function/method calls.
    pub(crate) fn add_kwargs_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some((callables, overload_idx, _, _)) =
            self.get_callables_from_call(handle, position)
            && let Some(callable) = callables.get(overload_idx).cloned()
            && let Some(params) = Self::normalize_singleton_function_type_into_params(callable)
        {
            for param in params {
                match param {
                    Param::Pos(name, ty, _)
                    | Param::PosOnly(Some(name), ty, _)
                    | Param::KwOnly(name, ty, _)
                    | Param::VarArg(Some(name), ty) => {
                        if name.as_str() != "self" {
                            completions.push(CompletionItem {
                                label: format!("{}=", name.as_str()),
                                detail: Some(ty.to_string()),
                                kind: Some(CompletionItemKind::VARIABLE),
                                ..Default::default()
                            });
                        }
                    }
                    Param::VarArg(None, _) | Param::Kwargs(_, _) | Param::PosOnly(None, _, _) => {}
                }
            }
        }
    }

    /// Gets docstring documentation for an attribute to display in completion items.
    pub(crate) fn get_docstring_for_attribute(
        &self,
        handle: &Handle,
        attr_info: &AttrInfo,
    ) -> Option<lsp_types::Documentation> {
        let definition = attr_info.definition.as_ref()?.clone();
        let attribute_definition = self.resolve_attribute_definition(
            handle,
            &attr_info.name,
            definition,
            attr_info.docstring_range,
            FindPreference::default(),
        );

        let (definition, Some(docstring_range)) = attribute_definition? else {
            return None;
        };
        let docstring = Docstring(docstring_range, definition.module);

        Some(lsp_types::Documentation::MarkupContent(
            lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: docstring.resolve().trim().to_owned(),
            },
        ))
    }
}
