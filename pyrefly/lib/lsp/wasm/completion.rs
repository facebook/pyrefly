/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionItemTag;
use lsp_types::InsertTextFormat;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::keywords::get_keywords;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

use crate::alt::attr::AttrInfo;
use crate::binding::binding::Key;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::types::Type;

/// Returns true if the client supports snippet completions in completion items.
pub(crate) fn supports_snippet_completions(capabilities: &lsp_types::ClientCapabilities) -> bool {
    capabilities
        .text_document
        .as_ref()
        .and_then(|t| t.completion.as_ref())
        .and_then(|c| c.completion_item.as_ref())
        .and_then(|ci| ci.snippet_support)
        .unwrap_or(false)
}

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

    /// Adds function/method completion inserts with parentheses, using snippets when supported.
    pub(crate) fn add_function_call_parens(
        completions: &mut [CompletionItem],
        supports_snippets: bool,
    ) {
        for item in completions {
            if item.insert_text.is_some() || item.text_edit.is_some() {
                continue;
            }
            if !matches!(
                item.kind,
                Some(CompletionItemKind::FUNCTION | CompletionItemKind::METHOD)
            ) {
                continue;
            }

            if supports_snippets {
                item.insert_text = Some(format!("{}($0)", item.label));
                item.insert_text_format = Some(InsertTextFormat::SNIPPET);
            } else {
                item.insert_text = Some(format!("{}()", item.label));
            }
        }
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

    /// Adds completions from the builtins module, optionally filtered by fuzzy match.
    pub(crate) fn add_builtins_autoimport_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some(builtin_handle) = self
            .import_handle(handle, ModuleName::builtins(), None)
            .finding()
        {
            let builtin_exports = self.get_exports(&builtin_handle);
            for (name, location) in builtin_exports.iter() {
                if let Some(identifier) = identifier
                    && SkimMatcherV2::default()
                        .smart_case()
                        .fuzzy_match(name.as_str(), identifier.as_str())
                        .is_none()
                {
                    continue;
                }
                let kind = match location {
                    ExportLocation::OtherModule(..) => continue,
                    ExportLocation::ThisModule(export) => export
                        .symbol_kind
                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                            Some(k.to_lsp_completion_item_kind())
                        }),
                };
                completions.push(CompletionItem {
                    label: name.as_str().to_owned(),
                    detail: None,
                    kind,
                    data: Some(serde_json::json!("builtin")),
                    ..Default::default()
                });
            }
        }
    }

    /// Adds completions for local variables and returns true if any were added.
    /// If an identifier is present, filters matches using fuzzy matching.
    pub(crate) fn add_local_variable_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) -> bool {
        let mut has_added_any = false;
        if let Some(bindings) = self.get_bindings(handle)
            && let Some(module_info) = self.get_module_info(handle)
        {
            for idx in bindings.available_definitions(position) {
                let key = bindings.idx_to_key(idx);
                let label = match key {
                    Key::Definition(id) => module_info.code_at(id.range()),
                    Key::Anywhere(id, _) => id,
                    _ => continue,
                };
                if let Some(identifier) = identifier
                    && SkimMatcherV2::default()
                        .fuzzy_match(label, identifier.as_str())
                        .is_none()
                {
                    continue;
                }
                let binding = bindings.get(idx);
                let ty = self.get_type(handle, key);
                let export_info = self.key_to_export(handle, key, FindPreference::default());

                let kind = if let Some((_, ref export)) = export_info {
                    export
                        .symbol_kind
                        .map_or(CompletionItemKind::VARIABLE, |k| {
                            k.to_lsp_completion_item_kind()
                        })
                } else {
                    binding
                        .symbol_kind()
                        .map_or(CompletionItemKind::VARIABLE, |k| {
                            k.to_lsp_completion_item_kind()
                        })
                };

                let is_deprecated = ty.as_ref().is_some_and(|t| {
                    if let Type::ClassDef(cls) = t {
                        self.ad_hoc_solve(handle, |solver| {
                            solver.get_metadata_for_class(cls).deprecation().is_some()
                        })
                        .unwrap_or(false)
                    } else {
                        t.function_deprecation().is_some()
                    }
                });
                let detail = ty.map(|t| t.to_string());
                let documentation = self.get_documentation_from_export(export_info);

                has_added_any = true;
                completions.push(CompletionItem {
                    label: label.to_owned(),
                    detail,
                    kind: Some(kind),
                    documentation,
                    tags: if is_deprecated {
                        Some(vec![CompletionItemTag::DEPRECATED])
                    } else {
                        None
                    },
                    ..Default::default()
                })
            }
        }
        has_added_any
    }
}
