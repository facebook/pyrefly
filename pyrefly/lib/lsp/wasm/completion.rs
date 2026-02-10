/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionItemLabelDetails;
use lsp_types::CompletionItemTag;
use lsp_types::InsertTextFormat;
use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::keywords::get_keywords;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ExprContext;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::alt::attr::AttrInfo;
use crate::binding::binding::Key;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::state::ide::import_regular_import_edit;
use crate::state::ide::insert_import_edit;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::IdentifierWithContext;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::MIN_CHARACTERS_TYPED_AUTOIMPORT;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::types::Type;

/// Sort text prefix for autoimport completions from public modules.
const SORT_AUTOIMPORT_PUBLIC: &str = "4a";
/// Sort text prefix for autoimport completions from private modules (deprioritized).
const SORT_AUTOIMPORT_PRIVATE: &str = "4b";
/// Default sort text for autoimport completions (used when no explicit sort text is set).
const SORT_AUTOIMPORT_DEFAULT: &str = "4";

/// Options that influence completion item formatting and behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompletionOptions {
    pub supports_completion_item_details: bool,
    pub complete_function_parens: bool,
    pub supports_snippet_completions: bool,
}

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
        let definition = attr_info.definition.clone();
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

    /// Adds literal completions for function call arguments based on parameter types.
    pub(crate) fn add_literal_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
        in_string_literal: bool,
    ) {
        if let Some((callables, chosen_overload_index, active_argument, _)) =
            self.get_callables_from_call(handle, position)
            && let Some(callable) = callables.get(chosen_overload_index)
            && let Some(params) =
                Self::normalize_singleton_function_type_into_params(callable.clone())
            && let Some(arg_index) = Self::active_parameter_index(&params, &active_argument)
            && let Some(param) = params.get(arg_index)
        {
            Self::add_literal_completions_from_type(
                param.as_type(),
                completions,
                in_string_literal,
            );
        }
    }

    /// Adds auto-import completions from exports of other modules using fuzzy matching.
    pub(crate) fn add_autoimport_completions(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        completions: &mut Vec<CompletionItem>,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) {
        // Auto-import can be slow. Let's only return results if there are no local
        // results for now. TODO: re-enable it once we no longer have perf issues.
        // We should not try to generate autoimport when the user has typed very few
        // characters. It's unhelpful to narrow down suggestions.
        if identifier.as_str().len() >= MIN_CHARACTERS_TYPED_AUTOIMPORT
            && let Some(ast) = self.get_ast(handle)
            && let Some(module_info) = self.get_module_info(handle)
        {
            let autoimport_sort_text = |module_name: &str| {
                if module_name.split('.').any(|part| part.starts_with('_')) {
                    SORT_AUTOIMPORT_PRIVATE
                } else {
                    SORT_AUTOIMPORT_PUBLIC
                }
            };
            for (handle_to_import_from, name, export) in
                self.search_exports_fuzzy(identifier.as_str())
            {
                // Using handle itself doesn't always work because handles can be made separately and have different hashes
                if handle_to_import_from.module() == handle.module()
                    || handle_to_import_from.module() == ModuleName::builtins()
                {
                    continue;
                }
                let module_description = handle_to_import_from.module().as_str().to_owned();
                let (insert_text, additional_text_edits, imported_module) = {
                    let (position, insert_text, module_name) = insert_import_edit(
                        &ast,
                        self.config_finder(),
                        handle.dupe(),
                        handle_to_import_from,
                        &name,
                        import_format,
                    );
                    let import_text_edit = TextEdit {
                        range: module_info.to_lsp_range(TextRange::at(position, TextSize::new(0))),
                        new_text: insert_text.clone(),
                    };
                    (insert_text, Some(vec![import_text_edit]), module_name)
                };
                let auto_import_label_detail = format!(" (import {imported_module})");

                completions.push(CompletionItem {
                    label: name,
                    detail: Some(insert_text),
                    kind: export
                        .symbol_kind
                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                            Some(k.to_lsp_completion_item_kind())
                        }),
                    additional_text_edits,
                    label_details: supports_completion_item_details.then_some(
                        CompletionItemLabelDetails {
                            detail: Some(auto_import_label_detail),
                            description: Some(module_description),
                        },
                    ),
                    tags: if export.deprecation.is_some() {
                        Some(vec![CompletionItemTag::DEPRECATED])
                    } else {
                        None
                    },
                    sort_text: Some(autoimport_sort_text(&imported_module).to_owned()),
                    ..Default::default()
                });
            }

            for module_name in self.search_modules_fuzzy(identifier.as_str()) {
                if module_name == handle.module() {
                    continue;
                }
                let module_name_str = module_name.as_str().to_owned();
                let module_sort_text = autoimport_sort_text(&module_name_str).to_owned();
                if let Some(module_handle) = self.import_handle(handle, module_name, None).finding()
                {
                    let (insert_text, additional_text_edits) = {
                        let (position, insert_text) =
                            import_regular_import_edit(&ast, module_handle);
                        let import_text_edit = TextEdit {
                            range: module_info
                                .to_lsp_range(TextRange::at(position, TextSize::new(0))),
                            new_text: insert_text.clone(),
                        };
                        (insert_text, Some(vec![import_text_edit]))
                    };
                    let auto_import_label_detail = format!(" (import {module_name_str})");

                    completions.push(CompletionItem {
                        label: module_name_str.clone(),
                        detail: Some(insert_text),
                        kind: Some(CompletionItemKind::MODULE),
                        additional_text_edits,
                        label_details: supports_completion_item_details.then_some(
                            CompletionItemLabelDetails {
                                detail: Some(auto_import_label_detail),
                                description: Some(module_name_str.clone()),
                            },
                        ),
                        sort_text: Some(module_sort_text),
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// Core completion implementation returning items and incomplete flag.
    pub(crate) fn completion_sorted_opt_with_incomplete(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        options: CompletionOptions,
    ) -> (Vec<CompletionItem>, bool) {
        let CompletionOptions {
            supports_completion_item_details,
            complete_function_parens,
            supports_snippet_completions,
        } = options;
        let mut result = Vec::new();
        let mut is_incomplete = false;
        let mut allow_function_call_parens = false;
        // Because of parser error recovery, `from x impo...` looks like `from x import impo...`
        // If the user might be typing the `import` keyword, add that as an autocomplete option.
        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ImportedName { module_name, .. },
            }) => {
                if let Some(handle) = self.import_handle(handle, module_name, None).finding() {
                    if "import".starts_with(identifier.as_str()) {
                        result.push(CompletionItem {
                            label: "import".to_owned(),
                            kind: Some(CompletionItemKind::KEYWORD),
                            ..Default::default()
                        })
                    }
                    let exports = self.get_exports(&handle);
                    for (name, export) in exports.iter() {
                        let is_deprecated = match export {
                            ExportLocation::ThisModule(export) => export.deprecation.is_some(),
                            ExportLocation::OtherModule(_, _) => false,
                        };
                        result.push(CompletionItem {
                            label: name.to_string(),
                            // todo(kylei): completion kind for exports
                            kind: Some(CompletionItemKind::VARIABLE),
                            tags: if is_deprecated {
                                Some(vec![CompletionItemTag::DEPRECATED])
                            } else {
                                None
                            },
                            ..Default::default()
                        })
                    }
                }
            }
            // TODO: Handle relative import (via ModuleName::new_maybe_relative)
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ImportedModule { .. },
            }) => self
                .import_prefixes(handle, ModuleName::from_name(identifier.id()))
                .iter()
                .for_each(|module_name| {
                    result.push(CompletionItem {
                        label: module_name
                            .components()
                            .last()
                            .unwrap_or(&Name::empty())
                            .to_string(),
                        detail: Some(module_name.to_string()),
                        kind: Some(CompletionItemKind::MODULE),
                        ..Default::default()
                    })
                }),
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => {
                allow_function_call_parens = true;
                if let Some(answers) = self.get_answers(handle)
                    && let Some(base_type) = answers.get_type_trace(base_range)
                {
                    self.ad_hoc_solve(handle, |solver| {
                        solver
                            .completions(base_type, None, true)
                            .iter()
                            .for_each(|x| {
                                let kind = match x.ty {
                                    Some(Type::BoundMethod(_)) => Some(CompletionItemKind::METHOD),
                                    Some(Type::Function(_) | Type::Overload(_)) => {
                                        Some(CompletionItemKind::FUNCTION)
                                    }
                                    Some(Type::Module(_)) => Some(CompletionItemKind::MODULE),
                                    Some(Type::ClassDef(_)) => Some(CompletionItemKind::CLASS),
                                    _ => Some(CompletionItemKind::FIELD),
                                };
                                let ty = &x.ty;
                                let detail =
                                    ty.clone().map(|t| t.as_lsp_string(LspDisplayMode::Hover));
                                let documentation = self.get_docstring_for_attribute(handle, x);
                                result.push(CompletionItem {
                                    label: x.name.as_str().to_owned(),
                                    detail,
                                    kind,
                                    documentation,
                                    sort_text: if x.is_reexport {
                                        Some("1".to_owned())
                                    } else {
                                        None
                                    },
                                    tags: if x.is_deprecated {
                                        Some(vec![CompletionItemTag::DEPRECATED])
                                    } else {
                                        None
                                    },
                                    ..Default::default()
                                });
                            });
                    });
                }
            }
            Some(IdentifierWithContext {
                identifier,
                context,
            }) => {
                if matches!(
                    context,
                    IdentifierContext::Expr(ExprContext::Load | ExprContext::Invalid)
                ) {
                    allow_function_call_parens = true;
                }
                if matches!(context, IdentifierContext::MethodDef { .. }) {
                    Self::add_magic_method_completions(&identifier, &mut result);
                }
                self.add_kwargs_completions(handle, position, &mut result);
                Self::add_keyword_completions(handle, &mut result);
                let has_local_completions = self.add_local_variable_completions(
                    handle,
                    Some(&identifier),
                    position,
                    &mut result,
                );
                if !has_local_completions {
                    self.add_autoimport_completions(
                        handle,
                        &identifier,
                        &mut result,
                        import_format,
                        supports_completion_item_details,
                    );
                }
                // Mark results as incomplete in the following cases so clients keep asking
                // for completions as the user types more:
                // 1. If identifier is below MIN_CHARACTERS_TYPED_AUTOIMPORT threshold,
                //    autoimport completions are skipped and will be checked once threshold
                //    is reached.
                // 2. If local completions exist and blocked autoimport completions,
                //    the local completions might not match as the user continues typing,
                //    and autoimport completions should then be shown.
                if identifier.as_str().len() < MIN_CHARACTERS_TYPED_AUTOIMPORT
                    || has_local_completions
                {
                    is_incomplete = true;
                }
                self.add_builtins_autoimport_completions(handle, Some(&identifier), &mut result);
            }
            None => {
                // todo(kylei): optimization, avoid duplicate ast walkss
                if let Some(mod_module) = self.get_ast(handle) {
                    let nodes = Ast::locate_node(&mod_module, position);
                    if nodes.is_empty() {
                        Self::add_keyword_completions(handle, &mut result);
                        self.add_local_variable_completions(handle, None, position, &mut result);
                        self.add_builtins_autoimport_completions(handle, None, &mut result);
                    }
                    let in_string_literal = nodes
                        .iter()
                        .any(|node| matches!(node, AnyNodeRef::ExprStringLiteral(_)));
                    self.add_match_literal_completions(
                        handle,
                        &nodes,
                        &mut result,
                        in_string_literal,
                    );
                    self.add_literal_completions(handle, position, &mut result, in_string_literal);
                    self.add_dict_key_completions(
                        handle,
                        mod_module.as_ref(),
                        position,
                        &mut result,
                    );
                    // in foo(x=<>, y=2<>), the first containing node is AnyNodeRef::Arguments(_)
                    // in foo(<>), the first containing node is AnyNodeRef::ExprCall
                    if let Some(first) = nodes.first()
                        && matches!(first, AnyNodeRef::ExprCall(_) | AnyNodeRef::Arguments(_))
                    {
                        self.add_kwargs_completions(handle, position, &mut result);
                    }
                }
            }
        }
        if complete_function_parens && allow_function_call_parens {
            Self::add_function_call_parens(&mut result, supports_snippet_completions);
        }
        for item in &mut result {
            let sort_text = if item
                .tags
                .as_ref()
                .is_some_and(|tags| tags.contains(&CompletionItemTag::DEPRECATED))
            {
                "9"
            } else if let Some(sort_text) = &item.sort_text {
                // 1 is reserved for re-exports
                sort_text.as_str()
            } else if item.additional_text_edits.is_some() {
                SORT_AUTOIMPORT_DEFAULT
            } else if item.label.starts_with("__") {
                "3"
            } else if item.label.as_str().starts_with("_") {
                "2"
            } else {
                "0"
            }
            .to_owned();
            item.sort_text = Some(sort_text);
        }
        (result, is_incomplete)
    }
}
