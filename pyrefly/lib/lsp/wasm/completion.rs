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
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ExprContext;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::alt::attr::AttrInfo;
use crate::binding::binding::Key;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::state::ide::common_alias_target_module;
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

/// Classification of a completion item's source, used for ranking.
#[derive(Clone, Copy, Default)]
pub(crate) enum CompletionSource {
    /// Keywords, variables, literals, builtins, dict keys, etc.
    #[default]
    Local,
    /// Defined in another module, exposed from this one.
    Reexport,
    /// Auto-import from a public module path.
    AutoimportPublic,
    /// Auto-import from a private module path (segment starts with `_`).
    AutoimportPrivate,
}

/// A completion item paired with ranking metadata.
pub(crate) struct RankedCompletion {
    pub(crate) item: CompletionItem,
    pub(crate) source: CompletionSource,
    pub(crate) is_incompatible: bool,
}

impl RankedCompletion {
    /// Wraps a `CompletionItem` with default (local, compatible) ranking metadata.
    pub(crate) fn new(item: CompletionItem) -> Self {
        Self {
            item,
            source: CompletionSource::Local,
            is_incompatible: false,
        }
    }
}

/// All completion ranking logic lives here. Assigns `sort_text` to each item
/// based on its source classification, name prefix, compatibility, and MRU rank.
///
/// `mru_rank` uses two levels of `Option`:
/// - `None` means the MRU system is not active (e.g. wasm path). sort_text uses
///   the compact base format (`"0"`, `"1"`, etc.).
/// - `Some(None)` means the MRU system is active but the item was not found in
///   the MRU list. sort_text uses the extended format with rank 9999 so it sorts
///   after MRU-matched items.
/// - `Some(Some(rank))` means the item was found at position `rank` in the MRU.
fn assign_sort_text(ranked: &mut RankedCompletion, mru_rank: Option<Option<usize>>) {
    let is_deprecated = ranked
        .item
        .tags
        .as_ref()
        .is_some_and(|tags| tags.contains(&CompletionItemTag::DEPRECATED));

    let base = if is_deprecated {
        "9".to_owned()
    } else {
        let base = match ranked.source {
            CompletionSource::AutoimportPrivate => "4b",
            CompletionSource::AutoimportPublic => "4a",
            CompletionSource::Reexport => "1",
            CompletionSource::Local => {
                if ranked.item.label.starts_with("__") {
                    "3"
                } else if ranked.item.label.starts_with('_') {
                    "2"
                } else {
                    "0"
                }
            }
        };
        if ranked.is_incompatible {
            format!("{base}z")
        } else {
            base.to_owned()
        }
    };

    match mru_rank {
        Some(Some(rank)) => {
            let rank = rank.min(9999);
            ranked.item.sort_text = Some(format!("{base}.{rank:04}.{}", ranked.item.label));
            ranked.item.preselect = if rank == 0 { Some(true) } else { None };
        }
        Some(None) => {
            ranked.item.sort_text = Some(format!("{base}.9999.{}", ranked.item.label));
        }
        None => {
            ranked.item.sort_text = Some(base);
        }
    }
}

fn autoimport_source(module_name: &str) -> CompletionSource {
    if module_name.split('.').any(|part| part.starts_with('_')) {
        CompletionSource::AutoimportPrivate
    } else {
        CompletionSource::AutoimportPublic
    }
}

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
    /// Adds a common alias auto-import completion (e.g. `np` -> `numpy`).
    /// Returns the module name that was aliased when a completion was added.
    fn add_common_alias_autoimport_completion(
        &self,
        handle: &Handle,
        ast: &ModModule,
        module_info: &Module,
        identifier_text: &str,
        supports_completion_item_details: bool,
        completions: &mut Vec<RankedCompletion>,
    ) -> Option<ModuleName> {
        let module_name_str = common_alias_target_module(identifier_text)?;
        let module_name = ModuleName::from_str(module_name_str);
        if module_name == handle.module() {
            return None;
        }
        let module_handle = self.import_handle(handle, module_name, None).finding()?;
        let (position, import_text, completion_label) =
            import_regular_import_edit(ast, module_handle, Some(identifier_text));
        let import_text_edit = TextEdit {
            range: module_info.to_lsp_range(TextRange::at(position, TextSize::new(0))),
            new_text: import_text.clone(),
        };
        let auto_import_label_detail = format!(" (import {module_name_str} as {identifier_text})");
        completions.push(RankedCompletion {
            item: CompletionItem {
                label: completion_label.clone(),
                detail: Some(import_text),
                kind: Some(CompletionItemKind::MODULE),
                additional_text_edits: Some(vec![import_text_edit]),
                label_details: supports_completion_item_details.then_some(
                    CompletionItemLabelDetails {
                        detail: Some(auto_import_label_detail),
                        description: Some(module_name_str.to_owned()),
                    },
                ),
                insert_text: Some(completion_label),
                ..Default::default()
            },
            source: autoimport_source(module_name_str),
            is_incompatible: false,
        });
        Some(module_name)
    }

    /// Adds completion items for literal types (e.g., `Literal["foo", "bar"]`).
    pub(crate) fn add_literal_completions_from_type(
        param_type: &Type,
        completions: &mut Vec<RankedCompletion>,
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
                completions.push(RankedCompletion::new(CompletionItem {
                    label,
                    kind: Some(CompletionItemKind::VALUE),
                    detail: Some(format!("{param_type}")),
                    insert_text: Some(insert_text),
                    ..Default::default()
                }));
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
        completions: &mut Vec<RankedCompletion>,
    ) {
        let typed = identifier.as_str();
        if !typed.is_empty() && !typed.starts_with("__") {
            return;
        }
        for name in dunder::MAGIC_METHOD_NAMES {
            if name.starts_with(typed) {
                completions.push(RankedCompletion::new(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::METHOD),
                    ..Default::default()
                }));
            }
        }
    }

    /// Adds completions for Python keywords (e.g., `if`, `for`, `class`, etc.).
    pub(crate) fn add_keyword_completions(
        handle: &Handle,
        completions: &mut Vec<RankedCompletion>,
    ) {
        get_keywords(handle.sys_info().version())
            .iter()
            .for_each(|name| {
                completions.push(RankedCompletion::new(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    ..Default::default()
                }))
            });
    }

    /// Adds function/method completion inserts with parentheses, using snippets when supported.
    pub(crate) fn add_function_call_parens(
        completions: &mut [RankedCompletion],
        supports_snippets: bool,
    ) {
        for ranked in completions {
            let item = &mut ranked.item;
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
        completions: &mut Vec<RankedCompletion>,
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
                            completions.push(RankedCompletion::new(CompletionItem {
                                label: format!("{}=", name.as_str()),
                                detail: Some(ty.to_string()),
                                kind: Some(CompletionItemKind::VARIABLE),
                                ..Default::default()
                            }));
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
        completions: &mut Vec<RankedCompletion>,
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
                completions.push(RankedCompletion::new(CompletionItem {
                    label: name.as_str().to_owned(),
                    detail: None,
                    kind,
                    data: Some(serde_json::json!("builtin")),
                    ..Default::default()
                }));
            }
        }
    }

    fn expected_call_argument_type(&self, handle: &Handle, position: TextSize) -> Option<Type> {
        let (callables, chosen_overload_index, active_argument, _) =
            self.get_callables_from_call(handle, position)?;
        let callable = callables.get(chosen_overload_index)?.clone();
        let params = Self::normalize_singleton_function_type_into_params(callable)?;
        let arg_index = Self::active_parameter_index(&params, &active_argument)?;
        let param = params.get(arg_index)?;
        Some(param.as_type().clone())
    }

    fn is_incompatible_with_expected_type(
        &self,
        handle: &Handle,
        expected_type: Option<&Type>,
        actual_type: Option<&Type>,
    ) -> bool {
        let Some(expected_type) = expected_type else {
            return false;
        };
        let Some(actual_type) = actual_type else {
            return false;
        };
        self.ad_hoc_solve(handle, |solver| {
            solver.is_subset_eq(actual_type, expected_type)
        })
        .map(|compatible| !compatible)
        .unwrap_or(false)
    }

    /// Adds completions for local variables and returns true if any were added.
    /// If an identifier is present, filters matches using fuzzy matching.
    pub(crate) fn add_local_variable_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        position: TextSize,
        expected_type: Option<&Type>,
        completions: &mut Vec<RankedCompletion>,
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
                let detail = ty.as_ref().map(|t| t.to_string());
                let documentation = self.get_documentation_from_export(export_info);
                let is_incompatible =
                    self.is_incompatible_with_expected_type(handle, expected_type, ty.as_ref());

                has_added_any = true;
                completions.push(RankedCompletion {
                    item: CompletionItem {
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
                    },
                    source: CompletionSource::Local,
                    is_incompatible,
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
        completions: &mut Vec<RankedCompletion>,
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
        completions: &mut Vec<RankedCompletion>,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) {
        // Auto-import can be slow. Let's only return results if there are no local
        // results for now. TODO: re-enable it once we no longer have perf issues.
        // We should not try to generate autoimport when the user has typed very few
        // characters. It's unhelpful to narrow down suggestions.
        if let Some(ast) = self.get_ast(handle)
            && let Some(module_info) = self.get_module_info(handle)
        {
            let identifier_text = identifier.as_str();
            let mut aliased_modules = SmallSet::new();
            if let Some(module_name) = self.add_common_alias_autoimport_completion(
                handle,
                &ast,
                &module_info,
                identifier_text,
                supports_completion_item_details,
                completions,
            ) {
                aliased_modules.insert(module_name);
            }

            if identifier_text.len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
                return;
            }
            for (handle_to_import_from, name, export) in self.search_exports_fuzzy(identifier_text)
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

                completions.push(RankedCompletion {
                    item: CompletionItem {
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
                        ..Default::default()
                    },
                    source: autoimport_source(&imported_module),
                    is_incompatible: false,
                });
            }

            for module_name in self.search_modules_fuzzy(identifier_text) {
                if module_name == handle.module() {
                    continue;
                }
                if aliased_modules.contains(&module_name) {
                    continue;
                }
                let module_name_str = module_name.as_str().to_owned();
                let source = autoimport_source(&module_name_str);
                if let Some(module_handle) = self.import_handle(handle, module_name, None).finding()
                {
                    let (import_text, additional_text_edits) = {
                        let (position, import_text, _) =
                            import_regular_import_edit(&ast, module_handle, None);
                        let import_text_edit = TextEdit {
                            range: module_info
                                .to_lsp_range(TextRange::at(position, TextSize::new(0))),
                            new_text: import_text.clone(),
                        };
                        (import_text, Some(vec![import_text_edit]))
                    };
                    let auto_import_label_detail = format!(" (import {module_name_str})");

                    completions.push(RankedCompletion {
                        item: CompletionItem {
                            label: module_name_str.clone(),
                            detail: Some(import_text),
                            kind: Some(CompletionItemKind::MODULE),
                            additional_text_edits,
                            label_details: supports_completion_item_details.then_some(
                                CompletionItemLabelDetails {
                                    detail: Some(auto_import_label_detail),
                                    description: Some(module_name_str.clone()),
                                },
                            ),
                            ..Default::default()
                        },
                        source,
                        is_incompatible: false,
                    });
                }
            }
        }
    }

    /// Suggest Literal values when completing inside a `match` value pattern.
    ///
    /// We can't reuse the call-argument literal completion path here because
    /// `case <value>:` isn't a call site, so we never get parameter types to
    /// infer literals from. Instead, we look for a match value/singleton
    /// pattern at the cursor and pull the `match` subject's type to surface
    /// its Literal members.
    pub(crate) fn add_match_literal_completions(
        &self,
        handle: &Handle,
        covering_nodes: &[AnyNodeRef],
        completions: &mut Vec<RankedCompletion>,
        in_string_literal: bool,
    ) {
        let mut is_match_value_pattern = false;
        let mut subject = None;
        for node in covering_nodes {
            match node {
                AnyNodeRef::PatternMatchValue(_) | AnyNodeRef::PatternMatchSingleton(_) => {
                    is_match_value_pattern = true;
                }
                AnyNodeRef::StmtMatch(stmt_match) => {
                    subject = Some(stmt_match.subject.as_ref());
                }
                _ => {}
            }
            if is_match_value_pattern && subject.is_some() {
                break;
            }
        }
        if !is_match_value_pattern {
            return;
        }
        let Some(subject) = subject else {
            return;
        };
        if let Some(subject_type) = self.get_type_trace(handle, subject.range()) {
            Self::add_literal_completions_from_type(&subject_type, completions, in_string_literal);
        }
    }

    /// Core completion implementation returning items and incomplete flag.
    pub(crate) fn completion_sorted_opt_with_incomplete<F>(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        options: CompletionOptions,
        mut mru_index: Option<F>,
    ) -> (Vec<CompletionItem>, bool)
    where
        F: FnMut(&CompletionItem) -> Option<usize>,
    {
        let CompletionOptions {
            supports_completion_item_details,
            complete_function_parens,
            supports_snippet_completions,
        } = options;
        let mut result: Vec<RankedCompletion> = Vec::new();
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
                        result.push(RankedCompletion::new(CompletionItem {
                            label: "import".to_owned(),
                            kind: Some(CompletionItemKind::KEYWORD),
                            ..Default::default()
                        }))
                    }
                    let exports = self.get_exports(&handle);
                    for (name, export) in exports.iter() {
                        let is_deprecated = match export {
                            ExportLocation::ThisModule(export) => export.deprecation.is_some(),
                            ExportLocation::OtherModule(_, _) => false,
                        };
                        result.push(RankedCompletion::new(CompletionItem {
                            label: name.to_string(),
                            // todo(kylei): completion kind for exports
                            kind: Some(CompletionItemKind::VARIABLE),
                            tags: if is_deprecated {
                                Some(vec![CompletionItemTag::DEPRECATED])
                            } else {
                                None
                            },
                            ..Default::default()
                        }))
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
                    result.push(RankedCompletion::new(CompletionItem {
                        label: module_name
                            .components()
                            .last()
                            .unwrap_or(&Name::empty())
                            .to_string(),
                        detail: Some(module_name.to_string()),
                        kind: Some(CompletionItemKind::MODULE),
                        ..Default::default()
                    }))
                }),
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => {
                let expected_type = self.expected_call_argument_type(handle, position);
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
                                let is_incompatible = self.is_incompatible_with_expected_type(
                                    handle,
                                    expected_type.as_ref(),
                                    ty.as_ref(),
                                );
                                let source = if x.is_reexport {
                                    CompletionSource::Reexport
                                } else {
                                    CompletionSource::Local
                                };
                                result.push(RankedCompletion {
                                    item: CompletionItem {
                                        label: x.name.as_str().to_owned(),
                                        detail,
                                        kind,
                                        documentation,
                                        tags: if x.is_deprecated {
                                            Some(vec![CompletionItemTag::DEPRECATED])
                                        } else {
                                            None
                                        },
                                        ..Default::default()
                                    },
                                    source,
                                    is_incompatible,
                                });
                            });
                    });
                }
            }
            Some(IdentifierWithContext {
                identifier,
                context,
            }) => {
                let expected_type = if matches!(
                    context,
                    IdentifierContext::Expr(ExprContext::Load | ExprContext::Invalid)
                ) {
                    self.expected_call_argument_type(handle, position)
                } else {
                    None
                };
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
                    expected_type.as_ref(),
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
                    let expected_type = self.expected_call_argument_type(handle, position);
                    let nodes = Ast::locate_node(&mod_module, position);
                    if nodes.is_empty() {
                        Self::add_keyword_completions(handle, &mut result);
                        self.add_local_variable_completions(
                            handle,
                            None,
                            position,
                            expected_type.as_ref(),
                            &mut result,
                        );
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
        for ranked in &mut result {
            let mru_rank = mru_index.as_mut().map(|index| (*index)(&ranked.item));
            assign_sort_text(ranked, mru_rank);
        }
        (result.into_iter().map(|r| r.item).collect(), is_incomplete)
    }
}
