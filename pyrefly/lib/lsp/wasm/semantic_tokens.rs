/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use lsp_types::SemanticToken;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Stmt;
use ruff_text_size::TextRange;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::export::exports::Export;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::semantic_tokens::SemanticTokenBuilder;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::semantic_tokens::disabled_ranges_for_module;
use crate::state::state::Transaction;

/// A binding that is verified to be a binding for a name in the source code.
/// This data structure carries the proof for the verification,
/// which includes the definition information, and the binding itself.
pub(crate) struct NamedBinding {
    pub definition_handle: Handle,
    pub definition_export: Export,
    pub key: Key,
}

impl Transaction<'_> {
    /// Bindings can contain synthetic bindings, which are not meaningful to end users.
    /// This function helps to filter out such bindings and only leave bindings that eventually
    /// jumps to a name in the source.
    pub(crate) fn named_bindings(&self, handle: &Handle, bindings: &Bindings) -> Vec<NamedBinding> {
        let mut named_bindings = Vec::new();
        for idx in bindings.keys::<Key>() {
            let key = bindings.idx_to_key(idx);
            if matches!(key, Key::Phi(..) | Key::Narrow(..)) {
                // These keys are always synthetic and never serves as a name definition.
                continue;
            }
            if let Some((definition_handle, definition_export)) = self.key_to_export(
                handle,
                key,
                FindPreference {
                    import_behavior: ImportBehavior::StopAtRenamedImports,
                    ..Default::default()
                },
            ) {
                named_bindings.push(NamedBinding {
                    definition_handle,
                    definition_export,
                    key: key.clone(),
                });
            }
        }
        named_bindings
    }

    pub fn semantic_tokens(
        &self,
        handle: &Handle,
        limit_range: Option<TextRange>,
        limit_cell_idx: Option<usize>,
    ) -> Option<Vec<SemanticToken>> {
        let module_info = self.get_module_info(handle)?;
        let bindings = self.get_bindings(handle)?;
        let ast = self.get_ast(handle)?;
        let legends = SemanticTokensLegends::new();
        let disabled_ranges = disabled_ranges_for_module(ast.as_ref(), handle.sys_info());
        let mut builder = SemanticTokenBuilder::new(limit_range, disabled_ranges);
        let mut symbol_kinds: HashMap<ShortIdentifier, (ModuleName, SymbolKind)> = HashMap::new();
        for NamedBinding {
            definition_handle,
            definition_export,
            key,
        } in self.named_bindings(handle, &bindings)
        {
            if let Export {
                symbol_kind: Some(symbol_kind),
                ..
            } = definition_export
            {
                let binding = bindings.get(bindings.key_to_idx(&key));
                let definition_module = match binding {
                    Binding::Import(module, _, _) | Binding::Module(module, ..) => *module,
                    _ => definition_handle.module(),
                };
                if let Key::Definition(short) = &key {
                    symbol_kinds.insert(short.clone(), (definition_module, symbol_kind));
                }
                builder.process_key(&key, definition_module, symbol_kind);
            }
        }
        for stmt in &ast.body {
            add_import_from_alias_tokens(&mut builder, stmt, &symbol_kinds);
        }
        builder.process_ast(&ast, &|range| self.get_type_trace(handle, range));
        Some(legends.convert_tokens_into_lsp_semantic_tokens(
            &builder.all_tokens_sorted(),
            module_info,
            limit_cell_idx,
        ))
    }
}

fn add_import_from_alias_tokens(
    builder: &mut SemanticTokenBuilder,
    stmt: &Stmt,
    symbol_kinds: &HashMap<ShortIdentifier, (ModuleName, SymbolKind)>,
) {
    if let Stmt::ImportFrom(import_from) = stmt {
        for alias in &import_from.names {
            if let Some(asname) = &alias.asname {
                let key = ShortIdentifier::new(asname);
                if let Some((definition_module, symbol_kind)) = symbol_kinds.get(&key) {
                    builder.process_range(alias.name.range, *definition_module, *symbol_kind);
                }
            }
        }
    }
    stmt.recurse(&mut |inner| add_import_from_alias_tokens(builder, inner, symbol_kinds));
}
