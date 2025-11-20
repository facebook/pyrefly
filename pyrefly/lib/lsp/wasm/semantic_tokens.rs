/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SemanticToken;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextRange;

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
                builder.process_key(&key, definition_handle.module(), symbol_kind)
            }
        }
        builder.process_ast(&ast, &|range| self.get_type_trace(handle, range));
        Some(legends.convert_tokens_into_lsp_semantic_tokens(
            &builder.all_tokens_sorted(),
            module_info,
            limit_cell_idx,
        ))
    }
}
