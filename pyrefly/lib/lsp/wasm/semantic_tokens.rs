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
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::semantic_tokens::SemanticTokenBuilder;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::semantic_tokens::disabled_ranges_for_module;
use crate::state::state::Transaction;

impl Transaction<'_> {
    pub fn semantic_tokens(
        &self,
        handle: &Handle,
        limit_range: Option<TextRange>,
        limit_cell_idx: Option<usize>,
    ) -> Option<Vec<SemanticToken>> {
        let module_info = self.get_module_info(handle)?;
        let ast = self.get_ast(handle)?;
        let legends = SemanticTokensLegends::new();
        let disabled_ranges = disabled_ranges_for_module(ast.as_ref(), handle.sys_info());
        let mut builder = SemanticTokenBuilder::new(limit_range, disabled_ranges);

        builder.process_ast(
            &ast,
            &|range| self.get_type_trace(handle, range),
            &|key: &Key| {
                let find_preference = FindPreference {
                    import_behavior: ImportBehavior::StopAtRenamedImports,
                    ..Default::default()
                };
                self.key_to_export(handle, key, find_preference)
                    .and_then(|(def_handle, export)| {
                        export.symbol_kind.map(|sk| (def_handle.module(), sk))
                    })
            },
        );

        Some(legends.convert_tokens_into_lsp_semantic_tokens(
            &builder.all_tokens_sorted(),
            module_info,
            limit_cell_idx,
        ))
    }
}
