/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;
use pyrefly_python::module::TextRangeWithModule;

use crate::state::lsp::MIN_CHARACTERS_TYPED_AUTOIMPORT;
use crate::state::state::Transaction;

impl Transaction<'_> {
    pub fn workspace_symbols(
        &self,
        query: &str,
    ) -> Option<Vec<(String, SymbolKind, TextRangeWithModule)>> {
        if query.len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
            return None;
        }
        let mut result = Vec::new();
        for (handle, name, export) in self.search_exports_fuzzy(query) {
            if let Some(module) = self.get_module_info(&handle) {
                let kind = export
                    .symbol_kind
                    .map_or(SymbolKind::VARIABLE, |k| k.to_lsp_symbol_kind());
                let location = TextRangeWithModule {
                    module,
                    range: export.location,
                };
                result.push((name, kind, location));
            }
        }
        Some(result)
    }
}
