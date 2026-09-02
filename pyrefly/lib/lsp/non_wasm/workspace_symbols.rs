/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_util::thread_pool::ThreadPool;

use crate::state::lsp::MIN_CHARACTERS_TYPED_AUTOIMPORT;
use crate::state::state::Transaction;

/// One `workspace/symbol` result, before it is converted to an LSP location.
pub struct WorkspaceSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub location: TextRangeWithModule,
}

impl Transaction<'_> {
    pub fn workspace_symbols(
        &self,
        query: &str,
        custom_thread_pool: Option<&ThreadPool>,
    ) -> Option<Vec<WorkspaceSymbol>> {
        if query.len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
            return None;
        }
        let mut result = Vec::new();
        for (definition, _, name, export) in self
            .search_exports_fuzzy(query, custom_thread_pool)
            .unwrap_or_default()
        {
            if let Some(module) = self.get_module_info(&definition) {
                result.push(WorkspaceSymbol {
                    name: name.to_string(),
                    kind: export
                        .symbol_kind
                        .map_or(SymbolKind::VARIABLE, |k| k.to_lsp_symbol_kind()),
                    location: TextRangeWithModule {
                        module,
                        range: export.location,
                    },
                });
            }
        }
        // Keep shared fuzzy ordering intact while preferring non-`__init__.py` matches here.
        result.sort_by_key(|symbol| symbol.location.module.path().is_init());
        Some(result)
    }
}
