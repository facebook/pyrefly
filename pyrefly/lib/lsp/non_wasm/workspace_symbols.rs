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
        let matches = self
            .search_workspace_symbols_fuzzy(query, custom_thread_pool)
            .unwrap_or_default();

        Some(
            matches
                .into_iter()
                .filter_map(|m| {
                    Some(WorkspaceSymbol {
                        name: m.name.to_string(),
                        kind: m
                            .kind
                            .map_or(SymbolKind::VARIABLE, |kind| kind.to_lsp_symbol_kind()),
                        location: TextRangeWithModule {
                            module: self.get_module_info(&m.handle)?,
                            range: m.range,
                        },
                    })
                })
                .collect(),
        )
    }
}
