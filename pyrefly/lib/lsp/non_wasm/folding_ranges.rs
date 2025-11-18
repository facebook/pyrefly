/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::FoldingRangeKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::folding::folding_ranges;
use ruff_text_size::TextRange;

use crate::state::state::Transaction;

impl Transaction<'_> {
    pub fn folding_ranges(
        &self,
        handle: &Handle,
    ) -> Option<Vec<(TextRange, Option<FoldingRangeKind>)>> {
        let ast = self.get_ast(handle)?;
        let module_info = self.get_module_info(handle)?;
        Some(folding_ranges(&module_info, &ast.body))
    }

    pub fn docstring_ranges(&self, handle: &Handle) -> Option<Vec<TextRange>> {
        let ranges = self.folding_ranges(handle)?;
        Some(
            ranges
                .into_iter()
                .filter_map(|(range, kind)| {
                    if kind == Some(FoldingRangeKind::Comment) {
                        Some(range)
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}
