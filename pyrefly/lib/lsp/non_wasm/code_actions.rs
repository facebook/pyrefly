/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use lsp_types::Location;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextRange;

use crate::ModuleInfo;

pub(crate) fn workspace_edit_for_refactor_edits(
    edits: Vec<(ModuleInfo, TextRange, String)>,
    to_lsp_location: impl Fn(&TextRangeWithModule) -> Option<Location>,
) -> Option<WorkspaceEdit> {
    let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();
    for (module, edit_range, new_text) in edits {
        let Some(lsp_location) = to_lsp_location(&TextRangeWithModule {
            module,
            range: edit_range,
        }) else {
            continue;
        };
        changes.entry(lsp_location.uri).or_default().push(TextEdit {
            range: lsp_location.range,
            new_text,
        });
    }
    if changes.is_empty() {
        None
    } else {
        Some(WorkspaceEdit {
            changes: Some(changes),
            ..Default::default()
        })
    }
}
