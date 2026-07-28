/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Lazy reference-count CodeLens creation and resolution.

use lsp_types::CodeLens;
use lsp_types::Command;
use lsp_types::DocumentSymbol;
use lsp_types::Location;
use lsp_types::SymbolKind;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_util::telemetry::EmptyResponseReason;
use ruff_text_size::TextRange;
use serde::Deserialize;
use serde::Serialize;

use crate::ModuleInfo;
use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::state::state::Transaction;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ReferenceCodeLensData {
    pub(super) uri: Url,
}

/// Collect unresolved lenses without performing definition lookup or global reference searches.
pub(super) fn unresolved_code_lenses(
    transaction: &Transaction<'_>,
    handle: &Handle,
    uri: &Url,
) -> Result<Vec<CodeLens>, EmptyResponseReason> {
    let symbols = transaction
        .symbols(handle, None)
        .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;
    let mut lenses = Vec::new();
    collect_symbol_lenses(&symbols, uri, &mut lenses);
    Ok(lenses)
}

/// Decode the opaque data attached to an unresolved reference-count lens.
pub(super) fn resolve_data(lens: &CodeLens) -> Result<ReferenceCodeLensData, String> {
    let data = lens
        .data
        .clone()
        .ok_or_else(|| "Reference CodeLens is missing resolve data".to_owned())?;
    serde_json::from_value(data)
        .map_err(|error| format!("Invalid reference CodeLens resolve data: {error}"))
}

/// Attach the reference command, excluding every location backed by a notebook cell.
pub(super) fn resolve_code_lens(
    mut lens: CodeLens,
    source_uri: &Url,
    results: Vec<(ModuleInfo, Vec<TextRange>)>,
    path_remapper: Option<&PathRemapper>,
) -> CodeLens {
    let mut locations = Vec::new();
    for (info, ranges) in results {
        let Some(uri) = module_info_to_uri(&info, path_remapper) else {
            continue;
        };
        for range in ranges {
            if info.to_cell_for_lsp(range.start()).is_some() {
                continue;
            }
            locations.push(Location {
                uri: uri.clone(),
                range: info.to_lsp_range(range),
            });
        }
    }

    let reference_count = locations.len();
    let title = if reference_count == 1 {
        "1 reference".to_owned()
    } else {
        format!("{reference_count} references")
    };
    lens.command = Some(Command {
        title,
        command: "editor.action.showReferences".to_owned(),
        arguments: Some(vec![
            serde_json::to_value(source_uri).expect("URI should serialize for code lens"),
            serde_json::to_value(lens.range.start)
                .expect("Position should serialize for code lens"),
            serde_json::to_value(&locations).expect("Locations should serialize for code lens"),
        ]),
        tooltip: None,
    });
    lens.data = None;
    lens
}

/// Traverse document symbols and create one unresolved lens per reference-countable definition.
fn collect_symbol_lenses(symbols: &[DocumentSymbol], uri: &Url, lenses: &mut Vec<CodeLens>) {
    for symbol in symbols {
        if matches!(
            symbol.kind,
            SymbolKind::CLASS | SymbolKind::FUNCTION | SymbolKind::METHOD
        ) {
            lenses.push(CodeLens {
                range: symbol.selection_range,
                command: None,
                data: Some(
                    serde_json::to_value(ReferenceCodeLensData { uri: uri.clone() })
                        .expect("Reference CodeLens data should serialize"),
                ),
            });
        }
        if let Some(children) = symbol.children.as_deref() {
            collect_symbol_lenses(children, uri, lenses);
        }
    }
}
