/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Experimental LSP method that shows the type of something using fully-qualified names
//! See https://github.com/facebook/pyrefly/issues/1181

use dupe::Dupe;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Position;
use lsp_types::TextDocumentIdentifier;
use lsp_types::request::Request;
use pyrefly_build::handle::Handle;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::display::TypeDisplayContext;
use serde::Deserialize;
use serde::Serialize;

use crate::state::require::Require;
use crate::state::state::Transaction;

#[derive(Debug)]
pub enum ProvideType {}

impl Request for ProvideType {
    type Params = ProvideTypeParams;
    type Result = Option<ProvideTypeResponse>;
    const METHOD: &'static str = "types/provide-type";
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvideTypeParams {
    pub text_document: TextDocumentIdentifier,
    pub positions: Vec<Position>,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
pub struct ProvideTypeResponse {
    pub contents: Vec<MarkupContent>,
}

pub fn provide_type(
    transaction: &mut Transaction<'_>,
    handle: &Handle,
    positions: Vec<Position>,
) -> Option<ProvideTypeResponse> {
    // This LSP method works for unopened files.
    // Check if the file is already loaded in memory. If not, load it.
    let was_loaded = transaction.get_module_info(handle).is_some();
    if !was_loaded {
        transaction.run(&[handle.dupe()], Require::Everything);
    }
    let info = transaction.get_module_info(handle)?;
    let mut contents = Vec::new();

    for position in positions {
        let text_size = info.from_lsp_position(position, None);
        if let Some(ty) = transaction.get_type_at(handle, text_size) {
            let mut c = TypeDisplayContext::new(&[&ty]);
            c.set_lsp_display_mode(LspDisplayMode::Hover);
            c.always_display_module_name();
            contents.push(MarkupContent {
                kind: MarkupKind::PlainText,
                value: c.display(&ty).to_string(),
            });
        } else {
            contents.push(MarkupContent {
                kind: MarkupKind::PlainText,
                value: String::new(),
            });
        }
    }
    Some(ProvideTypeResponse { contents })
}
