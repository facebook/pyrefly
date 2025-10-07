/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Experimental LSP method that shows the type of something using fully-qualified names
//! See https://github.com/facebook/pyrefly/issues/1181

use lsp_types::MarkupContent;
use lsp_types::Position;
use lsp_types::TextDocumentIdentifier;
use lsp_types::request::Request;
use serde::Deserialize;
use serde::Serialize;

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
