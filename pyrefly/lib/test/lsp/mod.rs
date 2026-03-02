/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]

mod call_hierarchy;
mod code_actions;
mod completion;
mod declaration;
mod definition;
mod diagnostic;
mod document_symbols;
mod folding_ranges;
mod hover;
mod hover_docstring;
mod hover_type;
mod implementation;
mod inlay_hint;
mod local_find_refs;
mod lsp_interaction;
mod rename;
mod semantic_tokens;
mod signature_help;
