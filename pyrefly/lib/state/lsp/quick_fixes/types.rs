/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeActionKind;
use pyrefly_python::module::Module;
use ruff_text_size::TextRange;

/// Description of a refactor edit that stays within the local workspace.
#[derive(Clone, Debug)]
pub struct LocalRefactorCodeAction {
    pub title: String,
    pub edits: Vec<(Module, TextRange, String)>,
    pub kind: CodeActionKind,
}
