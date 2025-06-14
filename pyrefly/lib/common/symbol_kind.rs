/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::vec;

use dupe::Dupe;
use lsp_types::CompletionItemKind;
use lsp_types::SemanticTokenModifier;
use lsp_types::SemanticTokenType;

/// The kind of symbol of a binding.
/// It will be displayed in IDEs with different icons.
/// https://adamcoster.com/blog/vscode-workspace-symbol-provider-purpose might give you an idea of
/// how it will look in VSCode.
#[derive(Debug, Clone, Copy, Dupe, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymbolKind {
    Module,
    Attribute,
    Variable,
    Constant,
    Parameter,
    TypeParameter,
    TypeAlias,
    Function,
    Class,
    Bool,
    Str,
}

impl SymbolKind {
    pub fn to_lsp_completion_item_kind(self) -> CompletionItemKind {
        match self {
            SymbolKind::Module => CompletionItemKind::MODULE,
            SymbolKind::Attribute => CompletionItemKind::FIELD,
            SymbolKind::Variable => CompletionItemKind::VARIABLE,
            SymbolKind::Constant => CompletionItemKind::CONSTANT,
            SymbolKind::Parameter => CompletionItemKind::VARIABLE,
            SymbolKind::TypeParameter => CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::TypeAlias => CompletionItemKind::INTERFACE,
            SymbolKind::Function => CompletionItemKind::FUNCTION,
            SymbolKind::Class => CompletionItemKind::CLASS,
            SymbolKind::Bool => CompletionItemKind::VALUE,
            SymbolKind::Str => CompletionItemKind::VALUE,
        }
    }

    pub fn display_for_hover(self) -> String {
        match self {
            SymbolKind::Module => "(module)".to_owned(),
            SymbolKind::Attribute => "(attribute)".to_owned(),
            SymbolKind::Variable => "(variable)".to_owned(),
            SymbolKind::Constant => "(constant)".to_owned(),
            SymbolKind::Parameter => "(parameter)".to_owned(),
            SymbolKind::TypeParameter => "(type parameter)".to_owned(),
            SymbolKind::TypeAlias => "(type alias)".to_owned(),
            SymbolKind::Function => "(function)".to_owned(),
            SymbolKind::Class => "(class)".to_owned(),
            SymbolKind::Bool | SymbolKind::Str => "".to_owned(),
        }
    }

    pub fn to_lsp_semantic_token_type_with_modifiers(
        self,
    ) -> (SemanticTokenType, Vec<SemanticTokenModifier>) {
        match self {
            SymbolKind::Module => (SemanticTokenType::NAMESPACE, vec![]),
            SymbolKind::Attribute => (SemanticTokenType::PROPERTY, vec![]),
            SymbolKind::Variable => (SemanticTokenType::VARIABLE, vec![]),
            SymbolKind::Constant => (
                SemanticTokenType::VARIABLE,
                vec![SemanticTokenModifier::READONLY],
            ),
            SymbolKind::Parameter => (SemanticTokenType::PARAMETER, vec![]),
            SymbolKind::TypeParameter => (SemanticTokenType::TYPE_PARAMETER, vec![]),
            SymbolKind::TypeAlias => (SemanticTokenType::INTERFACE, vec![]),
            // todo(samzhou19815): modifier for async
            SymbolKind::Function => (SemanticTokenType::FUNCTION, vec![]),
            SymbolKind::Class => (SemanticTokenType::CLASS, vec![]),
            SymbolKind::Bool => (SemanticTokenType::VARIABLE, vec![]),
            SymbolKind::Str => (SemanticTokenType::STRING, vec![]),
        }
    }
}
