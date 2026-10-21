/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CompletionItemKind;
use lsp_types::SemanticTokenModifiers;
use lsp_types::SemanticTokenTypes;

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
    Method,
    Class,
}

impl SymbolKind {
    pub fn to_lsp_symbol_kind(self) -> lsp_types::SymbolKind {
        match self {
            SymbolKind::Module => lsp_types::SymbolKind::Module,
            SymbolKind::Attribute => lsp_types::SymbolKind::Field,
            SymbolKind::Variable => lsp_types::SymbolKind::Variable,
            SymbolKind::Constant => lsp_types::SymbolKind::Constant,
            SymbolKind::Parameter => lsp_types::SymbolKind::Variable,
            SymbolKind::TypeParameter => lsp_types::SymbolKind::TypeParameter,
            SymbolKind::TypeAlias => lsp_types::SymbolKind::Interface,
            SymbolKind::Function => lsp_types::SymbolKind::Function,
            SymbolKind::Method => lsp_types::SymbolKind::Method,
            SymbolKind::Class => lsp_types::SymbolKind::Class,
        }
    }

    pub fn to_lsp_completion_item_kind(self) -> CompletionItemKind {
        match self {
            SymbolKind::Module => CompletionItemKind::Module,
            SymbolKind::Attribute => CompletionItemKind::Field,
            SymbolKind::Variable => CompletionItemKind::Variable,
            SymbolKind::Constant => CompletionItemKind::Constant,
            SymbolKind::Parameter => CompletionItemKind::Variable,
            SymbolKind::TypeParameter => CompletionItemKind::TypeParameter,
            SymbolKind::TypeAlias => CompletionItemKind::Interface,
            SymbolKind::Function => CompletionItemKind::Function,
            SymbolKind::Method => CompletionItemKind::Method,
            SymbolKind::Class => CompletionItemKind::Class,
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
            SymbolKind::Method => "(method)".to_owned(),
            SymbolKind::Class => "(class)".to_owned(),
        }
    }

    pub fn to_lsp_semantic_token_type_with_modifiers(
        self,
    ) -> (SemanticTokenTypes, Vec<SemanticTokenModifiers>) {
        match self {
            SymbolKind::Module => (SemanticTokenTypes::Namespace, vec![]),
            SymbolKind::Attribute => (SemanticTokenTypes::Property, vec![]),
            SymbolKind::Variable => (SemanticTokenTypes::Variable, vec![]),
            SymbolKind::Constant => (
                SemanticTokenTypes::Variable,
                vec![SemanticTokenModifiers::Readonly],
            ),
            SymbolKind::Parameter => (SemanticTokenTypes::Parameter, vec![]),
            SymbolKind::TypeParameter => (SemanticTokenTypes::TypeParameter, vec![]),
            SymbolKind::TypeAlias => (SemanticTokenTypes::Interface, vec![]),
            // todo(samzhou19815): modifier for async
            SymbolKind::Function => (SemanticTokenTypes::Function, vec![]),
            SymbolKind::Method => (SemanticTokenTypes::Method, vec![]),
            SymbolKind::Class => (SemanticTokenTypes::Class, vec![]),
        }
    }
}
