/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use lsp_types::SemanticToken;
use lsp_types::SemanticTokenModifier;
use lsp_types::SemanticTokenType;
use lsp_types::SemanticTokensLegend;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit as _;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::binding::Key;

pub struct SemanticTokensLegends {
    token_types_index: HashMap<SemanticTokenType, u32>,
    token_modifiers_index: HashMap<SemanticTokenModifier, u32>,
}

impl SemanticTokensLegends {
    pub fn lsp_semantic_token_legends() -> SemanticTokensLegend {
        SemanticTokensLegend {
            token_types: vec![
                SemanticTokenType::NAMESPACE,
                SemanticTokenType::TYPE,
                SemanticTokenType::CLASS,
                SemanticTokenType::ENUM,
                SemanticTokenType::INTERFACE,
                SemanticTokenType::STRUCT,
                SemanticTokenType::TYPE_PARAMETER,
                SemanticTokenType::PARAMETER,
                SemanticTokenType::VARIABLE,
                SemanticTokenType::PROPERTY,
                SemanticTokenType::ENUM_MEMBER,
                SemanticTokenType::EVENT,
                SemanticTokenType::FUNCTION,
                SemanticTokenType::METHOD,
                SemanticTokenType::MACRO,
                SemanticTokenType::KEYWORD,
                SemanticTokenType::MODIFIER,
                SemanticTokenType::COMMENT,
                SemanticTokenType::STRING,
                SemanticTokenType::NUMBER,
                SemanticTokenType::REGEXP,
                SemanticTokenType::OPERATOR,
                SemanticTokenType::DECORATOR,
            ],
            token_modifiers: vec![
                SemanticTokenModifier::DECLARATION,
                SemanticTokenModifier::DEFINITION,
                SemanticTokenModifier::READONLY,
                SemanticTokenModifier::STATIC,
                SemanticTokenModifier::DEPRECATED,
                SemanticTokenModifier::ABSTRACT,
                SemanticTokenModifier::ASYNC,
                SemanticTokenModifier::MODIFICATION,
                SemanticTokenModifier::DOCUMENTATION,
                SemanticTokenModifier::DEFAULT_LIBRARY,
            ],
        }
    }

    pub fn new() -> Self {
        let lsp_legend = Self::lsp_semantic_token_legends();
        let mut token_types_index = HashMap::new();
        let mut token_modifiers_index = HashMap::new();
        for (i, token_type) in lsp_legend.token_types.iter().enumerate() {
            token_types_index.insert(token_type.clone(), i as u32);
        }
        for (i, token_modifier) in lsp_legend.token_modifiers.iter().enumerate() {
            token_modifiers_index.insert(token_modifier.clone(), i as u32);
        }
        Self {
            token_types_index,
            token_modifiers_index,
        }
    }

    pub fn convert_tokens_into_lsp_semantic_tokens(
        &self,
        tokens: &[SemanticTokenWithFullRange],
        module_info: Module,
    ) -> Vec<SemanticToken> {
        let mut previous_line = 0;
        let mut previous_col = 0;
        let mut lsp_semantic_tokens = Vec::new();
        for token in tokens {
            let source_range = module_info.display_range(token.range);
            let length = token.range.len().to_u32();
            if length == 0 {
                continue;
            }
            let current_line = source_range.start.line.to_zero_indexed();
            let current_col = source_range.start.column.get() - 1;
            let (delta_line, delta_start) = {
                let delta_line = current_line - previous_line;
                let delta_start = if previous_line == current_line {
                    current_col - previous_col
                } else {
                    current_col
                };
                previous_line = current_line;
                previous_col = current_col;
                (delta_line, delta_start)
            };
            let token_type = *self.token_types_index.get(&token.token_type).unwrap();
            let mut token_modifiers_bitset = 0;
            for modifier in &token.token_modifiers {
                let index = *self.token_modifiers_index.get(modifier).unwrap();
                token_modifiers_bitset |= 1 << index;
            }
            lsp_semantic_tokens.push(SemanticToken {
                delta_line,
                delta_start,
                length,
                token_type,
                token_modifiers_bitset,
            });
        }
        lsp_semantic_tokens.dedup_by(|current, previous| {
            current.delta_line == 0 && current.delta_start == 0 && current.length == previous.length
        });
        lsp_semantic_tokens
    }

    #[cfg(test)]
    pub fn get_modifiers(&self, token_modifiers_bitset: u32) -> Vec<SemanticTokenModifier> {
        let mut modifiers = Vec::new();
        for (modifier, index) in &self.token_modifiers_index {
            let singleton_set = (1 << *index) as u32;
            if (token_modifiers_bitset & singleton_set) == singleton_set {
                modifiers.push(modifier.clone());
            }
        }
        // needed for a deterministic print ordering in tests
        modifiers.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        modifiers
    }
}

pub struct SemanticTokenWithFullRange {
    pub range: TextRange,
    pub token_type: SemanticTokenType,
    pub token_modifiers: Vec<SemanticTokenModifier>,
}

pub struct SemanticTokenBuilder {
    tokens: Vec<SemanticTokenWithFullRange>,
    limit_range: Option<TextRange>,
}

impl SemanticTokenBuilder {
    pub fn new(limit_range: Option<TextRange>) -> Self {
        Self {
            tokens: Vec::new(),
            limit_range,
        }
    }

    fn push_if_in_range(
        &mut self,
        range: TextRange,
        token_type: SemanticTokenType,
        token_modifiers: Vec<SemanticTokenModifier>,
    ) {
        if self.limit_range.is_none_or(|x| x.contains_range(range)) {
            self.tokens.push(SemanticTokenWithFullRange {
                range,
                token_type,
                token_modifiers,
            })
        }
    }

    pub fn process_key(
        &mut self,
        key: &Key,
        definition_module: ModuleName,
        symbol_kind: SymbolKind,
    ) {
        let reference_range = key.range();
        let (token_type, mut token_modifiers) =
            symbol_kind.to_lsp_semantic_token_type_with_modifiers();
        let is_default_library = {
            let module_name_str = definition_module.as_str();
            module_name_str == "builtins"
                || module_name_str == "typing"
                || module_name_str == "typing_extensions"
        };
        if is_default_library {
            token_modifiers.push(SemanticTokenModifier::DEFAULT_LIBRARY);
        }
        self.push_if_in_range(reference_range, token_type, token_modifiers);
    }

    fn process_arguments(&mut self, args: &Arguments) {
        for keyword in &args.keywords {
            if let Some(arg) = &keyword.arg {
                self.push_if_in_range(arg.range, SemanticTokenType::PARAMETER, Vec::new());
            }
        }
    }

    fn process_expr(&mut self, x: &Expr) {
        match x {
            Expr::Call(call) if let Expr::Attribute(attr) = call.func.as_ref() => {
                self.push_if_in_range(attr.attr.range(), SemanticTokenType::METHOD, Vec::new());
                attr.value.visit(&mut |x| self.process_expr(x));
                for arg in call.arguments.arguments_source_order() {
                    arg.value().visit(&mut |x| self.process_expr(x));
                }
                self.process_arguments(&call.arguments);
            }
            Expr::Call(call) => {
                self.process_arguments(&call.arguments);
                x.recurse(&mut |x| self.process_expr(x));
            }
            Expr::Attribute(attr) => {
                // todo(samzhou19815): if the class's base is Enum, it should be ENUM_MEMBER
                self.push_if_in_range(attr.attr.range(), SemanticTokenType::PROPERTY, Vec::new());
                attr.value.visit(&mut |x| self.process_expr(x));
            }
            _ => {
                x.recurse(&mut |x| self.process_expr(x));
            }
        }
    }

    pub fn process_ast(&mut self, ast: &ModModule) {
        ast.visit(&mut |e| self.process_expr(e));
    }

    pub fn all_tokens_sorted(self) -> Vec<SemanticTokenWithFullRange> {
        let mut tokens = self.tokens;
        tokens.sort_by(|a, b| a.range.start().cmp(&b.range.start()));
        tokens
    }
}
