/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SemanticToken;
use lsp_types::SemanticTokenType;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextRange;

use crate::alt::attr::AttrDefinition;
use crate::binding::binding::Key;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::lsp::attribute_symbol_kind_from_type;
use crate::state::semantic_tokens::SemanticTokenBuilder;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::semantic_tokens::disabled_ranges_for_module;
use crate::state::state::Transaction;

impl Transaction<'_> {
    pub fn semantic_tokens(
        &self,
        handle: &Handle,
        limit_range: Option<TextRange>,
        limit_cell_idx: Option<usize>,
        include_syntax_tokens: bool,
    ) -> Option<Vec<SemanticToken>> {
        let module_info = self.get_module_info(handle)?;
        let parsed = self.get_parsed_module(handle)?;
        let ast = parsed.module();
        let legends = SemanticTokensLegends::new();
        let disabled_ranges = disabled_ranges_for_module(ast.as_ref(), *handle.sys_info());
        let mut builder = SemanticTokenBuilder::new(limit_range, disabled_ranges);

        if include_syntax_tokens && let Some(tokens) = parsed.tokens() {
            builder.process_syntax_tokens(&tokens);
        }

        builder.process_ast(
            &ast,
            &|range, base_range, name| {
                let ty = self.get_type_trace(handle, range)?;
                let kind = attribute_symbol_kind_from_type(&ty)
                    .to_lsp_semantic_token_type_with_modifiers()
                    .0;
                // Enum-valued fields can have the same type as enum members, so
                // confirm that the attribute's definitions are enum members.
                if kind == SemanticTokenType::PROPERTY
                    && let Some(base) = self.get_type_trace(handle, base_range)
                    && self.ad_hoc_solve(handle, "semantic_token_enum_member", |solver| {
                        let definitions = solver.completions(base, Some(name), false);
                        !definitions.is_empty()
                            && definitions.iter().all(|info| {
                                matches!(&info.definition, AttrDefinition::FullyResolved { cls, .. }
                                    if solver.get_enum_member(cls, name).is_some())
                            })
                    }) == Some(true)
                {
                    return Some(SemanticTokenType::ENUM_MEMBER);
                }
                Some(kind)
            },
            &|key: &Key| {
                let find_preference = FindPreference {
                    import_behavior: ImportBehavior::StopAtRenamedImports,
                    ..Default::default()
                };
                self.key_to_export(handle, key, find_preference)
                    .and_then(|(def_handle, export)| {
                        export.symbol_kind.map(|sk| (def_handle.module(), sk))
                    })
            },
        );

        Some(legends.convert_tokens_into_lsp_semantic_tokens(
            &builder.all_tokens_sorted(),
            module_info,
            limit_range,
            limit_cell_idx,
        ))
    }
}
