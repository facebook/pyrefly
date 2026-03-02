/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type source information for hover display.
//!
//! This module provides functionality to display where a type came from,
//! such as narrowing conditions or first-use inference sites.

use lsp_types::Url;
use pyrefly_util::lined_buffer::DisplayPos;

/// Set the URL fragment to a position suitable for editor navigation.
/// Handles both regular source files (`L{line},{col}`) and notebook cells
/// (`{cell},L{line},{col}`).
pub fn set_display_pos_fragment(url: &mut Url, pos: DisplayPos) {
    let fragment = if let Some(cell) = pos.cell() {
        format!(
            "{},L{},{}",
            cell.get(),
            pos.line_within_cell().get(),
            pos.column()
        )
    } else {
        format!("L{},{}", pos.line_within_file().get(), pos.column())
    };
    url.set_fragment(Some(&fragment));
}

// Type source tracking is only available on non-wasm targets because it requires
// Url::from_file_path which is not available in wasm builds.
#[cfg(not(target_arch = "wasm32"))]
mod impl_ {
    use lsp_types::Url;
    use pyrefly_build::handle::Handle;
    use pyrefly_graph::index::Idx;
    use pyrefly_python::module::Module;
    use pyrefly_python::short_identifier::ShortIdentifier;
    use ruff_python_ast::ExprContext;
    use ruff_text_size::Ranged;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use starlark_map::small_set::SmallSet;

    use super::set_display_pos_fragment;
    use crate::binding::binding::Binding;
    use crate::binding::binding::FirstUse;
    use crate::binding::binding::Key;
    use crate::binding::bindings::Bindings;
    use crate::state::lsp::IdentifierContext;
    use crate::state::state::Transaction;

    fn format_type_source_location(module: &Module, range: TextRange) -> String {
        let display_pos = module.display_pos(range.start());
        let location = display_pos.to_string();
        let Ok(mut url) = Url::from_file_path(module.path().as_path()) else {
            return location;
        };
        set_display_pos_fragment(&mut url, display_pos);
        format!("[{}]({})", location, url)
    }

    fn format_code_snippet(module: &Module, range: TextRange) -> Option<String> {
        if range.is_empty() {
            return None;
        }
        let snippet = module.code_at(range);
        let cleaned = snippet
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .replace('`', "'");
        if cleaned.is_empty() {
            return None;
        }
        const MAX_LEN: usize = 80;
        let mut truncated = cleaned;
        if truncated.chars().count() > MAX_LEN {
            let max_chars = MAX_LEN.saturating_sub(3);
            let end_byte = truncated
                .char_indices()
                .nth(max_chars)
                .map(|(idx, _)| idx)
                .unwrap_or_else(|| truncated.len());
            truncated.truncate(end_byte);
            truncated.push_str("...");
        }
        Some(truncated)
    }

    fn narrow_source_for_key(
        bindings: &Bindings,
        module: &Module,
        start_idx: Idx<Key>,
    ) -> Option<String> {
        let mut current = start_idx;
        let mut seen = SmallSet::new();
        loop {
            if seen.contains(&current) {
                return None;
            }
            seen.insert(current);
            match bindings.get(current) {
                Binding::Forward(next) => current = *next,
                Binding::Narrow(_, op, _) => {
                    let key = bindings.idx_to_key(current);
                    let Key::Narrow(x) = key else {
                        return None;
                    };
                    let (name, op_range, _) = x.as_ref();
                    let location = format_type_source_location(module, *op_range);
                    let mut msg = format!("Narrowed by condition at {location}");
                    if let Some(snippet) = op
                        .as_python_snippet(name, &|range| format_code_snippet(module, range))
                        .or_else(|| format_code_snippet(module, *op_range))
                    {
                        msg.push_str(&format!(": `{snippet}`"));
                    }
                    return Some(msg);
                }
                _ => return None,
            }
        }
    }

    fn definition_short_identifier(bindings: &Bindings, key: &Key) -> Option<ShortIdentifier> {
        let mut current = bindings.key_to_idx(key);
        let mut seen = SmallSet::new();
        loop {
            if seen.contains(&current) {
                return None;
            }
            seen.insert(current);
            let current_key = bindings.idx_to_key(current);
            if let Key::Definition(short_identifier) = current_key {
                return Some(*short_identifier);
            }
            match bindings.get(current) {
                Binding::Forward(next)
                | Binding::Narrow(next, ..)
                | Binding::CompletedPartialType(next, ..)
                | Binding::PartialTypeWithUpstreamsCompleted(next, ..)
                | Binding::LoopPhi(next, ..) => current = *next,
                // All branches of a Phi node originate from the same variable definition,
                // so any branch will lead to the same Key::Definition. We follow the first.
                Binding::Phi(_, branches) if !branches.is_empty() => {
                    current = branches[0].value_key;
                }
                _ => return None,
            }
        }
    }

    fn first_use_source_for_key(
        bindings: &Bindings,
        module: &Module,
        key: &Key,
        hover_position: TextSize,
    ) -> Option<String> {
        let def_identifier = definition_short_identifier(bindings, key)?;
        let key = Key::CompletedPartialType(def_identifier);
        if !bindings.is_valid_key(&key) {
            return None;
        }
        let idx = bindings.key_to_idx(&key);
        match bindings.get(idx) {
            Binding::CompletedPartialType(_, FirstUse::UsedBy(use_idx)) => {
                let use_range = bindings.idx_to_key(*use_idx).range();
                if use_range.contains(hover_position) {
                    return None;
                }
                let location = format_type_source_location(module, use_range);
                let mut msg = format!("Inferred from first use at {location}");
                if let Some(snippet) = format_code_snippet(module, use_range) {
                    msg.push_str(&format!(": `{snippet}`"));
                }
                Some(msg)
            }
            _ => None,
        }
    }

    /// Collect type source information (narrowing, first-use inference) for hover display.
    pub fn type_sources_for_hover(
        transaction: &Transaction<'_>,
        handle: &Handle,
        position: TextSize,
    ) -> Vec<String> {
        let Some(bindings) = transaction.get_bindings(handle) else {
            return Vec::new();
        };
        let Some(module) = transaction.get_module_info(handle) else {
            return Vec::new();
        };
        let Some(identifier_with_context) = transaction.identifier_at(handle, position) else {
            return Vec::new();
        };
        let key = match identifier_with_context.context {
            IdentifierContext::Expr(expr_context) => match expr_context {
                ExprContext::Store => {
                    Key::Definition(ShortIdentifier::new(&identifier_with_context.identifier))
                }
                ExprContext::Load | ExprContext::Del | ExprContext::Invalid => {
                    Key::BoundName(ShortIdentifier::new(&identifier_with_context.identifier))
                }
            },
            // Type sources are only meaningful for expression-context identifiers (variables,
            // parameters). Other contexts like imports, type annotations, and decorators don't
            // have narrowing or first-use inference semantics.
            _ => return Vec::new(),
        };
        if !bindings.is_valid_key(&key) {
            return Vec::new();
        }
        let idx = bindings.key_to_idx(&key);
        let mut sources = Vec::new();
        if let Some(narrow_source) = narrow_source_for_key(&bindings, &module, idx) {
            sources.push(narrow_source);
        }
        if let Some(first_use_source) = first_use_source_for_key(&bindings, &module, &key, position)
        {
            sources.push(first_use_source);
        }
        sources
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use impl_::type_sources_for_hover;

#[cfg(target_arch = "wasm32")]
pub fn type_sources_for_hover(
    _transaction: &crate::state::state::Transaction<'_>,
    _handle: &pyrefly_build::handle::Handle,
    _position: ruff_text_size::TextSize,
) -> Vec<String> {
    Vec::new()
}
