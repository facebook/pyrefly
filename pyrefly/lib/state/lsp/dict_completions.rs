/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::types::Union;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Key;
use crate::lsp::wasm::completion::RankedCompletion;
use crate::types::types::Type;

#[derive(Clone)]
enum DictKeyLiteralContext {
    /// A key literal used to access an existing dict/TypedDict.
    /// Examples: `cfg["na|"]`, `cfg.get("na|")`.
    KeyAccess {
        base_expr: Expr,
        literal: ExprStringLiteral,
    },
    /// A key literal inside a dict literal being constructed.
    /// Example: `{"na|": 1}`.
    DictLiteral {
        dict: ExprDict,
        literal: ExprStringLiteral,
    },
}

impl DictKeyLiteralContext {
    fn literal_range(&self) -> TextRange {
        match self {
            Self::KeyAccess { literal, .. } | Self::DictLiteral { literal, .. } => literal.range(),
        }
    }

    fn base_range(&self) -> TextRange {
        // For key access, we want the container expression's type.
        // For dict literals, we want the literal's contextual type (e.g. a TypedDict in
        // `cfg: Config = {"na|": 1}`), which is attached to the literal's range.
        match self {
            Self::KeyAccess { base_expr, .. } => base_expr.range(),
            Self::DictLiteral { dict, .. } => dict.range(),
        }
    }

    fn base_expr(&self) -> Option<&Expr> {
        match self {
            Self::KeyAccess { base_expr, .. } => Some(base_expr),
            Self::DictLiteral { .. } => None,
        }
    }
}

impl<'a> super::Transaction<'a> {
    fn type_contains_typed_dict(ty: &Type) -> bool {
        match ty {
            Type::TypedDict(_) | Type::PartialTypedDict(_) => true,
            Type::Union(box Union { members, .. }) => {
                members.iter().any(Self::type_contains_typed_dict)
            }
            _ => false,
        }
    }

    fn expr_has_typed_dict_type(&self, handle: &Handle, expr: &Expr) -> bool {
        self.get_type_trace(handle, expr.range())
            .map(|ty| Self::type_contains_typed_dict(&ty))
            .unwrap_or(false)
    }

    /// Extracts typed dict access from `.get()` method calls.
    /// This handles both `d.get("key")` and `d["key"]` patterns - the subscript
    /// case is handled in `dict_key_string_literal_at`.
    fn typed_dict_get_string_literal(
        &self,
        handle: &Handle,
        call: &ExprCall,
    ) -> Option<(Expr, ExprStringLiteral)> {
        let Expr::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        if attr.attr.id.as_str() != "get" {
            return None;
        }
        if !self.expr_has_typed_dict_type(handle, attr.value.as_ref()) {
            return None;
        }
        // If there's already a string literal, we want to provide completions
        // for the key name inside the quotes (e.g., `d.get("k|")` -> suggest "key")
        if let Some(Expr::StringLiteral(lit)) = call.arguments.args.first() {
            return Some((attr.value.as_ref().clone(), lit.clone()));
        }
        if let Some(lit) =
            call.arguments
                .keywords
                .iter()
                .find_map(|kw| match (&kw.arg, &kw.value) {
                    (Some(id), Expr::StringLiteral(lit)) if id.id.as_str() == "key" => Some(lit),
                    _ => None,
                })
        {
            return Some((attr.value.as_ref().clone(), lit.clone()));
        }
        None
    }

    fn dict_key_string_literal_at(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
    ) -> Option<(Expr, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let mut best: Option<(u8, TextSize, Expr, ExprStringLiteral)> = None;
        for node in nodes {
            let candidate = match node {
                AnyNodeRef::ExprSubscript(sub) => {
                    if let Expr::StringLiteral(lit) = sub.slice.as_ref() {
                        Some((sub.value.as_ref().clone(), lit.clone()))
                    } else {
                        None
                    }
                }
                AnyNodeRef::ExprCall(call) => self.typed_dict_get_string_literal(handle, call),
                _ => None,
            };
            let Some((base_expr, literal)) = candidate else {
                continue;
            };
            let (priority, dist) = Self::string_literal_priority(position, literal.range());
            let should_update = match &best {
                Some((best_prio, best_dist, _, _)) => {
                    priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                }
                None => true,
            };
            if should_update {
                best = Some((priority, dist, base_expr, literal));
                if priority == 0 && dist == TextSize::from(0) {
                    break;
                }
            }
        }
        best.map(|(_, _, base_expr, literal)| (base_expr, literal))
    }

    fn string_literal_priority(position: TextSize, range: TextRange) -> (u8, TextSize) {
        if range.contains(position) {
            (0, TextSize::from(0))
        } else if position < range.start() {
            (1, range.start() - position)
        } else {
            (2, position - range.end())
        }
    }

    fn dict_key_literal_context(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
    ) -> Option<DictKeyLiteralContext> {
        // Prefer direct key access (`d["k"]` / `d.get("k")`) so we can reuse the base
        // expression for facet-based completions. Fall back to dict literal keys.
        if let Some((base_expr, literal)) =
            self.dict_key_string_literal_at(handle, module, position)
        {
            Some(DictKeyLiteralContext::KeyAccess { base_expr, literal })
        } else {
            Self::dict_literal_string_literal_at(module, position)
                .map(|(dict, literal)| DictKeyLiteralContext::DictLiteral { dict, literal })
        }
    }

    fn dict_literal_string_literal_at(
        module: &ModModule,
        position: TextSize,
    ) -> Option<(ExprDict, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let mut best: Option<(u8, TextSize, ExprDict, ExprStringLiteral)> = None;
        for node in nodes {
            let AnyNodeRef::ExprDict(dict) = node else {
                continue;
            };
            let mut best_in_dict: Option<(u8, TextSize, ExprStringLiteral)> = None;
            for item in &dict.items {
                let Some(key_expr) = item.key.as_ref() else {
                    continue;
                };
                let Expr::StringLiteral(literal) = key_expr else {
                    continue;
                };
                let (priority, dist) = Self::string_literal_priority(position, literal.range());
                let should_update = match &best_in_dict {
                    Some((best_prio, best_dist, _)) => {
                        priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                    }
                    None => true,
                };
                if should_update {
                    best_in_dict = Some((priority, dist, literal.clone()));
                    if priority == 0 && dist == TextSize::from(0) {
                        break;
                    }
                }
            }
            let Some((priority, dist, literal)) = best_in_dict else {
                continue;
            };
            let should_update = match &best {
                Some((best_prio, best_dist, _, _)) => {
                    priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                }
                None => true,
            };
            if should_update {
                best = Some((priority, dist, dict.clone(), literal));
                if priority == 0 && dist == TextSize::from(0) {
                    break;
                }
            }
        }
        best.map(|(_, _, dict, literal)| (dict, literal))
    }

    fn expression_facets(expr: &Expr) -> Option<(Identifier, Vec<FacetKind>)> {
        let mut facets = Vec::new();
        let mut current = expr;
        loop {
            match current {
                Expr::Subscript(sub) => {
                    match sub.slice.as_ref() {
                        Expr::NumberLiteral(ExprNumberLiteral {
                            value: Number::Int(idx),
                            ..
                        }) if idx.as_usize().is_some() => {
                            facets.push(FacetKind::Index(idx.as_usize().unwrap()))
                        }
                        Expr::StringLiteral(lit) => {
                            facets.push(FacetKind::Key(lit.value.to_string()))
                        }
                        _ => return None,
                    }
                    current = sub.value.as_ref();
                }
                Expr::Attribute(attr) => {
                    facets.push(FacetKind::Attribute(attr.attr.id.clone()));
                    current = attr.value.as_ref();
                }
                Expr::Name(name) => {
                    facets.reverse();
                    return Some((Ast::expr_name_identifier(name.clone()), facets));
                }
                _ => return None,
            }
        }
    }

    fn collect_typed_dict_keys(
        &self,
        handle: &Handle,
        base_type: Type,
    ) -> Option<BTreeMap<String, Type>> {
        self.ad_hoc_solve(handle, "typed_dict_keys", |solver| {
            let mut map = BTreeMap::new();
            let mut stack = vec![base_type];
            while let Some(ty) = stack.pop() {
                match ty {
                    Type::TypedDict(td) | Type::PartialTypedDict(td) => {
                        for (name, field) in solver.type_order().typed_dict_fields(&td) {
                            map.entry(name.to_string())
                                .or_insert_with(|| field.ty.clone());
                        }
                    }
                    Type::Union(box Union { members, .. }) => {
                        stack.extend(members.into_iter());
                    }
                    _ => {}
                }
            }
            map
        })
    }

    pub(crate) fn add_dict_key_completions(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
        completions: &mut Vec<RankedCompletion>,
    ) {
        let Some(context) = self.dict_key_literal_context(handle, module, position) else {
            return;
        };
        let literal_range = context.literal_range();
        // Allow the cursor to sit a few characters before the literal (e.g. between nested
        // subscripts) so completion requests fired just before the quotes still succeed.
        let allowance = TextSize::from(4);
        let lower_bound = literal_range
            .start()
            .checked_sub(allowance)
            .unwrap_or_else(|| TextSize::new(0));
        if position < lower_bound || position > literal_range.end() {
            return;
        }
        let mut suggestions: BTreeMap<String, Option<Type>> = BTreeMap::new();

        if let Some(base_expr) = context.base_expr()
            && let Some(bindings) = self.get_bindings(handle)
        {
            let base_info = if let Some((identifier, facets)) = Self::expression_facets(base_expr) {
                Some((identifier, facets))
            } else if let Expr::Name(name) = &base_expr {
                Some((Ast::expr_name_identifier(name.clone()), Vec::new()))
            } else {
                None
            };

            if let Some((identifier, facets)) = base_info {
                let short_id = ShortIdentifier::new(&identifier);
                let idx_opt = {
                    let bound_key = Key::BoundName(short_id);
                    if bindings.is_valid_key(&bound_key) {
                        Some(bindings.key_to_idx(&bound_key))
                    } else {
                        let def_key = Key::Definition(short_id);
                        if bindings.is_valid_key(&def_key) {
                            Some(bindings.key_to_idx(&def_key))
                        } else {
                            None
                        }
                    }
                };

                if let Some(idx) = idx_opt {
                    let facets_clone = facets.clone();
                    if let Some(keys) = self.ad_hoc_solve(handle, "dict_key_facets", |solver| {
                        let info = solver.get_idx(idx);
                        info.key_facets_at(&facets_clone)
                    }) {
                        for (key, ty_opt) in keys {
                            suggestions.entry(key).or_insert(ty_opt);
                        }
                    }
                }
            }
        }

        // For key access we query the container expression; for literals we query the
        // literal itself to pick up contextual TypedDict typing from assignments.
        if let Some(base_type) = self.get_type_trace(handle, context.base_range())
            && let Some(typed_keys) = self.collect_typed_dict_keys(handle, base_type)
        {
            for (key, ty) in typed_keys {
                let entry = suggestions.entry(key).or_insert(None);
                if entry.is_none() {
                    *entry = Some(ty);
                }
            }
        }

        if suggestions.is_empty() {
            return;
        }

        for (label, ty_opt) in suggestions {
            let detail = ty_opt.as_ref().map(|ty| ty.to_string());
            completions.push(RankedCompletion::new(CompletionItem {
                label,
                detail,
                kind: Some(CompletionItemKind::FIELD),
                ..Default::default()
            }));
        }
    }
}
