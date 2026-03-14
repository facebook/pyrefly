/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Expression walker for the CinderX type report.
//!
//! Walks every expression in a module's AST, looks up the inferred type,
//! and builds a deduplicated `TypeTable` plus a list of `LocatedType` entries.

use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::Class;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::type_info::TypeInfo;
use pyrefly_types::types::Type;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::alt::answers::Answers;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::module::module_info::ModuleInfo;
use crate::report::cinderx::convert::type_to_structured;
use crate::report::cinderx::types::LocatedType;
use crate::report::cinderx::types::TypeTable;
use crate::report::cinderx::types::TypeTableEntry;
use crate::state::state::Transaction;

/// Collected per-module type data for the CinderX report.
pub(crate) struct ModuleTypeData {
    /// Deduplicated type table entries.
    pub entries: Vec<TypeTableEntry>,
    /// Per-expression located type references into the table.
    pub locations: Vec<LocatedType>,
    /// All classes encountered during type conversion, for MRO collection.
    pub classes: Vec<Class>,
}

/// Collect per-expression types for a single module.
///
/// Walks every expression in the AST, looks up the inferred type via
/// `Answers`, converts it to a `StructuredType` entry in a shared
/// `TypeTable`, and records a `LocatedType` mapping source location
/// to table index.
///
/// Returns `None` if any of the required data (AST, answers, bindings,
/// module info) is unavailable for the given handle.
pub(crate) fn collect_module_types(
    transaction: &Transaction,
    handle: &Handle,
) -> Option<ModuleTypeData> {
    let ast = transaction.get_ast(handle)?;
    let module_info = transaction.get_module_info(handle)?;
    let answers = transaction.get_answers(handle)?;
    let bindings = transaction.get_bindings(handle)?;

    let mut table = TypeTable::new();
    let mut locations = Vec::new();
    let mut pending_class_traits: Vec<(usize, Class)> = Vec::new();

    walk_expressions(
        &ast,
        &module_info,
        &answers,
        &bindings,
        transaction,
        handle,
        &mut table,
        &mut locations,
        &mut pending_class_traits,
    );

    // Extract unique classes from pending_class_traits for MRO collection.
    let classes: Vec<Class> = pending_class_traits
        .into_iter()
        .map(|(_, cls)| cls)
        .collect();

    Some(ModuleTypeData {
        entries: table.into_entries(),
        locations,
        classes,
    })
}

/// Walk all expressions in the AST and collect types.
///
/// For `Expr::Attribute` or `Expr::Subscript` chains rooted at an `Expr::Name`
/// (e.g. `x.foo`, `x.foo.bar`, `x[0]`, `x["key"].bar`), also detects facet
/// narrows: if any level in the chain has a narrowed facet, re-resolves the
/// full chain on the unnarrowed base type and records the result so CinderX
/// can distinguish sound from unsound narrows.
fn walk_expressions(
    ast: &Arc<ModModule>,
    module_info: &ModuleInfo,
    answers: &Answers,
    bindings: &Bindings,
    transaction: &Transaction,
    handle: &Handle,
    table: &mut TypeTable,
    locations: &mut Vec<LocatedType>,
    pending_class_traits: &mut Vec<(usize, Class)>,
) {
    /// Try to find a binding key for a name expression.
    ///
    /// Checks `BoundName` (use/load site) first, then `Definition`
    /// (store site), returning `None` if neither exists in bindings.
    fn try_find_key_for_name(name: &ExprName, bindings: &Bindings) -> Option<Key> {
        let key = Key::BoundName(ShortIdentifier::expr_name(name));
        if bindings.is_valid_key(&key) {
            return Some(key);
        }
        let key = Key::Definition(ShortIdentifier::expr_name(name));
        if bindings.is_valid_key(&key) {
            return Some(key);
        }
        None
    }

    /// Look up the inferred type for an expression.
    ///
    /// For `Expr::Name` nodes, looks up via the binding key for precise
    /// results. For all other expressions, falls back to the trace map
    /// which is populated during solving.
    fn lookup_type(x: &Expr, answers: &Answers, bindings: &Bindings) -> Option<Type> {
        if let Expr::Name(name) = x
            && let Some(key) = try_find_key_for_name(name, bindings)
            && let Some(ty) = answers.get_type_at(bindings.key_to_idx(&key))
        {
            return Some(ty);
        }
        answers.get_type_trace(x.range())
    }

    /// Walk an expression chain of `Expr::Attribute` and `Expr::Subscript`
    /// nodes to find the root `Expr::Name` and collect `FacetKind`s from
    /// root to leaf.
    ///
    /// Handles all three facet kinds:
    /// - `Attribute(name)` from `x.foo`
    /// - `Index(n)` from `x[0]` (integer literal subscript)
    /// - `Key(s)` from `x["key"]` (string literal subscript)
    ///
    /// Returns `None` if the chain doesn't root at an `Expr::Name`
    /// (e.g. `f().attr`) or contains a non-literal subscript.
    fn extract_facet_chain(expr: &Expr) -> Option<(&ExprName, Vec<FacetKind>)> {
        // Must start with Attribute or Subscript to have a facet chain.
        if !matches!(expr, Expr::Attribute(_) | Expr::Subscript(_)) {
            return None;
        }
        let mut chain = Vec::new();
        let mut current = expr;
        loop {
            match current {
                Expr::Attribute(attr) => {
                    chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    current = attr.value.as_ref();
                }
                Expr::Subscript(sub) => {
                    match sub.slice.as_ref() {
                        Expr::NumberLiteral(ExprNumberLiteral {
                            value: Number::Int(idx),
                            ..
                        }) if idx.as_i64().is_some() => {
                            chain.push(FacetKind::Index(idx.as_i64().unwrap()));
                        }
                        Expr::StringLiteral(lit) => {
                            chain.push(FacetKind::Key(lit.value.to_string()));
                        }
                        _ => return None,
                    }
                    current = sub.value.as_ref();
                }
                Expr::Name(name) => {
                    chain.reverse();
                    return Some((name, chain));
                }
                _ => return None,
            }
        }
    }

    /// Check whether any level in a facet chain has a facet narrow.
    ///
    /// Walks the facet tree level by level using `type_at_facet` to check
    /// for narrows and `at_facet` to descend. Returns `true` as soon as
    /// any level has a narrowed type.
    fn has_facet_narrow_in_chain(type_info: &TypeInfo, chain: &[FacetKind]) -> bool {
        let mut current = type_info.clone();
        for facet in chain {
            if current.type_at_facet(facet).is_some() {
                return true;
            }
            current = current.at_facet(facet, Type::never);
        }
        false
    }

    /// Recursive expression visitor: looks up type, converts to structured
    /// form, records location, then recurses into child expressions.
    ///
    /// For attribute/subscript chains rooted at a name (`x.attr`, `x[0]`,
    /// `x["key"].bar`, etc.), checks whether any level in the chain has a
    /// facet narrow. If so, re-resolves the full chain on the unnarrowed
    /// base type to populate `unnarrowed_type` and `is_narrowed_mismatch`
    /// on the `LocatedType`.
    fn visit_expr(
        x: &Expr,
        parent: Option<&Expr>,
        module_info: &ModuleInfo,
        answers: &Answers,
        bindings: &Bindings,
        transaction: &Transaction,
        handle: &Handle,
        table: &mut TypeTable,
        locations: &mut Vec<LocatedType>,
        pending_class_traits: &mut Vec<(usize, Class)>,
    ) {
        if let Some(ty) = lookup_type(x, answers, bindings) {
            let range = x.range();
            let location = module_info
                .lined_buffer()
                .python_ast_range_for_expr(range, x, parent);
            let type_index = type_to_structured(&ty, table, pending_class_traits);

            // Detect facet narrows on attribute/subscript chains (x.attr, x[0], etc.).
            // When any level in the chain has a facet narrow, re-resolve the full
            // chain on the unnarrowed base type so CinderX can handle the unsound
            // narrow appropriately.
            let (unnarrowed_type, is_narrowed_mismatch) = if let Some((name, chain)) =
                extract_facet_chain(x)
                && let Some(key) = try_find_key_for_name(name, bindings)
                && let Some(type_info) = answers.get_idx(bindings.key_to_idx(&key))
                && type_info.has_facets()
                && has_facet_narrow_in_chain(&type_info, &chain)
            {
                // Some level in the chain has a facet narrow.
                // Re-resolve the full chain on the unnarrowed base type.
                let base_type = type_info.ty().clone();
                let unnarrowed_ty =
                    transaction.ad_hoc_solve(handle, "cinderx_unnarrow", |solver| {
                        let errors = solver.error_swallower();
                        let mut current_ty = base_type;
                        for facet in &chain {
                            current_ty = match facet {
                                FacetKind::Attribute(name) => solver.attr_infer_for_type(
                                    &current_ty,
                                    name,
                                    range,
                                    &errors,
                                    None,
                                ),
                                FacetKind::Index(idx) => {
                                    let synth = Expr::NumberLiteral(ExprNumberLiteral {
                                        node_index: AtomicNodeIndex::default(),
                                        range: TextRange::empty(TextSize::from(0)),
                                        value: Number::Int(Int::from(*idx as u64)),
                                    });
                                    solver.subscript_infer_for_type(
                                        &current_ty,
                                        &synth,
                                        range,
                                        &errors,
                                    )
                                }
                                FacetKind::Key(key) => {
                                    let synth =
                                        Ast::str_expr(key, TextRange::empty(TextSize::from(0)));
                                    solver.subscript_infer_for_type(
                                        &current_ty,
                                        &synth,
                                        range,
                                        &errors,
                                    )
                                }
                            };
                        }
                        current_ty
                    });
                match unnarrowed_ty {
                    Some(unnarrowed_ty) => {
                        let unnarrowed_idx =
                            type_to_structured(&unnarrowed_ty, table, pending_class_traits);
                        let is_mismatch =
                            table.hash_at(type_index) != table.hash_at(unnarrowed_idx);
                        (Some(unnarrowed_idx), is_mismatch)
                    }
                    // ad_hoc_solve returned None (module data unavailable); degrade gracefully.
                    None => (None, false),
                }
            } else {
                (None, false)
            };

            locations.push(LocatedType {
                location,
                type_index,
                unnarrowed_type,
                is_narrowed_mismatch,
            });
        }

        x.recurse(&mut |child| {
            visit_expr(
                child,
                Some(x),
                module_info,
                answers,
                bindings,
                transaction,
                handle,
                table,
                locations,
                pending_class_traits,
            )
        });
    }

    ast.visit(&mut |x: &Expr| {
        visit_expr(
            x,
            None,
            module_info,
            answers,
            bindings,
            transaction,
            handle,
            table,
            locations,
            pending_class_traits,
        )
    });
}
