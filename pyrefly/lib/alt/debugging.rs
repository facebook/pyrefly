/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::display::DisplayWithCtx;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::CalcId;
use crate::binding::binding::AnyIdx;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::binding::Keyed;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::graph::index::Idx;

/// Debugging helpers for the AnswersSolver.
///
/// These are all string-returning functions, which make them potentially less efficient
/// but more convenient than `Display` implementations because they are easy to use
/// for string-based comparisons for filtered debugging.
///
/// For example, one useful snippet in unit tests is:
///   let debug = self.show_current_module() == "main";
///   if debug {
///      ... dump some information that would be too verbose if printed for stdlib modules ...
///   }
#[expect(dead_code)]
impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn show_idx<K>(&self, idx: Idx<K>) -> String
    where
        K: Keyed,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.show_idx_with(self.bindings(), idx)
    }

    pub fn show_idx_with<K>(&self, bindings: &Bindings, idx: Idx<K>) -> String
    where
        K: Keyed,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        format!(
            "{}",
            bindings.idx_to_key(idx).display_with(bindings.module())
        )
    }

    pub fn show_binding_generic<K>(&self, binding: &K::Value) -> String
    where
        K: Keyed,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        format!("{}", binding.display_with(self.bindings()))
    }

    pub fn show_binding(&self, binding: &Binding) -> String {
        self.show_binding_generic::<Key>(binding)
    }

    pub fn show_binding_for<K: Keyed>(&self, idx: Idx<K>) -> String
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.show_binding_for_with(self.bindings(), idx)
    }

    pub fn show_binding_for_with<K: Keyed>(&self, bindings: &Bindings, idx: Idx<K>) -> String
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        format!("{}", bindings.get(idx).display_with(bindings))
    }

    pub fn show_current_module(&self) -> String {
        format!("{}", self.module().name())
    }

    pub fn show_current_idx(&self) -> String {
        match self.stack().peek() {
            None => {
                // In practice we'll never hit this debugging, but there's no need to panic if we do.
                "(None)".to_owned()
            }
            Some(calc_id) => format!("{calc_id}"),
        }
    }

    pub fn show_current_binding(&self) -> String {
        match self.stack().peek() {
            None => {
                // In practice we'll never hit this debugging, but there's no need to panic if we do.
                "(None)".to_owned()
            }
            Some(CalcId(bindings, idx)) => match idx {
                AnyIdx::Key(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyExpect(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyConsistentOverrideCheck(idx) => {
                    self.show_binding_for_with(&bindings, idx)
                }
                AnyIdx::KeyClass(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyTParams(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassBaseType(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassField(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyVariance(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassSynthesizedFields(idx) => {
                    self.show_binding_for_with(&bindings, idx)
                }
                AnyIdx::KeyExport(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyDecoratedFunction(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyUndecoratedFunction(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyAnnotation(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassMetadata(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassMro(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyLegacyTypeParam(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyYield(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyYieldFrom(idx) => self.show_binding_for_with(&bindings, idx),
            },
        }
    }

    pub fn show_current_stack(&self) -> impl Iterator<Item = String> {
        self.stack()
            .into_vec()
            .into_iter()
            .map(|calc_id| format!("{calc_id}"))
    }

    // Get a printable representation of the current cycle.
    //
    // Panics if called when there is no cycle (should only be used to debug cycle breaking).
    pub fn show_current_cycle(&self) -> impl Iterator<Item = String> {
        self.stack()
            .current_cycle()
            .unwrap()
            .into_iter()
            .map(|c| format!("{c}"))
    }
}
