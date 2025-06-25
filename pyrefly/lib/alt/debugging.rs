use dupe::Dupe;
use pyrefly_util::display::DisplayWithCtx;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::CalcId;
use crate::alt::answers::LookupAnswer;
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
/// but more convienient than `Display` implementations because they are easy to use
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
            bindings
                .idx_to_key(idx)
                .display_with(bindings.module_info())
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
        format!("{}", self.module_info().name())
    }

    fn show_any_idx_with(&self, bindings: &Bindings, idx: AnyIdx) -> String {
        let kind = idx.kind();
        let key = match idx {
            AnyIdx::Key(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyExpect(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyClass(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyTParams(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyClassField(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyVariance(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyClassSynthesizedFields(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyExport(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyFunction(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyAnnotation(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyClassMetadata(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyClassMro(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyLegacyTypeParam(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyYield(idx) => self.show_idx_with(bindings, idx),
            AnyIdx::KeyYieldFrom(idx) => self.show_idx_with(bindings, idx),
        };
        format!("{} :: {}", kind, key)
    }

    pub fn show_calc_id(&self, c: CalcId) -> String {
        match c {
            CalcId(bindings, idx) => {
                format!(
                    "{} . {}",
                    bindings.module_info().name(),
                    self.show_any_idx_with(&bindings, idx)
                )
            }
        }
    }

    pub fn show_current_idx(&self) -> String {
        match self.stack().peek() {
            None => {
                // In practice we'll never hit this debugging, but there's no need to panic if we do.
                "(None)".to_owned()
            }
            Some(c) => self.show_calc_id(c),
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
                AnyIdx::KeyClass(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyTParams(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassField(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyVariance(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyClassSynthesizedFields(idx) => {
                    self.show_binding_for_with(&bindings, idx)
                }
                AnyIdx::KeyExport(idx) => self.show_binding_for_with(&bindings, idx),
                AnyIdx::KeyFunction(idx) => self.show_binding_for_with(&bindings, idx),
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
            .map(|CalcId(bindings, idx)| {
                format!(
                    "{} . {}",
                    bindings.module_info().name(),
                    self.show_any_idx_with(&bindings, idx)
                )
            })
    }

    /// Return the current cycle, if we are at a (module, idx) that we've already seen in this thread.
    ///
    /// The answer will have the form
    /// - if there is no cycle, `None`
    /// - if there is a cycle, `Some(vec![(m0, i0), (m2, i2)...])`
    ///   where the order of (module, idx) pairs is recency (so starting with current
    ///   module and idx, and ending with the oldest).
    fn compute_current_cycle(&self) -> Option<Vec<CalcId>> {
        let mut rev_stack = self.stack().into_vec().into_iter().rev();
        let current = rev_stack.next()?;
        let mut cycle = Vec::with_capacity(rev_stack.len());
        cycle.push(current.dupe());
        for c in rev_stack {
            if c == current {
                return Some(cycle);
            }
            cycle.push(c);
        }
        None
    }

    // Get a printable representation of the current cycle.
    //
    // Panics if called when there is no cycle (should only be used to debug cycle breaking).
    pub fn show_current_cycle(&self) -> impl Iterator<Item = String> {
        self.compute_current_cycle()
            .unwrap()
            .into_iter()
            .map(|CalcId(bindings, idx)| {
                format!(
                    "{} . {}",
                    bindings.module_info().name(),
                    self.show_any_idx_with(&bindings, idx)
                )
            })
    }
}
