/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;

use itertools::Either;
use itertools::Itertools;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::Flow;
use crate::binding::scope::FlowInfo;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::Loop;
use crate::binding::scope::LoopExit;
use crate::config::error_kind::ErrorKind;
use crate::error::context::ErrorInfo;
use crate::graph::index::Idx;

// Represents a name we need to handle when merging flows.
struct MergeItem {
    // The key at which we will bind the result of the merge. Unlike all other keys
    // in our data structure, this one does not refer to a pre-existing binding coming
    // from upstream but rather the *output* of the merge.
    phi_idx: Idx<Key>,
    // Key of the default binding. This is only used in loops, where it is used
    // to say that when in doubt, the loop recursive Phi should solve to either
    // the binding lives at the top of the loop (if any) or the first assignment
    // in the first branch we encountered.
    default: Idx<Key>,
    // The set of bindings live at the end of each branch. This will not include
    // `merged_key` itself (which might be live if a branch of a loop does not
    // modify anything).
    branch_idxs: SmallSet<Idx<Key>>,
    // The flow styles from each branch in the merge
    flow_styles: Vec<FlowStyle>,
}

impl MergeItem {
    fn new(
        name: Name,
        range: TextRange,
        info: FlowInfo,
        visible_branches_len: usize,
        idx_for_promise: impl FnOnce(Key) -> Idx<Key>,
    ) -> Self {
        // We are promising to bind this key at the end of the merge (see `merged_flow_info`).
        //
        // Note that in loops, the speculative phi logic may have already inserted this key,
        // in which case `idx_for_promise` will just give us back the idx we aready created.
        let phi_idx = idx_for_promise(Key::Phi(name, range));
        let mut myself = Self {
            phi_idx,
            default: info.default,
            branch_idxs: SmallSet::new(),
            flow_styles: Vec::with_capacity(visible_branches_len),
        };
        myself.add_branch(info);
        myself
    }

    /// Add the flow info at the end of a branch to our merge item.
    fn add_branch(&mut self, info: FlowInfo) {
        if info.key != self.phi_idx {
            // Optimization: instead of x = phi(x, ...), we can skip the x.
            // Avoids a recursive solving step later.
            self.branch_idxs.insert(info.key);
        }
        self.flow_styles.push(info.style);
    }

    /// Get the flow info for an item in the merged flow, which is a combination
    /// of the `phi_key` that will have the merged type information and the merged
    /// flow styles.
    ///
    /// The binding for the phi key is typically a Phi, but if this merge is from a loop
    /// we'll wrap that in a Default, and if all branches were the same we'll
    /// just use a Forward instead.
    ///
    /// The default value will depend on whether we are still in a loop after the
    /// current merge. If so, we preserve the existing default; if not, the
    /// merged phi is the new default used for downstream loops.
    fn merged_flow_info(
        self,
        current_is_loop: bool,
        contained_in_loop: bool,
        insert_binding_idx: impl FnOnce(Idx<Key>, Binding),
    ) -> FlowInfo {
        insert_binding_idx(
            self.phi_idx,
            match () {
                _ if self.branch_idxs.len() == 1 => {
                    Binding::Forward(self.branch_idxs.into_iter().next().unwrap())
                }
                _ if current_is_loop => {
                    Binding::Default(self.default, Box::new(Binding::Phi(self.branch_idxs)))
                }
                _ => Binding::Phi(self.branch_idxs),
            },
        );
        FlowInfo {
            key: self.phi_idx,
            default: if contained_in_loop {
                self.default
            } else {
                self.phi_idx
            },
            style: FlowStyle::merged(self.flow_styles),
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    fn merge_flow(&mut self, mut xs: Vec<Flow>, range: TextRange, is_loop: bool) -> Flow {
        if xs.len() == 1 && xs[0].has_terminated {
            return xs.pop().unwrap();
        }

        // Hidden branches are branches where control flow terminates; visible ones are those
        // that appear to flow into the merge.
        //
        // We normally only merge the visible branches, but if nothing is visible no one is going to
        // fill in the Phi keys we promised. So just give up and use the hidden branches instead.
        let (hidden_branches, mut visible_branches): (Vec<_>, Vec<_>) =
            xs.into_iter().partition(|x| x.has_terminated);
        let no_next = visible_branches.is_empty();
        if visible_branches.is_empty() {
            visible_branches = hidden_branches;
        }

        // Collect all the branches into a `MergeItem` per name we need to merge
        let mut merge_items: SmallMap<Name, MergeItem> =
            SmallMap::with_capacity(visible_branches.first().map_or(0, |x| x.info.len()));
        let visible_branches_len = visible_branches.len();
        for flow in visible_branches {
            for (name, info) in flow.info.into_iter_hashed() {
                match merge_items.entry_hashed(name) {
                    Entry::Occupied(mut merge_item_entry) => {
                        merge_item_entry.get_mut().add_branch(info)
                    }
                    Entry::Vacant(e) => {
                        let name = e.key().clone();
                        e.insert(MergeItem::new(
                            name,
                            range,
                            info,
                            visible_branches_len,
                            |key| self.idx_for_promise(key),
                        ));
                    }
                };
            }
        }

        // For each name and merge item, produce the merged FlowInfo for our new Flow
        let mut res = SmallMap::with_capacity(merge_items.len());
        for (name, merge_item) in merge_items.into_iter_hashed() {
            res.insert_hashed(
                name,
                merge_item.merged_flow_info(is_loop, self.scopes.loop_depth() > 0, |key, value| {
                    self.insert_binding_idx(key, value);
                }),
            );
        }
        Flow {
            info: res,
            has_terminated: no_next,
        }
    }

    fn merge_into_current(&mut self, mut branches: Vec<Flow>, range: TextRange, is_loop: bool) {
        branches.push(mem::take(&mut self.scopes.current_mut().flow));
        self.scopes.current_mut().flow = self.merge_flow(branches, range, is_loop);
    }

    fn merge_loop_into_current(&mut self, branches: Vec<Flow>, range: TextRange) {
        self.merge_into_current(branches, range, true);
    }

    pub fn merge_branches_into_current(&mut self, branches: Vec<Flow>, range: TextRange) {
        self.merge_into_current(branches, range, false);
    }

    pub fn set_current_flow_to_merged_branches(&mut self, branches: Vec<Flow>, range: TextRange) {
        let flow = self.merge_flow(branches, range, false);
        self.scopes.replace_current_flow(flow);
    }

    /// Helper for loops, inserts a phi key for every name in the given flow.
    fn insert_phi_keys(&mut self, mut flow: Flow, range: TextRange) -> Flow {
        for (name, info) in flow.info.iter_mut() {
            // The promise is that we will insert a Phi binding when the control flow merges.
            info.key = self.idx_for_promise(Key::Phi(name.clone(), range));
        }
        flow
    }

    pub fn setup_loop(&mut self, range: TextRange, narrow_ops: &NarrowOps) {
        let base = mem::take(&mut self.scopes.current_mut().flow);
        // To account for possible assignments to existing names in a loop, we
        // speculatively insert phi keys upfront.
        self.scopes.current_mut().flow = self.insert_phi_keys(base.clone(), range);
        self.scopes
            .current_mut()
            .loops
            .push(Loop(vec![(LoopExit::NeverRan, base)]));
        self.bind_narrow_ops(narrow_ops, range);
    }

    pub fn teardown_loop(&mut self, range: TextRange, narrow_ops: &NarrowOps, orelse: Vec<Stmt>) {
        let done = self.scopes.finish_current_loop();
        let (breaks, other_exits): (Vec<Flow>, Vec<Flow>) =
            done.0.into_iter().partition_map(|(exit, flow)| match exit {
                LoopExit::Break => Either::Left(flow),
                LoopExit::NeverRan | LoopExit::Continue => Either::Right(flow),
            });
        // We associate a range to the non-`break` exits from the loop; it doesn't matter much what
        // it is as long as it's different from the loop's range.
        let other_range = TextRange::new(range.start(), range.start());
        if breaks.is_empty() {
            // When there are no `break`s, the loop condition is always false once the body has exited,
            // and any `orelse` always runs.
            self.merge_loop_into_current(other_exits, range);
            self.bind_narrow_ops(&narrow_ops.negate(), other_range);
            self.stmts(orelse);
        } else {
            // Otherwise, we negate the loop condition and run the `orelse` only when we don't `break`.
            self.merge_loop_into_current(other_exits, range);
            self.bind_narrow_ops(&narrow_ops.negate(), other_range);
            self.stmts(orelse);
            self.merge_loop_into_current(breaks, other_range);
        }
    }

    pub fn add_loop_exitpoint(&mut self, exit: LoopExit, range: TextRange) {
        let in_loop = self.scopes.add_loop_exitpoint(exit);
        if !in_loop {
            // Python treats break and continue outside of a loop as a syntax error.
            self.error(
                range,
                ErrorInfo::Kind(ErrorKind::ParseError),
                format!("Cannot `{exit}` outside loop"),
            );
        }
    }
}
