/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Iterative Tarjan's algorithm for finding strongly connected components (SCCs).
//!
//! Key insight: During DFS, a node is the root of an SCC iff its `lowlink` (the smallest
//! index reachable by following edges) equals its own `index` after visiting all descendants.
//! This means no node in its subtree can reach an ancestor outside the SCC.
//!
//! See https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

use std::hash::Hash;
use std::ops::Index;
use std::ops::IndexMut;

use append_only_vec::AppendOnlyVec;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use index_vec::IndexVec;
use index_vec::define_index_type;

pub struct Tarjan<T: Eq + Hash> {
    /// Hash table mapping node keys to their indices in `nodes`.
    index: HashTable<NodeIdx>,
    /// Storage for all nodes. Each entry contains the node's key and metadata.
    nodes: NodeVec<T>,
    /// Stack of nodes in the current DFS path. Nodes remain on the stack until their
    /// SCC is identified and popped. Being on the stack means the node is an ancestor
    /// of the current node and could form a cycle.
    stack: IndexVec<StackIdx, NodeIdx>,
    /// Used to assign `index` on nodes, the depth-first visit order.
    count: u32,
    /// Stack of remaining work. See `WorkItem`.
    work: Vec<WorkItem>,
    /// Discovered components, in toposort order.
    sccs: IndexVec<SccIdx, Box<[NodeIdx]>>,
}

impl<T: Eq + Hash> Tarjan<T> {
    pub fn new() -> Self {
        Self {
            index: HashTable::new(),
            nodes: NodeVec::new(),
            stack: IndexVec::new(),
            count: 0,
            work: Vec::new(),
            sccs: IndexVec::new(),
        }
    }

    pub fn iter_scc(&self, scc: SccIdx) -> impl Iterator<Item = &T> {
        self.sccs[scc].iter().map(|&h| &self.nodes[h].0)
    }

    /// Find the strongly connected component containing the given node.
    ///
    /// Returns a slice containing all nodes in the same SCC as the input node.
    /// The same `Tarjan` instance can be reused for multiple queries, and previously
    /// computed SCCs are cached.
    pub fn root(&mut self, root_key: T, visit: &impl Fn(&T, &mut dyn FnMut(T))) -> SccIdx {
        assert!(self.work.is_empty());

        let hash = Self::hash(&root_key);

        // Insert root. `curr_idx` tracks the current node's position on the stack as we traverse the graph.
        let mut curr_idx = match self.index.entry(
            hash,
            |idx| self.nodes[*idx].0 == root_key,
            Self::hasher(&self.nodes),
        ) {
            Entry::Occupied(entry) => {
                let node = self.nodes[*entry.get()].1.unwrap();
                return node.scc.unwrap();
            }
            Entry::Vacant(entry) => {
                let node = TarjanNode::new(&mut self.count);
                let node_idx = self.nodes.push((root_key, Some(node)));
                entry.insert(node_idx);
                self.visit(node_idx, None, visit)
            }
        };

        // Store the root node's index to get back to it after visiting all edges.
        let root_node_idx = self.stack[curr_idx];

        while let Some(item) = self.work.pop() {
            match item {
                WorkItem::Node(prev) => {
                    // After visiting all edges from this node, check for SCC.
                    let curr_node = *self.get_node(curr_idx).unwrap();

                    if curr_node.index == curr_node.lowlink {
                        // Current node is the root of an SCC: no descendant can reach an ancestor
                        // of this node, so all nodes from `curr_idx` to the top of the stack form the SCC.
                        let scc = self.stack.split_off(curr_idx).raw.into_boxed_slice();
                        let idx = self.sccs.push(scc);
                        for h in &self.sccs[idx] {
                            let n = self.nodes[*h].1.as_mut().unwrap();
                            n.scc = Some(idx);
                        }
                    }

                    // Update parent's lowlink with our final lowlink
                    if let Some(prev_idx) = prev {
                        let prev_node = self.get_node_mut(prev_idx).unwrap();
                        prev_node.lowlink = prev_node.lowlink.min(curr_node.lowlink);
                        curr_idx = prev_idx; // backtrack
                    }
                }
                WorkItem::Edge(succ_idx) => {
                    let succ_node_opt = &mut self.nodes[succ_idx].1;
                    match succ_node_opt {
                        None => {
                            // New node - insert and visit
                            *succ_node_opt = Some(TarjanNode::new(&mut self.count));
                            curr_idx = self.visit(succ_idx, Some(curr_idx), visit);
                        }
                        Some(succ_node) => {
                            let succ_index = succ_node.index;
                            if succ_node.on_stack() {
                                // Back edge to node still on stack
                                let curr_node = self.get_node_mut(curr_idx).unwrap();
                                curr_node.lowlink = curr_node.lowlink.min(succ_index)
                            }
                            // If successor is not on stack, it's a cross edge to an already-processed SCC.
                        }
                    }
                }
            }
        }

        let root_node = self.nodes[root_node_idx].1.unwrap();
        root_node.scc.unwrap()
    }

    // Push node onto the stack and return the stack index.
    fn visit(
        &mut self,
        node_idx: NodeIdx,
        prev: Option<StackIdx>,
        visit: &impl Fn(&T, &mut dyn FnMut(T)),
    ) -> StackIdx {
        let stack_idx = self.stack.push(node_idx);
        self.work.push(WorkItem::Node(prev));

        visit(&self.nodes[node_idx].0, &mut |key| {
            let hash = Self::hash(&key);
            let entry = self
                .index
                .entry(
                    hash,
                    |idx| self.nodes[*idx].0 == key,
                    Self::hasher(&self.nodes),
                )
                .or_insert_with(|| self.nodes.push((key, None)));
            self.work.push(WorkItem::Edge(*entry.get()))
        });

        stack_idx
    }

    fn get_node(&self, idx: StackIdx) -> Option<&TarjanNode> {
        self.nodes[self.stack[idx]].1.as_ref()
    }

    fn get_node_mut(&mut self, idx: StackIdx) -> Option<&mut TarjanNode> {
        self.nodes[self.stack[idx]].1.as_mut()
    }

    fn hash(key: &T) -> u64 {
        fxhash::hash64(key)
    }

    fn hasher(nodes: &NodeVec<T>) -> impl Fn(&NodeIdx) -> u64 {
        |idx: &NodeIdx| Self::hash(&nodes[*idx].0)
    }
}

define_index_type! {
    struct NodeIdx = u32;
}

define_index_type! {
    struct StackIdx = u32;
}

define_index_type! {
    pub struct SccIdx = u32;
}

/// Store nodes using AppendOnlyVec so that references remain stable as new nodes
/// are added during traversal.
struct NodeVec<T>(AppendOnlyVec<(T, Option<TarjanNode>)>);

impl<T> NodeVec<T> {
    fn new() -> Self {
        Self(AppendOnlyVec::new())
    }

    fn push(&self, value: (T, Option<TarjanNode>)) -> NodeIdx {
        NodeIdx::new(self.0.push(value))
    }
}

impl<T> Index<NodeIdx> for NodeVec<T> {
    type Output = (T, Option<TarjanNode>);

    fn index(&self, idx: NodeIdx) -> &Self::Output {
        &self.0[idx.index()]
    }
}

impl<T> IndexMut<NodeIdx> for NodeVec<T> {
    fn index_mut(&mut self, idx: NodeIdx) -> &mut Self::Output {
        &mut self.0[idx.index()]
    }
}

/// Metadata for a node in Tarjan's algorithm.
#[derive(Copy, Clone)]
struct TarjanNode {
    /// Order that this node was visited. Unique.
    index: u32,
    /// Smallest index of any node in the stack reachable from this node.
    lowlink: u32,
    /// Index into `sccs` vec, assigned when we pop this node from the stack.
    /// If Some(_), implies that this node is no longer on the stack.
    scc: Option<SccIdx>,
}

impl TarjanNode {
    fn new(count: &mut u32) -> Self {
        let index = *count;
        *count += 1;
        Self {
            index,
            lowlink: index,
            scc: None,
        }
    }

    /// Returns true if this node is on the DFS stack (i.e., is an ancestor of the
    /// current node). Back-edges to on-stack nodes indicate cycles and affect lowlink.
    fn on_stack(&self) -> bool {
        self.scc.is_none()
    }
}

/// Work items for the iterative DFS traversal.
enum WorkItem {
    /// Signal to backtrack after we have visited all edges from this node.
    /// Argument is the stack index of the previous node, or None for a root node.
    Node(Option<StackIdx>),
    /// Edge from the current node to node T.
    Edge(NodeIdx),
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn test_single_node() {
        let mut tarjan = Tarjan::new();
        let scc = tarjan.root(1, &|_, _| {});
        let scc: Vec<_> = tarjan.iter_scc(scc).copied().collect();
        assert_eq!(scc, vec![1]);
    }

    #[test]
    fn test_simple_cycle() {
        // 1 -> 2 -> 1 (cycle)
        let mut tarjan = Tarjan::new();
        let scc = tarjan.root(1, &|h, edge| match h {
            1 => edge(2),
            2 => edge(1),
            _ => {}
        });
        // Both nodes should be in the same SCC
        let scc: Vec<_> = tarjan.iter_scc(scc).copied().collect();
        assert_eq!(scc.len(), 2);
        assert!(scc.contains(&1));
        assert!(scc.contains(&2));
    }

    #[test]
    fn test_no_cycle() {
        // 1 -> 2 -> 3 (no cycle)
        let mut tarjan = Tarjan::new();
        let scc = tarjan.root(1, &|h, edge| match h {
            1 => edge(2),
            2 => edge(3),
            _ => {}
        });
        // Node 1 should be in its own SCC
        let scc: Vec<_> = tarjan.iter_scc(scc).copied().collect();
        assert_eq!(scc, vec![1]);
    }

    #[test]
    fn test_reuse_cache() {
        // 0 -> 1 -> 0 (cycle)
        let edges = [1, 0];
        let visits = [Cell::new(0), Cell::new(0)];
        let visit = |h: &usize, edge: &mut dyn FnMut(usize)| {
            let v = &visits[*h];
            v.set(v.get() + 1);
            edge(edges[*h]);
        };

        let mut tarjan = Tarjan::new();

        // First query - should visit both nodes once
        tarjan.root(0, &visit);
        assert_eq!(visits[0].get(), 1);
        assert_eq!(visits[1].get(), 1);

        // Query again - should use cached result, no additional visits
        tarjan.root(0, &visit);
        assert_eq!(visits[0].get(), 1);
        assert_eq!(visits[1].get(), 1);

        // Query the other node in the cycle - also cached, no additional visits
        tarjan.root(1, &visit);
        assert_eq!(visits[0].get(), 1);
        assert_eq!(visits[1].get(), 1);
    }

    #[test]
    fn test_nested_cycles_sharing_node() {
        // Graph where cycles share nodes, creating a larger SCC:
        //
        //     1 → 2 → 3
        //     ↑       ↓
        //     └───────┘
        //         ↓
        //     4 → 5 → 6
        //     ↑       ↓
        //     └───────┘
        //
        // But with an edge from 3 → 4 and 6 → 1, creating one big SCC
        //
        // All nodes {1,2,3,4,5,6} should be in the same SCC

        let visit = |h: &i32, edge: &mut dyn FnMut(i32)| match h {
            1 => edge(2),
            2 => edge(3),
            3 => {
                edge(1); // Back to cycle 1
                edge(4); // Into cycle 2
            }
            4 => edge(5),
            5 => edge(6),
            6 => {
                edge(4); // Back to cycle 2
                edge(1); // Back to cycle 1, connecting both cycles
            }
            _ => {}
        };

        let mut tarjan = Tarjan::new();
        let scc_idx = tarjan.root(1, &visit);
        let scc: Vec<_> = tarjan.iter_scc(scc_idx).copied().collect();

        // All 6 nodes should be in one SCC due to the interconnection
        assert_eq!(scc.len(), 6);
        for i in 1..=6 {
            assert!(scc.contains(&i), "SCC should contain node {}", i);
        }
    }

    #[test]
    fn test_diamond_with_cycle_at_bottom() {
        // Diamond pattern with a cycle at the bottom:
        //
        //       1
        //      / \
        //     2   3
        //      \ /
        //       4 ←→ 5 (cycle)
        //
        // SCCs: {1}, {2}, {3}, {4, 5}
        //
        // This tests that cross edges from 2 and 3 to the {4,5} SCC
        // correctly use on_stack=false after the SCC is formed.

        let visit = |h: &i32, edge: &mut dyn FnMut(i32)| match h {
            1 => {
                edge(2);
                edge(3);
            }
            2 => edge(4),
            3 => edge(4),
            4 => edge(5),
            5 => edge(4),
            _ => {}
        };

        let mut tarjan = Tarjan::new();

        // Query from 1
        let scc_1_idx = tarjan.root(1, &visit);
        let scc_1: Vec<_> = tarjan.iter_scc(scc_1_idx).copied().collect();
        assert_eq!(scc_1, vec![1]);

        // Verify the cycle SCC
        let scc_4_idx = tarjan.root(4, &visit);
        let scc_4: Vec<_> = tarjan.iter_scc(scc_4_idx).copied().collect();
        assert_eq!(scc_4.len(), 2);
        assert!(scc_4.contains(&4));
        assert!(scc_4.contains(&5));

        // Verify singletons
        let scc_2_idx = tarjan.root(2, &visit);
        let scc_2: Vec<_> = tarjan.iter_scc(scc_2_idx).copied().collect();
        assert_eq!(scc_2, vec![2]);

        let scc_3_idx = tarjan.root(3, &visit);
        let scc_3: Vec<_> = tarjan.iter_scc(scc_3_idx).copied().collect();
        assert_eq!(scc_3, vec![3]);
    }

    #[test]
    fn test_long_chain_with_back_edge() {
        // Long chain with a back edge from the end to the beginning:
        // 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 1
        //
        // All nodes should be in one SCC.

        let visit = |h: &i32, edge: &mut dyn FnMut(i32)| {
            if *h < 10 {
                edge(h + 1);
            } else {
                edge(1); // Back edge to start
            }
        };

        let mut tarjan = Tarjan::new();
        let scc_idx = tarjan.root(1, &visit);
        let scc: Vec<_> = tarjan.iter_scc(scc_idx).copied().collect();

        assert_eq!(scc.len(), 10);
        for i in 1..=10 {
            assert!(scc.contains(&i));
        }
    }

    #[test]
    fn test_multiple_back_edges_in_cycle() {
        // Cycle with multiple back edges:
        //
        // 1 → 2 → 3 → 4 → 5
        // ↑   ↑   ↑   ↑   ↓
        // └───┴───┴───┴───┘
        //
        // Node 5 has edges back to 1, 2, 3, and 4
        // All should be in one SCC.

        let visit = |h: &i32, edge: &mut dyn FnMut(i32)| match h {
            1 => edge(2),
            2 => edge(3),
            3 => edge(4),
            4 => edge(5),
            5 => {
                edge(1);
                edge(2);
                edge(3);
                edge(4);
            }
            _ => {}
        };

        let mut tarjan = Tarjan::new();
        let scc_idx = tarjan.root(1, &visit);
        let scc: Vec<_> = tarjan.iter_scc(scc_idx).copied().collect();

        assert_eq!(scc.len(), 5);
        for i in 1..=5 {
            assert!(scc.contains(&i));
        }
    }

    #[test]
    fn test_self_loop() {
        // Node with edge to itself
        // 1 → 1

        let mut tarjan = Tarjan::new();
        let scc_idx = tarjan.root(1, &|h, edge| {
            if *h == 1 {
                edge(1);
            }
        });
        let scc: Vec<_> = tarjan.iter_scc(scc_idx).copied().collect();

        assert_eq!(scc, vec![1]);
    }

    #[test]
    fn test_disconnected_sccs_queried_separately() {
        // Two disconnected cycles, queried in separate root() calls:
        //
        // SCC A: 1 → 2 → 1
        // SCC B: 3 → 4 → 3 (disconnected from A)
        use std::cell::Cell;

        let visits = Cell::new(0);
        let visit = |h: &i32, edge: &mut dyn FnMut(i32)| {
            visits.set(visits.get() + 1);
            match h {
                1 => edge(2),
                2 => edge(1),
                3 => edge(4),
                4 => edge(3),
                _ => {}
            }
        };

        let mut tarjan = Tarjan::new();

        // Query first SCC - should visit nodes 1 and 2
        let scc_a_idx = tarjan.root(1, &visit);
        let scc_a: Vec<_> = tarjan.iter_scc(scc_a_idx).copied().collect();
        assert_eq!(scc_a.len(), 2);
        assert!(scc_a.contains(&1));
        assert!(scc_a.contains(&2));
        assert_eq!(visits.get(), 2);

        // Query second SCC - should visit nodes 3 and 4
        let scc_b_idx = tarjan.root(3, &visit);
        let scc_b: Vec<_> = tarjan.iter_scc(scc_b_idx).copied().collect();
        assert_eq!(scc_b.len(), 2);
        assert!(scc_b.contains(&3));
        assert!(scc_b.contains(&4));
        assert_eq!(visits.get(), 4);

        // Verify caching - no additional visits
        let idx_2 = tarjan.root(2, &visit);
        assert_eq!(tarjan.iter_scc(idx_2).count(), 2);
        let idx_4 = tarjan.root(4, &visit);
        assert_eq!(tarjan.iter_scc(idx_4).count(), 2);
        assert_eq!(visits.get(), 4);
    }
}
