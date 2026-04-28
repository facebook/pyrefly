/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Demand tree collection for debugging and testing laziness.
//!
//! A [`DemandCollector`] records cross-module demand calls into a tree. A
//! collector is usually owned by a `Transaction`, scoping collection to a
//! single check run — so parallel checks don't interfere with one another.
//! Parent/child nesting is tracked via a per-thread scratch stack.
//!
//! The tree is machine-readable via serde. `#[serde(default)]` on optional
//! fields lets the schema grow without breaking existing consumers.

use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

use serde::Deserialize;
use serde::Serialize;

/// What kind of cross-module demand a node represents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DemandKind {
    /// A leaf event for an Exports-level demand (e.g. `module_exists`,
    /// `export_exists`). Carries the reason string identifying which
    /// `LookupExport` method triggered the demand.
    Exports { reason: String },
    /// A cross-module Answer lookup span. May have children if the
    /// computation recursively demanded data from other modules.
    /// `key` is the `Debug`-formatted lookup key.
    Answer { key: String },
}

/// A node in the demand tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandNode {
    /// The module that made the demand.
    pub from: String,
    /// The module the demand was made against.
    pub target: String,
    /// What kind of demand this was.
    pub kind: DemandKind,
    /// Nested demands made while computing this one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<DemandNode>,
}

thread_local! {
    /// Per-thread stack of in-flight parent spans, used to nest closed demand
    /// nodes under their caller. This is inherently per-thread scratch space:
    /// enter and exit always happen on the same thread, and the stack empties
    /// itself as long as every enter has a matching exit.
    static STACK: RefCell<Vec<DemandNode>> = const { RefCell::new(Vec::new()) };
}

/// A demand-tree collection session. Cloning produces another handle to the
/// same underlying roots (the collector is reference-counted internally).
#[derive(Clone, Default)]
pub struct DemandCollector {
    roots: Arc<Mutex<Vec<DemandNode>>>,
}

impl DemandCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a demand span for a cross-module Answer lookup. Hold the returned
    /// guard for the duration of the demand — dropping the guard (including
    /// on unwind) closes the span and attaches it to its parent, keeping
    /// enter/exit balanced even across panics. `key` is the lookup key,
    /// formatted via `Debug`.
    #[inline]
    pub fn enter(
        &self,
        from: impl fmt::Display,
        target: impl fmt::Display,
        key: impl fmt::Debug,
    ) -> DemandSpan<'_> {
        STACK.with(|stack| {
            stack.borrow_mut().push(DemandNode {
                from: from.to_string(),
                target: target.to_string(),
                kind: DemandKind::Answer {
                    key: format!("{key:?}"),
                },
                children: Vec::new(),
            });
        });
        DemandSpan { collector: self }
    }

    /// Record a leaf event for an Exports-level demand. `reason` identifies
    /// which `LookupExport` method triggered the demand.
    #[inline]
    pub fn exports_event(&self, from: impl fmt::Display, target: impl fmt::Display, reason: &str) {
        self.attach(DemandNode {
            from: from.to_string(),
            target: target.to_string(),
            kind: DemandKind::Exports {
                reason: reason.to_owned(),
            },
            children: Vec::new(),
        });
    }

    /// Attach a completed node either to the current parent on the stack or,
    /// if no parent is in flight, to the collector's shared root list.
    fn attach(&self, node: DemandNode) {
        let leftover = STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            match stack.last_mut() {
                Some(parent) => {
                    parent.children.push(node);
                    None
                }
                None => Some(node),
            }
        });
        if let Some(node) = leftover {
            self.roots.lock().unwrap().push(node);
        }
    }

    /// Take the collected tree roots, leaving the collector empty.
    pub fn take_roots(&self) -> Vec<DemandNode> {
        std::mem::take(&mut *self.roots.lock().unwrap())
    }
}

/// RAII guard for an in-flight Answer demand span. Dropping the guard —
/// whether via normal control flow or unwind — pops the span off the
/// per-thread stack and attaches it to its parent (or to the collector's
/// roots if no parent is in flight).
#[must_use = "demand span is closed when the guard is dropped; hold it for the duration of the demand"]
pub struct DemandSpan<'a> {
    collector: &'a DemandCollector,
}

impl Drop for DemandSpan<'_> {
    fn drop(&mut self) {
        if let Some(node) = STACK.with(|stack| stack.borrow_mut().pop()) {
            self.collector.attach(node);
        }
    }
}
