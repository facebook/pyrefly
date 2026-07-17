/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Demand tree collection for debugging and testing module demand-call laziness.
//!
//! Pyrefly's type-checker computes per-module data lazily: a module that
//! imports another may only need that other module's exports, not its full
//! answers or solutions. The demand tree records *which* cross-module
//! lookups (`LookupExport` / `LookupAnswer`) actually fire during a check
//! run, so authors can verify that dependencies stop at the earliest step
//! sufficient to resolve what callers need.
//!
//! See [`DemandCollector`] for collection mechanics and [`report_json`] for
//! the JSON serialization used by `pyrefly check --report-demand-tree`.
//!
//! The tree is machine-readable via serde. `#[serde(default)]` on optional
//! fields lets the schema grow without breaking existing consumers.

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;

use crate::lock::Mutex;

/// What kind of cross-module demand an edge represents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DemandKind {
    /// One Load-level demand call. `reason` identifies which
    /// `LookupExport` method triggered it. Load-level demands only
    /// confirm the target module is reachable; they do not force its
    /// exports to be computed.
    Load { reason: String },
    /// One Exports-level demand call (e.g. `export_exists`,
    /// `is_special_export`). `reason` identifies which `LookupExport`
    /// method triggered it. Each call records its own edge — multiple
    /// reasons against the same `(from, target)` pair produce
    /// multiple sibling edges, one per call.
    Exports { reason: String },
    /// A cross-module Answer lookup span. May have children if the
    /// computation recursively demanded data from other modules.
    /// `key` is the `Debug`-formatted lookup key.
    Answer { key: String },
}

/// One observed cross-module demand: an edge from `from` to `target`
/// carrying what was demanded (`kind`). Edges form a tree because
/// Answer demands open spans that nest other demands made while
/// computing them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandEdge {
    /// The module that made the demand.
    pub from: String,
    /// The module the demand was made against.
    pub target: String,
    /// What kind of demand this was.
    pub kind: DemandKind,
    /// Nested demands made while computing this one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<DemandEdge>,
}

thread_local! {
    /// Per-thread stack of in-flight parent spans, used to nest closed
    /// demand edges under their caller. This is inherently per-thread
    /// scratch space: enter and exit always happen on the same thread,
    /// and the stack empties itself as long as every enter has a
    /// matching exit.
    static STACK: RefCell<Vec<DemandEdge>> = const { RefCell::new(Vec::new()) };
}

/// A demand-tree collection session. Cloning produces another handle to the
/// same underlying roots (the collector is reference-counted internally).
#[derive(Clone, Default)]
pub struct DemandCollector {
    roots: Arc<Mutex<Vec<DemandEdge>>>,
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
            stack.borrow_mut().push(DemandEdge {
                from: from.to_string(),
                target: target.to_string(),
                kind: DemandKind::Answer {
                    key: format!("{key:?}"),
                },
                children: Vec::new(),
            });
        });
        DemandSpan {
            collector: self,
            _not_send: PhantomData,
        }
    }

    /// Record a leaf event for a Load-level demand. `reason` identifies
    /// which `LookupExport` method triggered the demand.
    #[inline]
    pub fn load_event(&self, from: impl fmt::Display, target: impl fmt::Display, reason: &str) {
        self.attach(DemandEdge {
            from: from.to_string(),
            target: target.to_string(),
            kind: DemandKind::Load {
                reason: reason.to_owned(),
            },
            children: Vec::new(),
        });
    }

    /// Record a leaf event for an Exports-level demand. `reason` identifies
    /// which `LookupExport` method triggered the demand.
    #[inline]
    pub fn exports_event(&self, from: impl fmt::Display, target: impl fmt::Display, reason: &str) {
        self.attach(DemandEdge {
            from: from.to_string(),
            target: target.to_string(),
            kind: DemandKind::Exports {
                reason: reason.to_owned(),
            },
            children: Vec::new(),
        });
    }

    /// Attach a completed edge either to the current parent on the stack or,
    /// if no parent is in flight, to the collector's shared root list.
    fn attach(&self, edge: DemandEdge) {
        let leftover = STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            match stack.last_mut() {
                Some(parent) => {
                    parent.children.push(edge);
                    None
                }
                None => Some(edge),
            }
        });
        if let Some(edge) = leftover {
            self.roots.lock().push(edge);
        }
    }

    /// Take the collected tree roots, leaving the collector empty.
    pub fn take_roots(&self) -> Vec<DemandEdge> {
        std::mem::take(&mut *self.roots.lock())
    }
}

/// RAII guard for an in-flight Answer demand span. Dropping the guard —
/// whether via normal control flow or unwind — pops the span off the
/// per-thread stack and attaches it to its parent (or to the collector's
/// roots if no parent is in flight).
///
/// `enter()` pushes onto the calling thread's `STACK` and `Drop::drop`
/// pops from the dropping thread's `STACK`, so a guard moved to a
/// different thread before drop would pop an unrelated entry. The
/// `PhantomData<*const ()>` makes `DemandSpan` `!Send` to rule that out
/// at compile time.
#[must_use = "demand span is closed when the guard is dropped; hold it for the duration of the demand"]
pub struct DemandSpan<'a> {
    collector: &'a DemandCollector,
    _not_send: PhantomData<*const ()>,
}

impl Drop for DemandSpan<'_> {
    fn drop(&mut self) {
        if let Some(edge) = STACK.with(|stack| stack.borrow_mut().pop()) {
            self.collector.attach(edge);
        }
    }
}

/// Serialize a demand tree alongside per-module step info as a pretty-
/// printed JSON document for `pyrefly check --report-demand-tree`.
/// `module_steps` should pair each module's name with the highest step
/// it reached during the run.
pub fn report_json(roots: &[DemandEdge], module_steps: &[(String, &'static str)]) -> String {
    #[derive(Serialize)]
    struct ModuleStep<'a> {
        module: &'a str,
        last_step: &'a str,
    }

    #[derive(Serialize)]
    struct Report<'a> {
        module_steps: Vec<ModuleStep<'a>>,
        demand_tree: &'a [DemandEdge],
    }

    let mut steps: Vec<ModuleStep> = module_steps
        .iter()
        .map(|(m, s)| ModuleStep {
            module: m.as_str(),
            last_step: s,
        })
        .collect();
    // Stable ordering makes diffing reports tractable.
    steps.sort_by(|a, b| a.module.cmp(b.module));

    let report = Report {
        module_steps: steps,
        demand_tree: roots,
    };
    serde_json::to_string_pretty(&report).expect("demand tree report should always serialize")
}
