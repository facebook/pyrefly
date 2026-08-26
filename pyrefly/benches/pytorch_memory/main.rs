/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Fully-indexed memory report for the pinned PyTorch checkout: how much memory
//! a language server holds once it has indexed the whole project at
//! `Require::Indexing`.
//!
//! A standalone binary rather than a criterion benchmark, and deliberately its
//! own target rather than another module of `pytorch_bench`. RSS is a property
//! of the process: `VmRSS` counts whatever else the process still holds and
//! `VmHWM` is the high-water mark over its whole life, so running this after the
//! walltime benchmarks in one process would report their memory, not this
//! index's. The `pytorch` criterion benchmark still times the cold index.
//!
//! Run with
//! `buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_memory_bench`
//! or `cargo bench -p pyrefly --bench pytorch_memory`.

#[path = "../pytorch/common.rs"]
#[expect(
    dead_code,
    reason = "shared with the criterion benches, which use the LSP helpers this binary does not"
)]
mod common;

use pyrefly_bench_harness::report_indexed_memory;

use crate::common::pytorch_root_or_skip;

/// The pinned checkout resolves ~2.5k project files; 1000 is a generous floor
/// that still catches the pin's layout changing under us.
const MIN_FILES: usize = 1000;

fn main() {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    report_indexed_memory(&root, MIN_FILES, "pytorch");
}
