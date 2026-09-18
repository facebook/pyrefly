/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Fully-indexed build benchmark against PyTorch: how long a cold whole-project
//! index takes.
//!
//! Timing only. The memory that index holds is reported by the standalone
//! `pytorch_memory` binary instead, because RSS is a property of the process and
//! this binary runs several allocation-heavy benchmarks in one.

use std::env::set_current_dir;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use pyrefly_bench_harness::index_project;

use crate::common::pytorch_root_or_skip;

/// The pinned checkout resolves ~2.5k project files; 1000 is a generous floor
/// that still catches the pin's layout changing under us.
const MIN_FILES: usize = 1000;

/// `BatchSize::PerIteration` gives every sample a fresh `State`, so each one is a
/// genuine cold index and teardown stays outside the timing. Criterion enforces a
/// floor of 10 samples, which is what these heavy walltime benchmarks use.
fn indexed_memory_torch(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    // Project mode resolves the project rooted at the cwd, so put the process
    // inside the corpus to exercise the real project-discovery path.
    set_current_dir(&root).expect("cd into the benchmark corpus");

    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    group.bench_function("indexed_memory", |b| {
        b.iter_batched(
            || (),
            |()| index_project(MIN_FILES, "pytorch"),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, indexed_memory_torch);
