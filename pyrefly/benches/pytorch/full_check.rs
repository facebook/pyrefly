/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Full-check batch benchmark against PyTorch: cold batch `check` throughput.
//! Unlike the interactive `cold_start` and `error_propagation` benches (which
//! drive the LSP server and measure latency), this runs exactly what `pyrefly
//! check` (project mode, no file args) does from inside the checkout — discover
//! the project, resolve its config, build handles for every project file, and
//! check them across all cores from a cold `State`. It is the proxy for
//! whole-project batch throughput rather than interactive latency.
//!
//! The shared PyTorch checkout harness lives in [`crate::common`].

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use pyrefly::commands::check::Handles;
use pyrefly::commands::files::FilesArgs;
use pyrefly::state::require::Require;
use pyrefly::state::state::State;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::thread_pool::ThreadCount;

use crate::common::pytorch_root_or_skip;

/// One cold `pyrefly check` of the current project: resolve the project (empty
/// file list → project mode, rooted at the cwd), then check every project file
/// across all cores from a fresh `State`. The transaction is committed so the
/// checked results (and the heavy `State`) are dropped outside the measured
/// region, keeping teardown out of the timing.
fn full_check() -> State {
    let (includes, config_finder, _upsell) =
        FilesArgs::get(Vec::new(), None, ConfigOverrideArgs::default(), None)
            .expect("resolving the PyTorch project");
    let files = config_finder
        .checkpoint(includes.files_iter())
        .expect("listing project files");
    let handles = Handles::new(files);
    let (loaded_handles, _, _) = handles.all(&config_finder);
    // Fail loudly if project discovery broke (e.g. the pin's layout changed)
    // rather than silently measuring an empty check. The pinned checkout resolves
    // ~2.5k project files; 1000 is a generous floor that still catches breakage.
    assert!(
        loaded_handles.len() > 1000,
        "full_check expected the whole PyTorch project (thousands of files), got {}",
        loaded_handles.len()
    );

    let state = State::new(config_finder, ThreadCount::AllThreads);
    // Mirror `pyrefly check`'s require levels (see `CheckArgs::get_required_levels`):
    // the checked files are required at `Errors`, while transitively-imported
    // dependencies (e.g. the bundled typeshed stdlib) default to `Exports`.
    // Using `Errors` as the default would error-check that dependency closure too,
    // measuring more work than a real check.
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction
        .as_mut()
        .run(&loaded_handles, Require::Errors, None);
    // Collect the errors like `run_inner` does — part of the check `pyrefly check`
    // measures; `black_box` keeps the collection from being optimized away.
    std::hint::black_box(transaction.as_mut().get_errors(&loaded_handles));
    state.commit_transaction(transaction, None);
    state
}

/// Whole-project batch-check throughput: time a cold `pyrefly check` over the
/// entire PyTorch checkout across all cores. `BatchSize::PerIteration` runs
/// `full_check` once per measured iteration behind a fresh `State`, so every
/// sample is a genuine cold check; the returned `State` is dropped outside the
/// timed region, keeping teardown out of the measurement. Criterion enforces a
/// floor of 10 samples, so we use the floor for this heavy walltime bench.
fn full_check_torch(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    // `pyrefly check` (project mode) checks the project rooted at the cwd, so put
    // the process inside the checkout to exercise the real project-discovery path.
    // Safe here because `full_check` is the last bench in the binary and the LSP
    // benches address files by absolute path.
    std::env::set_current_dir(&root).expect("cd into the PyTorch checkout");

    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    group.bench_function("full_check", |b| {
        b.iter_batched(|| (), |()| full_check(), BatchSize::PerIteration);
    });
    group.finish();
}

criterion_group!(benches, full_check_torch);
