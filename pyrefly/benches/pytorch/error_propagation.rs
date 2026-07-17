/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Error-propagation LSP benchmark against PyTorch: incremental recheck latency.
//! With the server already warm, rebind the `Parameter` export in
//! `torch/nn/__init__.py` to a non-type value and measure how long the resulting
//! error takes to surface in the distant dependent `_backward.py`. We rebind
//! rather than delete the export so the trigger is independent of filesystem
//! case-sensitivity: deleting `Parameter as Parameter` lets a case-insensitive
//! filesystem resolve `Parameter` to the `parameter.py` submodule, masking the
//! error. The warm-up happens outside the measured region.
//!
//! The shared PyTorch checkout harness lives in [`crate::common`].

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use lsp_types::Url;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use serde_json::json;

use crate::common::BACKWARD;
use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

/// File whose `Parameter` re-export `error_propagation` rebinds to trigger a
/// propagated error.
const NN_INIT: &str = "torch/nn/__init__.py";

/// Restores a file's original contents on drop, so the in-place edit
/// `error_propagation` makes never leaves the PyTorch checkout dirty.
struct RestoreOnDrop {
    path: PathBuf,
    original: String,
}

impl Drop for RestoreOnDrop {
    fn drop(&mut self) {
        let _ = fs::write(&self.path, &self.original);
    }
}

/// A warm server with both files open and initial diagnostics received — the
/// state right before the edit whose propagation `error_propagation` measures.
struct Prepared {
    interaction: LspInteraction,
    backward_path: PathBuf,
    modified_content: String,
    /// Held only for its `Drop`, which restores the edited file after each
    /// iteration; never read directly.
    #[allow(dead_code)]
    restore: RestoreOnDrop,
}

/// Warm-up (unmeasured): launch the server over all cores, open both files, and
/// wait for the initial check of each to settle. Type errors are forced on so the
/// server publishes diagnostics for the opened files.
fn prepare(root: &Path) -> Prepared {
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: lsp_args(),
        telemetry: Box::new(NoTelemetry),
        thread_count: ThreadCount::AllThreads,
        thrift_remapper: None,
    });
    interaction
        .client
        .set_timeouts(Duration::from_secs(120), Duration::from_secs(1800));
    interaction.set_root(root.to_path_buf());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Url::from_file_path(root).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    let nn_init_path = root.join(NN_INIT);
    let backward_path = root.join(BACKWARD);

    // Open the dependent (`_backward.py`) and let it settle *before* opening the
    // source (`torch/nn/__init__.py`), so its `torch.nn` import binds to the
    // source's filesystem handle. If the source were open first, the dependent
    // would bind to the source's in-memory handle instead, which the disk write +
    // save in `measure` (an `invalidate_disk` on the filesystem handle) never
    // reaches — the error would never propagate and the benchmark would hang.
    // `_backward.py`'s diagnostics only arrive once its import closure — reaching
    // `torch.nn` — has resolved, marking the end of cold start.
    interaction.client.did_open(BACKWARD);
    interaction
        .client
        .expect_publish_diagnostics_for_file(backward_path.clone())
        .unwrap();

    interaction.client.did_open(NN_INIT);
    interaction
        .client
        .expect_publish_diagnostics_for_file(nn_init_path.clone())
        .unwrap();

    let original = fs::read_to_string(&nn_init_path).unwrap();
    // Rebind `Parameter` to an `int`; dependents that use it as a type then error.
    let modified_content =
        format!("{original}\nParameter = 42  # pyrefly bench: force a propagated error\n");

    Prepared {
        interaction,
        backward_path,
        modified_content,
        restore: RestoreOnDrop {
            path: nn_init_path,
            original,
        },
    }
}

/// Measured region: apply the edit and wait for the resulting error to reach the
/// far file. `edit_file` writes to disk and notifies via the file watcher, since
/// dependents recheck off saved files. `Iterator[Parameter]` is now
/// `Iterator[int-instance]`, which is not a type — we key on that error message
/// rather than an exact diagnostic count, which varies by pyrefly version.
fn measure(p: &mut Prepared) {
    p.interaction.client.edit_file(NN_INIT, &p.modified_content);
    p.interaction
        .client
        .expect_publish_diagnostics_eventual_message_contains(
            p.backward_path.clone(),
            "Expected a type form, got instance of `int`",
        )
        .unwrap();
}

/// Incremental recheck latency: with a warm server, edit a heavily-imported file
/// and measure how long the resulting error takes to surface in a distant
/// dependent. `iter_batched` runs `prepare` (the cold start + open) unmeasured in
/// setup and times only `measure` (the edit + propagation); `BatchSize::PerIteration`
/// gives each iteration a fresh warm server, since the edit is not idempotent.
/// `RestoreOnDrop` (inside `Prepared`) reverts the edit after each iteration.
/// Criterion enforces a floor of 10 samples, so we use the floor.
fn error_propagation(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    group.bench_function("error_propagation", |b| {
        b.iter_batched(
            || prepare(&root),
            |mut prepared| {
                measure(&mut prepared);
                prepared
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, error_propagation);
