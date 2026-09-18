/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Cold-start LSP benchmark against PyTorch: latency from a cold server to the
//! first answered cross-file go-to-definition — a proxy for time-to-first-index.
//! From a *cold* server it runs the whole path a developer waits through when
//! opening a project: initialize, open a file deep in the dependency graph, and
//! answer the first cross-file go-to-definition (on `from torch.nn import
//! Parameter`). The response can only be produced once that file and its import
//! closure have been analyzed.
//!
//! The shared PyTorch checkout harness lives in [`crate::common`].

use std::path::Path;
use std::time::Duration;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Url;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::common::BACKWARD;
use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

/// Position of `Parameter` in `_backward.py`'s `from torch.nn import Parameter`
/// (0-indexed), the cross-file symbol the benchmark navigates to.
const PARAM_LINE: u32 = 9;
const PARAM_COL: u32 = 21;

/// Full cold start: launch a fresh server over all cores, initialize, open a deep
/// file, and resolve the first cross-file go-to-definition. Returns the
/// interaction so it is dropped (and the server torn down) outside the measured
/// region. All cores because a real IDE cold start parallelizes across the rayon
/// pool — which is exactly what time-to-first-index should reflect.
fn cold_start_definition(root: &Path) -> LspInteraction {
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: lsp_args(),
        telemetry: Box::new(NoTelemetry),
        thread_count: ThreadCount::AllThreads,
        thrift_remapper: None,
    });
    // The first check of a project this large can stay silent well past the
    // interactive-test defaults, so give the cold start generous headroom.
    interaction
        .client
        .set_timeouts(Duration::from_secs(120), Duration::from_secs(1800));
    interaction.set_root(root.to_path_buf());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Url::from_file_path(root).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open(BACKWARD);
    interaction
        .client
        .definition(BACKWARD, PARAM_LINE, PARAM_COL)
        .expect_response_with(|resp: Option<GotoDefinitionResponse>| {
            let resolved = match &resp {
                Some(GotoDefinitionResponse::Scalar(_)) => true,
                Some(GotoDefinitionResponse::Array(locs)) => !locs.is_empty(),
                Some(GotoDefinitionResponse::Link(links)) => !links.is_empty(),
                None => false,
            };
            // Fail fast instead of hanging until the receive timeout: a request
            // gets exactly one response, so returning `false` here would make
            // `expect_response_with` wait for a second one that never arrives.
            // An empty result means PARAM_LINE/PARAM_COL drifted (e.g. after a
            // pin bump) and needs updating.
            assert!(
                resolved,
                "go-to-definition at {BACKWARD}:{PARAM_LINE}:{PARAM_COL} returned no \
                 location; PARAM_LINE/PARAM_COL likely drifted after a pin bump"
            );
            true
        })
        .unwrap();

    interaction
}

/// Cold-start latency to the first answered cross-file navigation (a proxy for
/// time-to-first-index). `BatchSize::PerIteration` runs `cold_start_definition`
/// once per measured iteration behind a fresh server, so every sample is a
/// genuine cold start. Criterion enforces a floor of 10 samples (it panics
/// below that), so we use the floor for this heavy walltime bench; the returned
/// interaction is dropped outside the timed region, keeping server teardown out
/// of the measurement.
fn cold_start_go_to_definition(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    group.bench_function("cold_start_go_to_definition", |b| {
        b.iter_batched(
            || root.clone(),
            |root| cold_start_definition(&root),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, cold_start_go_to_definition);
