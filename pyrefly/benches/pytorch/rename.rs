/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Warm rename benchmarks against PyTorch. The paired measurements run the
//! same semantic rename with and without non-Python text occurrences, so their
//! difference captures the cost of walking and scanning the workspace.

use std::path::Path;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use lsp_types::request::Rename;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use serde_json::json;

use crate::common::BACKWARD;
use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

/// Position of `reverse_closure` in its definition in `_backward.py`. The name
/// has no occurrences outside Python files, so both cases return the same edits.
const SYMBOL_LINE: u32 = 39;
const SYMBOL_COL: u32 = 4;

fn prepare(root: &Path, text_occurrences: bool) -> LspInteraction {
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
            configuration: Some(Some(json!([{
                "pyrefly": {
                    "rename": {
                        "textOccurrences": text_occurrences
                    }
                }
            }]))),
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
        .definition(BACKWARD, SYMBOL_LINE, SYMBOL_COL)
        .expect_response_with(|response: Option<GotoDefinitionResponse>| response.is_some())
        .unwrap();
    interaction
}

fn measure(interaction: &mut LspInteraction, root: &Path) {
    interaction
        .client
        .send_request::<Rename>(json!({
            "textDocument": {
                "uri": Url::from_file_path(root.join(BACKWARD)).unwrap()
            },
            "position": {
                "line": SYMBOL_LINE,
                "character": SYMBOL_COL
            },
            "newName": "renamed_reverse_closure"
        }))
        .expect_response_with(|response: Option<WorkspaceEdit>| response.is_some())
        .unwrap();
}

fn rename(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    let mut group = c.benchmark_group("pytorch");

    for (name, text_occurrences) in [
        ("rename_semantic_only", false),
        ("rename_with_text_occurrences", true),
    ] {
        let mut interaction = prepare(&root, text_occurrences);
        group.bench_function(name, |b| {
            b.iter(|| measure(&mut interaction, &root));
        });
        interaction.shutdown().unwrap();
    }

    group.finish();
}

criterion_group!(benches, rename);
