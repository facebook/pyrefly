/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Warm `workspace/symbol` latency against a fully indexed PyTorch checkout.
//! Setup, indexing, and the first query happen before Criterion starts, so each
//! sample measures only an end-to-end workspace-symbol request over a warm index.

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use lsp_types::Url;
use lsp_types::WorkspaceSymbolResponse;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::common::BACKWARD;
use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

const INDEXED_SYMBOL: &str = "InitDeviceMeshTest";
const INDEXED_SYMBOL_FILE: &str = "test/distributed/test_device_mesh.py";

fn query(interaction: &LspInteraction) {
    interaction
        .client
        .send_workspace_symbol("init")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("unexpected workspace symbol response: {result:?}");
            };
            assert!(
                !symbols.is_empty(),
                "workspace symbol query returned no results"
            );
            black_box(symbols);
            true
        })
        .unwrap();
}

fn workspace_symbol(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    let mut interaction = None;
    group.bench_function("workspace_symbol_init", |b| {
        let interaction = interaction.get_or_insert_with(|| {
            let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
                args: LspArgs {
                    indexing_mode: IndexingMode::LazyBlocking,
                    workspace_indexing_limit: usize::MAX,
                    ..lsp_args()
                },
                telemetry: Box::new(NoTelemetry),
                thread_count: ThreadCount::AllThreads,
                thrift_remapper: None,
            });
            interaction.client.set_message_logging(false);
            interaction
                .client
                .set_timeouts(Duration::from_secs(120), Duration::from_secs(1800));
            interaction.set_root(root.clone());
            interaction
                .initialize(InitializeSettings {
                    workspace_folders: Some(vec![(
                        "pytorch".to_owned(),
                        Url::from_file_path(&root).unwrap(),
                    )]),
                    configuration: Some(None),
                    ..Default::default()
                })
                .unwrap();

            interaction.client.did_open(BACKWARD);
            let expected_uri = Url::from_file_path(root.join(INDEXED_SYMBOL_FILE)).unwrap();
            interaction
                .client
                .send_workspace_symbol(INDEXED_SYMBOL)
                .expect_response_with(|result| {
                    let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                        panic!("unexpected workspace symbol response: {result:?}");
                    };
                    assert!(
                        symbols.iter().any(|symbol| symbol.name == INDEXED_SYMBOL
                            && symbol.location.uri == expected_uri),
                        "workspace index does not contain {INDEXED_SYMBOL_FILE}"
                    );
                    true
                })
                .unwrap();
            query(&interaction);
            interaction
        });

        b.iter(|| query(interaction));
    });
    group.finish();

    if let Some(interaction) = interaction {
        interaction.shutdown().unwrap();
    }
}

criterion_group!(benches, workspace_symbol);
