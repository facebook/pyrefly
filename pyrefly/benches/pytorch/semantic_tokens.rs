/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Warm full-file semantic-token latency for PyTorch's `torch.nn.Module`.
//! Server setup, type checking, and the first request are outside the measurement.

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use lsp_types::SemanticTokensRequest;
use lsp_types::Uri;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use serde_json::json;

use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

const MODULE: &str = "torch/nn/modules/module.py";

fn query(interaction: &LspInteraction, uri: &Uri) {
    interaction
        .client
        .send_request::<SemanticTokensRequest>(json!({
            "textDocument": {"uri": uri.to_string()}
        }))
        .expect_response_with(|result| {
            let tokens = result.expect("semantic-token request returned no result");
            assert!(!tokens.data.is_empty(), "semantic-token response is empty");
            black_box(tokens);
            true
        })
        .unwrap();
}

fn semantic_tokens(c: &mut Criterion) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    let uri = Uri::from_file_path(root.join(MODULE)).unwrap();
    let mut group = c.benchmark_group("pytorch");
    group.sample_size(10);
    let mut interaction = None;
    group.bench_function("semantic_tokens_module", |b| {
        let interaction = interaction.get_or_insert_with(|| {
            let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
                args: lsp_args(),
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
                        Uri::from_file_path(&root).unwrap(),
                    )]),
                    configuration: Some(None),
                    // Editors with syntax grammars request semantic tokens only.
                    capabilities: Some(json!({
                        "textDocument": {
                            "semanticTokens": {
                                "requests": {"full": true},
                                "tokenTypes": [],
                                "tokenModifiers": [],
                                "formats": ["relative"],
                                "augmentsSyntaxTokens": true
                            }
                        }
                    })),
                    ..Default::default()
                })
                .unwrap();
            interaction.client.did_open(MODULE);
            query(&interaction, &uri);
            interaction
        });

        b.iter(|| query(interaction, &uri));
    });
    group.finish();

    if let Some(interaction) = interaction {
        interaction.shutdown().unwrap();
    }
}

criterion_group!(benches, semantic_tokens);
