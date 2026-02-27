/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Benchmark test for measuring code action latency in large codebases.
//!
//! Run manually with:
//! CODE_ACTION_BENCH_PATH=/path/to/repo \
//! CODE_ACTION_BENCH_FILE=relative/path.py \
//! CODE_ACTION_BENCH_RANGE=START_LINE:START_COL-END_LINE:END_COL \
//! CODE_ACTION_BENCH_ITERS=10 \
//! CODE_ACTION_BENCH_TITLE="Introduce parameter `param`" \
//! cargo test --release test_code_action_latency -- --ignored --nocapture

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use lsp_types::CodeAction;
use lsp_types::CodeActionOrCommand;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::CodeActionResolveRequest;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::init_thread_pool;
use serde_json::json;

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

fn parse_range_env(raw: &str) -> (u32, u32, u32, u32) {
    let (start, end) = raw
        .split_once('-')
        .unwrap_or_else(|| panic!("Invalid CODE_ACTION_BENCH_RANGE: {raw}"));
    let (start_line, start_col) = start
        .split_once(':')
        .unwrap_or_else(|| panic!("Invalid CODE_ACTION_BENCH_RANGE: {raw}"));
    let (end_line, end_col) = end
        .split_once(':')
        .unwrap_or_else(|| panic!("Invalid CODE_ACTION_BENCH_RANGE: {raw}"));
    let start_line = start_line
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("Invalid start line in CODE_ACTION_BENCH_RANGE: {raw}"));
    let start_col = start_col
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("Invalid start column in CODE_ACTION_BENCH_RANGE: {raw}"));
    let end_line = end_line
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("Invalid end line in CODE_ACTION_BENCH_RANGE: {raw}"));
    let end_col = end_col
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("Invalid end column in CODE_ACTION_BENCH_RANGE: {raw}"));
    (start_line, start_col, end_line, end_col)
}

fn print_stats(label: &str, samples: &[Duration]) {
    if samples.is_empty() {
        eprintln!("{label}: no samples");
        return;
    }
    let mut sorted = samples.to_vec();
    sorted.sort();
    let count = sorted.len() as u64;
    let total: Duration = sorted
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, v| acc + v);
    let mean_nanos = total.as_nanos() / count as u128;
    let mean = Duration::from_nanos(mean_nanos as u64);
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(((sorted.len() as f64) * 0.95).floor() as usize).min(sorted.len() - 1)];
    let p99 = sorted[(((sorted.len() as f64) * 0.99).floor() as usize).min(sorted.len() - 1)];
    eprintln!("{label}: count={count} mean={mean:?} p50={p50:?} p95={p95:?} p99={p99:?}");
}

#[test]
#[ignore] // Run manually with env vars. See module docs.
fn test_code_action_latency() {
    let repo_path =
        std::env::var("CODE_ACTION_BENCH_PATH").expect("CODE_ACTION_BENCH_PATH must be set");
    let file_rel =
        std::env::var("CODE_ACTION_BENCH_FILE").expect("CODE_ACTION_BENCH_FILE must be set");
    let range_raw = std::env::var("CODE_ACTION_BENCH_RANGE")
        .expect("CODE_ACTION_BENCH_RANGE must be set (START:COL-END:COL)");
    let iterations = std::env::var("CODE_ACTION_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let title_filter = std::env::var("CODE_ACTION_BENCH_TITLE").ok();

    let repo_root = PathBuf::from(&repo_path);
    assert!(repo_root.exists(), "Repo not found at {repo_path}");

    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    init_thread_pool(ThreadCount::AllThreads);
    interaction.set_root(repo_root.clone());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            workspace_folders: Some(vec![(
                "bench".to_owned(),
                Url::from_file_path(&repo_root).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    let file_path = repo_root.join(&file_rel);
    let uri = Url::from_file_path(&file_path).unwrap();
    interaction.client.did_open(&file_rel);

    // Warm up by waiting for diagnostics on the target file.
    interaction
        .client
        .expect_publish_diagnostics_for_file(file_path)
        .unwrap();

    let (start_line, start_col, end_line, end_col) = parse_range_env(&range_raw);

    let mut list_samples = Vec::new();
    let mut resolve_samples = Vec::new();

    for _ in 0..iterations {
        let action_to_resolve: RefCell<Option<CodeAction>> = RefCell::new(None);
        let list_start = Instant::now();
        interaction
            .client
            .send_request::<CodeActionRequest>(json!({
                "textDocument": { "uri": uri },
                "range": {
                    "start": { "line": start_line, "character": start_col },
                    "end": { "line": end_line, "character": end_col }
                },
                "context": {
                    "diagnostics": [],
                    "triggerKind": 1
                }
            }))
            .expect_response_with(|response| {
                let Some(actions) = response else {
                    return false;
                };
                for action in actions {
                    let CodeActionOrCommand::CodeAction(code_action) = action else {
                        continue;
                    };
                    if let Some(expected_title) = title_filter.as_ref() {
                        if code_action.title != *expected_title {
                            continue;
                        }
                    }
                    if code_action.data.is_some() {
                        *action_to_resolve.borrow_mut() = Some(code_action);
                        return true;
                    }
                }
                false
            })
            .unwrap();
        list_samples.push(list_start.elapsed());

        let Some(action) = action_to_resolve.into_inner() else {
            panic!("No resolvable code action found. Set CODE_ACTION_BENCH_TITLE to match one.");
        };

        let resolve_start = Instant::now();
        interaction
            .client
            .send_request::<CodeActionResolveRequest>(serde_json::to_value(action).unwrap())
            .expect_response_with(|resolved| resolved.edit.is_some())
            .unwrap();
        resolve_samples.push(resolve_start.elapsed());
    }

    eprintln!("\n==== Code Action Benchmark Results ====");
    print_stats("list", &list_samples);
    print_stats("resolve", &resolve_samples);
    eprintln!("======================================\n");

    interaction.shutdown().unwrap();
}
