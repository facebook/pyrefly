/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Benchmark test for measuring type error propagation latency in large codebases.
//!
//! This test opens the PyTorch codebase and measures how long it takes for
//! a type error to propagate from `torch/nn/__init__.py` to
//! `torch/distributed/pipelining/_backward.py` when removing an export.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use lsp_types::Url;
use pyrefly_util::fs_anyhow::read_to_string;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::init_thread_pool;
use serde_json::json;

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;

#[test]
#[ignore] // Run manually with: PYTORCH_PATH=/path/to/pytorch cargo test --release test_pytorch_error_propagation_latency -- --ignored --nocapture
fn test_pytorch_error_propagation_latency() {
    let pytorch_path =
        std::env::var("PYTORCH_PATH").expect("PYTORCH_PATH environment variable must be set");
    let pytorch_root = PathBuf::from(&pytorch_path);
    assert!(
        pytorch_root.exists(),
        "PyTorch not found at {}",
        pytorch_path
    );

    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    // Override the default 3-thread limit to use all available cores for realistic benchmarking
    init_thread_pool(ThreadCount::AllThreads);
    interaction.set_root(pytorch_root.clone());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Url::from_file_path(&pytorch_root).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    let nn_init_path_str = "torch/nn/__init__.py";
    let nn_init_path = pytorch_root.join(nn_init_path_str);
    let backward_path_str = "torch/distributed/pipelining/_backward.py";
    let backward_path = pytorch_root.join(backward_path_str);

    interaction.client.did_open(nn_init_path_str);
    interaction.client.did_open(backward_path_str);

    // Wait for the server to finish processing the did_open notifications.
    // We wait for diagnostics on nn_init_path first.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(nn_init_path.clone(), 0)
        .unwrap();
    // Then wait for diagnostics on backward_path. This file imports from torch.nn,
    // so its diagnostics will only be published after the server has finished
    // populating all files in the project path and resolving all imports.
    interaction
        .client
        .expect_publish_diagnostics_for_file(backward_path.clone())
        .unwrap();

    // Save the original content so we can restore it after the test
    let original_content = read_to_string(&nn_init_path).unwrap();
    let modified_content = original_content.replace("    Parameter as Parameter,\n", "");

    eprintln!("Removing 'Parameter as Parameter' export...");
    // Write the modified content to disk and notify via file watcher
    // (dependencies only work on saved files, not in-memory changes)
    interaction
        .client
        .did_change(nn_init_path_str, &modified_content);
    fs::write(&nn_init_path, modified_content).unwrap();
    interaction.client.file_modified(nn_init_path_str);
    interaction.client.did_save(nn_init_path_str);

    eprintln!("Starting timer...");

    let start = Instant::now();

    if let Err(e) = interaction
        .client
        .expect_publish_diagnostics_eventual_message_contains(
            backward_path,
            "Could not import `Parameter` from `torch.nn`",
        )
    {
        // Attempt to restore original content before propagating the error
        let _ = fs::write(&nn_init_path, &original_content);
        // Re-propagate the original error
        panic!("Diagnostics expectation failed: {}", e);
    }

    let elapsed = start.elapsed();
    eprintln!("\n========================================");
    eprintln!("  Total time: {:?}", elapsed);
    eprintln!("========================================\n");

    // Restore the original file content
    fs::write(&nn_init_path, &original_content).unwrap();

    interaction.shutdown().unwrap();
}
