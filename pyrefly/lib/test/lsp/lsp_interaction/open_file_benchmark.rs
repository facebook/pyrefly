/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Benchmark: time-to-first-diagnostics and peak memory when opening a file.
//!
//! This measures the cold-start latency a developer experiences when opening a
//! single (typically large) file in a project. The timer covers `didOpen` until
//! the opened file's `publishDiagnostics` arrives — the first thing the IDE
//! shows, and the earliest observable server output (`didOpen` itself has no
//! reply). Peak RSS is sampled from `/proc/self/status` afterwards, since the
//! server runs in a thread inside this test process.
//!
//! The benchmark is project-agnostic: the workspace root and target file are
//! supplied via environment variables, so no specific repository or file path is
//! committed here.
//!
//! - `PYREFLY_BENCH_ROOT` — workspace root (the project the config is discovered
//!   from).
//! - `PYREFLY_BENCH_FILE` — path to the file to open, relative to the root.
//!
//! When either variable is unset the benchmark is skipped.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::thread::available_parallelism;
use std::time::Instant;

use lsp_types::Url;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;

/// Peak resident set size of this process in bytes, read from `VmHWM` in
/// `/proc/self/status`. The LSP server runs in a thread of this process, so its
/// peak memory is reflected here.
fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let line = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .expect("VmHWM entry present in /proc/self/status");
    let kb: u64 = line
        .split_whitespace()
        .nth(1)
        .and_then(|n| n.parse().ok())
        .expect("VmHWM value is a number in kB");
    kb * 1024
}

/// Open the file named by `PYREFLY_BENCH_FILE` (relative to `PYREFLY_BENCH_ROOT`)
/// and measure time-to-first-diagnostics and peak RSS.
///
/// This is `#[ignore]`d, so run it explicitly (build in release for realistic
/// numbers):
/// ```text
/// PYREFLY_BENCH_ROOT=/path/to/project PYREFLY_BENCH_FILE=relative/path.py \
///   cargo test --release test_open_file_time_to_first_diagnostics -- \
///   --ignored --nocapture
/// ```
#[test]
#[ignore] // Manual benchmark; requires PYREFLY_BENCH_ROOT and PYREFLY_BENCH_FILE.
fn test_open_file_time_to_first_diagnostics() {
    let (Ok(root), Ok(file)) = (
        std::env::var("PYREFLY_BENCH_ROOT"),
        std::env::var("PYREFLY_BENCH_FILE"),
    ) else {
        eprintln!(
            "Skipping benchmark: set PYREFLY_BENCH_ROOT (workspace root) and \
             PYREFLY_BENCH_FILE (path relative to root) to run it."
        );
        return;
    };
    let root = PathBuf::from(root);
    let file_path = root.join(&file);
    assert!(
        file_path.exists(),
        "Target file not found at {}",
        file_path.display()
    );

    let args = LspArgs {
        // No indexing: this disables the project-wide population that `did_open`
        // would otherwise trigger (checking every file in the config, which both
        // dominates memory and races our peak-memory read). The opened file is
        // still checked and its imports still resolve via the source DB below, so
        // this isolates the deterministic cost of opening one file.
        indexing_mode: IndexingMode::None,
        workspace_indexing_limit: 0,
        // Block on the build system's source DB so imports resolve, matching the
        // realistic IDE experience rather than fallback heuristics.
        build_system_blocking: true,
    };
    // Use every available core. `ThreadCount::AllThreads` caps at 64, so to truly
    // use the maximum on a high-core box we pass the raw core count explicitly.
    let cores = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let thread_count = ThreadCount::NumThreads(NonZeroUsize::new(cores).unwrap());
    let mut interaction =
        LspInteraction::new_with_args(args, NoTelemetry, Some(thread_count), None);
    interaction.set_root(root.clone());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            workspace_folders: Some(vec![(
                "workspace".to_owned(),
                Url::from_file_path(&root).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    // Start timing: a developer opens the file. `didOpen` is a notification with
    // no reply, so we wait on the first server output — the opened file's
    // diagnostics, which is the first thing the IDE shows.
    let text = std::fs::read_to_string(&file_path).unwrap();
    let uri = Url::from_file_path(&file_path).unwrap();
    let start = Instant::now();
    interaction.client.did_open_uri(&uri, "python", text);
    interaction
        .client
        .expect_publish_diagnostics_for_file(file_path.clone())
        .unwrap();
    let elapsed = start.elapsed();

    let peak = peak_rss_bytes();

    eprintln!("\n========================================");
    eprintln!("  open-file benchmark: {}", file);
    eprintln!("  Threads:                      {}", cores);
    eprintln!("  Time to first diagnostics:    {:?}", elapsed);
    eprintln!(
        "  Peak memory (VmHWM):          {:.2} GiB ({} bytes)",
        peak as f64 / (1024.0 * 1024.0 * 1024.0),
        peak
    );
    eprintln!("========================================\n");

    interaction.shutdown().unwrap();
}
