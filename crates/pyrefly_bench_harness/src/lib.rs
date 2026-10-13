/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared harness for benchmarks that measure a fully-indexed project.
//!
//! `Require::Indexing` is the level the language server uses for the files it
//! indexes in the background, and it is what retains the find-references index
//! and the per-module symbol tables. Nothing in a batch `pyrefly check` reaches
//! it, so the cost of indexing is only visible to a benchmark that asks for it
//! explicitly.
//!
//! This lives in its own library because each benchmark is a separate
//! compilation unit and they cannot import each other's modules — the same
//! reason `pyrefly_lsp_test` exists. Benchmarks over different corpora (the
//! pinned PyTorch checkout, an internal repository) differ only in which
//! directory they point at, so they share [`index_project`] and
//! [`report_indexed_memory`] and supply the root.

use std::env::set_current_dir;
use std::fs::read_to_string;
use std::path::Path;

use pyrefly::commands::check::Handles;
use pyrefly::commands::files::FilesArgs;
use pyrefly::state::require::Require;
use pyrefly::state::state::State;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::thread_pool::ThreadCount;

/// A resident-set-size sample, in kB, read from `/proc/self/status`. `None` off
/// procfs platforms, so a benchmark that reports memory degrades to reporting
/// nothing rather than failing.
///
/// Sampling costs one small file read — tens of microseconds against benchmarks
/// that run for seconds — so it is safe to call from inside a timed region when
/// the measurement has to observe live state.
pub struct Rss {
    /// `VmRSS`: resident memory at the moment of the sample.
    pub current_kb: u64,
    /// `VmHWM`: the high-water mark over the whole process lifetime. Because it
    /// never decreases, a benchmark that runs several iterations reports the
    /// peak across all of them, not the peak of the last one.
    pub peak_kb: u64,
}

impl Rss {
    pub fn sample() -> Option<Self> {
        let status = read_to_string("/proc/self/status").ok()?;
        let field = |key: &str| {
            status
                .lines()
                .find_map(|l| l.strip_prefix(key)?.split_whitespace().next()?.parse().ok())
        };
        Some(Self {
            current_kb: field("VmRSS:")?,
            peak_kb: field("VmHWM:")?,
        })
    }
}

/// Index the project rooted at the current directory from cold: resolve it
/// (empty file list → project mode) and drive every project file to
/// `Require::Indexing` across all cores. Dependencies stay at the default
/// `Require::Exports`, matching the language server, which indexes the workspace
/// rather than its whole dependency closure.
///
/// `min_files` guards against silently measuring an empty index when project
/// discovery breaks; pass a floor comfortably below the corpus's real size.
///
/// Hands back the `State` so the caller can drop it outside any timed region,
/// along with the number of project files indexed.
pub fn index_project(min_files: usize, label: &str) -> (State, usize) {
    let (includes, config_finder, _upsell) =
        FilesArgs::get(Vec::new(), None, ConfigOverrideArgs::default(), None)
            .expect("resolving the project");
    let files = config_finder
        .checkpoint(includes.files_iter())
        .expect("listing project files");
    let handles = Handles::new(files);
    let (loaded_handles, _, _) = handles.all(&config_finder);
    assert!(
        loaded_handles.len() >= min_files,
        "{label}: expected at least {min_files} project files, got {} — project discovery is broken",
        loaded_handles.len()
    );

    let state = State::new(config_finder, ThreadCount::AllThreads);
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction
        .as_mut()
        .run(&loaded_handles, Require::Indexing, None);
    state.commit_transaction(transaction, None);

    (state, loaded_handles.len())
}

/// Index the corpus at `root` once and print the memory the indexed project
/// holds.
///
/// This is the whole body of a standalone per-corpus binary, and it has to stay
/// that way. RSS is a property of the process, not of the routine: `VmRSS`
/// counts whatever else the process still holds, and `VmHWM` is the high-water
/// mark over the process's whole life, so any earlier allocation-heavy work in
/// the same process inflates both. Measuring in a process that does nothing else
/// is what makes the figure describe this index.
///
/// Memory is reported rather than asserted, because RSS depends on the allocator
/// and the host, so a threshold would be flaky. The number's use is comparative —
/// run at two commits and diff the reported figures.
pub fn report_indexed_memory(root: &Path, min_files: usize, label: &str) {
    // Project mode resolves the project rooted at the cwd, so put the process
    // inside the corpus to exercise the real project-discovery path.
    set_current_dir(root).expect("cd into the benchmark corpus");

    let (state, files) = index_project(min_files, label);

    // Sampled before the state is dropped, so the figure describes a live index
    // rather than one being torn down.
    if let Some(rss) = Rss::sample() {
        println!(
            "{label}: files={files} current_rss_kB={} peak_rss_kB={}",
            rss.current_kb, rss.peak_kb,
        );
    }
    drop(state);
}
