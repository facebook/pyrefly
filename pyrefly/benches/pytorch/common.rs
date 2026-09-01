/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared harness for the PyTorch walltime benchmarks. The interactive cases
//! drive the real LSP server over all cores against a pinned, multi-gigabyte
//! PyTorch checkout, so they share the code that acquires
//! that checkout and the standard LSP args. These are walltime benchmarks —
//! threads, I/O, a long cold start — distinct from the deterministic,
//! single-threaded `micro` benchmarks.
//!
//! PyTorch is pinned to a single commit in `benches/pytorch_pin.bzl` (the same
//! file `BUCK` loads). Two providers feed the benches (see
//! [`pytorch_root_or_skip`]):
//! - internal (buck): `pyrefly/BUCK` fetches the pinned tarball from Manifold via
//!   `http_archive` and declares it as a resource of the benchmark target, so CI
//!   needs no github egress.
//! - OSS (cargo): there is no resource, so the bench shallow-clones PyTorch at
//!   the pinned rev into a per-rev cache dir the way `mypy_primer` grabs its
//!   inputs — no git submodule. When neither is available the benchmark skips,
//!   so `cargo bench` still works in a bare checkout.

use std::env::var;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;

/// PyTorch repo the OSS bench clones its pinned input from.
const PYTORCH_REPO: &str = "https://github.com/pytorch/pytorch.git";
/// The pin file `BUCK` also loads; we parse the rev out of it so the commit lives
/// in exactly one place. Bump it with `benches/update_revision.sh`.
const PYTORCH_PIN_BZL: &str = include_str!("../pytorch_pin.bzl");

/// File deep in the import graph the benchmarks navigate from / propagate to.
pub const BACKWARD: &str = "torch/distributed/pipelining/_backward.py";

/// Manual override pointing the benchmarks at an existing checkout, taking
/// precedence over every other source.
const BENCH_PATH_ENV: &str = "PYREFLY_PYTORCH_BENCH_PATH";

/// Marks a complete checkout. `clone_pytorch` renames into place last, so its
/// presence also rules out a half-populated tree from an interrupted clone.
const SENTINEL: &str = "torch/nn/__init__.py";

/// Pinned PyTorch commit, parsed from the `PYTORCH_BENCH_REV = "..."` line in
/// [`PYTORCH_PIN_BZL`].
fn pytorch_rev() -> &'static str {
    PYTORCH_PIN_BZL
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("PYTORCH_BENCH_REV = \"")?
                .strip_suffix('"')
        })
        .expect("PYTORCH_BENCH_REV not found in pytorch_pin.bzl")
}

/// Standard LSP args for the benchmarks.
///
/// `IndexingMode::None`: the benchmarks don't need the background workspace
/// index — the measured operations resolve through the opened files' import
/// closure on demand. Skipping the workspace index removes competing background
/// work, so the walltime reflects the operation under test rather than indexing
/// noise.
pub fn lsp_args() -> LspArgs {
    LspArgs {
        indexing_mode: IndexingMode::None,
        workspace_indexing_limit: 50,
        build_system_blocking: false,
    }
}

/// Checkout materialized by buck for the internal benchmark target, or `None`
/// outside an internal build.
///
/// The target declares the `http_archive` as a resource, so buck materializes it
/// alongside the binary and reports its path here at run time. That is a
/// supported runtime interface, so unlike reading the path out of the `buck-out`
/// layout it stays correct when rustc ran on a remote executor and the build
/// host's paths differ from this one's.
///
/// Buck always supplies that resource, so a lookup that fails or does not name a
/// checkout is a build fault, and this panics rather than returning `None`. The
/// acquisition fallbacks below exist for the open-source build and cannot help
/// here — falling through to them would end in a skip, and a skip exits 0 and
/// reads like a pass, which is the failure this whole function exists to remove.
#[cfg(fbcode_build)]
fn buck_resource_root() -> Option<PathBuf> {
    let root = buck_resources::get("pyrefly/pyrefly/pytorch-bench-src")
        .expect("the benchmark target declares pytorch-bench-src as a resource");
    assert!(
        root.join(SENTINEL).exists(),
        "the pytorch-bench-src resource at {} is not a PyTorch checkout: no {SENTINEL}",
        root.display(),
    );
    Some(root)
}

#[cfg(not(fbcode_build))]
fn buck_resource_root() -> Option<PathBuf> {
    None
}

/// Root of the pinned PyTorch checkout, or `None` if it can't be obtained.
///
/// Internally (buck, no github egress) the checkout is the Manifold
/// `http_archive` materialized as a resource of the benchmark target. Otherwise
/// (OSS `cargo bench`) the pinned source is shallow-cloned into a per-rev cache
/// dir under the system temp dir and reused by later runs. [`BENCH_PATH_ENV`]
/// overrides both. The returned path is a PyTorch repo root containing `torch/`.
fn pytorch_root() -> Option<PathBuf> {
    // Checked before the resource so that pointing the benchmarks at your own
    // checkout stays possible internally, where consulting the resource would
    // otherwise panic on a build whose resource is broken. An override that does
    // not name a checkout falls through rather than failing the run.
    let override_root = var(BENCH_PATH_ENV)
        .ok()
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    if let Some(root) = override_root
        && root.join(SENTINEL).exists()
    {
        return Some(root);
    }

    if let Some(root) = buck_resource_root() {
        return Some(root);
    }

    // Only an open-source build reaches here: no resource was configured, so the
    // pinned source has to be acquired directly. This is the one path where
    // coming up empty is an unconfigured environment rather than a fault, and so
    // the one path allowed to end in a skip.
    let rev = pytorch_rev();
    let root = std::env::temp_dir().join(format!("pyrefly-pytorch-bench-{rev}"));
    if root.join(SENTINEL).exists() {
        return Some(root);
    }
    clone_pytorch(rev, &root).then_some(root)
}

/// Shallow-clone PyTorch at `rev`, returning whether it succeeded. Clones into a
/// staging dir and renames into place only on success, so the sentinel file
/// [`pytorch_root`] checks can never appear on a half-populated (interrupted) clone.
fn clone_pytorch(rev: &str, dest: &Path) -> bool {
    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);
    let _ = fs::remove_dir_all(dest);

    let cloned = run(Command::new("git")
        .args([
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            PYTORCH_REPO,
        ])
        .arg(&staging))
        && run(Command::new("git")
            .arg("-C")
            .arg(&staging)
            .args(["fetch", "--depth", "1", "origin", rev]))
        && run(Command::new("git")
            .arg("-C")
            .arg(&staging)
            .args(["checkout", rev]));

    cloned && fs::rename(&staging, dest).is_ok()
}

/// Run `cmd` inheriting stdio (so git's progress and errors reach the user), and
/// report whether it exited successfully.
fn run(cmd: &mut Command) -> bool {
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

/// Like [`pytorch_root`], but prints a skip notice when the checkout can't be
/// obtained, so `cargo bench` still succeeds on a host without git or github egress.
pub fn pytorch_root_or_skip() -> Option<PathBuf> {
    let root = pytorch_root();
    if root.is_none() {
        eprintln!(
            "Skipping pytorch benchmark: could not obtain the pinned PyTorch \
             checkout. Needs git and github.com egress, or set \
             PYREFLY_PYTORCH_BENCH_PATH to an existing checkout."
        );
    }
    root
}
