/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Regression tests for `lookup_export_location` on a "cold" transaction.
//!
//! The TSP server resolves the source location of an exported name (e.g. for a
//! `TypeVar` reached via `convert_quantified`) through
//! `Server::resolve_export_location`, which runs `lookup_export_location` on a
//! fresh `state.transaction()`. A fresh transaction inherits its `Stdlib` map
//! from the last committed run, which is empty when no check has run yet (a cold
//! start). `lookup_export_location` demands the target module's exports, and
//! `demand` calls `get_stdlib`, which panics when the stdlib map is empty.
//!
//! The fix runs the target handle for `Require::Exports` first, which invokes
//! `compute_stdlib` and populates the map before the demand. These tests pin
//! that behavior: running first succeeds, and skipping the run reproduces the
//! original panic.

use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use ruff_python_ast::name::Name;

use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;

type Memory = Vec<(PathBuf, Option<Arc<FileContents>>)>;

/// Build a `State` that has never been run, so its committed `Stdlib` map is
/// empty, along with a handle and memory contents for a single in-memory module
/// that exports `X`. This reproduces the cold-start condition the TSP server
/// hits before any background check has populated stdlib.
fn cold_state() -> (State, Handle, Memory) {
    let mut env = TestEnv::new();
    env.add_with_path("foo", "foo.py", "X = 1\n");
    let sys_info = env.sys_info();
    let memory = env.get_memory();
    let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("foo"),
        ModulePath::memory(PathBuf::from("foo.py")),
        sys_info,
    );
    (state, handle, memory)
}

#[test]
fn test_lookup_export_location_after_run_does_not_panic() {
    let (state, handle, memory) = cold_state();
    let mut transaction = state.transaction();
    transaction.set_memory(memory);
    // Mirror `Server::resolve_export_location`: run the target handle for
    // `Require::Exports` so `compute_stdlib` populates the stdlib map before the
    // lookup demands it.
    transaction.run(&[handle.dupe()], Require::Exports, None);
    let location = transaction.lookup_export_location(&handle, &Name::new_static("X"));
    assert!(
        location.is_some(),
        "expected to resolve the location of `X` in the cold transaction"
    );
}

#[test]
#[should_panic]
fn test_lookup_export_location_without_run_panics_on_cold_transaction() {
    let (state, handle, memory) = cold_state();
    let mut transaction = state.transaction();
    transaction.set_memory(memory);
    // Without first running for `Require::Exports`, the cold transaction's stdlib
    // map is empty, so the demand triggered by the lookup panics in `get_stdlib`.
    // This is the bug that `resolve_export_location`'s `transaction.run(...)`
    // guards against.
    let _ = transaction.lookup_export_location(&handle, &Name::new_static("X"));
}
