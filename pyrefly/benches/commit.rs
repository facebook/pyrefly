/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! What `State::commit_transaction` costs, over a synthetic project.
//!
//! Committing takes the `State` write lock, which blocks every reader including
//! in-flight LSP requests, and then frees whatever the new state displaced.
//!
//! Loading a module copies its data out of the committed state, so a transaction
//! pays for every module it touches even when nothing about that module changed
//! — the commit writes all of them back. The three scenarios sweep from that
//! floor up to a full rebuild:
//!
//! - `clean` rechecks the project without changing anything, so the commit
//!   writes back every module and frees none of them.
//! - `invalidate_find` announces a created file, which is what the language
//!   server does on any watcher create or delete. Every module is marked
//!   dirty-find and every `LoaderFindCache` is discarded, so every import is
//!   re-resolved; modules still only rebuild if something resolves differently.
//! - `invalidate_all` rewrites every module, so the commit also holds the last
//!   reference to each one's previous AST, bindings, answers and solutions and
//!   frees them. Rare in practice, kept as the upper bound.
//!
//! The project is synthetic, but each module carries enough — a class with
//! methods, a few functions, imports of several earlier modules — to produce
//! real AST nodes, bindings, answers and solutions and a non-trivial dep graph.
//! Those are what a commit writes back and frees, so the costs here are the same
//! ones a real project pays.
//!
//! ```text
//! buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:commit_bench -- --bench
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::Bencher;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dupe::Dupe;
use pyrefly::state::load::FileContents;
use pyrefly::state::require::Require;
use pyrefly::state::state::CommittingTransaction;
use pyrefly::state::state::State;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::timer::set_timing_enabled;

/// Modules in the synthetic project. Large enough that per-module costs
/// dominate, small enough that a sample stays cheap enough to take many of.
const MODULES: usize = 500;

/// How many earlier modules each one imports. Gives every module a non-trivial
/// `deps`/`rdeps` map, which is cloned into the transaction and freed at commit.
const FANIN: usize = 3;

/// One module's source. Distinct per index, so no two modules can share derived
/// data, and substantial enough that its solutions cost something to free.
fn source(index: usize) -> String {
    let mut out = String::new();
    for dep in imports_of(index) {
        out.push_str(&format!("from mod{dep} import Value{dep}\n"));
    }
    out.push_str(&format!(
        "
class Value{index}:
    x: int
    y: str

    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    def combine(self, other: \"Value{index}\") -> \"Value{index}\":
        return Value{index}(self.x + other.x, self.y + other.y)

    def scaled(self, factor: int) -> list[int]:
        return [self.x * k for k in range(factor)]

def build{index}(count: int) -> list[Value{index}]:
    return [Value{index}(k, str(k)) for k in range(count)]

def total{index}(values: list[Value{index}]) -> int:
    return sum(v.x for v in values)
"
    ));
    for dep in imports_of(index) {
        out.push_str(&format!(
            "
def use{index}_{dep}(v: Value{dep}) -> int:
    return v.x
"
        ));
    }
    out
}

/// The earlier modules `index` imports.
fn imports_of(index: usize) -> impl Iterator<Item = usize> {
    let start = index.saturating_sub(FANIN);
    start..index
}

fn module_name(index: usize) -> ModuleName {
    ModuleName::from_string(format!("mod{index}"))
}

fn module_path(index: usize) -> ModulePath {
    ModulePath::memory(memory_path(index))
}

/// A `State` over the synthetic project, already checked and committed once, so
/// the measured commits displace real data rather than filling an empty map.
fn checked_state() -> (State, Vec<Handle>) {
    set_timing_enabled(false);
    let mut config = ConfigFile::default();
    // Pin the environment rather than inheriting whatever interpreter the host
    // has, so the benchmark is reproducible — and so `get_sys_info` below has
    // the values `configure` would otherwise have had to discover.
    config.python_environment.python_version = Some(PythonVersion::default());
    config.python_environment.python_platform = Some(PythonPlatform::default());
    config.python_environment.site_package_path = Some(Vec::new());
    let mut sourcedb = MapDatabase::new(config.get_sys_info());
    for index in 0..MODULES {
        sourcedb.insert(module_name(index), module_path(index));
    }
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    // The benchmark must not depend on whatever interpreter happens to be on the
    // host, and the modules import nothing outside themselves.
    config.interpreters.skip_interpreter_query = true;
    config.configure();
    let sys_info = config.get_sys_info();
    let config = ArcId::new(config);

    let handles = (0..MODULES)
        .map(|index| Handle::new(module_name(index), module_path(index), sys_info.dupe()))
        .collect::<Vec<_>>();

    // Only setup uses the thread pool: each measured span is a single
    // `commit_transaction`, which runs entirely on the calling thread. Checking
    // on the pool, as the language server does, means the commit frees data
    // that other threads allocated.
    let state = State::new(ConfigFinder::new_constant(config), ThreadCount::AllThreads);
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().set_memory(
        (0..MODULES)
            .map(|index| (memory_path(index), Some(Arc::new(contents(index)))))
            .collect(),
    );
    transaction.as_mut().run(&handles, Require::Errors, None);
    // A module that fails to load or resolve its imports would make every
    // commit nearly free without failing, so require the project to check clean.
    let errors = transaction.as_ref().get_errors(&handles).collect_errors();
    assert!(
        errors.ordinary.is_empty(),
        "synthetic project should check without errors, got {}: {:?}",
        errors.ordinary.len(),
        errors.ordinary.first(),
    );
    state.commit_transaction(transaction, None);
    (state, handles)
}

fn memory_path(index: usize) -> PathBuf {
    PathBuf::from(format!("/synthetic/mod{index}.py"))
}

fn contents(index: usize) -> FileContents {
    FileContents::from_source(source(index))
}

/// Recheck the project and leave a transaction ready to commit. This is the
/// `iter_batched_ref` setup, so none of it is timed.
fn prepare<'a>(state: &'a State, handles: &[Handle]) -> CommittingTransaction<'a> {
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().run(handles, Require::Errors, None);
    transaction
}

/// Rewrite every module so the whole project rebuilds, then recheck. Also the
/// `iter_batched_ref` setup, so untimed.
///
/// `nonce` must differ per call: `set_memory` compares contents and ignores a
/// write that changes nothing, so reusing text would leave the project clean and
/// silently turn this into [`prepare`].
fn prepare_all<'a>(
    state: &'a State,
    handles: &[Handle],
    nonce: usize,
) -> CommittingTransaction<'a> {
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().set_memory(
        (0..MODULES)
            .map(|index| {
                let mut text = source(index);
                text.push_str(&format!("\n_touch{nonce}: int = {nonce}\n"));
                (
                    memory_path(index),
                    Some(Arc::new(FileContents::from_source(text))),
                )
            })
            .collect(),
    );
    transaction.as_mut().run(handles, Require::Errors, None);
    transaction
}

/// Announce a created file, so `invalidate_events` triggers `invalidate_find`.
/// Also the `iter_batched_ref` setup, so untimed.
fn prepare_find<'a>(
    state: &'a State,
    handles: &[Handle],
    nonce: usize,
) -> CommittingTransaction<'a> {
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().invalidate_events(&CategorizedEvents {
        created: vec![PathBuf::from(format!("/synthetic/created{nonce}.py"))],
        ..Default::default()
    });
    transaction.as_mut().run(handles, Require::Errors, None);
    transaction
}

/// Time committing each transaction that `setup` prepares.
///
/// This uses `iter_batched_ref` because a `CommittingTransaction` holds the
/// committing-transaction lock and a read lock on `State` until it is committed.
/// CodSpeed's `iter_batched` calls `setup` a second time on the measured pass
/// while the first input is still alive, so the second `setup` would block on
/// the committing lock forever. `iter_batched_ref` passes each input to the
/// routine exactly once, which the `expect` below relies on; an extra call
/// panics rather than timing an empty commit.
fn commit_each<'a>(
    b: &mut Bencher<'_>,
    state: &'a State,
    mut setup: impl FnMut() -> CommittingTransaction<'a>,
) {
    b.iter_batched_ref(
        || Some(setup()),
        |transaction| {
            let transaction = transaction
                .take()
                .expect("iter_batched_ref passes each input to the routine once");
            state.commit_transaction(transaction, None)
        },
        BatchSize::PerIteration,
    );
}

fn commit(c: &mut Criterion) {
    let (state, handles) = checked_state();

    let mut group = c.benchmark_group("commit");
    // Every sample re-runs the project in setup, so samples are not free; this is
    // as many as fit in a run of a few tens of seconds.
    group.sample_size(50);
    group.bench_function("clean", |b| {
        commit_each(b, &state, || prepare(&state, &handles))
    });
    let mut find_nonce = 0;
    group.bench_function("invalidate_find", |b| {
        commit_each(b, &state, || {
            find_nonce += 1;
            prepare_find(&state, &handles, find_nonce)
        })
    });
    let mut all_nonce = 0;
    group.bench_function("invalidate_all", |b| {
        commit_each(b, &state, || {
            all_nonce += 1;
            prepare_all(&state, &handles, all_nonce)
        })
    });
    group.finish();
}

criterion_group!(benches, commit);
criterion_main!(benches);
