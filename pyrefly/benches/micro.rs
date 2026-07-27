/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Microbenchmarks for Pyrefly type checking, run with the `criterion` benchmark
//! harness.
//!
//! Each case builds a synthetic Python snippet that stresses one part of the
//! checker (enum member resolution, exhaustiveness, protocol structural matching,
//! narrowing, gradual-typing calls, type-variable joins, inferred typed dicts,
//! overload resolution) and times a single in-memory check of it. `SHARED_STATE`
//! pre-initializes the stdlib once, so only the snippet's check is measured, and
//! each case asserts its expected error count up front so a scenario that stops
//! exercising the intended path fails loudly instead of silently measuring
//! nothing.
//!
//! Build mode matters: must be optimized. Buck requires `@fbcode//mode/opt`
//! (or `opt-clang-thinlto` for final numbers); Cargo `cargo bench` builds the
//! optimized bench profile (release-like) by default, `cargo run` needs `--release`.
//!
//! Run with cargo: `cargo bench -p pyrefly --bench micro`
//! Run with buck: `buck run @fbcode//mode/opt fbcode//pyrefly/pyrefly:micro_bench -- --bench`

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dupe::Dupe;
use pyrefly::state::load::FileContents;
use pyrefly::state::require::Require;
use pyrefly::state::state::State;
use pyrefly_build::handle::Handle;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::timer::set_timing_enabled;

const BENCH_FILE: &str = "bench.py";

/// Single-threaded state with stdlib pre-initialized.
static SHARED_STATE: LazyLock<State> = LazyLock::new(|| {
    // Disable the type checker's profiling timers: each `Instant::now()` is a
    // `clock_gettime` syscall under instrumentation and the benchmark doesn't need them.
    set_timing_enabled(false);
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::default());
    let config = {
        let mut c = ConfigFile::default();
        c.python_environment.python_version = Some(PythonVersion::default());
        c.python_environment.python_platform = Some(PythonPlatform::default());
        // Skip interpreter discovery: `configure()` otherwise spawns `python3` to
        // probe site-packages. The snippets only use stdlib from bundled typeshed.
        c.interpreters.skip_interpreter_query = true;
        c.python_environment.site_package_path = Some(Vec::new());
        c.configure();
        ArcId::new(c)
    };
    let finder = ConfigFinder::new_constant(config);
    // Inline (no rayon pool): a pooled run would block this thread on a `futex`
    // while a worker does the check. Inline keeps the whole check on this thread.
    let state = State::new(finder, ThreadCount::Inline);
    // Force stdlib init by running an empty module.
    let h = Handle::new(
        ModuleName::from_str("_bench_init"),
        ModulePath::memory(PathBuf::from("_bench_init.py")),
        sys_info,
    );
    let mut t = state.new_committable_transaction(Require::Errors, None);
    t.as_mut().set_memory(vec![(
        PathBuf::from("_bench_init.py"),
        Some(Arc::new(FileContents::from_source(String::new()))),
    )]);
    t.as_mut().run(&[h], Require::Errors, None);
    state.commit_transaction(t, None);
    state
});

/// Run a type check on the given Python code and return the error count.
fn check_code(code: Arc<FileContents>) -> usize {
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::default());
    let h = Handle::new(
        ModuleName::from_str("bench"),
        ModulePath::memory(PathBuf::from(BENCH_FILE)),
        sys_info,
    );
    let mut t = SHARED_STATE.transaction();
    t.set_memory(vec![(PathBuf::from(BENCH_FILE), Some(code))]);
    t.run(&[h.dupe()], Require::Errors, None);
    let errors = t.get_errors([&h]);
    errors.collect_errors().ordinary.len()
}

/// Join a range `0..n` into a single string, formatting each index with `f` and
/// separating with `sep`. Most snippet generators are just a few of these joins
/// stitched together with fixed headers.
fn joined(n: usize, sep: &str, mut f: impl FnMut(usize) -> String) -> String {
    let mut out = String::new();
    for i in 0..n {
        if i != 0 {
            out.push_str(sep);
        }
        out.push_str(&f(i));
    }
    out
}

/// Enum with `count` members followed by a read of every member. Exercises enum
/// member resolution; well-typed, so no errors.
fn enum_member_reads(count: usize) -> String {
    let members = joined(count, "\n", |i| format!("    K{i} = {i}"));
    let reads = joined(count, "\n", |i| format!("_ = Palette.K{i}"));
    format!("from enum import Enum\nclass Palette(Enum):\n{members}\n{reads}")
}

/// Exhaustive `match` over a `count`-member enum with `assert_never` in the
/// wildcard arm. Because every member is covered, the wildcard is unreachable
/// and the snippet type checks clean.
fn exhaustive_match(count: usize) -> String {
    let members = joined(count, "\n", |i| format!("    S{i} = {i}"));
    let arms = joined(count, "\n", |i| {
        format!("        case Phase.S{i}:\n            return {i}")
    });
    format!(
        "from enum import Enum\nfrom typing import assert_never\n\
         class Phase(Enum):\n{members}\n\
         def classify(p: Phase) -> int:\n    match p:\n{arms}\n        case _:\n            assert_never(p)"
    )
}

/// A `Protocol` with `methods` members and a class implementing all of them with
/// the wrong return type, assigned to the protocol `bindings` times. Each binding
/// is one structural-mismatch error, so the snippet is expected to report
/// exactly `bindings` errors.
fn protocol_binding_mismatch(methods: usize, bindings: usize) -> String {
    let decls = joined(methods, "\n", |i| {
        format!("    def step{i}(self) -> int: ...")
    });
    let wrong = joined(methods, "\n", |i| {
        format!("    def step{i}(self) -> str:\n        return \"\"")
    });
    let uses = joined(bindings, "\n", |i| format!("bound{i}: Runner = Concrete()"));
    format!(
        "from typing import Protocol\n\
         class Runner(Protocol):\n{decls}\n\
         class Concrete:\n{wrong}\n{uses}"
    )
}

/// Union of `count` dataclasses read back through an exhaustive class-pattern
/// `match`. Exercises union construction plus pattern narrowing; no errors.
fn union_pattern_read(count: usize) -> String {
    let classes = joined(count, "\n", |i| {
        format!("@dataclass\nclass Node{i}:\n    weight: int")
    });
    let members = joined(count, " | ", |i| format!("Node{i}"));
    let arms = joined(count, "\n", |i| {
        format!("        case Node{i}():\n            return node.weight")
    });
    format!(
        "from dataclasses import dataclass\n{classes}\n\
         def total(node: {members}) -> int:\n    match node:\n{arms}"
    )
}

/// An `if`/`elif` `isinstance` ladder narrowing a union of `count` classes down
/// to a `str` return. Exercises isinstance-based narrowing; no errors.
fn isinstance_chain(count: usize) -> String {
    let classes = joined(count, "\n", |i| format!("class Tag{i}: ..."));
    let members = joined(count, " | ", |i| format!("Tag{i}"));
    let branches = joined(count, "\n", |i| {
        let kw = if i == 0 { "if" } else { "elif" };
        format!("    {kw} isinstance(value, Tag{i}):\n        return \"Tag{i}\"")
    });
    format!(
        "{classes}\n\
         def name_of(value: {members}) -> str:\n{branches}\n    return \"unknown\""
    )
}

/// A `*args: Any` function invoked with `count` positional int literals. Gradual
/// typing accepts the call, so no errors.
fn variadic_any_call(count: usize) -> String {
    let args = joined(count, ", ", |i| i.to_string());
    format!("from typing import Any\ndef accept(*args: Any) -> None: ...\naccept({args})")
}

/// A generic function with `count` `T`-typed parameters called with a rotating
/// mix of literal types, forcing the solver to join them into a single `T`. No
/// errors.
fn typevar_join(count: usize) -> String {
    let params = joined(count, ", ", |i| format!("p{i}: T"));
    let args = joined(count, ", ", |i| match i % 4 {
        0 => i.to_string(),
        1 => format!("\"lit{i}\""),
        2 => format!("{i}.25"),
        _ => "False".to_owned(),
    });
    format!(
        "from typing import TypeVar\nT = TypeVar(\"T\")\ndef unify({params}) -> T: ...\nunify({args})"
    )
}

/// A dict literal with `count` string keys and rotating value types, which
/// Pyrefly infers as an anonymous TypedDict. Exercises that inference; no errors.
fn inferred_typed_dict(count: usize) -> String {
    let entries = joined(count, ", ", |i| {
        let value = match i % 4 {
            0 => i.to_string(),
            1 => format!("\"val{i}\""),
            2 => "True".to_owned(),
            _ => "None".to_owned(),
        };
        format!("\"field{i}\": {value}")
    });
    format!("mapping = {{{entries}}}")
}

/// `overloads` `@overload` signatures, each with a distinct parameter type, plus
/// `calls` calls that rotate through matching argument literals so every branch
/// is resolved. Exercises overload dispatch; no errors.
fn overload_resolution(overloads: usize, calls: usize) -> String {
    const TYPES: [&str; 10] = [
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "list[int]",
        "None",
        "tuple[int, ...]",
        "dict[str, int]",
        "set[int]",
    ];
    const ARGS: [&str; 10] = [
        "7",
        "\"a\"",
        "1.5",
        "False",
        "b\"z\"",
        "[9]",
        "None",
        "(4, 5)",
        "{\"k\": 6}",
        "{1, 3}",
    ];
    let mut src = String::from("from typing import overload\n");
    for i in 0..overloads {
        let ty = TYPES[i % TYPES.len()];
        let _ = write!(src, "@overload\ndef choose(x: {ty}) -> {ty}: ...\n");
    }
    src.push_str("def choose(x): return x\n");
    for i in 0..calls {
        let _ = write!(src, "r{i} = choose({})\n", ARGS[i % ARGS.len()]);
    }
    src
}

/// Type-check `source` once to assert it produces `expected_errors`, then
/// register a criterion benchmark that repeats the check. The up-front assertion
/// guards against a scenario silently drifting to a different error count (and
/// therefore measuring something other than intended).
fn measure(c: &mut Criterion, name: &str, source: String, expected_errors: usize) {
    let code = Arc::new(FileContents::from_source(source));
    assert_eq!(
        check_code(code.dupe()),
        expected_errors,
        "benchmark `{name}` produced an unexpected error count"
    );
    c.bench_function(name, |b| b.iter(|| check_code(code.dupe())));
}

/// Smoke benchmark validating the harness end-to-end.
fn smoke(c: &mut Criterion) {
    measure(c, "smoke", "x: int = 1".to_owned(), 0);
}

fn enum_members(c: &mut Criterion) {
    measure(c, "enum_member_reads_512", enum_member_reads(512), 0);
}

fn enum_exhaustiveness(c: &mut Criterion) {
    measure(c, "exhaustive_match_48", exhaustive_match(48), 0);
}

fn protocol_mismatch(c: &mut Criterion) {
    // 10 bindings each assign a structurally-incompatible impl, so 10 errors.
    measure(
        c,
        "protocol_binding_mismatch_100x10",
        protocol_binding_mismatch(100, 10),
        10,
    );
}

fn union_narrowing(c: &mut Criterion) {
    measure(c, "union_pattern_read_32", union_pattern_read(32), 0);
}

fn isinstance_narrowing(c: &mut Criterion) {
    measure(c, "isinstance_chain_64", isinstance_chain(64), 0);
}

fn vararg_call(c: &mut Criterion) {
    measure(c, "variadic_any_call_256", variadic_any_call(256), 0);
}

fn typevar_mapping(c: &mut Criterion) {
    measure(c, "typevar_join_256", typevar_join(256), 0);
}

fn anon_typed_dict(c: &mut Criterion) {
    measure(c, "inferred_typed_dict_64", inferred_typed_dict(64), 0);
}

fn overloads(c: &mut Criterion) {
    measure(
        c,
        "overload_resolution_10x20",
        overload_resolution(10, 20),
        0,
    );
}

criterion_group!(
    benches,
    smoke,
    enum_members,
    enum_exhaustiveness,
    protocol_mismatch,
    union_narrowing,
    isinstance_narrowing,
    vararg_call,
    typevar_mapping,
    anon_typed_dict,
    overloads,
);
criterion_main!(benches);
