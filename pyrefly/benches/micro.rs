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
//! overload resolution, nested generic construction, and the Polars/pandas
//! schema-tracking dispatch chain run against non-DataFrame receivers) and times a
//! single in-memory check of it. `SHARED_STATE`
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

use std::collections::HashSet;
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
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
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::suggest::Candidate;
use pyrefly_util::suggest::best_suggestion;
use pyrefly_util::suggest::char_mask;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::timer::set_timing_enabled;
use ruff_python_ast::name::Name;

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
        c.root
            .errors
            .get_or_insert_default()
            .set_error_severity(ErrorKind::ImplicitAnyLambda, Severity::Error);
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

fn check_module(code: Arc<FileContents>, module: &str, path: &str) -> usize {
    check_module_at(code, module, path, Require::Errors)
}

/// Check a module at a given require level.
///
/// Below `Require::Errors` the module is loaded with `ErrorStyle::Never`, so
/// every error it produces is discarded. It is still bound: binding is what
/// produces the types a dependent module reads, and binding is also where an
/// unresolved name is noticed. Work done there on behalf of an error is
/// therefore still reachable at this level, and still thrown away.
fn check_module_at(code: Arc<FileContents>, module: &str, path: &str, require: Require) -> usize {
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::default());
    let h = Handle::new(
        ModuleName::from_str(module),
        ModulePath::memory(PathBuf::from(path)),
        sys_info,
    );
    let mut t = SHARED_STATE.transaction();
    t.set_memory(vec![(PathBuf::from(path), Some(code))]);
    t.run(&[h.dupe()], require, None);
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

/// Each layer pushes `Base[object]` inward. Treating the leaf's soft diagnostic as
/// a failed contextual attempt makes every layer retry, producing exponential work.
fn nested_generic_constructor_soft_error(depth: usize) -> String {
    let mut expression = "Leaf(lambda x: 0)".to_owned();
    for _ in 0..depth {
        expression = format!("Box({expression})");
    }
    format!(
        r#"
from typing import Generic, TypeVar

T = TypeVar("T", covariant=True)

class Base(Generic[T]): ...

class Box(Base[T], Generic[T]):
    def __init__(self, value: Base[T]) -> None: ...

class Leaf(Base[T], Generic[T]):
    def __init__(self, value: T) -> None: ...

result: Base[object] = {expression}
"#
    )
}

/// `count` reassignments through a user method returning its own class. Every call walks the Polars
/// dispatch chain but falls through on a `ClassType` receiver, the primary gate that schema tracking
/// costs ordinary method calls nothing.
fn user_method_calls(count: usize) -> String {
    let calls = joined(count, "\n", |_| "c = c.m()".to_owned());
    format!("class C:\n    def m(self) -> \"C\":\n        return self\nc = C()\n{calls}")
}

/// `count` reassignments through builtin `str` methods, exercising the same dispatch chain on a
/// builtin receiver with no errors.
fn builtin_method_calls(count: usize) -> String {
    const METHODS: [&str; 4] = ["upper", "lower", "strip", "title"];
    let calls = joined(count, "\n", |i| {
        format!("s = s.{}()", METHODS[i % METHODS.len()])
    });
    format!("s = \"x\"\n{calls}")
}

/// A class whose method names collide with Polars DataFrame methods, called `count` times. The
/// receiver is a `ClassType`, never a `DataFrame`, so every Polars guard returns `None`.
fn method_name_collision_calls(count: usize) -> String {
    let methods = joined(4, "\n", |i| {
        let name = ["select", "filter", "agg", "group_by"][i];
        format!("    def {name}(self, x: str) -> \"Fake\": return self")
    });
    let calls = joined(count, "\n", |i| match i % 3 {
        0 => "f = f.select(\"a\")".to_owned(),
        1 => "f = f.filter(\"a\")".to_owned(),
        _ => "f = f.group_by(\"a\").agg(\"b\")".to_owned(),
    });
    format!("class Fake:\n{methods}\nf = Fake()\n{calls}")
}

/// `count` instantiations of a trivial class, exercising the constructor-call dispatch in `call.rs`
/// with no errors.
fn constructor_calls(count: usize) -> String {
    let calls = joined(count, "\n", |_| "k = K()".to_owned());
    format!("class K: ...\n{calls}")
}

fn pandas_method_calls(count: usize) -> String {
    const METHODS: [&str; 4] = ["head", "drop", "rename", "copy"];
    let calls = joined(count, "\n", |i| {
        format!("df.{}()", METHODS[i % METHODS.len()])
    });
    format!(
        "class DataFrame:\n    def __init__(self, data: object = None) -> None: ...\n    def head(self) -> \"DataFrame\": ...\n    def drop(self) -> \"DataFrame\": ...\n    def rename(self) -> \"DataFrame\": ...\n    def copy(self) -> \"DataFrame\": ...\n\ndf = DataFrame({{\"a\": [1]}})\n{calls}"
    )
}

/// Type-check `source` once to assert it produces `expected_errors`, then
/// register a criterion benchmark that repeats the check. The up-front assertion
/// guards against a scenario silently drifting to a different error count (and
/// therefore measuring something other than intended).
fn measure(c: &mut Criterion, name: &str, source: String, expected_errors: usize) {
    measure_module(c, name, source, expected_errors, "bench", BENCH_FILE);
}

fn measure_module(
    c: &mut Criterion,
    name: &str,
    source: String,
    expected_errors: usize,
    module: &str,
    path: &str,
) {
    let code = Arc::new(FileContents::from_source(source));
    assert_eq!(
        check_module(code.dupe(), module, path),
        expected_errors,
        "benchmark `{name}` produced an unexpected error count"
    );
    c.bench_function(name, |b| b.iter(|| check_module(code.dupe(), module, path)));
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

fn nested_generic_constructor(c: &mut Criterion) {
    measure(
        c,
        "nested_generic_constructor_soft_error_12",
        nested_generic_constructor_soft_error(12),
        1,
    );
}

fn user_method_dispatch(c: &mut Criterion) {
    measure(c, "user_method_calls_256", user_method_calls(256), 0);
}

fn builtin_method_dispatch(c: &mut Criterion) {
    measure(c, "builtin_method_calls_256", builtin_method_calls(256), 0);
}

fn method_name_collision(c: &mut Criterion) {
    measure(
        c,
        "method_name_collision_calls_256",
        method_name_collision_calls(256),
        0,
    );
}

fn constructor_dispatch(c: &mut Criterion) {
    measure(c, "constructor_calls_256", constructor_calls(256), 0);
}

fn pandas_method_dispatch(c: &mut Criterion) {
    measure_module(
        c,
        "pandas_method_calls_256",
        pandas_method_calls(256),
        0,
        "pandas.core.frame",
        "pandas/core/frame.py",
    );
}
// ---------------------------------------------------------------------------
// Unknown-name suggestion search
// ---------------------------------------------------------------------------

/// The name every case searches for. Twelve characters, matching the real
/// unresolved names in the Configerator config that motivated this: they are
/// short thrift type names like `ColumnSchema` and `Int64ColumnType`.
const MISSING: &str = "ColumnSchema";

/// Names visible in the scope the missing name is looked up against.
///
/// `Spread` varies their length the way ordinary source does. `Long` makes
/// every one far longer than the missing name, which is the production shape:
/// a Thrift IDL wildcard-imported ~94,000 generated enum members, all around
/// forty characters, into a module scope. `SameLength` makes every one exactly
/// as long as the missing name, so no candidate can be rejected on length and
/// every single one reaches the distance computation.
enum ScopeShape {
    Spread,
    Long,
    SameLength,
}

/// One name of the given shape. Kept separate from `scope_names` so that the
/// whole-file cases below do not have to follow it as later commits change what
/// the direct cases need alongside each name.
fn scope_name(i: usize, shape: &ScopeShape) -> String {
    match shape {
        ScopeShape::Spread => format!("Cand{}{i:07}", "Q".repeat(i % 37)),
        ScopeShape::Long => format!("Cand{}{i:07}", "Q".repeat(29)),
        // Four, plus eight digits, is the length of `MISSING`.
        ScopeShape::SameLength => format!("Cand{i:08}"),
    }
}

/// Paired with each name's character length, the way the static scope records
/// it, so the benchmark measures the same work the binder does.
fn scope_names(count: usize, shape: &ScopeShape) -> Vec<(Name, u32, u32)> {
    (0..count)
        .map(|i| {
            let name = Name::new(scope_name(i, shape));
            let (char_len, mask) = name.as_str().chars().fold((0u32, 0u32), |(len, mask), c| {
                (len + 1, mask | char_mask(c))
            });
            (name, char_len, mask)
        })
        .collect()
}

/// Stand-in for the builtins wildcard, which every lookup searches after the
/// enclosing scopes. Typeshed's `builtins.pyi` exports a few hundred names and,
/// unlike generated scope names, they are short and varied.
fn builtin_names() -> Vec<Name> {
    (0..250)
        .map(|i| Name::new(format!("bltn_{}{i:03}", "z".repeat(i % 11))))
        .collect()
}

/// One `best_suggestion` call shaped like the real one: the names in scope
/// innermost first with their lengths already known, then the builtins at a
/// priority no scope can reach.
fn search(missing: &Name, scope: &[(Name, u32, u32)], builtins: &[Name]) -> Option<Name> {
    best_suggestion(
        missing,
        scope
            .iter()
            .enumerate()
            .map(|(i, (name, char_len, mask))| {
                Candidate::new(name, *char_len as usize, *mask, i % 4)
            })
            .chain(builtins.iter().map(|n| Candidate::measured(n, usize::MAX))),
    )
}

/// A whole file: `scope` names in scope, then `missing` references to a name
/// that is not.
///
/// The cases above call `best_suggestion` directly, which measures the search
/// itself. These reach it the way the binder does, so they also cover walking
/// the scopes to collect candidates and deciding whether the error is worth
/// reporting -- work that a direct call cannot see.
fn unresolved_names(scope: usize, missing: usize, shape: &ScopeShape) -> String {
    let mut source = String::new();
    for i in 0..scope {
        source.push_str(&format!("{} = 1\n", scope_name(i, shape)));
    }
    for i in 0..missing {
        source.push_str(&format!("_r{i} = {MISSING}\n"));
    }
    source
}

fn suggestion_whole_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("suggestion_whole_file");
    for (shape, label) in [
        (ScopeShape::Long, "unresolved_long"),
        (ScopeShape::Spread, "unresolved_spread"),
        (ScopeShape::SameLength, "unresolved_same_length"),
    ] {
        let name = format!("{label}_1k_scope_100_missing");
        let code = Arc::new(FileContents::from_source(unresolved_names(
            1000, 100, &shape,
        )));
        assert_eq!(
            check_module(code.dupe(), "bench", BENCH_FILE),
            100,
            "benchmark `{name}` produced an unexpected error count"
        );
        group.bench_function(&name, |b| {
            b.iter(|| check_module(code.dupe(), "bench", BENCH_FILE))
        });
    }
    group.finish();
}

/// The same shape as `unresolved_names`, except that the names bound to the
/// unresolved one are module-level exports rather than private.
///
/// A module below `Require::Errors` still solves its exported keys, so making
/// these exports is what guarantees the unresolved name is actually looked up
/// rather than left for a solve that never happens.
fn exported_unresolved_names(scope: usize, missing: usize, shape: &ScopeShape) -> String {
    let mut source = String::new();
    for i in 0..scope {
        let _ = writeln!(source, "{} = 1", scope_name(i, shape));
    }
    for i in 0..missing {
        let _ = writeln!(source, "r{i} = {MISSING}");
    }
    source
}

/// A file loaded below `Require::Errors`, where every error it produces is
/// discarded.
///
/// This is what a module pulled in behind the file the user is editing looks
/// like: bound, because something needs its types, but silenced. An unresolved
/// name in one is still noticed while binding, so working out what the author
/// might have meant is pure waste -- the answer has nowhere to go. The
/// benchmark is that waste, and it should cost almost nothing.
fn suggestion_discarded_errors(c: &mut Criterion) {
    let code = Arc::new(FileContents::from_source(exported_unresolved_names(
        1000,
        100,
        &ScopeShape::Long,
    )));
    // Checked, the same source has to produce one error per unresolved name, or
    // it no longer holds the names whose lookup this is here to measure.
    assert_eq!(
        check_module(code.dupe(), "bench", BENCH_FILE),
        100,
        "the source should hold one unresolved name per error"
    );
    assert_eq!(
        check_module_at(code.dupe(), "bench", BENCH_FILE, Require::Exports),
        0,
        "a module below `Require::Errors` should report nothing"
    );
    let mut group = c.benchmark_group("suggestion_whole_file");
    group.bench_function("discarded_errors_1k_scope_100_missing", |b| {
        b.iter(|| check_module_at(code.dupe(), "bench", BENCH_FILE, Require::Exports))
    });
    group.finish();
}

fn suggestion_search(c: &mut Criterion) {
    let builtins = builtin_names();
    let absent = Name::new(MISSING);

    // What almost every real lookup looks like: a handful of names in scope and
    // the builtins behind them. Worth measuring separately because the builtin
    // tail is a fixed cost paid on every unresolved name, however small the
    // file.
    let small = scope_names(200, &ScopeShape::Spread);
    c.bench_function("suggestion_small_scope", |b| {
        b.iter(|| black_box(search(black_box(&absent), &small, &builtins)))
    });

    // A genuine typo, so a match is found early and tightens the bound for
    // every candidate after it.
    let names = scope_names(100_000, &ScopeShape::Spread);
    let typo = Name::new(format!("Cand{}0000123", "Q".repeat(12)));
    c.bench_function("suggestion_typo_100k", |b| {
        b.iter(|| black_box(search(black_box(&typo), &names, &builtins)))
    });
}

// ---------------------------------------------------------------------------
// Suggestion search: real identifiers
// ---------------------------------------------------------------------------

/// Collect the identifiers in `source`, which needs no parser: the search only
/// ever sees names, and every name is an ASCII identifier run.
fn push_identifiers(source: &str, out: &mut HashSet<String>) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            // The search skips one-character candidates, so they are noise here.
            if i - start > 1 {
                out.insert(source[start..i].to_owned());
            }
        } else {
            i += 1;
        }
    }
}

/// Every identifier in the bundled stubs, ordered so the corpus is stable.
///
/// The generated shapes above differ only in a filler run and a digit tail, so
/// each one holds nearly the same characters as the next. That makes them an
/// honest test of the length filter and a misleading one for the character
/// mask, which rejects on precisely the variety they lack: measured against
/// these stubs a real name reaches the distance computation 6.6% of the time,
/// against the generated ones 58%. Only real names put that filter under the
/// pressure it sees in production.
static REAL_IDENTIFIERS: LazyLock<Vec<Name>> = LazyLock::new(|| {
    let mut set = HashSet::new();
    for contents in pyrefly_bundled::bundled_typeshed()
        .expect("bundled typeshed should extract")
        .values()
    {
        push_identifiers(contents, &mut set);
    }
    let (typeshed_third_party, _) = pyrefly_bundled::bundled_third_party_stubs()
        .expect("bundled typeshed third-party stubs should extract");
    for contents in typeshed_third_party.values() {
        push_identifiers(contents, &mut set);
    }
    for contents in pyrefly_bundled::bundled_third_party()
        .expect("bundled third-party stubs should extract")
        .values()
    {
        push_identifiers(contents, &mut set);
    }
    let mut names: Vec<String> = set.into_iter().collect();
    names.sort();
    names.into_iter().map(Name::new).collect()
});

/// `count` identifiers spread across the whole corpus. Striding rather than
/// taking a prefix keeps the sample from being all names beginning with `_`.
fn real_scope(count: usize) -> Vec<(Name, u32, u32)> {
    let stride = (REAL_IDENTIFIERS.len() / count).max(1);
    REAL_IDENTIFIERS
        .iter()
        .step_by(stride)
        .take(count)
        .map(|name| {
            let (char_len, mask) = name.as_str().chars().fold((0u32, 0u32), |(len, mask), c| {
                (len + 1, mask | char_mask(c))
            });
            (name.clone(), char_len, mask)
        })
        .collect()
}

/// The real builtins wildcard, which every lookup searches after the enclosing
/// scopes. The stand-in above is uniform where these are short and varied, and
/// this tail is paid on every unresolved name, so its character spread matters.
fn real_builtins() -> Vec<Name> {
    let files = pyrefly_bundled::bundled_typeshed().expect("bundled typeshed should extract");
    let contents = files
        .get(Path::new("builtins.pyi"))
        .expect("bundled typeshed should contain builtins.pyi");
    let mut set = HashSet::new();
    push_identifiers(contents, &mut set);
    let mut names: Vec<String> = set.into_iter().collect();
    names.sort();
    names.into_iter().map(Name::new).collect()
}

/// `name` with one character replaced, the way a real misspelling arrives.
fn misspell(name: &Name) -> Name {
    let chars: Vec<char> = name.as_str().chars().collect();
    let at = chars.len() / 2;
    let replacement = if chars[at] == 'x' { 'q' } else { 'x' };
    Name::new(
        chars
            .iter()
            .enumerate()
            .map(|(i, c)| if i == at { replacement } else { *c })
            .collect::<String>(),
    )
}

fn suggestion_real(c: &mut Criterion) {
    let builtins = real_builtins();
    let scope = real_scope(100_000);

    // No candidate is close, so every one has to be rejected. This is the shape
    // production hits on a file full of names from an import that failed.
    let unknown = Name::new("Zqxjkvwmpfghbd");
    assert!(
        search(&unknown, &scope, &builtins).is_none(),
        "`suggestion_real_unknown` found a match, so it no longer measures the rejection path"
    );
    c.bench_function("suggestion_real_unknown", |b| {
        b.iter(|| black_box(search(black_box(&unknown), &scope, &builtins)))
    });

    // An ordinary typo: the name it was meant to be is in scope, one edit away.
    let target = scope[scope.len() / 2].0.clone();
    let typo = misspell(&target);
    assert_eq!(
        search(&typo, &scope, &builtins).as_ref(),
        Some(&target),
        "`suggestion_real_typo` should recover the name it misspells"
    );
    c.bench_function("suggestion_real_typo", |b| {
        b.iter(|| black_box(search(black_box(&typo), &scope, &builtins)))
    });

    // The same, but misspelling one of the long generated names, which select a
    // wider distance tier and survive the length filter against more candidates.
    let longest = scope
        .iter()
        .max_by_key(|(name, _, _)| name.as_str().len())
        .expect("scope is not empty")
        .0
        .clone();
    let long_typo = misspell(&longest);
    assert_eq!(
        search(&long_typo, &scope, &builtins).as_ref(),
        Some(&longest),
        "`suggestion_real_long_typo` should recover the name it misspells"
    );
    c.bench_function("suggestion_real_long_typo", |b| {
        b.iter(|| black_box(search(black_box(&long_typo), &scope, &builtins)))
    });
}

/// A file whose unresolved references sit at the bottom of `depth` nested
/// scopes, so each lookup walks every enclosing scope rather than just the
/// module. Every fourth level is a class, whose names the code blocks nested
/// inside it cannot see, so the walk also has to do its skipping.
fn nested_scopes(depth: usize, per_scope: usize, missing: usize) -> String {
    let mut source = String::new();
    for level in 0..depth {
        let pad = "    ".repeat(level);
        if level % 4 == 3 && level + 1 < depth {
            let _ = writeln!(source, "{pad}class C{level}:");
        } else {
            let _ = writeln!(source, "{pad}def f{level}():");
        }
        let inner = "    ".repeat(level + 1);
        for i in 0..per_scope {
            let _ = writeln!(source, "{inner}n{level}_{i} = 1");
        }
    }
    let inner = "    ".repeat(depth);
    for i in 0..missing {
        let _ = writeln!(source, "{inner}_r{i} = {MISSING}");
    }
    source
}

fn suggestion_nested(c: &mut Criterion) {
    let code = Arc::new(FileContents::from_source(nested_scopes(16, 60, 20)));
    assert_eq!(
        check_module(code.dupe(), "bench", BENCH_FILE),
        20,
        "benchmark `suggestion_whole_file/nested_scopes` produced an unexpected error count"
    );
    let mut group = c.benchmark_group("suggestion_whole_file");
    group.bench_function("nested_scopes_16_deep", |b| {
        b.iter(|| check_module(code.dupe(), "bench", BENCH_FILE))
    });
    group.finish();
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
    nested_generic_constructor,
    user_method_dispatch,
    builtin_method_dispatch,
    method_name_collision,
    constructor_dispatch,
    pandas_method_dispatch,
    suggestion_search,
    suggestion_whole_file,
    suggestion_real,
    suggestion_nested,
    suggestion_discarded_errors,
);
criterion_main!(benches);
