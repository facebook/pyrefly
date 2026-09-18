---
name: benchmark-pyrefly
description: Run Pyrefly benchmarks locally via Buck or Cargo, including PyTorch real-world LSP benchmarks. Use when user asks to benchmark pyrefly performance, run pyrefly bench, compare cold start vs error propagation, or update PyTorch benchmark pin.
---

# Benchmarking Pyrefly

Use this skill when asked to benchmark pyrefly performance, run a pyrefly bench,
compare cold-start vs error-propagation latency, or update the PyTorch benchmark
pin. All benches live in `pyrefly/pyrefly/benches/` and use Criterion.

Build mode matters: **always build optimized, or the numbers are meaningless.**
Buck uses `@fbcode//mode/opt` (or `@fbcode//mode/opt-clang-thinlto` for final
numbers). Cargo `cargo bench` already uses the optimized `bench` profile;
`cargo run`/`cargo build` would need `--release`. Debug builds run 3-10x slower
and are not comparable.

Never run two benchmarks in parallel — they compete for CPU and memory and the
timings become unreliable.

## Micro benchmarks (fast)

Deterministic, single-threaded microbenchmarks of type-checker internals. Cheap
to run, good for a quick signal.

- Buck: `buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:micro_bench -- --bench`
- Cargo: `cargo bench --bench micro`

Source: `pyrefly/pyrefly/benches/micro.rs`.

## PyTorch benchmarks (heavy, walltime)

Three real-world benchmarks run over a pinned, multi-gigabyte PyTorch checkout
(15k+ Python files) across all cores. They are **walltime** benchmarks (threads,
I/O, a long cold start), not deterministic unit tests — distinct from the `micro`
benchmarks. Two drive the actual LSP server (`cold_start`, `error_propagation`);
one runs a cold batch check (`full_check`).

All three ship in **one target** — buck `pytorch_bench`, cargo bench
`pytorch` — and you select an individual one at runtime with a Criterion name
filter rather than picking a separate target. They live under
`pyrefly/pyrefly/benches/pytorch/`, one module file per bench:

- `pytorch/main.rs` — crate root; declares the modules and calls
  `criterion_main!` aggregating the benchmarks' Criterion groups.
- `pytorch/common.rs` — shared PyTorch-checkout acquisition harness and standard
  LSP args.
- `pytorch/cold_start.rs` — the cold-start benchmark. Fresh server per iteration;
  opens `torch/distributed/pipelining/_backward.py` and queries go-to-definition
  of the `Parameter` import. Proxy for time-to-first-index. Criterion id
  `pytorch/cold_start_go_to_definition`.
- `pytorch/error_propagation.rs` — the error-propagation benchmark. Warm server;
  edits `torch/nn/__init__.py` to rebind `Parameter` to an int and waits for the
  resulting type error to surface in the distant dependent `_backward.py`. Proxy
  for incremental edit-propagation latency. Criterion id
  `pytorch/error_propagation`.
- `pytorch/full_check.rs` — the full-check benchmark. Fresh `State` per iteration;
  runs exactly what `pyrefly check` (project mode, no file args) does from inside
  the checkout — discovers the project and checks every project file across all
  cores. Proxy for whole-project batch throughput (not interactive latency).
  Criterion id `pytorch/full_check`.

### Time cost

These are slow. A cold-start iteration is ~3-5 s on a 64-thread devvm;
error-propagation is ~2-3 s and full-check ~1-1.5 s per iteration. Criterion's
sample floor is 10, so budget roughly **2-4 minutes per benchmark**. Do not run
them in CI Sandcastle by default — they are manual/heavy (the PyTorch
`http_archive` dep is labeled `manual`).

### Run commands

Run all benchmarks:

```bash
# Buck (internal)
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench
# Cargo (OSS)
cargo bench --bench pytorch
```

Run just one, selecting it by Criterion name filter:

```bash
# Buck (internal) — cold start
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench cold_start
# Buck (internal) — error propagation
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench error_propagation
# Buck (internal) — full check
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench full_check
# Cargo (OSS)
cargo bench --bench pytorch -- cold_start
cargo bench --bench pytorch -- error_propagation
cargo bench --bench pytorch -- full_check
```

Useful flags (append after `--` for buck; pass directly for cargo):

- `--list` — list the benches in the binary instead of running them, e.g.
  `buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --list`.
- `--quick` — quicker, lower-confidence run.
- `--noplot` — skip Criterion report rendering. Required on a headless devserver
  with no gnuplot and no usable font: the plotters backend otherwise panics
  after a bench finishes with `BackendError(FontError(FontUnavailable))`.
  Timings are unaffected. Example:
  `buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench --noplot`.

### PyTorch source acquisition

The PyTorch source (~550 MB unpacked, 68 MB tar.gz) is pinned by a single 40-hex
commit in `pyrefly/pyrefly/benches/pytorch_pin.bzl` — there is no git submodule. That
file is loaded by `BUCK` and parsed by the shared `common` module. Two providers:

- **Internal (buck):** fetches the pinned tarball from Manifold via
  `http_archive` and passes its path to the benches in
  `PYREFLY_PYTORCH_BENCH_PATH`. No github egress; Buck CAS caches it.
- **OSS (cargo):** shallow-clones the pinned rev from github into a per-rev temp
  cache on first run (needs `git` + `github.com` egress), reused after.

Set `PYREFLY_PYTORCH_BENCH_PATH` to an existing checkout to bypass both. If the
checkout can't be obtained, the bench prints a skip notice and exits cleanly.

### Output location

Criterion writes to `target/criterion/` (HTML plots, CSV samples,
`estimates.json`). Because buck run / cargo bench run from the cell root, it
lands at the fbsource repo root `target/criterion/`. It is ephemeral — do not
commit it, and do not add repo-root ignore entries (it is already gitignored
appropriately).

## TSP benchmark (fast)

`benches/tsp.rs` -- buck `tsp_bench`, cargo bench `tsp`, Criterion id
`tsp/get_computed_type_unopened_cached`. Durable regression coverage for the
TSP unopened-file solve-reuse fix (D118537886): repeats an identical
`getComputedType` request, on a stable snapshot, against one never-opened,
unreferenced module deliberately expensive to solve, over the plain
in-process main connection. Server spawn, fixture generation, and the first
cold request are unmeasured; each measured iteration is one logical request.
Portable, no pinned checkout, no `manual` label.

```bash
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:tsp_bench -- --bench
cargo bench --bench tsp
```

## Updating the PyTorch pin

Run on a machine with github access (devvm/Sandcastle have no egress):

```bash
pyrefly/pyrefly/benches/update_revision.sh <40-hex-sha>
```

It clones the rev, archives a tarball, rewrites `pytorch_pin.bzl` (rev +
sha256), and uploads the tarball to Manifold.

**After a pin bump, re-check `PARAM_LINE` / `PARAM_COL` in
`pytorch/cold_start.rs`** — they encode the position of `Parameter` in
`_backward.py`, and the cold-start bench asserts if they drift.
