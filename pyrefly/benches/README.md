# PyTorch benchmarks

Two real-world walltime benchmarks that drive the actual Pyrefly LSP server over
a large, pinned PyTorch checkout (15k+ Python files) across all cores. They
measure interactive latency, not batch `check` throughput.

For the full command reference (all flags, micro benchmarks, cargo/buck forms),
see `.claude/skills/benchmark-pyrefly/SKILL.md`.

## The pin

The PyTorch source is pinned by a single 40-hex commit in `pytorch_pin.bzl` (rev
+ tarball sha256) — there is **no git submodule**. That one file is loaded by
`BUCK` and parsed by the shared `common` module, so the commit lives in exactly
one place. Two providers feed the checkout:

- **Internal (buck):** `pyrefly/BUCK` fetches the pinned tarball from Manifold
  via `http_archive` and passes its path to the benches in
  `PYREFLY_PYTORCH_BENCH_PATH`. No github egress; Buck CAS caches it.
- **OSS (cargo):** shallow-clones the pinned rev from github into a per-rev temp
  cache on first run (needs `git` + `github.com` egress), reused afterward.

Set `PYREFLY_PYTORCH_BENCH_PATH` to an existing checkout to bypass both. If the
checkout can't be obtained the bench prints a skip notice and exits cleanly.

## The two benches

They ship in **one target** — buck `pytorch_bench`, cargo bench `pytorch` — and
you select an individual one at runtime with a Criterion name filter rather than
picking a separate target. They live under `benches/pytorch/`, one module file
per bench:

- `pytorch/main.rs` — crate root; declares the modules and calls
  `criterion_main!` aggregating both benchmarks' Criterion groups.
- `pytorch/common.rs` — shared checkout-acquisition and LSP-args harness.
- `pytorch/cold_start.rs` — the cold-start benchmark. Fresh server per iteration;
  opens `torch/distributed/pipelining/_backward.py` and queries go-to-definition
  of the `Parameter` import. Proxy for time-to-first-index. Criterion id
  `pytorch/cold_start_go_to_definition`.
- `pytorch/error_propagation.rs` — the error-propagation benchmark. Warm server;
  edits `torch/nn/__init__.py` to rebind `Parameter` to an int and waits for the
  resulting type error to surface in the distant dependent `_backward.py`.
  Proxy for incremental edit-propagation latency. Criterion id
  `pytorch/error_propagation`.

## Running

**Build mode matters:** always build optimized or the numbers are meaningless.
Buck uses `@fbcode//mode/opt` (or `@fbcode//mode/opt-clang-thinlto` for final
numbers). Cargo `cargo bench` already uses the optimized `bench` profile. Debug
builds run 3-10x slower and are not comparable. Never run two benches in
parallel.

These are heavy walltime benchmarks: budget roughly 2-4 minutes each (Criterion's
sample floor is 10; ~3-5 s per cold-start iteration, ~2-3 s per
error-propagation iteration after warmup). They are manual/heavy and are not run
in CI Sandcastle by default (the `http_archive` dep is labeled `manual`).

```bash
# Both benchmarks
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench
cargo bench --bench pytorch

# Just one, selected by Criterion name filter
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench cold_start
buck2 run @fbcode//mode/opt fbcode//pyrefly/pyrefly:pytorch_bench -- --bench error_propagation
cargo bench --bench pytorch -- cold_start
cargo bench --bench pytorch -- error_propagation
```

Useful flags — append after `--` for buck, pass directly for cargo:

- `--list` — list the binary's benches instead of running them.
- `--quick` — quicker, lower-confidence run.
- `--noplot` — skip Criterion report rendering. Required on a headless devserver
  with no gnuplot and no usable font, where the plotters backend otherwise
  panics after a bench finishes with `BackendError(FontError(FontUnavailable))`.
  Timings are unaffected.

## Output

Criterion writes to `target/criterion/` (HTML plots, CSV samples,
`estimates.json`). Because buck run / cargo bench run from the cell root, it
lands at the fbsource repo root `target/criterion/`. It is throwaway — do not
commit it, and do not add repo-root ignore entries (it is already gitignored
appropriately).

## Updating the pin

Run on a machine with github access (devvm/Sandcastle have no egress):

```bash
./update_revision.sh <40-hex-sha>
```

It clones the rev, archives a tarball, rewrites `pytorch_pin.bzl` (rev +
sha256), and uploads the tarball to Manifold. **After a bump, re-check
`PARAM_LINE` / `PARAM_COL` in `pytorch/cold_start.rs`** — they encode the
position of `Parameter` in `_backward.py`, and the cold-start bench asserts if
they drift.
