# Pyrefly Benchmarks

Benchmarking scripts for comparing pyrefly against other Python type checkers.

Ported from https://github.com/lolpack/type_coverage_py

## Type Checking Speed Benchmark

Measures wall-clock execution time and peak memory usage of type checkers
(pyright, pyrefly, ty, mypy, zuban) across popular open-source Python packages.

### Prerequisites

Make a venv and install the type checkers you want to benchmark:

```bash
pip install pyright mypy pyrefly ty zuban
```

### Usage

```bash
# Run all checkers on all packages in install_envs.json
python3 typecheck_benchmark.py

# Run only pyrefly and mypy on the first 5 packages
python3 typecheck_benchmark.py -c pyrefly mypy -p 5

# Run specific packages with 10 measured runs and 2 warmup runs
python3 typecheck_benchmark.py -n requests flask django -r 10 -w 2

# Save results to a custom directory
python3 typecheck_benchmark.py -o ./my_results --os-name macos

# Use a custom install_envs.json
python3 typecheck_benchmark.py --install-envs /path/to/install_envs.json
```

### Options

| Flag | Description |
|------|-------------|
| `-p, --packages N` | Max number of packages to benchmark |
| `-n, --package-names NAME [NAME ...]` | Specific package names to benchmark |
| `-c, --checkers NAME [NAME ...]` | Type checkers to run (default: all five) |
| `-t, --timeout SECS` | Timeout per checker invocation (default: 300s) |
| `-r, --runs N` | Measured runs per checker per package (default: 5) |
| `-w, --warmup N` | Warmup runs to discard before measuring (default: 1) |
| `-o, --output DIR` | Output directory for JSON results |
| `--os-name NAME` | OS name for output filename (e.g., macos) |
| `--install-envs PATH` | Path to install_envs.json |

### Output

Results are saved as JSON in `results/` (or the directory specified by `-o`).
Each run produces a dated file (e.g., `benchmark_2025-03-17.json`) and a
`latest.json` symlink.

## LSP Benchmark (Go to Definition)

Measures `textDocument/definition` latency across LSP servers. For each run,
the script picks a random Python file and identifier, starts each server,
and times the Go to Definition response. All servers are run in parallel on
the same position for fair comparison.

`lsp_benchmark.py` has two modes:

1. **Multi-package mode** (`--install-envs`) — auto-clones packages from
   `install_envs.json` and benchmarks each one (recommended).
2. **Single-repo mode** (`--root`) — benchmarks an already-cloned repo.

### Prerequisites

The LSP servers must be available as commands. Typical commands:

- **pyrefly**: `pyrefly lsp`
- **ty**: `ty server`
- **pyright**: `pyright-langserver --stdio`
- **zuban**: `zubanls`

### Usage (multi-package mode — recommended)

```bash
# Run all checkers on all packages (auto-clones from GitHub)
python3 lsp_benchmark.py --install-envs install_envs.json

# Run only pyrefly on 3 packages with 10 runs each
python3 lsp_benchmark.py --install-envs install_envs.json -c pyrefly -p 3 -r 10

# Run specific packages
python3 lsp_benchmark.py --install-envs install_envs.json -n requests flask django -r 20

# Save results to a custom directory
python3 lsp_benchmark.py --install-envs install_envs.json -o ./my_results --os-name macos
```

### Usage (single-repo mode)

```bash
# Benchmark only pyrefly with JSON output
python3 lsp_benchmark.py \
    --root ~/projects/some-python-repo \
    --servers pyrefly \
    --pyrefly-cmd "pyrefly lsp" \
    --runs 20 \
    --json results.json

# Reproducible run with a fixed seed
python3 lsp_benchmark.py \
    --root ~/projects/some-python-repo \
    --pyrefly-cmd "pyrefly lsp" \
    --seed 42 --runs 50

# All four servers
python3 lsp_benchmark.py \
    --root ~/projects/some-python-repo \
    --pyrefly-cmd "pyrefly lsp" \
    --ty-cmd "ty server" \
    --pyright-cmd "pyright-langserver --stdio" \
    --zuban-cmd "zubanls" \
    --runs 20
```

### Options

| Flag | Description | Mode |
|------|-------------|------|
| `--install-envs PATH` | Path to install_envs.json (enables multi-package mode) | multi |
| `-p, --packages N` | Max number of packages to benchmark | multi |
| `-n, --package-names NAME [NAME ...]` | Specific package names to benchmark | multi |
| `-c, --checkers NAME [NAME ...]` | Type checkers to benchmark (default: all four) | multi |
| `-o, --output DIR` | Output directory for JSON results | multi |
| `--os-name NAME` | OS name for output filename | multi |
| `--root PATH` | Repo root / workspace folder (default: cwd) | single |
| `--servers LIST` | Comma-separated server names (default: auto-detect from provided commands) | single |
| `--pyrefly-cmd CMD` | Command to start pyrefly LSP | single |
| `--ty-cmd CMD` | Command to start ty LSP | single |
| `--pyright-cmd CMD` | Command to start pyright LSP | single |
| `--zuban-cmd CMD` | Command to start zuban LSP | single |
| `--trace` | Verbose LSP wire trace to stderr | single |
| `--json PATH` | Write JSON report to file | single |
| `--settings-json JSON` | JSON settings to send via `workspace/didChangeConfiguration` | single |
| `--pyright-disable-indexing` | Disable Pyright background indexing | single |
| `-r, --runs N` | Runs per package (default: 100 multi, 1 single) | both |
| `-s, --seed N` | RNG seed for reproducibility | both |
| `--timeout SECS` | Per-request timeout (default: 10s) | both |

### Output

Prints a summary table to stdout. With `--json` (single-repo) or `-o`
(multi-package), writes a detailed JSON report including per-run results,
latency percentiles (p50, p95), and per-server ok/found/valid rates.

## Package List (`install_envs.json`)

The `install_envs.json` file defines which packages to benchmark in the
type checking benchmark, including:

- GitHub URLs for cloning
- Subdirectories to type-check (`check_paths`)
- Whether to `pip install -e .` the package (`install`)
- Additional pip dependencies (`deps`)
