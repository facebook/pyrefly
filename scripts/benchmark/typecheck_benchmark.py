#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type checker speed benchmark.

This script measures execution time and memory usage of type checkers
(pyright, pyrefly, ty, mypy, zuban) across popular Python packages.

It does NOT measure error counts or false positives/negatives -- only
wall-clock latency and peak RSS memory per type checker per package.

Usage:
    python3 typecheck_benchmark.py [OPTIONS]

Examples:
    # Run all checkers on all packages
    python3 typecheck_benchmark.py

    # Run only pyrefly on 5 packages
    python3 typecheck_benchmark.py -c pyrefly -p 5

    # Run specific packages with 3 runs each
    python3 typecheck_benchmark.py -n requests flask django -r 3

    # Save results to a custom directory
    python3 typecheck_benchmark.py -o ./my_results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TypedDict

ROOT_DIR: Path = Path(__file__).parent

# The 5 type checkers to benchmark
DEFAULT_TYPE_CHECKERS: list[str] = ["pyright", "pyrefly", "ty", "mypy", "zuban"]

# Timeout per type checker invocation (seconds)
DEFAULT_TIMEOUT: int = 300


# ---------------------------------------------------------------------------
# TypedDicts for structured output
# ---------------------------------------------------------------------------


class RunStats(TypedDict):
    """Statistical summary of multiple benchmark runs."""

    min: float
    max: float
    mean: float
    median: float
    stddev: float


class TimingMetrics(TypedDict, total=False):
    """Timing metrics for a single type checker run on a single package."""

    ok: bool
    execution_time_s: float
    peak_memory_mb: float
    error_message: str | None
    runs: int
    execution_time_stats: RunStats
    peak_memory_stats: RunStats


class PackageResult(TypedDict, total=False):
    """Result of benchmarking a single package."""

    package_name: str
    github_url: str | None
    error: str | None
    metrics: dict[str, TimingMetrics]


class AggregateStats(TypedDict, total=False):
    """Aggregate statistics for one type checker across all packages."""

    packages_tested: int
    packages_failed: int
    avg_execution_time_s: float
    p50_execution_time_s: float
    p90_execution_time_s: float
    p95_execution_time_s: float
    max_execution_time_s: float
    total_execution_time_s: float
    avg_peak_memory_mb: float
    p90_peak_memory_mb: float
    p95_peak_memory_mb: float
    max_peak_memory_mb: float


class BenchmarkOutput(TypedDict, total=False):
    """Top-level JSON output."""

    timestamp: str
    date: str
    type_checkers: list[str]
    type_checker_versions: dict[str, str]
    package_count: int
    runs_per_package: int
    aggregate: dict[str, AggregateStats]
    results: list[PackageResult]


class ProcessResult(TypedDict, total=False):
    """Result from a subprocess run."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    execution_time_s: float
    peak_memory_mb: float


# ---------------------------------------------------------------------------
# Subprocess execution with memory measurement
# ---------------------------------------------------------------------------


def _parse_time_stderr(stderr: str) -> tuple[float, str]:
    """Extract peak RSS from /usr/bin/time output mixed into stderr.

    Handles both macOS (/usr/bin/time -l) and Linux (/usr/bin/time -v) output
    formats.  Returns (peak_memory_mb, stderr_without_time_output).
    """
    peak_mb = 0.0
    filtered_lines: list[str] = []
    in_time_output = False
    for line in stderr.splitlines(keepends=True):
        stripped = line.strip()

        # macOS: output starts with "  0.12 real  0.01 user  0.00 sys"
        if re.match(r"\d+\.\d+\s+real\s+", stripped):
            in_time_output = True
            continue

        # Linux: output starts with "Command being timed: ..."
        if stripped.startswith("Command being timed:"):
            in_time_output = True
            continue

        if in_time_output:
            # macOS: "  12345678  maximum resident set size" (bytes)
            m = re.match(r"(\d+)\s+maximum resident set size", stripped)
            if m:
                peak_mb = round(int(m.group(1)) / (1024 * 1024), 1)
                continue

            # Linux: "Maximum resident set size (kbytes): 12345"
            m = re.match(r"Maximum resident set size \(kbytes\):\s*(\d+)", stripped)
            if m:
                peak_mb = round(int(m.group(1)) / 1024, 1)
                continue

            # Other /usr/bin/time stat lines
            if re.match(r"(\d+\s+\w|.*:\s*\d)", stripped):
                continue

            # No longer in time output
            in_time_output = False

        filtered_lines.append(line)
    return peak_mb, "".join(filtered_lines)


def run_process_with_timeout(
    cmd: list[str],
    cwd: Path,
    timeout: int,
) -> ProcessResult:
    """Run a process with timeout, returning timing and peak memory.

    Peak memory is measured via /usr/bin/time on macOS (-l) and Linux (-v).
    """
    # Wrap with /usr/bin/time to get peak RSS from stderr
    actual_cmd = cmd
    use_time = sys.platform in ("darwin", "linux")
    if use_time:
        flag = "-l" if sys.platform == "darwin" else "-v"
        actual_cmd = ["/usr/bin/time", flag] + cmd

    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform != "win32":
        kwargs["start_new_session"] = True

    start_time = time.perf_counter()
    process = subprocess.Popen(actual_cmd, **kwargs)

    timed_out = False
    try:
        stdout, raw_stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                pass
        else:
            process.kill()
        stdout, raw_stderr = process.communicate(timeout=5)

    execution_time = time.perf_counter() - start_time

    # Extract peak memory from /usr/bin/time output
    if use_time:
        peak_memory_mb, clean_stderr = _parse_time_stderr(raw_stderr)
    else:
        peak_memory_mb = 0.0
        clean_stderr = raw_stderr

    if timed_out:
        return {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "timed_out": True,
            "execution_time_s": round(execution_time, 2),
            "peak_memory_mb": peak_memory_mb,
        }

    return {
        "stdout": stdout,
        "stderr": clean_stderr,
        "returncode": process.returncode,
        "timed_out": False,
        "execution_time_s": round(execution_time, 2),
        "peak_memory_mb": peak_memory_mb,
    }


# ---------------------------------------------------------------------------
# Package loading from install_envs.json
# ---------------------------------------------------------------------------


def load_install_envs(
    install_envs_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Load packages from install_envs.json.

    Returns only packages that have install: true or a non-empty deps list.
    """
    if install_envs_file is None:
        install_envs_file = ROOT_DIR / "install_envs.json"

    if not install_envs_file.exists():
        print(f"Error: {install_envs_file} not found")
        return []

    with open(install_envs_file, encoding="utf-8") as f:
        data = json.load(f)

    packages: list[dict[str, Any]] = []
    for pkg in data.get("packages", []):
        github_url = pkg.get("github_url", "")
        if not github_url:
            continue
        # Only include packages that have install config
        has_install = pkg.get("install", False)
        has_deps = bool(pkg.get("deps"))
        if not has_install and not has_deps:
            continue
        # Derive name from explicit field or github URL
        name = pkg.get("name") or github_url.rstrip("/").split("/")[-1]
        packages.append({**pkg, "name": name})

    return packages


# ---------------------------------------------------------------------------
# Type checker version detection
# ---------------------------------------------------------------------------


def get_type_checker_versions() -> dict[str, str]:
    """Get version strings for all type checkers."""
    versions: dict[str, str] = {}
    version_commands: dict[str, list[str]] = {
        "pyright": ["pyright", "--version"],
        "pyrefly": ["pyrefly", "--version"],
        "ty": ["ty", "--version"],
        "mypy": ["mypy", "--version"],
        "zuban": ["zuban", "--version"],
    }
    for name, cmd in version_commands.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                match = re.search(r"\d+\.\d+\.\d+", output)
                if match:
                    versions[name] = match.group(0)
                else:
                    parts = output.split()
                    versions[name] = parts[1] if len(parts) > 1 else parts[0]
            else:
                versions[name] = "unknown"
        except (subprocess.TimeoutExpired, OSError):
            versions[name] = "not installed"
    return versions


# ---------------------------------------------------------------------------
# Type checker availability
# ---------------------------------------------------------------------------


def is_type_checker_available(checker: str) -> bool:
    """Check if a type checker binary is available on PATH."""
    if checker == "mypy":
        result = subprocess.run(
            [sys.executable, "-c", "import mypy"],
            capture_output=True,
        )
        return result.returncode == 0

    cmd_map: dict[str, str] = {
        "pyright": "pyright",
        "pyrefly": "pyrefly",
        "ty": "ty",
        "zuban": "zuban",
    }
    binary = cmd_map.get(checker)
    if not binary:
        return False
    which_cmd = "where" if sys.platform == "win32" else "which"
    result = subprocess.run([which_cmd, binary], capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Cloning and dependency installation
# ---------------------------------------------------------------------------


def clone_package(
    github_url: str,
    name: str,
    dest: Path,
    timeout: int = 180,
) -> Path | None:
    """Shallow-clone a GitHub repository."""
    target = dest / name
    try:
        print(f"  Cloning {github_url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", github_url, str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  Failed to clone: {result.stderr[:200]}")
            return None
        return target
    except subprocess.TimeoutExpired:
        print(f"  Timeout cloning {github_url}")
        return None
    except Exception as e:
        print(f"  Error cloning: {e}")
        return None


def install_deps(package_path: Path, env_config: dict[str, Any]) -> bool:
    """Install dependencies for a package."""
    install_self = env_config.get("install", False)
    deps = env_config.get("deps", [])
    install_env = env_config.get("install_env", {})

    env = os.environ.copy()
    env.update(install_env)

    if install_self:
        print("  Installing package (pip install -e .)")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=package_path,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            if result.returncode != 0:
                print(f"  Warning: pip install -e . failed: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print("  Warning: pip install -e . timed out")
            return False

    if deps:
        print(
            f"  Installing deps: {', '.join(deps[:5])}{'...' if len(deps) > 5 else ''}"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + deps,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"  Warning: pip install deps failed: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print("  Warning: pip install deps timed out")
            return False

    return True


# ---------------------------------------------------------------------------
# Type checker runners (timing-only -- we don't parse error output)
# ---------------------------------------------------------------------------
# Dummy config methodology:
# Many packages ship their own type checker configs (mypy.ini, pyrightconfig.json,
# pyproject.toml [tool.*] sections, etc.) that can change strictness, enable/disable
# checks, or alter behavior in ways that skew benchmark comparisons.
#
# To ensure consistent benchmarking across packages, we write minimal config files
# that embed the check_paths (from install_envs.json) directly into each checker's
# native config format, then pass the config via CLI flag to override package-level
# config.  This means paths are NOT passed as CLI arguments — they live in the config:
#
#   pyright  -- pyrightconfig.json with "include": [paths] (written in-place)
#   mypy     -- [mypy] section with files = path1, path2 via --config-file
#   ty       -- [src] include = [paths] in ty.benchmark.toml via --config-file
#   pyrefly  -- project_includes = [paths] in pyrefly.benchmark.toml via --config
#   zuban    -- [mypy] section with files = path1, path2 via --config-file
# ---------------------------------------------------------------------------


def _write_dummy_pyright_config(
    package_path: Path,
    check_paths: list[str] | None = None,
) -> None:
    """Write minimal pyrightconfig.json for consistent benchmarking."""
    config = {
        "include": check_paths or ["."],
        "exclude": [],
        "typeCheckingMode": "basic",
    }
    with open(package_path / "pyrightconfig.json", "w", encoding="utf-8") as f:
        json.dump(config, f)


def _write_dummy_mypy_config(
    package_path: Path,
    check_paths: list[str] | None = None,
) -> Path:
    """Write minimal mypy config for consistent benchmarking.

    Excludes test directories to avoid fatal errors from duplicate
    conftest.py modules and syntax errors in test fixture files, which
    cause mypy to bail out with 'errors prevented further checking'
    without actually type-checking any code.
    """
    config_path = package_path / "mypy.benchmark.ini"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[mypy]\n")
        if check_paths:
            f.write(f"files = {', '.join(check_paths)}\n")
        # Exclude test dirs to avoid duplicate module / syntax errors
        # that cause mypy to bail out without checking anything
        f.write("exclude = (?x)(\n    /tests/\n    | /test_\n    | /testing/\n  )\n")
    return config_path


def _write_dummy_ty_config(
    package_path: Path,
    check_paths: list[str] | None = None,
) -> Path:
    """Write minimal ty.toml for consistent benchmarking."""
    config_path = package_path / "ty.benchmark.toml"
    with open(config_path, "w", encoding="utf-8") as f:
        if check_paths:
            f.write("[src]\n")
            paths_toml = ", ".join(f'"{p}"' for p in check_paths)
            f.write(f"include = [{paths_toml}]\n")
    return config_path


def _write_dummy_pyrefly_config(
    package_path: Path,
    check_paths: list[str] | None = None,
) -> Path:
    """Write minimal pyrefly.toml for consistent benchmarking."""
    config_path = package_path / "pyrefly.benchmark.toml"
    with open(config_path, "w", encoding="utf-8") as f:
        if check_paths:
            paths_toml = ", ".join(f'"{p}"' for p in check_paths)
            f.write(f"project_includes = [{paths_toml}]\n")
    return config_path


def _write_dummy_zuban_config(
    package_path: Path,
    check_paths: list[str] | None = None,
) -> Path:
    """Write minimal mypy-style config for zuban benchmarking."""
    config_path = package_path / "zuban.benchmark.ini"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[mypy]\n")
        if check_paths:
            f.write(f"files = {', '.join(check_paths)}\n")
    return config_path


def run_checker(
    checker: str,
    package_path: Path,
    check_paths: list[Path] | None,
    timeout: int,
) -> TimingMetrics:
    """Run a single type checker and return timing metrics only."""
    # Convert absolute check_paths to relative strings from package root
    rel_paths: list[str] | None = None
    if check_paths:
        rel_paths = [
            str(p.relative_to(package_path)) if p.is_absolute() else str(p)
            for p in check_paths
        ]

    if checker == "pyright":
        _write_dummy_pyright_config(package_path, rel_paths)
        cmd = ["pyright", "--outputjson"]
        cwd = package_path
    elif checker == "pyrefly":
        config_path = _write_dummy_pyrefly_config(package_path, rel_paths)
        cmd = ["pyrefly", "check", "--config", str(config_path)]
        cwd = package_path
    elif checker == "ty":
        config_path = _write_dummy_ty_config(package_path, rel_paths)
        cmd = ["ty", "check", "--config-file", str(config_path)]
        cwd = package_path
    elif checker == "mypy":
        config_path = _write_dummy_mypy_config(package_path, rel_paths)
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--config-file",
            str(config_path),
        ]
        # mypy needs explicit paths if no files= in config; when check_paths
        # are embedded in the config via files=, we still pass "." so mypy
        # has a target (it requires at least one positional arg or files=).
        if not rel_paths:
            cmd.append(str(package_path))
        cwd = package_path
    elif checker == "zuban":
        # zuban ignores files= from --config-file, so we must always pass
        # check paths as explicit positional arguments.
        cmd = ["zuban", "check"]
        if rel_paths:
            cmd.extend(rel_paths)
        else:
            cmd.append(".")
        cwd = package_path
    else:
        return {
            "ok": False,
            "execution_time_s": 0.0,
            "peak_memory_mb": 0.0,
            "error_message": f"Unknown checker: {checker}",
        }

    result = run_process_with_timeout(cmd, cwd=cwd, timeout=timeout)

    if result["timed_out"]:
        return {
            "ok": False,
            "execution_time_s": result["execution_time_s"],
            "peak_memory_mb": result.get("peak_memory_mb", 0.0),
            "error_message": "Timeout",
        }

    # Detect fatal errors: mypy exits with code 2 for fatal errors (code 1 = type errors found).
    # Also check for "errors prevented further checking" which means mypy bailed out early.
    stderr = result.get("stderr", "")
    stdout = result.get("stdout", "")
    combined_output = stderr + stdout

    if "errors prevented further checking" in combined_output:
        return {
            "ok": False,
            "execution_time_s": result["execution_time_s"],
            "peak_memory_mb": result.get("peak_memory_mb", 0.0),
            "error_message": "Fatal: errors prevented further checking",
        }

    # mypy return code 2 = fatal error (not type errors)
    if checker == "mypy" and result.get("returncode", 0) == 2:
        # Extract first error line from output for context
        first_error = ""
        for line in combined_output.splitlines():
            if "error:" in line.lower():
                first_error = line.strip()[:200]
                break
        return {
            "ok": False,
            "execution_time_s": result["execution_time_s"],
            "peak_memory_mb": result.get("peak_memory_mb", 0.0),
            "error_message": first_error or "Fatal error (exit code 2)",
        }

    return {
        "ok": True,
        "execution_time_s": result["execution_time_s"],
        "peak_memory_mb": result.get("peak_memory_mb", 0.0),
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def compute_percentile(values: Sequence[float | int], percentile: float) -> float:
    """Compute the given percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_values):
        return float(sorted_values[-1])
    fraction = index - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def compute_run_stats(values: list[float]) -> RunStats:
    """Compute min/max/mean/median/stddev for a list of values."""
    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "stddev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def compute_aggregate_stats(
    results: list[PackageResult],
    type_checkers: list[str],
) -> dict[str, AggregateStats]:
    """Compute aggregate timing statistics across all packages."""
    stats: dict[str, AggregateStats] = {}

    for checker in type_checkers:
        times: list[float] = []
        memories: list[float] = []
        packages_tested = 0
        packages_failed = 0

        for result in results:
            if result.get("error"):
                continue
            metrics = result.get("metrics", {}).get(checker)
            if not metrics:
                continue
            if not metrics.get("ok"):
                packages_failed += 1
                continue
            packages_tested += 1
            times.append(metrics["execution_time_s"])
            mem = metrics.get("peak_memory_mb", 0.0)
            if mem > 0:
                memories.append(mem)

        stats[checker] = {
            "packages_tested": packages_tested,
            "packages_failed": packages_failed,
            "avg_execution_time_s": round(sum(times) / len(times), 2) if times else 0.0,
            "p50_execution_time_s": round(compute_percentile(times, 50), 2),
            "p90_execution_time_s": round(compute_percentile(times, 90), 2),
            "p95_execution_time_s": round(compute_percentile(times, 95), 2),
            "max_execution_time_s": round(max(times), 2) if times else 0.0,
            "total_execution_time_s": round(sum(times), 2),
            "avg_peak_memory_mb": round(sum(memories) / len(memories), 1)
            if memories
            else 0.0,
            "p90_peak_memory_mb": round(compute_percentile(memories, 90), 1),
            "p95_peak_memory_mb": round(compute_percentile(memories, 95), 1),
            "max_peak_memory_mb": round(max(memories), 1) if memories else 0.0,
        }

    return stats


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    package_limit: int | None = None,
    package_names: list[str] | None = None,
    type_checkers: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    output_dir: Path | None = None,
    os_name: str | None = None,
    install_envs_file: Path | None = None,
    runs: int = 5,
    warmup: int = 1,
) -> Path:
    """Run the full benchmark suite.

    Args:
        package_limit: Max packages to benchmark.
        package_names: Specific package names to benchmark.
        type_checkers: Type checkers to run.
        timeout: Per-checker timeout in seconds.
        output_dir: Where to write JSON results.
        os_name: OS name for filename (ubuntu, macos, windows).
        install_envs_file: Path to install_envs.json.
        runs: Number of runs per checker per package.
        warmup: Number of warmup runs to discard before real runs.

    Returns:
        Path to the dated output JSON file.
    """
    if type_checkers is None:
        type_checkers = DEFAULT_TYPE_CHECKERS.copy()
    if output_dir is None:
        output_dir = ROOT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load packages
    packages = load_install_envs(install_envs_file)
    if not packages:
        print("Error: No packages found in install_envs.json")
        return output_dir / "empty.json"

    if package_names:
        name_set = set(package_names)
        packages = [p for p in packages if p["name"] in name_set]
        if not packages:
            print(f"Warning: None of the specified packages found: {package_names}")
            return output_dir / "empty.json"
    elif package_limit:
        packages = packages[:package_limit]

    # Header
    print("=" * 70)
    print("Type Checker Timing Benchmark")
    print("=" * 70)
    print(f"Packages: {len(packages)}")
    print(f"Type checkers: {', '.join(type_checkers)}")
    print(f"Timeout: {timeout}s per checker")
    print(f"Warmup runs: {warmup}")
    print(f"Measured runs: {runs}")
    print("=" * 70)

    versions = get_type_checker_versions()
    print("\nType Checker Versions:")
    for name, version in versions.items():
        if name in type_checkers:
            print(f"  {name}: {version}")
    print()

    all_results = _run_all(packages, type_checkers, timeout, runs, warmup)

    # Aggregate
    aggregate = compute_aggregate_stats(all_results, type_checkers)

    # Save
    output_file = _save_results(
        all_results,
        aggregate,
        type_checkers,
        versions,
        len(packages),
        output_dir,
        os_name,
        runs,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    _print_summary(aggregate, type_checkers)
    print(f"\nResults saved to: {output_file}")

    return output_file


def _run_all(
    packages: list[dict[str, Any]],
    type_checkers: list[str],
    timeout: int,
    runs: int = 5,
    warmup: int = 1,
) -> list[PackageResult]:
    """Run benchmarks for all packages."""
    all_results: list[PackageResult] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, pkg in enumerate(packages, 1):
            name = pkg["name"]

            print(f"\n[{i}/{len(packages)}] {name}")

            result = _benchmark_package(
                pkg, temp_path, type_checkers, timeout, runs, warmup
            )
            all_results.append(result)

    return all_results


def _benchmark_package(
    pkg: dict[str, Any],
    temp_path: Path,
    type_checkers: list[str],
    timeout: int,
    runs: int = 5,
    warmup: int = 1,
) -> PackageResult:
    """Benchmark a single package: clone, install deps, run checkers."""
    name = pkg["name"]
    github_url = pkg.get("github_url", "")

    if not github_url:
        return {
            "package_name": name,
            "github_url": None,
            "error": "No GitHub URL",
            "metrics": {},
        }

    # Clone
    package_path = clone_package(github_url, name, temp_path)
    if not package_path:
        return {
            "package_name": name,
            "github_url": github_url,
            "error": "Failed to clone",
            "metrics": {},
        }

    # Install deps
    success = install_deps(package_path, pkg)
    if not success:
        print("  Warning: dependency installation had issues, continuing anyway")

    # Resolve check_paths
    resolved_paths: list[Path] | None = None
    raw_check_paths = pkg.get("check_paths")
    if raw_check_paths:
        resolved_paths = [package_path / p for p in raw_check_paths]
        existing = [p for p in resolved_paths if p.exists()]
        if existing:
            resolved_paths = existing
            rel = [str(p.relative_to(package_path)) for p in resolved_paths]
            print(f"  Checking: {rel}")
        else:
            print("  Warning: check_paths don't exist, using full repo")
            resolved_paths = None

    # Run type checkers
    metrics: dict[str, TimingMetrics] = {}
    for checker in type_checkers:
        if not is_type_checker_available(checker):
            print(f"    Skipping {checker}: not installed")
            metrics[checker] = {
                "ok": False,
                "execution_time_s": 0.0,
                "peak_memory_mb": 0.0,
                "error_message": "Not installed",
            }
            continue

        total = warmup + runs
        print(
            f"    Running {checker}... "
            f"({warmup} warmup + {runs} measured run{'s' if runs > 1 else ''})"
        )
        times: list[float] = []
        memories: list[float] = []
        failed_metric: TimingMetrics | None = None

        for run_idx in range(total):
            is_warmup = run_idx < warmup
            label = (
                f"Warmup {run_idx + 1}/{warmup}"
                if is_warmup
                else f"Run {run_idx - warmup + 1}/{runs}"
            )
            print(f"      {label}...", end=" ")
            m = run_checker(checker, package_path, resolved_paths, timeout)
            if not m.get("ok"):
                print(f"Failed: {m.get('error_message', 'Unknown')}")
                if not is_warmup:
                    failed_metric = m
                    break
                # Warmup failure: skip this checker entirely
                failed_metric = m
                break
            peak = m.get("peak_memory_mb", 0)
            mem_str = f", {peak:.0f}MB" if peak > 0 else ""
            if is_warmup:
                print(f"{m['execution_time_s']:.1f}s{mem_str} (discarded)")
            else:
                print(f"{m['execution_time_s']:.1f}s{mem_str}")
                times.append(m["execution_time_s"])
                memories.append(m.get("peak_memory_mb", 0.0))

        if failed_metric is not None:
            failed_metric["runs"] = len(times)
            metrics[checker] = failed_metric
        elif not times:
            metrics[checker] = {
                "ok": False,
                "execution_time_s": 0.0,
                "peak_memory_mb": 0.0,
                "error_message": "No successful runs",
            }
        else:
            result_metric: TimingMetrics = {
                "ok": True,
                "execution_time_s": round(statistics.mean(times), 2),
                "peak_memory_mb": round(statistics.mean(memories), 2),
                "runs": runs,
            }
            if len(times) > 1:
                result_metric["execution_time_stats"] = compute_run_stats(times)
                result_metric["peak_memory_stats"] = compute_run_stats(memories)

            metrics[checker] = result_metric
            peak = result_metric["peak_memory_mb"]
            mem_str = f", {peak:.0f}MB" if peak > 0 else ""
            if len(times) > 1:
                time_stats = result_metric["execution_time_stats"]
                print(
                    f"      Mean: {result_metric['execution_time_s']:.1f}s{mem_str} "
                    f"(stddev: {time_stats['stddev']:.2f}s)"
                )
            else:
                print(f"      {result_metric['execution_time_s']:.1f}s{mem_str}")

    # Cleanup cloned repo
    shutil.rmtree(package_path, ignore_errors=True)

    return {
        "package_name": name,
        "github_url": github_url,
        "error": None,
        "metrics": metrics,
    }


def _save_results(
    results: list[PackageResult],
    aggregate: dict[str, AggregateStats],
    type_checkers: list[str],
    versions: dict[str, str],
    package_count: int,
    output_dir: Path,
    os_name: str | None = None,
    runs: int = 1,
) -> Path:
    """Save benchmark results to JSON."""
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y-%m-%d")

    if os_name:
        output_file = output_dir / f"benchmark_{date_str}_{os_name}.json"
        latest_file = output_dir / f"latest-{os_name}.json"
    else:
        output_file = output_dir / f"benchmark_{date_str}.json"
        latest_file = output_dir / "latest.json"

    output_data: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "date": date_str,
        "type_checkers": type_checkers,
        "type_checker_versions": {
            k: v for k, v in versions.items() if k in type_checkers
        },
        "package_count": package_count,
        "runs_per_package": runs,
        "aggregate": aggregate,
        "results": results,
    }
    if os_name:
        output_data["os"] = os_name

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"  {output_file}")
    print(f"  {latest_file}")
    return output_file


def _print_summary(
    stats: dict[str, AggregateStats],
    type_checkers: list[str],
) -> None:
    """Print a summary table."""
    print(
        f"\n{'Checker':<12} {'Pkgs':>5} {'Avg Time':>9} {'P95 Time':>9} {'Max Time':>9} {'Avg Mem':>8} {'Max Mem':>8}"
    )
    print("-" * 70)
    for checker in type_checkers:
        s = stats.get(checker, {})
        if s.get("packages_tested", 0) == 0:
            print(f"{checker:<12} {'N/A':>5}")
            continue
        avg_mem = s.get("avg_peak_memory_mb", 0)
        max_mem = s.get("max_peak_memory_mb", 0)
        print(
            f"{checker:<12} "
            f"{s.get('packages_tested', 0):>5} "
            f"{s.get('avg_execution_time_s', 0):>8.1f}s "
            f"{s.get('p95_execution_time_s', 0):>8.1f}s "
            f"{s.get('max_execution_time_s', 0):>8.1f}s "
            f"{avg_mem:>7.0f}M "
            f"{max_mem:>7.0f}M"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run type checker timing benchmarks across packages",
    )
    parser.add_argument(
        "--packages",
        "-p",
        type=int,
        default=None,
        help="Max number of packages to benchmark (default: all with install config)",
    )
    parser.add_argument(
        "--package-names",
        "-n",
        nargs="+",
        default=None,
        help="Specific package names to benchmark",
    )
    parser.add_argument(
        "--checkers",
        "-c",
        nargs="+",
        default=DEFAULT_TYPE_CHECKERS,
        help=f"Type checkers to benchmark (default: {' '.join(DEFAULT_TYPE_CHECKERS)})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per type checker in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--os-name",
        type=str,
        default=None,
        help="OS name for output filename (e.g., ubuntu, macos, windows)",
    )
    parser.add_argument(
        "--install-envs",
        type=Path,
        default=None,
        help="Path to install_envs.json (default: <script_dir>/install_envs.json)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=5,
        help="Number of runs per checker per package (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=1,
        help="Number of warmup runs to discard before real runs (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    run_benchmark(
        package_limit=args.packages,
        package_names=args.package_names,
        type_checkers=args.checkers,
        timeout=args.timeout,
        output_dir=args.output,
        os_name=args.os_name,
        install_envs_file=args.install_envs,
        runs=args.runs,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
