#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare pyrefly, pyright, and mypy on mypy_primer projects.

Two output modes:
  Error counts (default):  prints a summary table with error counts per project.
  Full errors (--output-json):  writes structured JSON with every error message.

Local-only script. Two-phase usage:

  Prerequisites:
    - A conda/venv environment with pyright and mypy installed:
      pip install pyright mypy

  Phase 1 — Clone projects (run outside conda, proxy works):
    python3 compare_typecheckers.py --clone-only --cache-dir /tmp/primer_cache

  Phase 2 — Run checkers on cached repos (run inside conda):
    # Error counts (table):
    python3 compare_typecheckers.py --cache-dir /tmp/primer_cache --reuse-cache -o /tmp/results.txt
    # Full error messages (JSON):
    python3 compare_typecheckers.py --cache-dir /tmp/primer_cache --reuse-cache --output-json /tmp/errors.json

  Or if cloning works in your env, do it all at once:
    python3 compare_typecheckers.py --pyrefly /path/to/pyrefly --cache-dir /tmp/primer_cache -o /tmp/results.txt

  Cleanup:
    python3 compare_typecheckers.py --cleanup --cache-dir /tmp/primer_cache
"""

import argparse
import configparser
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import venv
from datetime import datetime, timezone

from projects import get_mypy_primer_projects, Project


def run(
    cmd: str | list[str], debug: bool, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    """Run a command, suppressing output unless debug is set.

    Supports a 'timeout' kwarg (seconds). If the process exceeds the
    timeout, it is killed and an empty CompletedProcess is returned.
    """
    logging.debug(cmd)
    if not debug and "capture_output" not in kwargs:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    try:
        return subprocess.run(cmd, **kwargs)  # noqa
    except subprocess.TimeoutExpired:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        logging.warning(f"  Command timed out: {cmd_str[:80]}")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="CHECKER_TIMEOUT")


def build_pyrefly() -> str:
    """Build pyrefly via cargo in release mode and return the absolute binary path."""
    logging.info("Building pyrefly via cargo (release mode)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "pyrefly"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Failed to build pyrefly: {result.stderr}")
    # Find the binary in the target directory
    # Walk up from this script to find the repo root (where Cargo.toml lives)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    path = os.path.join(repo_root, "target", "release", "pyrefly")
    if not os.path.exists(path):
        raise RuntimeError(f"Built binary not found at {path}")
    logging.info(f"Built pyrefly at {path}")
    return path


def clone_projects(projects: list[Project], cache_dir: str, debug: bool) -> None:
    """Clone all projects into cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    for project in projects:
        repo_dir = os.path.join(cache_dir, project.name)
        if os.path.exists(repo_dir):
            logging.info(f"Skipping {project.name} (already exists)")
            continue
        logging.info(f"Cloning {project.name}...")
        run(
            ["git", "clone", "--depth=1", project.location, repo_dir],
            debug,
        )
        if project.revision:
            run(["git", "checkout", project.revision], debug, cwd=repo_dir)


def setup_project(
    project: Project, repo_dir: str, debug: bool, reuse: bool
) -> str | None:
    """Clone project, create venv, install deps + runtime deps.

    Returns site-package-path flags for pyrefly, or None if no deps.
    """
    if not reuse:
        run(
            ["git", "clone", "--depth=1", project.location, repo_dir],
            debug,
        )
        if project.revision:
            run(["git", "checkout", project.revision], debug, cwd=repo_dir)

    deps = [dep.format(repo=repo_dir) for dep in project.deps or []]
    venv_dir = os.path.join(repo_dir, ".venv")
    activate = os.path.join(venv_dir, "bin", "activate")

    if not reuse or not os.path.exists(venv_dir):
        venv.create(venv_dir, with_pip=True, clear=True)
        if deps:
            result = run(
                f". {activate} && pip install " + " ".join(deps),
                debug,
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                logging.warning(
                    f"Failed to install deps for {project.name}: {result.stderr}"
                )

        # Install runtime deps via pip install ., then uninstall just the project package
        _install_and_uninstall_project(activate, repo_dir, project.name, debug)

    # Get site-package paths for pyrefly
    site_paths = run(
        f". {activate} && python -c \"import site; print(' '.join('--site-package-path=' + p for p in site.getsitepackages()))\"",
        debug,
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return site_paths or None


def _install_and_uninstall_project(
    activate: str, repo_dir: str, project_name: str, debug: bool
) -> None:
    """Install project via pip to pull in runtime deps, then uninstall just the project package.

    Leaves transitive dependencies in the venv so type checkers can resolve
    imports, without the project's own source shadowing the repo checkout.
    """
    logging.debug(
        f"Installing runtime deps for {project_name} (pip install . then uninstall)"
    )
    result = run(
        f". {activate} && pip install .",
        debug,
        cwd=repo_dir,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        logging.debug(
            f"pip install . failed for {project_name} (non-fatal): {result.stderr}"
        )
        return

    pkg_name = _get_package_name(repo_dir)
    if pkg_name:
        logging.debug(f"Uninstalling {pkg_name} (keeping transitive deps)")
        run(
            f". {activate} && pip uninstall -y {pkg_name}",
            debug,
            cwd=repo_dir,
            shell=True,
        )

    # Clean build artifacts
    for d in ["build", "dist"]:
        p = os.path.join(repo_dir, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for entry in os.listdir(repo_dir):
        if entry.endswith(".egg-info"):
            shutil.rmtree(os.path.join(repo_dir, entry))


def _get_package_name(repo_dir: str) -> str | None:
    """Extract the installed package name from pyproject.toml or setup.cfg.

    Needed because the package name often differs from the repo name
    (e.g. repo 'python-chess' installs package 'chess'), and pip uninstall
    requires the exact package name.
    """
    pyproject = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproject):
        with open(pyproject) as f:
            for line in f:
                m = re.match(r'\s*name\s*=\s*["\'](.+?)["\']', line)
                if m:
                    return m.group(1)
    setup_cfg = os.path.join(repo_dir, "setup.cfg")
    if os.path.exists(setup_cfg):
        cfg = configparser.ConfigParser()
        cfg.read(setup_cfg)
        if cfg.has_option("metadata", "name"):
            return cfg.get("metadata", "name")
    return None


def parse_error_count(output: str) -> int:
    """Parse error count from pyrefly or pyright output.

    Pyrefly:  'INFO 4 errors (11 suppressed)' or 'INFO 3,418 errors' or 'INFO No errors'
    Pyright:  '2 errors, 0 warnings, 0 informations'
    """
    for line in reversed(output.splitlines()):
        if "No errors" in line:
            return 0
        m = re.search(r"([\d,]+)\s+errors?", line)
        if m:
            return int(m.group(1).replace(",", ""))
    return -1


def parse_full_errors_pyrefly(stdout: str) -> list[dict[str, object]]:
    """Parse pyrefly --output-format json output into structured error list.

    Pyrefly JSON format:
    {
      "errors": [
        {
          "line": 1, "column": 3, "stop_line": 1, "stop_column": 5,
          "path": "file.py", "name": "bad-return",
          "description": "...", "severity": "error"
        }
      ]
    }
    """
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logging.warning("Failed to parse pyrefly JSON output")
        return []
    errors = []
    for e in data.get("errors", []):
        errors.append(
            {
                "file": e.get("path", ""),
                "line": e.get("line", 0),
                "col": e.get("column", 0),
                "kind": e.get("name", ""),
                "message": e.get("description", ""),
                "severity": e.get("severity", "error"),
            }
        )
    return errors


def parse_full_errors_pyright(stdout: str, repo_dir: str) -> list[dict[str, object]]:
    """Parse pyright --outputjson output into structured error list.

    Pyright JSON format:
    {
      "generalDiagnostics": [
        {
          "file": "/absolute/path/to/file.py",
          "severity": "error",
          "message": "...",
          "range": {"start": {"line": 0, "character": 0}},
          "rule": "reportReturnType"
        }
      ]
    }

    Pyright uses 0-based line/col numbers; we normalize to 1-based.
    Absolute file paths are made relative to repo_dir.
    """
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logging.warning("Failed to parse pyright JSON output")
        return []
    errors = []
    # Ensure repo_dir ends with separator for clean prefix stripping.
    # Use realpath to resolve symlinks (e.g. /tmp -> /private/tmp on macOS)
    # so that the prefix matches pyright's resolved absolute paths.
    prefix = os.path.realpath(repo_dir) + os.sep
    for d in data.get("generalDiagnostics", []):
        range_info = d.get("range", {})
        start = range_info.get("start", {})
        file_path = d.get("file", "")
        # Make absolute paths relative to repo_dir
        if file_path.startswith(prefix):
            file_path = file_path[len(prefix) :]
        errors.append(
            {
                "file": file_path,
                "line": start.get("line", 0) + 1,
                "col": start.get("character", 0) + 1,
                "kind": d.get("rule", ""),
                "message": d.get("message", ""),
                "severity": d.get("severity", "error"),
            }
        )
    return errors


def parse_error_count_mypy(output: str) -> int:
    """Parse error count from mypy output.

    Mypy summary line: 'Found 42 errors in 10 files (checked 50 source files)'
    or: 'Success: no issues found in 50 source files'
    """
    for line in reversed(output.splitlines()):
        if "no issues found" in line.lower():
            return 0
        m = re.search(r"Found\s+(\d+)\s+errors?", line)
        if m:
            return int(m.group(1))
    return -1


def parse_full_errors_mypy(output: str) -> list[dict[str, object]]:
    """Parse mypy text output into structured error list.

    Mypy outputs errors as:
      file.py:10: error: Incompatible return type  [return-value]
      file.py:20:5: error: Message  [code]

    We only capture error/warning lines, not notes.
    """
    errors = []
    pattern = re.compile(
        r"^(.+?):(\d+)(?::(\d+))?: (error|warning): (.+?)(?:\s+\[(.+?)\])?\s*$"
    )
    for line in output.splitlines():
        m = pattern.match(line)
        if m:
            errors.append(
                {
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "col": int(m.group(3)) if m.group(3) else 0,
                    "kind": m.group(6) or "",
                    "message": m.group(5),
                    "severity": m.group(4),
                }
            )
    return errors


def generate_table(
    results: list[dict[str, object]], output_file: str | None, csv_file: str | None
) -> None:
    """Generate and print summary table, optionally write to file and CSV."""
    lines = []
    header = (
        f"{'Project':<35} {'Pyrefly Err':>12} {'Pyright Err':>12} {'Mypy Err':>12}"
        f" {'Pyrefly (s)':>12} {'Pyright (s)':>12} {'Mypy (s)':>12}"
    )
    separator = "-" * 123
    lines.append(header)
    lines.append(separator)
    for r in results:
        lines.append(
            f"{r['project']:<35} {r['pyrefly_errors']:>12} {r['pyright_errors']:>12} {r['mypy_errors']:>12}"
            f" {r['pyrefly_time']:>12} {r['pyright_time']:>12} {r['mypy_time']:>12}"
        )

    total_pyrefly = sum(
        r["pyrefly_errors"]
        for r in results
        if isinstance(r["pyrefly_errors"], int) and r["pyrefly_errors"] >= 0
    )
    total_pyright = sum(
        r["pyright_errors"]
        for r in results
        if isinstance(r["pyright_errors"], int) and r["pyright_errors"] >= 0
    )
    total_mypy = sum(
        r["mypy_errors"]
        for r in results
        if isinstance(r["mypy_errors"], int) and r["mypy_errors"] >= 0
    )
    lines.append(separator)
    lines.append(
        f"{'TOTAL':<35} {total_pyrefly:>12} {total_pyright:>12} {total_mypy:>12}"
    )

    table = "\n" + "\n".join(lines) + "\n"
    print(table)

    if output_file:
        with open(output_file, "w") as f:
            f.write(table)
        logging.info(f"Table written to {output_file}")

    if csv_file:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "project",
                    "pyrefly_errors",
                    "pyright_errors",
                    "mypy_errors",
                    "pyrefly_time",
                    "pyright_time",
                    "mypy_time",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Results written to {csv_file}")


def write_json_output(
    results: list[dict[str, object]],
    projects: list["Project"],
    output_path: str,
) -> None:
    """Write structured JSON with full error details for all projects.

    Output format:
    {
      "timestamp": "...",
      "projects": [
        {
          "name": "...",
          "url": "...",
          "pyrefly": {"errors": [...], "error_count": N, "duration_sec": T},
          "pyright": {"errors": [...], "error_count": N, "duration_sec": T},
          "mypy": {"errors": [...], "error_count": N, "duration_sec": T}
        }
      ]
    }
    """
    # Build a url lookup from the projects list
    url_by_name = {p.name: p.location for p in projects}
    project_entries = []
    for r in results:
        name = r["project"]
        entry: dict[str, object] = {
            "name": name,
            "url": url_by_name.get(name, ""),  # type: ignore[arg-type]
        }
        for checker in ("pyrefly", "pyright", "mypy"):
            error_list = r.get(f"{checker}_error_list", [])
            entry[checker] = {
                "errors": error_list,
                "error_count": r.get(f"{checker}_errors", 0),
                "duration_sec": r.get(f"{checker}_time", 0),
            }
        project_entries.append(entry)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "projects": project_entries,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(
        f"Wrote {len(project_entries)} projects to {output_path} ({size_mb:.1f} MB)"
    )


def _has_pyright_config(repo_dir: str) -> bool:
    """Check whether the repo has a pyright configuration.

    Pyright reads config from pyrightconfig.json, or from a [tool.pyright]
    section in pyproject.toml.
    """
    if os.path.exists(os.path.join(repo_dir, "pyrightconfig.json")):
        return True
    pyproject = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproject):
        with open(pyproject) as f:
            for line in f:
                if re.match(r"\s*\[tool\.pyright\]", line):
                    return True
    return False


def _remove_pyrefly_section(pyproject_path: str) -> None:
    """Remove any [tool.pyrefly] section from a pyproject.toml file.

    Needed because repos may have [tool.pyrefly] with error codes from a
    newer pyrefly version that cause config parse errors on older versions.
    """
    with open(pyproject_path) as f:
        lines = f.readlines()

    result = []
    in_pyrefly = False
    for line in lines:
        # Detect start of [tool.pyrefly] or [[tool.pyrefly.*]]
        if re.match(r"\s*\[+tool\.pyrefly", line):
            in_pyrefly = True
            continue
        # Detect start of a new section (not pyrefly)
        if (
            in_pyrefly
            and re.match(r"\s*\[", line)
            and not re.match(r"\s*\[+tool\.pyrefly", line)
        ):
            in_pyrefly = False
        if not in_pyrefly:
            result.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(result)


def extract_paths_from_cmd(cmd: str) -> list[str]:
    """Extract file/directory paths from a checker command like '{pyrefly} src tests'."""
    parts = cmd.split()
    return [p for p in parts[1:] if not p.startswith("-") and not p.startswith("{")]


def check_project(
    project: Project,
    repo_dir: str,
    pyrefly_bin: str,
    debug: bool,
    full_errors: bool = False,
) -> dict[str, object]:
    """Run pyrefly, pyright, and mypy on a project, return a results row.

    When full_errors is True, checkers run in JSON mode and the result
    includes per-error detail lists under '{checker}_error_list' keys.
    """
    site_paths = setup_project(project, repo_dir, debug, reuse=os.path.exists(repo_dir))

    paths = extract_paths_from_cmd(project.pyrefly_cmd)
    # Filter to paths that actually exist — repos restructure over time and
    # stale entries in projects.py cause pyright to exit without parseable output.
    existing = [p for p in paths if os.path.exists(os.path.join(repo_dir, p))]
    if paths and not existing:
        logging.warning(
            f"  {project.name}: none of the target paths exist ({paths}), falling back to '.'"
        )
    path_args = " ".join(existing) if existing else "."

    # Venv activate path for running mypy inside the project's venv
    venv_dir = os.path.join(repo_dir, ".venv")
    activate = os.path.join(venv_dir, "bin", "activate")

    # Pyrefly — init to auto-detect configs, exclude venv via CLI flag
    # (config-file project-excludes is unreliable, CLI flag works)
    # Clean any existing pyrefly config first to avoid version-incompatible
    # error codes (e.g., repos with [tool.pyrefly] using newer error names).
    pyrefly_toml = os.path.join(repo_dir, "pyrefly.toml")
    if os.path.exists(pyrefly_toml):
        os.remove(pyrefly_toml)
    pyproject_toml = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproject_toml):
        _remove_pyrefly_section(pyproject_toml)

    if _has_pyright_config(repo_dir):
        logging.info("  Found pyright config, migrating")
    else:
        logging.info("  No pyright config, using default init")
    run(f"{pyrefly_bin} init", debug, cwd=repo_dir, shell=True)

    # When full_errors is requested, run pyrefly in JSON mode
    output_format = " --output-format json" if full_errors else ""
    pyrefly_cmd = project.pyrefly_cmd.format(
        pyrefly=f'{pyrefly_bin} check --project-excludes "**/.venv/**"{output_format}'
    )
    if site_paths:
        pyrefly_cmd += " " + site_paths

    start = time.time()
    pr = run(
        pyrefly_cmd,
        debug,
        cwd=repo_dir,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    pyrefly_time = time.time() - start

    if full_errors:
        pyrefly_error_list = parse_full_errors_pyrefly(pr.stdout or "")
        pyrefly_errors = len(
            [e for e in pyrefly_error_list if e.get("severity") == "error"]
        )
    else:
        pyrefly_output = (pr.stdout or "") + (pr.stderr or "")
        pyrefly_errors = parse_error_count(pyrefly_output)
        pyrefly_error_list = None

    # Pyright — use same paths as pyrefly for apples-to-apples.
    # Point pyright at the project's venv Python so it resolves installed deps.
    # When full_errors is requested, run pyright in JSON mode.
    # 10 minute timeout — large projects can hang.
    # For projects where pyright hangs on '.', use a targeted path instead.
    _PYRIGHT_PATH_OVERRIDES = {
        # scipy-stubs: match pyrefly project-includes to avoid analyzing the
        # entire repo (which causes pyright to hang/OOM on JSON output)
        "scipy-stubs": "scipy-stubs scripts tests",
    }
    pyright_json_flag = "--outputjson " if full_errors else ""
    pyright_path_args = _PYRIGHT_PATH_OVERRIDES.get(project.name, path_args)
    venv_python = os.path.join(venv_dir, "bin", "python")
    pyright_venv_flag = (
        f"--pythonpath {venv_python} " if os.path.exists(venv_python) else ""
    )
    start = time.time()
    pp = run(
        f"pyright {pyright_venv_flag}{pyright_json_flag}{pyright_path_args}",
        debug,
        cwd=repo_dir,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    pyright_time = time.time() - start

    if full_errors:
        pyright_error_list = parse_full_errors_pyright(pp.stdout or "", repo_dir)
        pyright_errors = len(
            [e for e in pyright_error_list if e.get("severity") == "error"]
        )
    else:
        pyright_output = (pp.stdout or "") + (pp.stderr or "")
        pyright_errors = parse_error_count(pyright_output)
        pyright_error_list = None

    # Mypy — run inside the project's venv so it can resolve installed deps.
    # The mypy_cmd uses {mypy} placeholder, replaced with just "mypy" from the venv.
    # 10 minute timeout to avoid hanging on large projects.
    mypy_cmd = project.mypy_cmd.format(mypy="mypy")
    start = time.time()
    pm = run(
        f". {activate} && {mypy_cmd}",
        debug,
        cwd=repo_dir,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    mypy_time = time.time() - start

    mypy_output = (pm.stdout or "") + (pm.stderr or "")
    if full_errors:
        mypy_error_list = parse_full_errors_mypy(mypy_output)
        mypy_errors = len([e for e in mypy_error_list if e.get("severity") == "error"])
    else:
        mypy_errors = parse_error_count_mypy(mypy_output)
        mypy_error_list = None

    logging.info(
        f"  pyrefly: {pyrefly_errors} errors in {pyrefly_time:.1f}s | "
        f"pyright: {pyright_errors} errors in {pyright_time:.1f}s | "
        f"mypy: {mypy_errors} errors in {mypy_time:.1f}s"
    )
    result: dict[str, object] = {
        "project": project.name,
        "pyrefly_errors": pyrefly_errors,
        "pyright_errors": pyright_errors,
        "mypy_errors": mypy_errors,
        "pyrefly_time": round(pyrefly_time, 2),
        "pyright_time": round(pyright_time, 2),
        "mypy_time": round(mypy_time, 2),
    }
    if full_errors:
        result["pyrefly_error_list"] = pyrefly_error_list
        result["pyright_error_list"] = pyright_error_list
        result["mypy_error_list"] = mypy_error_list
        result["url"] = project.location
    return result


def get_projects(names: list[str] | None) -> list[Project]:
    """Get filtered list of projects that have all three checker commands."""
    projects = [
        p
        for p in get_mypy_primer_projects()
        if p.pyrefly_cmd and p.pyright_cmd and p.mypy_cmd and not p.skip_pyrefly
    ]
    if names:
        lower_names = {n.lower() for n in names}
        projects = [p for p in projects if p.name.lower() in lower_names]
    return projects


def resolve_repo_dir(
    project: Project,
    cache_dir: str | None,
    tmp_dir: str | None,
    reuse_cache: bool,
) -> str:
    """Determine the repo directory for a project, cleaning stale dirs if needed."""
    if cache_dir:
        repo_dir = os.path.join(cache_dir, project.name)
        if not reuse_cache and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        os.makedirs(cache_dir, exist_ok=True)
    else:
        repo_dir = os.path.join(tmp_dir, project.name)
    return repo_dir


def _apply_sharding(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    projects: list,
) -> list:
    """Validate sharding args and return the sharded project list."""
    if args.shard_index is None and args.num_shards is None:
        return projects
    if args.shard_index is None or args.num_shards is None:
        parser.error("--shard-index and --num-shards must be used together")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    projects = projects[args.shard_index :: args.num_shards]
    logging.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(projects)} projects"
    )
    return projects


def _run_checkers(
    projects: list,
    pyrefly_bin: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Run all three type checkers on each project."""
    results: list[dict[str, object]] = []
    tmp_dir = None if args.cache_dir else tempfile.mkdtemp()

    try:
        for project in projects:
            logging.info(f"=== {project.name} ===")
            try:
                repo_dir = resolve_repo_dir(
                    project, args.cache_dir, tmp_dir, args.reuse_cache
                )
                results.append(
                    check_project(
                        project,
                        repo_dir,
                        pyrefly_bin,
                        args.debug,
                        full_errors=bool(args.output_json),
                    )
                )
            except Exception as e:
                logging.error(f"  FAILED: {e}")
                results.append(
                    {
                        "project": project.name,
                        "pyrefly_errors": "ERR",
                        "pyright_errors": "ERR",
                        "mypy_errors": "ERR",
                        "pyrefly_time": 0,
                        "pyright_time": 0,
                        "mypy_time": 0,
                    }
                )
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pyrefly vs pyright vs mypy on mypy_primer projects"
    )
    parser.add_argument(
        "--pyrefly", help="Path to pyrefly binary (default: build via cargo)"
    )
    parser.add_argument("-k", "--project", nargs="+", help="Only run these projects")
    parser.add_argument("--cache-dir", help="Persistent cache dir for cloned repos")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing cached repos (skip clone, reuse venv if present)",
    )
    parser.add_argument(
        "--clone-only",
        action="store_true",
        help="Only clone projects into --cache-dir, don't run checkers",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove --cache-dir and all cloned repos/venvs, then exit",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--csv", help="Write results to CSV file")
    parser.add_argument("--output", "-o", help="Write summary table to a file")
    parser.add_argument(
        "--output-json",
        help="Write structured JSON with full error messages to this file",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Run only this shard (0-indexed). Requires --num-shards.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Total number of shards to split projects across.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    projects = get_projects(args.project)

    if not projects:
        logging.info("No matching projects found")
        return

    projects = _apply_sharding(parser, args, projects)

    # Cleanup mode: remove cache dir and exit
    if args.cleanup:
        if not args.cache_dir:
            parser.error("--cleanup requires --cache-dir")
        if os.path.exists(args.cache_dir):
            shutil.rmtree(args.cache_dir)
            logging.info(f"Removed {args.cache_dir}")
        else:
            logging.info(f"{args.cache_dir} does not exist, nothing to clean")
        return

    # Clone-only mode: just clone repos and exit
    if args.clone_only:
        if not args.cache_dir:
            parser.error("--clone-only requires --cache-dir")
        clone_projects(projects, args.cache_dir, args.debug)
        logging.info(f"Cloned {len(projects)} projects into {args.cache_dir}")
        return

    if not shutil.which("pyright"):
        parser.error(
            "pyright not found on PATH. Install it with: pip install pyright (in a conda env with its own pip)"
        )

    if not shutil.which("mypy"):
        parser.error("mypy not found on PATH. Install it with: pip install mypy")

    pyrefly_bin = os.path.abspath(args.pyrefly) if args.pyrefly else build_pyrefly()
    results = _run_checkers(projects, pyrefly_bin, args)

    if args.output_json:
        write_json_output(results, projects, args.output_json)
    else:
        generate_table(results, args.output, args.csv)


if __name__ == "__main__":
    main()
