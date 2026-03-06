#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare pyrefly vs pyright error counts on mypy_primer projects.

Local-only script. Two-phase usage:

  Prerequisites:
    - A conda/venv environment with pyright installed: pip install pyright

  Phase 1 — Clone projects (run outside conda, proxy works):
    python3 compare_typecheckers.py --clone-only --cache-dir /tmp/primer_cache

  Phase 2 — Run checkers on cached repos (run inside conda):
    python3 compare_typecheckers.py --cache-dir /tmp/primer_cache --reuse-cache -o /tmp/results.txt

  Or if cloning works in your env, do it all at once:
    python3 compare_typecheckers.py --pyrefly /path/to/pyrefly --cache-dir /tmp/primer_cache -o /tmp/results.txt

  Cleanup:
    python3 compare_typecheckers.py --cleanup --cache-dir /tmp/primer_cache
"""

import argparse
import configparser
import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import venv

from projects import get_mypy_primer_projects, Project


def run(
    cmd: str | list[str], debug: bool, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    """Run a command, suppressing output unless debug is set."""
    logging.debug(cmd)
    if not debug and "capture_output" not in kwargs:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    return subprocess.run(cmd, **kwargs)  # noqa


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
    venv_dir = os.path.join(repo_dir, "_primer_venv")
    activate = os.path.join(venv_dir, "bin", "activate")

    if not reuse or not os.path.exists(venv_dir):
        venv.create(venv_dir, with_pip=True, clear=True)
        if deps:
            result = run(
                f"source {activate} && pip install " + " ".join(deps),
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
        f"source {activate} && python -c \"import site; print(' '.join('--site-package-path=' + p for p in site.getsitepackages()))\"",
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
        f"source {activate} && pip install .",
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
            f"source {activate} && pip uninstall -y {pkg_name}",
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


def generate_table(
    results: list[dict[str, object]], output_file: str | None, csv_file: str | None
) -> None:
    """Generate and print summary table, optionally write to file and CSV."""
    lines = []
    header = f"{'Project':<35} {'Pyrefly Err':>12} {'Pyright Err':>12} {'Pyrefly (s)':>12} {'Pyright (s)':>12}"
    separator = "-" * 87
    lines.append(header)
    lines.append(separator)
    for r in results:
        lines.append(
            f"{r['project']:<35} {r['pyrefly_errors']:>12} {r['pyright_errors']:>12} {r['pyrefly_time']:>12} {r['pyright_time']:>12}"
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
    lines.append(separator)
    lines.append(f"{'TOTAL':<35} {total_pyrefly:>12} {total_pyright:>12}")

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
                    "pyrefly_time",
                    "pyright_time",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Results written to {csv_file}")


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


def extract_paths_from_cmd(cmd: str) -> list[str]:
    """Extract file/directory paths from a checker command like '{pyrefly} src tests'."""
    parts = cmd.split()
    return [p for p in parts[1:] if not p.startswith("-") and not p.startswith("{")]


def check_project(
    project: Project, repo_dir: str, pyrefly_bin: str, debug: bool
) -> dict[str, object]:
    """Run pyrefly and pyright on a project, return a results row."""
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

    # Pyrefly — init to auto-detect configs, exclude venv via CLI flag
    # (config-file project-excludes is unreliable, CLI flag works)
    if _has_pyright_config(repo_dir):
        # Migrate from pyright config for apples-to-apples comparison
        logging.info("  Found pyright config, migrating")
        run(
            f"{pyrefly_bin} init --non-interactive --migrate-from pyright",
            debug,
            cwd=repo_dir,
            shell=True,
        )
    else:
        # No pyright config; default init (uses mypy config if present)
        logging.info("  No pyright config, using default init")
        run(f"{pyrefly_bin} init --non-interactive", debug, cwd=repo_dir, shell=True)
    pyrefly_cmd = project.pyrefly_cmd.format(
        pyrefly=f'{pyrefly_bin} check --project-excludes "**/_primer_venv/**"'
    )
    if site_paths:
        pyrefly_cmd += " " + site_paths

    start = time.time()
    pr = run(
        pyrefly_cmd, debug, cwd=repo_dir, shell=True, capture_output=True, text=True
    )
    pyrefly_time = time.time() - start
    pyrefly_output = (pr.stdout or "") + (pr.stderr or "")
    pyrefly_errors = parse_error_count(pyrefly_output)

    # Pyright — use same paths as pyrefly for apples-to-apples
    start = time.time()
    pp = run(
        f"pyright {path_args}",
        debug,
        cwd=repo_dir,
        shell=True,
        capture_output=True,
        text=True,
    )
    pyright_time = time.time() - start
    pyright_output = (pp.stdout or "") + (pp.stderr or "")
    pyright_errors = parse_error_count(pyright_output)

    logging.info(
        f"  pyrefly: {pyrefly_errors} errors in {pyrefly_time:.1f}s | "
        f"pyright: {pyright_errors} errors in {pyright_time:.1f}s"
    )
    return {
        "project": project.name,
        "pyrefly_errors": pyrefly_errors,
        "pyright_errors": pyright_errors,
        "pyrefly_time": round(pyrefly_time, 2),
        "pyright_time": round(pyright_time, 2),
    }


def get_projects(names: list[str] | None) -> list[Project]:
    """Get filtered list of projects that have both pyrefly and pyright commands."""
    projects = [
        p
        for p in get_mypy_primer_projects()
        if p.pyrefly_cmd and p.pyright_cmd and not p.skip_pyrefly
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pyrefly vs pyright on mypy_primer projects"
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    projects = get_projects(args.project)

    if not projects:
        logging.info("No matching projects found")
        return

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

    pyrefly_bin = os.path.abspath(args.pyrefly) if args.pyrefly else build_pyrefly()
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
                    check_project(project, repo_dir, pyrefly_bin, args.debug)
                )
            except Exception as e:
                logging.error(f"  FAILED: {e}")
                results.append(
                    {
                        "project": project.name,
                        "pyrefly_errors": "ERR",
                        "pyright_errors": "ERR",
                        "pyrefly_time": 0,
                        "pyright_time": 0,
                    }
                )
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    generate_table(results, args.output, args.csv)


if __name__ == "__main__":
    main()
