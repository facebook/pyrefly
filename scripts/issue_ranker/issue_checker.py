#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run pyrefly, pyright, and mypy on code snippets from issues.

Writes each snippet to a temp file, runs all three checkers, and
returns structured error lists. Reuses parsers from compare_typecheckers.py.

If checkers report missing-import errors, we attempt to pip install the
missing packages and re-run once. Import errors that persist after the
retry are filtered out (snippets can't be expected to have all deps).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compare_typecheckers import (
    parse_full_errors_mypy,
    parse_full_errors_pyrefly,
    parse_full_errors_pyright,
)

from .dep_resolver import extract_module_from_error, install_missing_modules

# Error kinds that indicate a missing import — used to detect installable
# deps and to filter out unresolvable import errors after retry.
_IMPORT_ERROR_KINDS = {
    "missing-import",
    "missing-module",
    "reportMissingImports",
    "reportMissingModuleSource",
    "import",
    "import-not-found",
}


def _filter_import_errors(errors: list[dict]) -> list[dict]:
    """Remove import-related errors since snippets won't have resolved imports."""
    return [e for e in errors if e.get("kind", "") not in _IMPORT_ERROR_KINDS]


def _collect_import_errors(errors: list[dict]) -> list[dict]:
    """Return only the import-related errors from a list."""
    return [e for e in errors if e.get("kind", "") in _IMPORT_ERROR_KINDS]


def _run_checker(
    cmd: list[str], snippet_path: str, cwd: str
) -> subprocess.CompletedProcess[str]:
    """Run a checker command, returning the completed process."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        logging.warning(f"  Checker timed out: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="CHECKER_TIMEOUT")
    except FileNotFoundError:
        logging.debug(f"  Checker not found: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")


def _run_all_checkers(
    code_blocks: list[str],
    pyrefly_bin: str,
    has_pyright: bool,
    has_mypy: bool,
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Run pyrefly/pyright/mypy on all code blocks.

    Returns (pyrefly_errors, pyright_errors, mypy_errors, python_output_parts)
    with NO filtering applied — callers decide what to keep.
    """
    all_pyrefly: list[dict] = []
    all_pyright: list[dict] = []
    all_mypy: list[dict] = []
    python_output_parts: list[str] = []

    for i, code in enumerate(code_blocks):
        with tempfile.TemporaryDirectory() as tmpdir:
            snippet_file = os.path.join(tmpdir, f"snippet_{i}.py")
            with open(snippet_file, "w") as f:
                f.write(code)

            # Pyrefly (JSON output)
            result = _run_checker(
                [pyrefly_bin, "check", "--output-format", "json", snippet_file],
                snippet_file,
                tmpdir,
            )
            all_pyrefly.extend(parse_full_errors_pyrefly(result.stdout or ""))

            # Pyright (JSON output)
            if has_pyright:
                result = _run_checker(
                    ["pyright", "--outputjson", snippet_file],
                    snippet_file,
                    tmpdir,
                )
                all_pyright.extend(
                    parse_full_errors_pyright(result.stdout or "", tmpdir)
                )

            # Mypy
            if has_mypy:
                result = _run_checker(
                    ["mypy", "--no-error-summary", snippet_file],
                    snippet_file,
                    tmpdir,
                )
                all_mypy.extend(
                    parse_full_errors_mypy(
                        (result.stdout or "") + (result.stderr or "")
                    )
                )

            # Python3 execution — best-effort runtime check
            py_result = _run_checker(
                [sys.executable, snippet_file],
                snippet_file,
                tmpdir,
            )
            py_out = (py_result.stdout or "").strip()
            py_err = (py_result.stderr or "").strip()
            if py_out or (py_err and py_result.returncode != 0):
                part = f"[snippet {i}]"
                if py_out:
                    part += f" stdout: {py_out[:200]}"
                if py_err and py_result.returncode != 0:
                    part += f" error: {py_err[:200]}"
                python_output_parts.append(part)

    return all_pyrefly, all_pyright, all_mypy, python_output_parts


def check_snippets(
    code_blocks: list[str],
    pyrefly_bin: str,
) -> dict:
    """Run all three checkers on code snippets and return structured results.

    On the first run, if any checker reports missing-import errors, we attempt
    to pip install the missing packages and re-run all checkers once. After the
    final run, remaining import errors are filtered out.

    Also runs snippets with python3 to capture runtime output — useful
    when third-party deps make static checker results unreliable.

    Returns:
        {
            "pyrefly": [error, ...],
            "pyright": [error, ...],
            "mypy": [error, ...],
            "python_output": str,
            "snippet_count": int,
            "unresolved_deps": [module, ...],
        }
    """
    has_pyright = shutil.which("pyright") is not None
    has_mypy = shutil.which("mypy") is not None

    logging.info(
        f"    Checkers: pyrefly={pyrefly_bin}, "
        f"pyright={'yes' if has_pyright else 'NO'}, "
        f"mypy={'yes' if has_mypy else 'NO'}"
    )

    # First run.
    all_pyrefly, all_pyright, all_mypy, python_output_parts = _run_all_checkers(
        code_blocks, pyrefly_bin, has_pyright, has_mypy
    )

    # Collect import errors across all checkers and extract module names.
    import_errors = (
        _collect_import_errors(all_pyrefly)
        + _collect_import_errors(all_pyright)
        + _collect_import_errors(all_mypy)
    )
    missing_modules: set[str] = set()
    for err in import_errors:
        mod = extract_module_from_error(err)
        if mod:
            missing_modules.add(mod)

    # If we found missing modules, install them and re-run (one retry).
    unresolved_deps: list[str] = []
    if missing_modules:
        logging.info(f"    Missing modules detected: {sorted(missing_modules)}")
        installed, failed = install_missing_modules(missing_modules)
        if installed:
            logging.info("    Re-running checkers after installing deps...")
            all_pyrefly, all_pyright, all_mypy, python_output_parts = _run_all_checkers(
                code_blocks, pyrefly_bin, has_pyright, has_mypy
            )
        unresolved_deps = sorted(failed | (missing_modules - installed))

    # Filter remaining import errors (uninstallable deps).
    all_pyrefly = _filter_import_errors(all_pyrefly)
    all_pyright = _filter_import_errors(all_pyright)
    all_mypy = _filter_import_errors(all_mypy)

    logging.debug(
        f"    Results: pyrefly={len(all_pyrefly)} errors, "
        f"pyright={len(all_pyright)} errors, "
        f"mypy={len(all_mypy)} errors "
        f"(from {len(code_blocks)} snippets)"
    )
    if python_output_parts:
        logging.debug(f"    Python output: {'; '.join(python_output_parts)[:200]}")

    return {
        "pyrefly": all_pyrefly,
        "pyright": all_pyright,
        "mypy": all_mypy,
        "python_output": "; ".join(python_output_parts) if python_output_parts else "",
        "snippet_count": len(code_blocks),
        "unresolved_deps": unresolved_deps,
    }
