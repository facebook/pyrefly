# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-check pyrefly primer errors against mypy and pyright.

For each project in the primer diff:
1. Look up project config in projects.py (for mypy_cmd, pyright_cmd, deps)
2. Clone + set up venv with deps via setup_project(install_project=False)
3. Run mypy/pyright using the project's configured commands
4. LLM pass: send ALL pyrefly errors + ALL mypy/pyright errors, LLM labels
   each pyrefly error as also-flagged-by-mypy and/or also-flagged-by-pyright
5. Annotate each ErrorEntry with whether mypy/pyright also flag the same issue
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

from .parser import ErrorEntry, ProjectDiff

# scripts/ is not a Python package (no __init__.py) so relative imports can't
# reach compare_typecheckers or projects.  We add the parent directory to
# sys.path as the simplest workaround.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compare_typecheckers import (
    extract_paths_from_cmd,
    parse_full_errors_mypy,
    parse_full_errors_pyright,
    setup_project,
)
from llm_transport import (
    get_backend,
    call_llama_api,
    call_anthropic_api,
    extract_text,
    parse_json_list,
)
from projects import get_mypy_primer_projects, Project

_CHECKER_TIMEOUT = 300  # 5 minutes per checker
_BATCH_SIZE = 80  # max pyrefly errors per LLM call to avoid output truncation


def _find_project(name: str) -> Project | None:
    """Look up a project by name in the primer projects list."""
    for p in get_mypy_primer_projects():
        if p.name == name:
            return p
    return None


def _run_checker(
    cmd: str,
    repo_dir: str,
    checker_name: str,
    project_name: str,
) -> subprocess.CompletedProcess[str]:
    """Run a checker command in the project's venv."""
    activate = os.path.join(repo_dir, ".venv", "bin", "activate")
    full_cmd = f". {activate} && {cmd}"
    logging.info(f"  [cross-check] Running {checker_name} on {project_name}")
    try:
        return subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=_CHECKER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning(
            f"  [cross-check] {checker_name} timed out for {project_name}"
        )
        return subprocess.CompletedProcess(full_cmd, 1, stdout="", stderr="")


def _collect_checker_errors(
    project: Project,
    repo_dir: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run mypy and pyright, return (mypy_errors, pyright_errors).

    Each error is a dict with file, line, col, kind, message, severity.
    """
    mypy_errors: list[dict[str, object]] = []
    pyright_errors: list[dict[str, object]] = []

    if project.mypy_cmd:
        mypy_cmd = project.mypy_cmd.format(mypy="mypy")
        result = _run_checker(mypy_cmd, repo_dir, "mypy", project.name)
        mypy_errors = parse_full_errors_mypy(result.stdout or "")
        if result.returncode != 0 and not mypy_errors:
            logging.debug(
                f"  [cross-check] mypy exit={result.returncode}, "
                f"stderr={result.stderr if result.stderr else '(empty)'}"
            )
    else:
        logging.debug(f"  [cross-check] {project.name}: no mypy_cmd configured")

    if project.pyright_cmd:
        paths = extract_paths_from_cmd(project.pyright_cmd)
        path_args = " ".join(paths) if paths else "."
        python_path = os.path.join(repo_dir, ".venv", "bin", "python")
        pyright_cmd = (
            f"pyright --outputjson --pythonpath {python_path} {path_args}"
        )
        result = _run_checker(pyright_cmd, repo_dir, "pyright", project.name)
        pyright_errors = parse_full_errors_pyright(result.stdout or "", repo_dir)
        if result.returncode != 0 and not pyright_errors:
            logging.debug(
                f"  [cross-check] pyright exit={result.returncode}, "
                f"stderr={result.stderr if result.stderr else '(empty)'}"
            )
    else:
        logging.debug(
            f"  [cross-check] {project.name}: no pyright_cmd configured"
        )

    return mypy_errors, pyright_errors


def _format_checker_errors(
    errors: list[dict[str, object]], checker_name: str,
) -> str:
    """Format checker errors for the LLM prompt."""
    if not errors:
        return f"No {checker_name} errors."
    lines = [f"{checker_name} errors ({len(errors)} total):"]
    for e in errors:
        kind_str = f" [{e['kind']}]" if e.get("kind") else ""
        lines.append(f"  {e['file']}:{e['line']}: {e['message']}{kind_str}")
    return "\n".join(lines)


def _format_pyrefly_errors(entries: list[ErrorEntry], offset: int = 0) -> str:
    """Format pyrefly primer diff errors for the LLM prompt, indexed."""
    lines = []
    for i, e in enumerate(entries):
        lines.append(
            f"  [{offset + i}] {e.file_path}:{e.line_number}: {e.message} [{e.error_kind}]"
        )
    return f"Pyrefly new errors ({len(entries)} total):\n" + "\n".join(lines)


_MATCH_SYSTEM_PROMPT = """\
You are matching type checker errors across tools. Given pyrefly's new errors \
and the full error output from mypy and pyright on the same project, determine \
which pyrefly errors are also flagged by mypy and/or pyright.

Errors may not match exactly — different checkers:
- Report on slightly different lines (±5 lines is common)
- Use different error codes (e.g. pyrefly "bad-return" vs mypy "return-value" \
vs pyright "reportReturnType")
- Use different wording for the same issue

Match semantically: if mypy/pyright report a similar issue in the same file \
and roughly the same location, that counts as a match.

Respond with JSON only — an array with one entry per pyrefly error:
[{"index": 0, "mypy": true, "pyright": false}, ...]

Every pyrefly error index must appear exactly once. Use true/false, not \
strings. If a checker was not run (no errors section), use false for it."""


def _match_errors_batch(
    pyrefly_entries: list[ErrorEntry],
    mypy_errors: list[dict[str, object]],
    pyright_errors: list[dict[str, object]],
    offset: int,
    model: Optional[str],
) -> list[dict[str, bool]]:
    """Match a single batch of pyrefly errors against checker output via LLM."""
    backend, api_key = get_backend()
    if backend == "none":
        logging.warning("[cross-check] No LLM API key — skipping matching")
        return []

    user_prompt_parts = [
        _format_pyrefly_errors(pyrefly_entries, offset),
        "",
        _format_checker_errors(mypy_errors, "mypy"),
        "",
        _format_checker_errors(pyright_errors, "pyright"),
    ]
    user_prompt = "\n".join(user_prompt_parts)

    logging.info(
        f"  [cross-check] LLM matching: {len(pyrefly_entries)} pyrefly errors "
        f"(indices {offset}-{offset + len(pyrefly_entries) - 1}) "
        f"vs {len(mypy_errors)} mypy + {len(pyright_errors)} pyright"
    )

    try:
        if backend == "llama":
            result = call_llama_api(
                api_key, _MATCH_SYSTEM_PROMPT, user_prompt, model
            )
        else:
            result = call_anthropic_api(
                api_key, _MATCH_SYSTEM_PROMPT, user_prompt, model
            )

        text = extract_text(backend, result)
        logging.debug(
            f"  [cross-check] LLM response ({len(text)} chars): {text[:500]}..."
        )
        return parse_json_list(text)

    except Exception as e:
        logging.warning(f"  [cross-check] LLM matching failed: {e}")
        return []


def _filter_to_relevant_files(
    checker_errors: list[dict[str, object]],
    pyrefly_entries: list[ErrorEntry],
) -> list[dict[str, object]]:
    """Filter checker errors to only files that have pyrefly errors.

    Large projects can have tens of thousands of checker errors across
    hundreds of files. Sending all of them to the LLM would exceed the
    API token limit (200K). We filter to only files with pyrefly errors
    so the LLM sees every checker error in the relevant files — enough
    for accurate semantic matching without blowing past the token limit.
    """
    pyrefly_files = {e.file_path for e in pyrefly_entries}
    return [e for e in checker_errors if e.get("file") in pyrefly_files]


def _match_errors_with_llm(
    pyrefly_entries: list[ErrorEntry],
    mypy_errors: list[dict[str, object]],
    pyright_errors: list[dict[str, object]],
    model: Optional[str] = None,
) -> list[dict[str, bool]]:
    """LLM pass: label each pyrefly error as also-in-mypy / also-in-pyright.

    Filters checker errors to only files with pyrefly errors (to stay within
    the API token limit), then batches pyrefly errors to avoid output truncation.

    Returns a list of dicts with 'index', 'mypy', and 'pyright' bool fields.
    """
    if not pyrefly_entries:
        return []

    # Filter checker errors to files that have pyrefly errors.
    # Large projects (e.g. core with 28K pyright errors) would exceed the
    # 200K token API limit without this filtering.
    filtered_mypy = _filter_to_relevant_files(mypy_errors, pyrefly_entries)
    filtered_pyright = _filter_to_relevant_files(pyright_errors, pyrefly_entries)
    if len(filtered_mypy) != len(mypy_errors) or len(filtered_pyright) != len(pyright_errors):
        logging.info(
            f"  [cross-check] Filtered to relevant files: "
            f"mypy {len(mypy_errors)}->{len(filtered_mypy)}, "
            f"pyright {len(pyright_errors)}->{len(filtered_pyright)}"
        )

    # For small sets, single call. For large sets, batch to avoid
    # output truncation (each match entry is ~40 tokens).
    all_matches: list[dict[str, bool]] = []
    for batch_start in range(0, len(pyrefly_entries), _BATCH_SIZE):
        batch = pyrefly_entries[batch_start : batch_start + _BATCH_SIZE]
        batch_matches = _match_errors_batch(
            batch, filtered_mypy, filtered_pyright, batch_start, model
        )
        all_matches.extend(batch_matches)

    # Filter out malformed entries (LLM sometimes returns ints or strings
    # instead of dicts with index/mypy/pyright fields)
    all_matches = [m for m in all_matches if isinstance(m, dict)]

    # Validate completeness
    returned_indices = {m.get("index") for m in all_matches}
    expected_indices = set(range(len(pyrefly_entries)))
    missing = expected_indices - returned_indices
    if missing:
        logging.warning(
            f"  [cross-check] LLM omitted {len(missing)} indices: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''} "
            f"(returned {len(all_matches)}/{len(pyrefly_entries)})"
        )

    # Summarize
    mypy_matches = sum(1 for m in all_matches if m.get("mypy") is True)
    pyright_matches = sum(1 for m in all_matches if m.get("pyright") is True)
    both = sum(
        1
        for m in all_matches
        if m.get("mypy") is True and m.get("pyright") is True
    )
    neither = sum(
        1
        for m in all_matches
        if m.get("mypy") is not True and m.get("pyright") is not True
    )
    logging.info(
        f"  [cross-check] Match results: {len(all_matches)} entries — "
        f"mypy={mypy_matches}, pyright={pyright_matches}, "
        f"both={both}, pyrefly-only={neither}"
    )
    return all_matches


def cross_check_projects(
    projects: list[ProjectDiff],
    cache_dir: str | None = None,
    debug: bool = False,
    model: str | None = None,
) -> None:
    """Cross-check pyrefly primer errors against mypy/pyright via LLM labeling.

    For each project: clone, install deps (no pip install .), run mypy/pyright,
    then use an LLM to label which pyrefly errors are also flagged by the other
    checkers. Checker errors are filtered to files with pyrefly errors to stay
    within API token limits.

    Modifies projects in place — sets also_in_mypy / also_in_pyright
    on each ErrorEntry in project.added.
    """
    if not projects:
        return

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="primer_crosscheck_")
    else:
        os.makedirs(cache_dir, exist_ok=True)

    logging.info(f"[cross-check] Cross-checking {len(projects)} project(s)")

    for project_diff in projects:
        if not project_diff.added:
            continue

        project = _find_project(project_diff.name)
        if project is None:
            logging.info(
                f"  [cross-check] {project_diff.name}: not in projects.py, skipping"
            )
            continue

        if not project.mypy_cmd and not project.pyright_cmd:
            logging.info(
                f"  [cross-check] {project_diff.name}: no checker commands, skipping"
            )
            continue

        repo_dir = os.path.join(cache_dir, project_diff.name)

        # Clone + venv with explicit deps only (no pip install .)
        try:
            setup_project(
                project,
                repo_dir,
                debug,
                reuse=os.path.exists(repo_dir),
                install_project=False,
            )
        except Exception as e:
            logging.warning(
                f"  [cross-check] setup failed for {project_diff.name}: {e}"
            )
            continue

        # Install mypy/pyright into the project venv so they can access
        # the project's deps and plugins (e.g. envier mypy plugin).
        activate = os.path.join(repo_dir, ".venv", "bin", "activate")
        logging.info(
            f"  [cross-check] Installing mypy/pyright into {project_diff.name} venv"
        )
        try:
            pip_result = subprocess.run(
                f". {activate} && pip install mypy pyright",
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pip_result.returncode != 0:
                logging.warning(
                    f"  [cross-check] pip install failed for {project_diff.name}: "
                    f"{pip_result.stderr if pip_result.stderr else '(no stderr)'}"
                )
        except subprocess.TimeoutExpired:
            logging.warning(
                f"  [cross-check] pip install mypy/pyright timed out for {project_diff.name}"
            )

        # Run checkers
        mypy_errors, pyright_errors = _collect_checker_errors(project, repo_dir)
        project_diff.mypy_errors = mypy_errors
        project_diff.pyright_errors = pyright_errors
        logging.info(
            f"  [cross-check] {project_diff.name}: "
            f"mypy={len(mypy_errors)}, pyright={len(pyright_errors)}"
        )

        if not mypy_errors and not pyright_errors:
            # Checkers ran but found nothing — all entries are pyrefly-only.
            for entry in project_diff.added:
                entry.cross_checked = True
            logging.info(
                f"  [cross-check] {project_diff.name}: no checker errors, "
                f"all {len(project_diff.added)} errors are pyrefly-only"
            )
            continue

        # LLM labeling pass — dump full checker output, no filtering
        matches = _match_errors_with_llm(
            project_diff.added, mypy_errors, pyright_errors, model
        )

        # Apply annotations
        annotated = 0
        for match in matches:
            idx = match.get("index")
            if isinstance(idx, int) and 0 <= idx < len(project_diff.added):
                entry = project_diff.added[idx]
                entry.cross_checked = True
                if match.get("mypy") is True:
                    entry.also_in_mypy = True
                if match.get("pyright") is True:
                    entry.also_in_pyright = True
                annotated += 1

        logging.info(
            f"  [cross-check] {project_diff.name}: annotated "
            f"{annotated}/{len(project_diff.added)} errors"
        )

    # Final summary across all projects
    all_entries = [e for p in projects for e in p.added]
    checked = [e for e in all_entries if e.cross_checked]
    total_mypy = sum(1 for e in checked if e.also_in_mypy)
    total_pyright = sum(1 for e in checked if e.also_in_pyright)
    total_pyrefly_only = sum(
        1 for e in checked if not e.also_in_mypy and not e.also_in_pyright
    )
    unchecked = len(all_entries) - len(checked)
    logging.info(
        f"[cross-check] Done. {len(all_entries)} total errors, "
        f"{len(checked)} checked: {total_mypy} also-mypy, "
        f"{total_pyright} also-pyright, {total_pyrefly_only} pyrefly-only"
        + (f", {unchecked} not checked" if unchecked else "")
    )
