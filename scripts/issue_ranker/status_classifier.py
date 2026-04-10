#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Classify issue status using checker results + LLM analysis.

Uses a fast heuristic for obvious cases (no code, no errors) and an
LLM call for nuanced classification when checker results need
interpretation.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_transport import call_llm_json

HAIKU_MODEL = "claude-haiku-4-5-20251001"

VALID_STATUSES = {
    "false_positive",
    "false_negative",
    "already_fixed",
    "feature_request",
    "confirmed_bug",
}

_SYSTEM_PROMPT = """You are classifying the status of a GitHub issue for pyrefly, a Python type checker.

You receive the issue title, body, and the results of running three type checkers (pyrefly, pyright, mypy) on the issue's code snippet.

Classify the issue into exactly one status:
- "false_positive": Pyrefly reports an error but the code is actually valid (pyright/mypy agree the code is fine, or the error is clearly wrong based on the issue description)
- "false_negative": Pyrefly MISSES an error that it should report (pyright/mypy catch it, or the issue describes missing detection)
- "confirmed_bug": The error is real and all checkers agree, OR the error behavior described in the issue is confirmed by the checker output
- "already_fixed": The code snippet produces no relevant errors on pyrefly — the bug may have been fixed already
- "feature_request": The issue describes a feature that doesn't exist yet, not a bug in existing behavior

Consider:
- Error messages may differ between checkers — compare semantically, not literally
- reveal_type errors are informational, not real errors — ignore them when counting
- Import errors should be ignored (snippets won't have proper imports)
- Read the issue title/body to understand what the user is actually reporting

Respond with JSON:
{"status": "one of the five statuses", "reasoning": "1 sentence explaining why"}"""


def classify_status(
    checker_results: dict,
    issue_title: str = "",
    issue_body: str = "",
    unresolved_deps: list[str] | None = None,
) -> str:
    """Classify an issue's status using checker results and LLM analysis.

    For obvious cases (no code, no errors at all), uses a fast heuristic.
    For cases needing judgment, calls Haiku to analyze the checker output
    in context of the issue.

    When unresolved_deps is non-empty, checker results may be unreliable
    (checkers can't resolve third-party types). The LLM is told to rely
    on its own reasoning rather than trusting "0 errors."

    Returns one of:
      "false_positive"  — pyrefly errors incorrectly
      "false_negative"  — pyrefly misses errors it should catch
      "already_fixed"   — no errors on any checker
      "feature_request" — no code snippets to check
      "confirmed_bug"   — bug confirmed by checker output
    """
    # Fast path: no checker results means no code → feature request
    if not checker_results or checker_results.get("snippet_count", 0) == 0:
        return "feature_request"

    pyrefly = checker_results.get("pyrefly", [])
    pyright = checker_results.get("pyright", [])
    mypy = checker_results.get("mypy", [])

    # Filter out reveal_type from error counts (they're informational)
    pyrefly_real = [e for e in pyrefly if e.get("kind") != "reveal-type"]
    pyright_real = [
        e for e in pyright if "reveal_type" not in e.get("message", "").lower()
    ]
    mypy_real = [e for e in mypy if e.get("kind") != "note"]

    # When we have unresolved deps and no errors, don't trust the results —
    # the checkers couldn't resolve third-party types. Use LLM to classify
    # based on the issue description instead.
    if unresolved_deps and not pyrefly_real and not pyright_real and not mypy_real:
        logging.info(
            f"  Unresolved deps {unresolved_deps} — "
            f"checker results unreliable, using LLM classification"
        )
        return _llm_classify(
            pyrefly_real,
            pyright_real,
            mypy_real,
            issue_title,
            issue_body,
            unresolved_deps=unresolved_deps,
        )

    # Fast path: no real errors from any checker → already fixed
    if not pyrefly_real and not pyright_real and not mypy_real:
        return "already_fixed"

    # For all other cases, use LLM to classify
    return _llm_classify(
        pyrefly_real,
        pyright_real,
        mypy_real,
        issue_title,
        issue_body,
    )


def _format_errors(errors: list[dict], limit: int = 5) -> str:
    """Format error list for LLM prompt."""
    if not errors:
        return "  (no errors)"
    lines = []
    for e in errors[:limit]:
        kind = e.get("kind", "")
        line = e.get("line", "?")
        msg = e.get("message", "")[:100]
        lines.append(f"  L{line} [{kind}]: {msg}")
    if len(errors) > limit:
        lines.append(f"  ... and {len(errors) - limit} more errors")
    return "\n".join(lines)


def _llm_classify(
    pyrefly_real: list[dict],
    pyright_real: list[dict],
    mypy_real: list[dict],
    issue_title: str,
    issue_body: str,
    unresolved_deps: list[str] | None = None,
) -> str:
    """Use Haiku to classify the issue status based on checker output."""
    try:
        user_prompt = (
            f"Issue: {issue_title}\n"
            f"Body:\n{(issue_body or '')[:800]}\n\n"
            f"Pyrefly errors ({len(pyrefly_real)}):\n"
            f"{_format_errors(pyrefly_real)}\n\n"
            f"Pyright errors ({len(pyright_real)}):\n"
            f"{_format_errors(pyright_real)}\n\n"
            f"Mypy errors ({len(mypy_real)}):\n"
            f"{_format_errors(mypy_real)}\n"
        )

        if unresolved_deps:
            user_prompt += (
                f"\nWARNING: The following third-party dependencies could not be "
                f"resolved: {unresolved_deps}. The checker results above may be "
                f"UNRELIABLE — 0 errors likely means the checkers could not "
                f"analyze the code properly, NOT that the code is bug-free. "
                f"Base your classification on the issue description and code "
                f"semantics instead.\n"
            )

        parsed = call_llm_json(_SYSTEM_PROMPT, user_prompt, model=HAIKU_MODEL)

        status = parsed.get("status", "")
        reasoning = parsed.get("reasoning", "")
        if reasoning:
            logging.debug(f"  LLM reasoning: {reasoning}")

        if status in VALID_STATUSES:
            return status

        # If LLM returned an invalid status, fall back to heuristic
        logging.debug(f"  LLM returned invalid status '{status}', using heuristic")
        return _heuristic_classify(pyrefly_real, pyright_real, mypy_real)

    except Exception as e:
        logging.debug(f"  LLM classification failed: {e}, using heuristic")
        return _heuristic_classify(pyrefly_real, pyright_real, mypy_real)


def _heuristic_classify(
    pyrefly_real: list[dict],
    pyright_real: list[dict],
    mypy_real: list[dict],
) -> str:
    """Simple heuristic fallback when LLM is not available."""
    has_pyrefly = len(pyrefly_real) > 0
    has_other = len(pyright_real) > 0 or len(mypy_real) > 0

    if not has_pyrefly and not has_other:
        return "already_fixed"
    if has_pyrefly and not has_other:
        return "false_positive"
    if not has_pyrefly and has_other:
        return "false_negative"
    return "confirmed_bug"
