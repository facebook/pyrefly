#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 2: Primer impact — match issues against primer error data.

Mostly deterministic string matching. Small Haiku call only for fuzzy
matching of error kinds that don't directly match.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json, LLMError

HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _build_primer_index(primer_data: dict) -> dict:
    """Build an index of error kinds to affected projects and counts.

    Returns:
        {
            error_kind: {
                "projects": {project_name: count},
                "total_count": int,
            }
        }
    """
    index: dict[str, dict] = {}
    for project in primer_data.get("projects", []):
        name = project.get("name", "")
        for error in project.get("pyrefly", {}).get("errors", []):
            kind = error.get("kind", "")
            if not kind:
                continue
            if kind not in index:
                index[kind] = {"projects": {}, "total_count": 0}
            if name not in index[kind]["projects"]:
                index[kind]["projects"][name] = 0
            index[kind]["projects"][name] += 1
            index[kind]["total_count"] += 1
    return index


def _fuzzy_match_kind(issue_kind: str, primer_kinds: list[str]) -> str | None:
    """Try fuzzy matching of an issue error kind against primer kinds.

    First tries exact match, then substring containment, then falls back
    to a small Haiku call for semantic matching.
    """
    # Exact match
    if issue_kind in primer_kinds:
        return issue_kind

    # Normalize (strip hyphens, lowercase)
    norm_issue = issue_kind.lower().replace("-", "").replace("_", "")
    for pk in primer_kinds:
        norm_pk = pk.lower().replace("-", "").replace("_", "")
        if norm_issue == norm_pk:
            return pk

    # Substring match
    for pk in primer_kinds:
        if issue_kind.lower() in pk.lower() or pk.lower() in issue_kind.lower():
            return pk

    return None


def _llm_fuzzy_match(
    issue_kind: str, issue_title: str, primer_kinds: list[str]
) -> str | None:
    """Use Haiku to find the best matching primer error kind."""
    system = (
        "Match the issue error kind to the closest primer error kind. "
        'Respond with JSON: {"match": "exact_kind_name"} or '
        '{"match": null} if no match.'
    )
    user = (
        f"Issue: {issue_title}\n"
        f"Issue error kind: {issue_kind}\n"
        f"Available primer kinds: {', '.join(primer_kinds[:50])}"
    )

    try:
        parsed = call_llm_json(system, user, model=HAIKU_MODEL)
        match = parsed.get("match")
        if match and match in primer_kinds:
            return match
    except LLMError:
        pass

    return None


def compute_primer_impact(
    issues: list[dict],
    primer_data: dict | None,
    categorizations: dict[int, dict],
) -> dict[int, dict]:
    """Compute primer impact for all issues.

    Returns: {issue_number: {primer_project_count, primer_error_count, matched_projects, matched_kind}}
    """
    if not primer_data:
        return {
            i.get("number", 0): {
                "primer_project_count": 0,
                "primer_error_count": 0,
                "matched_projects": [],
                "matched_kind": "",
            }
            for i in issues
        }

    primer_index = _build_primer_index(primer_data)
    primer_kinds = list(primer_index.keys())
    results: dict[int, dict] = {}

    for issue in issues:
        num = issue.get("number", 0)
        cat = categorizations.get(num, {})

        # Try to extract error kind from checker results or category
        error_kinds: list[str] = []
        checker = issue.get("checker_results", {})
        if checker:
            for err in checker.get("pyrefly", []):
                kind = err.get("kind", "")
                if kind and kind not in error_kinds:
                    error_kinds.append(kind)

        # Also try the subcategory as a kind hint
        subcat = cat.get("subcategory", "")
        if subcat and subcat not in error_kinds:
            error_kinds.append(subcat)

        # Match each kind against primer index
        best_match = None
        for kind in error_kinds:
            match = _fuzzy_match_kind(kind, primer_kinds)
            if match:
                best_match = match
                break

        # If no deterministic match, try LLM fuzzy
        if not best_match and error_kinds:
            best_match = _llm_fuzzy_match(
                error_kinds[0], issue.get("title", ""), primer_kinds
            )

        if best_match and best_match in primer_index:
            entry = primer_index[best_match]
            results[num] = {
                "primer_project_count": len(entry["projects"]),
                "primer_error_count": entry["total_count"],
                "matched_projects": sorted(entry["projects"].keys()),
                "matched_kind": best_match,
            }
        else:
            results[num] = {
                "primer_project_count": 0,
                "primer_error_count": 0,
                "matched_projects": [],
                "matched_kind": "",
            }

        logging.debug(
            f"  #{num}: primer impact = {results[num]['primer_project_count']} projects, "
            f"{results[num]['primer_error_count']} errors"
        )

    return results
