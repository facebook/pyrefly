#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 2: Primer impact — match issues against primer error data.

Mostly deterministic string matching. Small Haiku call for fuzzy
matching of error kinds that don't directly match, and a second
Haiku call to assess how specific a matched pattern is to the issue.
"""

from __future__ import annotations

import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json, LLMError

HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _templatize(message: str) -> str:
    """Generalize an error message into a template by replacing
    backtick-quoted identifiers with '`_`'."""
    return re.sub(r"`[^`]+`", "`_`", message)


def _build_primer_index(primer_data: dict) -> dict:
    """Build an index of (error_kind, message_template) to affected projects and counts.

    Message templates are created by replacing backtick-quoted identifiers
    with '_', so e.g. "Argument `x` is not assignable to parameter `y`"
    becomes "Argument `_` is not assignable to parameter `_`".

    Returns:
        {
            (error_kind, template): {
                "projects": {project_name: count},
                "total_count": int,
            }
        }
    """
    index: dict[tuple[str, str], dict] = {}
    for project in primer_data.get("projects", []):
        name = project.get("name", "")
        for error in project.get("pyrefly", {}).get("errors", []):
            kind = error.get("kind", "")
            if not kind:
                continue
            template = _templatize(error.get("message", ""))
            key = (kind, template)
            if key not in index:
                index[key] = {"projects": {}, "total_count": 0}
            if name not in index[key]["projects"]:
                index[key]["projects"][name] = 0
            index[key]["projects"][name] += 1
            index[key]["total_count"] += 1
    return index


def _fuzzy_match_kind(issue_kind: str, primer_kinds: list[str]) -> str | None:
    """Try fuzzy matching of an issue error kind against primer kinds.

    Matches against the kind component extracted from (kind, template) keys.
    First tries exact match, then normalized match, then substring containment.
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


def _assess_pattern_specificity(
    issue_title: str,
    issue_body: str,
    matched_kind: str,
    matched_template: str,
    error_count: int,
    project_count: int,
) -> dict:
    """Use Haiku to assess how specific a primer error pattern is to this issue.

    Returns {"specificity": "high"|"medium"|"low", "note": "..."} where:
    - high: this issue is likely the primary cause of these errors
    - medium: this issue is one of several possible causes
    - low: this is a generic pattern with many unrelated root causes
    """
    system = (
        "You are analyzing whether a GitHub issue for a Python type checker "
        "is likely the specific root cause of an error pattern seen across "
        "real-world projects.\n\n"
        "Given the issue description and a matched error pattern with its "
        "frequency, assess specificity:\n"
        "- HIGH: The issue describes a specific bug that would directly "
        "produce this exact error pattern. Fixing this issue would likely "
        "fix most/all of these errors.\n"
        "- MEDIUM: The issue is one of several possible causes. The error "
        "pattern is somewhat specific but could be triggered by other bugs "
        "too.\n"
        "- LOW: The error pattern is generic (e.g., type mismatch, missing "
        "attribute) and many unrelated issues could produce it. This issue "
        "is just one of potentially dozens of root causes.\n\n"
        "Respond with JSON:\n"
        '{"specificity": "high"|"medium"|"low", '
        '"note": "1 sentence explaining why"}'
    )
    user = (
        f"Issue: {issue_title}\n"
        f"Description: {issue_body[:500]}\n\n"
        f"Matched error pattern: {matched_kind}: {matched_template}\n"
        f"This pattern appears {error_count} times across {project_count} projects.\n"
    )

    try:
        parsed = call_llm_json(system, user, model=HAIKU_MODEL)
        specificity = parsed.get("specificity", "medium")
        if specificity not in ("high", "medium", "low"):
            specificity = "medium"
        return {
            "specificity": specificity,
            "note": parsed.get("note", ""),
        }
    except LLMError:
        return {"specificity": "unknown", "note": ""}


def compute_primer_impact(
    issues: list[dict],
    primer_data: dict | None,
    categorizations: dict[int, dict],
) -> dict[int, dict]:
    """Compute primer impact for all issues.

    Matches by (kind, message_template) for precise counts.  Falls back
    to kind-only aggregation when no template match is found.

    Returns: {issue_number: {primer_project_count, primer_error_count,
              matched_projects, matched_kind, matched_template,
              pattern_specificity, specificity_note}}
    """
    if not primer_data:
        return {
            i.get("number", 0): {
                "primer_project_count": 0,
                "primer_error_count": 0,
                "matched_projects": [],
                "matched_kind": "",
                "matched_template": "",
                "pattern_specificity": "",
                "specificity_note": "",
            }
            for i in issues
        }

    primer_index = _build_primer_index(primer_data)
    # Unique kind strings for fuzzy/LLM fallback matching
    unique_kinds = sorted({k for k, _ in primer_index})
    results: dict[int, dict] = {}

    for issue in issues:
        num = issue.get("number", 0)
        cat = categorizations.get(num, {})

        # Extract (kind, template) pairs from checker results
        error_patterns: list[tuple[str, str]] = []
        error_kinds: list[str] = []
        checker = issue.get("checker_results", {})
        if checker:
            for err in checker.get("pyrefly", []):
                kind = err.get("kind", "")
                message = err.get("message", "")
                if kind:
                    template = _templatize(message)
                    pattern = (kind, template)
                    if pattern not in error_patterns:
                        error_patterns.append(pattern)
                    if kind not in error_kinds:
                        error_kinds.append(kind)

        # Also try the subcategory as a kind hint (no template available)
        subcat = cat.get("subcategory", "")
        if subcat and subcat not in error_kinds:
            error_kinds.append(subcat)

        # 1) Try exact (kind, template) match against primer index
        best_match = None
        best_count = 0
        for pattern in error_patterns:
            if pattern in primer_index:
                entry = primer_index[pattern]
                if entry["total_count"] > best_count:
                    best_match = pattern
                    best_count = entry["total_count"]

        # 2) Fallback: aggregate all primer entries for a matching kind
        if not best_match and error_kinds:
            for kind in error_kinds:
                matched_kind = _fuzzy_match_kind(kind, unique_kinds)
                if not matched_kind:
                    continue
                # Aggregate all templates for this kind
                agg_projects: dict[str, int] = {}
                agg_total = 0
                for (pk, _), entry in primer_index.items():
                    if pk == matched_kind:
                        for proj, cnt in entry["projects"].items():
                            agg_projects[proj] = agg_projects.get(proj, 0) + cnt
                        agg_total += entry["total_count"]
                if agg_total > 0:
                    results[num] = {
                        "primer_project_count": len(agg_projects),
                        "primer_error_count": agg_total,
                        "matched_projects": sorted(agg_projects.keys()),
                        "matched_kind": matched_kind,
                        "matched_template": "(all patterns)",
                    }
                    break

            # 3) LLM fuzzy match as last resort
            if num not in results and error_kinds:
                llm_match = _llm_fuzzy_match(
                    error_kinds[0], issue.get("title", ""), unique_kinds
                )
                if llm_match:
                    agg_projects = {}
                    agg_total = 0
                    for (pk, _), entry in primer_index.items():
                        if pk == llm_match:
                            for proj, cnt in entry["projects"].items():
                                agg_projects[proj] = agg_projects.get(proj, 0) + cnt
                            agg_total += entry["total_count"]
                    if agg_total > 0:
                        results[num] = {
                            "primer_project_count": len(agg_projects),
                            "primer_error_count": agg_total,
                            "matched_projects": sorted(agg_projects.keys()),
                            "matched_kind": llm_match,
                            "matched_template": "(all patterns)",
                        }

        # Record exact pattern match result
        if best_match and num not in results:
            entry = primer_index[best_match]
            results[num] = {
                "primer_project_count": len(entry["projects"]),
                "primer_error_count": entry["total_count"],
                "matched_projects": sorted(entry["projects"].keys()),
                "matched_kind": best_match[0],
                "matched_template": best_match[1],
            }

        # No match at all
        if num not in results:
            results[num] = {
                "primer_project_count": 0,
                "primer_error_count": 0,
                "matched_projects": [],
                "matched_kind": "",
                "matched_template": "",
                "pattern_specificity": "",
                "specificity_note": "",
            }

        # LLM specificity assessment for issues with a primer match
        r = results[num]
        if r["primer_error_count"] > 0:
            spec = _assess_pattern_specificity(
                issue_title=issue.get("title", ""),
                issue_body=issue.get("body", "") or "",
                matched_kind=r["matched_kind"],
                matched_template=r["matched_template"],
                error_count=r["primer_error_count"],
                project_count=r["primer_project_count"],
            )
            r["pattern_specificity"] = spec["specificity"]
            r["specificity_note"] = spec["note"]
        else:
            r["pattern_specificity"] = ""
            r["specificity_note"] = ""

        logging.debug(
            f"  #{num}: primer impact = {r['primer_project_count']} projects, "
            f"{r['primer_error_count']} errors"
            f" (template: {r['matched_template'][:50]},"
            f" specificity: {r['pattern_specificity']})"
        )

    return results
