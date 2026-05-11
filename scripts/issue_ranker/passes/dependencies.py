#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 3: Dependency graph — find blocking chains and duplicate clusters.

Single Opus call with all issue summaries for cross-issue reasoning.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json

OPUS_MODEL = "claude-opus-4-20250514"

_SYSTEM_PROMPT = """You are analyzing GitHub issues for pyrefly (a Python type checker) to find dependencies, duplicates, and blocking relationships.

Given a list of issue summaries and any known relationships (parent/child, duplicates, blocked-by) extracted from the issue tracker, identify:
1. Dependency groups: issues that must be fixed together or in sequence
2. Blocking chains: issue A blocks issue B (fixing A enables fixing B)
3. Duplicate clusters: issues that describe the same underlying problem

Known relationships are ground truth — always include them in your output. Build on them by discovering additional relationships from the issue content.

Respond with JSON:
{
  "dependency_groups": [
    {"name": "group label", "issues": [123, 456], "reason": "why they're related"}
  ],
  "blocking_chains": [
    {"blocker": 123, "blocked": [456, 789], "reason": "why 123 blocks the others"}
  ],
  "duplicate_clusters": [
    {"canonical": 123, "duplicates": [456], "reason": "why they're duplicates"}
  ]
}

Be conservative — only flag new relationships you're confident about based on the issue content. Don't over-connect issues just because they share a label."""


def build_dependencies(
    issues: list[dict],
    categorizations: dict[int, dict],
    relationships: dict,
) -> dict:
    """Analyze all issues to find dependency groups, blocking chains, and duplicates.

    Makes a single Opus call with all issue summaries (~20K tokens).
    """
    # Build compact summaries for each issue
    summaries = []
    for issue in issues:
        num = issue.get("number", 0)
        cat = categorizations.get(num, {})
        summary = (
            f"#{num}: {issue.get('title', '')}\n"
            f"  Category: {cat.get('category', '?')} / {cat.get('subcategory', '?')}\n"
            f"  Labels: {', '.join(issue.get('labels', []))}\n"
            f"  Status: {issue.get('status_classification', '?')}\n"
        )
        # Add first 200 chars of body for context
        body = (issue.get("body", "") or "")[:200]
        if body:
            summary += f"  Body: {body}...\n"
        summaries.append(summary)

    # Include known relationships from the resolver
    known_rels = []
    for parent, children in relationships.get("parent_child", {}).items():
        known_rels.append(
            f"  #{parent} is parent of {['#' + str(c) for c in children]}"
        )
    for issue_num, dupes in relationships.get("duplicates", {}).items():
        known_rels.append(
            f"  #{issue_num} marked as duplicate of {['#' + str(d) for d in dupes]}"
        )
    for issue_num, blockers in relationships.get("blocked_by", {}).items():
        known_rels.append(
            f"  #{issue_num} blocked by {['#' + str(b) for b in blockers]}"
        )
    for issue_num, blocked in relationships.get("blocks", {}).items():
        known_rels.append(f"  #{issue_num} blocks {['#' + str(b) for b in blocked]}")

    user_prompt = f"Issues ({len(issues)} total):\n\n" + "\n".join(summaries)
    if known_rels:
        user_prompt += "\n\nKnown relationships:\n" + "\n".join(known_rels)

    logging.info(
        f"  Sending {len(issues)} issue summaries to Opus for dependency analysis..."
    )

    parsed = call_llm_json(
        _SYSTEM_PROMPT,
        user_prompt,
        model=OPUS_MODEL,
        max_tokens=8192,
        timeout=300,
    )

    dep_graph = {
        "dependency_groups": parsed.get("dependency_groups", []),
        "blocking_chains": parsed.get("blocking_chains", []),
        "duplicate_clusters": parsed.get("duplicate_clusters", []),
    }

    n_groups = len(dep_graph["dependency_groups"])
    n_chains = len(dep_graph["blocking_chains"])
    n_dupes = len(dep_graph["duplicate_clusters"])
    logging.info(
        f"  Found {n_groups} dependency groups, {n_chains} blocking chains, "
        f"{n_dupes} duplicate clusters"
    )

    return dep_graph
