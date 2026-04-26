#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parse issue relationships: duplicates, blockers, parent-child."""

from __future__ import annotations

import logging
import re
from collections import defaultdict


# Patterns for relationship markers in issue bodies.
_DUPLICATE_RE = re.compile(
    r"(?:duplicate\s+of|duplicates?|dupe\s+of)\s+#(\d+)", re.IGNORECASE
)
_BLOCKED_BY_RE = re.compile(
    r"(?:blocked\s+by|depends\s+on|waiting\s+on)\s+#(\d+)", re.IGNORECASE
)
_BLOCKS_RE = re.compile(r"(?<!code )(?<!code)\b(?:blocks?)\s+#(\d+)", re.IGNORECASE)


def resolve_relationships(issues: list[dict]) -> dict:
    """Parse relationships from issue bodies and GraphQL sub-issue data.

    Returns:
        {
            "duplicates": {issue_num: [duplicate_of_num, ...]},
            "blocked_by": {issue_num: [blocking_num, ...]},
            "blocks": {issue_num: [blocked_num, ...]},
            "parent_child": {parent_num: [child_num, ...]},
            "duplicate_clusters": [[num, num, ...], ...],
        }
    """
    issue_nums = {i["number"] for i in issues}
    duplicates: dict[int, list[int]] = defaultdict(list)
    blocked_by: dict[int, list[int]] = defaultdict(list)
    blocks: dict[int, list[int]] = defaultdict(list)
    parent_child: dict[int, list[int]] = defaultdict(list)

    for issue in issues:
        num = issue["number"]
        body = issue.get("body", "") or ""

        # Parse text-based relationships
        for match in _DUPLICATE_RE.finditer(body):
            target = int(match.group(1))
            if target in issue_nums:
                duplicates[num].append(target)

        for match in _BLOCKED_BY_RE.finditer(body):
            target = int(match.group(1))
            if target in issue_nums:
                blocked_by[num].append(target)

        for match in _BLOCKS_RE.finditer(body):
            target = int(match.group(1))
            if target in issue_nums:
                blocks[num].append(target)

        # GraphQL sub-issues (parent_issues = this issue is tracked in those)
        for parent in issue.get("parent_issues", []):
            parent_num = parent["number"]
            if parent_num in issue_nums:
                parent_child[parent_num].append(num)

        # GraphQL tracked issues (sub_issues = children of this issue)
        for child in issue.get("sub_issues", []):
            child_num = child["number"]
            if child_num in issue_nums:
                parent_child[num].append(child_num)

    # Build duplicate clusters using union-find
    duplicate_clusters = _build_duplicate_clusters(duplicates)

    # Log relationship summary
    logging.info(
        f"  Relationships found: {sum(len(v) for v in duplicates.values())} duplicates, "
        f"{sum(len(v) for v in blocked_by.values())} blocked-by, "
        f"{sum(len(v) for v in blocks.values())} blocks, "
        f"{sum(len(v) for v in parent_child.values())} parent-child, "
        f"{len(duplicate_clusters)} duplicate clusters"
    )

    return {
        "duplicates": dict(duplicates),
        "blocked_by": dict(blocked_by),
        "blocks": dict(blocks),
        "parent_child": dict(parent_child),
        "duplicate_clusters": duplicate_clusters,
    }


def _build_duplicate_clusters(
    duplicates: dict[int, list[int]],
) -> list[list[int]]:
    """Group duplicate issues into clusters using union-find."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for issue_num, dupe_targets in duplicates.items():
        for target in dupe_targets:
            union(issue_num, target)

    clusters: dict[int, list[int]] = defaultdict(list)
    all_nums = set(parent.keys())
    for dup_list in duplicates.values():
        all_nums.update(dup_list)
    for num in all_nums:
        clusters[find(num)].append(num)

    return [sorted(c) for c in clusters.values() if len(c) > 1]
