#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for issue ranker LLM passes (categorize, score, dependencies, rank).

Requires ANTHROPIC_API_KEY or LLAMA_API_KEY in the environment.
Uses small synthetic fixture data — not real primer results.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from passes.categorize import CATEGORIES, categorize_issue
from passes.dependencies import build_dependencies
from passes.rank import rank_issues
from passes.score import score_issue

_HAS_API_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLAMA_API_KEY")
)
_SKIP_REASON = "No API key set (need ANTHROPIC_API_KEY or LLAMA_API_KEY)"

# ── synthetic fixtures ───────────────────────────────────────────────

_ISSUE_FP = {
    "number": 9001,
    "title": "Pyrefly false positive on valid overload",
    "body": (
        "Pyrefly flags this valid code:\n"
        "```python\n"
        "from typing import overload\n"
        "@overload\n"
        "def f(x: int) -> int: ...\n"
        "@overload\n"
        "def f(x: str) -> str: ...\n"
        "def f(x): return x\n"
        "```"
    ),
    "labels": ["bug", "false-positive"],
    "reactions_count": 5,
    "comments_count": 2,
    "milestone": "",
    "priority": "P1",
    "sub_issues": [],
    "comments": [],
}

_ISSUE_FEATURE = {
    "number": 9002,
    "title": "Support ParamSpec in decorators",
    "body": "Pyrefly doesn't support ParamSpec for decorator typing.",
    "labels": ["enhancement", "typing-spec"],
    "reactions_count": 10,
    "comments_count": 5,
    "milestone": "v1",
    "priority": "P0",
    "sub_issues": [],
    "comments": [],
}

_ISSUE_PERF = {
    "number": 9003,
    "title": "LSP hover takes 5 seconds on large files",
    "body": "Performance regression in hover provider for files >10K lines.",
    "labels": ["performance", "lsp"],
    "reactions_count": 3,
    "comments_count": 1,
    "milestone": "",
    "priority": "P2",
    "sub_issues": [],
    "comments": [],
}


# ── pass tests ───────────────────────────────────────────────────────


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestCategorizeIssue(unittest.TestCase):
    """Test that categorize_issue returns a valid category."""

    def test_categorize_returns_valid(self):
        result = categorize_issue(_ISSUE_FP)
        self.assertIn("category", result)
        self.assertIn(result["category"], CATEGORIES)
        self.assertIn("subcategory", result)
        self.assertIn("confidence", result)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestScoreIssue(unittest.TestCase):
    """Test that score_issue returns a valid score dict."""

    def test_score_returns_valid(self):
        categorization = {
            "category": "false_positive",
            "subcategory": "overload",
            "confidence": "high",
        }
        primer_impact = {
            "primer_project_count": 3,
            "primer_error_count": 12,
            "matched_kind": "bad-override",
        }
        dep_graph = {
            "dependency_groups": [],
            "blocking_chains": [],
            "duplicate_clusters": [],
        }

        result = score_issue(_ISSUE_FP, categorization, primer_impact, dep_graph)
        self.assertIn("priority_score", result)
        self.assertIsInstance(result["priority_score"], float)
        self.assertGreaterEqual(result["priority_score"], 0)
        self.assertLessEqual(result["priority_score"], 100)
        self.assertIn("rationale", result)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestBuildDependencies(unittest.TestCase):
    """Test that build_dependencies returns a valid dep graph."""

    def test_three_related_issues(self):
        issues = [_ISSUE_FP, _ISSUE_FEATURE, _ISSUE_PERF]
        categorizations = {
            9001: {"category": "false_positive", "subcategory": "overload"},
            9002: {"category": "missing_feature", "subcategory": "paramspec"},
            9003: {"category": "performance", "subcategory": "lsp"},
        }
        relationships = {
            "parent_child": {},
            "duplicates": {},
            "blocked_by": {},
        }

        result = build_dependencies(issues, categorizations, relationships)
        self.assertIn("dependency_groups", result)
        self.assertIn("blocking_chains", result)
        self.assertIn("duplicate_clusters", result)
        self.assertIsInstance(result["dependency_groups"], list)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestRankBatch(unittest.TestCase):
    """Test that rank_issues produces a valid ranking."""

    def test_rank_three_issues(self):
        issues = [_ISSUE_FP, _ISSUE_FEATURE, _ISSUE_PERF]
        scores = {
            9001: {"priority_score": 75, "rationale": "High impact FP"},
            9002: {"priority_score": 85, "rationale": "V1 blocker"},
            9003: {"priority_score": 55, "rationale": "Moderate perf issue"},
        }
        categorizations = {
            9001: {"category": "false_positive", "subcategory": "overload"},
            9002: {"category": "missing_feature", "subcategory": "paramspec"},
            9003: {"category": "performance", "subcategory": "lsp"},
        }
        primer_impacts = {
            9001: {"primer_project_count": 3, "primer_error_count": 12},
            9002: {"primer_project_count": 0, "primer_error_count": 0},
            9003: {"primer_project_count": 1, "primer_error_count": 5},
        }
        dep_graph = {
            "dependency_groups": [],
            "blocking_chains": [],
            "duplicate_clusters": [],
        }

        result = rank_issues(issues, scores, categorizations, primer_impacts, dep_graph)
        self.assertIn("ranked_issues", result)
        self.assertGreater(len(result["ranked_issues"]), 0)
        self.assertIn("priority_tiers", result)
        # Should have a v1_gap_analysis section
        self.assertIn("v1_gap_analysis", result)


if __name__ == "__main__":
    unittest.main()
