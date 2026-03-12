#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for issue_ranker deterministic modules.

Tests cover:
- code_extractor: regex matching, Python detection
- relationship_resolver: regex matching, union-find clusters
- dep_resolver: module extraction from errors, package mapping
- status_classifier: heuristic classification
- primer_impact: fuzzy matching, primer index building
- rank: mechanical tiering, V1 gap analysis
- report_formatter: markdown/JSON output
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code_extractor import _looks_like_python, _parse_code_response, extract_code_blocks
from dep_resolver import _module_to_package, extract_module_from_error
from passes.primer_impact import _build_primer_index, _fuzzy_match_kind
from passes.rank import _mechanical_tier
from relationship_resolver import (
    _BLOCKED_BY_RE,
    _BLOCKS_RE,
    _build_duplicate_clusters,
    _DUPLICATE_RE,
    resolve_relationships,
)
from report_formatter import format_json, format_markdown
from status_classifier import _format_errors, _heuristic_classify


# ──────────────────────────────────────────────────────────────────────
# code_extractor tests
# ──────────────────────────────────────────────────────────────────────
class TestLooksLikePython(unittest.TestCase):
    """Test _looks_like_python heuristic."""

    def test_def_statement(self):
        self.assertTrue(_looks_like_python("def foo():\n    pass"))

    def test_class_statement(self):
        self.assertTrue(_looks_like_python("class Foo:\n    pass"))

    def test_import_statement(self):
        self.assertTrue(_looks_like_python("import os"))

    def test_from_import(self):
        self.assertTrue(_looks_like_python("from typing import List"))

    def test_not_python(self):
        self.assertFalse(_looks_like_python("just some text"))

    def test_error_message(self):
        """Error messages should NOT match as Python."""
        self.assertFalse(_looks_like_python("error: Cannot find module 'foo'"))

    def test_config_text(self):
        self.assertFalse(
            _looks_like_python("[tool.pyright]\nreportMissingImports = false")
        )

    def test_indented_def(self):
        """def must be at start of line."""
        self.assertFalse(_looks_like_python("    some text\n    more text"))

    def test_multiline_with_def(self):
        self.assertTrue(_looks_like_python("# comment\ndef foo():\n    return 1"))


class TestExtractCodeBlocks(unittest.TestCase):
    """Test extract_code_blocks from markdown."""

    def test_python_fence(self):
        body = "Some text\n```python\nx: int = 1\n```\nMore text"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], "x: int = 1")

    def test_py_fence(self):
        body = "```py\ndef foo() -> int:\n    return 1\n```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 1)
        self.assertIn("def foo", blocks[0])

    def test_multiple_python_fences(self):
        body = "```python\na = 1\n```\ntext\n```python\nb = 2\n```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 2)

    def test_bare_fence_with_python_code(self):
        body = "```\nimport os\ndef foo():\n    pass\n```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 1)

    def test_bare_fence_without_python_code(self):
        """Bare fence with non-Python content should return empty."""
        body = "```\nsome random config text\nkey = value\n```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 0)

    def test_no_code_blocks(self):
        body = "This is just text with no code blocks."
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 0)

    def test_empty_body(self):
        self.assertEqual(extract_code_blocks(""), [])

    def test_indented_fence(self):
        body = "  ```python\n  x = 1\n  ```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 1)

    def test_python_fence_takes_priority_over_bare(self):
        """If there are tagged Python fences, bare fences are ignored."""
        body = "```python\ntagged = True\n```\n```\nimport os\n```"
        blocks = extract_code_blocks(body)
        self.assertEqual(len(blocks), 1)
        self.assertIn("tagged", blocks[0])


class TestParseCodeResponse(unittest.TestCase):
    """Test _parse_code_response LLM output parsing."""

    def test_clean_json(self):
        self.assertEqual(_parse_code_response('{"code": "x = 1"}'), "x = 1")

    def test_no_code_json(self):
        self.assertEqual(_parse_code_response('{"code": "NO_CODE"}'), "")

    def test_json_in_fence(self):
        text = '```json\n{"code": "y = 2"}\n```'
        self.assertEqual(_parse_code_response(text), "y = 2")

    def test_python_in_fence(self):
        text = "```python\ndef foo():\n    pass\n```"
        result = _parse_code_response(text)
        self.assertIn("def foo", result)

    def test_empty_code(self):
        self.assertEqual(_parse_code_response('{"code": ""}'), "")

    def test_invalid_input(self):
        self.assertEqual(_parse_code_response("random text"), "")


# ──────────────────────────────────────────────────────────────────────
# relationship_resolver tests
# ──────────────────────────────────────────────────────────────────────
class TestRelationshipRegexes(unittest.TestCase):
    """Test relationship regex patterns."""

    def test_duplicate_of(self):
        m = _DUPLICATE_RE.search("This is a duplicate of #123")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "123")

    def test_duplicates(self):
        m = _DUPLICATE_RE.search("duplicates #456")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "456")

    def test_dupe_of(self):
        m = _DUPLICATE_RE.search("dupe of #789")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "789")

    def test_blocked_by(self):
        m = _BLOCKED_BY_RE.search("blocked by #100")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "100")

    def test_depends_on(self):
        m = _BLOCKED_BY_RE.search("depends on #200")
        self.assertIsNotNone(m)

    def test_waiting_on(self):
        m = _BLOCKED_BY_RE.search("waiting on #300")
        self.assertIsNotNone(m)

    def test_blocks_issue(self):
        m = _BLOCKS_RE.search("This blocks #400")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "400")

    def test_block_issue(self):
        m = _BLOCKS_RE.search("block #500")
        self.assertIsNotNone(m)

    def test_code_block_no_false_positive(self):
        """'code block' should NOT match the blocks regex."""
        m = _BLOCKS_RE.search("Here is a code block #123 for reference")
        self.assertIsNone(m)

    def test_code_blocks_no_false_positive(self):
        """'code blocks' should NOT match."""
        m = _BLOCKS_RE.search("Use code blocks #456 for examples")
        self.assertIsNone(m)

    def test_blocks_at_word_boundary(self):
        """'blocks' at the start of a sentence should match."""
        m = _BLOCKS_RE.search("Blocks #789 from progressing")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "789")

    def test_this_blocks(self):
        m = _BLOCKS_RE.search("This issue blocks #100")
        self.assertIsNotNone(m)

    def test_case_insensitive(self):
        m = _DUPLICATE_RE.search("DUPLICATE OF #999")
        self.assertIsNotNone(m)


class TestResolveRelationships(unittest.TestCase):
    """Test the full resolve_relationships function."""

    def test_finds_duplicates(self):
        issues = [
            {"number": 1, "body": "duplicate of #2"},
            {"number": 2, "body": ""},
        ]
        result = resolve_relationships(issues)
        self.assertIn(1, result["duplicates"])
        self.assertEqual(result["duplicates"][1], [2])

    def test_ignores_unknown_issues(self):
        """References to issues not in the set should be ignored."""
        issues = [
            {"number": 1, "body": "duplicate of #999"},
        ]
        result = resolve_relationships(issues)
        self.assertEqual(result["duplicates"], {})

    def test_parent_child_from_graphql(self):
        issues = [
            {"number": 1, "body": "", "sub_issues": [{"number": 2}]},
            {"number": 2, "body": "", "parent_issues": [{"number": 1}]},
        ]
        result = resolve_relationships(issues)
        self.assertIn(1, result["parent_child"])
        self.assertIn(2, result["parent_child"][1])

    def test_none_body(self):
        """Issues with None body should not crash."""
        issues = [{"number": 1, "body": None}]
        result = resolve_relationships(issues)
        self.assertEqual(result["duplicates"], {})


class TestBuildDuplicateClusters(unittest.TestCase):
    """Test union-find duplicate clustering."""

    def test_single_cluster(self):
        duplicates = {1: [2], 2: [3]}
        clusters = _build_duplicate_clusters(duplicates)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0]), [1, 2, 3])

    def test_two_clusters(self):
        duplicates = {1: [2], 3: [4]}
        clusters = _build_duplicate_clusters(duplicates)
        self.assertEqual(len(clusters), 2)

    def test_empty(self):
        self.assertEqual(_build_duplicate_clusters({}), [])

    def test_transitive(self):
        """If 1→2 and 2→3, all three should be in one cluster."""
        duplicates = {1: [2], 2: [3]}
        clusters = _build_duplicate_clusters(duplicates)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0]), [1, 2, 3])


# ──────────────────────────────────────────────────────────────────────
# dep_resolver tests
# ──────────────────────────────────────────────────────────────────────
class TestExtractModuleFromError(unittest.TestCase):
    """Test extract_module_from_error with error dicts from each checker."""

    def test_pyrefly_missing_import(self):
        """Pyrefly: 'Could not resolve import of "numpy"'."""
        error = {
            "kind": "missing-import",
            "message": 'Could not resolve import of "numpy"',
        }
        self.assertEqual(extract_module_from_error(error), "numpy")

    def test_pyrefly_dotted_import(self):
        """Pyrefly: dotted module should return top-level name."""
        error = {
            "kind": "missing-import",
            "message": 'Could not resolve import of "numpy.typing"',
        }
        self.assertEqual(extract_module_from_error(error), "numpy")

    def test_pyrefly_missing_module(self):
        """Pyrefly: 'Module "foo" has no attribute ...'."""
        error = {
            "kind": "missing-module",
            "message": 'Module "requests" has no attribute "sessions"',
        }
        self.assertEqual(extract_module_from_error(error), "requests")

    def test_pyright_missing_import(self):
        """Pyright: 'Import "pandas" could not be resolved'."""
        error = {
            "kind": "reportMissingImports",
            "message": 'Import "pandas" could not be resolved',
        }
        self.assertEqual(extract_module_from_error(error), "pandas")

    def test_pyright_missing_import_from_source(self):
        """Pyright: 'Import "foo.bar" could not be resolved from source'."""
        error = {
            "kind": "reportMissingModuleSource",
            "message": 'Import "flask.views" could not be resolved from source',
        }
        self.assertEqual(extract_module_from_error(error), "flask")

    def test_mypy_missing_stub(self):
        """Mypy: 'Cannot find implementation or library stub for module named "cv2"'."""
        error = {
            "kind": "import",
            "message": 'Cannot find implementation or library stub for module named "cv2"',
        }
        self.assertEqual(extract_module_from_error(error), "cv2")

    def test_mypy_no_library_stub(self):
        """Mypy: 'No library stub file for module "yaml"'."""
        error = {
            "kind": "import",
            "message": 'No library stub file for module "yaml"',
        }
        self.assertEqual(extract_module_from_error(error), "yaml")

    def test_non_import_error(self):
        """Non-import errors should return None."""
        error = {
            "kind": "bad-return",
            "message": "Incompatible return type",
        }
        self.assertIsNone(extract_module_from_error(error))

    def test_empty_message(self):
        error = {"kind": "missing-import", "message": ""}
        self.assertIsNone(extract_module_from_error(error))


class TestModuleToPackage(unittest.TestCase):
    """Test module-to-pip-package mapping."""

    def test_pil_to_pillow(self):
        self.assertEqual(_module_to_package("PIL"), "pillow")

    def test_cv2_to_opencv(self):
        self.assertEqual(_module_to_package("cv2"), "opencv-python")

    def test_sklearn_to_scikit(self):
        self.assertEqual(_module_to_package("sklearn"), "scikit-learn")

    def test_yaml_to_pyyaml(self):
        self.assertEqual(_module_to_package("yaml"), "pyyaml")

    def test_identity_for_unknown(self):
        self.assertEqual(_module_to_package("torch"), "torch")


# ──────────────────────────────────────────────────────────────────────
# status_classifier tests
# ──────────────────────────────────────────────────────────────────────
class TestHeuristicClassify(unittest.TestCase):
    """Test _heuristic_classify fallback logic."""

    def test_no_errors(self):
        self.assertEqual(_heuristic_classify([], [], []), "already_fixed")

    def test_pyrefly_only(self):
        """Pyrefly errors only → false positive."""
        self.assertEqual(
            _heuristic_classify([{"kind": "bad-return", "message": "msg"}], [], []),
            "false_positive",
        )

    def test_other_only(self):
        """Only pyright/mypy errors → false negative."""
        self.assertEqual(
            _heuristic_classify([], [{"kind": "rule", "message": "msg"}], []),
            "false_negative",
        )

    def test_all_have_errors(self):
        """All checkers have errors → confirmed bug."""
        self.assertEqual(
            _heuristic_classify(
                [{"kind": "a", "message": "msg"}],
                [{"kind": "b", "message": "msg"}],
                [{"kind": "c", "message": "msg"}],
            ),
            "confirmed_bug",
        )

    def test_pyrefly_and_mypy(self):
        """Pyrefly + mypy → confirmed bug."""
        self.assertEqual(
            _heuristic_classify(
                [{"kind": "a", "message": "msg"}],
                [],
                [{"kind": "b", "message": "msg"}],
            ),
            "confirmed_bug",
        )


class TestFormatErrors(unittest.TestCase):
    """Test _format_errors helper."""

    def test_empty(self):
        self.assertEqual(_format_errors([]), "  (no errors)")

    def test_single_error(self):
        result = _format_errors(
            [{"kind": "bad-return", "line": 10, "message": "Bad type"}]
        )
        self.assertIn("L10", result)
        self.assertIn("bad-return", result)

    def test_limit(self):
        errors = [
            {"kind": f"err{i}", "line": i, "message": f"msg{i}"} for i in range(10)
        ]
        result = _format_errors(errors, limit=3)
        self.assertIn("7 more errors", result)


# ──────────────────────────────────────────────────────────────────────
# primer_impact tests
# ──────────────────────────────────────────────────────────────────────
class TestBuildPrimerIndex(unittest.TestCase):
    """Test _build_primer_index."""

    def test_basic_index(self):
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {"kind": "bad-return"},
                            {"kind": "bad-return"},
                            {"kind": "unknown-type"},
                        ]
                    },
                },
                {
                    "name": "proj2",
                    "pyrefly": {"errors": [{"kind": "bad-return"}]},
                },
            ]
        }
        index = _build_primer_index(primer_data)
        self.assertIn("bad-return", index)
        self.assertEqual(index["bad-return"]["total_count"], 3)
        self.assertEqual(len(index["bad-return"]["projects"]), 2)
        self.assertEqual(index["bad-return"]["projects"]["proj1"], 2)
        self.assertEqual(index["bad-return"]["projects"]["proj2"], 1)
        self.assertEqual(index["unknown-type"]["total_count"], 1)

    def test_empty_primer(self):
        self.assertEqual(_build_primer_index({}), {})

    def test_skips_empty_kind(self):
        primer_data = {
            "projects": [{"name": "p", "pyrefly": {"errors": [{"kind": ""}]}}]
        }
        self.assertEqual(_build_primer_index(primer_data), {})


class TestFuzzyMatchKind(unittest.TestCase):
    """Test _fuzzy_match_kind deterministic matching."""

    def test_exact_match(self):
        kinds = ["bad-return", "unknown-type", "bad-override"]
        self.assertEqual(_fuzzy_match_kind("bad-return", kinds), "bad-return")

    def test_normalized_match_hyphen_vs_underscore(self):
        kinds = ["bad-return", "unknown-type"]
        self.assertEqual(_fuzzy_match_kind("bad_return", kinds), "bad-return")

    def test_normalized_match_case(self):
        kinds = ["bad-return", "Unknown-Type"]
        self.assertEqual(_fuzzy_match_kind("BAD-RETURN", kinds), "bad-return")

    def test_substring_match_issue_in_primer(self):
        """Issue kind is a substring of primer kind."""
        kinds = ["report-missing-imports", "report-return-type"]
        self.assertEqual(
            _fuzzy_match_kind("missing-imports", kinds),
            "report-missing-imports",
        )

    def test_substring_match_primer_in_issue(self):
        """Primer kind is a substring of issue kind."""
        kinds = ["override"]
        self.assertEqual(_fuzzy_match_kind("bad-override-error", kinds), "override")

    def test_no_match(self):
        kinds = ["bad-return", "unknown-type"]
        self.assertIsNone(_fuzzy_match_kind("performance-issue", kinds))

    def test_empty_primer_kinds(self):
        self.assertIsNone(_fuzzy_match_kind("bad-return", []))

    def test_similar_but_different_variable_names(self):
        """Same error kind regardless of which variable triggered it."""
        kinds = ["bad-override"]
        # The kind itself is the same — variable names are in the message, not the kind
        self.assertEqual(_fuzzy_match_kind("bad-override", kinds), "bad-override")

    def test_mixed_separator_matching(self):
        """Hyphens and underscores should be treated equivalently."""
        kinds = ["report_missing_imports"]
        self.assertEqual(
            _fuzzy_match_kind("report-missing-imports", kinds),
            "report_missing_imports",
        )


# ──────────────────────────────────────────────────────────────────────
# rank tests (mechanical tiering, V1 gap)
# ──────────────────────────────────────────────────────────────────────
class TestMechanicalTier(unittest.TestCase):
    """Test _mechanical_tier fallback tiering."""

    def test_critical(self):
        issues = [{"number": 1}]
        scores = {1: {"priority_score": 90}}
        result = _mechanical_tier(issues, scores)
        self.assertEqual(result["ranked_issues"][0]["tier"], "critical")
        self.assertIn(1, result["priority_tiers"]["critical"])

    def test_high(self):
        issues = [{"number": 1}]
        scores = {1: {"priority_score": 70}}
        result = _mechanical_tier(issues, scores)
        self.assertEqual(result["ranked_issues"][0]["tier"], "high")

    def test_medium(self):
        issues = [{"number": 1}]
        scores = {1: {"priority_score": 50}}
        result = _mechanical_tier(issues, scores)
        self.assertEqual(result["ranked_issues"][0]["tier"], "medium")

    def test_low(self):
        issues = [{"number": 1}]
        scores = {1: {"priority_score": 30}}
        result = _mechanical_tier(issues, scores)
        self.assertEqual(result["ranked_issues"][0]["tier"], "low")

    def test_boundary_values(self):
        """Test exact boundary values."""
        issues = [{"number": i} for i in range(1, 5)]
        scores = {
            1: {"priority_score": 80},  # critical
            2: {"priority_score": 65},  # high
            3: {"priority_score": 45},  # medium
            4: {"priority_score": 44},  # low
        }
        result = _mechanical_tier(issues, scores)
        tiers = {r["number"]: r["tier"] for r in result["ranked_issues"]}
        self.assertEqual(tiers[1], "critical")
        self.assertEqual(tiers[2], "high")
        self.assertEqual(tiers[3], "medium")
        self.assertEqual(tiers[4], "low")

    def test_default_score(self):
        """Issues without scores should default to 50 (medium)."""
        issues = [{"number": 1}]
        scores = {}
        result = _mechanical_tier(issues, scores)
        self.assertEqual(result["ranked_issues"][0]["tier"], "medium")


# ──────────────────────────────────────────────────────────────────────
# report_formatter tests
# ──────────────────────────────────────────────────────────────────────
class TestFormatMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def _make_results(self):
        return {
            "ranking": {
                "ranked_issues": [
                    {"number": 100, "final_score": 90, "tier": "critical"},
                    {"number": 200, "final_score": 70, "tier": "high"},
                ],
                "priority_tiers": {
                    "critical": [100],
                    "high": [200],
                    "medium": [],
                    "low": [],
                },
                "v1_gap_analysis": {"v1_issue_count": 0},
            },
            "pass_results": {
                "categorizations": {
                    100: {"category": "false_positive", "subcategory": "bad-return"},
                    200: {"category": "missing_feature", "subcategory": "generics"},
                },
                "scores": {
                    100: {"priority_score": 90, "rationale": "High impact"},
                    200: {"priority_score": 70, "rationale": "Needed feature"},
                },
                "primer_impacts": {},
            },
            "timing": {"total": 42},
            "cost_estimate": 2.50,
        }

    def _make_issue_data(self):
        return {
            "issues": [
                {
                    "number": 100,
                    "title": "False positive on valid code",
                    "labels": ["P0"],
                    "url": "https://github.com/facebook/pyrefly/issues/100",
                    "reactions_count": 5,
                    "comments_count": 3,
                },
                {
                    "number": 200,
                    "title": "Support generics",
                    "labels": ["P1"],
                    "url": "https://github.com/facebook/pyrefly/issues/200",
                    "reactions_count": 2,
                    "comments_count": 1,
                },
            ]
        }

    def test_contains_title(self):
        md = format_markdown(self._make_results(), self._make_issue_data())
        self.assertIn("# Pyrefly Issue Priority Ranking", md)

    def test_contains_issue_links(self):
        md = format_markdown(self._make_results(), self._make_issue_data())
        self.assertIn("https://github.com/facebook/pyrefly/issues/100", md)
        self.assertIn("https://github.com/facebook/pyrefly/issues/200", md)

    def test_contains_tiers(self):
        md = format_markdown(self._make_results(), self._make_issue_data())
        self.assertIn("Critical (1 issues)", md)
        self.assertIn("High (1 issues)", md)

    def test_contains_cost(self):
        md = format_markdown(self._make_results(), self._make_issue_data())
        self.assertIn("$2.50", md)


class TestFormatJson(unittest.TestCase):
    """Test JSON output generation."""

    def test_basic_output(self):
        results = {
            "ranking": {
                "ranked_issues": [
                    {"number": 100, "final_score": 90, "tier": "critical"},
                ],
                "priority_tiers": {"critical": [100]},
                "v1_gap_analysis": {},
            },
            "pass_results": {
                "categorizations": {100: {"category": "bug"}},
                "scores": {100: {"priority_score": 90, "breakdown": {}}},
                "primer_impacts": {},
            },
            "timing": {"total": 10},
            "cost_estimate": 1.00,
        }
        issue_data = {
            "issues": [
                {
                    "number": 100,
                    "title": "Bug",
                    "labels": [],
                    "url": "",
                    "reactions_count": 0,
                    "comments_count": 0,
                    "milestone": "",
                }
            ]
        }
        output = format_json(results, issue_data)
        self.assertEqual(len(output["ranked_issues"]), 1)
        self.assertEqual(output["ranked_issues"][0]["number"], 100)
        self.assertIn("timestamp", output)
        self.assertEqual(output["cost_estimate"], 1.00)


if __name__ == "__main__":
    unittest.main()
