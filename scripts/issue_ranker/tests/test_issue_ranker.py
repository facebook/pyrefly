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

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code_extractor import _looks_like_python, _parse_code_response, extract_code_blocks
from dep_resolver import _module_to_package, extract_module_from_error
from llm_transport import LLMError
from passes.primer_impact import (
    _assess_pattern_specificity,
    _build_primer_index,
    _fuzzy_match_kind,
    _templatize,
    compute_primer_impact,
)
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
from v1_analysis import (
    _ensure_labels_exist,
    _get_current_managed_labels,
    _github_api,
    apply_labels,
    format_v1_report,
    generate_reasons,
    run_v1_analysis,
)


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
class TestTemplatize(unittest.TestCase):
    """Test _templatize message generalization."""

    def test_replaces_backtick_identifiers(self):
        msg = "Argument `x` is not assignable to parameter `y` with type `int`"
        self.assertEqual(
            _templatize(msg),
            "Argument `_` is not assignable to parameter `_` with type `_`",
        )

    def test_no_backticks(self):
        msg = "Some plain error message"
        self.assertEqual(_templatize(msg), "Some plain error message")

    def test_empty_string(self):
        self.assertEqual(_templatize(""), "")

    def test_adjacent_backtick_groups(self):
        msg = "`A` and `B`"
        self.assertEqual(_templatize(msg), "`_` and `_`")


class TestBuildPrimerIndex(unittest.TestCase):
    """Test _build_primer_index with (kind, template) tuple keys."""

    def test_basic_index(self):
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {"kind": "bad-return", "message": "Bad return `x`"},
                            {"kind": "bad-return", "message": "Bad return `y`"},
                            {"kind": "unknown-type", "message": "Unknown `T`"},
                        ]
                    },
                },
                {
                    "name": "proj2",
                    "pyrefly": {
                        "errors": [{"kind": "bad-return", "message": "Bad return `z`"}]
                    },
                },
            ]
        }
        index = _build_primer_index(primer_data)
        # All three bad-return errors share the same template
        key = ("bad-return", "Bad return `_`")
        self.assertIn(key, index)
        self.assertEqual(index[key]["total_count"], 3)
        self.assertEqual(len(index[key]["projects"]), 2)
        self.assertEqual(index[key]["projects"]["proj1"], 2)
        self.assertEqual(index[key]["projects"]["proj2"], 1)
        self.assertIn(("unknown-type", "Unknown `_`"), index)

    def test_different_templates_same_kind(self):
        """Different message templates within the same kind get separate entries."""
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {
                                "kind": "bad-argument-type",
                                "message": "Argument `x` is not assignable to parameter `y`",
                            },
                            {
                                "kind": "bad-argument-type",
                                "message": "Expected `int` arguments, got `str`",
                            },
                        ]
                    },
                }
            ]
        }
        index = _build_primer_index(primer_data)
        key1 = (
            "bad-argument-type",
            "Argument `_` is not assignable to parameter `_`",
        )
        key2 = ("bad-argument-type", "Expected `_` arguments, got `_`")
        self.assertIn(key1, index)
        self.assertIn(key2, index)
        self.assertEqual(index[key1]["total_count"], 1)
        self.assertEqual(index[key2]["total_count"], 1)

    def test_empty_primer(self):
        self.assertEqual(_build_primer_index({}), {})

    def test_skips_empty_kind(self):
        primer_data = {
            "projects": [
                {"name": "p", "pyrefly": {"errors": [{"kind": "", "message": "x"}]}}
            ]
        }
        self.assertEqual(_build_primer_index(primer_data), {})

    def test_missing_message_uses_empty_template(self):
        """Errors without a message field get an empty-string template."""
        primer_data = {
            "projects": [{"name": "p", "pyrefly": {"errors": [{"kind": "bad-return"}]}}]
        }
        index = _build_primer_index(primer_data)
        self.assertIn(("bad-return", ""), index)


class TestAssessPatternSpecificity(unittest.TestCase):
    """Test _assess_pattern_specificity LLM call."""

    @patch("passes.primer_impact.call_llm_json")
    def test_returns_specificity(self, mock_llm):
        mock_llm.return_value = {
            "specificity": "high",
            "note": "Bug specifically about ParamSpec forwarding",
        }
        result = _assess_pattern_specificity(
            issue_title="ParamSpec forwarding broken",
            issue_body="When using ParamSpec to forward args...",
            matched_kind="bad-argument-type",
            matched_template="Argument `_` is not assignable to parameter `_`",
            error_count=100,
            project_count=5,
        )
        self.assertEqual(result["specificity"], "high")
        self.assertIn("ParamSpec", result["note"])

    @patch("passes.primer_impact.call_llm_json")
    def test_normalizes_invalid_specificity(self, mock_llm):
        mock_llm.return_value = {"specificity": "banana", "note": "weird"}
        result = _assess_pattern_specificity("title", "body", "kind", "template", 10, 1)
        self.assertEqual(result["specificity"], "medium")

    @patch("passes.primer_impact.call_llm_json")
    def test_handles_llm_failure(self, mock_llm):
        mock_llm.side_effect = LLMError("timeout")
        result = _assess_pattern_specificity("title", "body", "kind", "template", 10, 1)
        self.assertEqual(result["specificity"], "unknown")
        self.assertEqual(result["note"], "")


class TestComputePrimerImpactMatching(unittest.TestCase):
    """Test that compute_primer_impact matches by (kind, template)."""

    @patch("passes.primer_impact._assess_pattern_specificity")
    def test_exact_template_match(self, mock_spec):
        """When checker_results have a matching (kind, template), use precise count."""
        mock_spec.return_value = {"specificity": "high", "note": "specific bug"}
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {
                                "kind": "bad-argument-type",
                                "message": "Argument `x` is not assignable to parameter `y`",
                            },
                        ]
                    },
                },
                {
                    "name": "proj2",
                    "pyrefly": {
                        "errors": [
                            {
                                "kind": "bad-argument-type",
                                "message": "Expected `int` arguments, got `str`",
                            },
                            {
                                "kind": "bad-argument-type",
                                "message": "Expected `int` arguments, got `str`",
                            },
                        ]
                    },
                },
            ]
        }
        issues = [
            {
                "number": 1,
                "title": "Test",
                "checker_results": {
                    "pyrefly": [
                        {
                            "kind": "bad-argument-type",
                            "message": "Argument `foo` is not assignable to parameter `bar`",
                        }
                    ]
                },
            }
        ]
        result = compute_primer_impact(issues, primer_data, {})
        # Should match only the first template (1 error), not all 3
        self.assertEqual(result[1]["primer_error_count"], 1)
        self.assertEqual(result[1]["primer_project_count"], 1)
        self.assertEqual(
            result[1]["matched_template"],
            "Argument `_` is not assignable to parameter `_`",
        )
        self.assertEqual(result[1]["matched_kind"], "bad-argument-type")
        # Specificity should be populated
        self.assertEqual(result[1]["pattern_specificity"], "high")
        mock_spec.assert_called_once()

    @patch("passes.primer_impact._assess_pattern_specificity")
    def test_fallback_to_kind_aggregation(self, mock_spec):
        """When no template matches, fall back to aggregating all patterns for the kind."""
        mock_spec.return_value = {"specificity": "low", "note": "generic pattern"}
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {
                                "kind": "bad-return",
                                "message": "Return type `int` mismatch",
                            },
                        ]
                    },
                },
            ]
        }
        issues = [
            {
                "number": 1,
                "title": "Test",
                "checker_results": {
                    "pyrefly": [
                        {
                            "kind": "bad-return",
                            "message": "Completely different message",
                        }
                    ]
                },
            }
        ]
        result = compute_primer_impact(issues, primer_data, {})
        # No template match, falls back to kind aggregation
        self.assertEqual(result[1]["primer_error_count"], 1)
        self.assertEqual(result[1]["matched_kind"], "bad-return")
        self.assertEqual(result[1]["matched_template"], "(all patterns)")
        self.assertEqual(result[1]["pattern_specificity"], "low")

    def test_no_primer_data(self):
        """Without primer data, all counts should be zero."""
        issues = [{"number": 1, "title": "Test"}]
        result = compute_primer_impact(issues, None, {})
        self.assertEqual(result[1]["primer_error_count"], 0)
        self.assertEqual(result[1]["matched_template"], "")
        self.assertEqual(result[1]["pattern_specificity"], "")

    @patch("passes.primer_impact._assess_pattern_specificity")
    def test_picks_highest_count_template(self, mock_spec):
        """When multiple templates match, pick the one with the highest count."""
        mock_spec.return_value = {"specificity": "medium", "note": ""}
        primer_data = {
            "projects": [
                {
                    "name": "proj1",
                    "pyrefly": {
                        "errors": [
                            {"kind": "err", "message": "Pattern `A`"},
                            {"kind": "err", "message": "Pattern `B`"},
                            {"kind": "err", "message": "Pattern `B`"},
                        ]
                    },
                },
            ]
        }
        issues = [
            {
                "number": 1,
                "title": "Test",
                "checker_results": {
                    "pyrefly": [
                        {"kind": "err", "message": "Pattern `x`"},
                        {"kind": "err", "message": "Pattern `y`"},
                    ]
                },
            }
        ]
        result = compute_primer_impact(issues, primer_data, {})
        # Both checker errors templatize to "Pattern `_`", which matches
        # both primer entries. The "Pattern `_`" template from "Pattern `B`"
        # has count 2, which is higher than "Pattern `A`" with count 1.
        # But they both templatize to the same thing, so they're the same key.
        self.assertEqual(result[1]["primer_error_count"], 3)
        self.assertEqual(result[1]["matched_template"], "Pattern `_`")


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


# ──────────────────────────────────────────────────────────────────────
# v1_analysis tests
# ──────────────────────────────────────────────────────────────────────
class TestFormatV1Report(unittest.TestCase):
    """Test V1 gap analysis report generation."""

    def _make_analysis(self):
        """Build a minimal analysis dict for testing."""
        return {
            "v1_gap": {
                "v1_issue_count": 3,
                "top_n_compared": 3,
                "overlap_count": 1,
                "overlap_percentage": 33.3,
                "overlap_issues": [100],
                "in_v1_not_top_ranked": [200],
                "in_top_ranked_not_v1": [300],
            },
            "ranked_issues": [
                {
                    "number": 100,
                    "title": "Overlap issue",
                    "final_score": 90,
                    "tier": "critical",
                },
                {
                    "number": 200,
                    "title": "Consider removing",
                    "final_score": 40,
                    "tier": "low",
                },
                {
                    "number": 300,
                    "title": "Consider adding",
                    "final_score": 85,
                    "tier": "critical",
                },
            ],
            "reasons": {
                "q2_reasons": {200: "stale, no activity"},
                "q3_reasons": {300: "high primer impact, 50 projects"},
            },
        }

    def test_contains_overview(self):
        md = format_v1_report(self._make_analysis())
        self.assertIn("V1 Gap Analysis with Reasons", md)
        self.assertIn("V1 issues: 3", md)
        self.assertIn("Overlap: 1 (33.3%)", md)

    def test_contains_verified_section(self):
        md = format_v1_report(self._make_analysis())
        self.assertIn("Verified V1 Issues (1)", md)
        self.assertIn("#100", md)
        self.assertIn("Overlap issue", md)

    def test_contains_q2_with_reason(self):
        md = format_v1_report(self._make_analysis())
        self.assertIn("Consider Removing from V1 (1)", md)
        self.assertIn("#200", md)
        self.assertIn("stale, no activity", md)

    def test_contains_q3_with_reason(self):
        md = format_v1_report(self._make_analysis())
        self.assertIn("Consider Adding to V1 (1)", md)
        self.assertIn("#300", md)
        self.assertIn("high primer impact, 50 projects", md)

    def test_empty_gap(self):
        """No V1 issues should produce empty sections."""
        analysis = {
            "v1_gap": {
                "v1_issue_count": 0,
                "top_n_compared": 0,
                "overlap_count": 0,
                "overlap_percentage": 0,
                "overlap_issues": [],
                "in_v1_not_top_ranked": [],
                "in_top_ranked_not_v1": [],
            },
            "ranked_issues": [],
            "reasons": {"q2_reasons": {}, "q3_reasons": {}},
        }
        md = format_v1_report(analysis)
        self.assertIn("Verified V1 Issues (0)", md)
        self.assertIn("Consider Removing from V1 (0)", md)
        self.assertIn("Consider Adding to V1 (0)", md)

    def test_missing_reasons(self):
        """Issues without LLM reasons should still appear (empty reason column)."""
        analysis = self._make_analysis()
        analysis["reasons"] = {"q2_reasons": {}, "q3_reasons": {}}
        md = format_v1_report(analysis)
        # Issue still appears even without a reason
        self.assertIn("#200", md)
        self.assertIn("#300", md)

    def test_sorted_by_score_descending(self):
        """Issues within each section should be sorted by score (highest first)."""
        analysis = self._make_analysis()
        analysis["v1_gap"]["in_v1_not_top_ranked"] = [200, 201]
        analysis["ranked_issues"].append(
            {
                "number": 201,
                "title": "Higher scored removal",
                "final_score": 60,
                "tier": "medium",
            }
        )
        analysis["reasons"]["q2_reasons"][201] = "borderline"
        md = format_v1_report(analysis)
        # #201 (score 60) should appear before #200 (score 40)
        pos_201 = md.index("#201")
        pos_200 = md.index("#200")
        self.assertLess(pos_201, pos_200)


class TestGithubApi(unittest.TestCase):
    """Test _github_api HTTP wrapper."""

    @patch("v1_analysis.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id": 1}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _github_api("GET", "https://api.github.com/test", "tok123")
        self.assertEqual(result, {"id": 1})

    @patch("v1_analysis.urllib.request.urlopen")
    def test_returns_none_on_204(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _github_api("DELETE", "https://api.github.com/test", "tok123")
        self.assertIsNone(result)

    @patch("v1_analysis.urllib.request.urlopen")
    def test_returns_none_on_404(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, io.BytesIO(b"")
        )
        result = _github_api("GET", "https://api.github.com/test", "tok123")
        self.assertIsNone(result)

    @patch("v1_analysis.urllib.request.urlopen")
    def test_raises_on_500(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "Server Error", {}, io.BytesIO(b"")
        )
        with self.assertRaises(urllib.error.HTTPError):
            _github_api("GET", "https://api.github.com/test", "tok123")

    @patch("v1_analysis.urllib.request.urlopen")
    def test_sends_body_as_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _github_api("POST", "https://api.github.com/test", "tok", {"key": "val"})
        # Verify the request was made with JSON body
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.data, b'{"key": "val"}')
        self.assertEqual(req.get_header("Content-type"), "application/json")


class TestEnsureLabelsExist(unittest.TestCase):
    """Test _ensure_labels_exist creates missing labels."""

    @patch("v1_analysis._github_api")
    def test_creates_missing_labels(self, mock_api):
        # All GET requests return None (label doesn't exist)
        mock_api.return_value = None
        _ensure_labels_exist("facebook/pyrefly", "tok")

        # 3 GETs (one per label) + 3 POSTs (create each)
        self.assertEqual(mock_api.call_count, 6)
        post_calls = [c for c in mock_api.call_args_list if c[0][0] == "POST"]
        self.assertEqual(len(post_calls), 3)

    @patch("v1_analysis._github_api")
    def test_skips_existing_labels(self, mock_api):
        # All GET requests return a label (already exists)
        mock_api.return_value = {"name": "exists"}
        _ensure_labels_exist("facebook/pyrefly", "tok")

        # 3 GETs, 0 POSTs
        self.assertEqual(mock_api.call_count, 3)
        post_calls = [c for c in mock_api.call_args_list if c[0][0] == "POST"]
        self.assertEqual(len(post_calls), 0)


class TestGetCurrentManagedLabels(unittest.TestCase):
    """Test _get_current_managed_labels fetches current label state."""

    @patch("v1_analysis._github_api")
    def test_fetches_labels_from_issues(self, mock_api):
        def side_effect(method, url, token, body=None):
            if method == "GET" and "v1-verified" in url and "page=1" in url:
                return [{"number": 10}, {"number": 20}]
            if method == "GET":
                return []
            return None

        mock_api.side_effect = side_effect
        result = _get_current_managed_labels("facebook/pyrefly", "tok")

        self.assertEqual(result[10], {"v1-verified"})
        self.assertEqual(result[20], {"v1-verified"})

    @patch("v1_analysis._github_api")
    def test_no_issues(self, mock_api):
        mock_api.return_value = []
        result = _get_current_managed_labels("facebook/pyrefly", "tok")
        self.assertEqual(result, {})

    @patch("v1_analysis._github_api")
    def test_paginates(self, mock_api):
        """Handles multiple pages of labeled issues."""
        page1 = [{"number": i} for i in range(100)]
        page2 = [{"number": 100}]

        verified_gets = {"count": 0}

        def side_effect(method, url, token, body=None):
            if method != "GET":
                return None
            if "v1-verified" not in url:
                return []
            verified_gets["count"] += 1
            if verified_gets["count"] == 1:
                return page1
            if verified_gets["count"] == 2:
                return page2
            return []

        mock_api.side_effect = side_effect
        result = _get_current_managed_labels("facebook/pyrefly", "tok")
        # 100 from page 1 + 1 from page 2 = 101 issues
        self.assertEqual(len(result), 101)


class TestApplyLabels(unittest.TestCase):
    """Test apply_labels diff-based orchestration."""

    @patch("v1_analysis._get_current_managed_labels", return_value={})
    @patch("v1_analysis._ensure_labels_exist")
    @patch("v1_analysis._github_api")
    def test_adds_labels_when_none_exist(self, mock_api, mock_ensure, mock_get):
        analysis = {
            "overlap_issues": [1, 2],
            "in_v1_not_top_ranked": [3],
            "in_top_ranked_not_v1": [4, 5],
        }
        result = apply_labels("facebook/pyrefly", analysis, "tok")

        self.assertEqual(result["added"], 5)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(result["unchanged"], 0)

    @patch("v1_analysis._get_current_managed_labels")
    @patch("v1_analysis._ensure_labels_exist")
    @patch("v1_analysis._github_api")
    def test_skips_already_correct_labels(self, mock_api, mock_ensure, mock_get):
        # Issue 1 already has v1-verified — should not be touched
        mock_get.return_value = {1: {"v1-verified"}}
        analysis = {
            "overlap_issues": [1],
            "in_v1_not_top_ranked": [],
            "in_top_ranked_not_v1": [],
        }
        result = apply_labels("facebook/pyrefly", analysis, "tok")

        self.assertEqual(result["added"], 0)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(result["unchanged"], 1)
        # No API calls for label changes
        self.assertEqual(mock_api.call_count, 0)

    @patch("v1_analysis._get_current_managed_labels")
    @patch("v1_analysis._ensure_labels_exist")
    @patch("v1_analysis._github_api")
    def test_removes_stale_and_adds_new(self, mock_api, mock_ensure, mock_get):
        # Issue 1 had v1-verified but should now have v1-consider-removing
        mock_get.return_value = {1: {"v1-verified"}}
        analysis = {
            "overlap_issues": [],
            "in_v1_not_top_ranked": [1],
            "in_top_ranked_not_v1": [],
        }
        result = apply_labels("facebook/pyrefly", analysis, "tok")

        self.assertEqual(result["added"], 1)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(result["unchanged"], 0)


class TestGenerateReasons(unittest.TestCase):
    """Test generate_reasons LLM integration."""

    @patch("v1_analysis.call_llm_json")
    def test_calls_haiku_and_normalizes_keys(self, mock_llm):
        mock_llm.return_value = {
            "q2_reasons": {"200": "stale issue"},
            "q3_reasons": {"300": "high impact"},
        }
        ranked = [
            {"number": 200, "title": "A", "final_score": 40, "tier": "low"},
            {"number": 300, "title": "B", "final_score": 85, "tier": "critical"},
        ]
        v1_gap = {
            "in_v1_not_top_ranked": [200],
            "in_top_ranked_not_v1": [300],
        }
        result = generate_reasons(ranked, v1_gap)

        # Keys normalized from str to int
        self.assertIn(200, result["q2_reasons"])
        self.assertIn(300, result["q3_reasons"])
        self.assertEqual(result["q2_reasons"][200], "stale issue")
        # Verify Haiku model used
        call_kwargs = mock_llm.call_args
        self.assertEqual(call_kwargs[1]["model"], "claude-haiku-4-5-20251001")

    @patch("v1_analysis.call_llm_json")
    def test_empty_issues_skips_llm(self, mock_llm):
        result = generate_reasons(
            [], {"in_v1_not_top_ranked": [], "in_top_ranked_not_v1": []}
        )
        mock_llm.assert_not_called()
        self.assertEqual(result, {"q2_reasons": {}, "q3_reasons": {}})

    @patch("v1_analysis.call_llm_json")
    def test_skips_unknown_issue_numbers(self, mock_llm):
        """Issue numbers in v1_gap not found in ranked_issues are skipped."""
        ranked = [{"number": 100, "title": "A", "final_score": 50, "tier": "medium"}]
        v1_gap = {
            "in_v1_not_top_ranked": [999],  # not in ranked_issues
            "in_top_ranked_not_v1": [],
        }
        result = generate_reasons(ranked, v1_gap)
        # 999 not in ranked_issues, so q2 list is empty → early return, no LLM call
        mock_llm.assert_not_called()
        self.assertEqual(result, {"q2_reasons": {}, "q3_reasons": {}})


class TestRunV1Analysis(unittest.TestCase):
    """Test run_v1_analysis end-to-end orchestration."""

    def _make_ranking_data(self):
        return {
            "ranked_issues": [
                {"number": 1, "title": "A", "final_score": 90, "tier": "critical"},
                {"number": 2, "title": "B", "final_score": 40, "tier": "low"},
                {"number": 3, "title": "C", "final_score": 85, "tier": "critical"},
            ],
            "v1_gap_analysis": {
                "v1_issue_count": 2,
                "top_n_compared": 2,
                "overlap_count": 1,
                "overlap_percentage": 50.0,
                "overlap_issues": [1],
                "in_v1_not_top_ranked": [2],
                "in_top_ranked_not_v1": [3],
            },
        }

    @patch("v1_analysis.apply_labels")
    @patch("v1_analysis.generate_reasons")
    def test_skips_labels_without_token(self, mock_reasons, mock_labels):
        mock_reasons.return_value = {"q2_reasons": {}, "q3_reasons": {}}
        import tempfile

        ranking_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self._make_ranking_data(), ranking_file)
        ranking_file.close()

        try:
            result = run_v1_analysis(
                ranking_json=ranking_file.name,
                output_md="",
                repo="facebook/pyrefly",
                github_token="",
                apply=True,
            )
            mock_labels.assert_not_called()
            self.assertEqual(result["labels_applied"], {})
        finally:
            os.unlink(ranking_file.name)

    @patch(
        "v1_analysis.apply_labels",
        return_value={"added": 1, "removed": 0, "unchanged": 0},
    )
    @patch("v1_analysis.generate_reasons")
    def test_applies_labels_with_token(self, mock_reasons, mock_labels):
        mock_reasons.return_value = {"q2_reasons": {}, "q3_reasons": {}}
        import tempfile

        ranking_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self._make_ranking_data(), ranking_file)
        ranking_file.close()

        try:
            result = run_v1_analysis(
                ranking_json=ranking_file.name,
                output_md="",
                repo="facebook/pyrefly",
                github_token="ghp_test",
                apply=True,
            )
            mock_labels.assert_called_once()
            self.assertEqual(
                result["labels_applied"],
                {"added": 1, "removed": 0, "unchanged": 0},
            )
        finally:
            os.unlink(ranking_file.name)

    @patch("v1_analysis.generate_reasons")
    def test_early_return_no_v1_data(self, mock_reasons):
        import tempfile

        data = {"ranked_issues": [], "v1_gap_analysis": {"v1_issue_count": 0}}
        ranking_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(data, ranking_file)
        ranking_file.close()

        try:
            result = run_v1_analysis(
                ranking_json=ranking_file.name,
                output_md="",
                repo="facebook/pyrefly",
                github_token="",
                apply=False,
            )
            mock_reasons.assert_not_called()
            self.assertEqual(result["reasons"], {})
        finally:
            os.unlink(ranking_file.name)

    @patch("v1_analysis.generate_reasons")
    def test_writes_report_to_file(self, mock_reasons):
        mock_reasons.return_value = {"q2_reasons": {}, "q3_reasons": {}}
        import tempfile

        ranking_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self._make_ranking_data(), ranking_file)
        ranking_file.close()

        output_file = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        output_file.close()

        try:
            run_v1_analysis(
                ranking_json=ranking_file.name,
                output_md=output_file.name,
                repo="facebook/pyrefly",
                github_token="",
                apply=False,
            )
            with open(output_file.name) as f:
                content = f.read()
            self.assertIn("V1 Gap Analysis with Reasons", content)
        finally:
            os.unlink(ranking_file.name)
            os.unlink(output_file.name)


if __name__ == "__main__":
    unittest.main()
