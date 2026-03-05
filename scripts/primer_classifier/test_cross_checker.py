# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cross-checker module.

Deterministic tests (mocked LLM):
    python -m pytest scripts/primer_classifier/test_cross_checker.py -v

Live LLM tests (requires CLASSIFIER_API_KEY or ANTHROPIC_API_KEY):
    python -m pytest scripts/primer_classifier/test_cross_checker.py -v -m slow
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from .cross_checker import (
    _filter_to_relevant_files,
    _format_checker_errors,
    _format_pyrefly_errors,
    _match_errors_with_llm,
    _MATCH_SYSTEM_PROMPT,
)
from .parser import ErrorEntry


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_pyrefly_entry(
    file: str, line: int, kind: str, message: str
) -> ErrorEntry:
    return ErrorEntry(
        severity="ERROR",
        file_path=file,
        location=f"{line}:1-10",
        message=message,
        error_kind=kind,
        raw_line=f"ERROR {file}:{line}:1-10: {message} [{kind}]",
    )


def _make_checker_error(
    file: str, line: int, kind: str, message: str
) -> dict[str, object]:
    return {
        "file": file,
        "line": line,
        "col": 1,
        "kind": kind,
        "message": message,
        "severity": "error",
    }


# ---------------------------------------------------------------------------
# Deterministic tests — formatting
# ---------------------------------------------------------------------------


class TestFilterToRelevantFiles:
    def test_filters_to_matching_files(self):
        entries = [
            _make_pyrefly_entry("a.py", 10, "err", "msg"),
            _make_pyrefly_entry("b.py", 20, "err", "msg"),
        ]
        checker_errors = [
            _make_checker_error("a.py", 5, "err", "msg"),
            _make_checker_error("b.py", 15, "err", "msg"),
            _make_checker_error("c.py", 99, "err", "unrelated"),
            _make_checker_error("d.py", 100, "err", "unrelated"),
        ]
        result = _filter_to_relevant_files(checker_errors, entries)
        assert len(result) == 2
        assert all(e["file"] in ("a.py", "b.py") for e in result)

    def test_empty_checker_errors(self):
        entries = [_make_pyrefly_entry("a.py", 10, "err", "msg")]
        result = _filter_to_relevant_files([], entries)
        assert result == []

    def test_no_matching_files(self):
        entries = [_make_pyrefly_entry("a.py", 10, "err", "msg")]
        checker_errors = [_make_checker_error("z.py", 1, "err", "msg")]
        result = _filter_to_relevant_files(checker_errors, entries)
        assert result == []

    def test_all_matching(self):
        entries = [_make_pyrefly_entry("a.py", 10, "err", "msg")]
        checker_errors = [
            _make_checker_error("a.py", 1, "err", "msg1"),
            _make_checker_error("a.py", 50, "err", "msg2"),
        ]
        result = _filter_to_relevant_files(checker_errors, entries)
        assert len(result) == 2


class TestFormatCheckerErrors:
    def test_empty(self):
        result = _format_checker_errors([], "mypy")
        assert result == "No mypy errors."

    def test_single_error(self):
        errors = [_make_checker_error("foo.py", 10, "return-value", "bad return")]
        result = _format_checker_errors(errors, "mypy")
        assert "mypy errors (1 total)" in result
        assert "foo.py:10" in result
        assert "bad return" in result

    def test_all_errors_included(self):
        errors = [
            _make_checker_error("foo.py", i, "err", f"msg {i}") for i in range(10)
        ]
        result = _format_checker_errors(errors, "pyright")
        assert "pyright errors (10 total)" in result
        # All errors should be included (no truncation)
        for i in range(10):
            assert f"msg {i}" in result


class TestFormatPyreflyErrors:
    def test_indexed_output(self):
        entries = [
            _make_pyrefly_entry("a.py", 1, "bad-return", "str is not int"),
            _make_pyrefly_entry("b.py", 2, "missing-attribute", "no attr foo"),
        ]
        result = _format_pyrefly_errors(entries)
        assert "[0]" in result
        assert "[1]" in result
        assert "a.py:1" in result
        assert "b.py:2" in result
        assert "2 total" in result

    def test_offset(self):
        entries = [
            _make_pyrefly_entry("a.py", 1, "bad-return", "msg"),
        ]
        result = _format_pyrefly_errors(entries, offset=5)
        assert "[5]" in result
        assert "[0]" not in result


# ---------------------------------------------------------------------------
# Deterministic tests — matching with mocked LLM
# ---------------------------------------------------------------------------


class TestMatchErrorsMocked:
    """Test _match_errors_with_llm with mocked API responses."""

    def _mock_llm_response(self, matches: list[dict]) -> dict:
        """Build a mock Anthropic API response containing JSON matches."""
        return {
            "content": [{"text": json.dumps(matches)}],
        }

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_all_co_reported(self, mock_api, mock_backend):
        mock_backend.return_value = ("anthropic", "fake-key")
        mock_api.return_value = self._mock_llm_response(
            [
                {"index": 0, "mypy": True, "pyright": True},
                {"index": 1, "mypy": True, "pyright": False},
            ]
        )

        entries = [
            _make_pyrefly_entry("a.py", 10, "bad-return", "str not int"),
            _make_pyrefly_entry("a.py", 20, "bad-assignment", "str not int"),
        ]

        mypy_errors = [_make_checker_error("a.py", 10, "return-value", "bad return")]
        pyright_errors = [_make_checker_error("a.py", 10, "reportReturnType", "bad return")]
        result = _match_errors_with_llm(entries, mypy_errors, pyright_errors)
        assert len(result) == 2
        assert result[0]["mypy"] is True
        assert result[0]["pyright"] is True
        assert result[1]["mypy"] is True
        assert result[1]["pyright"] is False

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_all_pyrefly_only(self, mock_api, mock_backend):
        mock_backend.return_value = ("anthropic", "fake-key")
        mock_api.return_value = self._mock_llm_response(
            [{"index": 0, "mypy": False, "pyright": False}]
        )

        entries = [_make_pyrefly_entry("a.py", 10, "bad-return", "str not int")]
        mypy_errors = [_make_checker_error("other.py", 99, "import", "unrelated")]
        result = _match_errors_with_llm(entries, mypy_errors, [])
        assert len(result) == 1
        assert result[0]["mypy"] is False
        assert result[0]["pyright"] is False

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_filters_checker_errors_to_relevant_files(self, mock_api, mock_backend):
        """Checker errors are filtered to only files with pyrefly errors."""
        mock_backend.return_value = ("anthropic", "fake-key")
        mock_api.return_value = self._mock_llm_response(
            [{"index": 0, "mypy": False, "pyright": False}]
        )

        entries = [_make_pyrefly_entry("a.py", 10, "bad-return", "str not int")]
        # Checker errors: one on same file (a.py), two on different files
        mypy_errors = [
            _make_checker_error("a.py", 5, "import", "relevant"),
            _make_checker_error("other.py", 99, "import", "unrelated"),
            _make_checker_error("third.py", 50, "attr-defined", "no attr"),
        ]
        _match_errors_with_llm(entries, mypy_errors, [])

        # Only the error on a.py should be sent to the LLM
        call_args = mock_api.call_args
        user_prompt = call_args[0][2]  # positional arg: user_prompt
        assert "a.py:5" in user_prompt
        assert "other.py:99" not in user_prompt
        assert "third.py:50" not in user_prompt

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_markdown_wrapped_response(self, mock_api, mock_backend):
        """LLM wraps JSON in markdown fences + explanation text."""
        mock_backend.return_value = ("anthropic", "fake-key")
        text = (
            "Here's my analysis:\n\n```json\n"
            '[{"index": 0, "mypy": true, "pyright": false}]\n'
            "```\n\nThe error is only flagged by mypy."
        )
        mock_api.return_value = {"content": [{"text": text}]}

        entries = [_make_pyrefly_entry("a.py", 10, "bad-return", "str not int")]
        mypy_errors = [_make_checker_error("a.py", 10, "return-value", "bad return")]
        result = _match_errors_with_llm(entries, mypy_errors, [])
        assert len(result) == 1
        assert result[0]["mypy"] is True

    @patch("primer_classifier.cross_checker.get_backend")
    def test_no_api_key(self, mock_backend):
        mock_backend.return_value = ("none", "")
        entries = [_make_pyrefly_entry("a.py", 10, "bad-return", "str not int")]
        result = _match_errors_with_llm(entries, [], [])
        assert result == []

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_llm_api_error(self, mock_api, mock_backend):
        """Gracefully handle API errors."""
        mock_backend.return_value = ("anthropic", "fake-key")
        mock_api.side_effect = Exception("API down")

        entries = [_make_pyrefly_entry("a.py", 10, "bad-return", "str not int")]
        mypy_errors = [_make_checker_error("a.py", 10, "return-value", "bad return")]
        result = _match_errors_with_llm(entries, mypy_errors, [])
        assert result == []

    @patch("primer_classifier.cross_checker.get_backend")
    @patch("primer_classifier.cross_checker.call_anthropic_api")
    def test_batching_large_set(self, mock_api, mock_backend):
        """Large error sets should be batched into multiple LLM calls."""
        mock_backend.return_value = ("anthropic", "fake-key")

        # Create 150 entries (more than _BATCH_SIZE=80)
        entries = [
            _make_pyrefly_entry("a.py", i, "err", f"msg {i}") for i in range(150)
        ]

        # Mock returns matches for whatever batch it receives
        def make_response(*args, **kwargs):
            user_prompt = args[2]
            # Parse out which indices are in this batch
            matches = []
            for i in range(150):
                if f"[{i}]" in user_prompt:
                    matches.append({"index": i, "mypy": True, "pyright": False})
            return {"content": [{"text": json.dumps(matches)}]}

        mock_api.side_effect = make_response

        result = _match_errors_with_llm(entries, [], [])
        assert len(result) == 150
        # Should have made 2 calls (80 + 70)
        assert mock_api.call_count == 2


# ---------------------------------------------------------------------------
# Live LLM tests — require API key
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMatchErrorsLive:
    """Live LLM matching tests — verifies the LLM correctly matches errors.

    Run with: python -m pytest scripts/primer_classifier/test_cross_checker.py -v -m slow
    """

    def test_all_same_errors_co_reported(self):
        """All 3 checkers flag the same obvious errors — should all match."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "test.py", 2, "bad-return",
                "Returned type `str` is not assignable to declared return type `int`"
            ),
            _make_pyrefly_entry(
                "test.py", 7, "bad-assignment",
                "`str` is not assignable to variable `x` with type `int`"
            ),
            _make_pyrefly_entry(
                "test.py", 9, "bad-argument-type",
                "Argument `int` is not assignable to parameter `name` with type `str`"
            ),
            _make_pyrefly_entry(
                "test.py", 15, "missing-attribute",
                "Object of class `Foo` has no attribute `nonexistent_method`"
            ),
        ]

        mypy_errors = [
            _make_checker_error(
                "test.py", 2, "return-value",
                'Incompatible return value type (got "str", expected "int")'
            ),
            _make_checker_error(
                "test.py", 7, "assignment",
                'Incompatible types in assignment (expression has type "str", variable has type "int")'
            ),
            _make_checker_error(
                "test.py", 9, "arg-type",
                'Argument 1 to "greet" has incompatible type "int"; expected "str"'
            ),
            _make_checker_error(
                "test.py", 15, "attr-defined",
                '"Foo" has no attribute "nonexistent_method"'
            ),
        ]

        pyright_errors = [
            _make_checker_error(
                "test.py", 2, "reportReturnType",
                'Type "Literal[\'hello\']" is not assignable to return type "int"'
            ),
            _make_checker_error(
                "test.py", 7, "reportAssignmentType",
                'Type "Literal[\'not an int\']" is not assignable to declared type "int"'
            ),
            _make_checker_error(
                "test.py", 9, "reportArgumentType",
                'Argument of type "Literal[42]" cannot be assigned to parameter "name" of type "str"'
            ),
            _make_checker_error(
                "test.py", 15, "reportAttributeAccessIssue",
                'Cannot access attribute "nonexistent_method" for class "Foo"'
            ),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, mypy_errors, pyright_errors)
        assert len(matches) == 4
        for m in matches:
            assert m.get("mypy") is True, f"Expected mypy=True for index {m.get('index')}"
            assert m.get("pyright") is True, f"Expected pyright=True for index {m.get('index')}"

    def test_pyrefly_only_errors(self):
        """Errors unique to pyrefly — mypy/pyright don't flag them."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "test.py", 5, "inconsistent-overload-default",
                "Default value for parameter `x` is inconsistent across overloads"
            ),
            _make_pyrefly_entry(
                "test.py", 10, "redundant-cast",
                "Redundant cast: `int` is the same type as `int`"
            ),
        ]

        # mypy/pyright have errors, but on completely different files/issues
        mypy_errors = [
            _make_checker_error("other.py", 100, "import", "Cannot find module foo"),
            _make_checker_error("other.py", 200, "syntax", "Syntax error"),
        ]
        pyright_errors = [
            _make_checker_error("other.py", 100, "reportMissingImports", "Import not found"),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, mypy_errors, pyright_errors)
        assert len(matches) == 2
        for m in matches:
            assert m.get("mypy") is not True, f"Expected mypy=False for index {m.get('index')}"
            assert m.get("pyright") is not True, f"Expected pyright=False for index {m.get('index')}"

    def test_mixed_co_reported_and_unique(self):
        """Mix of co-reported and pyrefly-only errors."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "main.py", 10, "bad-return",
                "Returned type `str` is not assignable to declared return type `int`"
            ),
            _make_pyrefly_entry(
                "main.py", 50, "redundant-cast",
                "Redundant cast: `int` is the same type as `int`"
            ),
        ]

        mypy_errors = [
            _make_checker_error(
                "main.py", 10, "return-value",
                'Incompatible return value type (got "str", expected "int")'
            ),
        ]
        pyright_errors = [
            _make_checker_error(
                "main.py", 10, "reportReturnType",
                'Type "str" is not assignable to return type "int"'
            ),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, mypy_errors, pyright_errors)
        assert len(matches) == 2

        m0 = next(m for m in matches if m.get("index") == 0)
        assert m0.get("mypy") is True, "bad-return should be flagged by mypy"
        assert m0.get("pyright") is True, "bad-return should be flagged by pyright"

        m1 = next(m for m in matches if m.get("index") == 1)
        assert m1.get("mypy") is not True, "redundant-cast should not be in mypy"
        assert m1.get("pyright") is not True, "redundant-cast should not be in pyright"

    def test_nearby_line_matching(self):
        """Errors on slightly different lines should still match."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "utils.py", 100, "bad-argument-type",
                "Argument `int` is not assignable to parameter `name` with type `str`"
            ),
        ]

        mypy_errors = [
            _make_checker_error(
                "utils.py", 102, "arg-type",
                'Argument 1 to "process" has incompatible type "int"; expected "str"'
            ),
        ]
        pyright_errors = [
            _make_checker_error(
                "utils.py", 97, "reportArgumentType",
                'Argument of type "int" cannot be assigned to parameter "name" of type "str"'
            ),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, mypy_errors, pyright_errors)
        assert len(matches) == 1
        assert matches[0].get("mypy") is True, "Same error ±2 lines should match"
        assert matches[0].get("pyright") is True, "Same error ±3 lines should match"

    def test_only_mypy_available(self):
        """Only mypy was run (no pyright errors) — pyright should be False."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "test.py", 5, "bad-return",
                "Returned type `str` is not assignable to declared return type `int`"
            ),
        ]

        mypy_errors = [
            _make_checker_error(
                "test.py", 5, "return-value",
                'Incompatible return value type (got "str", expected "int")'
            ),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, mypy_errors, [])
        assert len(matches) == 1
        assert matches[0].get("mypy") is True
        assert matches[0].get("pyright") is not True

    def test_only_pyright_available(self):
        """Only pyright was run (no mypy errors) — mypy should be False."""
        pyrefly_entries = [
            _make_pyrefly_entry(
                "test.py", 5, "missing-attribute",
                "Object of class `Foo` has no attribute `bar`"
            ),
        ]

        pyright_errors = [
            _make_checker_error(
                "test.py", 5, "reportAttributeAccessIssue",
                'Cannot access attribute "bar" for class "Foo"'
            ),
        ]

        matches = _match_errors_with_llm(pyrefly_entries, [], pyright_errors)
        assert len(matches) == 1
        assert matches[0].get("mypy") is not True
        assert matches[0].get("pyright") is True
