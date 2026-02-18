# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the primer classifier.

Run with: python -m pytest scripts/primer_classifier/test_classifier.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from .classifier import (
    Classification,
    _categorize_errors,
    _extract_class_name,
    _format_errors_for_llm,
    _is_all_internal_errors,
    _is_all_removals,
    _is_wording_change,
    classify_all,
    classify_project,
)
from .code_fetcher import _extract_referenced_modules, _github_url_to_owner_repo
from .formatter import format_json, format_markdown
from .llm_client import (
    CategoryVerdict,
    _extract_text_from_response,
    _get_backend,
    _parse_classification,
)
from .parser import ErrorEntry, ProjectDiff, parse_error_line, parse_primer_diff

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseErrorLine:
    def test_concise_format(self):
        line = "ERROR src/utils.py:10:5-20: some message [bad-argument]"
        entry = parse_error_line(line)
        assert entry is not None
        assert entry.severity == "ERROR"
        assert entry.file_path == "src/utils.py"
        assert entry.location == "10:5-20"
        assert entry.message == "some message"
        assert entry.error_kind == "bad-argument"

    def test_concise_format_multiline_location(self):
        line = "ERROR src/foo.py:2:5-4:10: multi line error [bad-return]"
        entry = parse_error_line(line)
        assert entry is not None
        assert entry.location == "2:5-4:10"
        assert entry.line_number == 2

    def test_github_actions_format(self):
        line = "::error file=src/main.py,line=10,col=5,endLine=10,endColumn=20,title=Pyrefly bad-return::Returned `int` but expected `str`"
        entry = parse_error_line(line)
        assert entry is not None
        assert entry.severity == "ERROR"
        assert entry.file_path == "src/main.py"
        assert entry.location == "10:5-20"
        assert entry.error_kind == "bad-return"
        assert entry.message == "Returned `int` but expected `str`"

    def test_github_actions_multiline_span(self):
        line = "::error file=a.py,line=1,col=1,endLine=5,endColumn=10,title=Pyrefly bad-return::msg"
        entry = parse_error_line(line)
        assert entry is not None
        assert entry.location == "1:1-5:10"

    def test_warning_format(self):
        line = " WARN src/foo.py:5:1: some warning [some-warn]"
        entry = parse_error_line(line)
        assert entry is not None
        assert entry.severity == "WARN"

    def test_non_matching_line(self):
        assert parse_error_line("this is not an error line") is None
        assert parse_error_line("") is None


class TestParsePrimerDiff:
    def test_empty(self):
        assert parse_primer_diff("") == []
        assert parse_primer_diff("   \n  \n  ") == []

    def test_all_removals(self):
        projects = parse_primer_diff(_load_fixture("all_removals.txt"))
        assert len(projects) == 1
        assert projects[0].name == "myproject"
        assert projects[0].url == "https://github.com/example/myproject"
        assert len(projects[0].added) == 0
        assert len(projects[0].removed) == 2

    def test_multi_project(self):
        projects = parse_primer_diff(_load_fixture("multi_project.txt"))
        assert len(projects) == 4
        names = [p.name for p in projects]
        assert names == ["project_a", "project_b", "project_c", "project_d"]

    def test_github_actions_format(self):
        projects = parse_primer_diff(_load_fixture("github_actions_format.txt"))
        assert len(projects) == 1
        assert len(projects[0].added) == 1
        assert len(projects[0].removed) == 1
        assert projects[0].added[0].error_kind == "bad-return"

    def test_deduplication(self):
        """Same error in both concise and ::error format should be deduped."""
        text = """dupproject (https://github.com/example/dupproject)
+ ERROR src/main.py:10:5-20: Returned `int` but expected `str` [bad-return]
+ ::error file=src/main.py,line=10,col=5,endLine=10,endColumn=20,title=Pyrefly bad-return::Returned `int` but expected `str`
"""
        projects = parse_primer_diff(text)
        assert len(projects) == 1
        assert len(projects[0].added) == 1  # deduped to 1


# ---------------------------------------------------------------------------
# Classifier heuristic tests
# ---------------------------------------------------------------------------


class TestHeuristics:
    def _make_entry(self, kind: str = "bad-return", msg: str = "msg") -> ErrorEntry:
        return ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message=msg,
            error_kind=kind,
            raw_line=f"ERROR src/foo.py:10:5-20: {msg} [{kind}]",
        )

    def test_all_removals(self):
        p = ProjectDiff(name="test", removed=[self._make_entry()])
        assert _is_all_removals(p) is True
        assert _is_all_internal_errors(p) is False

    def test_all_removals_false_when_has_additions(self):
        p = ProjectDiff(
            name="test",
            added=[self._make_entry()],
            removed=[self._make_entry()],
        )
        assert _is_all_removals(p) is False

    def test_all_internal_errors(self):
        e = self._make_entry(kind="internal-error", msg="panicked")
        p = ProjectDiff(name="test", added=[e])
        assert _is_all_internal_errors(p) is True

    def test_not_all_internal_errors_when_mixed(self):
        e1 = self._make_entry(kind="internal-error", msg="panicked")
        e2 = self._make_entry(kind="bad-return", msg="wrong return")
        p = ProjectDiff(name="test", added=[e1, e2])
        assert _is_all_internal_errors(p) is False

    def test_wording_change(self):
        added = self._make_entry(msg="new wording")
        removed = ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message="old wording",
            error_kind="bad-return",
            raw_line="ERROR src/foo.py:10:5-20: old wording [bad-return]",
        )
        p = ProjectDiff(name="test", added=[added], removed=[removed])
        assert _is_wording_change(p) is True

    def test_not_wording_change_different_kinds(self):
        added = self._make_entry(kind="bad-return")
        removed = self._make_entry(kind="bad-argument")
        p = ProjectDiff(name="test", added=[added], removed=[removed])
        assert _is_wording_change(p) is False

    def test_not_wording_change_different_counts(self):
        p = ProjectDiff(
            name="test",
            added=[self._make_entry(), self._make_entry()],
            removed=[self._make_entry()],
        )
        assert _is_wording_change(p) is False


class TestClassifyProject:
    def _make_entry(self, kind: str = "bad-return", msg: str = "msg") -> ErrorEntry:
        return ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message=msg,
            error_kind=kind,
            raw_line=f"ERROR src/foo.py:10:5-20: {msg} [{kind}]",
        )

    def test_all_removals_classified_as_improvement(self):
        p = ProjectDiff(name="test", removed=[self._make_entry()])
        result = classify_project(p, fetch_code=False, use_llm=False)
        assert result.verdict == "improvement"
        assert result.method == "heuristic"

    def test_internal_errors_classified_as_regression(self):
        e = self._make_entry(kind="internal-error")
        p = ProjectDiff(name="test", added=[e])
        result = classify_project(p, fetch_code=False, use_llm=False)
        assert result.verdict == "regression"
        assert result.method == "heuristic"

    def test_wording_change_classified_as_neutral(self):
        added = self._make_entry(msg="new msg")
        removed = ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message="old msg",
            error_kind="bad-return",
            raw_line="ERROR src/foo.py:10:5-20: old msg [bad-return]",
        )
        p = ProjectDiff(name="test", added=[added], removed=[removed])
        result = classify_project(p, fetch_code=False, use_llm=False)
        assert result.verdict == "neutral"
        assert result.method == "heuristic"

    def test_non_trivial_without_llm_is_ambiguous(self):
        p = ProjectDiff(name="test", added=[self._make_entry()])
        result = classify_project(p, fetch_code=False, use_llm=False)
        assert result.verdict == "ambiguous"
        assert result.method == "heuristic"


class TestClassifyAll:
    def _make_entry(self, kind: str = "bad-return") -> ErrorEntry:
        return ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message="msg",
            error_kind=kind,
            raw_line=f"ERROR src/foo.py:10:5-20: msg [{kind}]",
        )

    def test_counts(self):
        projects = [
            ProjectDiff(name="a", removed=[self._make_entry()]),  # improvement
            ProjectDiff(name="b", added=[self._make_entry(kind="internal-error")]),  # regression
            ProjectDiff(name="c", added=[self._make_entry()]),  # ambiguous (no LLM)
        ]
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 3
        assert result.improvements == 1
        assert result.regressions == 1
        assert result.ambiguous == 1


class TestClassifyFromFixtures:
    def test_all_removals_fixture(self):
        projects = parse_primer_diff(_load_fixture("all_removals.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.improvements == 1
        assert result.regressions == 0

    def test_internal_errors_fixture(self):
        projects = parse_primer_diff(_load_fixture("internal_errors.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.regressions == 1
        assert result.improvements == 0

    def test_wording_changes_fixture(self):
        projects = parse_primer_diff(_load_fixture("wording_changes.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.neutrals == 1

    def test_multi_project_fixture(self):
        projects = parse_primer_diff(_load_fixture("multi_project.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 4
        # project_a: all removals -> improvement
        # project_b: internal-error -> regression
        # project_c: wording change -> neutral
        # project_d: non-trivial -> ambiguous
        assert result.improvements == 1
        assert result.regressions == 1
        assert result.neutrals == 1
        assert result.ambiguous == 1

    def test_empty_fixture(self):
        projects = parse_primer_diff(_load_fixture("empty.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 0

    def test_mixed_changes_fixture(self):
        projects = parse_primer_diff(_load_fixture("mixed_changes.txt"))
        assert len(projects) == 1
        assert projects[0].name == "mixedproject"
        assert len(projects[0].added) == 2
        assert len(projects[0].removed) == 1
        # Non-trivial mixed change without LLM → ambiguous
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.ambiguous == 1


class TestRealFixtureParsing:
    """Verify that real primer diffs from actual PRs parse without errors."""

    REAL_DIR = FIXTURES_DIR / "real"

    def test_parse_all_real_fixtures(self):
        if not self.REAL_DIR.exists():
            pytest.skip("No real fixtures directory")
        for fixture in sorted(self.REAL_DIR.glob("*.txt")):
            projects = parse_primer_diff(fixture.read_text())
            assert len(projects) > 0, f"{fixture.name} should parse into at least one project"
            for p in projects:
                assert p.name, f"Project in {fixture.name} has no name"
                assert p.added or p.removed, f"Project {p.name} in {fixture.name} has no changes"


# ---------------------------------------------------------------------------
# Categorization tests
# ---------------------------------------------------------------------------


class TestCategorization:
    def test_extract_class_name(self):
        assert _extract_class_name("Object of class `Foo` has no attribute `bar`") == "Foo"
        assert _extract_class_name("no class here") is None

    def test_format_below_threshold_is_raw(self):
        """Below _CATEGORY_THRESHOLD, errors are listed individually."""
        entries = [
            ErrorEntry("ERROR", "a.py", "1:1", "msg", "bad-return", "ERROR a.py:1:1: msg [bad-return]"),
        ]
        p = ProjectDiff(name="test", added=entries)
        text = _format_errors_for_llm(p)
        assert "+ ERROR a.py:1:1: msg [bad-return]" in text
        assert "Error summary" not in text

    def test_format_above_threshold_uses_categories(self):
        """Above _CATEGORY_THRESHOLD, errors are grouped into categories."""
        entries = [
            ErrorEntry("ERROR", f"f{i}.py", f"{i}:1", f"Object of class `X` has no attribute `a{i}`", "missing-attribute", f"raw{i}")
            for i in range(10)
        ]
        p = ProjectDiff(name="test", added=entries)
        text = _format_errors_for_llm(p)
        assert "Error summary" in text
        assert "[missing-attribute]" in text


# ---------------------------------------------------------------------------
# LLM client tests (no actual API calls)
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_llama_preferred(self):
        with patch.dict(os.environ, {"LLAMA_API_KEY": "key1", "ANTHROPIC_API_KEY": "key2"}):
            backend, key = _get_backend()
            assert backend == "llama"
            assert key == "key1"

    def test_classifier_key_over_anthropic(self):
        env = {"CLASSIFIER_API_KEY": "ckey", "ANTHROPIC_API_KEY": "akey"}
        with patch.dict(os.environ, env, clear=True):
            backend, key = _get_backend()
            assert backend == "anthropic"
            assert key == "ckey"

    def test_anthropic_fallback(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "akey"}, clear=True):
            backend, key = _get_backend()
            assert backend == "anthropic"
            assert key == "akey"

    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            backend, _key = _get_backend()
            assert backend == "none"


class TestExtractTextFromResponse:
    def test_llama_format(self):
        resp = {"completion_message": {"content": {"text": "hello"}}}
        assert _extract_text_from_response("llama", resp) == "hello"

    def test_anthropic_format(self):
        resp = {"content": [{"text": "hello"}]}
        assert _extract_text_from_response("anthropic", resp) == "hello"


class TestParseClassification:
    def test_clean_json(self):
        text = '{"verdict": "regression", "reason": "test"}'
        result = _parse_classification(text)
        assert result["verdict"] == "regression"

    def test_markdown_fenced_json(self):
        text = '```json\n{"verdict": "improvement", "reason": "ok"}\n```'
        result = _parse_classification(text)
        assert result["verdict"] == "improvement"

    def test_json_embedded_in_text(self):
        text = 'Here is my analysis:\n{"verdict": "neutral", "reason": "wording"}\nDone.'
        result = _parse_classification(text)
        assert result["verdict"] == "neutral"

    def test_nested_json_with_categories(self):
        obj = {
            "verdict": "regression",
            "reason": "overall bad",
            "categories": [
                {"category": "missing-attr", "verdict": "regression", "reason": "false positives"}
            ],
        }
        text = json.dumps(obj)
        result = _parse_classification(text)
        assert result["verdict"] == "regression"
        assert len(result["categories"]) == 1

    def test_needs_files_response(self):
        text = '{"needs_files": ["foo/bar.py", "baz/qux.py"]}'
        result = _parse_classification(text)
        assert result["needs_files"] == ["foo/bar.py", "baz/qux.py"]

    def test_garbage_raises(self):
        from .llm_client import LLMError

        with pytest.raises(LLMError):
            _parse_classification("this is not json at all")


# ---------------------------------------------------------------------------
# Code fetcher tests
# ---------------------------------------------------------------------------


class TestGitHubUrlParsing:
    def test_valid_url(self):
        result = _github_url_to_owner_repo("https://github.com/facebook/pyrefly")
        assert result == ("facebook", "pyrefly")

    def test_url_with_git_suffix(self):
        result = _github_url_to_owner_repo("https://github.com/facebook/pyrefly.git")
        assert result == ("facebook", "pyrefly")

    def test_non_github_url(self):
        assert _github_url_to_owner_repo("https://gitlab.com/user/repo") is None

    def test_invalid_url(self):
        assert _github_url_to_owner_repo("not a url") is None


class TestExtractReferencedModules:
    def test_dotted_reference(self):
        entry = ErrorEntry(
            severity="ERROR",
            file_path="src/child.py",
            location="10:1",
            message="overrides parent class `base.module.ParentClass`",
            error_kind="bad-override",
            raw_line="raw",
        )
        paths = _extract_referenced_modules(entry)
        # The function converts dotted paths to file paths, trying longest prefix first
        assert any("base/module" in p for p in paths)

    def test_no_references(self):
        entry = ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="1:1",
            message="simple error with no references",
            error_kind="bad-return",
            raw_line="raw",
        )
        assert _extract_referenced_modules(entry) == []


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def _make_result(self):
        from .classifier import ClassificationResult

        return ClassificationResult(
            total_projects=2,
            regressions=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj_a",
                    verdict="regression",
                    reason="false positive",
                    added_count=3,
                    removed_count=0,
                    method="llm",
                ),
                Classification(
                    project_name="proj_b",
                    verdict="improvement",
                    reason="removed false positives",
                    added_count=0,
                    removed_count=5,
                    method="heuristic",
                ),
            ],
        )

    def test_contains_project_names(self):
        md = format_markdown(self._make_result())
        assert "proj_a" in md
        assert "proj_b" in md

    def test_contains_verdict_sections(self):
        md = format_markdown(self._make_result())
        assert "Regression" in md
        assert "Improvement" in md

    def test_empty_result(self):
        from .classifier import ClassificationResult

        md = format_markdown(ClassificationResult())
        assert "No diffs" in md or "All clear" in md

    def test_with_categories(self):
        from .classifier import ClassificationResult

        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="overall",
                    added_count=10,
                    method="llm",
                    categories=[
                        CategoryVerdict("missing-attr", "regression", "false positives"),
                        CategoryVerdict("bad-return", "improvement", "real bugs"),
                    ],
                ),
            ],
        )
        md = format_markdown(result)
        assert "missing-attr" in md
        assert "bad-return" in md


class TestFormatJson:
    def test_valid_json(self):
        from .classifier import ClassificationResult

        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="good",
                    method="heuristic",
                ),
            ],
        )
        output = format_json(result)
        data = json.loads(output)
        assert data["summary"]["total_projects"] == 1
        assert len(data["classifications"]) == 1
        assert data["classifications"][0]["verdict"] == "improvement"
