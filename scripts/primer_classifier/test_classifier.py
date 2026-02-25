# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic unit tests for the primer classifier (no real API calls).

Run with: python -m pytest scripts/primer_classifier/test_classifier.py -v

For live LLM tests, see test_llm_live.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from .classifier import (
    _extract_class_name,
    _format_errors_for_llm,
    _is_all_internal_errors,
    _is_wording_change,
    _truncate_source_context,
    Classification,
    ClassificationResult,
    Suggestion,
    SuggestionResult,
    classify_all,
    classify_project,
)
from .code_fetcher import _extract_referenced_modules, _github_url_to_owner_repo
from .formatter import format_json, format_markdown
from .llm_client import (
    _build_suggestion_user_prompt,
    _build_user_prompt,
    _build_verdict_prompt,
    _build_verdict_system_prompt,
    _extract_text_from_response,
    _get_backend,
    _parse_classification,
    assign_verdict_with_llm,
    CategoryVerdict,
    generate_suggestions,
    LLMResponse,
)
from .parser import ErrorEntry, parse_error_line, parse_primer_diff, ProjectDiff
from .test_helpers import (
    MOCK_GT_SCENARIO_A_RESPONSE,
    MOCK_GT_SCENARIO_B_RESPONSE,
    MOCK_MIXED_SCENARIO_RESPONSE,
    MOCK_OVERRIDE_SCENARIO_RESPONSE,
    MOCK_TYPE_CHECKING_SCENARIO_RESPONSE,
    MOCK_VARIANCE_SCENARIO_RESPONSE,
    MOCK_VARIANCE_SUGGESTION_RESPONSE,
    GT_BAD_OVERRIDE_ARGS_DIFF,
    GT_PROTOCOL_SUBTYPING_DIFF,
    GT_TYPE_CHECKING_DIFF,
    TYPE_CHECKING_DIFF,
    VARIANCE_DIFF,
    build_all_improvements_scenario,
    build_classification_result,
    build_gt_all_neutral_scenario,
    build_gt_bad_override_args_scenario,
    build_gt_protocol_subtyping_scenario,
    build_gt_pure_improvement_scenario,
    build_gt_type_checking_scenario,
    build_override_scenario,
    build_type_checking_scenario,
    build_variance_scenario,
    load_fixture,
    make_error_entry,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "unit"


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
        projects = parse_primer_diff(load_fixture("all_removals.txt"))
        assert len(projects) == 1
        assert projects[0].name == "myproject"
        assert projects[0].url == "https://github.com/example/myproject"
        assert len(projects[0].added) == 0
        assert len(projects[0].removed) == 2

    def test_multi_project(self):
        projects = parse_primer_diff(load_fixture("multi_project.txt"))
        assert len(projects) == 4
        names = [p.name for p in projects]
        assert names == ["project_a", "project_b", "project_c", "project_d"]

    def test_github_actions_format(self):
        projects = parse_primer_diff(load_fixture("github_actions_format.txt"))
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

    def test_all_removals_without_llm_is_ambiguous(self):
        p = ProjectDiff(name="test", removed=[self._make_entry()])
        result = classify_project(p, fetch_code=False, use_llm=False)
        assert result.verdict == "ambiguous"
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
            ProjectDiff(name="a", removed=[self._make_entry()]),  # ambiguous (no LLM)
            ProjectDiff(
                name="b", added=[self._make_entry(kind="internal-error")]
            ),  # regression
            ProjectDiff(name="c", added=[self._make_entry()]),  # ambiguous (no LLM)
        ]
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 3
        assert result.improvements == 0
        assert result.regressions == 1
        assert result.ambiguous == 2


class TestClassifyFromFixtures:
    def test_all_removals_fixture(self):
        projects = parse_primer_diff(load_fixture("all_removals.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.ambiguous == 1
        assert result.regressions == 0

    def test_internal_errors_fixture(self):
        projects = parse_primer_diff(load_fixture("internal_errors.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.regressions == 1
        assert result.improvements == 0

    def test_wording_changes_fixture(self):
        projects = parse_primer_diff(load_fixture("wording_changes.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.neutrals == 1

    def test_multi_project_fixture(self):
        projects = parse_primer_diff(load_fixture("multi_project.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 4
        # project_a: all removals -> ambiguous (needs LLM to determine FP vs FN)
        # project_b: internal-error -> regression
        # project_c: wording change -> neutral
        # project_d: non-trivial -> ambiguous
        assert result.improvements == 0
        assert result.regressions == 1
        assert result.neutrals == 1
        assert result.ambiguous == 2

    def test_empty_fixture(self):
        projects = parse_primer_diff(load_fixture("empty.txt"))
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.total_projects == 0

    def test_mixed_changes_fixture(self):
        projects = parse_primer_diff(load_fixture("mixed_changes.txt"))
        assert len(projects) == 1
        assert projects[0].name == "mixedproject"
        assert len(projects[0].added) == 2
        assert len(projects[0].removed) == 1
        # Non-trivial mixed change without LLM -> ambiguous
        result = classify_all(projects, fetch_code=False, use_llm=False)
        assert result.ambiguous == 1


class TestRealFixtureParsing:
    """Verify that real primer diffs from actual PRs parse without errors."""

    REAL_DIR = Path(__file__).parent / "fixtures" / "real"

    def test_parse_all_real_fixtures(self):
        if not self.REAL_DIR.exists():
            pytest.skip("No real fixtures directory")
        for fixture in sorted(self.REAL_DIR.glob("*.txt")):
            projects = parse_primer_diff(fixture.read_text())
            assert (
                len(projects) > 0
            ), f"{fixture.name} should parse into at least one project"
            for p in projects:
                assert p.name, f"Project in {fixture.name} has no name"
                assert (
                    p.added or p.removed
                ), f"Project {p.name} in {fixture.name} has no changes"


# ---------------------------------------------------------------------------
# Categorization tests
# ---------------------------------------------------------------------------


class TestCategorization:
    def test_extract_class_name(self):
        assert (
            _extract_class_name("Object of class `Foo` has no attribute `bar`") == "Foo"
        )
        assert _extract_class_name("no class here") is None

    def test_format_below_threshold_is_raw(self):
        """Below _CATEGORY_THRESHOLD, errors are listed individually."""
        entries = [
            ErrorEntry(
                "ERROR",
                "a.py",
                "1:1",
                "msg",
                "bad-return",
                "ERROR a.py:1:1: msg [bad-return]",
            ),
        ]
        p = ProjectDiff(name="test", added=entries)
        text = _format_errors_for_llm(p)
        assert "+ ERROR a.py:1:1: msg [bad-return]" in text
        assert "Error summary" not in text

    def test_format_above_threshold_uses_categories(self):
        """Above _CATEGORY_THRESHOLD, errors are grouped into categories."""
        entries = [
            ErrorEntry(
                "ERROR",
                f"f{i}.py",
                f"{i}:1",
                f"Object of class `X` has no attribute `a{i}`",
                "missing-attribute",
                f"raw{i}",
            )
            for i in range(10)
        ]
        p = ProjectDiff(name="test", added=entries)
        text = _format_errors_for_llm(p)
        assert "Error summary" in text
        assert "[missing-attribute]" in text

    def test_truncate_source_context_none(self):
        assert _truncate_source_context(None, "some errors") is None

    def test_truncate_source_context_fits(self):
        """Small context should pass through unchanged."""
        ctx = "def foo():\n    return 42\n"
        result = _truncate_source_context(ctx, "errors")
        assert result == ctx

    def test_truncate_source_context_too_large(self):
        """Oversized context should be truncated with a marker."""
        from .classifier import _MAX_PROMPT_CHARS

        # Make errors text consume most of the budget
        huge_errors = "x" * (_MAX_PROMPT_CHARS - 100)
        ctx = "line\n" * 1000
        result = _truncate_source_context(ctx, huge_errors)
        # Should be truncated or None due to budget exhaustion
        assert result is None or "[... source context truncated" in result


# ---------------------------------------------------------------------------
# LLM client tests (no actual API calls)
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_llama_preferred(self):
        with patch.dict(
            os.environ, {"LLAMA_API_KEY": "key1", "ANTHROPIC_API_KEY": "key2"}
        ):
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
        text = (
            'Here is my analysis:\n{"verdict": "neutral", "reason": "wording"}\nDone.'
        )
        result = _parse_classification(text)
        assert result["verdict"] == "neutral"

    def test_nested_json_with_categories(self):
        obj = {
            "verdict": "regression",
            "reason": "overall bad",
            "categories": [
                {
                    "category": "missing-attr",
                    "verdict": "regression",
                    "reason": "false positives",
                }
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

    def test_pass1_response_without_verdict(self):
        """Pass 1 responses have reason but no verdict — should parse OK."""
        text = json.dumps({
            "spec_check": "N/A",
            "runtime_behavior": "N/A",
            "mypy_pyright": "N/A",
            "removal_assessment": "These were false positives",
            "pr_attribution": "N/A",
            "reason": "The removed errors were false positives from inference failures",
            "categories": [{"category": "missing-attr", "reason": "false positives"}],
        })
        result = _parse_classification(text)
        assert "verdict" not in result
        assert result["reason"] == "The removed errors were false positives from inference failures"
        assert len(result["categories"]) == 1

    def test_pass1_embedded_in_text_without_verdict(self):
        """Pass 1 response embedded in text should be found by reason key."""
        text = 'Analysis:\n{"reason": "false positives removed", "pr_attribution": "N/A"}\nDone.'
        result = _parse_classification(text)
        assert result["reason"] == "false positives removed"


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
                    pr_attribution="check_for_imported_final_reassignment() in pyrefly/lib/binding/binding.rs",
                    categories=[
                        CategoryVerdict(
                            "bad-assignment", "regression", "false positives"
                        ),
                    ],
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

    def test_has_table_header(self):
        md = format_markdown(self._make_result())
        assert "| Project |" in md
        assert "| Verdict |" in md
        assert "| Root Cause |" in md

    def test_has_collapsible_details(self):
        md = format_markdown(self._make_result())
        assert "<details>" in md
        assert "<summary>Detailed analysis</summary>" in md
        assert "</details>" in md

    def test_linkifies_function_names(self):
        md = format_markdown(self._make_result())
        assert "github.com" in md
        assert "check_for_imported_final_reassignment()" in md

    def test_empty_result(self):
        md = format_markdown(ClassificationResult())
        assert "No diffs" in md or "All clear" in md

    def test_with_categories(self):
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
                        CategoryVerdict(
                            "missing-attr", "regression", "false positives"
                        ),
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


# ---------------------------------------------------------------------------
# PR diff attribution tests
# ---------------------------------------------------------------------------


class TestBuildUserPromptWithDiff:
    def test_includes_diff_when_provided(self):
        prompt = _build_user_prompt(
            errors_text="+ ERROR a.py:1:1: msg [bad-return]",
            source_context=None,
            change_type="additions only",
            pyrefly_diff="diff --git a/alt/answers.rs\n+fn new_logic() {}",
        )
        assert "Pyrefly PR diff" in prompt
        assert "alt/answers.rs" in prompt

    def test_omits_diff_when_none(self):
        prompt = _build_user_prompt(
            errors_text="+ ERROR a.py:1:1: msg [bad-return]",
            source_context=None,
            change_type="additions only",
            pyrefly_diff=None,
        )
        assert "Pyrefly PR diff" not in prompt

    def test_omits_diff_when_empty(self):
        prompt = _build_user_prompt(
            errors_text="+ ERROR a.py:1:1: msg [bad-return]",
            source_context=None,
            change_type="additions only",
            pyrefly_diff="",
        )
        assert "Pyrefly PR diff" not in prompt


class TestPrAttributionParsing:
    def test_pr_attribution_parsed_from_response(self):
        text = json.dumps({
            "spec_check": "N/A",
            "runtime_behavior": "N/A",
            "mypy_pyright": "N/A",
            "removal_assessment": "N/A",
            "pr_attribution": "Change to overload_resolution() in alt/answers.rs",
            "reason": "Fixed false positives",
            "verdict": "improvement",
        })
        result = _parse_classification(text)
        assert result["pr_attribution"] == "Change to overload_resolution() in alt/answers.rs"

    def test_pr_attribution_defaults_to_empty(self):
        text = json.dumps({
            "reason": "test",
            "verdict": "regression",
        })
        result = _parse_classification(text)
        assert result.get("pr_attribution", "") == ""


class TestLLMResponsePrAttribution:
    def test_pr_attribution_field(self):
        resp = LLMResponse(
            verdict="improvement",
            reason="removed false positives",
            pr_attribution="Change in alt/answers.rs fixed overload resolution",
        )
        assert resp.pr_attribution == "Change in alt/answers.rs fixed overload resolution"

    def test_pr_attribution_default_empty(self):
        resp = LLMResponse(verdict="neutral", reason="wording change")
        assert resp.pr_attribution == ""


class TestClassificationPrAttribution:
    def test_pr_attribution_field(self):
        c = Classification(
            project_name="test",
            verdict="improvement",
            reason="good",
            pr_attribution="Change in solver.rs",
        )
        assert c.pr_attribution == "Change in solver.rs"

    def test_pr_attribution_default_empty(self):
        c = Classification(
            project_name="test",
            verdict="neutral",
            reason="wording",
        )
        assert c.pr_attribution == ""


class TestClassifyAllWithDiff:
    def _make_entry(self, kind: str = "bad-return") -> ErrorEntry:
        return ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message="msg",
            error_kind=kind,
            raw_line=f"ERROR src/foo.py:10:5-20: msg [{kind}]",
        )

    def test_pyrefly_diff_accepted(self):
        """classify_all accepts pyrefly_diff without error."""
        projects = [
            ProjectDiff(
                name="b", added=[self._make_entry(kind="internal-error")]
            ),
        ]
        result = classify_all(
            projects,
            fetch_code=False,
            use_llm=False,
            pyrefly_diff="diff --git a/foo.rs",
        )
        assert result.total_projects == 1
        assert result.regressions == 1


class TestFormatterPrAttribution:
    def test_markdown_shows_attribution(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="removed false positives",
                    method="llm",
                    pr_attribution="Change in alt/answers.rs fixed overload resolution",
                ),
            ],
        )
        md = format_markdown(result)
        assert "**Attribution:**" in md
        assert "alt/answers.rs" in md

    def test_markdown_hides_na_attribution(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="removed false positives",
                    method="llm",
                    pr_attribution="N/A",
                ),
            ],
        )
        md = format_markdown(result)
        assert "Attribution" not in md

    def test_markdown_hides_empty_attribution(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="removed false positives",
                    method="llm",
                ),
            ],
        )
        md = format_markdown(result)
        assert "Attribution" not in md

    def test_json_includes_attribution(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="good",
                    method="llm",
                    pr_attribution="Change in solver.rs",
                ),
            ],
        )
        output = format_json(result)
        data = json.loads(output)
        assert data["classifications"][0]["pr_attribution"] == "Change in solver.rs"

    def test_json_includes_empty_attribution(self):
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
        assert data["classifications"][0]["pr_attribution"] == ""


class TestTruncateSourceContextWithDiff:
    def test_diff_len_reduces_budget(self):
        """Source context budget should shrink when pyrefly_diff is present."""
        from .classifier import _MAX_PROMPT_CHARS

        # Create a source context large enough that the diff budget matters.
        errors = "x" * 1000
        ctx = "a" * (_MAX_PROMPT_CHARS - 20_000)

        # Without diff len, context should pass through (or be truncated less)
        result_no_diff = _truncate_source_context(ctx, errors, pyrefly_diff_len=0)
        # With a large diff len, context should be truncated further or dropped
        result_with_diff = _truncate_source_context(
            ctx, errors, pyrefly_diff_len=_MAX_PROMPT_CHARS // 2
        )

        assert result_no_diff is not None
        if result_with_diff is None:
            # Budget exhausted entirely — diff ate the remaining space
            pass
        else:
            assert len(result_with_diff) < len(result_no_diff)


# ---------------------------------------------------------------------------
# Two-pass classification tests
# ---------------------------------------------------------------------------


class TestBuildVerdictPrompt:
    def test_includes_reasoning(self):
        reason = "The removed errors were false positives from inference failures"
        categories = [
            CategoryVerdict("missing-attr", "", "attributes exist via inheritance"),
            CategoryVerdict("bad-return", "", "return type mismatch is real"),
        ]
        prompt = _build_verdict_prompt(reason, categories)
        assert reason in prompt
        assert "missing-attr" in prompt
        assert "attributes exist via inheritance" in prompt
        assert "bad-return" in prompt

    def test_empty_categories(self):
        prompt = _build_verdict_prompt("simple reasoning", [])
        assert "simple reasoning" in prompt
        assert "Per-category" not in prompt


class TestBuildVerdictSystemPrompt:
    def test_contains_verdict_rules(self):
        prompt = _build_verdict_system_prompt()
        assert "improvement" in prompt
        assert "regression" in prompt
        assert "neutral" in prompt
        assert "verdict" in prompt


class TestTwoPassClassifyProject:
    def _make_entry(self, kind: str = "bad-return", msg: str = "msg") -> ErrorEntry:
        return ErrorEntry(
            severity="ERROR",
            file_path="src/foo.py",
            location="10:5-20",
            message=msg,
            error_kind=kind,
            raw_line=f"ERROR src/foo.py:10:5-20: {msg} [{kind}]",
        )

    def test_pass1_returns_empty_verdict(self):
        """classify_with_llm (pass 1) should return empty verdict."""
        pass1_response = {
            "reason": "Removed errors were false positives",
            "pr_attribution": "N/A",
            "categories": [{"category": "missing-attr", "reason": "FP"}],
        }
        # Mock the API call to return a pass 1 response (no verdict)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(pass1_response)}]},
            ):
                from .llm_client import classify_with_llm

                result = classify_with_llm(
                    errors_text="+ ERROR a.py:1:1: msg [bad-return]",
                )
                assert result.verdict == ""
                assert result.reason == "Removed errors were false positives"
                assert len(result.categories) == 1
                assert result.categories[0].verdict == ""

    def test_assign_verdict_improvement(self):
        """assign_verdict_with_llm should assign 'improvement' for false-positive reasoning."""
        verdict_response = {
            "verdict": "improvement",
            "categories": [{"category": "missing-attr", "verdict": "improvement"}],
        }
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(verdict_response)}]},
            ):
                categories = [CategoryVerdict("missing-attr", "", "false positives")]
                verdict, updated_cats = assign_verdict_with_llm(
                    "Removed errors were false positives from inference failures",
                    categories,
                )
                assert verdict == "improvement"
                assert updated_cats[0].verdict == "improvement"
                assert updated_cats[0].reason == "false positives"

    def test_assign_verdict_regression(self):
        """assign_verdict_with_llm should assign 'regression' for real-bug reasoning."""
        verdict_response = {"verdict": "regression"}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(verdict_response)}]},
            ):
                verdict, _ = assign_verdict_with_llm(
                    "Removed errors were catching real bugs",
                    [],
                )
                assert verdict == "regression"

    def test_two_pass_end_to_end(self):
        """Full two-pass flow: classify_project calls pass 1 then pass 2."""
        pass1_response = {
            "reason": "These missing-attribute errors are false positives",
            "pr_attribution": "Change in solver.rs",
            "spec_check": "N/A",
            "runtime_behavior": "N/A",
            "mypy_pyright": "N/A",
            "removal_assessment": "False positives",
        }
        pass2_response = {"verdict": "improvement"}

        p = ProjectDiff(name="test", removed=[self._make_entry()])

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                mock_api.side_effect = [
                    # Pass 1: reasoning
                    {"content": [{"text": json.dumps(pass1_response)}]},
                    # Pass 2: verdict
                    {"content": [{"text": json.dumps(pass2_response)}]},
                ]
                result = classify_project(p, fetch_code=False, use_llm=True)
                assert result.verdict == "improvement"
                assert result.reason == "These missing-attribute errors are false positives"
                assert result.pr_attribution == "Change in solver.rs"
                assert result.method == "llm"
                # Verify both passes were called
                assert mock_api.call_count == 2

    def test_two_pass_with_categories(self):
        """Two-pass flow with per-category verdicts."""
        pass1_response = {
            "reason": "Mixed results",
            "pr_attribution": "N/A",
            "categories": [
                {"category": "missing-attr", "reason": "false positives from inheritance"},
                {"category": "bad-return", "reason": "real type errors caught"},
            ],
        }
        pass2_response = {
            "verdict": "regression",
            "categories": [
                {"category": "missing-attr", "verdict": "improvement"},
                {"category": "bad-return", "verdict": "regression"},
            ],
        }

        # Use different error kinds so the wording-change heuristic doesn't match
        p = ProjectDiff(
            name="test",
            added=[self._make_entry(kind="missing-attribute", msg="no attr x")],
            removed=[self._make_entry(kind="bad-return", msg="wrong return")],
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                mock_api.side_effect = [
                    {"content": [{"text": json.dumps(pass1_response)}]},
                    {"content": [{"text": json.dumps(pass2_response)}]},
                ]
                result = classify_project(p, fetch_code=False, use_llm=True)
                assert result.verdict == "regression"
                assert len(result.categories) == 2
                assert result.categories[0].verdict == "improvement"
                assert result.categories[0].reason == "false positives from inheritance"
                assert result.categories[1].verdict == "regression"

    def test_two_pass_with_file_request(self):
        """Multi-pass file fetching still works, verdict assigned after final pass."""
        needs_files_response = {"needs_files": ["foo/bar.py"]}
        pass1_response = {
            "reason": "After seeing source: false positives",
            "pr_attribution": "N/A",
        }
        pass2_response = {"verdict": "improvement"}

        p = ProjectDiff(
            name="test",
            url="https://github.com/example/test",
            removed=[self._make_entry()],
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                mock_api.side_effect = [
                    # Pass 1, attempt 1: needs files
                    {"content": [{"text": json.dumps(needs_files_response)}]},
                    # Pass 1, attempt 2 (with files): reasoning
                    {"content": [{"text": json.dumps(pass1_response)}]},
                    # Pass 2: verdict
                    {"content": [{"text": json.dumps(pass2_response)}]},
                ]
                with patch(
                    "primer_classifier.classifier.fetch_files_by_path",
                    return_value="def bar(): pass",
                ):
                    result = classify_project(p, fetch_code=True, use_llm=True)
                    assert result.verdict == "improvement"
                    assert mock_api.call_count == 3


# ---------------------------------------------------------------------------
# Pass 3: Suggestion generation tests (all mocked)
# ---------------------------------------------------------------------------


class TestBuildSuggestionUserPromptFormatting:
    """Verify the serialized prompt contains project names, verdicts, reasons, counts, diff."""

    def test_contains_project_info(self):
        result = build_variance_scenario()
        prompt = _build_suggestion_user_prompt(result, VARIANCE_DIFF)
        assert "variance_proj_0" in prompt
        assert "variance_proj_4" in prompt
        assert "REGRESSION" in prompt
        assert "variance" in prompt.lower()
        assert "protocol" in prompt.lower()
        assert "+5/-0" in prompt
        assert "variance.rs" in prompt
        assert "is_protocol" in prompt

    def test_contains_attribution(self):
        result = build_type_checking_scenario()
        prompt = _build_suggestion_user_prompt(result, TYPE_CHECKING_DIFF)
        assert "Stricter final-variable checking" in prompt

    def test_mixed_verdicts_shown(self):
        """Both regressions and improvements appear in the prompt."""
        regression = build_classification_result(
            2, "regression", "variance-mismatch", "too broad", "diff", "reg"
        )
        improvement = build_classification_result(
            1, "improvement", "missing-attribute", "FP removed", "diff", "imp"
        )
        merged = ClassificationResult(
            classifications=regression.classifications + improvement.classifications,
            total_projects=3,
            regressions=2,
            improvements=1,
        )
        prompt = _build_suggestion_user_prompt(merged, "diff --git a/foo.rs")
        assert "REGRESSION" in prompt
        assert "IMPROVEMENT" in prompt
        assert "reg_0" in prompt
        assert "imp_0" in prompt


class TestGenerateSuggestionsParsing:
    """Mock API to return known JSON, verify it parses into SuggestionResult."""

    def test_parses_suggestions(self):
        result = build_variance_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_VARIANCE_SUGGESTION_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, VARIANCE_DIFF)
                assert len(suggestion.suggestions) == 1
                assert suggestion.suggestions[0].confidence == "high"
                assert "variance.rs" in suggestion.suggestions[0].files[0]
                assert suggestion.summary == "Variance check needs protocol guard"
                assert suggestion.has_regressions is True


class TestGenerateSuggestionsSkipsWhenNoRegressions:
    """Verify generate_suggestions() returns early without calling the API."""

    def test_skips_api_call(self):
        result = build_all_improvements_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                suggestion = generate_suggestions(result, "diff --git a/foo.rs")
                assert suggestion.suggestions == []
                assert suggestion.has_regressions is False
                mock_api.assert_not_called()


class TestClassifyAllWithSuggestFlag:
    """Mock all LLM calls, verify full pipeline produces suggestions."""

    def test_full_pipeline(self):
        pass1_response = {
            "reason": "Variance check too broad",
            "pr_attribution": "Removed is_protocol() guard",
        }
        pass2_response = {"verdict": "regression"}
        pass3_response = {
            "summary": "Restore protocol guard",
            "suggestions": [
                {
                    "description": "Add is_protocol() check",
                    "files": ["variance.rs"],
                    "confidence": "high",
                    "reasoning": "Guard was removed",
                },
            ],
        }

        entry = make_error_entry(kind="variance-mismatch", msg="variance issue")
        projects = [ProjectDiff(name="test_proj", added=[entry])]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                mock_api.side_effect = [
                    {"content": [{"text": json.dumps(pass1_response)}]},  # Pass 1
                    {"content": [{"text": json.dumps(pass2_response)}]},  # Pass 2
                    {"content": [{"text": json.dumps(pass3_response)}]},  # Pass 3
                ]
                result = classify_all(
                    projects,
                    fetch_code=False,
                    use_llm=True,
                    generate_suggestion=True,
                )
                assert result.regressions == 1
                assert result.suggestion is not None
                assert len(result.suggestion.suggestions) == 1
                assert result.suggestion.suggestions[0].description == "Add is_protocol() check"
                assert mock_api.call_count == 3


class TestSuggestionInMarkdownOutput:
    """Verify format_markdown() renders the suggestion section with GitHub links."""

    def test_renders_suggestion(self):
        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="too strict",
                    added_count=5,
                    method="llm",
                ),
            ],
            suggestion=SuggestionResult(
                suggestions=[
                    Suggestion(
                        description="Restore protocol guard",
                        files=["pyrefly/lib/alt/class/variance.rs"],
                        confidence="high",
                        reasoning="Guard was removed too broadly",
                    ),
                ],
                summary="Variance check needs narrowing",
                has_regressions=True,
            ),
        )
        md = format_markdown(result)
        assert "Suggested Fix" in md
        assert "Restore protocol guard" in md
        assert "github.com/facebook/pyrefly" in md
        assert "pyrefly/lib/alt/class/variance.rs" in md
        assert "high" in md
        assert "Guard was removed too broadly" in md
        assert "Variance check needs narrowing" in md


class TestSuggestionInJsonOutput:
    """Verify format_json() includes the suggestion key with file URLs."""

    def test_includes_suggestion(self):
        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="too strict",
                    method="llm",
                ),
            ],
            suggestion=SuggestionResult(
                suggestions=[
                    Suggestion(
                        description="Fix variance",
                        files=["pyrefly/lib/alt/class/variance.rs"],
                        confidence="medium",
                        reasoning="Scope too broad",
                    ),
                ],
                summary="Needs fix",
                has_regressions=True,
            ),
        )
        output = format_json(result)
        data = json.loads(output)
        assert "suggestion" in data
        assert data["suggestion"]["summary"] == "Needs fix"
        assert len(data["suggestion"]["suggestions"]) == 1
        assert data["suggestion"]["suggestions"][0]["confidence"] == "medium"
        assert "file_urls" in data["suggestion"]["suggestions"][0]
        assert "github.com/facebook/pyrefly" in data["suggestion"]["suggestions"][0]["file_urls"][0]


class TestSuggestionOmittedWhenNone:
    """Verify formatters handle suggestion=None gracefully."""

    def test_markdown_no_suggestion_section(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="good",
                    method="llm",
                ),
            ],
        )
        md = format_markdown(result)
        assert "Suggested Fixes" not in md

    def test_json_no_suggestion_key(self):
        result = ClassificationResult(
            total_projects=1,
            improvements=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="improvement",
                    reason="good",
                    method="llm",
                ),
            ],
        )
        output = format_json(result)
        data = json.loads(output)
        assert "suggestion" not in data


# ---------------------------------------------------------------------------
# Known-good-answer scenario tests (mock LLM, verify right data reaches it)
# ---------------------------------------------------------------------------


class TestScenarioVarianceToBroad:
    """Scenario 1: Variance inference too broad — removed is_protocol() guard."""

    def test_prompt_contains_regression_details(self):
        result = build_variance_scenario()
        prompt = _build_suggestion_user_prompt(result, VARIANCE_DIFF)
        assert "variance" in prompt.lower()
        assert "protocol" in prompt.lower()
        assert "is_protocol" in prompt
        assert "variance.rs" in prompt

    def test_mock_suggestion_parses_correctly(self):
        result = build_variance_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_VARIANCE_SCENARIO_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, VARIANCE_DIFF)
                assert len(suggestion.suggestions) == 1
                s = suggestion.suggestions[0]
                assert "protocol" in s.reasoning.lower()
                assert "variance.rs" in s.files[0]
                assert s.confidence == "high"


class TestScenarioTypeCheckingExempt:
    """Scenario 2: TYPE_CHECKING final assignment regressions."""

    def test_prompt_contains_all_projects(self):
        result = build_type_checking_scenario()
        prompt = _build_suggestion_user_prompt(result, TYPE_CHECKING_DIFF)
        assert "urllib3" in prompt
        assert "trio" in prompt
        assert "zulip" in prompt
        assert "ibis" in prompt
        assert "type_checking" in prompt.lower() or "TYPE_CHECKING" in prompt
        assert "bad-assignment" in prompt

    def test_mock_suggestion_correct(self):
        result = build_type_checking_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_TYPE_CHECKING_SCENARIO_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, TYPE_CHECKING_DIFF)
                assert len(suggestion.suggestions) == 1
                text = suggestion.suggestions[0].description + " " + suggestion.suggestions[0].reasoning
                assert any(
                    kw in text.lower()
                    for kw in ["type_checking", "exempt", "final"]
                )


class TestScenarioBadOverride:
    """Scenario 3: bad-override flood across multiple projects."""

    def test_prompt_contains_override_info(self):
        result = build_override_scenario()
        diff = "diff --git a/pyrefly/lib/alt/class/override.rs\n+stricter checks"
        prompt = _build_suggestion_user_prompt(result, diff)
        assert "bad-override" in prompt
        assert "jax" in prompt
        assert "bokeh" in prompt
        assert "poetry" in prompt
        assert "artigraph" in prompt

    def test_mock_suggestion_references_override(self):
        result = build_override_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_OVERRIDE_SCENARIO_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, "diff override.rs")
                assert len(suggestion.suggestions) >= 1
                assert "override" in suggestion.suggestions[0].reasoning.lower()


class TestScenarioPureImprovement:
    """Scenario 4: Pure improvement — no suggestion needed."""

    def test_skips_api_and_returns_empty(self):
        result = build_all_improvements_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                suggestion = generate_suggestions(result, "diff --git a/foo.rs")
                assert suggestion.suggestions == []
                assert not suggestion.has_regressions
                mock_api.assert_not_called()


class TestScenarioMixed:
    """Scenario 5: Mixed — regressions + improvements."""

    def test_prompt_contains_both(self):
        reg = build_classification_result(
            2, "regression", "variance-mismatch", "too broad", "variance.rs", "reg"
        )
        imp = build_classification_result(
            3, "improvement", "missing-attribute", "FP removed", "resolve.rs", "imp"
        )
        merged = ClassificationResult(
            classifications=reg.classifications + imp.classifications,
            total_projects=5,
            regressions=2,
            improvements=3,
        )
        prompt = _build_suggestion_user_prompt(merged, "diff --git a/variance.rs")
        # Both regressions and improvements in the prompt
        assert "REGRESSION" in prompt
        assert "IMPROVEMENT" in prompt
        assert "reg_0" in prompt
        assert "imp_0" in prompt

    def test_suggestion_targets_regression(self):
        reg = build_classification_result(
            2, "regression", "variance-mismatch", "too broad", "variance.rs", "reg"
        )
        imp = build_classification_result(
            3, "improvement", "missing-attribute", "FP removed", "resolve.rs", "imp"
        )
        merged = ClassificationResult(
            classifications=reg.classifications + imp.classifications,
            total_projects=5,
            regressions=2,
            improvements=3,
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_MIXED_SCENARIO_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(merged, "diff variance.rs")
                assert len(suggestion.suggestions) >= 1
                assert suggestion.has_regressions is True
                # Suggestion targets variance, not the improvements
                assert "variance" in suggestion.suggestions[0].description.lower()


# ---------------------------------------------------------------------------
# Ground-truth scenario tests (mock LLM)
# ---------------------------------------------------------------------------


class TestGroundTruthPromptConstruction:
    """Verify ground-truth scenario data appears correctly in the user prompt."""

    def test_scenario_a_prompt_has_real_projects(self):
        result = build_gt_protocol_subtyping_scenario()
        prompt = _build_suggestion_user_prompt(result, GT_PROTOCOL_SUBTYPING_DIFF)
        assert "jax" in prompt
        assert "bokeh" in prompt
        assert "poetry" in prompt
        assert "artigraph" in prompt
        assert "hydra-zen" in prompt
        assert "class_metadata.rs" in prompt
        assert "calculate_abstract_members" in prompt or "class_body_fields" in prompt

    def test_scenario_a_prompt_has_aggregate_info(self):
        result = build_gt_protocol_subtyping_scenario()
        prompt = _build_suggestion_user_prompt(result, GT_PROTOCOL_SUBTYPING_DIFF)
        assert "Regression error kinds:" in prompt
        assert "bad-override" in prompt
        assert "Affected projects:" in prompt

    def test_scenario_b_prompt_has_type_checking(self):
        result = build_gt_type_checking_scenario()
        prompt = _build_suggestion_user_prompt(result, GT_TYPE_CHECKING_DIFF)
        assert "urllib3" in prompt
        assert "trio" in prompt
        assert "zulip" in prompt
        assert "ibis" in prompt
        assert "bad-assignment" in prompt
        assert "check_for_imported_final_reassignment" in prompt

    def test_scenario_b_prompt_has_aggregate_info(self):
        result = build_gt_type_checking_scenario()
        prompt = _build_suggestion_user_prompt(result, GT_TYPE_CHECKING_DIFF)
        assert "Regression error kinds:" in prompt
        assert "bad-assignment" in prompt
        assert "Affected projects:" in prompt

    def test_scenario_c_prompt_has_override_info(self):
        result = build_gt_bad_override_args_scenario()
        prompt = _build_suggestion_user_prompt(result, GT_BAD_OVERRIDE_ARGS_DIFF)
        assert "jax" in prompt
        assert "bad-override" in prompt
        assert "subset.rs" in prompt


class TestGroundTruthMockParsing:
    """Mock API returns ground-truth-quality answers; verify parsing + new fields."""

    def test_scenario_a_parses_with_new_fields(self):
        result = build_gt_protocol_subtyping_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_GT_SCENARIO_A_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, GT_PROTOCOL_SUBTYPING_DIFF)
                assert len(suggestion.suggestions) == 1
                s = suggestion.suggestions[0]
                assert s.affected_projects == ["jax", "bokeh", "poetry", "artigraph"]
                assert s.error_kinds_fixed == ["bad-override"]
                assert "class_metadata.rs" in s.files[0]
                assert "calculate_abstract_members" in s.description

    def test_scenario_b_parses_with_new_fields(self):
        result = build_gt_type_checking_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
                return_value={"content": [{"text": json.dumps(MOCK_GT_SCENARIO_B_RESPONSE)}]},
            ):
                suggestion = generate_suggestions(result, GT_TYPE_CHECKING_DIFF)
                assert len(suggestion.suggestions) == 1
                s = suggestion.suggestions[0]
                assert s.affected_projects == ["urllib3", "trio", "zulip", "ibis"]
                assert s.error_kinds_fixed == ["bad-assignment"]
                assert "bindings.rs" in s.files[0]


class TestGroundTruthSkipsWhenNoRegressions:
    """Scenarios D and E: no regressions should skip LLM call."""

    def test_scenario_d_pure_improvement_skips(self):
        result = build_gt_pure_improvement_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                suggestion = generate_suggestions(result, "diff --git a/foo.rs")
                assert suggestion.suggestions == []
                assert not suggestion.has_regressions
                mock_api.assert_not_called()

    def test_scenario_e_all_neutral_skips(self):
        result = build_gt_all_neutral_scenario()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch(
                "primer_classifier.llm_client._call_anthropic_api",
            ) as mock_api:
                suggestion = generate_suggestions(result, "diff --git a/foo.rs")
                assert suggestion.suggestions == []
                assert not suggestion.has_regressions
                mock_api.assert_not_called()


class TestGroundTruthFormatterNewFields:
    """Verify new Suggestion fields render correctly in markdown and JSON."""

    def test_markdown_shows_affected_projects(self):
        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="too strict",
                    added_count=5,
                    method="llm",
                ),
            ],
            suggestion=SuggestionResult(
                suggestions=[
                    Suggestion(
                        description="Fix calculate_abstract_members()",
                        files=["pyrefly/lib/alt/class/class_metadata.rs"],
                        confidence="high",
                        reasoning="Guard was missing",
                        affected_projects=["jax", "bokeh", "poetry"],
                        error_kinds_fixed=["bad-override"],
                    ),
                ],
                summary="Protocol subtyping fix needed",
                has_regressions=True,
            ),
        )
        md = format_markdown(result)
        assert "Affected projects: jax, bokeh, poetry" in md
        assert "`bad-override`" in md
        assert "Fixes:" in md
        # New table format assertions
        assert "| Project |" in md
        assert "<details>" in md
        assert "Suggested Fix" in md
        # Function name should be linkified in the suggestion
        assert "calculate_abstract_members()" in md
        assert "github.com" in md

    def test_json_includes_new_fields(self):
        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="too strict",
                    method="llm",
                ),
            ],
            suggestion=SuggestionResult(
                suggestions=[
                    Suggestion(
                        description="Fix function",
                        files=["pyrefly/lib/alt/class/class_metadata.rs"],
                        confidence="high",
                        reasoning="Guard needed",
                        affected_projects=["jax", "bokeh"],
                        error_kinds_fixed=["bad-override", "bad-instantiation"],
                    ),
                ],
                summary="Fix needed",
                has_regressions=True,
            ),
        )
        output = format_json(result)
        data = json.loads(output)
        s = data["suggestion"]["suggestions"][0]
        assert s["affected_projects"] == ["jax", "bokeh"]
        assert s["error_kinds_fixed"] == ["bad-override", "bad-instantiation"]

    def test_markdown_omits_empty_new_fields(self):
        """When new fields are empty, they should not appear in output."""
        result = ClassificationResult(
            total_projects=1,
            regressions=1,
            classifications=[
                Classification(
                    project_name="proj",
                    verdict="regression",
                    reason="too strict",
                    added_count=5,
                    method="llm",
                ),
            ],
            suggestion=SuggestionResult(
                suggestions=[
                    Suggestion(
                        description="Fix variance",
                        files=["pyrefly/lib/alt/class/variance.rs"],
                        confidence="medium",
                        reasoning="Scope too broad",
                    ),
                ],
                summary="Needs fix",
                has_regressions=True,
            ),
        )
        md = format_markdown(result)
        assert "Affected projects:" not in md
        assert "Fixes:" not in md
