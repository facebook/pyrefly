# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Live LLM tests for the primer classifier (real API calls).

These tests require an ANTHROPIC_API_KEY environment variable and are marked
with @pytest.mark.slow so they don't run by default.

Run with: python -m pytest scripts/primer_classifier/test_llm_live.py -v -m slow
"""

from __future__ import annotations

import pytest

from .llm_client import generate_suggestions
from .test_helpers import (
    GT_BAD_OVERRIDE_ARGS_DIFF,
    GT_PROTOCOL_SUBTYPING_DIFF,
    GT_TYPE_CHECKING_DIFF,
    VARIANCE_DIFF,
    TYPE_CHECKING_DIFF,
    assert_actionable,
    build_all_improvements_scenario,
    build_gt_all_neutral_scenario,
    build_gt_bad_override_args_scenario,
    build_gt_protocol_subtyping_scenario,
    build_gt_pure_improvement_scenario,
    build_gt_type_checking_scenario,
    build_variance_scenario,
    build_type_checking_scenario,
)


# ---------------------------------------------------------------------------
# Live LLM suggestion quality tests (Pass 3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSuggestionQualityLive:
    """Live LLM tests — requires ANTHROPIC_API_KEY.

    Run with: python -m pytest scripts/primer_classifier/test_llm_live.py -v -m slow
    """

    def test_variance_too_broad_live(self):
        result = build_variance_scenario()
        diff = VARIANCE_DIFF
        suggestion = generate_suggestions(result, diff)
        assert len(suggestion.suggestions) >= 1
        text = " ".join(
            s.description + " " + s.reasoning for s in suggestion.suggestions
        )
        assert "protocol" in text.lower(), f"Expected 'protocol', got: {text}"
        # Should reference the actual file from the diff
        all_files = [f for s in suggestion.suggestions for f in s.files]
        assert any(
            "variance" in f for f in all_files
        ), f"Expected variance file reference, got: {all_files}"

    def test_type_checking_exempt_live(self):
        result = build_type_checking_scenario()
        diff = TYPE_CHECKING_DIFF
        suggestion = generate_suggestions(result, diff)
        assert len(suggestion.suggestions) >= 1
        text = " ".join(
            s.description + " " + s.reasoning for s in suggestion.suggestions
        )
        assert any(
            kw in text.lower() for kw in ["type_checking", "exempt", "final"]
        ), f"Expected TYPE_CHECKING/exempt/final, got: {text}"
        # Should reference the actual file from the diff
        all_files = [f for s in suggestion.suggestions for f in s.files]
        assert any(
            "stmt" in f or "binding" in f for f in all_files
        ), f"Expected stmt/binding file reference, got: {all_files}"

    def test_no_regressions_skips_live(self):
        result = build_all_improvements_scenario()
        suggestion = generate_suggestions(result, "diff --git a/foo.rs")
        assert suggestion.suggestions == []
        assert not suggestion.has_regressions


# ---------------------------------------------------------------------------
# Ground-truth live LLM quality tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGroundTruthLive:
    """Live LLM tests with ground-truth scenarios — requires ANTHROPIC_API_KEY.

    Run with: python -m pytest scripts/primer_classifier/test_llm_live.py -v -m slow
    """

    def test_scenario_a_protocol_subtyping_live(self):
        """PR #2428 -> #2492: LLM should name calculate_abstract_members or class_metadata."""
        result = build_gt_protocol_subtyping_scenario()
        diff = GT_PROTOCOL_SUBTYPING_DIFF
        suggestion = generate_suggestions(result, diff)
        assert len(suggestion.suggestions) >= 1
        text = " ".join(
            s.description + " " + s.reasoning for s in suggestion.suggestions
        )
        assert any(
            kw in text.lower()
            for kw in ["calculate_abstract_members", "class_metadata", "synthesized", "class_body_fields"]
        ), f"Expected function/file reference, got: {text}"
        all_files = [f for s in suggestion.suggestions for f in s.files]
        assert any(
            "class_metadata" in f for f in all_files
        ), f"Expected class_metadata file reference, got: {all_files}"
        # Actionability check on first suggestion
        assert_actionable(suggestion.suggestions[0])

    def test_scenario_b_type_checking_live(self):
        """PR #2389: LLM should name check_for_imported_final_reassignment or bindings."""
        result = build_gt_type_checking_scenario()
        diff = GT_TYPE_CHECKING_DIFF
        suggestion = generate_suggestions(result, diff)
        assert len(suggestion.suggestions) >= 1
        text = " ".join(
            s.description + " " + s.reasoning for s in suggestion.suggestions
        )
        assert any(
            kw in text.lower()
            for kw in ["check_for_imported_final_reassignment", "bindings", "type_checking"]
        ), f"Expected function/file reference, got: {text}"
        all_files = [f for s in suggestion.suggestions for f in s.files]
        assert any(
            "bindings" in f for f in all_files
        ), f"Expected bindings file reference, got: {all_files}"
        # Check affected projects
        all_projects = [p for s in suggestion.suggestions for p in s.affected_projects]
        for proj in ["urllib3", "trio", "zulip", "ibis"]:
            assert proj in all_projects, (
                f"Expected {proj} in affected_projects, got: {all_projects}"
            )
        # Actionability check
        assert_actionable(suggestion.suggestions[0])

    def test_scenario_c_bad_override_args_live(self):
        """PR #2322: LLM should reference subset.rs or args/kwargs/Any."""
        result = build_gt_bad_override_args_scenario()
        diff = GT_BAD_OVERRIDE_ARGS_DIFF
        suggestion = generate_suggestions(result, diff)
        assert len(suggestion.suggestions) >= 1
        text = " ".join(
            s.description + " " + s.reasoning for s in suggestion.suggestions
        )
        assert any(
            kw in text.lower()
            for kw in ["subset", "args", "kwargs", "any"]
        ), f"Expected override/args reference, got: {text}"
        assert_actionable(suggestion.suggestions[0])

    def test_scenario_d_pure_improvement_skips_live(self):
        """Pure improvement scenario should skip API call."""
        result = build_gt_pure_improvement_scenario()
        suggestion = generate_suggestions(result, "diff --git a/foo.rs")
        assert suggestion.suggestions == []
        assert not suggestion.has_regressions

    def test_scenario_e_all_neutral_skips_live(self):
        """All neutral scenario should skip API call."""
        result = build_gt_all_neutral_scenario()
        suggestion = generate_suggestions(result, "diff --git a/foo.rs")
        assert suggestion.suggestions == []
        assert not suggestion.has_regressions
