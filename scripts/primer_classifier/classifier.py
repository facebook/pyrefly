# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestrate classification of primer diff entries.

Applies heuristics for trivially obvious cases, fetches source code,
then delegates to the LLM for everything else.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .code_fetcher import fetch_files_by_path, fetch_source_context
from .cross_checker import cross_check_projects
from .llm_client import (
    assign_verdict_with_llm,
    CategoryVerdict,
    classify_with_llm,
    critique_reasoning,
    generate_suggestions,
    LLMError,
)
from .parser import ErrorEntry, ProjectDiff

# Rough token estimation: ~2 chars per token. Code tokenizes into shorter
# tokens than prose (brackets, operators, short identifiers each consume a
# full token), so 2 is genuinely conservative. We target 170K tokens to
# leave headroom below Anthropic's 200K limit.
_CHARS_PER_TOKEN = 2
_MAX_PROMPT_TOKENS = 170_000
_MAX_PROMPT_CHARS = _MAX_PROMPT_TOKENS * _CHARS_PER_TOKEN


@dataclass
class Classification:
    """Classification result for a single project."""

    project_name: str
    verdict: str  # "regression", "improvement", "neutral", "ambiguous"
    reason: str
    added_count: int = 0
    removed_count: int = 0
    method: str = "heuristic"  # "heuristic" or "llm"
    categories: list[CategoryVerdict] = field(default_factory=list)
    pr_attribution: str = ""
    error_kinds: list[str] = field(default_factory=list)
    cross_check_stats: str = ""  # e.g. "3/10 mypy, 5/10 pyright, 2/10 pyrefly-only"


@dataclass
class Suggestion:
    """A single actionable suggestion for fixing regressions."""

    description: str  # What to change, in plain English
    files: list[str]  # Pyrefly source files to modify
    confidence: str  # "high", "medium", "low"
    reasoning: str  # Why this fixes the regressions
    affected_projects: list[str] = field(default_factory=list)  # Projects impacted
    error_kinds_fixed: list[str] = field(default_factory=list)  # Error kinds addressed


@dataclass
class SuggestionResult:
    """Aggregate suggestion result from Pass 3."""

    suggestions: list[Suggestion]
    summary: str
    has_regressions: bool
    raw_response: Optional[dict] = None


@dataclass
class ClassificationResult:
    """Full classification result across all projects."""

    classifications: list[Classification] = field(default_factory=list)
    total_projects: int = 0
    regressions: int = 0
    improvements: int = 0
    neutrals: int = 0
    ambiguous: int = 0
    suggestion: Optional[SuggestionResult] = None


def _is_all_internal_errors(project: ProjectDiff) -> bool:
    """Check if all added errors are internal-error (pyrefly bug/panic)."""
    return len(project.added) > 0 and all(
        e.error_kind == "internal-error" for e in project.added
    )


_INFERENCE_FAILURE_MARKERS = ("Never", "@_")


def _has_inference_failure_markers(entries: list[ErrorEntry]) -> bool:
    """Check if any entries contain Never or @_ types (inference failures)."""
    return any(
        marker in e.message for e in entries for marker in _INFERENCE_FAILURE_MARKERS
    )


def _is_wording_change(project: ProjectDiff) -> bool:
    """Check if changes are just wording changes at identical locations.

    A wording change means: for every added error, there's a removed error
    at the same file:line with the same error kind, and vice versa. The
    message text differs, but the error kind and location are the same.

    Does NOT classify as wording change if new messages introduce Never or @_
    types, since that indicates a type inference regression even at the same
    location.
    """
    if not project.added or not project.removed:
        return False
    if len(project.added) != len(project.removed):
        return False

    # If new messages contain inference failure markers, this isn't a
    # simple wording change — the type checker's behavior changed materially.
    if _has_inference_failure_markers(project.added):
        return False

    # Build sets of (file, line, error_kind) for added and removed
    added_keys = {(e.file_path, e.line_number, e.error_kind) for e in project.added}
    removed_keys = {(e.file_path, e.line_number, e.error_kind) for e in project.removed}
    return added_keys == removed_keys


def _is_near_wording_change(project: ProjectDiff) -> bool:
    """Check if the diff is mostly wording changes with minor residual noise.

    Catches cases where most added/removed errors share the same location+kind
    (indicating type inference changes in the message text), with at most a
    few unmatched entries that don't alter the overall picture.

    Returns True when >= 80% of entries on each side match a counterpart,
    and the unmatched count is <= 2 on each side. Returns False if new
    messages introduce Never or @_ types (inference regression signals).
    """
    if not project.added or not project.removed:
        return False

    if _has_inference_failure_markers(project.added):
        return False

    added_keys = [(e.file_path, e.line_number, e.error_kind) for e in project.added]
    removed_keys = [(e.file_path, e.line_number, e.error_kind) for e in project.removed]

    # Count how many added entries have a matching removed entry
    removed_multiset: dict[tuple[str, int, str], int] = {}
    for k in removed_keys:
        removed_multiset[k] = removed_multiset.get(k, 0) + 1

    matched_added = 0
    remaining = dict(removed_multiset)
    for k in added_keys:
        if remaining.get(k, 0) > 0:
            matched_added += 1
            remaining[k] -= 1

    matched_removed = sum(
        removed_multiset[k] - remaining.get(k, 0) for k in removed_multiset
    )

    unmatched_added = len(project.added) - matched_added
    unmatched_removed = len(project.removed) - matched_removed

    if unmatched_added > 2 or unmatched_removed > 2:
        return False

    # At least 80% of each side must be matched
    added_ratio = matched_added / len(project.added) if project.added else 0
    removed_ratio = matched_removed / len(project.removed) if project.removed else 0
    return added_ratio >= 0.8 and removed_ratio >= 0.8


def _is_stubs_project(project: ProjectDiff) -> bool:
    """Check if this is a type stubs project (e.g. django-stubs, boto3-stubs)."""
    return project.name.endswith("-stubs")


_CATEGORY_THRESHOLD = 5  # Use categories instead of individual errors above this


def _extract_class_name(message: str) -> Optional[str]:
    """Extract the class name from an error message like 'Object of class `Foo`...'."""
    m = re.search(r"class `([^`]+)`", message)
    return m.group(1) if m else None


def _categorize_errors(
    entries: list[ErrorEntry], prefix: str, cross_check: bool = False
) -> str:
    """Group errors by (error_kind, class_name) and produce a category summary.

    Instead of listing 131 individual errors, output something like:
      missing-attribute on `Commit`: 45 errors across 8 files
        e.g. "Object of class `Commit` has no attribute `tree`"
        Files: dulwich/diff.py, dulwich/repo.py, dulwich/worktree.py, ...
    """
    # Group by (error_kind, class_name)
    groups: dict[tuple[str, str], list[ErrorEntry]] = defaultdict(list)
    for entry in entries:
        class_name = _extract_class_name(entry.message) or "unknown"
        groups[(entry.error_kind, class_name)].append(entry)

    # Sort groups by count (largest first)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

    lines = []
    for (kind, cls), group_entries in sorted_groups:
        files = sorted({e.file_path for e in group_entries})
        file_list = ", ".join(files[:4])
        if len(files) > 4:
            file_list += f", ... ({len(files)} files total)"

        # For reveal-type, show the actual revealed types so the LLM can
        # assess whether type resolution improved or degraded.
        if kind == "reveal-type":
            types_seen = sorted(
                {
                    m.group(1)
                    for e in group_entries
                    if (m := re.search(r"revealed type: (.+?)(?:\s*\[|$)", e.message))
                }
            )
            types_str = ", ".join(types_seen[:8])
            if len(types_seen) > 8:
                types_str += f", ... ({len(types_seen)} total)"
            line = f"{prefix} [{kind}]: {len(group_entries)} occurrence(s)"
            line += f"\n    Types revealed: {types_str}"
            line += f"\n    Files: {file_list}"
        else:
            attrs = sorted(
                {
                    m.group(1)
                    for e in group_entries
                    if (m := re.search(r"no attribute `([^`]+)`", e.message))
                }
            )
            example = group_entries[0].message
            line = f"{prefix} [{kind}] on `{cls}`: {len(group_entries)} error(s)"
            line += f"\n    Example: {example}"
            if attrs:
                attr_str = ", ".join(f"`{a}`" for a in attrs[:6])
                if len(attrs) > 6:
                    attr_str += f", ... ({len(attrs)} total)"
                line += f"\n    Attributes: {attr_str}"
            line += f"\n    Files: {file_list}"
        # Add cross-check stats for added errors
        if cross_check and prefix == "+":
            line += "\n" + _cross_check_stats(group_entries)
        lines.append(line)

    return "\n".join(lines)


def _cross_check_counts(
    entries: list[ErrorEntry],
) -> tuple[int, int, int, int]:
    """Aggregate cross-check counts for a list of entries.

    Returns (total_checked, in_mypy, in_pyright, pyrefly_only).
    Only counts entries where cross_checked is True.
    """
    checked = [e for e in entries if e.cross_checked]
    total = len(checked)
    in_mypy = sum(1 for e in checked if e.also_in_mypy)
    in_pyright = sum(1 for e in checked if e.also_in_pyright)
    pyrefly_only = sum(
        1 for e in checked if not e.also_in_mypy and not e.also_in_pyright
    )
    return total, in_mypy, in_pyright, pyrefly_only


def _cross_check_tag(entry: ErrorEntry) -> str:
    """Build a cross-check annotation tag for an error entry."""
    if not entry.cross_checked:
        return ""  # not checked — don't annotate
    if not entry.also_in_mypy and not entry.also_in_pyright:
        return " [pyrefly-only]"
    parts = []
    parts.append("mypy: yes" if entry.also_in_mypy else "mypy: no")
    parts.append("pyright: yes" if entry.also_in_pyright else "pyright: no")
    return f" [{', '.join(parts)}]"


def _cross_check_stats(entries: list[ErrorEntry]) -> str:
    """Build aggregate cross-check stats for a group of errors."""
    total, in_mypy, in_pyright, _ = _cross_check_counts(entries)
    if total == 0:
        return ""
    return f"    Cross-check: {in_mypy}/{total} also in mypy, {in_pyright}/{total} also in pyright"


def _format_errors_for_llm(
    project: ProjectDiff, cross_check: bool = False
) -> str:
    """Format errors for the LLM prompt.

    For projects with <= CATEGORY_THRESHOLD errors, list each individually.
    For larger projects, group by error category to give the LLM a
    higher-level view of the pattern instead of hundreds of lines.
    """
    total = len(project.added) + len(project.removed)

    if total <= _CATEGORY_THRESHOLD:
        # Small project: list individually
        lines = []
        for entry in project.added:
            tag = _cross_check_tag(entry) if cross_check else ""
            lines.append(f"+ {entry.raw_line}{tag}")
        for entry in project.removed:
            lines.append(f"- {entry.raw_line}")
        return "\n".join(lines)

    # Large project: categorize
    parts = [
        f"Error summary ({len(project.added)} added, {len(project.removed)} removed):"
    ]
    parts.append("")

    # For reveal-type changes, note the type transitions factually
    added_reveal = [e for e in project.added if e.error_kind == "reveal-type"]
    removed_reveal = [e for e in project.removed if e.error_kind == "reveal-type"]
    if added_reveal and removed_reveal:
        removed_types = {
            m.group(1)
            for e in removed_reveal
            if (m := re.search(r"revealed type: (.+?)(?:\s*\[|$)", e.message))
        }
        added_types = {
            m.group(1)
            for e in added_reveal
            if (m := re.search(r"revealed type: (.+?)(?:\s*\[|$)", e.message))
        }
        if "@_" in removed_types and "@_" not in added_types:
            parts.append(
                f"reveal_type changed from @_ (unknown/unresolved) to concrete types "
                f"({', '.join(sorted(added_types)[:5])}).\n"
            )

    if project.added:
        parts.append("NEW errors (added on PR branch):")
        parts.append(_categorize_errors(project.added, "+", cross_check))
    if project.removed:
        if project.added:
            parts.append("")
        parts.append("REMOVED errors (no longer reported):")
        parts.append(_categorize_errors(project.removed, "-"))

    return "\n".join(parts)


def _truncate_source_context(
    source_context: Optional[str],
    errors_text: str,
    pyrefly_diff_len: int = 0,
) -> Optional[str]:
    """Truncate source context if the combined prompt would exceed API limits.

    The errors text and pyrefly diff are kept intact (they're essential).
    Source context is progressively truncated if needed.
    """
    if source_context is None:
        return None

    # Budget for source context = total limit - errors - diff - system prompt overhead
    overhead_chars = 4000 * _CHARS_PER_TOKEN  # system prompt + formatting
    available_chars = (
        _MAX_PROMPT_CHARS - len(errors_text) - pyrefly_diff_len - overhead_chars
    )

    if available_chars <= 0:
        print(
            f"Warning: errors text alone is ~{len(errors_text) // _CHARS_PER_TOKEN} tokens, "
            "skipping source context to stay within API limits.",
            file=sys.stderr,
        )
        return None

    if len(source_context) <= available_chars:
        return source_context

    # Truncate to budget, cutting at a line boundary when possible
    truncated = source_context[:available_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > available_chars // 2:
        truncated = truncated[:last_newline]

    print(
        f"Warning: truncated source context from ~{len(source_context) // _CHARS_PER_TOKEN} "
        f"to ~{len(truncated) // _CHARS_PER_TOKEN} tokens to stay within API limits.",
        file=sys.stderr,
    )
    return truncated + "\n\n[... source context truncated due to token limit ...]"


def _determine_change_type(project: ProjectDiff) -> str:
    """Describe the type of change for the LLM prompt."""
    if project.added and not project.removed:
        return "additions only (new errors on PR branch)"
    elif project.removed and not project.added:
        return "removals only (errors fixed on PR branch)"
    else:
        return "mixed (some errors added, some removed)"


def _compute_structural_signals(
    project: ProjectDiff, cross_check: bool = False
) -> str:
    """Compute structural signals about the project and errors for the LLM.

    Returns a string of signals to append to the user prompt, helping the
    LLM make better decisions about test files, stubs projects, and inference
    failures.
    """
    signals: list[str] = []

    # Cross-check signal: summarize how many errors are also in mypy/pyright
    if cross_check and project.added:
        total, in_mypy, in_pyright, pyrefly_only = _cross_check_counts(
            project.added
        )
        if total > 0:
            signals.append(
                f"STRUCTURAL SIGNAL — CROSS-CHECK: {in_mypy}/{total} new errors also reported by mypy, "
                f"{in_pyright}/{total} by pyright, {pyrefly_only}/{total} pyrefly-only. "
                "Pyrefly-only errors are more likely regressions."
            )

    # Stubs project flag
    if _is_stubs_project(project):
        signals.append(
            "STRUCTURAL SIGNAL: This is a type STUBS project (provides type annotations "
            "for another library). New errors here are almost always regressions — the stubs "
            "are extensively tested against mypy/pyright."
        )

    # Test file ratio — advisory only, not deterministic.
    # Test code CAN have real type issues (e.g., wrong fixture types),
    # so this is a soft signal, not a hard rule.
    all_entries = project.added + project.removed
    test_entries = [
        e
        for e in all_entries
        if any(
            p in e.file_path
            for p in (
                "/tests/",
                "/test_",
                "_test.py",
                "conftest.py",
                "/testing/",
            )
        )
    ]
    if (
        test_entries
        and len(test_entries) == len(all_entries)
        and len(all_entries) >= 10
    ):
        signals.append(
            "STRUCTURAL SIGNAL: All errors are in test files. Consider whether these are "
            "genuine type issues (e.g., wrong fixture return types, missing attributes on real objects) "
            "or false positives from patterns the type checker can't model (mocking, parametrize, etc.)."
        )

    # Never/@_ inference failure detection — only count Never and @_ as strong
    # signals. Unknown on its own is ambiguous: it can indicate untyped code
    # that is *correctly* flagged as incompatible with a typed API.
    never_count = sum(
        1
        for e in project.added
        if any(marker in e.message for marker in ("Never", "@_"))
    )
    if never_count > 0:
        signals.append(
            f"STRUCTURAL SIGNAL: {never_count}/{len(project.added)} new error(s) contain "
            "`Never` or `@_` types — these strongly suggest type inference failures, not real bugs."
        )

    return "\n".join(signals)


def _get_best_source_context(
    project: ProjectDiff,
    fetch_code: bool,
) -> Optional[str]:
    """Fetch and combine source context for the most important errors.

    Fetches context for up to 5 error locations to keep the LLM prompt
    manageable while providing enough context for accurate classification.
    """
    if not fetch_code:
        return None

    # Prioritize added errors (new problems), then removed
    entries_to_fetch: list[tuple[str, ErrorEntry]] = []
    for entry in project.added[:3]:
        entries_to_fetch.append(("+", entry))
    for entry in project.removed[:2]:
        entries_to_fetch.append(("-", entry))

    contexts: list[str] = []
    for prefix, entry in entries_to_fetch:
        ctx = fetch_source_context(project, entry)
        if ctx:
            contexts.append(
                f"--- {prefix} {entry.file_path}:{entry.location} [{entry.error_kind}] ---\n"
                f"{ctx.snippet}"
            )

    return "\n\n".join(contexts) if contexts else None


def _truncate_pyrefly_diff(pyrefly_diff: str, errors_text: str) -> str:
    """Truncate the pyrefly PR diff if it would exceed API limits.

    The PR diff gets priority over source context but must still fit
    within the overall token budget alongside the errors text.
    """
    overhead_chars = 4000 * _CHARS_PER_TOKEN  # system prompt + formatting
    # Leave room for at least some source context
    max_diff_chars = _MAX_PROMPT_CHARS - len(errors_text) - overhead_chars
    # Reserve up to half the remaining budget for source context
    max_diff_chars = min(max_diff_chars, (_MAX_PROMPT_CHARS - overhead_chars) // 2)

    if max_diff_chars <= 0 or len(pyrefly_diff) <= max_diff_chars:
        return pyrefly_diff

    truncated = pyrefly_diff[:max_diff_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_diff_chars // 2:
        truncated = truncated[:last_newline]

    print(
        f"Warning: truncated pyrefly PR diff from ~{len(pyrefly_diff) // _CHARS_PER_TOKEN} "
        f"to ~{len(truncated) // _CHARS_PER_TOKEN} tokens to stay within API limits.",
        file=sys.stderr,
    )
    return truncated + "\n\n[... pyrefly PR diff truncated due to token limit ...]"


def classify_project(
    project: ProjectDiff,
    fetch_code: bool = True,
    use_llm: bool = True,
    model: Optional[str] = None,
    pyrefly_diff: Optional[str] = None,
    pr_description: Optional[str] = None,
    cross_check: bool = False,
) -> Classification:
    """Classify a single project's changes.

    Applies heuristics first. If the case is non-trivial and use_llm is True,
    fetches source code and delegates to the LLM. When pyrefly_diff is provided,
    it is included in each LLM call for PR attribution.
    """
    # Build cross-check stats string if cross-checking was enabled
    cc_stats = ""
    if cross_check and project.added:
        total, in_mypy, in_pyright, pyrefly_only = _cross_check_counts(
            project.added
        )
        if total > 0:
            cc_stats = (
                f"{in_mypy}/{total} mypy, {in_pyright}/{total} pyright, "
                f"{pyrefly_only}/{total} pyrefly-only"
            )

    base = Classification(
        project_name=project.name,
        verdict="ambiguous",
        reason="",
        added_count=len(project.added),
        removed_count=len(project.removed),
        error_kinds=sorted(set(
            e.error_kind
            for e in (*project.added, *project.removed)
            if e.error_kind
        )),
        cross_check_stats=cc_stats,
    )

    # Heuristic 1: All additions are internal-error → regression
    if _is_all_internal_errors(project):
        base.verdict = "regression"
        base.reason = (
            f"Added {len(project.added)} internal-error(s) — "
            "indicates a pyrefly bug or panic."
        )
        base.method = "heuristic"
        return base

    # Heuristic 2: Wording change (same file:line:kind, different message)
    if _is_wording_change(project):
        base.verdict = "neutral"
        base.reason = (
            "Same errors at same locations with same error kinds — "
            "message wording changed, no behavioral impact."
        )
        base.method = "heuristic"
        return base

    # Heuristic 3: Near-wording change (>= 80% matched locations, <= 2 unmatched)
    if _is_near_wording_change(project):
        base.verdict = "neutral"
        base.reason = (
            "Most errors at same locations with same error kinds — "
            "message wording changed with minor residual noise, no significant behavioral impact."
        )
        base.method = "heuristic"
        return base

    # Non-trivial case: use LLM if available
    if not use_llm:
        base.verdict = "ambiguous"
        base.reason = (
            f"Non-trivial change ({len(project.added)} added, "
            f"{len(project.removed)} removed). LLM classification not available."
        )
        base.method = "heuristic"
        return base

    errors_text = _format_errors_for_llm(project, cross_check)
    source_context = _get_best_source_context(project, fetch_code)

    # Truncate pyrefly diff if needed, then account for its size in source budget
    truncated_diff = None
    diff_len = 0
    if pyrefly_diff:
        truncated_diff = _truncate_pyrefly_diff(pyrefly_diff, errors_text)
        diff_len = len(truncated_diff)

    source_context = _truncate_source_context(source_context, errors_text, diff_len)
    change_type = _determine_change_type(project)
    structural_signals = _compute_structural_signals(project, cross_check)

    try:
        llm_result = classify_with_llm(
            errors_text=errors_text,
            source_context=source_context,
            change_type=change_type,
            model=model,
            structural_signals=structural_signals,
            pyrefly_diff=truncated_diff,
            pr_description=pr_description,
        )

        # Multi-pass: if the LLM requests additional files, fetch and retry
        # Allow up to 2 extra rounds (3 total) so the LLM can chase
        # parent class → type definition chains.
        combined_context = source_context or ""
        for _pass in range(2):
            if not llm_result.needs_files or not fetch_code:
                break
            print(
                f"  LLM requested files: {llm_result.needs_files}",
                file=sys.stderr,
            )
            extra_context = fetch_files_by_path(project, llm_result.needs_files)
            if not extra_context:
                break
            if combined_context:
                combined_context += "\n\n"
            combined_context += extra_context
            truncated = _truncate_source_context(
                combined_context, errors_text, diff_len
            )
            llm_result = classify_with_llm(
                errors_text=errors_text,
                source_context=truncated,
                change_type=change_type,
                model=model,
                structural_signals=structural_signals,
                pyrefly_diff=truncated_diff,
                pr_description=pr_description,
            )

        # If the LLM still wants files after all passes, give up
        if llm_result.needs_files:
            base.verdict = "ambiguous"
            base.reason = (
                f"LLM requested additional files that could not be resolved. "
                f"Non-trivial change ({len(project.added)} added, "
                f"{len(project.removed)} removed)."
            )
            base.method = "llm"
            return base

        # Pass 1.5: self-critique the reasoning for factual errors
        try:
            critiqued_reason, critiqued_categories = critique_reasoning(
                llm_result.reason,
                llm_result.categories,
                errors_text,
                source_context,
                model,
            )
        except LLMError as e:
            print(
                f"  Warning: self-critique failed for {project.name}: {e}, "
                "using original reasoning",
                file=sys.stderr,
            )
            critiqued_reason = llm_result.reason
            critiqued_categories = llm_result.categories

        # Pass 2: assign verdict based on (critiqued) reasoning
        verdict, categories_with_verdicts = assign_verdict_with_llm(
            critiqued_reason, critiqued_categories, model
        )

        base.verdict = verdict
        base.reason = critiqued_reason
        base.method = "llm"
        base.categories = categories_with_verdicts
        base.pr_attribution = llm_result.pr_attribution
    except LLMError as e:
        print(
            f"Warning: LLM classification failed for {project.name}: {e}",
            file=sys.stderr,
        )
        base.verdict = "ambiguous"
        base.reason = (
            f"LLM classification failed: {e}. "
            f"Non-trivial change ({len(project.added)} added, "
            f"{len(project.removed)} removed)."
        )
        base.method = "heuristic"

    return base


def _enforce_cross_project_consistency(
    classifications: list[Classification],
) -> None:
    """Enforce verdict consistency across projects that share error kinds.

    When multiple LLM-classified projects share the same error kind(s) and
    have conflicting verdicts, the majority verdict wins. This prevents the
    classifier from saying "overload resolution improved" for one project
    and "overload resolution regressed" for another with the same pattern.

    Modifies classifications in place.
    """
    # Only consider LLM-classified projects with clear verdicts
    llm_classified = [
        c for c in classifications
        if c.method == "llm" and c.verdict in ("regression", "improvement")
    ]
    if len(llm_classified) < 2:
        return

    # Group projects by their error kinds (using frozenset for hashability)
    kind_to_projects: dict[str, list[Classification]] = defaultdict(list)
    for c in llm_classified:
        for kind in c.error_kinds:
            kind_to_projects[kind].append(c)

    # For each error kind shared by multiple projects, check consistency
    already_adjusted: set[str] = set()
    for kind, group in kind_to_projects.items():
        if len(group) < 2:
            continue

        verdicts = [c.verdict for c in group]
        if len(set(verdicts)) <= 1:
            continue  # already consistent

        # Count verdicts
        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        majority = max(verdict_counts, key=lambda v: verdict_counts[v])
        minority_count = sum(
            c for v, c in verdict_counts.items() if v != majority
        )

        # Only enforce if majority is clear (> minority)
        if verdict_counts[majority] <= minority_count:
            continue

        # Update minority projects to match majority
        adjusted_names = []
        for c in group:
            if c.verdict != majority and c.project_name not in already_adjusted:
                old = c.verdict
                c.verdict = majority
                adjusted_names.append(c.project_name)
                already_adjusted.add(c.project_name)

        if adjusted_names:
            print(
                f"  Cross-project consistency [{kind}]: "
                f"{', '.join(adjusted_names)} adjusted to {majority} "
                f"(vote: {verdict_counts})",
                file=sys.stderr,
            )


def classify_all(
    projects: list[ProjectDiff],
    fetch_code: bool = True,
    use_llm: bool = True,
    model: Optional[str] = None,
    pyrefly_diff: Optional[str] = None,
    generate_suggestion: bool = False,
    pr_description: Optional[str] = None,
    cross_check: bool = False,
) -> ClassificationResult:
    """Classify all projects in a primer diff.

    Returns a ClassificationResult with per-project classifications
    and summary counts. When pyrefly_diff is provided, it is passed
    to each per-project LLM call for PR attribution.

    When cross_check is True, runs mypy/pyright on each project and uses
    an LLM to match errors semantically before classification.

    When generate_suggestion is True and regressions or ambiguous results
    exist, runs Pass 3 to produce an aggregate source code suggestion.
    """
    result = ClassificationResult(total_projects=len(projects))

    # Cross-check: run mypy/pyright and annotate errors before classifying
    if cross_check:
        print(
            f"Running mypy/pyright cross-check on {len(projects)} project(s)...",
            file=sys.stderr,
        )
        cross_check_projects(projects, model=model)

    for project in projects:
        classification = classify_project(
            project,
            fetch_code=fetch_code,
            use_llm=use_llm,
            model=model,
            pyrefly_diff=pyrefly_diff,
            pr_description=pr_description,
            cross_check=cross_check,
        )
        result.classifications.append(classification)

    # Enforce cross-project consistency before counting verdicts
    _enforce_cross_project_consistency(result.classifications)

    # Count verdicts after consistency enforcement
    for classification in result.classifications:
        if classification.verdict == "regression":
            result.regressions += 1
        elif classification.verdict == "improvement":
            result.improvements += 1
        elif classification.verdict == "neutral":
            result.neutrals += 1
        else:
            result.ambiguous += 1

    # Pass 3: aggregate suggestion generation
    if generate_suggestion and use_llm and (result.regressions > 0 or result.ambiguous > 0):
        try:
            suggestion_result = generate_suggestions(
                result, pyrefly_diff or "", model
            )
            result.suggestion = suggestion_result
        except Exception as e:
            print(
                f"Warning: suggestion generation failed: {e}",
                file=sys.stderr,
            )

    return result
