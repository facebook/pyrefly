# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestrate classification of primer diff entries.

Applies heuristics for trivially obvious cases, fetches source code,
then delegates to the LLM for everything else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from .code_fetcher import fetch_source_context
from .llm_client import LLMError, classify_with_llm
from .parser import ErrorEntry, ProjectDiff


@dataclass
class Classification:
    """Classification result for a single project."""

    project_name: str
    verdict: str  # "regression", "improvement", "neutral", "ambiguous"
    reason: str
    added_count: int = 0
    removed_count: int = 0
    method: str = "heuristic"  # "heuristic" or "llm"


@dataclass
class ClassificationResult:
    """Full classification result across all projects."""

    classifications: list[Classification] = field(default_factory=list)
    total_projects: int = 0
    regressions: int = 0
    improvements: int = 0
    neutrals: int = 0
    ambiguous: int = 0


def _is_all_removals(project: ProjectDiff) -> bool:
    """Check if a project has only removed errors (no additions)."""
    return len(project.removed) > 0 and len(project.added) == 0


def _is_all_internal_errors(project: ProjectDiff) -> bool:
    """Check if all added errors are internal-error (pyrefly bug/panic)."""
    return (
        len(project.added) > 0
        and all(e.error_kind == "internal-error" for e in project.added)
    )


def _is_wording_change(project: ProjectDiff) -> bool:
    """Check if changes are just wording changes at identical locations.

    A wording change means: for every added error, there's a removed error
    at the same file:line with the same error kind, and vice versa. The
    message text differs, but the error kind and location are the same.
    """
    if not project.added or not project.removed:
        return False
    if len(project.added) != len(project.removed):
        return False

    # Build sets of (file, line, error_kind) for added and removed
    added_keys = {(e.file_path, e.line_number, e.error_kind) for e in project.added}
    removed_keys = {(e.file_path, e.line_number, e.error_kind) for e in project.removed}
    return added_keys == removed_keys


def _format_errors_for_llm(project: ProjectDiff) -> str:
    """Format all errors in a project for the LLM prompt."""
    lines = []
    for entry in project.added:
        lines.append(f"+ {entry.raw_line}")
    for entry in project.removed:
        lines.append(f"- {entry.raw_line}")
    return "\n".join(lines)


def _determine_change_type(project: ProjectDiff) -> str:
    """Describe the type of change for the LLM prompt."""
    if project.added and not project.removed:
        return "additions only (new errors on PR branch)"
    elif project.removed and not project.added:
        return "removals only (errors fixed on PR branch)"
    else:
        return "mixed (some errors added, some removed)"


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


def classify_project(
    project: ProjectDiff,
    fetch_code: bool = True,
    use_llm: bool = True,
) -> Classification:
    """Classify a single project's changes.

    Applies heuristics first. If the case is non-trivial and use_llm is True,
    fetches source code and delegates to the LLM.
    """
    base = Classification(
        project_name=project.name,
        verdict="ambiguous",
        reason="",
        added_count=len(project.added),
        removed_count=len(project.removed),
    )

    # Heuristic 1: All removals → improvement (removed false positives)
    if _is_all_removals(project):
        base.verdict = "improvement"
        base.reason = (
            f"Removed {len(project.removed)} error(s) — "
            "no new errors introduced. Likely removed false positives."
        )
        base.method = "heuristic"
        return base

    # Heuristic 2: All additions are internal-error → regression
    if _is_all_internal_errors(project):
        base.verdict = "regression"
        base.reason = (
            f"Added {len(project.added)} internal-error(s) — "
            "indicates a pyrefly bug or panic."
        )
        base.method = "heuristic"
        return base

    # Heuristic 3: Wording change (same file:line:kind, different message)
    if _is_wording_change(project):
        base.verdict = "neutral"
        base.reason = (
            f"Same errors at same locations with same error kinds — "
            "message wording changed, no behavioral impact."
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

    errors_text = _format_errors_for_llm(project)
    source_context = _get_best_source_context(project, fetch_code)
    change_type = _determine_change_type(project)

    try:
        llm_result = classify_with_llm(
            errors_text=errors_text,
            source_context=source_context,
            change_type=change_type,
        )
        base.verdict = llm_result.verdict
        base.reason = llm_result.reason
        base.method = "llm"
    except LLMError as e:
        print(f"Warning: LLM classification failed for {project.name}: {e}", file=sys.stderr)
        base.verdict = "ambiguous"
        base.reason = (
            f"LLM classification failed: {e}. "
            f"Non-trivial change ({len(project.added)} added, "
            f"{len(project.removed)} removed)."
        )
        base.method = "heuristic"

    return base


def classify_all(
    projects: list[ProjectDiff],
    fetch_code: bool = True,
    use_llm: bool = True,
) -> ClassificationResult:
    """Classify all projects in a primer diff.

    Returns a ClassificationResult with per-project classifications
    and summary counts.
    """
    result = ClassificationResult(total_projects=len(projects))

    for project in projects:
        classification = classify_project(
            project,
            fetch_code=fetch_code,
            use_llm=use_llm,
        )
        result.classifications.append(classification)

        if classification.verdict == "regression":
            result.regressions += 1
        elif classification.verdict == "improvement":
            result.improvements += 1
        elif classification.verdict == "neutral":
            result.neutrals += 1
        else:
            result.ambiguous += 1

    return result
