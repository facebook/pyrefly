# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render classification results as GitHub markdown or JSON."""

from __future__ import annotations

import json

from .classifier import ClassificationResult


_VERDICT_EMOJI = {
    "regression": "\u274c",    # red X
    "improvement": "\u2705",   # green check
    "neutral": "\u2796",       # heavy minus
    "ambiguous": "\u2753",     # question mark
}

_VERDICT_LABEL = {
    "regression": "Regression",
    "improvement": "Improvement",
    "neutral": "Neutral",
    "ambiguous": "Needs Review",
}


def format_markdown(result: ClassificationResult) -> str:
    """Render classification results as a GitHub PR comment in markdown."""
    if not result.classifications:
        return (
            "## Primer Diff Classification\n\n"
            "No diffs to classify. All clear."
        )

    lines: list[str] = []
    lines.append("## Primer Diff Classification\n")

    # Summary line
    parts = []
    if result.regressions:
        parts.append(f"{_VERDICT_EMOJI['regression']} {result.regressions} regression(s)")
    if result.improvements:
        parts.append(f"{_VERDICT_EMOJI['improvement']} {result.improvements} improvement(s)")
    if result.neutrals:
        parts.append(f"{_VERDICT_EMOJI['neutral']} {result.neutrals} neutral")
    if result.ambiguous:
        parts.append(f"{_VERDICT_EMOJI['ambiguous']} {result.ambiguous} needs review")
    lines.append(" | ".join(parts) + f" | {result.total_projects} project(s) total\n")

    # Group by verdict for easier scanning
    for verdict in ("regression", "improvement", "neutral", "ambiguous"):
        group = [c for c in result.classifications if c.verdict == verdict]
        if not group:
            continue

        emoji = _VERDICT_EMOJI[verdict]
        label = _VERDICT_LABEL[verdict]
        lines.append(f"### {emoji} {label} ({len(group)})\n")

        for c in group:
            change_summary = _format_change_counts(c.added_count, c.removed_count)
            lines.append(f"**{c.project_name}** ({change_summary})")
            if c.categories:
                for cat in c.categories:
                    cat_emoji = _VERDICT_EMOJI.get(cat.verdict, "")
                    lines.append(f"> {cat_emoji} **{cat.category}**: {cat.reason}")
                if c.reason:
                    lines.append(f">\n> *Overall:* {c.reason}\n")
                else:
                    lines.append("")
            else:
                lines.append(f"> {c.reason}\n")

    # Footer
    lines.append("---")
    lines.append(
        "<sub>Classification by primer-classifier"
        f" ({_method_summary(result)})</sub>"
    )

    return "\n".join(lines)


def _format_change_counts(added: int, removed: int) -> str:
    """Format added/removed counts as a compact string."""
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return ", ".join(parts) if parts else "no changes"


def _method_summary(result: ClassificationResult) -> str:
    """Summarize which classification methods were used."""
    heuristic = sum(1 for c in result.classifications if c.method == "heuristic")
    llm = sum(1 for c in result.classifications if c.method == "llm")
    parts = []
    if heuristic:
        parts.append(f"{heuristic} heuristic")
    if llm:
        parts.append(f"{llm} LLM")
    return ", ".join(parts) if parts else "no classifications"


def format_json(result: ClassificationResult) -> str:
    """Render classification results as JSON."""
    data = {
        "summary": {
            "total_projects": result.total_projects,
            "regressions": result.regressions,
            "improvements": result.improvements,
            "neutrals": result.neutrals,
            "ambiguous": result.ambiguous,
        },
        "classifications": [
            {
                "project": c.project_name,
                "verdict": c.verdict,
                "reason": c.reason,
                "added_count": c.added_count,
                "removed_count": c.removed_count,
                "method": c.method,
                "categories": [
                    {
                        "category": cat.category,
                        "verdict": cat.verdict,
                        "reason": cat.reason,
                    }
                    for cat in c.categories
                ] if c.categories else [],
            }
            for c in result.classifications
        ],
    }
    return json.dumps(data, indent=2)
