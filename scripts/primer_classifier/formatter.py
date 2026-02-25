# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render classification results as GitHub markdown or JSON."""

from __future__ import annotations

import json
import re

from .classifier import ClassificationResult

_GITHUB_BASE_URL = "https://github.com/facebook/pyrefly/blob/main/"

_VERDICT_EMOJI = {
    "regression": "\u274c",  # red X
    "improvement": "\u2705",  # green check
    "neutral": "\u2796",  # heavy minus
    "ambiguous": "\u2753",  # question mark
}

_VERDICT_LABEL = {
    "regression": "Regression",
    "improvement": "Improvement",
    "neutral": "Neutral",
    "ambiguous": "Needs Review",
}

# Matches paths that look like pyrefly source files (ending in .rs, .py, .toml, etc.)
_SOURCE_FILE_PATTERN = re.compile(
    r"(?:pyrefly/|crates/|lsp/)[\w/]+\.\w+"
)

# Matches function-name patterns like `word_word()` in text
_FUNCTION_NAME_PATTERN = re.compile(r"(\w+)\(\)")


def _linkify_file(path: str) -> str:
    """Convert a pyrefly source file path to a GitHub markdown link.

    Turns 'pyrefly/lib/alt/class/variance.rs' into
    '[`pyrefly/lib/alt/class/variance.rs`](https://github.com/...)'
    """
    return f"[`{path}`]({_GITHUB_BASE_URL}{path})"


def _extract_file_from_text(text: str) -> str | None:
    """Find the first pyrefly source file path in a string.

    Used to associate function names with their containing file when
    both appear in the same text.
    """
    m = _SOURCE_FILE_PATTERN.search(text)
    return m.group(0) if m else None


def _linkify_function_in_text(text: str, file_path: str | None = None) -> str:
    """Replace function-name patterns like `func()` with GitHub-linked markdown.

    If file_path is provided, links the function name to that file's GitHub URL.
    If not, attempts to extract a file path from the surrounding text first.
    """
    resolved_file = file_path or _extract_file_from_text(text)

    def replacer(match: re.Match) -> str:
        func_name = match.group(1)
        # Skip common non-function words that happen to precede ()
        if func_name in ("e", "i", "s", "x", "a", "the"):
            return match.group(0)
        if resolved_file:
            url = f"{_GITHUB_BASE_URL}{resolved_file}"
            return f"[`{func_name}()`]({url})"
        return f"`{func_name}()`"

    return _FUNCTION_NAME_PATTERN.sub(replacer, text)


def _linkify_files_in_text(text: str) -> str:
    """Replace pyrefly source file paths and function names in text with GitHub links.

    Linkifies paths that look like pyrefly source files (starting with
    pyrefly/, crates/, or lsp/). Also linkifies function names when a file
    path is found in the same text. Avoids double-linkifying paths already
    inside markdown links.
    """
    file_path = _extract_file_from_text(text)

    def replacer(match: re.Match) -> str:
        path = match.group(0)
        # Check if already inside a markdown link [...](...)
        start = match.start()
        preceding = text[:start]
        if preceding.endswith("`") or preceding.endswith("("):
            return path
        return _linkify_file(path)

    result = _SOURCE_FILE_PATTERN.sub(replacer, text)
    result = _linkify_function_in_text(result, file_path)
    return result


def _extract_root_cause(c) -> str:
    """Extract a linkified root cause string from a classification's pr_attribution.

    Looks for function names and file paths in the attribution text. If found,
    returns the function name linked to the file's GitHub URL. Otherwise returns
    a truncated attribution string.
    """
    attr = c.pr_attribution
    if not attr or attr == "N/A":
        return ""
    file_path = _extract_file_from_text(attr)
    func_match = _FUNCTION_NAME_PATTERN.search(attr)
    if func_match and file_path:
        func_name = func_match.group(1)
        url = f"{_GITHUB_BASE_URL}{file_path}"
        return f"[`{func_name}()`]({url})"
    if func_match:
        return f"`{func_match.group(1)}()`"
    # Truncate long attribution text for the table
    if len(attr) > 60:
        return attr[:57] + "..."
    return attr


def _extract_error_kind(c) -> str:
    """Extract the primary error kind from a classification's categories."""
    if c.categories:
        return f"`{c.categories[0].category}`"
    return ""


def format_markdown(result: ClassificationResult) -> str:
    """Render classification results as a GitHub PR comment in markdown.

    Layout: summary line, overview table, collapsible detailed analysis,
    suggested fixes, and a footer.
    """
    if not result.classifications:
        return "## Primer Diff Classification\n\n" "No diffs to classify. All clear."

    lines: list[str] = []
    lines.append("## Primer Diff Classification\n")

    # Summary line
    parts = []
    if result.regressions:
        parts.append(
            f"{_VERDICT_EMOJI['regression']} {result.regressions} regression(s)"
        )
    if result.improvements:
        parts.append(
            f"{_VERDICT_EMOJI['improvement']} {result.improvements} improvement(s)"
        )
    if result.neutrals:
        parts.append(f"{_VERDICT_EMOJI['neutral']} {result.neutrals} neutral")
    if result.ambiguous:
        parts.append(f"{_VERDICT_EMOJI['ambiguous']} {result.ambiguous} needs review")
    lines.append(" | ".join(parts) + f" | {result.total_projects} project(s) total\n")

    # Overview table
    lines.append("| Project | Verdict | Changes | Error Kind | Root Cause |")
    lines.append("|---------|---------|---------|------------|------------|")
    for c in result.classifications:
        emoji = _VERDICT_EMOJI.get(c.verdict, "")
        label = _VERDICT_LABEL.get(c.verdict, c.verdict)
        change_summary = _format_change_counts(c.added_count, c.removed_count)
        error_kind = _extract_error_kind(c)
        root_cause = _extract_root_cause(c)
        lines.append(
            f"| {c.project_name} | {emoji} {label} | {change_summary} "
            f"| {error_kind} | {root_cause} |"
        )
    lines.append("")

    # Collapsible detailed analysis
    lines.append("<details>")
    lines.append("<summary>Detailed analysis</summary>\n")

    for verdict in ("regression", "improvement", "neutral", "ambiguous"):
        group = [c for c in result.classifications if c.verdict == verdict]
        if not group:
            continue

        emoji = _VERDICT_EMOJI[verdict]
        label = _VERDICT_LABEL[verdict]
        lines.append(f"#### {emoji} {label} ({len(group)})\n")

        for c in group:
            change_summary = _format_change_counts(c.added_count, c.removed_count)
            lines.append(f"**{c.project_name}** ({change_summary})")
            if c.categories:
                for cat in c.categories:
                    cat_emoji = _VERDICT_EMOJI.get(cat.verdict, "")
                    lines.append(f"> {cat_emoji} **{cat.category}**: {cat.reason}")
                if c.reason:
                    lines.append(f">\n> *Overall:* {c.reason}")
                if c.pr_attribution and c.pr_attribution != "N/A":
                    lines.append(
                        f">\n> **Attribution:** "
                        f"{_linkify_files_in_text(c.pr_attribution)}"
                    )
                lines.append("")
            else:
                lines.append(f"> {c.reason}")
                if c.pr_attribution and c.pr_attribution != "N/A":
                    lines.append(
                        f"> **Attribution:** "
                        f"{_linkify_files_in_text(c.pr_attribution)}"
                    )
                lines.append("")

    lines.append("</details>\n")

    # Suggested Fixes (Pass 3)
    if result.suggestion and result.suggestion.suggestions:
        lines.append("### Suggested Fix\n")
        lines.append(f"**Summary:** {result.suggestion.summary}\n")
        for i, s in enumerate(result.suggestion.suggestions, 1):
            lines.append(f"**{i}. {_linkify_files_in_text(s.description)}**")
            if s.files:
                linked_files = ", ".join(_linkify_file(f) for f in s.files)
                lines.append(f"> Files: {linked_files}")
            lines.append(f"> Confidence: {s.confidence}")
            if s.affected_projects:
                lines.append(
                    f"> Affected projects: {', '.join(s.affected_projects)}"
                )
            if s.error_kinds_fixed:
                kinds_str = ", ".join(f"`{k}`" for k in s.error_kinds_fixed)
                lines.append(f"> Fixes: {kinds_str}")
            lines.append(f"> {_linkify_files_in_text(s.reasoning)}")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        "<sub>Classification by primer-classifier" f" ({_method_summary(result)})</sub>"
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
                "pr_attribution": c.pr_attribution,
                "categories": (
                    [
                        {
                            "category": cat.category,
                            "verdict": cat.verdict,
                            "reason": cat.reason,
                        }
                        for cat in c.categories
                    ]
                    if c.categories
                    else []
                ),
            }
            for c in result.classifications
        ],
    }
    if result.suggestion is not None:
        data["suggestion"] = {
            "summary": result.suggestion.summary,
            "has_regressions": result.suggestion.has_regressions,
            "suggestions": [
                {
                    "description": s.description,
                    "files": s.files,
                    "file_urls": [
                        f"{_GITHUB_BASE_URL}{f}" for f in s.files
                    ],
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                    "affected_projects": s.affected_projects,
                    "error_kinds_fixed": s.error_kinds_fixed,
                }
                for s in result.suggestion.suggestions
            ],
        }
    return json.dumps(data, indent=2)
