#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 4: Priority scoring — weighted score for each issue.

One Sonnet call per issue. Uses all prior pass results plus optional
typing spec context. Scores 0-100 based on weighted signals.

CRITICAL: The LLM is NOT told which issues are in the V1 milestone.
It scores purely based on signals, enabling blind validation.
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json, get_backend, LLMError
from spec_fetcher import get_spec_excerpt

SONNET_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are scoring the priority of GitHub issues for pyrefly, a Python type checker. Score each issue 0-100 based on its impact on user adoption and type checker quality.

SCORING WEIGHTS:

| Weight | Signal | Description |
|--------|--------|-------------|
| Highest (x3) | False positive impact | Issues where pyrefly errors but pyright/mypy don't. These drive users away — if pyrefly flags valid code, people stop using it. High primer frequency amplifies this. |
| Highest (x3) | Performance | Memory usage, type check speed, LSP responsiveness. Performance issues directly block adoption at scale. |
| High (x2) | Team-assigned priority | P0/P1/P2 priorities set by the team in GitHub Projects. These reflect real-world urgency. P0 → score 80+, P1 → score 65+, P2 → score 50+. However, priorities CAN be stale or wrong — if the issue looks trivial or already fixed, override accordingly. Unset priority means no signal (don't penalize). |
| Medium (x1.5) | False negatives + spec compliance | Real type errors pyrefly misses that pyright/mypy catch. Spec compliance gaps (TypeVar, ParamSpec, overloads, narrowing, exhaustiveness). These affect correctness but are less urgent than false positives since users don't notice missing checks. |
| Medium (x1.5) | Actionability | How ready is this issue to be worked on? Score HIGH if: clear minimal repro, root cause identified, well-scoped, a developer could pick it up and start working on it. Score LOW if: vague description, no repro, can't reproduce, unclear root cause, needs more investigation, stale with no resolution. IMPORTANT: Actionability is about clarity and readiness, NOT implementation difficulty. A well-described issue with a clear repro is actionable even if the fix is architecturally complex. Do NOT penalize issues for being hard to implement. |
| High (x2) | IDE and usability | Language server features: hover, completions, go-to-def, diagnostics. Important for adoption — a broken IDE experience drives users away. Score IDE bugs 60-80, IDE features 45-65. |
| Medium (x1.5) | Primer breadth | How many primer projects show this error pattern. Primer counts are matched by error message pattern. Check the pattern specificity assessment: HIGH means this issue likely causes most of those errors, MEDIUM means it is one of several causes, LOW means the pattern is generic with many unrelated root causes. Weight primer counts accordingly. |
| High (x2.5) | Strategic adoption | Issues tagged pytorch or google — these are top-priority adoption targets. Blocking issues for these ecosystems should score 75+. |
| Medium (x1.5) | Ecosystem adoption | Issues tagged pydantic, sqlalchemy, or other framework labels — important but lower priority than strategic targets. |
| Lower (x0.5) | Edge cases | Few projects affected, low engagement, stale issues, niche scenarios. |

CRITICAL: Implementation difficulty or complexity of the fix must NEVER lower an issue's score. A hard-to-fix bug is just as important as an easy-to-fix bug. Score based on user impact and adoption risk, not engineering effort.

BONUS SIGNALS (additive):
- Duplicate/merged count: widespread problem → boost proportionally.
- Reaction count: community demand signal.
- Blocker count: if this issue BLOCKS other issues, boost significantly.

ACTIONABILITY GUIDELINES:
- Has a Python code snippet that reproduces the issue → actionable
- Root cause discussed and narrowed down in comments → actionable
- "I can't share the project" / no repro steps → NOT actionable
- Stale with no resolution or assignment → less actionable
- Active discussion with team members investigating → somewhat actionable
- Marked as question/needs-info → NOT actionable until resolved

IMPORTANT: You are NOT told which issues are in any milestone. Score purely based on the signals above.

Respond with JSON:
{
  "priority_score": 0-100,
  "breakdown": {
    "false_positive_impact": 0-100,
    "performance": 0-100,
    "team_priority": 0-100,
    "correctness": 0-100,
    "actionability": 0-100,
    "ide_usability": 0-100,
    "primer_breadth": 0-100,
    "adoption_impact": 0-100,
    "community_demand": 0-100
  },
  "rationale": "1-2 sentence explanation of the score"
}"""


def score_issue(  # noqa: C901
    issue: dict,
    categorization: dict,
    primer_impact: dict,
    dep_graph: dict,
) -> dict:
    """Score a single issue using Sonnet.

    Returns: {"priority_score": float, "breakdown": dict, "rationale": str}
    """
    backend, _ = get_backend()
    if backend == "none":
        raise LLMError("No API key set")

    num = issue.get("number", 0)
    title = issue.get("title", "")
    body = (issue.get("body", "") or "")[:1000]
    labels = issue.get("labels", [])

    # Gather context from prior passes
    cat = categorization.get("category", "unknown")
    subcat = categorization.get("subcategory", "unknown")
    status = issue.get("status_classification", "unknown")

    p_projects = primer_impact.get("primer_project_count", 0)
    p_errors = primer_impact.get("primer_error_count", 0)
    p_kind = primer_impact.get("matched_kind", "")
    p_template = primer_impact.get("matched_template", "")
    p_specificity = primer_impact.get("pattern_specificity", "")
    p_spec_note = primer_impact.get("specificity_note", "")

    reactions = issue.get("reactions_count", 0)
    comments = issue.get("comments_count", 0)
    sub_issues = len(issue.get("sub_issues", []))
    board_priority = issue.get("priority", "")  # P0, P1, P2, wish, or ""

    # Check if this issue is in any duplicate cluster
    dup_count = 0
    for cluster in dep_graph.get("duplicate_clusters", []):
        if num in cluster.get("duplicates", []) or num == cluster.get("canonical"):
            dup_count = len(cluster.get("duplicates", [])) + 1

    # Check how many issues this one blocks (blockers should be prioritized)
    blocks_count = 0
    for chain in dep_graph.get("blocking_chains", []):
        if chain.get("blocker") == num:
            blocks_count += len(chain.get("blocked", []))

    # Fetch spec excerpt if relevant
    spec_excerpt = ""
    if p_kind:
        spec_excerpt = get_spec_excerpt(p_kind)
    checker = issue.get("checker_results", {})
    if checker and not spec_excerpt:
        for err in checker.get("pyrefly", [])[:1]:
            kind = err.get("kind", "")
            if kind:
                spec_excerpt = get_spec_excerpt(kind)
                break

    user_prompt = (
        f"Issue #{num}: {title}\n"
        f"Labels: {', '.join(labels)}\n"
        f"Board priority: {board_priority or 'not set'}\n"
        f"Category: {cat} / {subcat}\n"
        f"Status: {status}\n"
        f"Reactions: {reactions}, Comments: {comments}, Sub-issues: {sub_issues}\n"
        f"Duplicate count: {dup_count}\n"
        f"Blocks count: {blocks_count} (issues blocked by this one)\n"
        f"Primer impact: {p_projects} projects, {p_errors} errors matching"
        f" pattern: {p_kind}: {p_template}\n"
        f"  Pattern specificity: {p_specificity or 'n/a'}"
        f"{(' — ' + p_spec_note) if p_spec_note else ''}\n"
    )

    # Add dep resolution info so LLM knows when to distrust checker results
    unresolved_deps = issue.get("unresolved_deps", [])
    if unresolved_deps:
        user_prompt += (
            f"\nDEPENDENCY WARNING: This issue uses third-party libraries "
            f"({', '.join(unresolved_deps)}) that could not be installed. "
            f"Checker results (0 errors) may be MISLEADING — the checkers "
            f"cannot analyze code without resolved dependencies. Use your own "
            f"reasoning about the issue description to assess severity.\n"
        )

    # Add python execution output if available
    python_output = (issue.get("checker_results") or {}).get("python_output", "")
    if python_output:
        user_prompt += f"\nPython runtime output: {python_output[:300]}\n"

    user_prompt += f"\nBody:\n{body}\n"

    # Add discussion summary from comments (for actionability assessment)
    issue_comments = issue.get("comments", [])
    if issue_comments:
        comment_lines = []
        for c in issue_comments[:5]:  # First 5 comments
            author = c.get("author", "")
            comment_body = (c.get("body", "") or "")[:200]
            if comment_body:
                comment_lines.append(f"  [{author}]: {comment_body}")
        if comment_lines:
            user_prompt += (
                f"\nDiscussion ({len(issue_comments)} comments):\n"
                + "\n".join(comment_lines)
                + "\n"
            )
    if spec_excerpt:
        user_prompt += f"\nRelevant typing spec:\n{spec_excerpt[:500]}\n"

    parsed = call_llm_json(_SYSTEM_PROMPT, user_prompt, model=SONNET_MODEL)

    return {
        "priority_score": float(parsed.get("priority_score", 50)),
        "breakdown": parsed.get("breakdown", {}),
        "rationale": parsed.get("rationale", "No rationale provided"),
    }


def score_all(
    issues: list[dict],
    categorizations: dict[int, dict],
    primer_impacts: dict[int, dict],
    dep_graph: dict,
) -> dict[int, dict]:
    """Score all issues, returning {issue_number: score_result}.

    Makes one Sonnet call per issue with brief delays.
    """
    results: dict[int, dict] = {}
    total = len(issues)

    for i, issue in enumerate(issues):
        num = issue.get("number", 0)
        cat = categorizations.get(num, {})
        primer = primer_impacts.get(num, {})

        try:
            score = score_issue(issue, cat, primer, dep_graph)
            results[num] = score
            logging.info(
                f"  [{i + 1}/{total}] #{num}: score={score['priority_score']:.0f} "
                f"— {score['rationale'][:60]}"
            )
        except LLMError as e:
            # Retry once before giving up
            logging.warning(
                f"  [{i + 1}/{total}] #{num}: scoring failed, retrying: {e}"
            )
            time.sleep(2)
            try:
                score = score_issue(issue, cat, primer, dep_graph)
                results[num] = score
                logging.info(
                    f"  [{i + 1}/{total}] #{num}: retry succeeded, score={score['priority_score']:.0f}"
                )
            except LLMError as e2:
                logging.error(
                    f"  [{i + 1}/{total}] #{num}: scoring failed after retry: {e2}"
                )
                results[num] = {
                    "priority_score": -1.0,
                    "breakdown": {},
                    "rationale": f"SCORING FAILED: {e2}",
                    "failed": True,
                }

        if i < total - 1:
            time.sleep(0.5)

    return results
