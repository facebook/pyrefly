#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate ranking output in markdown and JSON formats."""

from __future__ import annotations

from datetime import datetime, timezone


def format_markdown(results: dict, issue_data: dict) -> str:  # noqa: C901
    """Generate a human-readable markdown ranking report.

    Includes V1 gap analysis as a blind validation metric.
    """
    ranking = results.get("ranking", {})
    pass_results = results.get("pass_results", {})
    timing = results.get("timing", {})
    cost = results.get("cost_estimate", 0)

    issues_by_num = {i["number"]: i for i in issue_data.get("issues", [])}
    categorizations = pass_results.get("categorizations", {})
    scores = pass_results.get("scores", {})
    primer_impacts = pass_results.get("primer_impacts", {})

    lines = [
        "# Pyrefly Issue Priority Ranking",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Pipeline: {timing.get('total', '?')}s, ~${cost:.2f} estimated*",
        "",
    ]

    # Summary
    ranked_issues = ranking.get("ranked_issues", [])
    summary = ranking.get("summary", "")
    if summary:
        lines.append(f"**Summary:** {summary}")
        lines.append("")

    lines.append(f"**Total issues ranked:** {len(ranked_issues)}")
    lines.append("")

    # Priority tiers
    tiers = ranking.get("priority_tiers", {})
    if tiers:
        lines.append("## Priority Tiers")
        lines.append("")
        for tier_name in ["critical", "high", "medium", "low"]:
            tier_issues = tiers.get(tier_name, [])
            if tier_issues:
                lines.append(f"### {tier_name.title()} ({len(tier_issues)} issues)")
                lines.append("")
                for num in tier_issues:
                    issue = issues_by_num.get(num, {})
                    title = issue.get("title", f"Issue #{num}")
                    score = scores.get(num, {}).get("priority_score", "?")
                    lines.append(
                        f"- [**#{num}**](https://github.com/facebook/pyrefly/issues/{num}) [{score}] {title}"
                    )
                lines.append("")

    # Ranked list with details
    lines.append("## Full Ranking")
    lines.append("")
    lines.append(
        f"| {'Rank':>4} | {'Issue':>6} | {'Score':>5} | {'Tier':<8} "
        f"| {'Category':<20} | {'Title':<50} |"
    )
    lines.append(f"|{'-' * 6}|{'-' * 8}|{'-' * 7}|{'-' * 10}|{'-' * 22}|{'-' * 52}|")
    for i, ranked in enumerate(ranked_issues, 1):
        num = ranked.get("number", 0)
        final_score = ranked.get("final_score", 50)
        tier = ranked.get("tier", "?")
        cat = categorizations.get(num, {}).get("category", "?")
        issue = issues_by_num.get(num, {})
        title = issue.get("title", "?")[:50]
        link = f"[#{num}](https://github.com/facebook/pyrefly/issues/{num})"
        lines.append(
            f"| {i:>4} | {link} | {final_score:>5.0f} | {tier:<8} "
            f"| {cat:<20} | {title:<50} |"
        )
    lines.append("")

    # Detailed breakdown (top 20)
    lines.append("## Detailed Analysis (Top 20)")
    lines.append("")
    for i, ranked in enumerate(ranked_issues[:20], 1):
        num = ranked.get("number", 0)
        issue = issues_by_num.get(num, {})
        cat = categorizations.get(num, {})
        score_data = scores.get(num, {})
        primer = primer_impacts.get(num, {})

        lines.append(
            f"### {i}. [#{num}](https://github.com/facebook/pyrefly/issues/{num}): {issue.get('title', '?')}"
        )
        lines.append("")
        lines.append(
            f"**Score:** {ranked.get('final_score', '?')} | "
            f"**Tier:** {ranked.get('tier', '?')} | "
            f"**Category:** {cat.get('category', '?')} / {cat.get('subcategory', '?')}"
        )
        lines.append(
            f"**Labels:** {', '.join(issue.get('labels', []))} | "
            f"**Reactions:** {issue.get('reactions_count', 0)} | "
            f"**Comments:** {issue.get('comments_count', 0)}"
        )
        if primer.get("primer_project_count", 0) > 0:
            lines.append(
                f"**Primer Impact:** {primer['primer_project_count']} projects, "
                f"{primer['primer_error_count']} errors (kind: {primer.get('matched_kind', '?')})"
            )
        lines.append(f"**Rationale:** {score_data.get('rationale', 'N/A')}")
        if ranked.get("adjustment_reason"):
            lines.append(f"**Adjustment:** {ranked['adjustment_reason']}")
        url = issue.get("url", f"https://github.com/facebook/pyrefly/issues/{num}")
        lines.append(f"**URL:** {url}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Batch recommendations
    batches = ranking.get("batch_recommendations", [])
    if batches:
        lines.append("## Batch Fix Recommendations")
        lines.append("")
        for batch in batches:
            issue_nums = batch.get("issues", [])
            lines.append(f"- **Issues {issue_nums}:** {batch.get('reason', '')}")
        lines.append("")

    # V1 Gap Analysis
    gap = ranking.get("v1_gap_analysis", {})
    if gap.get("v1_issue_count", 0) > 0:
        lines.append("## V1 Milestone Gap Analysis (Blind Validation)")
        lines.append("")
        lines.append(
            "The LLM was NOT told which issues are in V1. This comparison "
            "validates whether the scoring prompt captures the right priorities."
        )
        lines.append("")
        lines.append(f"- **V1 issues:** {gap['v1_issue_count']}")
        lines.append(
            f"- **Top {gap['top_n_compared']} overlap:** {gap['overlap_count']} ({gap['overlap_percentage']}%)"
        )
        lines.append("")
        if gap.get("overlap_issues"):
            lines.append(f"**In both V1 and top-ranked:** {gap['overlap_issues']}")
        if gap.get("in_v1_not_top_ranked"):
            lines.append(
                f"**In V1 but NOT top-ranked (prompt may underweight):** {gap['in_v1_not_top_ranked']}"
            )
        if gap.get("in_top_ranked_not_v1"):
            lines.append(
                f"**Top-ranked but NOT in V1 (possibly missing from V1):** {gap['in_top_ranked_not_v1']}"
            )
        lines.append("")
    elif gap.get("note"):
        lines.append("## V1 Gap Analysis")
        lines.append("")
        lines.append(f"*{gap['note']}*")
        lines.append("")

    # Timing
    lines.append("## Pipeline Timing")
    lines.append("")
    for key, val in timing.items():
        lines.append(f"- {key}: {val}s")
    lines.append(f"- Estimated cost: ${cost:.2f}")
    lines.append("")

    return "\n".join(lines)


def format_json(results: dict, issue_data: dict) -> dict:
    """Generate machine-readable JSON output.

    Returns a dict ready for json.dump().
    """
    ranking = results.get("ranking", {})
    pass_results = results.get("pass_results", {})

    issues_by_num = {i["number"]: i for i in issue_data.get("issues", [])}
    categorizations = pass_results.get("categorizations", {})
    scores = pass_results.get("scores", {})
    primer_impacts = pass_results.get("primer_impacts", {})

    ranked_issues = []
    for ranked in ranking.get("ranked_issues", []):
        num = ranked.get("number", 0)
        issue = issues_by_num.get(num, {})
        ranked_issues.append(
            {
                "number": num,
                "title": issue.get("title", ""),
                "url": issue.get("url", ""),
                "final_score": ranked.get("final_score", 50),
                "tier": ranked.get("tier", "medium"),
                "category": categorizations.get(num, {}).get("category", ""),
                "subcategory": categorizations.get(num, {}).get("subcategory", ""),
                "labels": issue.get("labels", []),
                "milestone": issue.get("milestone", ""),
                "reactions": issue.get("reactions_count", 0),
                "comments": issue.get("comments_count", 0),
                "primer_impact": primer_impacts.get(num, {}),
                "score_breakdown": scores.get(num, {}).get("breakdown", {}),
                "rationale": scores.get(num, {}).get("rationale", ""),
                "adjustment_reason": ranked.get("adjustment_reason", ""),
            }
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ranked_issues": ranked_issues,
        "priority_tiers": ranking.get("priority_tiers", {}),
        "v1_gap_analysis": ranking.get("v1_gap_analysis", {}),
        "batch_recommendations": ranking.get("batch_recommendations", []),
        "timing": results.get("timing", {}),
        "cost_estimate": results.get("cost_estimate", 0),
    }
