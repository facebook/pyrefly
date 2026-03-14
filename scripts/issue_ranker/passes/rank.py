#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 5: Final ranking and gap analysis.

Batched Opus calls with scored issues for final ordering and
V1 milestone gap analysis (validation metric).

Splits issues into batches of ~50 to avoid API timeouts.
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json, get_backend, LLMError

OPUS_MODEL = "claude-opus-4-20250514"

_SYSTEM_PROMPT = """You are producing the final priority ranking of GitHub issues for pyrefly, a Python type checker.

You receive all issues with their scores from prior passes. Your job:
1. Produce a final ordered list (highest priority first)
2. Group issues into priority tiers (Critical, High, Medium, Low)
3. Identify any score adjustments based on cross-issue dependencies
4. Note any issues that should be addressed together as a batch

Respond with JSON:
{
  "ranked_issues": [
    {
      "number": 123,
      "final_score": 85,
      "tier": "critical|high|medium|low",
      "adjustment_reason": "optional reason if score was adjusted from prior pass"
    }
  ],
  "priority_tiers": {
    "critical": [123, 456],
    "high": [789],
    "medium": [101],
    "low": [102]
  },
  "batch_recommendations": [
    {"issues": [123, 456], "reason": "These share a root cause and should be fixed together"}
  ],
  "summary": "1-2 sentence summary of the ranking"
}"""


BATCH_SIZE = 50
MAX_BATCH_RETRIES = 2


def _build_dep_info(dep_graph: dict) -> list[str]:
    """Build dependency info strings for the prompt."""
    dep_info = []
    for group in dep_graph.get("dependency_groups", []):
        dep_info.append(
            f"Dependency group: {group.get('name', '?')} — issues {group.get('issues', [])}"
        )
    for chain in dep_graph.get("blocking_chains", []):
        dep_info.append(
            f"Blocking: #{chain.get('blocker', '?')} blocks {chain.get('blocked', [])}"
        )
    return dep_info


def _rank_batch(
    batch_issues: list[dict],
    scores: dict[int, dict],
    categorizations: dict[int, dict],
    primer_impacts: dict[int, dict],
    dep_graph: dict,
    batch_num: int,
    total_batches: int,
    total_issue_count: int,
    score_distribution: str,
) -> dict:
    """Rank a single batch of issues via Opus."""
    summaries = []
    for issue in batch_issues:
        num = issue.get("number", 0)
        score = scores.get(num, {})
        cat = categorizations.get(num, {})
        primer = primer_impacts.get(num, {})

        summary = (
            f"#{num}: {issue.get('title', '')} "
            f"[score: {score.get('priority_score', 50):.0f}]\n"
            f"  Category: {cat.get('category', '?')} / {cat.get('subcategory', '?')}\n"
            f"  Labels: {', '.join(issue.get('labels', []))}\n"
            f"  Primer: {primer.get('primer_project_count', 0)} projects, "
            f"{primer.get('primer_error_count', 0)} errors\n"
            f"  Reactions: {issue.get('reactions_count', 0)}, "
            f"Comments: {issue.get('comments_count', 0)}\n"
            f"  Rationale: {score.get('rationale', '?')}\n"
        )
        summaries.append(summary)

    dep_info = _build_dep_info(dep_graph)

    user_prompt = (
        f"Batch {batch_num}/{total_batches} — "
        f"{len(batch_issues)} issues (of {total_issue_count} total), "
        f"pre-sorted by scoring pass.\n\n"
        f"Score distribution across ALL issues: {score_distribution}\n\n"
        f"Use these tier thresholds consistently:\n"
        f"  Critical: score >= 80\n"
        f"  High: score 65-79\n"
        f"  Medium: score 45-64\n"
        f"  Low: score < 45\n\n"
        f"Issues:\n\n" + "\n".join(summaries)
    )
    if dep_info:
        user_prompt += "\n\nDependencies (across all issues):\n" + "\n".join(dep_info)

    return call_llm_json(
        _SYSTEM_PROMPT,
        user_prompt,
        model=OPUS_MODEL,
        max_tokens=8192,
        timeout=300,
    )


def rank_issues(  # noqa: C901
    issues: list[dict],
    scores: dict[int, dict],
    categorizations: dict[int, dict],
    primer_impacts: dict[int, dict],
    dep_graph: dict,
) -> dict:
    """Produce the final ranking of all issues.

    Splits issues into batches of ~50 and makes one Opus call per batch
    to avoid API timeouts. Merges batch results by final_score.
    Returns the full ranking result including V1 gap analysis.
    """
    backend, _ = get_backend()
    if backend == "none":
        raise LLMError("No API key set")

    # Pre-sort issues by Pass 4 score (descending)
    sorted_issues = sorted(
        issues,
        key=lambda i: scores.get(i.get("number", 0), {}).get("priority_score", 0),
        reverse=True,
    )

    # Compute score distribution for context
    all_scores = [
        scores.get(i.get("number", 0), {}).get("priority_score", 0)
        for i in sorted_issues
    ]
    if all_scores:
        score_distribution = (
            f"max={max(all_scores):.0f}, min={min(all_scores):.0f}, "
            f"median={sorted(all_scores)[len(all_scores) // 2]:.0f}, "
            f"mean={sum(all_scores) / len(all_scores):.0f}"
        )
    else:
        score_distribution = "N/A"

    # Split into batches
    batches = []
    for i in range(0, len(sorted_issues), BATCH_SIZE):
        batches.append(sorted_issues[i : i + BATCH_SIZE])

    total_batches = len(batches)
    logging.info(
        f"  Splitting {len(issues)} issues into {total_batches} batches "
        f"of ~{BATCH_SIZE} for Opus ranking..."
    )

    # Process each batch
    all_ranked = []
    all_batch_recs = []
    tiers = {"critical": [], "high": [], "medium": [], "low": []}

    for batch_idx, batch in enumerate(batches, 1):
        logging.info(
            f"  Batch {batch_idx}/{total_batches}: "
            f"{len(batch)} issues (scores "
            f"{scores.get(batch[0].get('number', 0), {}).get('priority_score', 0):.0f}"
            f"–{scores.get(batch[-1].get('number', 0), {}).get('priority_score', 0):.0f})"
        )

        try:
            batch_result = _rank_batch(
                batch,
                scores,
                categorizations,
                primer_impacts,
                dep_graph,
                batch_idx,
                total_batches,
                len(issues),
                score_distribution,
            )
        except (LLMError, Exception) as e:
            # Retry failed batches before falling back to mechanical tiering
            batch_result = None
            for retry in range(1, MAX_BATCH_RETRIES + 1):
                delay = 10 * retry
                logging.warning(
                    f"  Batch {batch_idx} failed: {e}, "
                    f"retrying in {delay}s ({retry}/{MAX_BATCH_RETRIES})..."
                )
                time.sleep(delay)
                try:
                    batch_result = _rank_batch(
                        batch,
                        scores,
                        categorizations,
                        primer_impacts,
                        dep_graph,
                        batch_idx,
                        total_batches,
                        len(issues),
                        score_distribution,
                    )
                    logging.info(f"  Batch {batch_idx} succeeded on retry {retry}")
                    break
                except (LLMError, Exception) as retry_e:
                    e = retry_e
                    continue

            if batch_result is None:
                logging.error(
                    f"  Batch {batch_idx} failed after {MAX_BATCH_RETRIES} retries: "
                    f"{e}, using score-based fallback"
                )
                batch_result = _mechanical_tier(batch, scores)

        # Collect ranked issues from this batch
        for ri in batch_result.get("ranked_issues", []):
            all_ranked.append(ri)

        # Collect tier assignments
        for tier_name in tiers:
            tiers[tier_name].extend(
                batch_result.get("priority_tiers", {}).get(tier_name, [])
            )

        # Collect batch recommendations
        all_batch_recs.extend(batch_result.get("batch_recommendations", []))

    # Sort all ranked issues by final_score descending
    all_ranked.sort(key=lambda r: r.get("final_score", 0), reverse=True)

    ranking = {
        "ranked_issues": all_ranked,
        "priority_tiers": tiers,
        "batch_recommendations": all_batch_recs,
        "summary": f"Ranked {len(all_ranked)} issues across {total_batches} batches.",
    }

    # Add V1 gap analysis — compare LLM ranking against actual V1 milestone
    v1_issues = {
        i["number"]
        for i in issues
        if any(kw in (i.get("milestone") or "").lower() for kw in ("v1", "1.0"))
    }

    ranked_nums = [r["number"] for r in all_ranked]
    top_n = len(v1_issues) if v1_issues else 10
    top_ranked = set(ranked_nums[:top_n])

    if v1_issues:
        overlap = v1_issues & top_ranked
        v1_only = v1_issues - top_ranked
        ranked_only = top_ranked - v1_issues
        overlap_pct = len(overlap) / len(v1_issues) * 100 if v1_issues else 0

        ranking["v1_gap_analysis"] = {
            "v1_issue_count": len(v1_issues),
            "top_n_compared": top_n,
            "overlap_count": len(overlap),
            "overlap_percentage": round(overlap_pct, 1),
            "overlap_issues": sorted(overlap),
            "in_v1_not_top_ranked": sorted(v1_only),
            "in_top_ranked_not_v1": sorted(ranked_only),
        }
        logging.info(
            f"  V1 gap analysis: {overlap_pct:.0f}% overlap "
            f"({len(overlap)}/{len(v1_issues)} V1 issues in top {top_n})"
        )
    else:
        ranking["v1_gap_analysis"] = {
            "v1_issue_count": 0,
            "note": "No V1 milestone issues found — gap analysis skipped",
        }

    return ranking


def _mechanical_tier(batch_issues: list[dict], scores: dict[int, dict]) -> dict:
    """Fallback: assign tiers mechanically based on score thresholds."""
    ranked = []
    tiers = {"critical": [], "high": [], "medium": [], "low": []}

    for issue in batch_issues:
        num = issue.get("number", 0)
        score_val = scores.get(num, {}).get("priority_score", 50)
        if score_val >= 80:
            tier = "critical"
        elif score_val >= 65:
            tier = "high"
        elif score_val >= 45:
            tier = "medium"
        else:
            tier = "low"

        ranked.append(
            {
                "number": num,
                "final_score": score_val,
                "tier": tier,
            }
        )
        tiers[tier].append(num)

    return {
        "ranked_issues": ranked,
        "priority_tiers": tiers,
        "batch_recommendations": [],
    }
