#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""V1 gap analysis with LLM-generated reasons and GitHub label management.

Post-processes ranking.json to:
1. Generate concise reasons for why issues should be reconsidered (one Haiku call)
2. Apply GitHub labels: v1-consider-adding, v1-consider-removing, v1-verified
3. Generate a team-readable markdown report

IMPORTANT: Label cleanup only touches labels introduced by this workflow
(v1-verified, v1-consider-adding, v1-consider-removing). No other labels
are ever modified or removed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_transport import call_llm_json

# Labels managed by this workflow. Only these are ever created/removed.
LABEL_VERIFIED = "v1-verified"
LABEL_CONSIDER_ADDING = "v1-consider-adding"
LABEL_CONSIDER_REMOVING = "v1-consider-removing"

_MANAGED_LABELS = {
    LABEL_VERIFIED: {
        "color": "0e8a16",
        "description": "In both V1 milestone and top-ranked (verified by ranking pipeline)",
    },
    LABEL_CONSIDER_ADDING: {
        "color": "fbca04",
        "description": "Top-ranked but not in V1 milestone (consider adding)",
    },
    LABEL_CONSIDER_REMOVING: {
        "color": "e4e669",
        "description": "In V1 milestone but not top-ranked (consider removing)",
    },
}


def generate_reasons(ranked_issues: list[dict], v1_gap: dict) -> dict:
    """Generate concise reasons for Q2 (consider removing) and Q3 (consider adding).

    Makes a single Haiku LLM call with issue summaries from ranking.json.
    Returns {"q2_reasons": {num: reason, ...}, "q3_reasons": {num: reason, ...}}.
    """
    issues_by_num = {i["number"]: i for i in ranked_issues}

    # Build Q2 issue summaries (in V1 but NOT top-ranked)
    q2_issues = []
    for num in v1_gap.get("in_v1_not_top_ranked", []):
        issue = issues_by_num.get(num)
        if not issue:
            continue
        q2_issues.append(
            {
                "number": num,
                "title": issue["title"],
                "final_score": issue["final_score"],
                "tier": issue["tier"],
                "category": issue.get("category", ""),
                "labels": issue.get("labels", []),
                "primer_impact": issue.get("primer_impact", {}),
                "rationale": issue.get("rationale", ""),
            }
        )

    # Build Q3 issue summaries (top-ranked but NOT in V1)
    q3_issues = []
    for num in v1_gap.get("in_top_ranked_not_v1", []):
        issue = issues_by_num.get(num)
        if not issue:
            continue
        q3_issues.append(
            {
                "number": num,
                "title": issue["title"],
                "final_score": issue["final_score"],
                "tier": issue["tier"],
                "category": issue.get("category", ""),
                "labels": issue.get("labels", []),
                "primer_impact": issue.get("primer_impact", {}),
                "rationale": issue.get("rationale", ""),
            }
        )

    if not q2_issues and not q3_issues:
        return {"q2_reasons": {}, "q3_reasons": {}}

    system_prompt = (
        "You are a technical analyst reviewing a type checker's issue backlog. "
        "Given two lists of issues, provide a concise reason (1-2 sentences, max 25 words) "
        "for each issue.\n\n"
        "For Q2 (consider REMOVING from V1 milestone): explain the WEAKNESS — why this "
        "issue is LESS important than it seems. Focus on why it should be deprioritized: "
        "stale/inactive, epic/tracker not directly actionable, blocked/needs-discussion, "
        "false negative (less urgent than false positive), niche scenario, low engagement, "
        "borderline score. Do NOT describe the issue's importance or impact — describe "
        "what makes it a weak candidate for V1. Do NOT cite implementation difficulty or "
        "complexity as a reason to deprioritize — difficulty is not a deterrent for V1.\n\n"
        "For Q3 (consider ADDING to V1 milestone): explain the STRENGTH — why this "
        "issue is MORE important than its current milestone suggests. Focus on: "
        "high primer impact (N projects, N errors), strategic adoption tags (google, "
        "pytorch, pydantic), false positive driving users away, common Python idiom.\n\n"
        "NOTE: Primer impact numbers are matched by error message pattern, not by "
        "specific root cause. Each issue includes a pattern specificity assessment "
        "(high/medium/low) from an LLM that estimates how likely this issue is the "
        "primary cause of those errors. When specificity is low, say 'pattern appears "
        "in N projects' rather than implying this specific issue causes all those "
        "errors. When specificity is high, the counts are a stronger signal.\n\n"
        "Return JSON with exactly this structure:\n"
        '{"q2_reasons": {"123": "reason...", ...}, "q3_reasons": {"456": "reason...", ...}}'
    )

    user_prompt = json.dumps(
        {
            "q2_issues_consider_removing": q2_issues,
            "q3_issues_consider_adding": q3_issues,
        }
    )

    logging.info(
        f"Generating reasons for {len(q2_issues)} Q2 and {len(q3_issues)} Q3 issues"
    )
    result = call_llm_json(
        system_prompt,
        user_prompt,
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
    )

    # Normalize keys to ints for consistency
    q2 = {int(k): v for k, v in result.get("q2_reasons", {}).items()}
    q3 = {int(k): v for k, v in result.get("q3_reasons", {}).items()}
    return {"q2_reasons": q2, "q3_reasons": q3}


def _github_api(
    method: str, url: str, token: str, body: dict | None = None
) -> dict | list | None:
    """Make a GitHub REST API request. Returns parsed JSON or None on 404/422."""
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if data else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            return None
        raise


def _ensure_labels_exist(repo: str, token: str) -> None:
    """Create managed labels if they don't already exist on the repo."""
    api = f"https://api.github.com/repos/{repo}/labels"
    for name, meta in _MANAGED_LABELS.items():
        result = _github_api("GET", f"{api}/{urllib.parse.quote(name)}", token)
        if result is None:
            logging.info(f"Creating label '{name}'")
            _github_api(
                "POST",
                api,
                token,
                {
                    "name": name,
                    "color": meta["color"],
                    "description": meta["description"],
                },
            )
        else:
            logging.debug(f"Label '{name}' already exists")


def _cleanup_managed_labels(repo: str, token: str) -> int:
    """Remove all managed labels from all issues. Returns count of labels removed.

    IMPORTANT: This ONLY removes labels that our workflow introduced
    (v1-verified, v1-consider-adding, v1-consider-removing). No other
    labels on any issue are ever touched.
    """
    api = f"https://api.github.com/repos/{repo}"
    removed = 0
    for label_name in _MANAGED_LABELS:
        # Fetch all issues that have this specific managed label
        page = 1
        while True:
            url = (
                f"{api}/issues?labels={urllib.parse.quote(label_name)}"
                f"&state=open&per_page=100&page={page}"
            )
            issues = _github_api("GET", url, token) or []
            if not issues:
                break
            for issue in issues:
                num = issue["number"]
                # Remove ONLY this specific managed label from the issue
                delete_url = (
                    f"{api}/issues/{num}/labels/{urllib.parse.quote(label_name)}"
                )
                _github_api("DELETE", delete_url, token)
                removed += 1
                logging.debug(f"Removed '{label_name}' from #{num}")
            if len(issues) < 100:
                break
            page += 1
    return removed


def apply_labels(repo: str, analysis: dict, github_token: str) -> dict:
    """Apply V1 analysis labels to GitHub issues.

    IMPORTANT: Only manages labels introduced by this workflow:
    v1-verified, v1-consider-adding, v1-consider-removing.
    No other labels are ever created, modified, or removed.

    Steps:
    1. Ensure the three managed labels exist on the repo
    2. Remove managed labels from all issues (clean slate)
    3. Apply fresh labels based on current analysis
    """
    logging.info("Ensuring managed labels exist on repo")
    _ensure_labels_exist(repo, github_token)

    logging.info(
        "Cleaning up old managed labels (only v1-verified/consider-adding/removing)"
    )
    removed = _cleanup_managed_labels(repo, github_token)
    logging.info(f"Removed {removed} stale managed labels")

    api = f"https://api.github.com/repos/{repo}/issues"
    applied = {"verified": 0, "consider_adding": 0, "consider_removing": 0}

    # Apply v1-verified to overlap issues
    for num in analysis.get("overlap_issues", []):
        _github_api(
            "POST",
            f"{api}/{num}/labels",
            github_token,
            {
                "labels": [LABEL_VERIFIED],
            },
        )
        applied["verified"] += 1

    # Apply v1-consider-removing to Q2 issues
    for num in analysis.get("in_v1_not_top_ranked", []):
        _github_api(
            "POST",
            f"{api}/{num}/labels",
            github_token,
            {
                "labels": [LABEL_CONSIDER_REMOVING],
            },
        )
        applied["consider_removing"] += 1

    # Apply v1-consider-adding to Q3 issues
    for num in analysis.get("in_top_ranked_not_v1", []):
        _github_api(
            "POST",
            f"{api}/{num}/labels",
            github_token,
            {
                "labels": [LABEL_CONSIDER_ADDING],
            },
        )
        applied["consider_adding"] += 1

    logging.info(
        f"Applied labels: {applied['verified']} verified, "
        f"{applied['consider_adding']} consider-adding, "
        f"{applied['consider_removing']} consider-removing"
    )
    return applied


def _escape_md_table(text: str) -> str:
    """Escape characters that break markdown table cells."""
    return text.replace("|", "\\|").replace("`", "\\`")


def format_v1_report(analysis: dict) -> str:
    """Generate a markdown report for the V1 gap analysis with reasons.

    The analysis dict contains:
    - v1_gap: the gap analysis from ranking.json
    - ranked_issues: list of all ranked issues from ranking.json
    - reasons: output from generate_reasons()
    """
    v1_gap = analysis["v1_gap"]
    issues_by_num = {i["number"]: i for i in analysis["ranked_issues"]}
    q2_reasons = analysis.get("reasons", {}).get("q2_reasons", {})
    q3_reasons = analysis.get("reasons", {}).get("q3_reasons", {})

    v1_count = v1_gap.get("v1_issue_count", 0)
    top_n = v1_gap.get("top_n_compared", 0)
    overlap_count = v1_gap.get("overlap_count", 0)
    overlap_pct = v1_gap.get("overlap_percentage", 0)

    overlap_issues = v1_gap.get("overlap_issues", [])
    q2_issues = v1_gap.get("in_v1_not_top_ranked", [])
    q3_issues = v1_gap.get("in_top_ranked_not_v1", [])

    lines = [
        "## V1 Gap Analysis with Reasons",
        "",
        "### Overview",
        f"- V1 issues: {v1_count} | Top-{top_n} ranked: {top_n} "
        f"| Overlap: {overlap_count} ({overlap_pct}%)",
        "",
    ]

    # Verified V1 Issues (overlap)
    lines.append(f"### Verified V1 Issues ({len(overlap_issues)})")
    lines.append("")
    if overlap_issues:
        lines.append("| # | Title | Score | Tier |")
        lines.append("|---|-------|-------|------|")
        for num in sorted(
            overlap_issues,
            key=lambda n: issues_by_num.get(n, {}).get("final_score", 0),
            reverse=True,
        ):
            issue = issues_by_num.get(num, {})
            title = _escape_md_table(issue.get("title", f"Issue #{num}")[:60])
            score = issue.get("final_score", "?")
            tier = issue.get("tier", "?")
            lines.append(
                f"| [#{num}](https://github.com/facebook/pyrefly/issues/{num}) "
                f"| {title} | {score} | {tier} |"
            )
        lines.append("")

    # Consider Removing from V1 (Q2)
    lines.append(f"### Consider Removing from V1 ({len(q2_issues)})")
    lines.append("")
    if q2_issues:
        lines.append("| # | Title | Score | Tier | Reason |")
        lines.append("|---|-------|-------|------|--------|")
        for num in sorted(
            q2_issues,
            key=lambda n: issues_by_num.get(n, {}).get("final_score", 0),
            reverse=True,
        ):
            issue = issues_by_num.get(num, {})
            title = _escape_md_table(issue.get("title", f"Issue #{num}")[:60])
            score = issue.get("final_score", "?")
            tier = issue.get("tier", "?")
            reason = _escape_md_table(q2_reasons.get(num, ""))
            lines.append(
                f"| [#{num}](https://github.com/facebook/pyrefly/issues/{num}) "
                f"| {title} | {score} | {tier} | {reason} |"
            )
        lines.append("")

    # Consider Adding to V1 (Q3)
    lines.append(f"### Consider Adding to V1 ({len(q3_issues)})")
    lines.append("")
    if q3_issues:
        lines.append("| # | Title | Score | Tier | Reason |")
        lines.append("|---|-------|-------|------|--------|")
        for num in sorted(
            q3_issues,
            key=lambda n: issues_by_num.get(n, {}).get("final_score", 0),
            reverse=True,
        ):
            issue = issues_by_num.get(num, {})
            title = _escape_md_table(issue.get("title", f"Issue #{num}")[:60])
            score = issue.get("final_score", "?")
            tier = issue.get("tier", "?")
            reason = _escape_md_table(q3_reasons.get(num, ""))
            lines.append(
                f"| [#{num}](https://github.com/facebook/pyrefly/issues/{num}) "
                f"| {title} | {score} | {tier} | {reason} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_v1_analysis(
    ranking_json: str,
    output_md: str,
    repo: str,
    github_token: str,
    apply: bool,
) -> dict:
    """Run V1 gap analysis: load ranking, generate reasons, apply labels, write report.

    Args:
        ranking_json: Path to ranking.json from the ranking pipeline.
        output_md: Path to write the markdown report.
        repo: GitHub repo in "owner/repo" format.
        github_token: GitHub token for label management (empty to skip).
        apply: Whether to apply labels to GitHub issues.

    Returns:
        Analysis dict with v1_gap, reasons, and label results.
    """
    logging.info(f"Loading ranking data from {ranking_json}")
    with open(ranking_json) as f:
        ranking_data = json.load(f)

    ranked_issues = ranking_data.get("ranked_issues", [])
    v1_gap = ranking_data.get("v1_gap_analysis", {})

    if v1_gap.get("v1_issue_count", 0) == 0:
        logging.warning("No V1 gap data found in ranking.json — nothing to analyze")
        return {"v1_gap": v1_gap, "reasons": {}, "labels_applied": {}}

    logging.info(
        f"V1 gap: {v1_gap['v1_issue_count']} V1 issues, "
        f"{v1_gap['overlap_count']} overlap ({v1_gap['overlap_percentage']}%), "
        f"{len(v1_gap.get('in_v1_not_top_ranked', []))} Q2, "
        f"{len(v1_gap.get('in_top_ranked_not_v1', []))} Q3"
    )

    # Generate reasons with one Haiku call
    reasons = generate_reasons(ranked_issues, v1_gap)
    logging.info(
        f"Generated {len(reasons.get('q2_reasons', {}))} Q2 reasons, "
        f"{len(reasons.get('q3_reasons', {}))} Q3 reasons"
    )

    analysis = {
        "v1_gap": v1_gap,
        "ranked_issues": ranked_issues,
        "reasons": reasons,
        "labels_applied": {},
    }

    # Apply labels if requested
    if apply and github_token:
        analysis["labels_applied"] = apply_labels(repo, v1_gap, github_token)
    elif apply and not github_token:
        logging.warning("--apply-labels requested but no GITHUB_TOKEN set, skipping")

    # Write markdown report
    report = format_v1_report(analysis)
    if output_md:
        with open(output_md, "w") as f:
            f.write(report)
        logging.info(f"V1 analysis report written to {output_md}")
    else:
        print(report)

    return analysis
