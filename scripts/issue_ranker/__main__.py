#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for the issue ranking pipeline.

Usage:
  # Collect issue data from GitHub (needs GITHUB_TOKEN)
  python3 -m scripts.issue_ranker --mode collect \
    --pyrefly /path/to/pyrefly \
    --output /tmp/issue_data.json

  # Run ranking pipeline (needs ANTHROPIC_API_KEY or LLAMA_API_KEY)
  python3 -m scripts.issue_ranker --mode rank \
    --primer-data /tmp/primer_errors.json \
    --issue-data /tmp/issue_data.json \
    --output /tmp/ranking.md --output-json /tmp/ranking.json

  # Full pipeline (collect + rank in one step)
  python3 -m scripts.issue_ranker --mode full \
    --pyrefly /path/to/pyrefly \
    --primer-data /tmp/primer_errors.json \
    --output /tmp/ranking.md --output-json /tmp/ranking.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .code_extractor import extract_code_blocks, repair_snippet
from .github_issues import fetch_issues
from .issue_checker import check_snippets
from .pipeline import run_pipeline
from .relationship_resolver import resolve_relationships
from .report_formatter import format_json, format_markdown
from .status_classifier import classify_status

# Labels that indicate type-checking issues worth extracting code for.
_TYPECHECKING_LABELS = {
    "typechecking",
    "contextual-typing",
    "scoping-control-flow",
    "narrowing",
    "overloads",
    "conformance",
    "typeshed",
}

# Labels that indicate NON-typechecking issues (skip code extraction).
_SKIP_CODE_LABELS = {
    "performance",
    "language-server",
    "documentation",
    "configuration",
    "build-fails",
    "onboarding",
    "notebook",
}


def _is_typechecking_issue(issue: dict) -> bool:
    """Check if an issue is about type-checking (worth extracting code for).

    We only try to extract and run code for type-checking bugs/features.
    Performance, IDE, config, and documentation issues are skipped.
    """
    labels = set(issue.get("labels", []))
    title = (issue.get("title", "") or "").lower()

    # False positive/negative issues are always typechecking issues
    if "false positive" in title or "false negative" in title:
        return True

    # If it has any typechecking-related label, it's a typechecking issue
    if labels & _TYPECHECKING_LABELS:
        return True

    # If it has a skip label, it's not a typechecking issue
    if labels & _SKIP_CODE_LABELS:
        return False

    # If it's an epic, skip code extraction
    if "epic" in labels:
        return False

    # For unlabeled issues, check the title for type-checking keywords.
    # Use word boundaries to avoid false positives (e.g. "error handling").
    type_keywords = [
        "type ",
        "typing",
        "infer",
        "narrow",
        "overload",
        "generic",
        "typevar",
        "paramspec",
        "literal",
        "union",
        "protocol",
        "dataclass",
        "false positive",
        "false negative",
        "type error",
        "type check",
    ]
    return any(kw in title for kw in type_keywords)


def collect_issue_data(args: argparse.Namespace) -> dict:  # noqa: C901
    """Collect and enrich issue data from GitHub.

    Collection phases:
    1. Fetch issues from GitHub API
    2. Categorize: decide which issues are type-checking issues
    3. Extract code blocks (regex + LLM fallback, type-checking only)
    4. Repair broken snippets with LLM (type-checking only)
    5. Run pyrefly/pyright/mypy on snippets (type-checking only)
    6. Classify issue status from checker results
    7. Resolve relationships (duplicates, blockers, parents)
    """
    logging.info("=== Phase 1: Fetching issues from GitHub ===")
    start = time.time()
    label_list = (
        [lb.strip() for lb in args.labels.split(",") if lb.strip()]
        if args.labels
        else None
    ) or None
    issues = fetch_issues(
        labels=label_list,
        limit=args.limit,
    )
    logging.info(f"Fetched {len(issues)} issues in {time.time() - start:.1f}s")

    # Phase 2: Categorize issues
    logging.info("=== Phase 2: Categorizing issues ===")
    tc_issues = []
    non_tc_issues = []
    for issue in issues:
        if _is_typechecking_issue(issue):
            issue["_is_typechecking"] = True
            tc_issues.append(issue)
        else:
            issue["_is_typechecking"] = False
            non_tc_issues.append(issue)
    logging.info(
        f"  {len(tc_issues)} typechecking issues, "
        f"{len(non_tc_issues)} non-typechecking (skipping code extraction)"
    )

    # Phase 3: Extract code blocks (only for type-checking issues)
    logging.info("=== Phase 3: Extracting code blocks (typechecking issues only) ===")
    for i, issue in enumerate(issues):
        if issue["_is_typechecking"]:
            issue["code_blocks"] = extract_code_blocks(
                issue.get("body", "") or "", use_llm=True
            )
            if issue["code_blocks"]:
                logging.debug(
                    f"  [{i + 1}/{len(issues)}] #{issue['number']}: "
                    f"{len(issue['code_blocks'])} blocks"
                )
        else:
            issue["code_blocks"] = []
    has_code = sum(1 for i in issues if i["code_blocks"])
    logging.info(f"  {has_code}/{len(tc_issues)} typechecking issues have code blocks")

    # Phase 4: Repair broken snippets (only for issues with code)
    logging.info("=== Phase 4: Repairing broken snippets (LLM) ===")
    repair_count = 0
    for issue in issues:
        if issue["code_blocks"]:
            repaired = []
            for block in issue["code_blocks"]:
                fixed = repair_snippet(
                    block,
                    issue_title=issue.get("title", ""),
                    issue_body=issue.get("body", "") or "",
                )
                if fixed != block:
                    repair_count += 1
                repaired.append(fixed)
            issue["code_blocks"] = repaired
    logging.info(f"  Repaired {repair_count} snippets")

    # Phase 5: Run type checkers on snippets (with auto-install retry)
    logging.info("=== Phase 5: Running type checkers on snippets ===")
    pyrefly_bin = args.pyrefly
    if pyrefly_bin:
        checked = 0
        for issue in issues:
            if issue["code_blocks"]:
                logging.info(
                    f"  [{checked + 1}/{has_code}] Checking #{issue['number']}..."
                )
                issue["checker_results"] = check_snippets(
                    issue["code_blocks"], pyrefly_bin
                )
                # Propagate unresolved deps for status classification.
                issue["unresolved_deps"] = issue["checker_results"].get(
                    "unresolved_deps", []
                )
                issue["has_unresolved_deps"] = len(issue["unresolved_deps"]) > 0
                checked += 1
            else:
                issue["checker_results"] = {}
                issue["unresolved_deps"] = []
                issue["has_unresolved_deps"] = False
        logging.info(f"  Checked {checked} issues")
    else:
        logging.info("  Skipping (no --pyrefly binary provided)")
        for issue in issues:
            issue["checker_results"] = {}
            issue["unresolved_deps"] = []
            issue["has_unresolved_deps"] = False

    # Phase 6: Classify issue status (LLM + heuristic)
    logging.info("=== Phase 6: Classifying issue status ===")
    for issue in issues:
        issue["status_classification"] = classify_status(
            issue["checker_results"],
            issue_title=issue.get("title", ""),
            issue_body=issue.get("body", "") or "",
            unresolved_deps=issue.get("unresolved_deps"),
        )

    # Phase 7: Resolve relationships
    logging.info("=== Phase 7: Resolving relationships ===")
    relationships = resolve_relationships(issues)

    # Clean up internal fields
    for issue in issues:
        issue.pop("_is_typechecking", None)

    issue_data = {
        "issues": issues,
        "relationships": relationships,
        "metadata": {
            "labels_filter": args.labels,
            "issue_count": len(issues),
            "issues_with_code": has_code,
            "typechecking_issues": len(tc_issues),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(issue_data, f, indent=2, default=str)
        logging.info(f"Issue data written to {args.output}")

    return issue_data


def run_ranking(args: argparse.Namespace, issue_data: dict | None = None) -> None:
    """Run the 5-pass LLM ranking pipeline."""
    if issue_data is None:
        if not args.issue_data:
            print("Error: --issue-data required for rank mode", file=sys.stderr)
            sys.exit(1)
        with open(args.issue_data) as f:
            issue_data = json.load(f)

    primer_data = None
    if args.primer_data:
        with open(args.primer_data) as f:
            primer_data = json.load(f)

    pass_results_cache = getattr(args, "pass_results", None)
    results = run_pipeline(issue_data, primer_data, pass_results_cache)

    if args.output:
        md = format_markdown(results, issue_data)
        with open(args.output, "w") as f:
            f.write(md)
        logging.info(f"Ranking report written to {args.output}")
    else:
        print(format_markdown(results, issue_data))

    if args.output_json:
        json_out = format_json(results, issue_data)
        with open(args.output_json, "w") as f:
            json.dump(json_out, f, indent=2)
        logging.info(f"JSON results written to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pyrefly GitHub Issue Ranking Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "rank", "full"],
        required=True,
        help="collect: fetch+enrich issues; rank: run LLM pipeline; full: both",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated GitHub labels to filter (default: all open issues)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max issues to fetch (for testing)",
    )
    parser.add_argument(
        "--pyrefly",
        help="Path to pyrefly binary for checking code snippets",
    )
    parser.add_argument(
        "--primer-data",
        help="Path to primer_errors.json (from compare_typecheckers.py)",
    )
    parser.add_argument(
        "--issue-data",
        help="Path to issue_data.json (from --mode collect)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for ranking.md",
    )
    parser.add_argument(
        "--output-json",
        help="Output file path for ranking.json",
    )
    parser.add_argument(
        "--pass-results",
        help="Path to cache intermediate pass results (saves/loads passes 1-4). "
        "If the file exists with valid data, passes 1-4 are skipped.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.mode == "collect":
        collect_issue_data(args)
    elif args.mode == "rank":
        run_ranking(args)
    elif args.mode == "full":
        issue_data = collect_issue_data(args)
        run_ranking(args, issue_data)


if __name__ == "__main__":
    main()
