#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pass 1: Categorize each issue using Haiku (fast, cheap classification).

One LLM call per issue. Classifies each issue into a category and
subcategory based on its title, body, and checker results.
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm_json, LLMError

HAIKU_MODEL = "claude-haiku-4-5-20251001"

CATEGORIES = [
    "false_positive",
    "missing_feature",
    "type_inference_bug",
    "import_resolution",
    "performance",
    "ide_feature",
    "spec_compliance",
    "edge_case",
]

_SYSTEM_PROMPT = f"""You are categorizing GitHub issues for pyrefly, a Python type checker.

Classify each issue into exactly one category:
- false_positive: Pyrefly reports an error that is incorrect (the code is valid)
- missing_feature: Pyrefly lacks support for a Python typing feature
- type_inference_bug: Pyrefly infers the wrong type for an expression
- import_resolution: Issues with finding/resolving imports
- performance: Memory, speed, or LSP responsiveness issues
- ide_feature: IDE/editor integration features (hover, completion, etc.)
- spec_compliance: Pyrefly doesn't match the typing spec behavior
- edge_case: Unusual or corner-case behavior

Also assign a subcategory (free-form, 2-4 words describing the specific issue).

Assign a confidence level: high, medium, or low.

Respond with JSON only:
{{"category": "one of {CATEGORIES}", "subcategory": "short description", "confidence": "high|medium|low"}}"""


def categorize_issue(issue: dict) -> dict:
    """Categorize a single issue using Haiku.

    Returns: {"category": str, "subcategory": str, "confidence": str}
    """
    # Build compact user prompt
    title = issue.get("title", "")
    body = (issue.get("body", "") or "")[:1500]  # Truncate long bodies
    labels = ", ".join(issue.get("labels", []))

    checker = issue.get("checker_results", {})
    checker_summary = ""
    if checker:
        pf = len(checker.get("pyrefly", []))
        pr = len(checker.get("pyright", []))
        my = len(checker.get("mypy", []))
        checker_summary = (
            f"\nChecker results: pyrefly={pf} errors, pyright={pr}, mypy={my}"
        )

    user_prompt = (
        f"Issue #{issue.get('number', '?')}: {title}\n"
        f"Labels: {labels}\n"
        f"Body:\n{body}"
        f"{checker_summary}"
    )

    parsed = call_llm_json(_SYSTEM_PROMPT, user_prompt, model=HAIKU_MODEL)

    category = parsed.get("category", "edge_case")
    if category not in CATEGORIES:
        category = "edge_case"

    return {
        "category": category,
        "subcategory": parsed.get("subcategory", "unknown"),
        "confidence": parsed.get("confidence", "low"),
    }


def categorize_all(issues: list[dict]) -> dict[int, dict]:
    """Categorize all issues, returning {issue_number: categorization}.

    Makes one Haiku call per issue with brief delays between calls.
    """
    results: dict[int, dict] = {}
    total = len(issues)

    for i, issue in enumerate(issues):
        num = issue.get("number", 0)
        try:
            cat = categorize_issue(issue)
            results[num] = cat
            logging.info(
                f"  [{i + 1}/{total}] #{num}: {cat['category']} "
                f"({cat['subcategory']}, {cat['confidence']})"
            )
        except LLMError as e:
            logging.warning(f"  [{i + 1}/{total}] #{num}: categorization failed: {e}")
            results[num] = {
                "category": "edge_case",
                "subcategory": "classification_failed",
                "confidence": "low",
            }

        if i < total - 1:
            time.sleep(0.3)

    return results
