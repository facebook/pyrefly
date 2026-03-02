#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GitHub API client for fetching pyrefly issues.

Uses GraphQL for efficient single-query fetching of issues with labels,
milestones, reactions, sub-issues, comments count, and project priority.
Zero pip deps — uses urllib.request + ssl_utils.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from primer_classifier.ssl_utils import get_ssl_context

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
OWNER = "facebook"
REPO = "pyrefly"


def _get_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable not set. "
            "Create a PAT at https://github.com/settings/tokens"
        )
    return token


def _graphql_query(token: str, query: str, variables: dict) -> dict:
    """Execute a GitHub GraphQL query."""
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        GITHUB_GRAPHQL_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ctx = get_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            limit_total = resp.headers.get("X-RateLimit-Limit", "?")
            logging.debug(
                f"  GitHub API rate limit: {remaining}/{limit_total} remaining"
            )
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"GitHub GraphQL API returned {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GitHub API network error: {e.reason}") from e


# Issue fields shared by both query variants.
_ISSUE_FIELDS = """
        number
        title
        body
        createdAt
        url
        milestone {
          title
        }
        labels(first: 20) {
          nodes {
            name
          }
        }
        reactions {
          totalCount
        }
        comments {
          totalCount
        }
        comments_preview: comments(first: 10) {
          nodes {
            body
            author {
              login
            }
            createdAt
          }
        }
        trackedInIssues(first: 5) {
          nodes {
            number
            title
          }
        }
        trackedIssues(first: 20) {
          nodes {
            number
            title
          }
        }
"""

# ProjectV2 fields for fetching priority (requires read:project scope).
_PROJECT_FIELDS = """
        projectItems(first: 5) {
          nodes {
            fieldValues(first: 10) {
              nodes {
                ... on ProjectV2ItemFieldSingleSelectValue {
                  name
                  field {
                    ... on ProjectV2SingleSelectField {
                      name
                    }
                  }
                }
              }
            }
          }
        }
"""


def _build_query(include_projects: bool) -> str:
    """Build the GraphQL query, optionally including project fields."""
    project_block = _PROJECT_FIELDS if include_projects else ""
    return f"""
query($owner: String!, $repo: String!, $labels: [String!], $after: String) {{
  repository(owner: $owner, name: $repo) {{
    issues(
      first: 100,
      after: $after,
      states: [OPEN],
      labels: $labels,
      orderBy: {{field: CREATED_AT, direction: DESC}}
    ) {{
      pageInfo {{
        hasNextPage
        endCursor
      }}
      nodes {{
{_ISSUE_FIELDS}
{project_block}
      }}
    }}
  }}
}}
"""


def _extract_priority(node: dict) -> str:
    """Extract priority value from GitHub Projects custom fields."""
    for item in node.get("projectItems", {}).get("nodes", []):
        for fv in item.get("fieldValues", {}).get("nodes", []):
            field_name = (fv.get("field") or {}).get("name", "")
            if field_name.lower() in ("priority", "p"):
                return fv.get("name", "")
    return ""


def _parse_issue_node(node: dict, has_projects: bool) -> dict:
    """Parse a GraphQL issue node into our issue dict format."""
    priority = _extract_priority(node) if has_projects else ""
    return {
        "number": node["number"],
        "title": node["title"],
        "body": node.get("body", ""),
        "created_at": node["createdAt"],
        "url": node.get("url", ""),
        "milestone": (node.get("milestone") or {}).get("title", ""),
        "labels": [lbl["name"] for lbl in node.get("labels", {}).get("nodes", [])],
        "reactions_count": node.get("reactions", {}).get("totalCount", 0),
        "comments_count": node.get("comments", {}).get("totalCount", 0),
        "comments": [
            {
                "author": (c.get("author") or {}).get("login", ""),
                "body": c.get("body", ""),
                "created_at": c.get("createdAt", ""),
            }
            for c in node.get("comments_preview", {}).get("nodes", [])
        ],
        "priority": priority,
        "parent_issues": [
            {"number": p["number"], "title": p["title"]}
            for p in node.get("trackedInIssues", {}).get("nodes", [])
        ],
        "sub_issues": [
            {"number": s["number"], "title": s["title"]}
            for s in node.get("trackedIssues", {}).get("nodes", [])
        ],
    }


def _has_scope_error(data: dict) -> bool:
    """Check if the GraphQL response has INSUFFICIENT_SCOPES errors."""
    for err in data.get("errors", []):
        if err.get("type") == "INSUFFICIENT_SCOPES":
            return True
    return False


def fetch_issues(
    labels: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Fetch open issues from facebook/pyrefly, optionally filtered by labels.

    Tries to include GitHub Projects priority data. If the token lacks
    read:project scope, falls back to fetching without project fields.
    """
    token = _get_token()
    all_issues: list[dict] = []
    cursor = None
    page = 0

    # Try with project fields first; fall back if scope is missing.
    include_projects = True
    query = _build_query(include_projects=True)

    while True:
        page += 1
        logging.info(f"  Fetching issues page {page}...")
        variables: dict = {
            "owner": OWNER,
            "repo": REPO,
            "labels": labels,
            "after": cursor,
        }
        data = _graphql_query(token, query, variables)

        # On first page, check for scope errors and fall back.
        if page == 1 and include_projects and _has_scope_error(data):
            logging.warning(
                "Token lacks read:project scope — fetching without "
                "project priorities. Add read:project to your token "
                "at https://github.com/settings/tokens to include "
                "P0/P1/P2 priority data."
            )
            include_projects = False
            query = _build_query(include_projects=False)
            data = _graphql_query(token, query, variables)

        if data.get("errors"):
            # Log non-scope errors but continue.
            non_scope = [
                e for e in data["errors"] if e.get("type") != "INSUFFICIENT_SCOPES"
            ]
            if non_scope:
                logging.warning(f"GraphQL errors: {non_scope}")

        repo = data.get("data", {}).get("repository", {})
        issues_data = repo.get("issues", {})
        nodes = issues_data.get("nodes", [])

        for node in nodes:
            issue = _parse_issue_node(node, include_projects)
            all_issues.append(issue)
            if limit and len(all_issues) >= limit:
                logging.info(f"  Reached limit of {limit} issues")
                return all_issues

        logging.info(
            f"  Page {page}: {len(nodes)} issues (total so far: {len(all_issues)})"
        )
        page_info = issues_data.get("pageInfo", {})
        if not page_info.get("hasNextPage", False):
            break
        cursor = page_info.get("endCursor")

    logging.info(f"  Total: {len(all_issues)} issues fetched")
    v1_count = sum(
        1
        for i in all_issues
        if any(kw in (i.get("milestone") or "").lower() for kw in ("v1", "1.0"))
    )
    if v1_count:
        logging.info(f"  V1 milestone issues: {v1_count}")

    # Log priority distribution
    priority_counts: dict[str, int] = {}
    for i in all_issues:
        p = i.get("priority", "") or "unset"
        priority_counts[p] = priority_counts.get(p, 0) + 1
    if any(k != "unset" for k in priority_counts):
        logging.info(
            f"  Priority distribution: {dict(sorted(priority_counts.items()))}"
        )

    return all_issues
