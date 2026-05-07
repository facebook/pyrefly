#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
GitHub Utilities

Provides a GitHub API client, bot filtering, contributor analysis, and
bug-between-tags helpers used by the release-notes generator and other scripts.

Consolidated from:
  - src/utils/github_client.py
  - src/utils/bot_filter.py
  - src/scripts/bugs_between_tags.py
  - src/scripts/list_contributors.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Bot filtering
# ---------------------------------------------------------------------------


def is_bot(username: str, user_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a user is a bot

    Args:
        username: GitHub username
        user_data: Optional user data from GitHub API

    Returns:
        True if the user is a bot, False otherwise
    """
    if not username:
        return False

    # Check if username ends with [bot]
    if username.endswith("[bot]"):
        return True

    # Check user type if available
    if user_data and user_data.get("type") == "Bot":
        return True

    # Common bot patterns
    bot_patterns = [
        "dependabot",
        "renovate",
        "greenkeeper",
        "snyk-bot",
        "codecov",
        "coveralls",
        "github-actions",
        "netlify",
        "vercel",
        "imgbot",
        "restyled-io",
        "stale",
        "allcontributors",
    ]

    username_lower = username.lower()
    return any(pattern in username_lower for pattern in bot_patterns)


# ---------------------------------------------------------------------------
# Environment / .env loading
# ---------------------------------------------------------------------------


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file

    Args:
        env_path: Path to .env file. If None, looks for .env in current directory and parent directories
    """
    if env_path:
        env_file = Path(env_path)
    else:
        # Look for .env file in current directory and walk up to find it
        current = Path.cwd()
        env_file = None

        # Check current directory and up to 3 parent directories
        for _ in range(4):
            candidate = current / ".env"
            if candidate.exists():
                env_file = candidate
                break
            current = current.parent

    if env_file and env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Only set if not already in environment
                    if key and not os.getenv(key):
                        os.environ[key] = value


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------


class GitHubClient:
    """GitHub API client wrapper using standard library (urllib)"""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client

        Args:
            token: GitHub personal access token. If not provided, will use GITHUB_TOKEN env var

        Raises:
            ValueError: If no token is provided and GITHUB_TOKEN env var is not set
        """
        # Try to load .env file if token not provided and env var not set
        if not token and not os.getenv("GITHUB_TOKEN"):
            load_env_file()

        auth_token = token or os.getenv("GITHUB_TOKEN")

        if not auth_token:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable or pass token to constructor."
            )

        self.token = auth_token
        self.base_url = "https://api.github.com"

    def _make_request(self, url: str) -> Any:
        """
        Make a request to the GitHub API

        Args:
            url: API endpoint URL

        Returns:
            Parsed JSON response
        """
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OSS-Health-Checker-Python",
        }

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req) as response:
                data = response.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise Exception(f"GitHub API error: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def get_all_commits(
        self,
        owner: str,
        repo: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        branch: Optional[str] = None,
        max_commits: int = 10000,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all repository commits with pagination support

        Args:
            owner: Repository owner
            repo: Repository name
            since: Only commits after this date (ISO 8601 format)
            until: Only commits before this date (ISO 8601 format)
            branch: Only commits from this branch (branch name or SHA)
            max_commits: Maximum number of commits to fetch
            per_page: Number of results per page

        Returns:
            List of commit dictionaries
        """
        all_commits = []
        page = 1

        while len(all_commits) < max_commits:
            url = f"{self.base_url}/repos/{owner}/{repo}/commits?per_page={per_page}&page={page}"
            if since:
                url += f"&since={since}"
            if until:
                url += f"&until={until}"
            if branch:
                url += f"&sha={branch}"

            try:
                commits = self._make_request(url)
                if not commits:
                    break

                all_commits.extend(commits)

                if len(commits) < per_page:
                    break

                page += 1
            except Exception as e:
                print(f"Warning: Error fetching page {page}: {e}")
                break

        return all_commits[:max_commits]

    def close(self):
        """Close the GitHub client connection (no-op for urllib)"""
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ---------------------------------------------------------------------------
# Contributor analysis
# ---------------------------------------------------------------------------


class ContributorAnalyzer:
    """Analyzes and lists repository contributors"""

    def __init__(self, client: GitHubClient, owner: str, repo: str):
        """
        Initialize analyzer

        Args:
            client: GitHubClient instance
            owner: Repository owner
            repo: Repository name
        """
        self.client = client
        self.owner = owner
        self.repo = repo

    def get_contributors_by_date_range(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Get contributors within a date range

        Args:
            since: Start date (ISO format)
            until: End date (ISO format)
            branch: Only include commits from this branch (e.g., 'main')

        Returns:
            Dictionary mapping contributor ID to their info (name/username, commit count, type)
        """
        print("\n🔍 Fetching commits", end="")
        if since:
            print(f" since {since}", end="")
        if until:
            print(f" until {until}", end="")
        if branch:
            print(f" on branch '{branch}'", end="")
        print("...")

        commits = self.client.get_all_commits(
            self.owner,
            self.repo,
            since=since,
            until=until,
            branch=branch,
            max_commits=100000,
        )

        print(f"✓ Found {len(commits)} commits")
        if branch:
            print(f"ℹ️  Note: Filtered to '{branch}' branch only")
        else:
            print("ℹ️  Note: Date filtering includes commits on all branches")

        return self._extract_contributors(commits)

    def get_contributors_by_tags(self, from_tag: str, to_tag: str) -> Dict[str, Dict]:
        """
        Get contributors between two release tags

        Args:
            from_tag: Starting tag
            to_tag: Ending tag

        Returns:
            Dictionary mapping contributor ID to their info (name/username, commit count, type)
        """
        print(f"\n🔍 Fetching commits between tags '{from_tag}' and '{to_tag}'...")

        url = f"{self.client.base_url}/repos/{self.owner}/{self.repo}/compare/{from_tag}...{to_tag}"

        try:
            comparison = self.client._make_request(url)
            commits = comparison.get("commits", [])

            print(f"✓ Found {len(commits)} commits between tags")
            print(
                "ℹ️  Note: Tag comparison shows commits in the direct lineage between releases"
            )

            return self._extract_contributors(commits)

        except Exception as e:
            print(f"❌ Error comparing tags: {e}", file=sys.stderr)
            print("   Make sure both tags exist in the repository", file=sys.stderr)
            sys.exit(1)

    def _extract_contributors(self, commits: List[Dict]) -> Dict[str, Dict]:
        """
        Extract unique contributors from commits

        Args:
            commits: List of commit dictionaries

        Returns:
            Dictionary mapping contributor ID to their info (bots excluded)
        """
        contributors = {}

        for commit in commits:
            contributor_id = None
            contributor_name = None
            contributor_type = None

            # Try to get GitHub user first
            if "author" in commit and commit["author"]:
                author_login = commit["author"].get("login")
                if author_login:
                    if is_bot(author_login, commit["author"]):
                        continue
                    contributor_id = f"user:{author_login}"
                    contributor_name = author_login
                    contributor_type = "github_user"

            # Fallback to commit author name if no GitHub user
            if not contributor_id and "commit" in commit:
                author_info = commit["commit"].get("author", {})
                author_name = author_info.get("name")
                author_email = author_info.get("email")

                if author_name:
                    if is_bot(author_name):
                        continue
                    contributor_id = f"name:{author_name}"
                    contributor_name = author_name
                    contributor_type = "name_based"

                    if author_email:
                        contributor_name = f"{author_name} <{author_email}>"

            # Add or update contributor
            if contributor_id:
                if contributor_id not in contributors:
                    contributors[contributor_id] = {
                        "name": contributor_name,
                        "type": contributor_type,
                        "commit_count": 0,
                    }
                contributors[contributor_id]["commit_count"] += 1

        return contributors


def display_contributors(
    contributors: Dict[str, Dict],
    sort_by: str = "commits",
    show_count: bool = True,
    show_type: bool = False,
):
    """
    Display contributors list

    Args:
        contributors: Dictionary of contributors
        sort_by: Sort order ('commits', 'name')
        show_count: Whether to show commit counts
        show_type: Whether to show contributor type (GitHub user vs name-based)
    """
    print("\n" + "=" * 80)
    print("  CONTRIBUTORS")
    print("=" * 80)

    # Separate GitHub users from name-based contributors
    github_users = {k: v for k, v in contributors.items() if v["type"] == "github_user"}
    name_based = {k: v for k, v in contributors.items() if v["type"] == "name_based"}

    print(f"\n  Total unique contributors: {len(contributors)}")
    print(f"    - GitHub users: {len(github_users)}")
    print(f"    - Name-based: {len(name_based)}")

    # Display GitHub users
    if github_users:
        print("\n" + "-" * 80)
        print(f"  GitHub Users ({len(github_users)})")
        print("-" * 80)

        github_sorted = sorted(
            github_users.items(),
            key=lambda x: (-x[1]["commit_count"], x[1]["name"].lower())
            if sort_by == "commits"
            else (x[1]["name"].lower(), -x[1]["commit_count"]),
        )

        for i, (_contributor_id, info) in enumerate(github_sorted, 1):
            output = f"  {i:4d}. {info['name']}"
            if show_count:
                output += f" ({info['commit_count']} commits)"
            print(output)

    # Display name-based contributors
    if name_based:
        print("\n" + "-" * 80)
        print(f"  Name-based Contributors ({len(name_based)})")
        print("-" * 80)

        name_sorted = sorted(
            name_based.items(),
            key=lambda x: (-x[1]["commit_count"], x[1]["name"].lower())
            if sort_by == "commits"
            else (x[1]["name"].lower(), -x[1]["commit_count"]),
        )

        for i, (_contributor_id, info) in enumerate(name_sorted, 1):
            output = f"  {i:4d}. {info['name']}"
            if show_count:
                output += f" ({info['commit_count']} commits)"
            print(output)

    # Print comma-separated list with @ handles for GitHub users
    if github_users:
        print("\n" + "-" * 80)
        print("  GitHub Handles (copy-paste ready)")
        print("-" * 80)
        github_sorted = sorted(
            github_users.items(),
            key=lambda x: (-x[1]["commit_count"], x[1]["name"].lower())
            if sort_by == "commits"
            else (x[1]["name"].lower(), -x[1]["commit_count"]),
        )
        handles = [f"@{info['name']}" for _, info in github_sorted]
        print(f"\n  {', '.join(handles)}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    total_commits = sum(c["commit_count"] for c in contributors.values())
    avg_commits = total_commits / len(contributors) if contributors else 0
    max_commits = max((c["commit_count"] for c in contributors.values()), default=0)
    min_commits = min((c["commit_count"] for c in contributors.values()), default=0)

    print(f"\n  Total contributors: {len(contributors)}")
    print(f"  Total commits:      {total_commits:,}")
    print(f"  Average commits:    {avg_commits:.1f} per contributor")
    if contributors:
        top = max(contributors.values(), key=lambda x: x["commit_count"])
        print(f"  Max commits:        {max_commits} (by {top['name']})")
    else:
        print(f"  Max commits:        {max_commits}")
    print(f"  Min commits:        {min_commits}")


# ---------------------------------------------------------------------------
# Bugs between tags
# ---------------------------------------------------------------------------


def get_tag_date(client: GitHubClient, owner: str, repo: str, tag: str) -> datetime:
    """
    Get the date of a git tag

    Args:
        client: GitHubClient instance
        owner: Repository owner
        repo: Repository name
        tag: Tag name (e.g., 'v1.0.0')

    Returns:
        datetime of the tag's commit
    """
    url = f"{client.base_url}/repos/{owner}/{repo}/git/refs/tags/{tag}"
    try:
        ref = client._make_request(url)
    except Exception as e:
        raise ValueError(f"Tag '{tag}' not found: {e}")

    obj_type = ref["object"]["type"]
    obj_sha = ref["object"]["sha"]

    if obj_type == "tag":
        # Annotated tag - need to get the tag object first
        tag_url = f"{client.base_url}/repos/{owner}/{repo}/git/tags/{obj_sha}"
        tag_obj = client._make_request(tag_url)
        commit_sha = tag_obj["object"]["sha"]
    else:
        # Lightweight tag - points directly to commit
        commit_sha = obj_sha

    commit_url = f"{client.base_url}/repos/{owner}/{repo}/git/commits/{commit_sha}"
    commit = client._make_request(commit_url)
    date_str = commit["committer"]["date"]

    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def fetch_bug_issues(
    client: GitHubClient,
    owner: str,
    repo: str,
    start_date: datetime,
    end_date: datetime,
    issue_type: str = "Bug",
) -> List[Dict[str, Any]]:
    """
    Fetch issues with a given issue type that were closed as completed between two dates.

    Args:
        client: GitHubClient instance
        owner: Repository owner
        repo: Repository name
        start_date: Start date (from-tag date)
        end_date: End date (to-tag date)
        issue_type: Issue type name to filter by (e.g., "Bug")

    Returns:
        List of matching issue dictionaries
    """
    matching_issues: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = (
            f"{client.base_url}/repos/{owner}/{repo}/issues"
            f"?state=closed&per_page=100&page={page}"
            f"&sort=updated&direction=desc"
            f"&since={start_date.isoformat()}"
        )

        # Retry on transient server errors (e.g. 503)
        max_retries = 3
        batch = None
        for attempt in range(max_retries):
            try:
                batch = client._make_request(url)
                break
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    wait = 2**attempt
                    print(f"    Received 503, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        if not batch:
            break

        for issue in batch:
            # Skip pull requests (they appear in the issues endpoint)
            if "pull_request" in issue:
                continue

            # Filter by issue type (not label)
            issue_type_obj = issue.get("type")
            if not issue_type_obj or issue_type_obj.get("name") != issue_type:
                continue

            # Only include issues closed as completed
            if issue.get("state_reason") != "completed":
                continue

            closed_at_str = issue.get("closed_at")
            if not closed_at_str:
                continue

            closed_at = datetime.fromisoformat(closed_at_str.replace("Z", "+00:00"))

            if start_date <= closed_at <= end_date:
                matching_issues.append(issue)

        if len(batch) < 100:
            break

        page += 1

    return matching_issues


def display_bug_results(
    issues: List[Dict[str, Any]],
    owner: str,
    repo: str,
    from_tag: str,
    to_tag: str,
    start_date: datetime,
    end_date: datetime,
    issue_type: str,
):
    """
    Display the list of bug issues

    Args:
        issues: List of issue dictionaries
        owner: Repository owner
        repo: Repository name
        from_tag: Starting tag name
        to_tag: Ending tag name
        start_date: Start date
        end_date: End date
        issue_type: Issue type that was filtered on
    """
    print("\n" + "=" * 80)
    print("  BUG ISSUES CLOSED AS COMPLETED BETWEEN TAGS")
    print("=" * 80)
    print(f"\n  Repository:   {owner}/{repo}")
    print(f"  Tags:         {from_tag} -> {to_tag}")
    print(f"  Date Range:   {start_date.date()} to {end_date.date()}")
    print(f"  Issue Type:   {issue_type}")
    print(f"  Total Found:  {len(issues)}")

    if not issues:
        print("\n  No matching issues found.")
        return

    print("\n" + "-" * 80)
    for i, issue in enumerate(issues, 1):
        closed_at = datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00"))
        labels = ", ".join(lbl["name"] for lbl in issue.get("labels", []))
        print(f"\n  {i}. #{issue['number']}: {issue['title']}")
        print(f"     Closed:  {closed_at.date()}")
        print(f"     Labels:  {labels}")
        print(f"     URL:     {issue['html_url']}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def bugs_between_tags_main():
    """CLI entry point for bugs-between-tags"""
    parser = argparse.ArgumentParser(
        description="Fetch bug issues closed as completed between two release tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch bugs closed between two tags
  python3 github_utils.py bugs facebook/react --from-tag v18.0.0 --to-tag v18.2.0

  # Use a custom issue type name
  python3 github_utils.py bugs facebook/react --from-tag v18.0.0 --to-tag v18.2.0 --type bug
        """,
    )

    parser.add_argument(
        "repository", help="Repository in format owner/repo (e.g., facebook/react)"
    )
    parser.add_argument(
        "--from-tag", required=True, help="Starting release tag (e.g., v1.0.0)"
    )
    parser.add_argument(
        "--to-tag", required=True, help="Ending release tag (e.g., v1.1.0)"
    )
    parser.add_argument(
        "--type",
        default="Bug",
        help="Issue type name to filter by (default: Bug)",
    )

    args = parser.parse_args(sys.argv[2:])

    # Parse repository
    if "/" not in args.repository:
        print("Error: Repository must be in format owner/repo", file=sys.stderr)
        sys.exit(1)

    owner, repo = args.repository.split("/", 1)

    print("\n" + "=" * 80)
    print("  OSS Health Checker - Bugs Between Tags")
    print("=" * 80)
    print(f"\n  Repository: {owner}/{repo}")
    print(f"  Tags:       {args.from_tag} -> {args.to_tag}")
    print(f"  Type:       {args.type}")

    try:
        with GitHubClient() as client:
            print("\n  Looking up tag dates...")
            start_date = get_tag_date(client, owner, repo, args.from_tag)
            end_date = get_tag_date(client, owner, repo, args.to_tag)
            print(f"    {args.from_tag}: {start_date.date()}")
            print(f"    {args.to_tag}: {end_date.date()}")

            print("\n  Fetching closed bug issues...")
            issues = fetch_bug_issues(
                client, owner, repo, start_date, end_date, args.type
            )

            display_bug_results(
                issues,
                owner,
                repo,
                args.from_tag,
                args.to_tag,
                start_date,
                end_date,
                args.type,
            )

            print("\n  Analysis complete!\n")

    except ValueError as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        print(
            "\nPlease set GITHUB_TOKEN environment variable or create a .env file",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n  Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def list_contributors_main():
    """CLI entry point for list-contributors"""
    parser = argparse.ArgumentParser(
        description="List repository contributors with various filtering options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List contributors from last 30 days
  python3 github_utils.py contributors facebook/react --since 2024-11-01

  # List contributors in a date range
  python3 github_utils.py contributors facebook/react --since 2024-01-01 --until 2024-06-30

  # List contributors between two release tags
  python3 github_utils.py contributors facebook/react --from-tag v18.0.0 --to-tag v18.2.0
        """,
    )

    parser.add_argument(
        "repository", help="Repository in format owner/repo (e.g., facebook/react)"
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--since", help="Show contributors since this date (YYYY-MM-DD)"
    )
    filter_group.add_argument(
        "--from-tag", help="Show contributors from this tag onwards"
    )

    parser.add_argument(
        "--until",
        help="Show contributors until this date (YYYY-MM-DD). Used with --since",
    )
    parser.add_argument(
        "--to-tag", help="Show contributors up to this tag. Used with --from-tag"
    )
    parser.add_argument(
        "--branch",
        help="Filter to specific branch (e.g., main). Only works with date filtering, not tag filtering",
    )

    parser.add_argument(
        "--sort-by",
        choices=["commits", "name"],
        default="commits",
        help="Sort contributors by commit count or name (default: commits)",
    )
    parser.add_argument("--no-count", action="store_true", help="Hide commit counts")

    args = parser.parse_args(sys.argv[2:])

    # Parse repository
    if "/" not in args.repository:
        print("❌ Error: Repository must be in format owner/repo", file=sys.stderr)
        sys.exit(1)

    owner, repo = args.repository.split("/", 1)

    print("\n" + "=" * 80)
    print("  🔍 Contributor List")
    print("=" * 80)
    print(f"\n  Repository: {owner}/{repo}")

    try:
        with GitHubClient() as client:
            analyzer = ContributorAnalyzer(client, owner, repo)

            if args.from_tag and not args.to_tag:
                print(
                    "❌ Error: --to-tag is required when using --from-tag",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.to_tag and not args.from_tag:
                print(
                    "❌ Error: --from-tag is required when using --to-tag",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.branch and (args.from_tag or args.to_tag):
                print(
                    "❌ Error: --branch cannot be used with tag filtering (--from-tag/--to-tag)",
                    file=sys.stderr,
                )
                print(
                    "   Branch filtering only works with date filtering (--since/--until)",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.from_tag and args.to_tag:
                print(f"  Filter: Tags {args.from_tag} → {args.to_tag}")
                contributors = analyzer.get_contributors_by_tags(
                    args.from_tag, args.to_tag
                )
            else:
                if args.since or args.until or args.branch:
                    filter_desc = "Filter: "
                    if args.since:
                        filter_desc += f"Since {args.since}"
                    if args.until:
                        filter_desc += (
                            f" Until {args.until}"
                            if args.since
                            else f"Until {args.until}"
                        )
                    if args.branch:
                        filter_desc += (
                            f" Branch: {args.branch}"
                            if (args.since or args.until)
                            else f"Branch: {args.branch}"
                        )
                    print(f"  {filter_desc}")
                else:
                    print("  Filter: All time")

                contributors = analyzer.get_contributors_by_date_range(
                    args.since, args.until, args.branch
                )

            display_contributors(
                contributors, sort_by=args.sort_by, show_count=not args.no_count
            )

            print("\n✅ Analysis complete!\n")

    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        print(
            "\nPlease set GITHUB_TOKEN environment variable or create a .env file",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Unified CLI entry point with subcommands"""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python github_utils.py <command> [options]")  # noqa: T201
        print()
        print("Commands:")
        print("  bugs           Fetch bug issues closed between two release tags")
        print("  contributors   List repository contributors")
        print()
        print(
            "Run 'python github_utils.py <command> --help' for command-specific help."
        )
        sys.exit(0)

    command = sys.argv[1]

    if command == "bugs":
        bugs_between_tags_main()
    elif command == "contributors":
        list_contributors_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Use 'bugs' or 'contributors'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Analysis cancelled by user.")
        sys.exit(0)
