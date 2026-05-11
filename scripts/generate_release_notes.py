#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Release Notes Generator

Generates release notes for a GitHub repository by gathering commit, contributor,
and issue data between two refs (tags or branches), then using an LLM to produce
polished markdown from a user-provided template.

The to_ref argument can be a tag or a branch name. When it's a branch, the script
uses the latest commit on that branch, allowing you to prepare release notes before
the release tag is created.

Usage:
    python generate_release_notes.py facebook/pyrefly v0.1.0 v0.2.0
    python generate_release_notes.py facebook/pyrefly v0.1.0 main --version v0.2.0
    python generate_release_notes.py facebook/pyrefly v0.1.0 v0.2.0 -o notes.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_RELEASE_NOTES_DIR: Path = _SCRIPT_DIR.parent / "release_notes"

try:
    # Buck build: use fully-qualified module path
    from pyrefly.scripts.github_utils import (
        ContributorAnalyzer,
        get_tag_date,
        GitHubClient,
        load_env_file,
    )
except ImportError:
    # Standalone / GitHub Actions: fall back to relative import via sys.path
    sys.path.insert(0, str(_SCRIPT_DIR))
    from github_utils import (  # type: ignore[import-not-found]  # noqa: F811
        ContributorAnalyzer,  # pyrefly: ignore
        get_tag_date,  # pyrefly: ignore
        GitHubClient,  # pyrefly: ignore
        load_env_file,  # pyrefly: ignore
    )


def resolve_ref_date(
    client: GitHubClient, owner: str, repo: str, ref: str
) -> Tuple[datetime, str, Optional[str]]:
    """
    Resolve a git ref (tag or branch name) to its commit date.

    Tries to resolve as a tag first. If that fails, tries as a branch.

    Returns:
        Tuple of (datetime, ref_type, commit_sha) where ref_type is "tag" or
        "branch" and commit_sha is the HEAD commit SHA (only set for branches).
    """
    try:
        dt = get_tag_date(client, owner, repo, ref)
        return dt, "tag", None
    except ValueError:
        pass

    url = f"{client.base_url}/repos/{owner}/{repo}/branches/{ref}"
    try:
        branch = client._make_request(url)
        commit_sha: str = branch["commit"]["sha"]
        commit_date: str = branch["commit"]["commit"]["committer"]["date"]
        dt = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
        return dt, "branch", commit_sha
    except Exception:
        raise ValueError(f"'{ref}' is not a valid tag or branch in {owner}/{repo}")


def get_commits_between_tags(
    client: GitHubClient, owner: str, repo: str, from_tag: str, to_tag: str
) -> List[Dict[str, Any]]:
    """Get all commits between two tags using the GitHub compare API."""
    url = f"{client.base_url}/repos/{owner}/{repo}/compare/{from_tag}...{to_tag}"
    try:
        comparison = client._make_request(url)
        return comparison.get("commits", [])
    except Exception as e:
        raise ValueError(f"Could not compare tags {from_tag}...{to_tag}: {e}")


def get_closed_issues_between_dates(
    client: GitHubClient, owner: str, repo: str, since: str, until: str
) -> List[Dict[str, Any]]:
    """
    Get issues (not PRs) closed between two ISO 8601 date strings.

    Paginates through all results.
    """
    issues: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = (
            f"{client.base_url}/repos/{owner}/{repo}/issues"
            f"?state=closed&since={since}&per_page=100&page={page}"
            f"&sort=updated&direction=asc"
        )
        page_issues = client._make_request(url)
        if not page_issues:
            break

        for issue in page_issues:
            if "pull_request" in issue:
                continue
            closed_at = issue.get("closed_at", "")
            if closed_at and since <= closed_at <= until:
                issues.append(issue)

        if len(page_issues) < 100:
            break
        page += 1

    return issues


def filter_bug_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return issues identified as bugs by either issue type or label."""
    bug_issues: List[Dict[str, Any]] = []
    for issue in issues:
        issue_type = issue.get("type")
        if issue_type and issue_type.get("name", "").lower() == "bug":
            bug_issues.append(issue)
            continue

        labels = [label.get("name", "").lower() for label in issue.get("labels", [])]
        if "bug" in labels:
            bug_issues.append(issue)
    return bug_issues


# ---------------------------------------------------------------------------
# LLM provider helpers
# ---------------------------------------------------------------------------


def call_openai(user_prompt: str, system_prompt: str, api_key: str) -> str:
    """Generate text via the OpenAI Chat Completions API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps(
        {
            "model": "gpt-4o",
            "max_tokens": 16384,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode()

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
    return result["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_user_prompt(
    template: str,
    commits: List[Dict[str, Any]],
    contributors: Dict[str, Dict[str, Any]],
    bug_issues: List[Dict[str, Any]],
    all_closed_issues: List[Dict[str, Any]],
    from_tag: str,
    to_tag: str,
    owner: str,
    repo: str,
    archived_notes: str = "",
) -> str:
    """
    Assemble the user prompt that is sent to the LLM alongside the system prompt.

    Includes the template, raw commit log, issue lists, and contributor handles.
    """
    commit_lines: List[str] = []
    for c in commits:
        sha = c.get("sha", "")[:7]
        msg = c.get("commit", {}).get("message", "").strip()
        author = (
            c.get("author", {}).get("login", "unknown")
            if c.get("author")
            else "unknown"
        )
        indented_msg = msg.replace("\n", "\n  ")
        commit_lines.append(f"- {sha} ({author}): {indented_msg}")

    bug_lines: List[str] = []
    for issue in bug_issues:
        body = (issue.get("body") or "")[:500]
        bug_lines.append(f"- #{issue['number']}: {issue['title']}\n  Body: {body}")

    all_bug_numbers = ", ".join(f"#{i['number']}" for i in bug_issues)

    closed_lines = [
        f"- #{issue['number']}: {issue['title']}" for issue in all_closed_issues
    ]

    sorted_contribs = sorted(contributors.values(), key=lambda x: -x["commit_count"])
    contrib_parts: List[str] = []
    for c in sorted_contribs:
        if c["type"] == "github_user":
            contrib_parts.append(f"@{c['name']}")
        else:
            name = c["name"].split(" <")[0]
            contrib_parts.append(name)
    handles = ", ".join(contrib_parts)

    today = datetime.now().strftime("%B %d, %Y")

    return f"""Generate release notes for the following release.

**Project:** {owner}/{repo}
**Version:** {to_tag}
**Previous version:** {from_tag}
**Release date:** {today}
**Total commits:** {len(commits)}
**Total contributors:** {len(contributors)}
**Total issues closed:** {len(all_closed_issues)}
**Bug issues closed:** {len(bug_issues)}

---

## Markdown template to follow

```markdown
{template}
```

---

## Commits between {from_tag} and {to_tag} ({len(commits)} total)

{chr(10).join(commit_lines) if commit_lines else "No commits found."}

---

## Bug issues closed ({len(bug_issues)})

{chr(10).join(bug_lines) if bug_lines else "None"}

All bug issue numbers: {all_bug_numbers}

IMPORTANT: For the bug fixes section, pick the 10 most important/interesting bugs
from the list above and write detailed bullet points for those only. Then add a final
"And more!" bullet point listing ALL remaining issue numbers (the ones you did NOT
describe in detail) as a comma-separated list.

---

## All closed issues ({len(all_closed_issues)})

{chr(10).join(closed_lines) if closed_lines else "None"}

---

## Contributors

{handles}

---

## Previous release notes (for style and tone reference)

{archived_notes if archived_notes else "No archived release notes provided."}

---

Fill in every placeholder in the template using the data above. Output only the final
markdown — no commentary, no code fences wrapping the entire document.
"""


# ---------------------------------------------------------------------------
# Post-generation validation / formatting fixes
# ---------------------------------------------------------------------------


def _fix_table_bullets(content: str) -> Tuple[str, List[str]]:
    """Ensure each item in New & Improved table cells starts with '- '.

    The LLM sometimes omits the bullet prefix on some items or uses an
    inconsistent marker (e.g. '•').  This normalises every item to '- '.
    """
    warnings: List[str] = []
    lines = content.split("\n")
    result: List[str] = []

    for line in lines:
        if not (line.startswith("|") and "**" in line):
            result.append(line)
            continue

        cells = line.split("|")
        if len(cells) < 4:
            result.append(line)
            continue

        area_cell = cells[1]
        content_cell = cells[2]

        items = content_cell.split("<br><br>")
        fixed_items: List[str] = []
        needs_fix = False

        for item in items:
            stripped = item.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                fixed_items.append(stripped)
            elif stripped.startswith("\u2022 "):
                fixed_items.append("- " + stripped[2:])
                needs_fix = True
            else:
                fixed_items.append("- " + stripped)
                needs_fix = True

        if needs_fix:
            new_content = " <br><br>".join(fixed_items)
            new_line = f"|{area_cell}| {new_content} |"
            result.append(new_line)
            warnings.append(
                f"Fixed bullet formatting in table row: {area_cell.strip()}"
            )
        else:
            result.append(line)

    return "\n".join(result), warnings


def _fix_bug_fix_count(content: str, max_fixes: int = 10) -> Tuple[str, List[str]]:
    """Ensure no more than *max_fixes* detailed bug-fix bullet points.

    Any overflow bullets are collapsed into the "And more!" line.
    """
    warnings: List[str] = []
    lines = content.split("\n")

    bug_section_start: Optional[int] = None
    bug_section_end: Optional[int] = None
    for i, line in enumerate(lines):
        if "\U0001f41b" in line:  # Bug emoji
            bug_section_start = i
        elif bug_section_start is not None and (
            (line.startswith("## ") and "\U0001f41b" not in line)
            or (line.strip() == "---" and i > bug_section_start + 2)
        ):
            bug_section_end = i
            break

    if bug_section_start is None:
        return content, warnings
    if bug_section_end is None:
        bug_section_end = len(lines)

    section_lines = lines[bug_section_start:bug_section_end]

    header_lines: List[str] = []
    bug_bullets: List[str] = []
    and_more_line: Optional[str] = None
    trailing_lines: List[str] = []

    state = "header"
    for line in section_lines:
        if state == "header":
            if re.match(r"^- #\d+", line):
                bug_bullets.append(line)
                state = "bullets"
            else:
                header_lines.append(line)
        elif state == "bullets":
            if re.match(r"^- #\d+", line):
                bug_bullets.append(line)
            elif line.startswith("- And more!"):
                and_more_line = line
                state = "trailing"
            else:
                trailing_lines.append(line)
                state = "trailing"
        else:
            trailing_lines.append(line)

    if len(bug_bullets) <= max_fixes:
        return content, warnings

    keep = bug_bullets[:max_fixes]
    overflow = bug_bullets[max_fixes:]

    overflow_numbers: List[str] = []
    for b in overflow:
        match = re.match(r"^- #(\d+)", b)
        if match:
            overflow_numbers.append(f"#{match.group(1)}")

    existing_numbers: List[str] = []
    if and_more_line:
        existing_numbers = re.findall(r"#\d+", and_more_line)

    all_remaining = overflow_numbers + existing_numbers
    new_and_more = f"- And more! {', '.join(all_remaining)}"

    warnings.append(
        f"Reduced bug-fix bullets from {len(bug_bullets)} to {max_fixes} "
        f"(moved {len(overflow)} to 'And more!')"
    )

    new_section = header_lines + keep + [new_and_more]
    return (
        "\n".join(
            lines[:bug_section_start]
            + new_section
            + trailing_lines
            + lines[bug_section_end:]
        ),
        warnings,
    )


def validate_and_fix_release_notes(
    content: str, max_bug_fixes: int = 10
) -> Tuple[str, List[str]]:
    """Run post-generation formatting checks and fixes on the release notes.

    Fixes applied:
    1. Ensures all items in "New & Improved" table cells have a '- ' prefix.
    2. Caps detailed bug-fix bullet points at *max_bug_fixes*, moving extras
       into the "And more!" line.

    Returns:
        Tuple of (fixed_content, list_of_warnings).
    """
    warnings: List[str] = []

    content, w = _fix_table_bullets(content)
    warnings.extend(w)

    content, w = _fix_bug_fix_count(content, max_bug_fixes)
    warnings.extend(w)

    return content, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate AI-powered release notes from GitHub data "
            "between two refs (tags or branches)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Between two tags
  python generate_release_notes.py facebook/pyrefly v0.1.0 v0.2.0

  # Pre-release: from a tag to a branch (draft notes before tagging)
  python generate_release_notes.py facebook/pyrefly v0.1.0 main --version v0.2.0

  # Custom output path
  python generate_release_notes.py facebook/pyrefly v0.1.0 v0.2.0 -o notes.md

Environment variables:
  GITHUB_TOKEN        GitHub personal access token (or set in .env)
  OPENAI_API_KEY      Required when --provider is openai (default)
        """,
    )

    parser.add_argument(
        "repo",
        help="Repository in 'owner/repo' format (e.g. facebook/pyrefly)",
    )
    parser.add_argument("from_tag", help="Starting tag (exclusive)")
    parser.add_argument(
        "to_ref",
        help="Ending tag or branch name (e.g. v0.2.0 or main)",
    )
    parser.add_argument(
        "--version",
        help=(
            "Version string for the release notes. "
            "Defaults to to_ref. Useful when to_ref is a branch name "
            "(e.g. --version v0.2.0 when to_ref is 'main')."
        ),
    )
    parser.add_argument(
        "--template",
        help=(
            "Path to a markdown template file. "
            "Defaults to release_notes_template.md in the release_notes directory."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Path to a custom system prompt file. "
            "Defaults to prompt.md in the release_notes directory."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: release-notes-<version>.md)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="LLM provider (default: openai)",
    )

    return parser.parse_args()


def _get_llm_config(
    provider: str,
) -> Tuple[str, Callable[[str, str, str], str]]:
    """Resolve the LLM provider to an API key and generate function."""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY not set. "
                "Add it to your .env file or export it as an environment variable.",
                file=sys.stderr,
            )
            sys.exit(1)
        return api_key, call_openai

    print(f"Error: Unknown provider '{provider}'.", file=sys.stderr)
    sys.exit(1)


def _load_files(
    args: argparse.Namespace,
) -> Tuple[str, str, Path]:
    """Load template, system prompt, and resolve output path."""
    template_path = (
        Path(args.template)
        if args.template
        else _RELEASE_NOTES_DIR / "release_notes_template.md"
    )
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    template = template_path.read_text()

    prompt_path = Path(args.prompt) if args.prompt else _RELEASE_NOTES_DIR / "prompt.md"
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text().strip()

    version = args.version or args.to_ref
    safe_tag = version.replace("/", "-")
    output_path = (
        Path(args.output)
        if args.output
        else _RELEASE_NOTES_DIR / f"release-notes-{safe_tag}.md"
    )

    return template, system_prompt, output_path


def run_generate(
    owner: str,
    repo: str,
    from_tag: str,
    to_ref: str,
    version: str,
    template: str,
    system_prompt: str,
    output_path: Path,
    generate_fn: Callable[[str, str, str], str],
    api_key: str,
) -> None:
    """Core release-notes generation logic.

    Gathers data from GitHub, builds the LLM prompt, calls the provider,
    validates the output, and writes the result. This function is used by
    both the OSS CLI and the internal Meta wrapper.
    """
    with GitHubClient() as client:
        print(
            f"\nGenerating release notes for {owner}/{repo}  {from_tag} -> {to_ref}\n"
        )

        print("  Resolving refs...")
        from_dt = get_tag_date(client, owner, repo, from_tag)
        to_dt, to_ref_type, to_sha = resolve_ref_date(client, owner, repo, to_ref)

        if to_ref_type == "branch":
            assert to_sha is not None
            print(f"    {from_tag} (tag): {from_dt.date()}")
            print(f"    {to_ref} (branch HEAD): {to_dt.date()} @ {to_sha[:10]}")
            to_date = datetime.now(timezone.utc).isoformat()
        else:
            print(f"    {from_tag}: {from_dt.date()}")
            print(f"    {to_ref}: {to_dt.date()}")
            to_date = to_dt.isoformat()

        from_date = from_dt.isoformat()

        print("  Fetching commits...")
        commits = get_commits_between_tags(client, owner, repo, from_tag, to_ref)
        print(f"  Found {len(commits)} commits")

        contributor_analyzer = ContributorAnalyzer(client, owner, repo)
        contributors = contributor_analyzer._extract_contributors(commits)
        print(f"  Found {len(contributors)} contributors")

        print("  Fetching closed issues...")
        all_closed_issues = get_closed_issues_between_dates(
            client, owner, repo, from_date, to_date
        )
        bug_issues = filter_bug_issues(all_closed_issues)
        print(
            f"  Found {len(all_closed_issues)} closed issues ({len(bug_issues)} bugs)"
        )

        archived_notes_path = _RELEASE_NOTES_DIR / "release_notes_archived.md"
        archived_notes = ""
        if archived_notes_path.exists():
            archived_notes = archived_notes_path.read_text()
            print("  Loaded archived release notes for style reference")

        user_prompt = build_user_prompt(
            template,
            commits,
            contributors,
            bug_issues,
            all_closed_issues,
            from_tag,
            version,
            owner,
            repo,
            archived_notes,
        )

        print("\n  Generating release notes with LLM...")
        result = generate_fn(user_prompt, system_prompt, api_key)

        release_notes, fix_warnings = validate_and_fix_release_notes(result)
        if fix_warnings:
            print("\n  Post-generation fixes applied:")
            for w in fix_warnings:
                print(f"    - {w}")

        if not release_notes.endswith("\n"):
            release_notes += "\n"
        output_path.write_text(release_notes)
        print(f"\n  Release notes written to: {output_path}\n")


def main() -> None:
    args = _parse_args()

    if "/" not in args.repo:
        print("Error: Repository must be in 'owner/repo' format.", file=sys.stderr)
        sys.exit(1)
    owner, repo = args.repo.split("/", 1)

    load_env_file()
    api_key, generate_fn = _get_llm_config(args.provider)
    template, system_prompt, output_path = _load_files(args)
    version = args.version or args.to_ref

    try:
        run_generate(
            owner,
            repo,
            args.from_tag,
            args.to_ref,
            version,
            template,
            system_prompt,
            output_path,
            generate_fn,
            api_key,
        )
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"\nHTTP Error {e.code}: {e.reason}", file=sys.stderr)
        if body:
            try:
                detail = json.loads(body)
                print(f"  {json.dumps(detail, indent=2)[:500]}", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"  {body[:500]}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
