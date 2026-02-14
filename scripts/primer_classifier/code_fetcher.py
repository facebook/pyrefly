# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch source code from GitHub for error locations.

Uses the GitHub raw content API (raw.githubusercontent.com) to fetch files.
Sends the full file content (up to MAX_FILE_LINES) so the LLM has complete
context for classification. Also fetches referenced files (e.g. parent classes)
when they can be identified from the error message.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .parser import ErrorEntry, ProjectDiff


def _get_ssl_context() -> ssl.SSLContext:
    """Get an SSL context that works on macOS (certifi or system certs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    ctx = ssl.create_default_context()
    try:
        urllib.request.urlopen("https://raw.githubusercontent.com", timeout=5, context=ctx)
    except urllib.error.URLError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:
        pass
    return ctx


_ssl_ctx = _get_ssl_context()


@dataclass
class SourceContext:
    """Source code context for an error location."""

    file_content: str  # the full file content
    snippet: str  # the extracted context around the error
    start_line: int  # 1-based start line of the snippet
    end_line: int  # 1-based end line of the snippet
    error_line: int  # 1-based line of the error within the file


MAX_FILE_LINES = 1000


def _extract_context(content: str, error_line: int) -> tuple[str, int, int]:
    """Format the file content with the error line marked.

    Sends the whole file up to MAX_FILE_LINES. For files larger than that,
    centers a window around the error line.

    Returns (snippet, start_line, end_line) where lines are 1-based.
    """
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return "", 1, 1

    error_idx = max(0, min(error_line - 1, total - 1))

    if total <= MAX_FILE_LINES:
        start, end = 0, total - 1
    else:
        # Center window around the error line
        half = MAX_FILE_LINES // 2
        start = max(0, error_idx - half)
        end = min(total - 1, start + MAX_FILE_LINES - 1)
        if end == total - 1:
            start = max(0, end - MAX_FILE_LINES + 1)

    snippet_lines = []
    for i in range(start, end + 1):
        marker = " >>> " if i == error_idx else "     "
        snippet_lines.append(f"{i + 1:5d}{marker}{lines[i]}")

    return "\n".join(snippet_lines), start + 1, end + 1


def _github_url_to_owner_repo(url: str) -> Optional[tuple[str, str]]:
    """Extract (owner, repo) from a GitHub URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    return None


def _fetch_file_from_github(
    owner: str,
    repo: str,
    file_path: str,
    ref: str = "HEAD",
) -> Optional[str]:
    """Fetch a file from GitHub using the raw content URL.

    Uses GITHUB_TOKEN if available for higher rate limits.
    """
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    req = urllib.request.Request(url)

    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")

    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None


def _resolve_project_ref(owner: str, repo: str) -> str:
    """Get the default branch for a repo. Falls back to 'main'."""
    token = os.environ.get("GITHUB_TOKEN")
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("default_branch", "main")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return "main"


# Cache to avoid re-fetching the same file for multiple errors
_file_cache: dict[str, Optional[str]] = {}


def _extract_referenced_modules(entry: ErrorEntry) -> list[str]:
    """Extract module/class paths referenced in the error message.

    For errors like 'overrides parent class `_PythonVersionInfo`' or
    'function `fnmatch.fnmatch`', extract the dotted path and convert
    to a potential file path.
    """
    # Match backtick-quoted dotted names like `foo.bar.Baz`
    refs = re.findall(r"`([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)`", entry.message)
    paths = []
    for ref in refs:
        # Convert dotted module path to file path (e.g. foo.bar.baz -> foo/bar/baz.py)
        # Try progressively shorter prefixes (the last part might be a class/function)
        parts = ref.split(".")
        for i in range(len(parts), 0, -1):
            candidate = "/".join(parts[:i]) + ".py"
            if candidate != entry.file_path:
                paths.append(candidate)
                break
    return paths


def _fetch_and_cache(
    owner: str, repo: str, file_path: str, ref: str,
) -> Optional[str]:
    """Fetch a file with caching."""
    cache_key = f"{owner}/{repo}/{ref}/{file_path}"
    if cache_key in _file_cache:
        return _file_cache[cache_key]
    content = _fetch_file_from_github(owner, repo, file_path, ref)
    _file_cache[cache_key] = content
    return content


def fetch_source_context(
    project: ProjectDiff,
    entry: ErrorEntry,
    ref: Optional[str] = None,
) -> Optional[SourceContext]:
    """Fetch source code context for a single error entry.

    Fetches the full file containing the error. Also attempts to fetch
    files referenced in the error message (e.g. parent classes, called
    functions) and appends them to the snippet.

    Returns None if the project URL is not a GitHub URL or the file
    can't be fetched.
    """
    if not project.url:
        return None

    parsed = _github_url_to_owner_repo(project.url)
    if not parsed:
        return None

    owner, repo = parsed
    if ref is None:
        ref = _resolve_project_ref(owner, repo)

    content = _fetch_and_cache(owner, repo, entry.file_path, ref)
    if content is None:
        return None

    snippet, start, end = _extract_context(content, entry.line_number)

    # Try to fetch referenced files (parent classes, called functions)
    ref_paths = _extract_referenced_modules(entry)
    for ref_path in ref_paths[:3]:  # limit to 3 extra files
        ref_content = _fetch_and_cache(owner, repo, ref_path, ref)
        if ref_content:
            ref_lines = ref_content.splitlines()
            if len(ref_lines) <= MAX_FILE_LINES:
                numbered = "\n".join(
                    f"{i + 1:5d}     {line}" for i, line in enumerate(ref_lines)
                )
                snippet += f"\n\n--- Referenced file: {ref_path} ---\n{numbered}"

    return SourceContext(
        file_content=content,
        snippet=snippet,
        start_line=start,
        end_line=end,
        error_line=entry.line_number,
    )


def clear_cache() -> None:
    """Clear the file fetch cache."""
    _file_cache.clear()
