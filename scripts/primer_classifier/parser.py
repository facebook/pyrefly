# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parse mypy_primer diff output into structured data.

The primer diff has two error formats per entry:

1. Concise format:
    ERROR path/to/file.py:10:5-20: some message [error-kind]

2. GitHub Actions format:
    ::error file=path/to/file.py,line=10,col=5,endLine=10,endColumn=20,title=Pyrefly error-kind::message

Projects are separated by blank lines. Lines starting with '+' are new errors
(on the PR branch), lines starting with '-' are removed errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ErrorEntry:
    """A single error line from the primer diff."""

    severity: str  # "ERROR", "WARN", "INFO"
    file_path: str  # relative path within the project
    location: str  # e.g. "10:5-20" or "2:5" or "2:5-4:10"
    message: str  # the error message text
    error_kind: str  # e.g. "bad-return", "missing-import"
    raw_line: str  # the original line from the diff

    @property
    def line_number(self) -> int:
        """Extract the starting line number from the location string."""
        return int(self.location.split(":")[0])


@dataclass
class ProjectDiff:
    """All added/removed errors for a single OSS project."""

    name: str
    url: Optional[str] = None  # GitHub URL if present
    added: list[ErrorEntry] = field(default_factory=list)
    removed: list[ErrorEntry] = field(default_factory=list)


# Matches the concise error format:
#   ERROR path/to/file.py:10:5-20: some message [error-kind]
#   WARN path/to/file.py:2:5: some message [error-kind]
# Severity labels: "ERROR" (5 chars), " WARN" (5 chars), " INFO" (5 chars)
_ERROR_RE = re.compile(
    r"^\s*(ERROR|WARN|INFO)\s+"  # severity (stripped of padding)
    r"(\S+)"  # file path
    r":"
    r"(\d+:\d+(?:-\d+(?::\d+)?)?)"  # location: line:col or line:col-endcol or line:col-endline:endcol
    r":\s+"
    r"(.*?)"  # message
    r"\s+\[([a-z][a-z0-9-]*)\]"  # [error-kind]
    r"\s*$"
)

# Matches the GitHub Actions annotation format:
#   ::error file=path.py,line=10,col=5,endLine=10,endColumn=20,title=Pyrefly error-kind::message
# The title contains "Pyrefly " followed by the error kind (e.g. "Pyrefly bad-return").
_GITHUB_ACTIONS_RE = re.compile(
    r"^::(?:error|warning|notice)\s+"
    r"file=([^,]+),"  # file path
    r"line=(\d+),"  # start line
    r"col=(\d+),"  # start column
    r"endLine=(\d+),"  # end line
    r"endColumn=(\d+),"  # end column
    r"title=Pyrefly\s+([a-z][a-z0-9-]*)"  # error kind from title
    r"::(.*)"  # message (everything after ::)
    r"$"
)

# Project header: "project_name" or "project_name (https://github.com/...)"
_PROJECT_HEADER_RE = re.compile(
    r"^([^\s(]+(?:\s+[^\s(]+)*?)"  # project name (non-greedy, avoids capturing URL)
    r"(?:\s+\((\S+)\))?"  # optional (url)
    r"\s*$"
)


def parse_error_line(line: str) -> Optional[ErrorEntry]:
    """Parse a single error line (without the +/- prefix) into an ErrorEntry.

    Handles both the concise format (ERROR file:line:col: msg [kind])
    and the GitHub Actions format (::error file=...,title=Pyrefly kind::msg).
    Returns None if the line doesn't match either format.
    """
    # Try concise format first
    m = _ERROR_RE.match(line)
    if m:
        return ErrorEntry(
            severity=m.group(1).strip(),
            file_path=m.group(2),
            location=m.group(3),
            message=m.group(4),
            error_kind=m.group(5),
            raw_line=line,
        )

    # Try GitHub Actions format
    m = _GITHUB_ACTIONS_RE.match(line)
    if m:
        file_path = m.group(1)
        start_line = m.group(2)
        start_col = m.group(3)
        end_line = m.group(4)
        end_col = m.group(5)
        error_kind = m.group(6)
        # The message may contain URL-encoded newlines (%0A); take first line
        message = m.group(7).split("%0A")[0].strip()

        # Build a location string consistent with the concise format
        if start_line == end_line:
            location = f"{start_line}:{start_col}-{end_col}"
        else:
            location = f"{start_line}:{start_col}-{end_line}:{end_col}"

        # Map ::error/::warning/::notice to severity
        if line.startswith("::error"):
            severity = "ERROR"
        elif line.startswith("::warning"):
            severity = "WARN"
        else:
            severity = "INFO"

        return ErrorEntry(
            severity=severity,
            file_path=file_path,
            location=location,
            message=message,
            error_kind=error_kind,
            raw_line=line,
        )

    return None


def _dedup_key(entry: ErrorEntry) -> tuple[str, str, str]:
    """Key for deduplicating entries that appear in both concise and ::error format."""
    return (entry.file_path, entry.location, entry.error_kind)


def parse_primer_diff(text: str) -> list[ProjectDiff]:
    """Parse the full primer diff text into a list of ProjectDiff objects.

    The diff is split into project sections by blank lines. The first line
    of each section is the project name (with optional GitHub URL). Subsequent
    lines starting with '+' or '-' are added/removed errors.

    Deduplicates entries that appear in both concise and GitHub Actions format.
    """
    if not text.strip():
        return []

    projects: list[ProjectDiff] = []
    # Split on blank lines to get project sections
    sections = re.split(r"\n\n+", text.strip())

    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # First line is the project header
        header_line = lines[0].strip()
        # Skip lines that look like diff lines (start with +/-)
        if header_line.startswith(("+", "-")):
            continue

        header_match = _PROJECT_HEADER_RE.match(header_line)
        if not header_match:
            continue

        project = ProjectDiff(
            name=header_match.group(1).strip(),
            url=header_match.group(2),
        )

        # Track seen entries to deduplicate concise vs ::error duplicates
        added_seen: set[tuple[str, str, str]] = set()
        removed_seen: set[tuple[str, str, str]] = set()

        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("+"):
                content = stripped[1:].strip()
                entry = parse_error_line(content)
                if entry:
                    key = _dedup_key(entry)
                    if key not in added_seen:
                        added_seen.add(key)
                        project.added.append(entry)
            elif stripped.startswith("-"):
                content = stripped[1:].strip()
                entry = parse_error_line(content)
                if entry:
                    key = _dedup_key(entry)
                    if key not in removed_seen:
                        removed_seen.add(key)
                        project.removed.append(entry)

        # Only include projects that have at least one change
        if project.added or project.removed:
            projects.append(project)

    return projects
