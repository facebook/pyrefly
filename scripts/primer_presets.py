#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Break a mypy_primer diff down by the preset that enables each error.

Which kinds a preset reports is read from `error_presets.json`.

The counts are approximate in one respect: presets also flip behavior settings
(`check-unannotated-defs`, `strict-callable-subtyping`, ...) that change which
errors get produced at all, not just which ones get displayed. An error that
only exists because `all` enables strict subtyping is still counted under
`default` if its kind is one `default` reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# Presets to report, in order.
PRESETS: tuple[str, ...] = ("default", "strict")

# Sits next to this module, both in the repo and in the Buck-built binaries
# that depend on it as a resource.
PRESETS_PATH: Path = Path(__file__).resolve().parent / "error_presets.json"

# Trailing `[error-kind]` tag on a Pyrefly error line. Both the internal
# runner's `format_diff` and mypy_primer's concise output end lines this way.
_KIND_TAG_RE = re.compile(r"\[([a-z][a-z0-9-]*)\]\s*$")


def summarize(added: Iterable[str], removed: Iterable[str]) -> str:
    """Format per-preset counts from added/removed error kind names.

    Returns e.g. `+7/-2 (default), +8/-2 (strict)`. An error kind missing from
    `error_presets.json` counts under no preset, which can only happen if the
    file is stale relative to the binary that produced the diff.
    """
    added = list(added)
    removed = list(removed)
    enabled_by = json.loads(PRESETS_PATH.read_text())["enabled_by"]
    parts = []
    for preset in PRESETS:
        a = sum(1 for kind in added if preset in enabled_by.get(kind, ()))
        r = sum(1 for kind in removed if preset in enabled_by.get(kind, ()))
        parts.append(f"+{a}/-{r} ({preset})")
    return ", ".join(parts)


def kinds_from_diff(text: str) -> tuple[list[str], list[str]]:
    """Pull added/removed error kind names out of a mypy_primer diff."""
    added: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(("+", "-")):
            continue
        match = _KIND_TAG_RE.search(line)
        if match:
            (added if line.startswith("+") else removed).append(match.group(1))
    return added, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diff_file", help="mypy_primer diff output, or - for stdin")
    args = parser.parse_args()
    if args.diff_file == "-":
        text = sys.stdin.read()
    else:
        with open(args.diff_file) as f:
            text = f.read()
    added, removed = kinds_from_diff(text)
    print(summarize(added, removed))


if __name__ == "__main__":
    main()
