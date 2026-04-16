#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build a rolling history of primer comparison results.

Reads the current run's primer_errors.json and an optional previous
primer_history.json, produces an updated history file with up to 7 days
of data, and renders a markdown summary with a trend table.

Usage:
  python3 primer_history.py --current /tmp/primer_errors.json \
      [--previous /tmp/prev/primer_history.json] \
      --output-history /tmp/primer_history.json \
      --output-markdown /tmp/primer_summary.md
"""

import argparse
import json
from datetime import datetime, timezone

MAX_HISTORY_DAYS = 7


def _to_int(val: object) -> int:
    """Coerce error count to int. 'ERR' or other non-int values become -1."""
    return val if isinstance(val, int) else -1


def load_current(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {"date": now, "projects": {}}
    for p in data["projects"]:
        name = p["name"]
        entry["projects"][name] = {
            "pyrefly": _to_int(p["pyrefly"]["error_count"]),
            "pyright": _to_int(p["pyright"]["error_count"]),
            "mypy": _to_int(p["mypy"]["error_count"]),
        }
    return entry


def load_history(path: str | None) -> list[dict]:
    if path is None:
        return []
    try:
        with open(path) as f:
            return json.load(f).get("history", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def build_history(previous: list[dict], current: dict) -> list[dict]:
    # Replace any existing entry for the same date
    history = [e for e in previous if e["date"] != current["date"]]
    history.append(current)
    # Keep only the last MAX_HISTORY_DAYS entries
    history.sort(key=lambda e: e["date"])
    return history[-MAX_HISTORY_DAYS:]


def render_markdown(history: list[dict]) -> str:
    lines = []
    dates = [e["date"] for e in history]

    # Totals trend table
    lines.append("# Primer Error Trend\n")
    header = "| Checker | " + " | ".join(dates) + " |"
    sep = "|---------|" + "|".join(["--------:"] * len(dates)) + "|"
    lines.append(header)
    lines.append(sep)

    for checker in ("pyrefly", "pyright", "mypy"):
        row = f"| {checker} |"
        for entry in history:
            total = sum(
                v[checker] for v in entry["projects"].values() if v[checker] > 0
            )
            row += f" {total:,} |"
        lines.append(row)

    # Delta from previous run
    if len(history) >= 2:
        prev = history[-2]
        curr = history[-1]
        lines.append(f"\n**Change from {prev['date']} to {curr['date']}:**\n")
        for checker in ("pyrefly", "pyright", "mypy"):
            prev_total = sum(
                v[checker] for v in prev["projects"].values() if v[checker] > 0
            )
            curr_total = sum(
                v[checker] for v in curr["projects"].values() if v[checker] > 0
            )
            delta = curr_total - prev_total
            sign = "+" if delta > 0 else ""
            lines.append(
                f"- {checker}: {sign}{delta:,} ({prev_total:,} → {curr_total:,})"
            )

    # Per-project table for latest run (all 3 checkers)
    latest = history[-1]
    sorted_projects = sorted(
        latest["projects"].items(),
        key=lambda x: x[1]["pyrefly"],
        reverse=True,
    )

    lines.append("\n## Per-project errors (latest run)\n")
    lines.append("| Project | Pyrefly | Pyright | Mypy |")
    lines.append("|---------|--------:|--------:|-----:|")
    for name, counts in sorted_projects:
        pe = "CRASH" if counts["pyrefly"] < 0 else f"{counts['pyrefly']:,}"
        pr = "CRASH" if counts["pyright"] < 0 else f"{counts['pyright']:,}"
        my = "CRASH" if counts["mypy"] < 0 else f"{counts['mypy']:,}"
        lines.append(f"| {name} | {pe} | {pr} | {my} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build primer history and render markdown"
    )
    parser.add_argument(
        "--current", required=True, help="Path to current primer_errors.json"
    )
    parser.add_argument("--previous", help="Path to previous primer_history.json")
    parser.add_argument(
        "--output-history",
        required=True,
        help="Output path for updated history JSON",
    )
    parser.add_argument(
        "--output-markdown",
        required=True,
        help="Output path for markdown summary",
    )
    args = parser.parse_args()

    current = load_current(args.current)
    previous = load_history(args.previous)
    history = build_history(previous, current)

    with open(args.output_history, "w") as f:
        json.dump({"history": history}, f, indent=2)

    md = render_markdown(history)
    with open(args.output_markdown, "w") as f:
        f.write(md)

    # Also print to stdout for piping to GITHUB_STEP_SUMMARY
    print(md)


if __name__ == "__main__":
    main()
