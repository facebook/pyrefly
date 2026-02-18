# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for the primer classifier.

Usage:
    python -m scripts.primer_classifier --diff-file path/to/diff.txt [options]

Options:
    --diff-file FILE     Path to the primer diff text file (required)
    --dry-run            Parse only, skip LLM classification
    --fetch-code         Fetch source code from GitHub (default: True with LLM)
    --no-fetch-code      Skip fetching source code
    --output-format FMT  Output format: "json" or "markdown" (default: markdown)
    --model MODEL        LLM model to use (default: depends on backend)
"""

from __future__ import annotations

import argparse
import sys

from .classifier import classify_all
from .formatter import format_json, format_markdown
from .parser import parse_primer_diff


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="primer_classifier",
        description="Classify mypy_primer diff output for pyrefly PRs",
    )
    parser.add_argument(
        "--diff-file",
        required=True,
        help="Path to the primer diff text file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and apply heuristics only, skip LLM classification",
    )
    parser.add_argument(
        "--fetch-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fetch source code from GitHub (default: enabled when using LLM)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use (default: depends on backend)",
    )

    args = parser.parse_args()

    # Read the diff file
    try:
        with open(args.diff_file) as f:
            diff_text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.diff_file}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    # Parse
    projects = parse_primer_diff(diff_text)
    if not projects:
        if args.output_format == "json":
            print('{"summary": {"total_projects": 0}, "classifications": []}')
        else:
            print("No diffs to classify.")
        return 0

    print(
        f"Parsed {len(projects)} project(s) from diff",
        file=sys.stderr,
    )

    # Determine fetch_code setting
    use_llm = not args.dry_run
    if args.fetch_code is None:
        fetch_code = use_llm  # fetch code when using LLM
    else:
        fetch_code = args.fetch_code

    # Classify
    result = classify_all(
        projects,
        fetch_code=fetch_code,
        use_llm=use_llm,
        model=args.model,
    )

    # Output
    if args.output_format == "json":
        print(format_json(result))
    else:
        print(format_markdown(result))

    # Return non-zero if there are regressions (useful for CI)
    return 1 if result.regressions > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
