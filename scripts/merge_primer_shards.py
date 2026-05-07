#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Merge per-shard JSON files from compare_typecheckers.py into a single file.

Usage:
  python3 merge_primer_shards.py shard_0.json shard_1.json ... -o merged.json

Each shard file is the output of:
  python3 compare_typecheckers.py --shard-index N --num-shards M --output-json shard_N.json
"""

import argparse
import json
import logging
import os
import sys


def merge_shards(shard_paths: list[str]) -> dict[str, object]:
    """Merge multiple shard JSON files into a single combined result.

    Takes the timestamp from the first shard and concatenates all project
    entries. Deduplicates by project name (first shard wins if duplicated,
    though duplicates should not occur with correct sharding).
    """
    all_projects: list[dict[str, object]] = []
    timestamp = None
    seen: set[str] = set()

    for path in shard_paths:
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Malformed JSON in shard file {path}: {e}")
                raise
        if timestamp is None:
            timestamp = data.get("timestamp", "")
        for proj in data.get("projects", []):
            name = proj.get("name", "")
            if name not in seen:
                seen.add(name)
                all_projects.append(proj)
            else:
                logging.warning(f"Duplicate project '{name}' in {path}, skipping")

    return {
        "timestamp": timestamp or "",
        "projects": all_projects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-shard JSON outputs from compare_typecheckers.py"
    )
    parser.add_argument(
        "shards",
        nargs="+",
        help="Paths to shard JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for merged JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Validate all shard files exist
    missing = [p for p in args.shards if not os.path.exists(p)]
    if missing:
        logging.error(f"Missing shard files: {missing}")
        sys.exit(1)

    merged = merge_shards(args.shards)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    logging.info(
        f"Merged {len(merged['projects'])} projects from {len(args.shards)} shards "
        f"into {args.output} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
