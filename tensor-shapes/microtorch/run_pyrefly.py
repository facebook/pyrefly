#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type check the microtorch shape suites with Pyrefly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import check_suites, pyrefly_command  # noqa: E402
from suites import SUITES  # noqa: E402

PACKAGE_ROOT: Path = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyrefly", type=Path, default=None)
    parser.add_argument("--buck", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=[suite.name for suite in SUITES],
    )
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args()

    selected = (
        [suite for suite in SUITES if suite.name in args.suite]
        if args.suite
        else SUITES
    )
    return check_suites(
        pyrefly=pyrefly_command(
            explicit=args.pyrefly, buck=args.buck, release=args.release
        ),
        package_root=PACKAGE_ROOT,
        suites=selected,
        nocapture=args.nocapture,
        check_stubs=False,
    )


if __name__ == "__main__":
    sys.exit(main())
