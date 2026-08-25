#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute the jax shape-stub suites against the real jax.

Run this with an interpreter that has jax installed -- normally the shared
virtualenv from bootstrap_venv.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import run_suites  # noqa: E402
from suites import SUITES  # noqa: E402


PACKAGE_ROOT: Path = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=[suite.name for suite in SUITES],
        help="run only the named suite; repeatable, defaults to all",
    )
    args = parser.parse_args()

    selected = (
        [suite for suite in SUITES if suite.name in args.suite]
        if args.suite
        else SUITES
    )
    run_suites(library="jax", package_root=PACKAGE_ROOT, suites=selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
