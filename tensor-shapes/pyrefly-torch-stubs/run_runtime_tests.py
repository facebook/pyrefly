# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SUITES = {
    "annotation": [
        "test_annotation_runtime.py",
        "test_annotation_runtime_future.py",
    ],
    "model": [
        "test_model_runtime.py",
    ],
}


def add_import_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def main() -> int:
    torch_root_default = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-root", type=Path, default=torch_root_default)
    parser.add_argument(
        "--suite",
        choices=("all",) + tuple(SUITES),
        action="append",
        default=[],
    )
    args = parser.parse_args()

    torch_root = args.torch_root.resolve()
    runtime_tests_root = torch_root / "test" / "runtime_tests"

    add_import_path(torch_root.parent / "pyrefly-shape-extensions")
    add_import_path(torch_root / "examples" / "runtime")

    suites = args.suite or ["all"]
    if "all" in suites:
        suites = list(SUITES)

    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    for suite in suites:
        for pattern in SUITES[suite]:
            test_suite.addTests(
                loader.discover(str(runtime_tests_root), pattern=pattern)
            )

    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
