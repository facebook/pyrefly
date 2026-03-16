#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run all deterministic unit tests for the issue ranker and compare_typecheckers.

Usage:
    python3 -m scripts.issue_ranker.tests.run_tests
    # or from the tests directory:
    python3 run_tests.py
"""

import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=__file__.rsplit("/", 1)[0],
        pattern="test_*.py",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
