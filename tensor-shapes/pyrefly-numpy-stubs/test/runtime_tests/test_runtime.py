# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


class NumpyRuntimeTest(unittest.TestCase):
    def test_runtime_suites(self) -> None:
        numpy_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(numpy_root))
        try:
            import run_runtime_tests

            self.assertIn("examples", run_runtime_tests.SUITES)
            total_tests = run_runtime_tests.run_suites(numpy_root, ["all"])
        finally:
            sys.path.pop(0)

        self.assertGreater(total_tests, 0)

    def test_numpy_rejects_float_array_indices(self) -> None:
        logits = np.ones((5, 3))
        float_indices = np.ones(5)

        with self.assertRaises(IndexError):
            logits[float_indices, float_indices]

    def test_numpy_rejects_mismatched_paired_index_lengths(self) -> None:
        logits = np.ones((5, 3))
        rows = np.arange(2)
        columns = np.zeros(3, dtype=np.int64)

        with self.assertRaises(IndexError):
            logits[rows, columns]
