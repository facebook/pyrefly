# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path


SUITES = {
    "arithmetic": (Path("test/test_arithmetic.py"),),
    "broadcasting": (Path("test/test_broadcasting.py"),),
    "creation-basics": (Path("test/test_creation_basics.py"),),
    "dtype-properties": (Path("test/test_dtype_properties.py"),),
    "examples": (Path("examples/stats.py"), Path("examples/physical_science.py")),
    "indexing": (Path("test/test_indexing.py"),),
    "linalg": (Path("test/test_linalg.py"),),
    "math-ufuncs": (Path("test/test_math_ufuncs.py"),),
    "random": (Path("test/test_random.py"),),
    "reductions": (Path("test/test_reductions.py"),),
}
