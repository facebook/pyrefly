# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import Suite  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_STUB_ROOTS = tuple(
    PACKAGE_ROOT.parent / package
    for package in (
        "pyrefly-torch-stubs",
        "pyrefly-numpy-stubs",
        "pyrefly-jax-stubs",
    )
)

SUITES: list[Suite] = [
    Suite(
        name="operations",
        patterns=("test/test_*.py",),
        expectations=True,
        extra_search_paths=TEST_STUB_ROOTS,
    )
]
