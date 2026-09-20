# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import Suite  # noqa: E402

SUITES: list[Suite] = [
    Suite(name="stubs", patterns=("microtorch.pyi", "shape_extensions.pyi")),
    Suite(name="positive", patterns=("test/test_*.py",)),
    Suite(
        name="examples",
        patterns=("examples/*/sandbox.py",),
        expectations=True,
    ),
]
