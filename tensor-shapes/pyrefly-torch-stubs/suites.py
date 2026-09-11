# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import Suite  # noqa: E402

_JAXTYPING_ROOT: Path = Path(__file__).resolve().parent / "test" / "jaxtyping"
_JAXTYPING_FIXTURES: Path = _JAXTYPING_ROOT / "fixtures"
_JAXTYPING_CONFIG: Path = _JAXTYPING_ROOT / "pyrefly.toml"

# Unlike the numpy and jax suites, these files are only type checked here; the
# torch runtime tests are separate unittest modules under test/runtime_tests.
SUITES: list[Suite] = [
    Suite(name="torch-examples", patterns=("examples/*.py", "examples/runtime/*.py")),
    Suite(name="torch-positive", patterns=("test/test_*.py",)),
    Suite(
        name="torch-negative",
        patterns=("test/negative_tests/test_*.py",),
        expectations=True,
    ),
    Suite(
        name="jaxtyping-positive",
        patterns=("test/jaxtyping/test_*.py",),
        python_version="3.12",
        config=_JAXTYPING_CONFIG,
        extra_search_paths=(_JAXTYPING_FIXTURES,),
    ),
    Suite(
        name="jaxtyping-negative",
        patterns=("test/jaxtyping/negative_tests/test_*.py",),
        python_version="3.12",
        config=_JAXTYPING_CONFIG,
        expectations=True,
        extra_search_paths=(_JAXTYPING_FIXTURES,),
    ),
]
