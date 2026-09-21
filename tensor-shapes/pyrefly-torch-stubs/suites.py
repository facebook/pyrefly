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
_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_STATIC_ONLY_TESTS: set[str] = {
    "test_module_forward_attribute.py",
    "test_tensor_base_members.py",
    "test_tensor_constructor.py",
}

OPERATION_SUITES: list[Suite] = [
    Suite(
        name=f"torch-{path.stem.removeprefix('test_').replace('_', '-')}",
        patterns=(f"test/{path.name}",),
        expectations=True,
        strict_callable_subtyping=True,
    )
    for path in sorted((_PACKAGE_ROOT / "test").glob("test_*.py"))
    if path.name not in _STATIC_ONLY_TESTS
]

STATIC_SUITES: list[Suite] = [
    Suite(
        name=f"torch-{path.removeprefix('test_').removesuffix('.py').replace('_', '-')}",
        patterns=(f"test/{path}",),
        expectations=True,
        strict_callable_subtyping=True,
    )
    for path in sorted(_STATIC_ONLY_TESTS)
]

# Root operation suites run both here and through `run_runtime_tests.py`.
# `expectations=True` makes every `# E:` marker part of the static assertion.
# Legacy suites under `test/old` remain static-only until they are migrated.
SUITES: list[Suite] = [
    *OPERATION_SUITES,
    *STATIC_SUITES,
    Suite(name="torch-examples", patterns=("examples/*.py", "examples/runtime/*.py")),
    Suite(
        name="torch-positive",
        patterns=("test/old/test_*.py",),
        strict_callable_subtyping=True,
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
