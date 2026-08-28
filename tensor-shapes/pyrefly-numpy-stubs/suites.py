# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import Suite  # noqa: E402


_PACKAGE_ROOT: Path = Path(__file__).resolve().parent


def _suites() -> list[Suite]:
    """One suite per test file, discovered rather than enumerated.

    Listing the files by hand would let a new `test/test_*.py` be silently
    neither type checked nor executed while the runner still reported success.
    Names are derived from the file name, so `test_math_ufuncs.py` stays
    addressable as `--suite math-ufuncs`.

    Every numpy suite sets `expectations`: these files pair a `# E:` marker with
    the runtime error NumPy itself raises, so both halves of a rejection are
    asserted together.
    """

    paths = sorted((_PACKAGE_ROOT / "test").glob("test_*.py"))
    if not paths:
        raise ValueError(f"no test files under {_PACKAGE_ROOT / 'test'}")
    suites = [
        Suite(
            name=path.stem.removeprefix("test_").replace("_", "-"),
            patterns=(f"test/{path.name}",),
            expectations=True,
        )
        for path in paths
    ]
    suites.append(
        Suite(name="examples", expectations=True, patterns=("examples/*.py",))
    )
    return suites


SUITES: list[Suite] = _suites()
