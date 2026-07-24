# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from suites import SUITES


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_shape_extensions(shape_extension_root: Path) -> None:
    if "shape_extensions" not in sys.modules:
        load_module(
            "shape_extensions",
            shape_extension_root / "shape_extensions" / "__init__.py",
        )


def run_test_file(path: Path) -> int:
    import shape_extensions

    current_test: str | None = None
    assert_shape_calls: dict[str, int] = {}
    original_assert_shape = shape_extensions.assert_shape

    def assert_shape(x: Any, shape: Any) -> Any:
        if current_test is not None:
            assert_shape_calls[current_test] += 1
        return original_assert_shape(x, shape)

    # Patch before importing the test module so `from shape_extensions import assert_shape`
    # binds the wrapper that counts runtime assertions.
    shape_extensions.assert_shape = assert_shape
    try:
        module = load_module(f"_numpy_shape_test_{path.stem}", path)
        tests = [
            (name, value)
            for name, value in sorted(vars(module).items())
            if name.startswith("test_") and callable(value)
        ]
        if not tests:
            raise AssertionError(f"{path} does not define any test functions")
        for name, test in tests:
            current_test = name
            assert_shape_calls[name] = 0
            test()
            current_test = None
            if assert_shape_calls[name] == 0:
                raise AssertionError(f"{path}::{name} did not execute assert_shape")
    finally:
        shape_extensions.assert_shape = original_assert_shape

    total_assert_shape_calls = sum(assert_shape_calls.values())
    print(f"PASS {path.name} ({len(tests)} tests, {total_assert_shape_calls} shapes)")
    return len(tests)


def run_suites(numpy_root: Path, suites: list[str]) -> int:
    if "all" in suites:
        suites = list(SUITES)

    total_tests = 0
    for suite in suites:
        for filename in SUITES[suite]:
            total_tests += run_test_file(numpy_root.resolve() / filename)
    print(f"PASS {len(suites)} suites ({total_tests} tests)")
    return total_tests


def main() -> int:
    numpy_root_default = Path(__file__).resolve().parent
    tensor_shapes_root_default = (
        Path(os.environ["TENSOR_SHAPES_ROOT"])
        if "TENSOR_SHAPES_ROOT" in os.environ
        else numpy_root_default.parent
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-root", type=Path, default=numpy_root_default)
    parser.add_argument(
        "--tensor-shapes-root",
        type=Path,
        default=tensor_shapes_root_default,
    )
    parser.add_argument(
        "--shape-extension-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--suite",
        choices=("all",) + tuple(SUITES),
        action="append",
        default=[],
    )
    args = parser.parse_args()

    if args.shape_extension_root is None:
        moved_shape_extension_root = (
            args.tensor_shapes_root / "pyrefly-shape-extensions"
        )
        shape_extension_root = (
            moved_shape_extension_root
            if moved_shape_extension_root.exists()
            else args.tensor_shapes_root
        )
    else:
        shape_extension_root = args.shape_extension_root

    load_shape_extensions(shape_extension_root.resolve())
    suites = args.suite or ["all"]
    run_suites(args.numpy_root, suites)
    return 0


if __name__ == "__main__":
    sys.exit(main())
