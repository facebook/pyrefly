#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path


SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SCRIPT_DIR.parent

RUST_TEST_FILTERS: tuple[str, ...] = (
    "shaped_array",
    "shape_dsl",
    "jaxtyping",
    "test_intvar_type_parameter_marker_imports_are_used",
    "test_tensor_shapes",
    "pytorch_efficiency_lint",
    "expand_with_bounds",
)

# `pyrefly_types` shape tests live in modules and under names that do not all
# contain "shape" (e.g. the `dimension` canonicalization module and `int`
# display tests), so match on several substrings rather than "shape" alone.
TYPES_TEST_FILTERS: tuple[str, ...] = ("shape", "int", "dimension")

BUCK_TYPES_TARGET: str = "fbcode//pyrefly/crates/pyrefly_types:pyrefly_types"
BUCK_RUST_TARGET: str = "pyrefly:pyrefly_library"

BUCK_STATIC_CORPUS_TARGETS: tuple[str, ...] = (
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/examples:torch_examples_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_all_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_error_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_error_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_arithmetic_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_broadcasting_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_creation_basics_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_dtype_properties_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_examples_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_indexing_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_linalg_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_math_ufuncs_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_random_static_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_reductions_static_test",
)

BUCK_RUNTIME_TARGETS: tuple[str, ...] = (
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:annotation_runtime_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/test:model_runtime_test",
    "fbcode//pyrefly/tensor-shapes/pyrefly-numpy-stubs:numpy_runtime_test",
)


def print_step(message: str) -> None:
    print(f"\033[92mRunning {message}...\033[0m", flush=True)


def run(args: Sequence[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    start = time.time()
    subprocess.run(args, cwd=REPO_ROOT, check=True)
    print(f"Finished in {time.time() - start:.2f} seconds.", flush=True)


def select_mode(mode: str) -> str:
    if mode == "auto":
        mode = "cargo"
    if mode == "cargo" and shutil.which("cargo") is None:
        print("cargo is not on PATH; falling back to buck mode.", flush=True)
        mode = "buck"
    if mode == "buck" and shutil.which("buck") is None:
        raise RuntimeError("buck mode requested, but `buck` is not on PATH")
    return mode


def run_cargo_rust_tests() -> None:
    print_step("Cargo build")
    run(["cargo", "build", "-p", "pyrefly"])
    print_step("Cargo pyrefly_types shape tests")
    run(["cargo", "test", "-p", "pyrefly_types", "--", *TYPES_TEST_FILTERS])
    for test_filter in RUST_TEST_FILTERS:
        print_step(f"Cargo Rust tests matching {test_filter}")
        run(
            [
                "cargo",
                "test",
                "-p",
                "pyrefly",
                "--lib",
                test_filter,
                "--",
                "--include-ignored",
            ]
        )


def run_cargo_static_corpus(nocapture: bool) -> None:
    extra_args = ["--nocapture"] if nocapture else []
    print_step("torch static tensor-shape corpus")
    run(
        [
            sys.executable,
            "tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py",
            *extra_args,
        ]
    )
    print_step("numpy static tensor-shape corpus")
    run(
        [
            sys.executable,
            "tensor-shapes/pyrefly-numpy-stubs/run_pyrefly.py",
            *extra_args,
        ]
    )


def run_cargo_runtime_tests() -> None:
    print_step("torch runtime tests")
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "tensor-shapes/pyrefly-torch-stubs/test/runtime_tests",
        ]
    )
    print_step("numpy runtime tests")
    run(
        [
            sys.executable,
            "tensor-shapes/pyrefly-numpy-stubs/run_runtime_tests.py",
        ]
    )


def run_buck_rust_tests() -> None:
    print_step("Buck pyrefly_types shape tests")
    run(["buck", "test", BUCK_TYPES_TARGET, "--", *TYPES_TEST_FILTERS])
    for test_filter in RUST_TEST_FILTERS:
        print_step(f"Buck Rust tests matching {test_filter}")
        run(
            [
                "buck",
                "test",
                BUCK_RUST_TARGET,
                "--",
                test_filter,
                "--run-disabled",
                "--return-zero-on-skips",
            ]
        )


def run_buck_corpus(include_runtime_tests: bool) -> None:
    print_step("Buck tensor-shape corpus")
    targets = list(BUCK_STATIC_CORPUS_TARGETS)
    if include_runtime_tests:
        targets.extend(BUCK_RUNTIME_TARGETS)
    run(["buck", "test", *targets, "--", "--run-disabled", "--return-zero-on-skips"])


def run_shape_tests(
    *,
    mode: str,
    include_runtime_tests: bool,
    nocapture: bool,
) -> None:
    selected_mode = select_mode(mode)
    print(f"Using {selected_mode} mode.", flush=True)
    if selected_mode == "cargo":
        run_cargo_rust_tests()
        run_cargo_static_corpus(nocapture)
        if include_runtime_tests:
            run_cargo_runtime_tests()
    else:
        run_buck_rust_tests()
        run_buck_corpus(include_runtime_tests)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pyrefly's shape-relevant Rust and tensor-shape corpus tests."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "cargo", "buck"),
        default="auto",
        help=(
            "Test runner mode. The default prefers cargo and falls back to buck "
            "when cargo is not on PATH."
        ),
    )
    parser.add_argument(
        "--include-runtime-tests",
        action="store_true",
        help="Also run tensor-shape runtime tests. These are slower and are off by default.",
    )
    parser.add_argument(
        "--nocapture",
        action="store_true",
        help=(
            "Stream Pyrefly output from the cargo-mode static tensor-shape corpus "
            "runners. Has no effect on the Rust unit-test filters or on buck mode."
        ),
    )
    args = parser.parse_args()
    run_shape_tests(
        mode=args.mode,
        include_runtime_tests=args.include_runtime_tests,
        nocapture=args.nocapture,
    )


if __name__ == "__main__":
    main()
