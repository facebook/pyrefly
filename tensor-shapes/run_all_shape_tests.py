#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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
    # Both binaries are required: the Rust tests below shell out to `buck`, and
    # the stub suites reach Pyrefly through `buck2` in `shape_testing.py`.
    missing = [name for name in ("buck", "buck2") if shutil.which(name) is None]
    if mode == "buck" and missing:
        raise RuntimeError(
            f"buck mode requested, but {' and '.join(missing)} is not on PATH"
        )
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


def run_static_corpus(nocapture: bool, buck: bool) -> None:
    print_step("static tensor-shape corpus")
    run(
        [sys.executable, "tensor-shapes/run_tests.py", "--static-only"]
        + (["--buck"] if buck else [])
        + (["--nocapture"] if nocapture else [])
    )


def run_runtime_tests(buck: bool) -> None:
    print_step("runtime tests")
    run(
        [sys.executable, "tensor-shapes/run_tests.py", "--runtime-only"]
        + (["--buck"] if buck else [])
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


def run_shape_tests(
    *,
    mode: str,
    include_runtime_tests: bool,
    nocapture: bool,
) -> None:
    selected_mode = select_mode(mode)
    print(f"Using {selected_mode} mode.", flush=True)
    buck = selected_mode == "buck"
    if buck:
        run_buck_rust_tests()
    else:
        run_cargo_rust_tests()
    # The stub corpus runs through the same runner either way; the build tool
    # only decides where the Pyrefly binary comes from.
    run_static_corpus(nocapture, buck)
    if include_runtime_tests:
        run_runtime_tests(buck)


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
            "Stream Pyrefly output from the static tensor-shape corpus instead of "
            "printing it only on failure. Has no effect on the Rust unit-test filters."
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
