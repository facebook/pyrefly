#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run every tensor-shape stub test: static and runtime, for every library.

This is the single entry point CI uses, internally and on GitHub, so that all
of the shape coverage lands in one job rather than one job per library. The
per-package `run_pyrefly.py` and `run_runtime_tests.py` remain the things to
reach for while iterating on a single library.

Requires a Pyrefly binary, and the shared virtualenv from bootstrap_venv.py for
the runtime half. `--static-only` drops the virtualenv requirement entirely,
which is the usual mode when changing Pyrefly rather than the stubs. Nothing
here downloads anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from shape_testing import pyrefly_command, TENSOR_SHAPES_ROOT, venv_python


PACKAGES: tuple[str, ...] = (
    "pyrefly-torch-stubs",
    "pyrefly-numpy-stubs",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pyrefly", type=Path, default=None)
    parser.add_argument(
        "--buck",
        action="store_true",
        help="run Pyrefly out of Buck instead of a locally built binary",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="use the Cargo release build instead of debug",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="interpreter with torch/numpy/jax installed (default: the shared virtualenv)",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="only type check; needs no virtualenv",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="only execute the suites against the real libraries",
    )
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args()

    if args.static_only and args.runtime_only:
        raise SystemExit("--static-only and --runtime-only are mutually exclusive")

    # Resolve both toolchains before running anything, so a missing virtualenv
    # fails immediately rather than after several minutes of type checking.
    pyrefly = (
        None
        if args.runtime_only
        else pyrefly_command(
            explicit=args.pyrefly, buck=args.buck, release=args.release
        )
    )
    python = (
        None
        if args.static_only
        else venv_python(
            args.python,
            extra_hint=(
                "Pass --static-only to run just the type checking, which needs no\n"
                "virtualenv. That is usually the right mode when working on Pyrefly\n"
                "itself rather than on the stubs, since CI runs the runtime tests.\n\n"
            ),
        )
    )

    failures: list[str] = []
    for package in PACKAGES:
        package_root = TENSOR_SHAPES_ROOT / package
        if pyrefly is not None:
            step = f"{package} static"
            print(f"\n=== {step} ===", flush=True)
            command = [sys.executable, str(package_root / "run_pyrefly.py")]
            # Forward the already-resolved binary rather than re-passing the
            # flags. `--pyrefly`, $PYREFLY and $CARGO_TARGET_DIR may all be
            # relative to this process's directory, and the child runs from a
            # different one. A single-element command is a binary path; anything
            # longer is the `buck2 run` invocation, which needs no resolving.
            if len(pyrefly) == 1:
                command.extend(["--pyrefly", pyrefly[0]])
            else:
                command.append("--buck")
            if args.nocapture:
                command.append("--nocapture")
            if not run(command):
                failures.append(step)
        if python is not None:
            step = f"{package} runtime"
            print(f"\n=== {step} ===", flush=True)
            if not run([str(python), str(package_root / "run_runtime_tests.py")]):
                failures.append(step)

    if failures:
        print("\nFAILED: " + ", ".join(failures), file=sys.stderr, flush=True)
        return 1
    print("\nAll tensor-shape tests passed.", flush=True)
    return 0


def run(command: list[str]) -> bool:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=TENSOR_SHAPES_ROOT).returncode == 0


if __name__ == "__main__":
    sys.exit(main())
