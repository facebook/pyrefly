# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from suites import SUITES


def default_pyrefly(repo_root: Path) -> Path:
    target_dir = (
        Path(os.environ["CARGO_TARGET_DIR"])
        if "CARGO_TARGET_DIR" in os.environ
        else repo_root / "target"
    )
    executable = "pyrefly.exe" if os.name == "nt" else "pyrefly"
    return target_dir / "debug" / executable


def run_check(
    *,
    pyrefly_command: list[str],
    numpy_root: Path,
    numpy_stubs_root: Path,
    shape_extension_root: Path,
    suite: str,
    nocapture: bool,
) -> int:
    check_files = [file.as_posix() for file in SUITES[suite]]
    command = [
        *pyrefly_command,
        "check",
        "--config",
        os.devnull,
        "--python-version",
        "3.13",
        "--expectations",
        "--search-path",
        str(numpy_stubs_root),
        "--search-path",
        str(shape_extension_root),
    ]
    command.extend(check_files)

    if nocapture:
        print("+ " + " ".join(command), flush=True)
        return subprocess.run(command, cwd=numpy_root).returncode

    result = subprocess.run(
        command,
        cwd=numpy_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("+ " + " ".join(command), flush=True)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
    else:
        print(f"PASS {suite} ({len(check_files)} files)", flush=True)
    return result.returncode


def main() -> int:
    numpy_root_default = Path(__file__).resolve().parent
    repo_root_default = numpy_root_default.parent.parent
    tensor_shapes_root_default = (
        Path(os.environ["TENSOR_SHAPES_ROOT"])
        if "TENSOR_SHAPES_ROOT" in os.environ
        else repo_root_default / "tensor-shapes"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyrefly",
        type=Path,
        default=Path(os.environ["PYREFLY"])
        if "PYREFLY" in os.environ
        else default_pyrefly(repo_root_default),
    )
    parser.add_argument(
        "--buck",
        action="store_true",
        help="run Pyrefly through the internal Buck target instead of a local binary",
    )
    parser.add_argument("--numpy-root", type=Path, default=numpy_root_default)
    parser.add_argument(
        "--tensor-shapes-root",
        type=Path,
        default=tensor_shapes_root_default,
    )
    parser.add_argument(
        "--numpy-stubs-root",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--nocapture",
        action="store_true",
        help="stream Pyrefly output instead of printing it only on failure",
    )
    args = parser.parse_args()

    numpy_stubs_root = args.numpy_stubs_root or args.tensor_shapes_root
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
    suites = args.suite or ["all"]
    if "all" in suites:
        suites = list(SUITES)
    pyrefly_command = (
        ["buck2", "run", "fbcode//pyrefly:pyrefly", "--"]
        if args.buck
        else [str(args.pyrefly.resolve())]
    )

    for suite in suites:
        result = run_check(
            pyrefly_command=pyrefly_command,
            numpy_root=args.numpy_root.resolve(),
            numpy_stubs_root=numpy_stubs_root.resolve(),
            shape_extension_root=shape_extension_root.resolve(),
            suite=suite,
            nocapture=args.nocapture,
        )
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
