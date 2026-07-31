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


SUITES = (
    "torch-examples",
    "torch-positive",
    "torch-negative",
    "jaxtyping-positive",
    "jaxtyping-negative",
)


def default_pyrefly(repo_root: Path, release: bool) -> Path:
    target_dir = (
        Path(os.environ["CARGO_TARGET_DIR"])
        if "CARGO_TARGET_DIR" in os.environ
        else repo_root / "target"
    )
    executable = "pyrefly.exe" if os.name == "nt" else "pyrefly"
    return target_dir / ("release" if release else "debug") / executable


def resolve_pyrefly(pyrefly: Path) -> Path:
    """Resolve the pyrefly executable, tolerating a missing `.exe` suffix so
    callers (e.g. CI) can pass an OS-agnostic path."""
    if not pyrefly.exists():
        with_exe = pyrefly.with_name(pyrefly.name + ".exe")
        if with_exe.exists():
            return with_exe.resolve()
    return pyrefly.resolve()


def files(pattern: str, torch_root: Path) -> list[str]:
    return [
        str(path.relative_to(torch_root)) for path in sorted(torch_root.glob(pattern))
    ]


def run_check(
    *,
    pyrefly: Path,
    torch_root: Path,
    torch_stubs_root: Path,
    shape_extension_root: Path,
    suite: str,
    nocapture: bool,
) -> int:
    jaxtyping_root = torch_root / "test" / "jaxtyping"
    shape_search_roots = [torch_stubs_root, shape_extension_root]
    if suite == "torch-examples":
        python_version = "3.13"
        search_paths = shape_search_roots
        check_files = files("examples/*.py", torch_root) + files(
            "examples/runtime/*.py", torch_root
        )
        expectations = False
    elif suite == "torch-positive":
        python_version = "3.13"
        search_paths = shape_search_roots
        check_files = files("test/test_*.py", torch_root)
        expectations = False
    elif suite == "torch-negative":
        python_version = "3.13"
        search_paths = shape_search_roots
        check_files = files("test/negative_tests/test_*.py", torch_root)
        expectations = True
    elif suite == "jaxtyping-positive":
        python_version = "3.12"
        search_paths = [jaxtyping_root / "fixtures"] + shape_search_roots
        check_files = files("test/jaxtyping/test_*.py", torch_root)
        expectations = False
    elif suite == "jaxtyping-negative":
        python_version = "3.12"
        search_paths = [jaxtyping_root / "fixtures"] + shape_search_roots
        check_files = files("test/jaxtyping/negative_tests/test_*.py", torch_root)
        expectations = True
    else:
        raise ValueError(f"unknown suite: {suite}")

    if not check_files:
        raise ValueError(f"no files found for suite: {suite}")

    command = [
        str(pyrefly),
        "check",
        "--config",
        os.devnull,
        "--python-version",
        python_version,
    ]
    for search_path in search_paths:
        command.extend(["--search-path", str(search_path)])
    if expectations:
        command.append("--expectations")
    command.extend(check_files)

    if nocapture:
        print("+ " + " ".join(command), flush=True)
        return subprocess.run(command, cwd=torch_root).returncode

    result = subprocess.run(
        command,
        cwd=torch_root,
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
    torch_root_default = Path(__file__).resolve().parent
    tensor_shapes_root_default = torch_root_default.parent
    repo_root_default = tensor_shapes_root_default.parent
    parser = argparse.ArgumentParser()
    # Default deferred until after parsing so `--release` can pick the profile.
    parser.add_argument("--pyrefly", type=Path, default=None)
    parser.add_argument(
        "--release",
        action="store_true",
        help="resolve the default pyrefly from target/release instead of target/debug",
    )
    parser.add_argument("--torch-root", type=Path, default=torch_root_default)
    parser.add_argument(
        "--torch-stubs-root",
        type=Path,
        default=(
            Path(os.environ["TORCH_STUBS_ROOT"])
            if "TORCH_STUBS_ROOT" in os.environ
            else torch_root_default
        ),
    )
    parser.add_argument(
        "--shape-extension-root",
        type=Path,
        default=(
            Path(os.environ["SHAPE_EXTENSION_ROOT"])
            if "SHAPE_EXTENSION_ROOT" in os.environ
            else tensor_shapes_root_default / "pyrefly-shape-extensions"
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("all",) + SUITES,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--nocapture",
        action="store_true",
        help="stream Pyrefly output instead of printing it only on failure",
    )
    args = parser.parse_args()

    if args.pyrefly is not None:
        pyrefly = args.pyrefly
    elif "PYREFLY" in os.environ:
        pyrefly = Path(os.environ["PYREFLY"])
    else:
        pyrefly = default_pyrefly(repo_root_default, args.release)
    pyrefly = resolve_pyrefly(pyrefly)

    suites = args.suite or ["all"]
    if "all" in suites:
        suites = list(SUITES)

    for suite in suites:
        result = run_check(
            pyrefly=pyrefly,
            torch_root=args.torch_root.resolve(),
            torch_stubs_root=args.torch_stubs_root.resolve(),
            shape_extension_root=args.shape_extension_root.resolve(),
            suite=suite,
            nocapture=args.nocapture,
        )
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
