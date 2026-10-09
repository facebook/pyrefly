#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check that `Shaped` annotations are readable by checkers that ignore them.

The rest of the tensor-shape suites ask whether Pyrefly infers the right shapes.
These suites also ask whether a library annotated with `Shaped` stays readable
by tools that know nothing about shapes, so each case runs under Pyrefly, mypy,
pyright and ty.

Each case directory holds a library annotated with `Shaped` and consumers of it
that never import `shape_extensions`. File names decide which checkers run:

- `*_external.py` pins the types a checker that ignores the strings infers.
  Pyrefly skips it.
- `*_pyrefly.py` pins the shapes Pyrefly infers from the strings. The other
  checkers skip it.
- `*_real_numpy.py` pins what Pyrefly infers against real numpy. Only the
  real-numpy check runs it.
- Everything else runs under every checker.

Pyrefly runs with `pyrefly.toml`, against the Pyrefly array stubs. A case may
also have `pyrefly-real-numpy.toml`, which checks `downstream.py` and the
`*_real_numpy.py` files against real numpy, as a consumer without the stubs
would.

Requires the shared tensor-shapes virtualenv, which supplies both the array
libraries and the three external checkers. Run `bootstrap_venv.py` first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shape_testing import (
    pyrefly_command,
    resolve_executable,
    venv_python,
    venv_site_packages,
)

COMPATIBILITY_ROOT: Path = Path(__file__).resolve().parent
PACKAGE_ROOT: Path = COMPATIBILITY_ROOT.parent


def run(
    command: list[str], cwd: Path, env: dict[str, str] | None = None
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=None if env is None else {**os.environ, **env},
        )
    except FileNotFoundError:
        return (
            False,
            f"Checker {command[0]} not found; run tensor-shapes/bootstrap_venv.py",
        )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def check_case(
    case: Path, pyrefly: list[str], python: Path
) -> list[tuple[str, bool, str]]:
    """Run every checker over one case directory."""

    sources = sorted(path.name for path in case.glob("*.py"))
    real_numpy_sources = [name for name in sources if name.endswith("_real_numpy.py")]
    sources = [name for name in sources if name not in real_numpy_sources]
    pyrefly_sources = [name for name in sources if not name.endswith("_external.py")]
    external_sources = [name for name in sources if not name.endswith("_pyrefly.py")]
    site_packages = ["--site-packages-path", str(venv_site_packages(python))]
    bin_dir = python.parent

    checks: list[tuple[str, list[str], dict[str, str] | None]] = [
        (
            "pyrefly",
            [*pyrefly, "check", *pyrefly_sources, "--config", "pyrefly.toml"]
            + site_packages,
            None,
        ),
        # `--python-executable` and `--pythonpath` point mypy and pyright at the
        # virtualenv, so they resolve the real array libraries. `shape_extensions`
        # is not installed there, so its root goes on the import path separately.
        # Getting this wrong is not silent: the library's types collapse to `Any`
        # and the `assert_type` calls downstream fail.
        (
            "mypy",
            [
                str(resolve_executable(bin_dir / "mypy")),
                f"--cache-dir={os.devnull}",
                # Use the types from `shape_extensions` without reporting on it.
                # These suites are about the case files; the package has
                # pre-existing mypy findings of its own that are not.
                "--follow-imports=silent",
                "--python-executable",
                str(python),
                *external_sources,
            ],
            {"MYPYPATH": str(PACKAGE_ROOT)},
        ),
        (
            "pyright",
            [
                str(resolve_executable(bin_dir / "pyright")),
                "--pythonpath",
                str(python),
                *external_sources,
            ],
            {"PYTHONPATH": str(PACKAGE_ROOT)},
        ),
        # ty needs the version stated explicitly; it otherwise assumes an older
        # Python and cannot find `typing.assert_type`.
        (
            "ty",
            [
                str(resolve_executable(bin_dir / "ty")),
                "check",
                "--python",
                str(python),
                "--python-version",
                "3.13",
                "--extra-search-path",
                ".",
                "--extra-search-path",
                str(PACKAGE_ROOT),
                *external_sources,
            ],
            None,
        ),
    ]
    if (case / "pyrefly-real-numpy.toml").exists():
        checks.append(
            (
                "pyrefly (real numpy)",
                [
                    *pyrefly,
                    "check",
                    "downstream.py",
                    *real_numpy_sources,
                    "--config",
                    "pyrefly-real-numpy.toml",
                ]
                + site_packages,
                None,
            )
        )
    return [(name, *run(command, case, env)) for name, command, env in checks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", type=Path, help="Interpreter of the tensor-shapes virtualenv"
    )
    parser.add_argument("--pyrefly", type=Path, help="Pyrefly binary to use")
    parser.add_argument("--buck", action="store_true", help="Run Pyrefly through buck2")
    args = parser.parse_args()

    python = venv_python(args.python)
    pyrefly = pyrefly_command(explicit=args.pyrefly, buck=args.buck)

    failed = False
    for case in sorted(
        path
        for path in COMPATIBILITY_ROOT.iterdir()
        if path.is_dir() and (path / "pyrefly.toml").exists()
    ):
        print(f"\n{case.name}")
        for name, ok, output in check_case(case, pyrefly, python):
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                failed = True
                print("\n".join(f"        {line}" for line in output.splitlines()))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
