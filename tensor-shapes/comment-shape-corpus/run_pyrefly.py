#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check independent comment-shape demos without importing their real projects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CORPUS_ROOT: Path = Path(__file__).resolve().parent
TENSOR_SHAPES_ROOT: Path = CORPUS_ROOT.parent
NUMPY_STUBS_ROOT: Path = TENSOR_SHAPES_ROOT / "pyrefly-numpy-stubs"

sys.path.insert(0, str(TENSOR_SHAPES_ROOT))

from shape_testing import (  # noqa: E402
    check_suites,
    pyrefly_command,
    Suite,
    venv_python,
    venv_site_packages,
)


def projects() -> list[str]:
    """Each immediate directory with a README is an independently checked demo."""
    names = sorted(
        path.name
        for path in CORPUS_ROOT.iterdir()
        if path.is_dir() and (path / "README.md").is_file()
    )
    if not names:
        raise ValueError(f"no project demos under {CORPUS_ROOT}")
    return names


def suites(project: str) -> list[Suite]:
    return [
        Suite(
            name=f"{project}-modules",
            patterns=("*.pyi",),
            extra_search_paths=(NUMPY_STUBS_ROOT,),
        ),
        Suite(
            name=f"{project}-examples",
            patterns=("examples/*.py",),
            extra_search_paths=(NUMPY_STUBS_ROOT,),
            expectations=True,
        ),
    ]


def main() -> int:
    available = projects()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", choices=available, default=[])
    parser.add_argument("--pyrefly", type=Path, help="use an already-built Pyrefly")
    parser.add_argument("--buck", action="store_true", help="build Pyrefly with Buck")
    parser.add_argument("--release", action="store_true", help="use Cargo release mode")
    numpy = parser.add_mutually_exclusive_group()
    numpy.add_argument("--python", type=Path, help="NumPy virtualenv interpreter")
    numpy.add_argument(
        "--numpy-root",
        type=Path,
        help="directory containing canonical numpy sources/stubs; no installation needed",
    )
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args()

    if args.numpy_root is not None:
        numpy_root = args.numpy_root.resolve()
        if not (numpy_root / "numpy").is_dir():
            parser.error(f"{numpy_root} must contain a numpy directory")
    else:
        numpy_root = venv_site_packages(venv_python(args.python))
    pyrefly = pyrefly_command(
        explicit=args.pyrefly, buck=args.buck, release=args.release
    )

    failed = 0
    for project in args.project or available:
        # These are module excerpts, not PEP 561 packages. Their .pyi files are
        # the first explicit suite, instead of check_suites' *-stubs convention.
        result = check_suites(
            pyrefly=pyrefly,
            package_root=CORPUS_ROOT / project,
            suites=suites(project),
            check_stubs=False,
            nocapture=args.nocapture,
            site_package_paths=(numpy_root,),
        )
        if result:
            failed = result
    return failed


if __name__ == "__main__":
    sys.exit(main())
