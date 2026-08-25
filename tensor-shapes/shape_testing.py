# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared harness for the tensor-shape stub test suites.

Each stub package validates its stubs two ways: Pyrefly checks the test files
statically, and the same files execute against the real library. This module
holds the parts that do not vary between packages -- locating a Pyrefly binary,
locating the shared virtualenv, and running suites.

Nothing here touches the network. The virtualenv is created only by
`bootstrap_venv.py`, so a caller without egress gets an actionable error naming
the fix instead of a hang or an opaque proxy failure.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


TENSOR_SHAPES_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = TENSOR_SHAPES_ROOT.parent
SHAPE_EXTENSIONS_ROOT: Path = TENSOR_SHAPES_ROOT / "pyrefly-shape-extensions"

DEFAULT_VENV: Path = Path.home() / ".tensor-shapes-venv"

_BOOTSTRAP_HINT = """\
Create it with:

    python3 {bootstrap}

Bootstrapping downloads packages, so unlike the test runners it needs network
access. On a Meta machine that means fwdproxy, and an agent sandboxed without
egress must be relaunched with `claude --secure-internet-mode`.

Set $TENSOR_SHAPES_VENV to use a virtualenv somewhere other than the default.\
"""


@dataclass(frozen=True, kw_only=True)
class Suite:
    """A named group of test files that are checked, and usually also run.

    `patterns` are globs relative to the stub package root. A suite is the unit
    of both reporting and iteration, so keep them small enough that a developer
    can rerun one while working on a single area of the stubs.
    """

    name: str
    patterns: tuple[str, ...]
    python_version: str = "3.13"
    # Search paths beyond the stub tree and `shape_extensions`, for suites that
    # need extra fixtures on the path.
    extra_search_paths: tuple[Path, ...] = ()
    # Run Pyrefly with `--expectations`, matching reported errors against `# E:`
    # comments. This is how a suite pairs a static rejection with the runtime
    # error the library itself raises, so the numpy and jax suites enable it
    # everywhere. It is not merely a stricter mode: it also counts errors that
    # are otherwise suppressed, which is why the torch corpus enables it only
    # for the dedicated negative-test directories.
    expectations: bool = False

    def files(self, package_root: Path) -> list[str]:
        paths = sorted(
            {path for pattern in self.patterns for path in package_root.glob(pattern)}
        )
        if not paths:
            raise ValueError(
                f"suite {self.name!r} matched no files under {package_root}"
            )
        return [str(path.relative_to(package_root)) for path in paths]


def venv_python(explicit: Path | None = None, *, extra_hint: str = "") -> Path:
    """Resolve the interpreter that has the shaped libraries installed.

    Order: an explicit `--python`, then $TENSOR_SHAPES_VENV, then the default
    virtualenv. This never creates anything -- see the module docstring.

    Only the runtime tests need this. Type checking resolves the stubs through
    `--search-path` and never imports the real library, so it runs with no
    virtualenv at all; `extra_hint` is how a caller offering a static-only mode
    advertises it here.
    """

    # Made absolute throughout, because callers pass these on to child processes
    # that run from a different directory. Deliberately not `Path.resolve()`:
    # `<venv>/bin/python` is a symlink to the base interpreter, and following it
    # yields an interpreter that cannot see the virtualenv's site-packages.
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--python {explicit} does not exist")
        return Path(os.path.abspath(explicit))

    if "TENSOR_SHAPES_VENV" in os.environ:
        venv = Path(os.path.abspath(os.environ["TENSOR_SHAPES_VENV"]))
        source = "$TENSOR_SHAPES_VENV"
    else:
        venv = DEFAULT_VENV
        source = "the default location"

    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    if os.name == "nt":
        python = python.with_suffix(".exe")
    if not python.exists():
        hint = _BOOTSTRAP_HINT.format(
            bootstrap=(TENSOR_SHAPES_ROOT / "bootstrap_venv.py").relative_to(REPO_ROOT)
        )
        raise SystemExit(
            f"No tensor-shapes virtualenv at {venv} ({source}).\n\n{extra_hint}{hint}"
        )
    return python


def pyrefly_command(
    *,
    explicit: Path | None = None,
    buck: bool = False,
    release: bool = False,
) -> list[str]:
    """Resolve how to invoke Pyrefly, as an argv prefix.

    Order: an explicit `--pyrefly`, then `--buck`, then $PYREFLY, then a Cargo
    build. Buck and Cargo are both supported because the internal checkout
    often has only one of them on PATH.
    """

    # Explicit flags beat the environment, so that `--buck` cannot be silently
    # overridden by a $PYREFLY left in someone's shell profile -- which would
    # type check against a stale binary while looking like it rebuilt.
    if explicit is not None:
        return [str(_resolve_executable(explicit))]
    if buck:
        return ["buck2", "run", "fbcode//pyrefly:pyrefly", "--"]
    if "PYREFLY" in os.environ:
        return [str(_resolve_executable(Path(os.environ["PYREFLY"])))]

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target"))
    profile = "release" if release else "debug"
    built = target_dir / profile / "pyrefly"
    if not _resolve_executable(built).exists():
        raise SystemExit(
            f"No Pyrefly binary at {built}.\n\n"
            f"Build one with `cargo build{' --release' if release else ''}`, or "
            "pass `--buck` to run it out of Buck, or point `--pyrefly`/$PYREFLY "
            "at an existing binary."
        )
    return [str(_resolve_executable(built))]


def _resolve_executable(path: Path) -> Path:
    """Tolerate a missing `.exe` so callers can pass an OS-agnostic path."""

    if not path.exists():
        with_exe = path.with_name(path.name + ".exe")
        if with_exe.exists():
            return with_exe.resolve()
    return path.resolve()


def check_suites(
    *,
    pyrefly: list[str],
    package_root: Path,
    suites: list[Suite],
    nocapture: bool = False,
) -> int:
    """Type check every suite, returning the last nonzero exit code.

    Every suite runs even after one fails. The whole static pass takes seconds,
    so stopping early would only make a developer rediscover the next failure on
    the following run.
    """

    if not suites:
        raise ValueError(f"no suites to check under {package_root}")

    failed = 0
    for suite in suites:
        files = suite.files(package_root)
        command = [
            *pyrefly,
            "check",
            "--config",
            os.devnull,
            "--python-version",
            suite.python_version,
        ]
        if suite.expectations:
            command.append("--expectations")
        for search_path in (
            *suite.extra_search_paths,
            package_root,
            SHAPE_EXTENSIONS_ROOT,
        ):
            command.extend(["--search-path", str(search_path)])
        command.extend(files)

        if nocapture:
            print("+ " + " ".join(command), flush=True)
            result = subprocess.run(command, cwd=package_root)
            if result.returncode != 0:
                failed = result.returncode
            continue

        result = subprocess.run(
            command,
            cwd=package_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("+ " + " ".join(command), flush=True)
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            failed = result.returncode
            continue
        print(f"PASS {suite.name} ({len(files)} files)", flush=True)
    return failed


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_suites(
    *,
    library: str,
    package_root: Path,
    suites: list[Suite],
) -> int:
    """Execute the suites against the real library, returning the test count.

    This is the other half of the guarantee: the same `assert_shape` calls that
    Pyrefly verified statically are checked against the shapes the library
    actually produces, so a stub that is self-consistent but wrong still fails.
    """

    # `shape_extensions` is loaded from the working copy rather than from the
    # virtualenv, so runtime tests exercise local edits to the helpers.
    if "shape_extensions" not in sys.modules:
        _load_module(
            "shape_extensions",
            SHAPE_EXTENSIONS_ROOT / "shape_extensions" / "__init__.py",
        )
    import shape_extensions

    if not suites:
        raise ValueError(f"no suites to run under {package_root}")

    total = 0
    for suite in suites:
        for filename in suite.files(package_root):
            total += _run_test_file(
                library=library,
                path=package_root / filename,
                shape_extensions=shape_extensions,
            )
    print(f"PASS {len(suites)} suites ({total} tests)", flush=True)
    return total


def _run_test_file(*, library: str, path: Path, shape_extensions: Any) -> int:
    current_test: str | None = None
    assert_shape_calls: dict[str, int] = {}
    original_assert_shape: Callable[..., Any] = shape_extensions.assert_shape

    def counting_assert_shape(x: Any, shape: Any) -> Any:
        if current_test is not None:
            assert_shape_calls[current_test] += 1
        return original_assert_shape(x, shape)

    # Patch before importing so that a module-level
    # `from shape_extensions import assert_shape` binds the counting wrapper.
    shape_extensions.assert_shape = counting_assert_shape
    try:
        module = _load_module(f"_{library}_shape_test_{path.stem}", path)
        tests = [
            (name, value)
            for name, value in sorted(vars(module).items())
            if name.startswith("test_") and callable(value)
        ]
        if not tests:
            raise AssertionError(f"{path} does not define any test functions")
        # A module lists in GRADUAL_SHAPE_RUNTIME_TESTS the tests whose static
        # shape is gradual. Those may fall back to plain runtime assertions,
        # because assert_shape currently also demands an exact static shape.
        # TODO(stroxler): Define how assert_shape should handle gradual static shapes.
        gradual_shape_tests = set(getattr(module, "GRADUAL_SHAPE_RUNTIME_TESTS", ()))
        unknown_markers = gradual_shape_tests - {name for name, _ in tests}
        if unknown_markers:
            raise AssertionError(
                f"{path} marks unknown gradual-shape tests: {sorted(unknown_markers)}"
            )
        for name, test in tests:
            current_test = name
            assert_shape_calls[name] = 0
            test()
            current_test = None
            # A test that asserts no shapes passes vacuously and would hide a
            # regression, so treat it as a failure rather than a pass.
            if assert_shape_calls[name] == 0 and name not in gradual_shape_tests:
                raise AssertionError(f"{path}::{name} did not execute assert_shape")
    finally:
        shape_extensions.assert_shape = original_assert_shape

    shapes = sum(assert_shape_calls.values())
    print(f"PASS {path.name} ({len(tests)} tests, {shapes} shapes)", flush=True)
    return len(tests)
