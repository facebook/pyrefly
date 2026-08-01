#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Test that everything works well
"""

from __future__ import annotations

import abc
import argparse
import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import final, Generator, Iterable

SCRIPT_PATH: Path = Path(__file__).parent


class Colors(Enum):
    # Copied from https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclasses.dataclass(frozen=True)
class TestFlags:
    run_fmt: bool
    run_lint: bool
    run_test: bool
    run_tensor_shapes: bool
    run_conformance: bool
    run_jsonschema: bool


def print_running(msg: str) -> None:
    print(Colors.OKGREEN.value + "Running " + msg + "..." + Colors.ENDC.value)


@contextmanager
def timing() -> Generator[None, None, None]:
    start = time.time()
    yield
    duration = time.time() - start
    print(f"Finished in {duration:.2f} seconds.")


def run(
    args: Iterable[str],
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Runs a command (args) in a new process.
    If the command fails, raise CalledProcessError.
    If the command passes, return CompletedProcess.
    If capture_output is False, print to the console, otherwise record it as CompletedProcess.stdout/stderr.
    If error is specified, print error on stderr when there is a CalledProcessError.
    """
    # On Ci stderr gets out of order with stdout. To avoid this, we need to flush stdout/stderr first.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        # @lint-ignore FIXIT1 NoUnsafeExecRule
        result = subprocess.run(
            tuple(args),
            # We'd like to use the capture_output argument,
            # but that isn't available in Python 3.6 which we use on Windows
            stdout=subprocess.PIPE if capture_output else sys.stdout,
            stderr=subprocess.PIPE if capture_output else sys.stderr,
            check=True,
            encoding="utf-8",
            env=env,
        )
        return result
    except subprocess.CalledProcessError as e:
        # Print the console info if we were capturing it
        if capture_output:
            print(e.stdout, file=sys.stdout)
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


class Executor(abc.ABC):
    @abc.abstractmethod
    def chdir(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def rustfmt(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def clippy(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def test(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def tensor_shapes(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def conformance(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def jsonschema(self) -> None:
        raise NotImplementedError()


@final
class CargoExecutor(Executor):
    def chdir(self) -> None:
        # Change to the pyrefly directory
        script_dir = SCRIPT_PATH.absolute()
        os.chdir(str(script_dir))

    def rustfmt(self) -> None:
        # rustfmt.toml's import options are nightly-only; use nightly when present to
        # apply them and avoid stable rustfmt's "unstable features" warnings.
        try:
            # @lint-ignore FIXIT1 NoUnsafeExecRule
            has_nightly = (
                subprocess.run(
                    ["cargo", "+nightly", "fmt", "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
        except OSError:
            has_nightly = False
        run(["cargo", "+nightly", "fmt"] if has_nightly else ["cargo", "fmt"])

    def clippy(self) -> None:
        run(["cargo", "clippy"])

    def test(self) -> None:
        run(["cargo", "build"])
        run(["cargo", "test"])
        scrut_path = shutil.which("scrut")
        if scrut_path is None:
            print(
                Colors.WARNING.value
                + "Scrut is not installed, skipping scrut tests."
                + Colors.ENDC.value
            )
            return
        script_dir = SCRIPT_PATH.absolute()
        cargo_target_dir = Path(
            os.environ.get("CARGO_TARGET_DIR", script_dir / "target")
        )
        pyrefly = (
            cargo_target_dir
            / "debug"
            / ("pyrefly.exe" if os.name == "nt" else "pyrefly")
        )
        jq_path = shutil.which("jq")
        run(
            [scrut_path, "test", "test"],
            env={
                "PYREFLY": str(pyrefly),
                "TYPESHED_ROOT": str(
                    script_dir / "crates" / "pyrefly_bundled" / "third_party"
                ),
                "JQ": jq_path if jq_path else "",
                "TEST_PY": str(script_dir / "test.py"),
                "PYREFLY_PY": str(script_dir / "pyrefly" / "python"),
                "PATH": os.environ.get("PATH", ""),
            },
        )

    def tensor_shapes(self) -> None:
        run(["cargo", "build"])
        # The runners resolve the debug pyrefly themselves, so we don't pass `--pyrefly`.
        run([sys.executable, "tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py"])
        run([sys.executable, "tensor-shapes/pyrefly-numpy-stubs/run_pyrefly.py"])

    def conformance(self) -> None:
        cargo_target_dir = os.environ.get("CARGO_TARGET_DIR", "target")
        run(
            [
                sys.executable,
                "conformance/conformance_output.py",
                "conformance/third_party",
                "--executable",
                f"{cargo_target_dir}/debug/pyrefly",
            ]
        )

    def jsonschema(self) -> None:
        run(["python3", "schemas/validate_schemas.py"])


@final
class BuckExecutor(Executor):
    def chdir(self) -> None:
        # Change to the pyrefly directory
        script_dir = SCRIPT_PATH.absolute()
        os.chdir(str(script_dir))

    def rustfmt(self) -> None:
        run(["arc", "f"])

    def clippy(self) -> None:
        run(
            [
                "arc",
                "rust-clippy",
                "...",
            ]
        )

    def test(self) -> None:
        if "SANDCASTLE_NONCE" in os.environ:
            print("Skipping tests on CI because they're already scheduled.")
            return
        res = run(
            [
                "buck2",
                "uquery",
                "kind('rust_test|rust_library', ...)",
            ],
            capture_output=True,
        )
        tests = [line.strip() for line in res.stdout.splitlines()] + [
            "test/...",
        ]
        run(
            ["buck2", "test"]
            + tests
            + ["--", "--run-disabled", "--return-zero-on-skips"]
        )

    def tensor_shapes(self) -> None:
        if "SANDCASTLE_NONCE" in os.environ:
            print(
                "Skipping tensor shape tests on CI because they're already scheduled."
            )
            return
        run(
            [
                "buck2",
                "test",
                "tensor-shapes/pyrefly-torch-stubs/examples:torch_examples_test",
                "tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_all_test",
                "tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_error_test",
                "tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_test",
                "tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_error_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_arithmetic_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_broadcasting_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_creation_basics_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_dtype_properties_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_examples_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_indexing_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_linalg_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_math_ufuncs_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_random_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_reductions_static_test",
                "tensor-shapes/pyrefly-numpy-stubs:numpy_runtime_test",
                "--",
                "--run-disabled",
                "--return-zero-on-skips",
            ]
        )

    def conformance(self) -> None:
        run(
            [
                "buck2",
                "run",
                "conformance:conformance_output_script",
                "--",
                "./conformance/third_party",
            ]
        )

    def jsonschema(self) -> None:
        run(
            [
                "buck2",
                "test",
                "--reuse-current-config",
                "schemas:test",
            ]
        )


def run_tests(executor: Executor, test_flags: TestFlags) -> None:
    if test_flags.run_fmt:
        print_running("formatting")
        with timing():
            executor.rustfmt()

    if test_flags.run_lint:
        print_running("linting")
        with timing():
            executor.clippy()

    if test_flags.run_test:
        print_running("tests")
        with timing():
            executor.test()

    if test_flags.run_tensor_shapes:
        print_running("tensor shape tests")
        with timing():
            executor.tensor_shapes()

    if test_flags.run_conformance:
        print_running("conformance tests")
        with timing():
            executor.conformance()

    if test_flags.run_jsonschema:
        print_running("jsonschema tests")
        with timing():
            executor.jsonschema()


def get_executor(mode: str) -> Executor:
    if mode == "auto":
        mode = "buck" if (SCRIPT_PATH / "pyrefly" / "BUCK").is_file() else "cargo"
    return BuckExecutor() if mode == "buck" else CargoExecutor()


def main(mode: str, test_flags: TestFlags) -> None:
    executor = get_executor(mode)
    executor.chdir()
    run_tests(executor, test_flags)


def invoke_main() -> None:
    parser = argparse.ArgumentParser(description="Pyrefly test script")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["buck", "cargo", "auto"],
        default="auto",
        help=(
            "Build the project with buck or cargo."
            "Default is auto-detect based on the existence of BUCK file."
        ),
    )
    # Requires Python 3.9+
    parser.add_argument(
        "--fmt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run code formatting or not",
    )
    parser.add_argument(
        "--lint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run code linting or not",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run testing or not",
    )
    parser.add_argument(
        "--tensor-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run tensor shape tests or not",
    )
    parser.add_argument(
        "--conformance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run conformance test or not",
    )
    parser.add_argument(
        "--jsonschema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run jsonschema test or not",
    )
    args = parser.parse_args()
    try:
        main(
            args.mode,
            TestFlags(
                run_fmt=args.fmt,
                run_lint=args.lint,
                run_test=args.test,
                run_tensor_shapes=args.tensor_shapes,
                run_conformance=args.conformance,
                run_jsonschema=args.jsonschema,
            ),
        )
    except KeyboardInterrupt:
        # no stack trace on interrupt
        sys.exit(signal.SIGINT)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    invoke_main()  # pragma: no cover
