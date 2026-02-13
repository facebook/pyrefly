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
    run_conformance: bool


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
    def conformance(self) -> None:
        raise NotImplementedError()


@final
class CargoExecutor(Executor):
    def chdir(self) -> None:
        # Change to the pyrefly directory
        script_dir = SCRIPT_PATH.absolute()
        os.chdir(str(script_dir))

    def rustfmt(self) -> None:
        run(["cargo", "fmt"])

    def clippy(self) -> None:
        run(["cargo", "clippy"])

    def test(self) -> None:
        run(["cargo", "build"])
        run(["cargo", "test"])
        script_dir = SCRIPT_PATH.absolute()
        scrut_path = shutil.which("scrut")
        jq_path = shutil.which("jq")
        if scrut_path is not None:
            run(
                [scrut_path, "test", "test"],
                env={
                    "PYREFLY": str(script_dir / "target" / "debug" / "pyrefly"),
                    "TYPESHED_ROOT": str(
                        script_dir / "crates" / "pyrefly_bundled" / "third_party"
                    ),
                    "JQ": jq_path if jq_path else "",
                    "TEST_PY": str(script_dir / "test.py"),
                    "PYREFLY_PY": str(script_dir / "pyrefly" / "python"),
                    "TENSOR_TEST_ROOT": str(script_dir / "test" / "tensor_shapes"),
                    "PATH": os.environ.get("PATH", ""),
                },
            )
        else:
            print(
                Colors.WARNING.value
                + "Scrut is not installed, skipping scrut tests."
                + Colors.ENDC.value
            )

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
        tests = [line.strip() for line in res.stdout.splitlines()] + ["test:"]
        run(
            ["buck2", "test"]
            + tests
            + ["--", "--run-disabled", "--return-zero-on-skips"]
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

    if test_flags.run_conformance:
        print_running("conformance tests")
        with timing():
            executor.conformance()


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
        "--conformance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run conformance test or not",
    )
    args = parser.parse_args()
    try:
        main(
            args.mode,
            TestFlags(
                run_fmt=args.fmt,
                run_lint=args.lint,
                run_test=args.test,
                run_conformance=args.conformance,
            ),
        )
    except KeyboardInterrupt:
        # no stack trace on interrupt
        sys.exit(signal.SIGINT)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    invoke_main()  # pragma: no cover
