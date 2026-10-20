# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_compatibility_checks


class CompatibilityRunnerTest(unittest.TestCase):
    def test_main_only_checks_directories_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "case").mkdir()
            (root / "case" / "pyrefly.toml").touch()
            (root / "tests").mkdir()
            with (
                patch.object(run_compatibility_checks, "COMPATIBILITY_ROOT", root),
                patch.object(
                    run_compatibility_checks, "venv_python", return_value=root
                ),
                patch.object(
                    run_compatibility_checks,
                    "pyrefly_command",
                    return_value=["pyrefly"],
                ),
                patch.object(
                    run_compatibility_checks, "check_case", return_value=[]
                ) as check,
                patch.object(sys, "argv", ["run_compatibility_checks.py"]),
            ):
                self.assertEqual(run_compatibility_checks.main(), 0)

        check.assert_called_once_with(root / "case", ["pyrefly"], root)

    def test_resolves_virtualenv_checker_executables(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            case.mkdir()
            (case / "downstream.py").touch()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python.exe"
            for checker in ("mypy", "pyright", "ty"):
                (bin_dir / f"{checker}.exe").touch()

            with (
                patch.object(
                    run_compatibility_checks, "venv_site_packages", return_value=root
                ),
                patch.object(
                    run_compatibility_checks, "run", return_value=(True, "")
                ) as run,
            ):
                checks = run_compatibility_checks.check_case(case, ["pyrefly"], python)

        self.assertTrue(all(ok for _, ok, _ in checks))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [Path(command[0]).name for command in commands[1:]],
            ["mypy.exe", "pyright.exe", "ty.exe"],
        )
        self.assertIn(f"--cache-dir={os.devnull}", commands[1])

    @patch.object(
        run_compatibility_checks.subprocess, "run", side_effect=FileNotFoundError
    )
    def test_missing_checker_reports_setup_instructions(self, _run: object) -> None:
        ok, message = run_compatibility_checks.run(
            ["/venv/bin/mypy", "--version"], Path(".")
        )
        self.assertFalse(ok)
        self.assertIn("/venv/bin/mypy", message)
        self.assertIn("bootstrap_venv.py", message)
