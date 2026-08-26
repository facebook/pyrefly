# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_pyrefly


class CorpusRunnerTest(unittest.TestCase):
    def test_discovers_only_documented_projects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scipy").mkdir()
            (root / "scipy" / "README.md").write_text("demo")
            (root / "scratch").mkdir()
            (root / "README.md").write_text("corpus")
            with patch.object(run_pyrefly, "CORPUS_ROOT", root):
                self.assertEqual(run_pyrefly.projects(), ["scipy"])

    def test_checks_modules_before_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fft.pyi").write_text("def f() -> None: ...")
            (root / "examples").mkdir()
            (root / "examples" / "fft.py").write_text("from fft import f")
            modules, examples = run_pyrefly.suites("scipy")
            self.assertEqual(modules.files(root), ["fft.pyi"])
            self.assertEqual(examples.files(root), ["examples/fft.py"])
            self.assertTrue(examples.expectations)
            self.assertEqual(
                modules.extra_search_paths, (run_pyrefly.NUMPY_STUBS_ROOT,)
            )

    def test_empty_project_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite in run_pyrefly.suites("scipy"):
                with self.assertRaises(ValueError):
                    suite.files(root)

    @patch.object(run_pyrefly, "check_suites", return_value=0)
    @patch.object(run_pyrefly, "pyrefly_command", return_value=["pyrefly"])
    @patch.object(run_pyrefly, "venv_python")
    def test_source_root_needs_no_scipy_or_virtualenv(
        self,
        venv_python: Mock,
        _pyrefly_command: Mock,
        check_suites: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "numpy").mkdir()
            with patch.object(run_pyrefly, "projects", return_value=["scipy"]):
                with patch.object(
                    run_pyrefly.sys,
                    "argv",
                    ["run_pyrefly.py", "--numpy-root", str(root)],
                ):
                    self.assertEqual(run_pyrefly.main(), 0)
            venv_python.assert_not_called()
            self.assertEqual(
                check_suites.call_args.kwargs["site_package_paths"], (root,)
            )
            self.assertFalse(check_suites.call_args.kwargs["check_stubs"])
            self.assertEqual(len(check_suites.call_args.kwargs["suites"]), 2)
