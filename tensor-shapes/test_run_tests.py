#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import call, Mock, patch

import run_tests
from run_tests import shaped_array_reference_lines


class ShapedArrayCorpusGuardTest(unittest.TestCase):
    def test_finds_every_reference(self) -> None:
        source = "\n".join(
            [
                "from shape_extensions import shaped_array",
                "@shaped_array(shape='Shape')",
                "# shaped_array must not remain in comments either",
                "class Ordinary: ...",
            ]
        )

        self.assertEqual(shaped_array_reference_lines(source), [1, 2, 3])

    @patch.object(run_tests, "shaped_array_references", return_value=[])
    @patch.object(run_tests, "run", return_value=True)
    @patch.object(run_tests, "venv_python", return_value=Path("/venv/bin/python"))
    @patch.object(run_tests, "pyrefly_command", return_value=["pyrefly"])
    def test_static_only_forwards_python_to_partial_stub_packages(
        self,
        _pyrefly_command: Mock,
        venv_python: Mock,
        run: Mock,
        _shaped_array_references: Mock,
    ) -> None:
        with patch.object(
            run_tests.sys,
            "argv",
            ["run_tests.py", "--static-only", "--python", "/custom/python"],
        ):
            self.assertEqual(run_tests.main(), 0)

        venv_python.assert_called_once_with(Path("/custom/python"))
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        run_tests.sys.executable,
                        str(
                            run_tests.TENSOR_SHAPES_ROOT
                            / "pyrefly-torch-stubs/run_pyrefly.py"
                        ),
                        "--pyrefly",
                        "pyrefly",
                        "--python",
                        "/venv/bin/python",
                    ]
                ),
                call(
                    [
                        run_tests.sys.executable,
                        str(
                            run_tests.TENSOR_SHAPES_ROOT
                            / "pyrefly-numpy-stubs/run_pyrefly.py"
                        ),
                        "--pyrefly",
                        "pyrefly",
                        "--python",
                        "/venv/bin/python",
                    ]
                ),
                call(
                    [
                        run_tests.sys.executable,
                        str(
                            run_tests.TENSOR_SHAPES_ROOT
                            / "pyrefly-jax-stubs/run_pyrefly.py"
                        ),
                        "--pyrefly",
                        "pyrefly",
                        "--python",
                        "/venv/bin/python",
                    ]
                ),
            ],
        )
