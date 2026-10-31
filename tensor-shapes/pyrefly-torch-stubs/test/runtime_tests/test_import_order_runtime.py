# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""`shape_extensions` patches Torch whichever of the two is imported first.

Each test runs in a fresh interpreter, because the import order is the subject.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

SHAPE_EXTENSIONS_ROOT = Path(__file__).resolve().parents[3] / "pyrefly-shape-extensions"


def run_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env={**os.environ, "PYTHONPATH": str(SHAPE_EXTENSIONS_ROOT)},
        capture_output=True,
        text=True,
    )


class ImportOrderTest(unittest.TestCase):
    def assert_script_passes(self, script: str) -> None:
        result = run_script(script)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shape_extensions_alone_does_not_import_torch(self) -> None:
        self.assert_script_passes(
            """
            import sys

            import shape_extensions

            assert "torch" not in sys.modules
            """
        )

    def test_torch_imported_first(self) -> None:
        self.assert_script_passes(
            """
            import torch
            import shape_extensions

            assert torch.Tensor[2, 3] is torch.Tensor
            assert torch.nn.Linear[2, 3] is torch.nn.Linear
            """
        )

    def test_shape_extensions_imported_first(self) -> None:
        self.assert_script_passes(
            """
            import sys

            import shape_extensions
            import torch

            assert torch.Tensor[2, 3] is torch.Tensor
            assert torch.nn.Conv2d[2, 3] is torch.nn.Conv2d
            assert not any(
                type(finder).__name__ == "_PostImportHook" for finder in sys.meta_path
            )
            assert type(torch.__spec__.loader).__name__ != "_PostImportLoader"
            """
        )

    def test_annotations_evaluate_when_torch_is_imported_later(self) -> None:
        self.assert_script_passes(
            """
            from shape_extensions import IntVar

            import torch

            N = IntVar("N")

            def double(x: torch.Tensor[N, 3]) -> torch.Tensor[N * 2, 3]:
                return torch.cat([x, x])

            assert double(torch.zeros(4, 3)).shape == (8, 3)
            """
        )
