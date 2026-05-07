# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for setup_primer_deps.py.

Run with: python -m pytest scripts/test_setup_primer_deps.py -v
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from setup_primer_deps import main


class TestMain:
    """Tests for the main() entry point."""

    def test_missing_arg(self, capsys):
        """Should print usage and return 1 when no project name given."""
        with patch.object(sys, "argv", ["setup_primer_deps.py"]):
            assert main() == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

    @patch("setup_primer_deps.get_mypy_primer_projects")
    def test_unknown_project(self, mock_projects, capsys):
        """Unknown project name should skip gracefully (return 0)."""
        mock_projects.return_value = []
        with patch.object(sys, "argv", ["setup_primer_deps.py", "nonexistent"]):
            assert main() == 0

    @patch("setup_primer_deps.os.path.exists", return_value=False)
    @patch("setup_primer_deps.setup_project")
    @patch("setup_primer_deps.get_mypy_primer_projects")
    def test_known_project_no_venv(self, mock_projects, mock_setup, mock_exists, capsys):
        """Known project but no venv created — should succeed without printing paths."""
        project = MagicMock()
        project.name = "testproj"
        project.deps = ["dep1"]
        mock_projects.return_value = [project]

        with patch.object(sys, "argv", ["setup_primer_deps.py", "testproj"]):
            assert main() == 0

        mock_setup.assert_called_once_with(
            project, ".", debug=False, reuse=True, install_project=False
        )
        captured = capsys.readouterr()
        # No site-package-path flags printed since venv doesn't exist
        assert "--site-package-path" not in captured.out

    @patch("setup_primer_deps.subprocess.run")
    @patch("setup_primer_deps.os.path.exists", return_value=True)
    @patch("setup_primer_deps.setup_project")
    @patch("setup_primer_deps.get_mypy_primer_projects")
    def test_known_project_with_venv(
        self, mock_projects, mock_setup, mock_exists, mock_run, capsys
    ):
        """Known project with venv — should print site-package-path flags."""
        project = MagicMock()
        project.name = "testproj"
        project.deps = ["dep1", "dep2"]
        mock_projects.return_value = [project]

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="--site-package-path=/path/to/site-packages\n",
        )

        with patch.object(sys, "argv", ["setup_primer_deps.py", "testproj"]):
            assert main() == 0

        captured = capsys.readouterr()
        assert "--site-package-path=/path/to/site-packages" in captured.out

    @patch("setup_primer_deps.subprocess.run")
    @patch("setup_primer_deps.os.path.exists", return_value=True)
    @patch("setup_primer_deps.setup_project")
    @patch("setup_primer_deps.get_mypy_primer_projects")
    def test_venv_python_fails(
        self, mock_projects, mock_setup, mock_exists, mock_run, capsys
    ):
        """Venv exists but python -c fails — should succeed without printing paths."""
        project = MagicMock()
        project.name = "testproj"
        project.deps = []
        mock_projects.return_value = [project]

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
        )

        with patch.object(sys, "argv", ["setup_primer_deps.py", "testproj"]):
            assert main() == 0

        captured = capsys.readouterr()
        assert "--site-package-path" not in captured.out

    @patch("setup_primer_deps.setup_project", side_effect=Exception("clone failed"))
    @patch("setup_primer_deps.get_mypy_primer_projects")
    def test_setup_project_failure(self, mock_projects, mock_setup):
        """setup_project() raises — should propagate the exception."""
        project = MagicMock()
        project.name = "testproj"
        project.deps = ["dep1"]
        mock_projects.return_value = [project]

        with patch.object(sys, "argv", ["setup_primer_deps.py", "testproj"]):
            with pytest.raises(Exception, match="clone failed"):
                main()
