# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These tests cover a script that ships inside the VS Code extension rather than
# anything Buck builds, so they have no Buck target either. `test.py` and
# `pyrefly.yml` run them with `python -m unittest`.
# @lint-ignore-every AUTODEPS2

"""Tests for `find_pyrefly.py`.

Reaching outside the selected environment is the failure these guard against, so
each test builds a real virtual environment and drives the script the way the
extension does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import venv
from pathlib import Path

FINDER: Path = Path(__file__).resolve().parent.parent / "resources" / "find_pyrefly.py"
BINARY_NAME: str = "pyrefly" + (sysconfig.get_config_var("EXE") or "")


def plant_binary(directory: Path) -> Path:
    """Put something that looks like an installed Pyrefly in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BINARY_NAME
    path.write_text("")
    path.chmod(0o755)
    return path.resolve()


class FindPyreflyTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

        # `PYTHONUSERBASE` relocates the per-user install schemes on every
        # platform, which keeps a real `~/.local/bin/pyrefly` on the machine
        # running these tests from deciding any result.
        self.environment: dict[str, str] = {
            **os.environ,
            "PYTHONUSERBASE": str(self.root / "userbase"),
            "HOME": str(self.root / "home"),
            "USERPROFILE": str(self.root / "home"),
        }

        builder = venv.EnvBuilder(with_pip=False)
        self.venv_path = self.root / "venv"
        try:
            context = builder.ensure_directories(self.venv_path)
            builder.create(self.venv_path)
        except (OSError, subprocess.CalledProcessError) as error:
            # Some distributions ship `venv` without the pieces it needs, and the
            # extension falls back to its bundled binary there anyway.
            self.skipTest(f"could not create a virtual environment: {error}")
        self.python = Path(context.env_exe)

    def find_pyrefly(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.python), str(FINDER)],
            capture_output=True,
            text=True,
            env=self.environment,
        )

    def scheme_scripts_directories(self) -> dict[str, Path]:
        """Every install scheme's script directory, as the venv sees them."""
        program = (
            "import json, sysconfig;"
            "print(json.dumps({s: sysconfig.get_path('scripts', s)"
            " for s in sysconfig.get_scheme_names()}))"
        )
        result = subprocess.run(
            [str(self.python), "-c", program],
            capture_output=True,
            text=True,
            env=self.environment,
            check=True,
        )
        return {name: Path(path) for name, path in json.loads(result.stdout).items()}

    def foreign_scripts_directories(self) -> list[Path]:
        """Script directories that do not belong to the virtual environment.

        Excludes the whole environment tree rather than just the interpreter's
        own directory, because the `nt` schemes name a sibling of it. Anything
        outside the temporary directory is a real directory on this machine, so
        it is dropped rather than written to.
        """
        return sorted(
            {
                path
                for path in self.scheme_scripts_directories().values()
                if not path.is_relative_to(self.venv_path)
                and path.is_relative_to(self.root)
            }
        )

    def test_reports_nothing_for_an_environment_without_pyrefly(self) -> None:
        result = self.find_pyrefly()
        self.assertEqual(result.stdout.strip(), "")
        # Exiting zero is what lets the extension tell an empty environment apart
        # from the search itself failing.
        self.assertEqual(result.returncode, 0)

    def test_finds_a_binary_installed_in_the_environment(self) -> None:
        expected = plant_binary(self.python.parent)
        result = self.find_pyrefly()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), str(expected))

    def test_ignores_binaries_outside_the_environment(self) -> None:
        foreign = self.foreign_scripts_directories()
        self.assertNotEqual(foreign, [], "expected at least one per-user scheme")
        for directory in foreign:
            plant_binary(directory)

        result = self.find_pyrefly()
        self.assertEqual(
            result.stdout.strip(),
            "",
            "a binary from outside the selected environment was reported",
        )
        self.assertEqual(result.returncode, 0)

    def test_prefers_the_environment_over_a_binary_outside_it(self) -> None:
        for directory in self.foreign_scripts_directories():
            plant_binary(directory)
        expected = plant_binary(self.python.parent)

        result = self.find_pyrefly()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), str(expected))


if __name__ == "__main__":
    sys.exit(unittest.main())
