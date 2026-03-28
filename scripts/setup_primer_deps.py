#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Set up project dependencies for mypy_primer runs.

Called by pyrefly_primer_wrapper.sh to install a project's explicit deps
into a venv without doing a full pip install. Prints site-package-path
flags for pyrefly to stdout on success.

Usage:
    python3 setup_primer_deps.py <project_name>
"""

import logging
import os
import subprocess
import sys

# This script lives in scripts/, so imports work directly.
from compare_typecheckers import setup_project
from projects import get_mypy_primer_projects

logging.basicConfig(level=logging.INFO, format="[primer-deps] %(message)s")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project_name>", file=sys.stderr)
        return 1

    name = sys.argv[1]
    project = next((p for p in get_mypy_primer_projects() if p.name == name), None)
    if project is None:
        logging.warning(f"Project {name} not found in projects.py — skipping deps")
        return 0

    logging.info(f"Setting up deps for {name} (deps={project.deps})")
    setup_project(project, ".", debug=False, reuse=True, install_project=False)
    logging.info(f"setup_project() succeeded for {name}")

    # Print site-package-path flags to stdout, one per line, for the wrapper
    # to read safely via readarray (handles paths with spaces).
    venv_python = os.path.join(".venv", "bin", "python")
    if os.path.exists(venv_python):
        # Get site packages from the venv's Python, not the current one.
        result = subprocess.run(
            [
                venv_python,
                "-c",
                "import site\n"
                "for p in site.getsitepackages():\n"
                "    print('--site-package-path=' + p)",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())
