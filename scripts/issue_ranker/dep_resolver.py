#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Detect and install missing third-party dependencies from checker errors.

Instead of pre-scanning code for imports, we let the type checkers run first
and extract missing module names from their error output. This is more accurate
because the checkers already know which modules are stdlib vs third-party.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys

# Map module names to pip package names where they differ.
_MODULE_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "attrs": "attrs",
    "google.protobuf": "protobuf",
    "pydantic_xml": "pydantic-xml",
    "pydantic_settings": "pydantic-settings",
}

# Patterns for extracting the missing module name from checker error messages.
# Each checker phrases "module not found" differently.
_MISSING_MODULE_PATTERNS = [
    # pyrefly: 'Could not resolve import of "foo"'
    re.compile(r'Could not resolve import of "([^"]+)"'),
    # pyrefly: 'Module "foo" has no attribute ...' (missing-module)
    re.compile(r'Module "([^"]+)" has no attribute'),
    # Pyright: 'Import "foo.bar" could not be resolved from source'
    re.compile(r'Import "([^"]+)" could not be resolved from source'),
    # Pyright: 'Import "foo" could not be resolved'
    re.compile(r'Import "([^"]+)" could not be resolved'),
    # mypy: 'Cannot find implementation or library stub for module named "foo"'
    re.compile(
        r'Cannot find implementation or library stub for module named "([^"]+)"'
    ),
    # mypy: 'Library stubs not installed for "foo"'
    re.compile(r'Library stubs not installed for "([^"]+)"'),
    # mypy: 'No library stub file for module "foo"'
    re.compile(r'No library stub file for module "([^"]+)"'),
]


def _module_to_package(module: str) -> str:
    """Map a Python module name to its pip package name."""
    return _MODULE_TO_PACKAGE.get(module, module)


def extract_module_from_error(error: dict) -> str | None:
    """Extract the missing module name from a checker import error.

    Handles error formats from pyrefly, pyright, and mypy. Returns the
    top-level module name (e.g. "numpy" from "numpy.typing"), or None
    if the error doesn't contain a recognizable missing-module pattern.
    """
    message = error.get("message", "")
    for pattern in _MISSING_MODULE_PATTERNS:
        m = pattern.search(message)
        if m:
            full_module = m.group(1)
            return full_module.split(".")[0]
    return None


def _install_individually(
    packages: dict[str, str],
    installed: set[str],
    failed: set[str],
) -> None:
    """Try to pip install each package individually, updating installed/failed sets."""
    for module, package in packages.items():
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                installed.add(module)
            else:
                failed.add(module)
                logging.warning(f"  Failed to install {package}: {r.stderr[:100]}")
        except subprocess.TimeoutExpired:
            failed.add(module)
            logging.warning(f"  Timeout installing {package}")


def install_missing_modules(
    modules: set[str],
) -> tuple[set[str], set[str]]:
    """Try to pip install the given modules.

    Uses _module_to_package() for known name mappings (e.g. PIL -> pillow).
    Attempts a batch install first, falling back to individual installs.

    Returns:
        (installed, failed) -- sets of module names.
    """
    if not modules:
        return set(), set()

    packages = {m: _module_to_package(m) for m in modules}
    installed: set[str] = set()
    failed: set[str] = set()

    # Batch install for efficiency.
    pkg_list = sorted(set(packages.values()))
    logging.info(f"  Installing {len(pkg_list)} packages: {pkg_list}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pkg_list,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            installed = set(modules)
            logging.info(f"  Successfully installed: {pkg_list}")
        else:
            logging.info("  Batch install failed, trying individually...")
            _install_individually(packages, installed, failed)
    except subprocess.TimeoutExpired:
        logging.warning("  Batch pip install timed out, trying individually...")
        _install_individually(packages, installed, failed)

    if installed:
        logging.info(f"  Installed: {sorted(installed)}")
    if failed:
        logging.warning(f"  Failed to install: {sorted(failed)}")

    return installed, failed
