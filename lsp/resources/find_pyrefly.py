# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Print the path of the `pyrefly` binary installed for the interpreter running
this script, or exit with status 1 if there is none."""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

DISTRIBUTION = "pyrefly"
BINARY_NAME = DISTRIBUTION + sysconfig.get_config_var("EXE")


def candidates() -> Iterator[Path]:
    """Yield plausible locations of the binary, most authoritative first."""
    # The distribution's RECORD names the script relative to its site-packages
    # directory, which pins the exact copy the installer wrote. The binary is a
    # wheel script rather than a console script entry point, so RECORD is the
    # only metadata that mentions it.
    try:
        dist = distribution(DISTRIBUTION)
    except PackageNotFoundError:
        pass
    else:
        for file in dist.files or ():
            if file.name == BINARY_NAME:
                yield Path(dist.locate_file(file))

    # An install with missing or stale metadata still leaves the binary in one of
    # this interpreter's script directories: every install scheme it knows about
    # (which covers venv, prefix, user, and distro-patched schemes such as
    # Debian's), plus the directory the interpreter itself was launched from.
    for scheme in sysconfig.get_scheme_names():
        yield Path(sysconfig.get_path("scripts", scheme), BINARY_NAME)
    yield Path(sys.executable).parent / BINARY_NAME

    # As a last resort, attempt a which for "pyrefly"
    if (pyrefly_which := shutil.which(DISTRIBUTION)):
        yield pyrefly_which


def main() -> None:
    # Python 3.11 and later filter missing files out of `dist.files`, but earlier
    # versions do not, so a stale RECORD still has to be checked against disk.
    seen = set()
    for candidate in candidates():
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and os.access(path, os.X_OK):
            print(path)
            return
    sys.exit(1)


if __name__ == "__main__":
    main()
