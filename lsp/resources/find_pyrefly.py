# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script ships inside the VS Code extension and is run by the user's own
# Python interpreter, so no build system produces it and it has no Buck target.
# @lint-ignore-every AUTODEPS2

"""Print the path of the `pyrefly` binary installed for the interpreter running
this script, or print nothing if it has none.

Exiting is reserved for failure: an uncaught exception here means the search
itself broke, which the caller must be able to tell apart from an environment
that simply has no Pyrefly in it."""

from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Iterator
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path

DISTRIBUTION = "pyrefly"
BINARY_NAME = DISTRIBUTION + (sysconfig.get_config_var("EXE") or "")


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

    yield Path(sysconfig.get_path("scripts"), BINARY_NAME)
    yield Path(sys.executable).parent / BINARY_NAME


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


if __name__ == "__main__":
    main()
