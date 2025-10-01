# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import sysconfig


def get_pyrefly_bin() -> str:
    # Try to figure out where the pyrefly executable is installed.
    # First look for a global installation.
    pyrefly_exe = "pyrefly" + sysconfig.get_config_var("EXE")
    pyrefly_bin = os.path.join(sysconfig.get_path("scripts"), pyrefly_exe)
    if os.path.isfile(pyrefly_bin):
        return pyrefly_bin
    # Next look for a user-local installation.
    if sys.version_info >= (3, 10):  # get_preferred_scheme is new in 3.10
        user_scheme = sysconfig.get_preferred_scheme("user")
        pyrefly_bin = os.path.join(
            sysconfig.get_path("scripts", scheme=user_scheme), pyrefly_exe
        )
        if os.path.isfile(pyrefly_bin):
            return pyrefly_bin
    # We didn't find the executable; let's just hope it's on the path.
    return "pyrefly"


if __name__ == "__main__":
    proc = subprocess.run([get_pyrefly_bin(), *sys.argv[1:]])
    sys.exit(proc.returncode)
