import subprocess

from numpy.typing import ArrayLike

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

def use(app: str | None = None, gl: str | None = None): ...
def run_subprocess(command: ArrayLike, return_code: bool = False, **kwargs) -> tuple[str, str, int]: ...
def test(*args, **kwargs): ...
