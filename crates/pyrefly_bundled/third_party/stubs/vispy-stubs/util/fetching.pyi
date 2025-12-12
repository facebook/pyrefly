# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import os
import shutil
import sys
import time
import urllib.request
from math import log
from os import path as op

from ..util.config import config

###############################################################################
# Vispy data directory

def load_data_file(fname: str, directory: str | None = None, force_download: str | bool = False) -> str: ...

###############################################################################
# File downloading (most adapted from mne-python)

class ProgressBar:
    spinner_symbols: list = ...
    template: str = ...

    def __init__(
        self,
        max_value: int,
        initial_value: int = 0,
        mesg: str = "",
        max_chars: int = 40,
        progress_character: str = ".",
        spinner: bool = False,
    ): ...
    def update(self, cur_value: float, mesg: str | None = None): ...
    def update_with_increment_value(self, increment_value: int, mesg: str | None = None): ...

def _chunk_read(response, local_file, chunk_size=65536, initial_size=0): ...
def _chunk_write(chunk, local_file, progress): ...
def _fetch_file(url, file_name, print_destination=True): ...
def sizeof_fmt(num): ...
