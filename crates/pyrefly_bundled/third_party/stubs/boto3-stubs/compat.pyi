"""
Type annotations for boto3.compat module.

Copyright 2024 Vlad Emelianov
"""

import os
from typing import Any

SOCKET_ERROR: type[ConnectionError]

rename_file = os.rename

def filter_python_deprecation_warnings() -> None: ...
def is_append_mode(fileobj: Any) -> bool: ...
