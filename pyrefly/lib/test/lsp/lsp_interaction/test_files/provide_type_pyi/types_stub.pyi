# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional


class MyClass:
    def __init__(self, value: int) -> None: ...

    def get_value(self) -> int: ...

    def process_data(self, data: List[str]) -> Optional[str]: ...


def utility_function(x: int, y: str) -> bool: ...


MY_CONSTANT: int
