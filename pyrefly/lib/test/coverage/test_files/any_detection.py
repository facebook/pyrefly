# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


def typed_func(x: int, y: str) -> bool:
    return True


def any_return(x: int) -> Any:
    return x


def any_param(x: Any) -> int:
    return 0


def untyped_func(x, y):
    return True


any_var: Any = None
typed_var: int = 42
untyped_var = "hello"
